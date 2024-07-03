import os.path as osp
from collections import OrderedDict
import math
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from models import maple_clip as clip
from .simple_tokenizer import SimpleTokenizer as _Tokenizer


_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg["name"]
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'MaPLe',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": cfg["n_ctx"]}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg["n_ctx"]
        ctx_init = cfg["ctx_init"] if len(cfg["ctx_init"]) > 0 else None
        dtype = clip_model.dtype
        self.dtype = dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        # Default is 1, which is compound shallow prompting
        assert cfg["prompt_depth"] >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = cfg["prompt_depth"]  # max=12, but will create 11 such shared prompts

        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.prompt_prefix = prompt_prefix
        print('MaPLe design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")
        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        self.proj = nn.Linear(ctx_dim, 768)
        self.proj.half()
        self.ctx = nn.Parameter(ctx_vectors)
        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        # Minimum can be 1, which defaults to shallow MaPLe
        # compound prompts
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        # Also make corresponding projection layers, for each prompt
        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        # Before returning, need to transform
        # prompts to 768 for the visual side
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        return prompts, self.proj(self.ctx), self.compound_prompts_text, visual_deep_prompts   # pass here original, as for visual 768 is required


    # encode specific text that is different from self.classnames
    def encode_text(self, text: list, token_embedding):
        n_cls = len(text)
        text = [name.replace("_", " ") for name in text]
        name_lens = [len(_tokenizer.encode(name)) for name in text]
        prompts = [self.prompt_prefix + " " + name + "." for name in text]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
        with torch.no_grad():
            dtype = self.dtype
            embedding = token_embedding(tokenized_prompts).type(dtype)
            embedding = embedding.to(tokenized_prompts.device)
 
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(n_cls, -1, -1)

        prefix = embedding[:, :1, :]  # SOS
        suffix = embedding[:, 1 + self.n_ctx :, :]  # CLS, EOS

        prompts = self.construct_prompts(ctx, prefix, suffix)

        # Before returning, need to transform
        # prompts to 768 for the visual side
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        return prompts, tokenized_prompts, self.proj(self.ctx), self.compound_prompts_text, visual_deep_prompts   # pass here original, as for visual 768 is required


    def get_visual_prompt(self,):
        # Before returning, need to transform
        # prompts to 768 for the visual side
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        return self.proj(self.ctx), visual_deep_prompts   # pass here original, as for visual 768 is required



class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
        image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()

        return logits

    # custom label set for tree cut
    def forward_custom_label_set(self, image, text: list, token_embedding):
        text_features = self.encode_text(text=text, token_embedding=token_embedding, normalize=True)
        image_features = self.encode_image(image, normalize=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


    def encode_text(self, text: list, token_embedding, normalize=True):

        prompts, tokenized_prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner.encode_text(text, token_embedding)
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)

        if normalize:
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def encode_image(self, image, normalize=True):
        shared_ctx, deep_compound_prompts_vision = self.prompt_learner.get_visual_prompt()
        image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)

        if normalize:
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features




def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MaPLe(nn.Module):
    def __init__(self, cfg, classnames):
        super().__init__()
        self.cfg = cfg
        self.check_cfg(self.cfg)
        self.classnames = classnames
        self.build_model()


    def check_cfg(self, cfg):
        assert cfg["prec"] in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg

        print("Loading CLIP (backbone:" + cfg["name"] + ") into Maple")
        clip_model = load_clip_to_cpu(cfg)

        if cfg["prec"] == "fp32" or cfg["prec"] == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, self.classnames, clip_model)
        self.token_embedding = clip_model.token_embedding
        #self.logit_scale = self.model.logit_scale        

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

    def forward(self, image, labelset=None, treecut_node=None):
        output = {}
        # use the init classname for classification
        if labelset is None:
            output["logits_per_image"] = self.model(image)
        else:
            output["logits_per_image"] = self.model.forward_custom_label_set(image, labelset, self.token_embedding) 
        return output

    # given a list of classes, return the text feature of those classes
    # return output: (len(text), feat_size)
    def encode_text(self, text : list, normalize=True):
        text_features = self.model.encode_text(text, self.token_embedding, normalize=normalize)
        return text_features

    # encode image and return image feature
    # return output: (bz, feat_size)
    def encode_image(self, image, normalize=True):
        image_features = self.model.encode_image(image, normalize=normalize)
        return image_features


    # load the pretrained_ckpt from author
    def load_author_pretrained_ckpt(self, model_path):   
        map_location = None if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(model_path, map_location=map_location)
        state_dict = checkpoint["state_dict"]
        epoch = checkpoint["epoch"]

        #print(state_dict.keys())
        # Ignore fixed token vectors
        if "prompt_learner.token_prefix" in state_dict:
            del state_dict["prompt_learner.token_prefix"]

        if "prompt_learner.token_suffix" in state_dict:
            del state_dict["prompt_learner.token_suffix"]

        #print(state_dict["prompt_learner.ctx"]) 
        self.model.load_state_dict(state_dict, strict=False)

        # double check the state dict is loaded
        assert torch.equal(self.model.prompt_learner.ctx, state_dict["prompt_learner.ctx"])

    # load the pretrained_ckpt from author
    def load_custom_ckpt(self, model_path):
        map_location = None if torch.cuda.is_available() else "cpu"
        state_dict = torch.load(model_path, map_location=map_location)

        # Ignore fixed token vectors
        if "token_prefix" in state_dict:
           del state_dict["token_prefix"]

        if "token_suffix" in state_dict:
           del state_dict["token_suffix"]

        #print(state_dict.keys())

        name_to_copy = ["prompt_learner.proj.weight", "prompt_learner.compound_prompts_text.0", "prompt_learner.compound_prompt_projections.0.bias", "prompt_learner.proj.bias", "prompt_learner.compound_prompt_projections.0.weight", "prompt_learner.compound_prompt_projections.1.weight", "prompt_learner.ctx", "prompt_learner.compound_prompt_projections.1.bias", "prompt_learner.compound_prompts_text.1"]
        #print(self.model.state_dict().keys())

        for n in name_to_copy:
            self.model.state_dict()[n].copy_(state_dict["model." + n])
        
            assert torch.equal(self.model.state_dict()[n], state_dict["model." + n])




def load(cfg, classnames: list):
    model = MaPLe(cfg, classnames)
    return model


