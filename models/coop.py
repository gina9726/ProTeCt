import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from models import clip
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

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg["n_ctx"]
        ctx_init = cfg["ctx_init"] if len(cfg["ctx_init"]) > 0 else None

        dtype = clip_model.dtype
        self.dtype = dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg["csc"]:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.prompt_prefix = prompt_prefix
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg["class_token_position"]

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

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
        token_prefix = embedding[:, :1, :]  # SOS
        token_suffix = embedding[:, 1 + self.n_ctx :, :]  # CLS, EOS

        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(n_cls, -1, -1)

        prefix = token_prefix
        suffix = token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(n_cls):
                name_len = name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts, tokenized_prompts




class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
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
        prompts, tokenized_prompts = self.prompt_learner.encode_text(text, token_embedding)
        text_features = self.text_encoder(prompts, tokenized_prompts)

        if normalize:
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def encode_image(self, image, normalize=True):
        image_features = self.image_encoder(image.type(self.dtype))

        if normalize:
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features


class CoOp(nn.Module):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

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
        print("Loading CLIP (backbone:" + cfg["name"] + ") into CoOp")
        clip_model = load_clip_to_cpu(cfg)

        if cfg["prec"] == "fp32" or cfg["prec"] == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        # this is no optimizable
        self.token_embedding = clip_model.token_embedding
        for name, param in self.token_embedding.named_parameters():
            param.requires_grad_(False)
        self.token_embedding.eval()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, self.classnames, clip_model)
        self.logit_scale = self.model.logit_scale        

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
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

        # Ignore fixed token vectors
        if "token_prefix" in state_dict:
           del state_dict["token_prefix"]

        if "token_suffix" in state_dict:
           del state_dict["token_suffix"]

        # set strict=False
        state_dict["prompt_learner.ctx"] = state_dict["ctx"]
        state_dict.pop("ctx")

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

        #print(state_dict["prompt_learner.ctx"])
        name_to_copy = ["prompt_learner.ctx"]
        #print(self.model.state_dict().keys())

        for n in name_to_copy:
            self.model.state_dict()[n].copy_(state_dict["model." + n])
        
            assert torch.equal(self.model.state_dict()[n], state_dict["model." + n])



def load(cfg, classnames: list):
    model = CoOp(cfg, classnames)
    return model

