### treecut_generator.py
# Generate random tree cuts by sampling a Bernoulli random variable at
# each internal node of the given tree.
# Author: Tz-Ying Wu
###

import torch
import torch.nn.functional as F
from torch import nn

class TreecutGenerator(nn.Module):
    """Module to perform dropout at each node.
        To get a random cut during training, simply run
            label_set, tgt_mapping = treecut_generator.get_randomcut()
        which will produce a binary mask indicating the used labels.
    """
    def __init__(self, dependence, masks, codewords, dropout_rate=0.1):
        super(TreecutGenerator, self).__init__()

        self.dropout_rate = dropout_rate
        self.n_intnl_node, self.n_param = masks.shape
        self.dependence = dependence.cuda()
        self.dependence_cnt = self.dependence.sum(dim=1)
        masks = masks.unsqueeze(0).cuda()
        self.masks = torch.cat(((1 - masks).bool(), masks.bool()))
        self.codewords = codewords.cuda()

    @torch.no_grad()
    def forward(self, p=None):
        x = torch.ones((self.n_intnl_node)).float().cuda()
        if p is None:
            x = F.dropout(x, self.dropout_rate) * (1 - self.dropout_rate)
        else:
            x = F.dropout(x, p) * (1 - p)

        return x

    @torch.no_grad()
    def get_randomcut(self, p=None):
        # sampling
        P = self.forward(p)
        P[0] = 1 # let root node always pass

        # consider node dependence
        cond = (self.dependence.matmul(P) == self.dependence_cnt)
        P *= cond

        # generate the mask for the sampled label set
        mask = 1 - self.masks[P.long(), range(self.n_intnl_node)].int()
        mask = 1 - torch.sum(mask, dim=0).bool().int()
        mask = mask.bool()

        tgt_mapping = torch.where(self.codewords[mask, :])[0]

        return mask, tgt_mapping


def treecut_generator(**kwargs):
    model = TreecutGenerator(**kwargs)
    return model


