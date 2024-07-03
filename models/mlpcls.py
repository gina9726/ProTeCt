### mlpcls.py
# Module of MLP classifier.
# Author: Gina Wu @ 01/22
###

import torch
from torch import nn
import torch.nn.functional as F


class MLPCls(nn.Module):

    def __init__(self, feat_size, n_class, norm=0, nonlinear='relu', bn=False, dp=False):
        super(MLPCls, self).__init__()
        self.n_class = n_class
        self.norm = norm

        layers = []
        for i in range(len(feat_size)-1):
            layers.append(nn.Linear(feat_size[i], feat_size[i+1]))
            if bn:
                layers.append(nn.BatchNorm1d(feat_size[i+1]))
            if nonlinear != 'none':
                layers.append(nn.ReLU(inplace=True))
            if dp:
                layers.append(nn.Dropout())

        self.mlp = nn.Sequential(*layers)

        if self.norm > 0:
            self.out = nn.Linear(feat_size[-1], n_class, bias=False)
        else:
            self.out = nn.Linear(feat_size[-1], n_class)

    def forward(self, x, feat=False):

        x = self.mlp(x)

        if self.norm > 0:
            x = F.normalize(x, p=2, dim=1) * self.norm
            weight = F.normalize(self.out.weight, p=2, dim=1) * self.norm
            y = torch.mm(x, weight.t())
        else:
            y = self.out(x)

        if feat:
            return y, x
        return y


def mlpcls(**kwargs):
    model = MLPCls(**kwargs)
    return model


if __name__ == '__main__':
    import pdb
    model = mlpcls(**{'feat_size': [100, 100], 'n_class': 10})
    x = torch.randn([4, 100])
    y = model(x)
    pdb.set_trace()


