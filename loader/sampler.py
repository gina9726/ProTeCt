### sampler.py
# Functions to setup data sampler.
# Author: Tz-Ying Wu
###

import numpy as np
from collections import Counter
import torch
from torch.utils.data.sampler import Sampler
import logging
import copy


logger = logging.getLogger('mylogger')

def get_sampler(dataset, smplr_dict):

    name = smplr_dict['name']
    logging.info('Using {} sampler'.format(name))

    if name == 'random':
        return None

    param_dict = copy.deepcopy(smplr_dict)
    param_dict.pop('name')
    sampler = _get_sampler_instance(name)
    sampler = sampler(dataset, **param_dict)

    return sampler


def _get_sampler_instance(name):
    try:
        return {
            'class_balanced': BalancedSampler,
            'nshot': NshotSampler
        }[name]
    except:
        raise BaseException('{} sampler not available'.format(name))


class BalancedSampler(Sampler):
    def __init__(self, dataset, indices=None, tgt_transform=None):

        self.indices = list(range(len(dataset))) if indices is None else indices

        class_ids = np.asarray(dataset.target)[self.indices]
        if tgt_transform is not None:
            class_ids = list(map(tgt_transform, class_ids))

        self.n_samples = len(class_ids)

        # compute class frequencies and set them as sampling weights
        counts = Counter(class_ids)
        get_freq = lambda x: 1.0 / counts[x]
        self.weights = torch.DoubleTensor(list(map(get_freq, class_ids)))

    def __iter__(self):
        sampled_idx = torch.multinomial(self.weights, self.n_samples, replacement=True)
        return (self.indices[i] for i in sampled_idx)

    def __len__(self):
        return self.n_samples


class NshotSampler(Sampler):
    def __init__(self, dataset, nshot=0, seed=0, tgt_transform=None):

        self.nshot = nshot
        self.seed = seed

        class_ids = np.asarray(dataset.target)
        if tgt_transform is not None:
            class_ids = list(map(tgt_transform, class_ids))
        classes = np.unique(class_ids)

        if nshot > 0:
            self.sampled_idx = []
            for i in classes:
                g_cpu = torch.Generator()
                g_cpu.manual_seed(int(seed + i))
                idx = np.where(class_ids == i)[0]
                selected = torch.randperm(len(idx), generator=g_cpu)[:nshot]
                if nshot > 1:
                    self.sampled_idx.extend(idx[selected].tolist())
                else:
                    self.sampled_idx.append(idx[selected])
            assert len(self.sampled_idx) == nshot * len(classes)
        else:
            self.sampled_idx = list(range(len(class_ids)))

    def __iter__(self):
        g_cpu = torch.Generator()
        g_cpu.manual_seed(int(self.seed))
        idx = torch.randperm(len(self.sampled_idx), generator=g_cpu)
        return (self.sampled_idx[i] for i in idx)

    def __len__(self):
        return len(self.sampled_idx)


