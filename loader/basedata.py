### basedata.py
# Data loader for general datasets.
# Author: Tz-Ying Wu
###

import os
import logging
import numpy as np
from collections import Counter
import torch.utils.data as data

from .img_flist import ImageFilelist
from .sampler import get_sampler
from .collate import get_collate_fn
from .transforms import get_transform


logger = logging.getLogger('mylogger')

def BaseDataLoader(cfg, splits, batch_size):
    """Function to build data loader(s) for the specified splits given the parameters.
    """
    data_root = cfg.get('data_root', '/path/to/dataset')
    if not os.path.isdir(data_root):
        raise Exception('{} does not exist'.format(data_root))

    num_workers = cfg.get('n_workers', 4)

    smplr_dict = cfg.get('sampler', {'name': 'random'})

    data_loader = dict()
    for split in splits:

        aug = True if 'train' in split else False
        trans = get_transform(cfg.get('transform', 'imagenet'), aug)

        data_list = cfg.get(split, None)
        if not os.path.isfile(data_list):
            raise Exception('{} not available'.format(data_list))

        dataset = ImageFilelist(root_dir=data_root, flist=data_list, transform=trans)
        if 'train' in split:
            cls_num_list = []
            counter = Counter(dataset.target)
            n_class = len(counter)
            for i in range(n_class):
                cls_num_list.append(counter.get(i, 1e-7))
            data_loader['cls_num_list'] = np.asarray(cls_num_list)

        shuffle = True if 'train' in split else False
        drop_last = cfg.get('drop_last', False) if 'train' in split else False
        rot = cfg.get('rot', False) if 'train' in split else False
        collate_fn = get_collate_fn(rot)
        if ('train' in split) and (smplr_dict['name'] != 'random'):
            sampler = get_sampler(dataset, smplr_dict)
            data_loader[split] = data.DataLoader(
                dataset, batch_size=batch_size, sampler=sampler, shuffle=False, collate_fn=collate_fn,
                drop_last=drop_last, pin_memory=True, num_workers=num_workers
            )
        else:
            data_loader[split] = data.DataLoader(
                dataset, batch_size=batch_size,  sampler=None, shuffle=shuffle, collate_fn=collate_fn,
                drop_last=drop_last, pin_memory=True, num_workers=num_workers
            )

        logger.info("{split}: {size}".format(split=split, size=len(dataset)))

    # load parameter names
    classes = np.load(cfg['classes'])
    data_loader['param_names'] = classes

    logger.info("Building data loader with {} workers".format(num_workers))

    return data_loader


# unit-test
if __name__ == '__main__':
    import pdb
    cfg = {
        'data_root': '/data/ILSVRC/2012',
        'train': '/data/imagenet/gt_train.txt',
        'val': '/data/imagenet/gt_valid.txt',
        'n_workers': 4,
        'rot': False,
        'sampler': {'name': 'class_balanced'},
        'transform': 'imagenet'
    }
    splits = ['train', 'val']
    data_loader = BaseDataLoader(cfg, splits, batch_size=4)
    for (step, value) in enumerate(data_loader['train']):
        if len(value) > 3:
            img, label, index, rot_label = value
        else:
            img, label, index = value
        pdb.set_trace()

