### Hierdata.py
# Data loader for hierarchical datasets.
# Author: Tz-Ying Wu
###

import os
import logging
import numpy as np
from collections import Counter
import torch.utils.data as data
import torch

from .utils import prepro_node_name
from .img_flist import ImageFilelist
from .sampler import get_sampler
from .collate import get_collate_fn
from .transforms import get_transform


logger = logging.getLogger('mylogger')

def HierDataLoader(cfg, splits, batch_size):
    """Function to build data loader(s) for the specified splits given the parameters.
    """
    data_root = cfg.get('data_root', '/path/to/dataset')
    if not os.path.isdir(data_root):
        raise Exception('{} does not exist'.format(data_root))

    num_workers = cfg.get('n_workers', 4)

    smplr_dict = cfg.get('sampler', {'name': 'random'})

    data_loader = dict()
    for split in splits:

        bz = batch_size if 'train' in split else cfg.get('eval_batch_size', 128)
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
                dataset, batch_size=bz, sampler=sampler, shuffle=False, collate_fn=collate_fn,
                drop_last=drop_last, pin_memory=True, num_workers=num_workers
            )
        else:
            data_loader[split] = data.DataLoader(
                dataset, batch_size=bz,  sampler=None, shuffle=shuffle, collate_fn=collate_fn,
                drop_last=drop_last, pin_memory=True, num_workers=num_workers
            )

        logger.info("{split}: {size}".format(split=split, size=len(dataset)))

    # retrieve semantic information from the class hierarchy
    tree = np.load(cfg['hierarchy'], allow_pickle=True).tolist()

    # get the parameter list of each internal node
    tree.gen_param_lists()
    n_intnl_node = len(tree.intnl_nodes)
    intnl_nodes = [tree.nodes.get(tree.intnl_nodes[i]).param_list for i in range(n_intnl_node)]

    # get the parameter list of leaf nodes
    n_leaf_node = len(tree.leaf_nodes)
    leaf_nodes = np.asarray([tree.get_nodeId(tree.leaf_nodes[i]) - 1 for i in range(n_leaf_node)])

    # get semantic names in the hierarchy
    nodes = sorted([v for v in tree.nodes.values()], key=lambda x: x.node_id)[1:]
    param_names = [prepro_node_name(x.name) for x in nodes]

    # generate codewords for all the nodes
    tree.gen_codewords('class')    # for treecut
    codewords = np.asarray([x.codeword for x in nodes])

    # generate the dependence matrix of the internal nodes
    tree.gen_dependence()

    # generate parameter masks for internal nodes
    masks = -np.ones((n_intnl_node, len(param_names)))
    intnl_params = np.asarray([tree.get_nodeId(tree.intnl_nodes[i]) - 1 for i in range(n_intnl_node)])
    for i in range(n_intnl_node):
        # passed parameters
        leaf_idx = (tree.sublabels[:, i] >= 0)
        masks[i, leaf_nodes[leaf_idx]] = 1
        intnl_idx = (tree.dependence[:, i] == 1)
        masks[i, intnl_params[intnl_idx]] = 1
        # blocked parameters
        intnl_idx = (tree.dependence[i, 1:] == 1)
        masks[i, intnl_params[1:][intnl_idx]] = 0

    data_loader['tree_info'] = {
        'dependence': torch.from_numpy(tree.dependence).float(),
        'masks': torch.from_numpy(masks).int(),
        'codewords': torch.from_numpy(codewords).int()
    }
    data_loader['sublabels'] = torch.from_numpy(tree.sublabels).float()
    data_loader['intnl_nodes'] = intnl_nodes
    data_loader['leaf_nodes'] = leaf_nodes
    data_loader['param_names'] = param_names

    logger.info("Building data loader with {} workers".format(num_workers))

    return data_loader


# unit-test
if __name__ == '__main__':
    import pdb
    cfg = {
        'data_root': '/raw/cifar100',
        'train': '/data/cifar100/gt_train.txt',
        'val': '/data/cifar100/gt_valid.txt',
        'hierarchy': '/data/cifar100/tree.npy',
        'n_workers': 4,
        'rot': False,
        'sampler': {'name': 'class_balanced'},
        'transform': 'clip'
    }
    splits = ['train', 'val']
    data_loader = HierDataLoader(cfg, splits, batch_size=4)
    for (step, value) in enumerate(data_loader['train']):
        if len(value) > 3:
            img, label, index, rot_label = value
        else:
            img, label, index = value
        pdb.set_trace()

