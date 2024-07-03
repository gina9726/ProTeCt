### prepro.py
# Scripts to preprocess hierarchical datasets and generate the splits.
# Author: Tz-Ying Wu
###

import numpy as np
import os
import argparse
import json
import sys
sys.path.append('../loader')
from treelibs import Tree


def write_file(file_name, data_list):

    with open(file_name, 'w') as f:
        for data in data_list:
            f.write('{},{},{}\n'.format(data[1][0], data[1][1], data[0]))

def main(out_dir):

    # build tree
    data_root = args.data
    tree = Tree(data_root)
    if args.display:
        tree.show()
    np.save(os.path.join(out_dir, 'tree.npy'), tree)

    # save leaf nodes
    leaf_id = {v: k for k, v in tree.leaf_nodes.items()}       # node_name: leaf_node_id
    print('number of classes: {}'.format(len(leaf_id)))
    np.save(os.path.join(out_dir, 'leaf_nodes.npy'), leaf_id)

    # load subsamples
    if args.subsample:
        fnames = json.load(open(args.subsample, 'r'))['fnames']
        split = args.subsample.split('/')[-1].replace('.json', '')
    else:
        split = 'all'

    # label data
    data = []
    for root, dirs, files in os.walk(data_root, topdown=True):
        if len(dirs) == 0:
            cls = root.split('/')[-1]
            while cls not in leaf_id:
                cls = tree.get_parent(cls)

            print('labeling {} as {} ...'.format(root, cls))
            label = leaf_id.get(cls, -1)

            if label < 0:
                raise ValueError('{} is not labeled'.format(cls))

        if args.subsample:
            for img in files:
                if img in fnames:
                    impath = os.path.join(root, img)
                    data.append((impath, label))
        else:
            for img in files:
                impath = os.path.join(root, img)
                data.append((impath, label))

    # export data
    data = [(i, x) for i, x in enumerate(data)]
    write_file(os.path.join(out_dir, 'gt_{}.txt'.format(split)), data)


if __name__ == '__main__':
    global args

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data',
        type=str,
        help='image data root',
    )
    parser.add_argument(
        '--out',
        type=str,
        help='output file basename',
    )
    parser.add_argument(
        '--display',
        type=bool,
        default=True,
        help='show tree hierachies',
    )
    parser.add_argument(
        '--subsample',
        type=str,
        help='subsample json file',
    )

    args = parser.parse_args()
    out_dir = os.path.join('prepro/data', '{}'.format(args.out))
    print('Data is saved to {}'.format(out_dir))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    main(out_dir)

