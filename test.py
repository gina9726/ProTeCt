### test.py
# Main script for inference.
###

import time
import argparse
import os
import pdb
import numpy as np
import pandas as pd
import yaml
import random
import shutil
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append('loader')

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from loader import get_dataloader, get_prompt_template
from models import get_model
from metrics import accuracy, averageMeter
from utils import get_logger

def main():

    if not torch.cuda.is_available():
        raise SystemExit('GPU is needed')

    os.system('echo $CUDA_VISIBLE_DEVICES')

    # setup data loader
    splits = ['test']
    data_loader = get_dataloader(cfg['data'], splits, cfg['data']['batch_size'])
    # hierarchical information
    param_names = data_loader['param_names']
    leaf_nodes = data_loader['leaf_nodes']
    intnl_nodes = data_loader['intnl_nodes']

    #load treecuts
    forest = torch.load(cfg['treecuts'])
    assert args.ntree <= len(forest)

    # setup model
    n_gpu = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init_classname = None
    # change the initial label space
    if cfg.get("init_label_set", "leaf") == "leaf":
        init_classname = [param_names[i] for i in leaf_nodes]

    model = get_model(cfg['model'], init_classname).cuda()

    # load the best model for evaluation
    checkpoint = os.path.join(args.folder, "ckpt", args.eval_ckpt + ".pth")
    if os.path.isfile(checkpoint):
		model.load_custom_ckpt(checkpoint)
    else:
        print(f'checkpoint ({checkpoint}) not found!')
    model.eval()

    # leaf accuracy
    df = pd.DataFrame(param_names)
    label_set = list(df.iloc[leaf_nodes][0])
    leaf_acc = inference(model, data_loader['test'], label_set)
    print(f'Leaf acc: {leaf_acc}')
    logger.info(f'Leaf acc: {leaf_acc}')

    # MTA evaluation
    print(f'Testing {args.ntree} treecuts')
    logger.info(f'Testing {args.ntree} treecuts')
    treecut_accs = []
    for i in range(args.ntree):
        label_set, tgt_mapping = forest[i]
        tgt_mapping = torch.tensor(tgt_mapping).long().cuda()
        acc = inference(model, data_loader['test'], label_set, tgt_mapping)
        treecut_accs.append(acc)
        print(f'acc-{i+1}: {acc}')
        logger.info(f'acc-{i+1}: {acc}')
    treecut_accs = np.array(treecut_accs)

    print(f"[ Treecut acc ]: Avg: {treecut_accs.mean()}\tStd: {treecut_accs.std()}")
    logger.info(f"[ Treecut acc ]: Avg: {treecut_accs.mean()}\tStd: {treecut_accs.std()}")
    np.save(logdir / 'treecut_accs.npy', treecut_accs)
    logger.info(f"======")


@torch.no_grad()
def inference(model, data_loader, param_names, tgt_mapping=None):
    # setup average meters
    meter = {}
    meter["acc"] = averageMeter()

    model.eval()

    for (step, value) in enumerate(tqdm(data_loader)):
        image = value[0].cuda()
        target = value[1].cuda()
        index = value[2]
        if tgt_mapping is not None:
            target = tgt_mapping[target]

        output = model(image, param_names)
        logits = output['logits_per_image']
        acc = accuracy(logits, target, topk=(1,))[0]
        meter["acc"].update(acc.item(), image.size(0))

    return meter["acc"].avg



if __name__ == '__main__':
    global cfg, args, logger, logdir
    parser = argparse.ArgumentParser(description='config')
    parser.add_argument(
        '--folder',
        type=str,
        help='Path to the folder',
    )
    parser.add_argument(
        '--eval_dataset',
        type=str,
        default='',
        help='Dataset for evaluation',
    )
    parser.add_argument(
        '--ntree',
        type=int,
        default=25,
        help='number of treecuts to evaluate'
    )
    parser.add_argument(
        '--bz',
        type=int,
        default=1024,
        help='batch size',
    )
    parser.add_argument(
        '--debug',
        default=False,
        action="store_true",
        help='debug mode',
    )
    args = parser.parse_args()
    args.eval_ckpt = ''

    # check if there is any yml files in the folder
    yml_file = None
    for f in os.listdir(args.folder):
        if f[-3:] == "yml":
            yml_file = args.folder + "/" + f
        elif f == "ckpt":
            assert os.path.exists(os.path.join(args.folder, f, "last.pth"))
            if os.path.exists(os.path.join(args.folder, f, "best.pth")):
                args.eval_ckpt = "best"
            else:
                args.eval_ckpt = "last"

    #assert args.eval_ckpt is not None
    assert yml_file is not None

    with open(yml_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.SafeLoader)

    cfg['data']['eval_batch_size'] = args.bz
    if args.eval_dataset != '':
        cfg['data']['data_root'] = os.path.join("./prepro/raw", args.eval_dataset)
        cfg['data']['test'] = os.path.join("./prepro/data", args.eval_dataset, "gt_test.txt")
        cfg['data']['hierarchy'] = os.path.join("./prepro/data", args.eval_dataset, "tree.npy")
        if args.eval_dataset == 'imagenet-a' or args.eval_dataset == 'imagenet-r':
            cfg['treecuts'] = Path('prepro/data', args.eval_dataset, f'treecuts_25_multi.pkl')
        else:
            cfg['treecuts'] = Path('prepro/data', cfg['data']['name'], f'treecuts_25_multi.pkl')
        logdir = Path(args.folder) / args.eval_dataset / f'treecuts_{args.ntree}'
    else:
        cfg['treecuts'] = Path('prepro/data', cfg['data']['name'], f'treecuts_25_multi.pkl')
        logdir = Path(args.folder) / f'treecuts_{args.ntree}'

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    print("RUNDIR: {}".format(logdir))

    logger = get_logger(logdir)
    logger.info("Start logging")

    main()

