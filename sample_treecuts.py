### sample_treecuts.py
# Scripts for sampling treecuts for evaluation.
# Author: Tz-Ying Wu
###

import argparse
import os
import pandas as pd
import numpy as np
import yaml
import random
import shutil
import pdb
import sys
sys.path.append('loader')

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from utils import get_logger, print_args
from loader import get_dataloader, get_prompt_template
from models import get_model

def main():

    print_args(args, cfg)

    # setup random seed
    torch.manual_seed(cfg.get('seed', 1))
    torch.cuda.manual_seed(cfg.get('seed', 1))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # setup data loader
    splits = ['test']
    data_loader = get_dataloader(cfg['data'], splits, cfg['data']['batch_size'])
    param_names = data_loader['param_names']

    # setup treecut generator
    model_dict = cfg['treecut_generator']
    model_dict.update(data_loader['tree_info'])
    treecut_generator = get_model(model_dict).cuda()

    # sample treecuts
    treecut_generator.train()
    df = pd.DataFrame(param_names)
    forest = set()
    trees = {}
    runs = 0
    if args.multi:
        dropout_rates = [0.1, 0.3, 0.5, 0.7, 0.9]
        ntree_per_dp = cfg['n_tree'] / len(dropout_rates)
        print(f'dropout_rates: {dropout_rates}')
        logger.info(f'dropout_rates: {dropout_rates}')
        print(f'ntree_per_dp: {ntree_per_dp}')
        logger.info(f'ntree_per_dp: {ntree_per_dp}')
        for dp in dropout_rates[::-1]:
            tree_cnt = 0
            while tree_cnt < ntree_per_dp:
                label_set, tgt_mapping = treecut_generator.get_randomcut(dp)
                param_idx = torch.where(label_set)[0]
                text = tuple(df.iloc[param_idx.cpu()][0])
                if text not in forest:
                    tree_cnt += 1
                    trees[len(forest)] = [text, tgt_mapping.cpu().numpy().tolist()]
                    forest.add(text)
                runs += 1
            print(f'sample {len(forest)} trees with dp={dp} and {runs} runs')
            logger.info(f'sample {len(forest)} trees with dp={dp} and {runs} runs')
        save_path = os.path.join(logdir, f"treecuts_{len(forest)}_multi.pkl")
    else:
        while len(forest) < cfg['n_tree']:
            label_set, tgt_mapping = treecut_generator.get_randomcut()
            param_idx = torch.where(label_set)[0]
            text = tuple(df.iloc[param_idx.cpu()][0])
            if text not in forest:
                trees[len(forest)] = [text, tgt_mapping.cpu().numpy().tolist()]
                forest.add(text)
            runs += 1

        print(f'sample {len(forest)} trees with {runs} runs')
        logger.info(f'sample {len(forest)} trees with {runs} runs')
        save_path = os.path.join(logdir, f"treecuts_{len(forest)}.pkl")

    torch.save(trees, save_path)
    print(f'results saved to {save_path}')

if __name__ == '__main__':
    global cfg, args, logger, logdir
    parser = argparse.ArgumentParser(description='config')
    parser.add_argument(
        '--config',
        nargs='?',
        type=str,
        help='Configuration file to use',
    )
    parser.add_argument(
        '--debug',
        action = "store_true",
        default=False,
        help='Run in debug mode',
    )
    parser.add_argument(
        '--multi',
        action = "store_true",
        default=False,
        help='Sample treecuts with multiple dropout rate i.e. [0.1, 0.3, 0.5, 0.7, 0.9]',
    )
    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.SafeLoader)

    if args.debug:
        outdir = 'runs/debug'
    else:
        outdir = 'runs'

    logdir = os.path.join(outdir, cfg['data']['name'], 'treecuts', cfg['name'])

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    print("RUNDIR: {}".format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info("Start logging")

    main()
 
