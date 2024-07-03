### train.py
# Main script for training.
###

import time
import argparse
import os
import pdb
import numpy as np
import yaml
import random
import shutil
from tqdm import tqdm
import sys
sys.path.append('loader')

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from loader import get_dataloader, get_prompt_template
from models import get_model
from utils import get_logger, print_args
from optim.optimizer import build_optimizer
from optim.lr_scheduler import build_lr_scheduler
from tensorboardX import SummaryWriter

from engine import train_one_epoch, eval_one_epoch

def main():

    if not torch.cuda.is_available():
        raise SystemExit('GPU is needed')

    os.system('echo $CUDA_VISIBLE_DEVICES')

    # print args
    print_args(args, cfg)
    if "eval_freq" not in cfg:
        cfg["eval_freq"] = -1

    # setup random seed
    torch.manual_seed(cfg.get('seed', 1))
    torch.cuda.manual_seed(cfg.get('seed', 1))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # setup data loader
    splits = ['train', 'test']
    data_loader = get_dataloader(cfg['data'], splits, cfg['data']['batch_size'])
    # hierarchical information
    param_names = data_loader['param_names']
    sublabels = data_loader['sublabels'].cuda()
    leaf_nodes = data_loader['leaf_nodes']
    intnl_nodes = data_loader['intnl_nodes']

    # setup model
    n_gpu = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # change the initial label space
    if cfg["init_label_set"] == "leaf":
        init_classname = [param_names[i] for i in leaf_nodes]

    treecut_generator = None
    if cfg.get('treecut_generator', None) is not None:
        model_dict = cfg['treecut_generator']
        model_dict.update(data_loader['tree_info'])
        treecut_generator = get_model(model_dict).cuda()
    
    model = get_model(cfg['model'], init_classname).cuda()

    #if n_gpu > 1:
    #    print(f"Multiple GPUs detected (n_gpus={n_gpu}), use all of them!")
    #    model = nn.DataParallel(model)

    # setup optimizer
    optim_param = []
    optim_name = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            optim_param.append(param)
            optim_name.add(name)
    print(f"Parameters to be updated: {optim_name}")
    optim = build_optimizer(optim_param, cfg['optim'])
    sched = build_lr_scheduler(optim, cfg['optim'])

    # training epochs
    best_result = 0
    for epoch in tqdm(range(cfg["optim"]["max_epoch"])):
        model.train()
        if treecut_generator is not None:
            train_meters = train_one_epoch(model, optim, sched, data_loader['train'], param_names, device, epoch, cfg, args, treecut_generator, leaf_nodes, intnl_nodes, sublabels)
        else:
            train_meters = train_one_epoch(model, optim, sched, data_loader['train'], param_names, device, epoch, cfg, args)

        # logging for training
        curr_lr = optim.param_groups[0]['lr']
        writer.add_scalar('train/lr', curr_lr, epoch + 1)
        logger.info(f"=== Epoch {epoch} done !! ===")
        for k in train_meters:
            writer.add_scalar(f"train/{k}", train_meters[k], epoch + 1)
            if k[0] == 'n':
                continue
            print(f"Average result of {k} over epoch: " + str(train_meters[k]))
            logger.info(f"Average result of {k} over epoch: " + str(train_meters[k]))
        logger.info(f"======")

        if cfg["eval_freq"] != -1 and (epoch + 1) % cfg["eval_freq"] == 0:
            model.eval()
            eval_meters = eval_one_epoch(model, data_loader['test'], param_names, device, epoch, cfg, args, leaf_nodes, intnl_nodes, sublabels)
            # logging for training
            logger.info(f"=== Evaluating {epoch}  !! ===")
            for k in eval_meters:
                writer.add_scalar(f"val/{k}", eval_meters[k], epoch + 1)
                if k[0] == 'n':
                    continue
                print(f"Average result of {k} on the test set: " + str(eval_meters[k]))
                logger.info(f"Average result of {k} on the test set: " + str(eval_meters[k]))

            result = (eval_meters["acc"] + eval_meters["consistency"]) / 2

            print(f"Best result : {best_result} ; Epoch result : {result}" )
            logger.info(f"Best result : {best_result} ; Epoch result : {result}" )

            if result >= best_result:
                best_result = result
                print("Saving the best model")
                # save the best model
                torch.save(model.state_dict(), os.path.join(logdir, "ckpt", "best.pth"))
        torch.save(model.state_dict(), os.path.join(logdir, "ckpt", "last.pth"))
        logger.info(f"======")

        # debug mode
        if args.debug:
            break

    # load the best model for evaluation
    if os.path.exists(os.path.join(logdir, "ckpt", "best.pth")):
        logger.info(f"=== Evaluating Best model  !! ===")
        model.load_state_dict(torch.load(os.path.join(logdir, "ckpt", "best.pth")))
    else:
        logger.info(f"=== Evaluating Last model  !! ===")
        model.load_state_dict(torch.load(os.path.join(logdir, "ckpt", "last.pth")))
    model.eval()
    eval_meters = eval_one_epoch(model, data_loader['test'], param_names, device, -1, cfg, args, leaf_nodes, intnl_nodes, sublabels)
    result = {x for x in eval_meters.items() if x[0][0] != 'n'}

    print(f"Final result : {result}" )
    logger.info(f"Final result : {result}" )
    logger.info(f"======")




if __name__ == '__main__':
    global cfg, args, writer, logger, logdir
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
        '--trial',
        type=str,
        help='different trial of the exp',
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.SafeLoader)

    if args.debug:
        outdir = 'runs/debug'
    else:
        outdir = 'runs'

    logdir = os.path.join(outdir, cfg['data']['name'], cfg['model']['arch'], cfg['exp'], "trial_" + args.trial)

    if not os.path.exists(os.path.join(logdir, "ckpt")):
        os.makedirs(os.path.join(logdir, "ckpt"))

    writer = SummaryWriter(log_dir=logdir)

    print("RUNDIR: {}".format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info("Start logging")

    main()
