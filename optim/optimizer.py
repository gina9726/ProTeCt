"""
Modified from https://github.com/KaiyangZhou/Dassl.pytorch/blob/ee55e298884301f44625557154a8b8f10d867c6c/dassl/optim/optimizer.py
"""

import warnings
import torch
import torch.nn as nn


AVAI_OPTIMS = ["adam", "amsgrad", "sgd", "rmsprop", "adamw"]



def build_optimizer(param_groups, optim_cfg):
    """A function wrapper for building an optimizer.
    Args:
        param_groups: If provided, directly optimize param_groups and abandon model
        optim_cfg (CfgNode): optimization config.
    """
    name = optim_cfg["name"] if "name" in optim_cfg else None
    lr = optim_cfg["lr"] if "lr" in optim_cfg else None
    weight_decay = optim_cfg["weight_decay"] if "weight_decay" in optim_cfg else 0
    momentum = optim_cfg["momentum"] if "momentum" in optim_cfg else 0
    sgd_dampening = optim_cfg["sgd_dampning"] if "sgd_dampning" in optim_cfg else None
    sgd_nesterov = optim_cfg["sgd_nesterov"] if "sgd_nesterov" in optim_cfg else False
    rmsprop_alpha = optim_cfg["rmsprop_alpha"] if "rmsprop_alpha" in optim_cfg else None
    adam_beta1 = optim_cfg["adam_beta1"] if "adam_beta1" in optim_cfg else None
    adam_beta2 = optim_cfg["adam_beta2"] if "adam_beta2" in optim_cfg else None
    staged_lr = optim_cfg["staged_lr"] if "staged_lr" in optim_cfg else None 
    new_layers = optim_cfg["new_layers"] if "new_layers" in optim_cfg else None  
    base_lr_mult = optim_cfg["base_lr_mult"] if "base_lr_mult" in optim_cfg else None

    if name == "adam":
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
        )

    elif name == "amsgrad":
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
            amsgrad=True,
        )

    elif name == "sgd":
        optimizer = torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=sgd_dampening,
            nesterov=sgd_nesterov,
        )

    elif name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            alpha=rmsprop_alpha,
        )

    elif name == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
        )
    else:
        raise NotImplementedError(f"Optimizer {optim} not implemented yet!")

    return optimizer



