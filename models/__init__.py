### __init__.py
# Get model instance with designated parameters.
# Author: Tz-Ying Wu
###

import copy
import torch.nn as nn
import logging

from .resnet import resnet10, resnet18, resnet50
from .linearcls import linearcls
from .mlpcls import mlpcls
from .clip import load as clip_model
from .coop import load as coop_model
from .cocoop import load as cocoop_model
from .maple import load as maple_model
from .treecut_generator import treecut_generator

logger = logging.getLogger('mylogger')


def get_model(model_dict, init_classname=None, verbose=False):

    name = model_dict['arch']
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop('arch')

    if 'resnet' in name:
        model = model(**param_dict)
        model.fc = nn.Identity()

    elif name == 'clip':
        model, _ = model(**param_dict)

    elif name in ['coop', 'cocoop', 'maple']:
        model = model(param_dict, init_classname)

    elif name == 'treecut_generator':
        model = model(**param_dict)

    else:
        model = model(**param_dict)

    if verbose:
        logger.info(model)

    return model


def _get_model_instance(name):
    try:
        return {
            'resnet10': resnet10,
            'resnet18': resnet18,
            'resnet50': resnet50,
            'linearcls': linearcls,
            'mlpcls': mlpcls,
            'clip': clip_model,
            'coop': coop_model,
            'cocoop': cocoop_model,
            'maple': maple_model,
            'treecut_generator': treecut_generator,
        }[name]
    except:
        raise BaseException('Model {} not available'.format(name))


