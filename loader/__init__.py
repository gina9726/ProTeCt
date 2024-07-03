### __init__.py
# Function to get different types of data loader instances.
# Author: Tz-Ying Wu
###

from .basedata import BaseDataLoader
from .hierdata import HierDataLoader
from .utils import get_prompt_template


def get_dataloader(cfg, splits, batch_size):
    loader = _get_loader_instance(cfg['loader'])
    data_loader = loader(cfg, splits, batch_size)
    return data_loader


def _get_loader_instance(name):
    try:
        return {
            'BaseDataLoader': BaseDataLoader,
            'HierDataLoader': HierDataLoader,
        }[name]
    except:
        raise BaseException('Loader type {} not available'.format(name))


