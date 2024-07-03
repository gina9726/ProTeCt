### transforms.py
# Define transform functions.
# Author: Tz-Ying Wu
###

from PIL import Image
from torchvision import transforms
import logging

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


logger = logging.getLogger('mylogger')

def get_transform(name, aug):
    logging.info('Using {} transform'.format(name))
    transform = _get_transform_instance(name)

    if name == 'clip':
        return transform(n_px=224)
    else:
        return transform(aug)

def _get_transform_instance(name):
    try:
        return {
            'cifar': cifar_transform,
            'imagenet': imagenet_transform,
            'clip': clip_transform
        }[name]
    except:
        raise BaseException('{} transform not available'.format(name))

### From https://github.com/openai/CLIP/blob/main/clip/clip.py
def _convert_image_to_rgb(image):
    return image.convert("RGB")

def clip_transform(n_px):
    return transforms.Compose([
        transforms.Resize(n_px, interpolation=BICUBIC),
        transforms.CenterCrop(n_px),
        _convert_image_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
###

def cifar_transform(aug=True):
    if aug:
        trans = transforms.Compose([
            transforms.Pad(4),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    return trans

def imagenet_transform(aug=True):
    if aug:
        trans = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return trans

