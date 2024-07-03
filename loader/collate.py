### rotdata.py
# Collate function for loading rotated images and labels.
# Author: Tz-Ying Wu
###

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate


def get_collate_fn(rot):
    if rot:
        return rot_collate
    else:
        return default_collate


def get_rotated_data(data, angle):
    assert angle in [0, 90, 180, 270]
    img, target, index = data
    target = torch.tensor([target])
    angle = torch.tensor([angle])
    index = torch.tensor([index])
    if angle == 90:
        img = np.transpose(img, (1, 2, 0))
        img = np.transpose(np.flipud(img), (2, 1, 0))
        img = torch.from_numpy(img.copy())
    elif angle == 180:
        img = np.transpose(img, (1, 2, 0))
        img = np.flipud(np.fliplr(img))
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img.copy())
    elif angle == 270:
        img = np.transpose(img, (1, 2, 0))
        img = np.transpose(np.fliplr(img), (2, 1, 0))
        img = torch.from_numpy(img.copy())

    return (img, target, index, angle//90)


class RotationBatch:
    def __init__(self, data):
        rot_data = []
        for i in range(4):
            rot_data.extend(list(map(lambda x: get_rotated_data(x, i*90), data)))
        transposed_data = list(zip(*rot_data))
        self.img = torch.stack(transposed_data[0], 0)
        self.cls_lbl = torch.stack(transposed_data[1], 0).squeeze()
        self.index = torch.stack(transposed_data[2], 0).squeeze()
        self.rot_lbl = torch.stack(transposed_data[3], 0).squeeze()

    def pin_memory(self):
        self.img = self.img.pin_memory()
        self.cls_lbl = self.cls_lbl.pin_memory()
        self.index = self.index.pin_memory()
        self.rot_lbl = self.rot_lbl.pin_memory()
        return self


def rot_collate(batch):
    batch = RotationBatch(batch)
    return batch.img, batch.cls_lbl, batch.index, batch.rot_lbl


# unit-test
if __name__ == '__main__':
    import torch.utils.data as data
    import torchvision.transforms as transforms
    from img_flist import ImageFilelist
    import pdb
    cfg = {
        'data_root': '/data8/gina/dataset/DomainNet',
        'val': '/data8/gina/dataset/DomainNet/clipart_test.txt',
        'n_workers': 4,
    }
    trans = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 9.456, 0.406], [0.229, 0.224, 0.225])
        ])
    dataset = ImageFilelist(root_dir=cfg['data_root'], flist=cfg['val'], transform=trans)
    data_loader = data.DataLoader(dataset, batch_size=4, collate_fn=rot_collate, pin_memory=True)
    for (step, value) in enumerate(data_loader):
        img, label, index, rot_lbl = value
        pdb.set_trace()


