import torch, torchvision
from pathlib import Path
from PIL import Image
from .base import _BaseDataset


__all__ = ['InatCUBVal']


def _read_inat_file(fname):
    with open(fname) as f:
        lines = f.read().strip().split('\n')
    imgs, labels = [], []
    for line in sorted(lines):
        lab, img = line.split('/')
        imgs.append(line)
        labels.append(lab)
    return imgs, labels


class InatCUBVal(_BaseDataset):
    def __init__(self, root, transform=None, target_transform=None,
                 data_file='images-full.txt'):
        self.root = Path(root)
        if self.root.name == 'test':
            self.root = self.root.parent
        self.transform = transform
        self.target_transform = target_transform

        self.imfolder = 'images'

        images, labels = _read_inat_file(self.root/data_file)
        self.classes = sorted(list(set(labels)))
        self.class_to_idx = {c:i for i, c in enumerate(self.classes)}
        self.targets = [self.class_to_idx[c] for c in labels]
        self.imgs = images

