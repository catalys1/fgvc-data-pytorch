from pathlib import Path

from PIL import Image
import torch, torchvision

from .base import _BaseDataset


__all__ = ['InatCUB']


def _read_inat_file(fname):
    with open(fname) as f:
        lines = f.read().strip().split('\n')
    imgs, labels = [], []
    for line in sorted(lines):
        lab, img = line.split('/')
        imgs.append(line)
        labels.append(lab)
    return imgs, labels


class InatCUB(_BaseDataset):

    name = 'iCub'
    image_file = 'images.txt'
    bounding_box_file = 'bounding_boxes.txt'

    def __init__(self, root, transform=None, target_transform=None, train=False, load_bboxes=False):
        # train is ignored, there for compatibility
        self.root = Path(root)
        if self.root.name in ['train', 'val', 'test']:
            self.root = self.root.parent
        self.transform = transform
        self.target_transform = target_transform

        self.imfolder = 'images'

        images, labels = _read_inat_file(self.root/self.image_file)
        self.classes = sorted(list(set(labels)))
        self.class_to_idx = {c:i for i, c in enumerate(self.classes)}
        self.targets = [self.class_to_idx[c] for c in labels]
        self.imgs = images

        if load_bboxes:
            boxes = open(self.root/self.bounding_box_file).read().strip().split('\n')
            bboxes = []
            for box in boxes:
                bboxes.append([float(x) for x in box.split(' ')])
            self.bboxes = bboxes
