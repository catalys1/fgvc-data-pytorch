import torch, torchvision
from pathlib import Path
from PIL import Image
from scipy.io import loadmat
from .base import _BaseDataset


__all__ = ['OxfordFlowers']


def _read_labels(fname):
    labels = loadmat(fname)['labels'].ravel()
    return [x.item() for x in labels]


def _read_split(fname):
    splits = loadmat(fname)
    split = {}
    for k in 'trnid valid tstid'.split():
        s = 0 if k == 'tstid' else 1
        for id in splits[k].ravel():
            split[id.item()] = s
    return split


class OxfordFlowers(_BaseDataset):
    '''The Oxford Flowers datasets, consisting of 102 categories of flower'''
    label_file = 'imagelabels.mat'
    split_file = 'setid.mat'
        
    def __init__(self, root, transform=None, target_transform=None, train=True):
        self.root = Path(root)
        if self.root.name == 'train':
            is_train = True
            self.root = self.root.parent
        elif self.root.name == 'test':
            is_train = False
            self.root = self.root.parent
        else:
            is_train = train
        self.transform = transform
        self.target_transform = target_transform
        self.train = is_train

        self.imfolder = 'jpg'

        files = sorted([x.name for x in
                        self.root.joinpath(self.imfolder).iterdir()])
        labels = _read_labels(self.root/self.label_file)
        split = _read_split(self.root/self.split_file)
        classes = sorted(list(set(labels)))
        class_to_idx = {c:i for i, c in enumerate(classes)}

        imgs, targets = [], []
        for im, targ in zip(files, labels):
            if split[int(im[6:11])] == is_train:
                imgs.append(im)
                targets.append(class_to_idx[targ])
        self.imgs = imgs
        self.targets = targets
        self.classes = classes
        self.class_to_idx = class_to_idx

