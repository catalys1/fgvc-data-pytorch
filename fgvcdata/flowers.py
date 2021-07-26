from pathlib import Path

from PIL import Image
from scipy.io import loadmat
import torch, torchvision

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
    '''The Oxford Flowers dataset, consisting of 102 categories of flower.
    
    https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html
    '''
    name = 'Oxford Flowers 102'
    label_file = 'imagelabels.mat'
    split_file = 'setid.mat'
    url_files = {
        '102flowers.tgz':
        'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz',
        'imagelabels.mat':
        'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat',
        'setid.mat':
        'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat',
        'README.txt':
        'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/README.txt',
    }

    def _setup(self):
        if self.load_bboxes:
            raise AttributeError('Oxford Flowers does not have any available bounding boxes')
        self.imfolder = 'jpg'
        files = sorted([x.name for x in
                        self.root.joinpath(self.imfolder).iterdir()])
        labels = _read_labels(self.root/self.label_file)
        split = _read_split(self.root/self.split_file)
        classes = sorted(list(set(labels)))
        class_to_idx = {c:i for i, c in enumerate(classes)}

        imgs, targets = [], []
        for im, targ in zip(files, labels):
            if split[int(im[6:11])] == self.train:
                imgs.append(im)
                targets.append(class_to_idx[targ])
        self.imgs = imgs
        self.targets = targets
        self.classes = classes
        self.class_to_idx = class_to_idx

