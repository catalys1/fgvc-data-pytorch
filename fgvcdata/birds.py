import torch, torchvision
from pathlib import Path
from PIL import Image
from .base import _BaseDataset


__all__ = ['CUB', 'CUBPlus', 'NABirds', 'InatCUBVal']


def _parse_ints(vals):
    for i in range(len(vals)):
        try:
            vals[i] = int(vals[i])
        except:
            pass
    return vals


def _read_file(fname):
    with open(fname) as f:
        txt = f.read().strip()
        res = dict(_parse_ints(x.split(' ', 1)) for x in txt.split('\n'))
    return res


class _BirdData(_BaseDataset):
    image_file = 'images.txt'
    train_test_split_file = 'train_test_split.txt'
    class_names_file = 'classes.txt'
    image_class_labels_file = 'image_class_labels.txt'

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

        self.imfolder = 'images'

        images = _read_file(self.root/self.image_file)
        split = _read_file(self.root/self.train_test_split_file)
        classes = _read_file(self.root/self.class_names_file)
        labels = _read_file(self.root/self.image_class_labels_file)

        imgs, targets = [], []
        cls = set()
        for id in sorted(images.keys()):
            if id in labels and split[id] == is_train:
                imgs.append(images[id])
                targets.append(labels[id])
                cls.add(labels[id])
        self.imgs = imgs
        self.targets = targets

        self.class_to_idx = {}
        idx_shift = {}
        i = 0
        for k, v in sorted(classes.items()):
            if k in cls:
                self.class_to_idx[v] = i
                idx_shift[k] = i
                i += 1
        self.classes = [x[0] for x in sorted(self.class_to_idx.items(),
                                             key=lambda a:a[1])]
        self.targets = [idx_shift[i] for i in self.targets]


class NABirds(_BirdData):
    '''The NA Birds dataset, consisting of 555 categories of birds'''


class CUB(_BirdData):
    '''The classic CUB Birds dataset, consisting of 200 categories of birds'''


class CUBPlus(_BirdData):
    '''The CUB Birds dataset with expert-validated labels'''
    image_class_labels_file = 'EXPERT_IMAGE_CLASS_LABELS.txt'


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

