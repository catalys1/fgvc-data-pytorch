import torch, torchvision
from pathlib import Path
from PIL import Image
from scipy.io import loadmat
from .base import _BaseDataset


__all__ = ['StanfordCars']


def _read_anno_file(fname):
    anno = loadmat(fname)['annotations'][0]
    files = [x[5].item() for x in anno]
    targets = [x[4].item() for x in anno]
    return files, targets


def _read_class_file(fname):
    class_names = loadmat(fname)['class_names'][0]
    names = [x[0].item() for x in class_names]
    class_to_idx = {v: i for i, v in enumerate(names)}
    return names, class_to_idx


class StanfordCars(_BaseDataset):
    '''The Stanford Cars datasets, consisting of 196 categories of cars'''
    train_anno_file = 'devkit/cars_train_annos.mat'
    test_anno_file = 'devkit/cars_test_annos_withlabels.mat'
    class_file = 'devkit/cars_meta.mat'
        
    def __init__(self, root, transform=None, target_transform=None, train=True):
        self.root = Path(root)
        if self.root.name in ['cars_train', 'train']:
            is_train = True
            self.root = self.root.parent
        elif self.root.name in ['cars_test', 'test']:
            is_train = False
            self.root = self.root.parent
        else:
            is_train = train
        self.transform = transform
        self.target_transform = target_transform
        self.train = is_train

        self.imfolder = 'cars_' + ('train' if is_train else 'test')
        anno_file = self.train_anno_file if is_train else self.test_anno_file

        files, labels = _read_anno_file(self.root/anno_file)
        classes, class_to_idx = _read_class_file(self.root/self.class_file)

        imgs, targets = [], []
        for im, targ in zip(files, labels):
            imgs.append(im)
            targets.append(targ - 1)
        self.imgs = imgs
        self.targets = targets
        self.classes = classes
        self.class_to_idx = class_to_idx

