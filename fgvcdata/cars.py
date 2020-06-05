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
    '''The Stanford Cars dataset, consisting of 196 categories of cars.
    
    https://ai.stanford.edu/~jkrause/cars/car_dataset.html
    '''
    name = 'Stanford Cars'
    train_anno_file = 'devkit/cars_train_annos.mat'
    test_anno_file = 'devkit/cars_test_annos_withlabels.mat'
    class_file = 'devkit/cars_meta.mat'
    url_files = {
        'cars_train.tgz':
        'http://imagenet.stanford.edu/internal/car196/cars_train.tgz',
        'cars_test.tgz':
        'http://imagenet.stanford.edu/internal/car196/cars_test.tgz',
        'car_devkit.tgz':
        'https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz',
    }
        
    def _setup(self):
        self.imfolder = 'cars_' + ('train' if self.train else 'test')
        anno_file = self.train_anno_file if self.train else self.test_anno_file

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

