import torch, torchvision
from pathlib import Path
from PIL import Image
from scipy.io import loadmat
from .base import _BaseDataset


__all__ = ['StanfordDogs']


def _read_anno_file(fname):
    anno = loadmat(fname)
    files = [x.item() for x in anno['file_list'].ravel()]
    targets = [x.item() for x in anno['labels'].ravel()]
    return files, targets


class StanfordDogs(_BaseDataset):
    '''The Stanford Dogs dataset, consisting of 120 categories of dog sourced
    from ImageNet.
    
    http://vision.stanford.edu/aditya86/ImageNetDogs/
    '''
    name = 'Stanford Dogs'
    train_anno_file = 'train_list.mat'
    test_anno_file = 'test_list.mat'
    url_files = {
        'images.tar':
        'http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar',
        'annotations.tar':
        'http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar',
        'lists.tar':
        'http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar',
        'README.txt':
        'http://vision.stanford.edu/aditya86/ImageNetDogs/README.txt',
    }
        
    def _setup(self):
        self.imfolder = 'Images'
        anno_file = self.train_anno_file if self.train else self.test_anno_file

        files, labels = _read_anno_file(self.root/anno_file)

        imgs, targets = [], []
        class_to_idx = {}
        for im, targ in zip(files, labels):
            imgs.append(im)
            targets.append(targ - 1)
            class_to_idx[im.split('/')[0]] = targ - 1
        self.imgs = imgs
        self.targets = targets
        self.classes = list(class_to_idx.keys())
        self.class_to_idx = class_to_idx

