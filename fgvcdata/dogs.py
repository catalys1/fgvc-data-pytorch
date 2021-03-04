import torch, torchvision
from pathlib import Path
from PIL import Image
from scipy.io import loadmat
import re
from .base import _BaseDataset


__all__ = ['StanfordDogs', 'TsinghuaDogs']


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

        if self.load_bboxes:
            anno = loadmat(self.root/anno_file)['annotation_list']
            regex = r'<xmin>(\d+)</xmin>.*<ymin>(\d+)</ymin>.*<xmax>(\d*)</xmax>.*<ymax>(\d+)</ymax>'
            regex = re.compile(regex, flags=re.DOTALL)
            boxes = []
            for a in anno:
                path = self.root.joinpath('Annotation', a[0].item())
                content = open(path).read()
                x1, y1, x2, y2 = [float(x) for x in re.search(regex, content).groups()]
                boxes.append([x1, y1, x2-x1, y2-y1])
            self.bboxes = boxes


class TsinghuaDogs(_BaseDataset):
    '''The Tsinghua Dogs dataset, consisting of 130 categories of dog.

    https://cg.cs.tsinghua.edu.cn/ThuDogs/
    '''
    name = 'Tsinghua Dogs'
    train_anno_file = 'TrainAndValList/train.lst'
    val_anno_file = 'TrainAndValList/validation.lst'

    def _setup(self):
        self.imfolder = 'low-resolution'
        anno_file = self.train_anno_file if self.train else self.val_anno_file

        # odd byte at beginning of file, thats why [1:]
        files = open(self.root/anno_file).read().strip().split('\n')[1:]
        files = [x[3:] for x in files] # remove ".//" prefix
        files.sort()

        imgs, targets = [], []
        class_to_idx = {}
        i = 0
        for f in files:
            targ = f.split('/')[0]
            if targ not in class_to_idx:
                class_to_idx[targ] = i
                i += 1
            imgs.append(f)
            targets.append(class_to_idx[targ])
        self.imgs = imgs
        self.targets = targets
        self.classes = list(class_to_idx.keys())
        self.class_to_idx = class_to_idx
