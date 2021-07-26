import json
from pathlib import Path

from PIL import Image
from scipy.io import loadmat
import torch, torchvision

from .base import _BaseDataset


__all__ = ['StanfordDogs', 'TsinghuaDogs']


def _read_anno_file(fname):
    anno = loadmat(fname)
    files = [x.item() for x in anno['file_list'].ravel()]
    targets = [x.item() for x in anno['labels'].ravel()]
    return files, targets


def _load_bbox_anno_files(filelist, tag='bndbox'):
    import xml.etree.ElementTree as etree
    boxes = []
    for f in filelist:
        root = etree.parse(f).getroot()
        elements = root.findall(f'.//{tag}')
        # some images have multiple bounding boxes
        bbs = []
        for el in elements:
            x1, y1, x2, y2 = [float(x.text) for x in el.findall('.//')]
            bbs.append([x1, y1, x2-x1, y2-y1])
        boxes.append(bbs)
    return boxes


def _load_bbox_json(fname):
    return json.load(open(fname))


def _cache_bbox_json(boxes, fname):
    try:
        json.dump(boxes, open(fname, 'w'))
    except PermissionError as e:
        print('Unable to cache bounding boxes (permission denied)')
        print(e)


class StanfordDogs(_BaseDataset):
    '''The Stanford Dogs dataset, consisting of 120 categories of dog sourced
    from ImageNet.
    
    http://vision.stanford.edu/aditya86/ImageNetDogs/
    '''
    name = 'Stanford Dogs'
    train_anno_file = 'train_list.mat'
    test_anno_file = 'test_list.mat'
    train_bounding_box_file = 'train_bbox.json'
    test_bounding_box_file = 'test_bbox.json'
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
            bbox_file = self.root.joinpath(self.train_bounding_box_file if self.train
                                           else self.test_bounding_box_file)
            if bbox_file.is_file():
                self.bboxes = _load_bbox_json(bbox_file)
            else:
                anno = loadmat(self.root/anno_file)['annotation_list']
                paths = [self.root.joinpath('Annotation', a[0].item()) for a in anno]
                bboxes = _load_bbox_anno_files(paths)
                _cache_bbox_json(bboxes, bbox_file)
                self.bboxes = bboxes


class TsinghuaDogs(_BaseDataset):
    '''The Tsinghua Dogs dataset, consisting of 130 categories of dog.

    https://cg.cs.tsinghua.edu.cn/ThuDogs/
    '''
    name = 'Tsinghua Dogs'
    train_anno_file = 'TrainAndValList/train.lst'
    val_anno_file = 'TrainAndValList/validation.lst'
    train_bounding_box_file = 'train_bbox.json'
    test_bounding_box_file = 'test_bbox.json'

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

        if self.load_bboxes:
            bbox_file = self.root.joinpath(self.train_bounding_box_file if self.train
                                           else self.test_bounding_box_file)
            if bbox_file.is_file():
                self.bboxes = _load_bbox_json(bbox_file)
            else:
                paths = [self.root.joinpath('Low-Annotations', x+'.xml') for x in self.imgs]
                bboxes = _load_bbox_anno_files(paths, 'bodybndbox')
                _cache_bbox_json(bboxes, bbox_file)
                self.bboxes = bboxes
