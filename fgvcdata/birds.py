from pathlib import Path

from PIL import Image
import torch, torchvision

from .base import _BaseDataset


__all__ = ['CUB', 'CUBPlus', 'NABirds']


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
    bounding_box_file = 'bounding_boxes.txt'

    def _setup(self):
        self.imfolder = 'images'

        images = _read_file(self.root/self.image_file)
        split = _read_file(self.root/self.train_test_split_file)
        classes = _read_file(self.root/self.class_names_file)
        labels = _read_file(self.root/self.image_class_labels_file)

        imgs, targets = [], []
        cls = set()
        for id in sorted(images.keys()):
            if id in labels and split[id] == self.train:
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

        if self.load_bboxes:
            boxes = _read_file(self.root/self.bounding_box_file)
            bboxes = []
            for id in sorted(images.keys()):
                if id in labels and split[id] == self.train:
                    bboxes.append([float(x) for x in boxes[id].split(' ')])
            self.bboxes = bboxes


class NABirds(_BirdData):
    '''The NABirds dataset, consisting of 555 categories of birds.
    
    https://dl.allaboutbirds.org/nabirds
    '''
    name = 'NABirds'


class CUB(_BirdData):
    '''The classic CUB birds dataset, consisting of 200 categories of birds.
    
    http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
    '''
    name = 'Caltech UCSD Birds (CUB)'
    url_files = {
        'CUB_200_2011.tgz':
            'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz',
        'README.txt':
            'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/README.txt'
    }


class CUBPlus(CUB):
    '''The CUB++ birds dataset -- CUB with expert-validated labels'''
    name = 'Caltech UCSD Birds (CUB++)'
    image_class_labels_file = 'cubplus_image_class_labels.txt'

