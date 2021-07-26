from pathlib import Path

from PIL import Image
import torch, torchvision

from .base import _BaseDataset


__all__ = ['Aircraft']


def _read_file(fname):
    with open(fname) as f:
        anno = [x.split(' ',1) for x in f.read().strip().split('\n')]
    files = [x[0] for x in anno]
    labels = [x[1] for x in anno]
    return files, labels


class Aircraft(_BaseDataset):
    '''The Oxford FGVC Aircraft dataset, consisting of 100 categories of
    aircraft.
    
    http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/
    '''
    name = 'FGVC Aircraft'
    train_file = 'data/images_variant_trainval.txt'
    test_file = 'data/images_variant_test.txt'
    bounding_box_file = 'data/images_box.txt'
    url_files = {
        'fgvc-aircraft-2013b.tar.gz':
        'http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz'
    }
        
    def _setup(self):
        self.imfolder = 'data/images'
        anno_file = self.train_file if self.train else self.test_file

        files, labels = _read_file(self.root/anno_file)
        classes = sorted(list(set(labels)))
        class_to_idx = {c:i for i, c in enumerate(classes)}

        imgs, targets = [], []
        for im, targ in zip(files, labels):
            imgs.append(im+'.jpg')
            targets.append(class_to_idx[targ])
        self.imgs = imgs
        self.targets = targets
        self.classes = classes
        self.class_to_idx = class_to_idx

        if self.load_bboxes:
            names, boxes = _read_file(self.root/self.bounding_box_file)
            boxes = {n+'.jpg': b for n, b in zip(names, boxes)}
            bboxes = []
            for im in self.imgs:
                box = [float(x) for x in boxes[im].split(' ')]
                # x1,y1,x2,y2 -> x1,y1,w,h
                box[2], box[3] = box[2]-box[0], box[3]-box[1] 
                bboxes.append(box)
            self.bboxes = bboxes
