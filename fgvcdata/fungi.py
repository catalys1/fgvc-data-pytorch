from pathlib import Path

from PIL import Image
import torch, torchvision

from .base import _BaseDataset


__all__ = ['DanishFungi']



def parse_csv(fpath,wanted_key_list):
    f = open(fpath,'r')
    lines = [x.strip().split(',') for x in f.readlines()]
    header = lines[0]
    #print(header)
    #print(lines[1])
    inds = []
    for k,t in wanted_key_list:
        for c in range(len(header)):
            if header[c]==k:
                inds.append( (k,c,t) )
    #print(inds)
    records = []
    for parts in lines[1:]:
        p2 = []
        infrag = False
        for part in parts:
            if len(part)==0:
                p2.append(part)
            elif infrag:
                if part[-1]=='"':
                    p2[-1] += ','+part[:-1]
                    infrag = False
                else:
                    p2[-1] += ','+part
            elif part[0]=='"':
                if part[-1]=='"':
                    p2.append(part[1:-1])
                else:
                    p2.append(part[1:])
                    infrag = True
            else:
                p2.append(part)
        parts = p2
        #print( len(parts),parts)
        needed = {k:t(parts[ind]) for k,ind,t in inds}
        records.append( needed )
    return records
    
    
    
def get_images_classes_and_labels( db_data ):
    
    imgs = [x['image_path'] for x in db_data]
    images = dict([ (i,'images/'+img) for i,img in zip(range(1,len(imgs)+1),imgs)])
    lbls = [x['species']+' ('+str(x['taxonID'])+')' for x in db_data]
    classes = sorted(set(lbls))
    classes = dict([ (i,cl) for i,cl in zip(range(1,len(classes)+1),list(classes)) ])
    rev_class = dict([ (classes[i],i) for i in classes.keys() ])
    #print(rev_class)
    labels = dict([ (i,rev_class[lbl]) for i,lbl in zip(range(1,len(lbls)+1),lbls)])
    return images,classes,labels

class DanishFungi(_BaseDataset):
    ''' The Danish Fungi FGVC dataset from Lukas Picek et al.
    https://sites.google.com/view/danish-fungi-dataset
    Contains 32753 training images and 3640 test images across 139 categories.
    '''

    TRAIN_FILE = 'DF20M-train_metadata_PROD.csv'
    TEST_FILE  = 'DF20M-public_test_metadata_PROD.csv'
    KEY_LIST   = [('ImageUniqueID',str),('image_path',str),('taxonID',lambda x:int(float(x))),('species',str)]

    def _setup(self):
        self.imfolder = 'images'

        db_data = parse_csv( self.root/(self.TRAIN_FILE if self.train else self.TEST_FILE), self.KEY_LIST )

        images,classes,labels = get_images_classes_and_labels(db_data)
        split = dict([(i,self.train) for i in images.keys()])

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

