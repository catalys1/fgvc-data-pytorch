from PIL import Image


class _BaseDataset(object):
    def __init__(self, root, transform=None, target_transform=None, train=True):
        raise NotImplementedError

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path = self.root/self.imfolder/self.imgs[index]
        target = self.targets[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
