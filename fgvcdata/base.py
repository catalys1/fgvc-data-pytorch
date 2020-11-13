from PIL import Image
from pathlib import Path
import os
from torchvision.datasets.utils import download_url, extract_archive


def download_and_extract(url, download_root, extract_root=None, filename=None,
                                 md5=None, remove_finished=False):
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)

    download_url(url, download_root, filename, md5)

    archive = os.path.join(download_root, filename)
    print("Extracting {} to {}".format(archive, extract_root))
    try:
        extract_archive(archive, extract_root, remove_finished)
    except ValueError:
        # OK if the file isn't an archive
        pass


class _BaseDataset(object):
    '''Base class for FGVC datasets. Should not be used directly.'''
    def __init__(self, root, transform=None, target_transform=None, train=True,
                 download=False):
        self.root = Path(root)
        if self.root.name == 'train':
            is_train = True
            self.root = self.root.parent
        elif self.root.name in ['test', 'val']:
            is_train = False
            self.root = self.root.parent
        else:
            is_train = train
        self.transform = transform
        self.target_transform = target_transform
        self.train = is_train

        if download: self.download()
        self._setup()

    def _setup(self):
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

    def __repr__(self):
        head = '{} Dataset ({}.{})'.format(
            self.name, self.__class__.__module__, self.__class__.__name__)
        body = ['Images: {}'.format(len(self)),
                'Root: {}'.format(str(self.root)),
                'Transform: {}'.format(self.transform)]
        lines = [head]+[' '*2 + line for line in body]
        return '\n'.join(lines)

    def download(self):
        if self.root.is_dir():
            print('{} already exists - skipping download'.format(str(self.root)))
            return
        for fname, url in self.url_files.items():
            download_and_extract(url, self.root, filename=fname)
