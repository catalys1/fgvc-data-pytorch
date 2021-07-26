# FGVC Datasets in PyTorch

The fgvcdata package implements a common API for working with Fine-Grained Visual
Categorization (FGVC) datasets in PyTorch.

## Purpose
FGVC is becoming a popular area in
computer vision, with new methods being published regularly at top CV
conferences. Often, the method authors provide PyTorch code to aid in
reproducing their work. Each code base ends up
implementing their own version of a PyTorch API to utilize several common
datasets. This leads to unnecessary duplicated effort, and often
complicates things when trying to work with multiple different models
and datasets.

The purposes of this package are
- to provide a unified interface for using FGVC datasets
- to make it easy to to get up and running with these datasets
- to make it convenient to add new datasets to the training process


## Installation

Install with `pip`
```bash
pip install fgvcdata
```

The fgvcdata package should now be available:
```python
>>> import fgvcdata
>>> fgvcdata.datasets
['CUB', 'CUBPlus', 'NABirds', 'StanfordCars', 'StanfordDogs', 'Aircraft', 'OxfordFlowers']
```

## Datasets

These are the currently available datasets:
- [FGVC Aircraft](http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)
- [Caltech UCSD Birds (CUB-200)](http://www.vision.caltech.edu/visipedia/CUB-200.html)
- [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
- [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/)
- [Oxford Flowers 102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)
- [North American Birds (NABirds)](https://dl.allaboutbirds.org/nabirds)

These datasets are publicly available online for use in research; we do not host or distribute them, or make any claims about their quality or fairness.

### Accessing the Data

The images and metadata for most of the datasets can be downloaded directly through the library, similar to `torchvision.datasets`. Further processing shouldn't be necessary.

The dataset classes in this package expect the data to be organized inside a root folder, containing the various image folders and metadata files in the same form as they are presented for download by the dataset host.

## Contributing

Contributions are welcome! Please feel free to open issues related to bug fixes, or to fix them and submit a pull request. I'd also be happy to discuss updates/changes to the library to make it more useful/usable, as well as for adding new datasets -- feel free to open an issue.
