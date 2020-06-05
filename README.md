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

Currently, the best method of installation is to clone this repository and
install with `pip` in editable mode:
```bash
git clone https://github.com/catalys1/fgvc-data-pytorch.git
pip install -e fgvc-data-pytorch
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
