# FGVC Datasets in PyTorch

This package implements a common API for working with Fine-Grained Visual
Categorization (FGVC) datasets in PyTorch.r

FGVC is becoming a popular area in
computer vision, with new methods being published regularly at top CV
conferences. Often, the method authors provide PyTorch code to help aid in
reproducing and extending their work. Most of these methods are evaluated on
some subset of a group of popular FGVC datasets, and each code base ends up
implementing their own version of a PyTorch API to utilize the data for
training convolutional networks. This leads to unnecessary duplicated effort, 
as well as issues when trying to apply models to datasets not used by the
original authors.

The purpose of this package is to provide a unified interface for using FGVC
datasets, so that it is easy to get up and running with these datasets, as
well as try out other datasets on new models and techniques.
