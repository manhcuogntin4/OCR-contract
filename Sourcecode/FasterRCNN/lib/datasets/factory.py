# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Copyright (c) 2016 Haoming Wang
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

from datasets.pascal_voc import pascal_voc
from datasets.coco import coco
from datasets.inria import inria
from datasets.axa import axa
import numpy as np

# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

# Set up coco_2014_<split>
for year in ['2014']:
    for split in ['train', 'val', 'minival', 'valminusminival']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
    for split in ['test', 'test-dev']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

inria_devkit_path = '/home/ubuntu/code/py-faster-rcnn/data/INRIA_Person_devkit' # for AWS EC2
for split in ['train', 'test']:
    name = '{}_{}'.format('inria', split)
    __sets[name] = (lambda split=split: inria(split, inria_devkit_path))

axa_devkit_path = '/home/ubuntu/code/py-faster-rcnn/data/axa_rentree_dataset' # for AWS EC2
for split in ['train', 'test']:
    name = '{}_{}'.format('axa', split)
    __sets[name] = (lambda split=split: axa(split, axa_devkit_path))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
