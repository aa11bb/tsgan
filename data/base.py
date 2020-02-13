from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

Dataset = collections.namedtuple('Dataset', ['X', 'y'])
Datasets = collections.namedtuple('Datasets', ['train', 'valid', 'test', 'nclass'])
