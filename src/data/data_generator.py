#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0,'.')

# Datasets
from data.context import Context_dataset
from data.bottles import Bottle_dataset


def load_data(args, embedding_size):
    if args.dataset == 'context':
        return Context_dataset(args, embedding_size)

    elif args.dataset == 'bottles':
        return Bottle_dataset(args, embedding_size)
    else:
        raise NameError(args.dataset + ' not implemented!')




