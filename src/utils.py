#-*- coding: utf-8 -*-
from __future__ import print_function, division

"""
Pytorch useful tools.
"""

import torch
import os
import errno
import numpy as np


def save_checkpoint(net, best_perf, directory, file_name, data_weights):
    print('---------- SAVING MODEL ----------')
    if not os.path.isdir(directory):
        os.makedirs(directory)
    checkpoint_file = os.path.join(directory, file_name + '_' + data_weights + '.weights')
    torch.save(net.state_dict(), checkpoint_file)
    print('Model Saved as:', checkpoint_file)
    with open (directory+'results.txt','w') as fp:
        fp.write(str(round(100*best_perf, 3))+'\n')

def load_checkpoint(model_file):
    if os.path.isfile(model_file):
        print("=> loading model '{}'".format(model_file))
        checkpoint = torch.load(model_file)
        return checkpoint
    else:
        print("=> no model found at '{}'".format(model_file))
        raise OSError(errno.ENOENT, os.strerror(errno.ENOENT), model_file)

def get_num_classes(args):
    data_path = args.data_path
    if args.dataset == 'context':
        classes_file = data_path+'/Context/classes.txt'
    else:
        classes_file = data_path+'/Drink_Bottle/classes.txt'

    num_classes = sum(1 for line in open(classes_file))
    return num_classes


def get_weight_criterion(dataset):
    if dataset == 'context':
        weights = [1, 1, 2.2, 1, 1, 1, 1, 1, 4.5, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2.2, 1.5, 1, 1, 1, 1]
    elif dataset =='bottles':
        weights = [1, 1, 1, 1, 2, 1.5, 1.5, 1, 1.5, 1, 1, 1.5, 1, 1, 1, 1.5, 1, 1, 1, 1]
    return weights

def get_embedding_size(embedding):
    if embedding == 'w2vec' or embedding == 'fasttext' or embedding == 'glove':
        embedding_size = 300
    elif embedding =='bert':
        embedding_size = 666
    elif embedding == 'phoc':
        embedding_size = 604
    elif embedding == 'fisher':
        embedding_size = 38400

    return embedding_size