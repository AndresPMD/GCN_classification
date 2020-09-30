#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.utils.data as data
from torchvision import transforms
import os
import pickle
import random
import pdb
import sys
import json
from PIL import Image

sys.path.insert(0, '.')

import numpy as np
from skimage import io


def Context_dataset(args, embedding_size):
    # Random seed
    np.random.seed(args.seed)

    # Getting the classes and annotations
    # ******
    data_path = args.data_path
    with open(data_path+'/Context/data/split_'+ str(args.split) +'.json','r') as fp:
        gt_annotations = json.load(fp)

    # Load Embedding according to OCR
    if args.embedding == 'w2vec' or args.embedding == 'fasttext' or args.embedding == 'glove':
        if args.ocr == 'google_ocr':
            with open(data_path + '/Context/' + args.ocr + '/text_embeddings/Context_' + args.embedding + '.json', 'r') as fp:
                text_embedding = json.load(fp)
        else:
            with open(data_path + '/Context/' + args.ocr + '/text_embeddings/Context_' + args.embedding + '.pickle','rb') as fp:
                text_embedding = pickle.load(fp)
    elif args.embedding =='phoc':
        text_embedding = {'embedding':'phoc'}
    elif args.embedding == 'fisher':
        text_embedding = {'embedding':'fisher'}
    else:
        print('OCR SELECTED NOT IMPLEMENTED')

    # Load Local features from Faster R-CNN VG

    with open(args.data_path + '/Context/context_local_feats.npy', 'rb') as fp:
        local_feats = np.load(fp, encoding='bytes')

    # Create img_name to index of local features
    with open(args.data_path + '/Context/context_local_feats_image_ids.txt', 'r') as fp:
        image_ids = fp.readlines()
    image_name2features_index = {}
    for item in image_ids:
        img_name = item.strip().split(',')[0].split('/')[-1].replace('\'', '')
        idx = item.strip().split(',')[1].replace(')', '').replace(' ','')
        image_name2features_index[img_name] = idx

    # BBOXES LOADING FOR TEXT FEATURES
    # Load BBOXES of Scene Text
    with open(data_path + '/Context/google_ocr/bboxes/Context_bboxes.json', 'r') as fp:
        text_bboxes = json.load(fp)

    # Load BBOXES of Local Visual Features
    with open(data_path + '/Context/context_bboxes.npy', 'rb') as fp:
        local_bboxes = np.load(fp, encoding='bytes')


    # Data Loaders

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_loader = Context_Train(args, gt_annotations, text_embedding, embedding_size, local_feats, image_name2features_index, text_bboxes, local_bboxes, train_transform)

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_loader = Context_Test(args, gt_annotations, text_embedding, embedding_size, local_feats, image_name2features_index, text_bboxes, local_bboxes, test_transform)

    return train_loader, test_loader, gt_annotations, text_embedding


class Context_Train(data.Dataset):
    def __init__(self, args, gt_annotations, text_embedding, embedding_size, local_feats, image_name2features_index, text_bboxes, local_bboxes, transform=None):

        self.args = args
        self.gt_annotations = gt_annotations
        self.text_embedding = text_embedding
        self.embedding_size = embedding_size
        self.transform = transform
        self.image_list = list(gt_annotations['train'].keys())

        self.image_name2features_index = image_name2features_index
        self.local_feats = local_feats
        self.text_bboxes = text_bboxes
        self.local_bboxes = local_bboxes

    def __len__(self):
        return len(self.gt_annotations['train'])

    def __getitem__(self, index):
        data_path = self.args.data_path
        assert index <= len(self), 'index range error'

        image_name = self.image_list[index].rstrip()
        image_path = data_path+'/Context/data/JPEGImages/' + image_name
        img = Image.open(image_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        img_class = self.gt_annotations['train'][image_name]
        label = np.zeros(28)
        label[int(img_class) - 1] = 1
        label = torch.from_numpy(label)
        label = label.type(torch.FloatTensor)

        if self.args.embedding == 'w2vec' or self.args.embedding == 'fasttext' or self.args.embedding == 'glove' or self.args.embedding == 'bert':
            text_embedding = np.asarray(self.text_embedding[image_name])
        elif self.args.embedding == 'phoc':
            with open (data_path + '/Context/yolo_phoc/'+image_name[:-3]+'json') as fp:
                phocs = json.load(fp)
                text_embedding = np.resize(phocs, (np.shape(phocs)[0], 604))
        elif self.args.embedding == 'fisher':
            if self.args.ocr == 'yolo_phoc':
                relative_path = '/Context/old_fisher_vectors/'
            elif self.args.ocr == 'e2e_mlt':
                relative_path = '/Context/fasttext_fisher/'
            else: print('Not Implemented')
            with open (data_path + relative_path +image_name[:-3]+'json')as fp:
                fisher_vector = json.load(fp)
                text_embedding = np.resize(fisher_vector, (1, 38400))
        # FISHER VECTORS DO NOT NEED MAX TEXTUAL
        if self.args.embedding != 'fisher':
            text_features = np.zeros((self.args.max_textual, self.embedding_size))
            if np.shape(text_embedding)[0] == 0:
                text_embedding = np.zeros((1,self.embedding_size))
            elif np.shape(text_embedding)[0] > self.args.max_textual:
                text_embedding = text_embedding[0:self.args.max_textual]
            text_features[:len(text_embedding)] = text_embedding
        else:
            text_features = text_embedding

        text_features = torch.from_numpy(text_features)
        text_features = text_features.type(torch.FloatTensor)

        # SCENE TEXT BBOXES ONLY FOR GOOGLE OCR
        text_bboxes = np.asarray(self.text_bboxes[image_name])
        if self.args.ocr == 'google_ocr':
            text_bboxes_features = np.zeros((self.args.max_textual, 4))
            if np.shape(text_bboxes)[0] == 0:
                text_bboxes = np.zeros((1, 4))
            elif np.shape(text_bboxes)[0] > self.args.max_textual:
                text_bboxes = text_bboxes[0:self.args.max_textual]
            text_bboxes_features[:len(text_bboxes)] = text_bboxes
        else:
            # NO BBOXES FOR OTHER OCRs
            text_bboxes_features = np.zeros((self.args.max_textual, 4))
        text_bboxes_features = torch.from_numpy(text_bboxes_features)
        text_bboxes_features = text_bboxes_features.type(torch.FloatTensor)

        # LOCAL VISUAL FEATURES
        local_features_index = self.image_name2features_index[image_name]
        local_features = self.local_feats[int(local_features_index)]
        local_features = torch.from_numpy(local_features[:int(self.args.max_visual)][:])
        local_features = local_features.type(torch.FloatTensor)
        # LOCAL VISUAL BBOXES
        local_bboxes_features = self.local_bboxes[int(local_features_index)]
        local_bboxes_features = torch.from_numpy(local_bboxes_features[:int(self.args.max_visual)][:])
        local_bboxes_features = local_bboxes_features.type(torch.FloatTensor)

        return img, label, text_features, local_features, text_bboxes_features, local_bboxes_features, image_name


class Context_Test(data.Dataset):
    def __init__(self, args, gt_annotations, text_embedding, embedding_size, local_feats, image_name2features_index, text_bboxes, local_bboxes, transform=None):
        self.args = args
        self.gt_annotations = gt_annotations
        self.text_embedding = text_embedding
        self.embedding_size = embedding_size
        self.transform = transform
        self.image_list = list(gt_annotations['test'].keys())

        self.image_name2features_index = image_name2features_index
        self.local_feats = local_feats
        self.text_bboxes = text_bboxes
        self.local_bboxes = local_bboxes

    def __len__(self):
        return len(self.gt_annotations['test'])

    def __getitem__(self, index):
        data_path = self.args.data_path
        assert index <= len(self), 'index range error'
        image_name = self.image_list[index].rstrip()
        image_path = data_path+ '/Context/data/JPEGImages/' + image_name
        img = Image.open(image_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        img_class = self.gt_annotations['test'][image_name]
        label = np.zeros(28)
        label[int(img_class) - 1] = 1
        label = torch.from_numpy(label)
        label = label.type(torch.FloatTensor)

        if self.args.embedding == 'w2vec' or self.args.embedding == 'fasttext' or self.args.embedding == 'glove' or self.args.embedding == 'bert':
            text_embedding = np.asarray(self.text_embedding[image_name])
        elif self.args.embedding == 'phoc':
            with open (data_path + '/Context/yolo_phoc/'+image_name[:-3]+'json') as fp:
                phocs = json.load(fp)
                text_embedding = np.resize(phocs, (np.shape(phocs)[0], 604))

        elif self.args.embedding == 'fisher':
            if self.args.ocr == 'yolo_phoc':
                relative_path = '/Context/old_fisher_vectors/'
            elif self.args.ocr == 'e2e_mlt':
                relative_path = '/Context/fasttext_fisher/'
            else: print('Not Implemented')
            with open (data_path + relative_path +image_name[:-3]+'json')as fp:
                fisher_vector = json.load(fp)
                text_embedding = np.resize(fisher_vector, (1, 38400))
        # FISHER VECTORS DO NOT NEED MAX TEXTUAL
        if self.args.embedding != 'fisher':
            text_features = np.zeros((self.args.max_textual, self.embedding_size))
            if np.shape(text_embedding)[0] == 0:
                text_embedding = np.zeros((1,self.embedding_size))
            elif np.shape(text_embedding)[0] > self.args.max_textual:
                text_embedding = text_embedding[0:self.args.max_textual]
            text_features[:len(text_embedding)] = text_embedding
        else:
            text_features = text_embedding

        text_features = torch.from_numpy(text_features)
        text_features = text_features.type(torch.FloatTensor)
        # SCENE TEXT BBOXES ONLY FOR GOOGLE OCR
        text_bboxes = np.asarray(self.text_bboxes[image_name])
        if self.args.ocr == 'google_ocr':
            text_bboxes_features = np.zeros((self.args.max_textual, 4))
            if np.shape(text_bboxes)[0] == 0:
                text_bboxes = np.zeros((1, 4))
            elif np.shape(text_bboxes)[0] > self.args.max_textual:
                text_bboxes = text_bboxes[0:self.args.max_textual]
            text_bboxes_features[:len(text_bboxes)] = text_bboxes
        else:
            # NO BBOXES FOR OTHER OCRs
            text_bboxes_features = np.zeros((self.args.max_textual, 4))
        text_bboxes_features = torch.from_numpy(text_bboxes_features)
        text_bboxes_features = text_bboxes_features.type(torch.FloatTensor)

        # LOCAL VISUAL FEATURES
        local_features_index = self.image_name2features_index[image_name]
        local_features = self.local_feats[int(local_features_index)]
        local_features = torch.from_numpy(local_features[:int(self.args.max_visual)][:])
        local_features = local_features.type(torch.FloatTensor)
        # LOCAL VISUAL BBOXES
        local_bboxes_features = self.local_bboxes[int(local_features_index)]
        local_bboxes_features = torch.from_numpy(local_bboxes_features[:int(self.args.max_visual)][:])
        local_bboxes_features = local_bboxes_features.type(torch.FloatTensor)

        return img, label, text_features, local_features, text_bboxes_features, local_bboxes_features, image_name





