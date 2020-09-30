#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.utils.data as data
from torchvision import transforms
import os
import pickle
import random
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
    if args.embedding == 'w2vec' or args.embedding == 'fasttext' or args.embedding == 'glove' or args.embedding =='bert':
        with open(data_path + '/Context/' + args.ocr + '/text_embeddings/Context_' + args.embedding + '.pickle','rb') as fp:
            text_embedding = pickle.load(fp)
    elif args.embedding =='phoc':
        text_embedding = {'embedding':'phoc'}
    elif args.embedding == 'fisher':
        text_embedding = {'embedding':'fisher'}
    else:
        print('OCR SELECTED NOT IMPLEMENTED')
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

    train_loader = Context_Train(args, gt_annotations, text_embedding, embedding_size, train_transform)

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_loader = Context_Test(args, gt_annotations, text_embedding, embedding_size, test_transform, )

    return train_loader, test_loader, gt_annotations, text_embedding


class Context_Train(data.Dataset):
    def __init__(self, args, gt_annotations, text_embedding, embedding_size, transform=None):

        self.args = args
        self.gt_annotations = gt_annotations
        self.text_embedding = text_embedding
        self.embedding_size = embedding_size
        self.transform = transform
        self.image_list = list(gt_annotations['train'].keys())
        #Random.shuffle(self.image_list)


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
        # Negative Image Generation
        while(1):
            neg_image_name = self.image_list[random.randint(0,len(self.gt_annotations['train'])-1)].rstrip()
            neg_img_class = self.gt_annotations['train'][neg_image_name]
            if neg_img_class != img_class:
                break
        neg_label = np.zeros(28)
        neg_label[int(neg_img_class) - 1] = 1
        neg_label = torch.from_numpy(neg_label)
        neg_label = neg_label.type(torch.FloatTensor)
        neg_image_path = data_path + '/Context/data/JPEGImages/' + neg_image_name
        neg_img = Image.open(neg_image_path).convert('RGB')
        if self.transform:
            neg_img = self.transform(neg_img)
        
        # Positive Image Generation
        while (1):
            pos_image_name = self.image_list[random.randint(0, len(self.gt_annotations['train'])-1)].rstrip()
            pos_img_class = self.gt_annotations['train'][pos_image_name]
            if pos_img_class == img_class:
                break
        pos_label = np.zeros(28)
        pos_label[int(pos_img_class) - 1] = 1
        pos_label = torch.from_numpy(pos_label)
        pos_label = pos_label.type(torch.FloatTensor)
        pos_image_path = data_path + '/Context/data/JPEGImages/' + pos_image_name
        pos_img = Image.open(pos_image_path).convert('RGB')
        if self.transform:
            pos_img = self.transform(pos_img)

        if self.args.embedding == 'w2vec' or self.args.embedding == 'fasttext' or self.args.embedding == 'glove' or self.args.embedding == 'bert':
            text_embedding = np.asarray(self.text_embedding[image_name])
            pos_text_embedding = np.asarray(self.text_embedding[pos_image_name])
            neg_text_embedding = np.asarray(self.text_embedding[neg_image_name])
        elif self.args.embedding == 'phoc':
            with open (data_path + '/Context/yolo_phoc/'+image_name[:-3]+'json') as fp:
                phocs = json.load(fp)
                text_embedding = np.resize(phocs, (np.shape(phocs)[0], 604))
        elif self.args.embedding == 'fisher':
            with open (data_path + '/Context/fisher_vectors/'+image_name[:-3]+'json')as fp:
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
            # NEGATIVE MINING
            neg_text_features = np.zeros((self.args.max_textual, self.embedding_size))
            if np.shape(neg_text_embedding)[0] == 0:
                neg_text_embedding = np.zeros((1, self.embedding_size))
            elif np.shape(neg_text_embedding)[0] > self.args.max_textual:
                neg_text_embedding = neg_text_embedding[0:self.args.max_textual]
            neg_text_features[:len(neg_text_embedding)] = neg_text_embedding
            # POSITIVE MINING
            pos_text_features = np.zeros((self.args.max_textual, self.embedding_size))
            if np.shape(pos_text_embedding)[0] == 0:
                pos_text_embedding = np.zeros((1, self.embedding_size))
            elif np.shape(pos_text_embedding)[0] > self.args.max_textual:
                pos_text_embedding = pos_text_embedding[0:self.args.max_textual]
            pos_text_features[:len(pos_text_embedding)] = pos_text_embedding


        else:
            text_features = text_embedding


        text_features = torch.from_numpy(text_features)
        text_features = text_features.type(torch.FloatTensor)
        neg_text_features = torch.from_numpy(neg_text_features)
        neg_text_features = text_features.type(torch.FloatTensor)
        pos_text_features = torch.from_numpy(pos_text_features)
        pos_text_features = text_features.type(torch.FloatTensor)

        return img, label, text_features, pos_img, pos_label, pos_text_features, neg_img, neg_label, neg_text_features

class Context_Test(data.Dataset):
    def __init__(self, args, gt_annotations, text_embedding, embedding_size, transform=None):
        self.args = args
        self.gt_annotations = gt_annotations
        self.text_embedding = text_embedding
        self.embedding_size = embedding_size
        self.transform = transform
        self.image_list = list(gt_annotations['test'].keys())

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
            with open (data_path + '/Context/fisher_vectors/'+image_name[:-3]+'json')as fp:
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

        return img, label, text_features





