# -*- coding: utf-8 -*-

"""
    Fine-grained Classification based on textual cues
"""

# Python modules
import torch

import torch.nn as nn
import time
import torch
import numpy as np
import glob
import os
import json
from PIL import Image, ImageDraw, ImageFile

import torchvision
from torch.autograd import Variable
from torchvision import transforms

import pdb
import sys

# Own modules

from utils import *
from options import *
from data.data_generator import *
from models.models import load_model
from custom_optim import *

# __author__ = "Andres Mafla Delgado;
# __email__ = "amafla@cvc.uab.cat;

# READS A LIST OF IMAGES BELONGING TO CON-TEXT OR BOTTLES AND EXTRACTS THE FEATURES AND PROBS - blur experiments

def test(args, net, cuda, num_classes, gt_annotations, text_embedding, local_feats, image_name2features_index, text_bboxes, local_bboxes):

    processed_imgs = 0
    # Switch to evaluation mode
    net.eval()

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if not os.path.exists(args.input_dir + 'single_features_' + args.dataset + '/'):
        os.mkdir(args.input_dir + 'single_features_' + args.dataset + '/')

    with torch.no_grad():
        image_list = os.listdir(args.input_dir + args.dataset + '/')

        for image in image_list:
            sample_size = 1
            img_path = args.input_dir + args.dataset + '/' + image
            img = Image.open(img_path).convert('RGB')
            data = test_transform(img)
            data = data.view(-1, 3, 224, 224)

            # Labels

            if args.dataset == 'context':
                label = np.zeros(28)
            elif args.dataset == 'bottles':
                label = np.zeros(20)
                image = 'images/15/' + image

            img_class = gt_annotations['test'][image]
            label[int(img_class) - 1] = 1
            labels = torch.from_numpy(label)
            labels = labels.type(torch.FloatTensor)

            # Textual Features
            if args.blur == 'none' or args.blur == 'visual':
                if args.dataset == 'bottles':
                    image = image.replace('images/','')
                text_embedding_sample = np.asarray(text_embedding[image])
                text_features = np.zeros((args.max_textual, 300))
                if np.shape(text_embedding_sample)[0] == 0:
                    text_embedding_sample = np.zeros((1, 300))
                elif np.shape(text_embedding_sample)[0] > args.max_textual:
                    text_embedding_sample = text_embedding_sample[0:args.max_textual]
                text_features[:len(text_embedding_sample)] = text_embedding_sample

            elif args.blur == 'text':
                text_features = np.zeros((15,300))

            text_features = torch.from_numpy(text_features)
            text_features = text_features.type(torch.FloatTensor)
            text_features = text_features.view(sample_size, 15, 300)

            # SCENE TEXT BBOXES ONLY FOR GOOGLE OCR
            if args.blur == 'none' or args.blur =='visual':
                text_bboxes_sample = np.asarray(text_bboxes[image])
                text_bboxes_features = np.zeros((args.max_textual, 4))
                if np.shape(text_bboxes_sample)[0] == 0:
                    text_bboxes_sample = np.zeros((1, 4))
                elif np.shape(text_bboxes_sample)[0] > args.max_textual:
                    text_bboxes_sample = text_bboxes_sample[0:args.max_textual]
                text_bboxes_features[:len(text_bboxes_sample)] = text_bboxes_sample
            elif args.blur == 'text':
                text_bboxes_features = np.zeros((15,4))

            text_bboxes_features = torch.from_numpy(text_bboxes_features)
            text_bboxes_features = text_bboxes_features.type(torch.FloatTensor)
            text_bboxes_features = text_bboxes_features.view(sample_size, 15, 4)

            # LOCAL VISUAL FEATURES

            if args.blur == 'none' or args.blur == 'text':
                local_features_index = image_name2features_index[image]
                local_feats_sample = local_feats[int(local_features_index)]
                local_feats_sample = torch.from_numpy(local_feats_sample[:int(args.max_visual)][:])
            elif args.blur == 'visual':
                local_feats_sample = np.zeros((36, 2048))
                local_feats_sample = torch.from_numpy(local_feats_sample)
            local_feats_sample = local_feats_sample.type(torch.FloatTensor)
            local_feats_sample = local_feats_sample.view(sample_size, 36, 2048)
            # LOCAL VISUAL BBOXES
            if args.blur == 'none' or args.blur == 'text':
                local_bboxes_features = local_bboxes[int(local_features_index)]
                local_bboxes_features = torch.from_numpy(local_bboxes_features[:int(args.max_visual)][:])
            elif args.blur == 'visual':
                local_bboxes_features = np.zeros ((36,4))
                local_bboxes_features = torch.from_numpy(local_bboxes_features)
            local_bboxes_features = local_bboxes_features.type(torch.FloatTensor)
            local_bboxes_features = local_bboxes_features.view(sample_size, 36, 4)

            # pdb.set_trace()
            # Move to GPU

            if cuda:
                data, labels, text_features, local_feats_sample, text_bboxes_features, local_bboxes_features = data.cuda(), labels.cuda(), text_features.cuda(),\
                                                                                             local_feats_sample.cuda(), text_bboxes_features.cuda(), local_bboxes_features.cuda()
            data = Variable(data)

            output, attn_mask, affinity_matrix = net(data, text_features, sample_size, local_feats_sample, text_bboxes_features,
                                                     local_bboxes_features)

            softmax = nn.Softmax(dim=1)
            features = softmax(output)
            features = features.cpu().numpy()
            features = features.tolist()
            if args.dataset == 'bottles':
                image = image.replace('images/15/','')
            with open (args.input_dir + 'single_features_' + args.dataset + '/'+ image.split('.')[0] + '.json','w') as fp:
                json.dump(features, fp)

            processed_imgs += 1

    print ('Process Completed - %d Processed Images' %(processed_imgs))
    return

def main():
    print('Preparing data')
    data_path = args.base_dir

    if args.dataset == 'context':
        num_classes = 28
        weight_file = '/SSD/GCN_classification/best/context_fullGCN_bboxes_fasttext_google_ocr_concat_mean_split1/checkpoint_context.weights'

        with open(data_path + '/Context/data/split_1.json', 'r') as fp:
            gt_annotations = json.load(fp)
        with open(data_path + '/Context/' + args.ocr + '/text_embeddings/Context_' + args.embedding + '.json','r') as fp:
            text_embedding = json.load(fp)

        # Load Local features from Faster R-CNN VG
        with open(data_path + '/Context/context_local_feats.npy', 'rb') as fp:
            local_feats = np.load(fp, encoding='bytes')

            # Create img_name to index of local features
        with open(data_path + '/Context/context_local_feats_image_ids.txt', 'r') as fp:
            image_ids = fp.readlines()
        image_name2features_index = {}
        for item in image_ids:
            img_name = item.strip().split(',')[0].split('/')[-1].replace('\'', '')
            idx = item.strip().split(',')[1].replace(')', '').replace(' ', '')
            image_name2features_index[img_name] = idx

        # BBOXES LOADING FOR TEXT FEATURES
        # Load BBOXES of Scene Text
        with open(data_path + '/Context/google_ocr/bboxes/Context_bboxes.json', 'r') as fp:
            text_bboxes = json.load(fp)

        # Load BBOXES of Local Visual Features
        with open(data_path + '/Context/context_bboxes.npy', 'rb') as fp:
            local_bboxes = np.load(fp, encoding='bytes')

    else:
        num_classes = 20
        weight_file = '/SSD/GCN_classification/best/bottles_fullGCN_bboxes_fasttext_google_ocr_concat_mean_split2/checkpoint_bottles.weights'

        with open(data_path + '/Drink_Bottle/split_2.json', 'r') as fp:
            gt_annotations = json.load(fp)
        with open(data_path + '/Drink_Bottle/' + args.ocr + '/text_embeddings/Drink_Bottle_' + args.embedding + '.json','r') as fp:
            text_embedding = json.load(fp)

        # Load Local features from Faster R-CNN VG
        with open(data_path + '/Drink_Bottle/bottles_local_feats.npy', 'rb') as fp:
            local_feats = np.load(fp, encoding='bytes')

            # Create img_name to index of local features
        with open(data_path + '/Drink_Bottle/bottles_local_feats_image_ids.txt', 'r') as fp:
            image_ids = fp.readlines()
        image_name2features_index = {}
        for item in image_ids:
            # Sample: ('/SSD/Datasets/Drink_Bottle/images/14/982.jpg', 0)
            img_name = item.strip().split(',')[0].replace('\'', '').split('/')[-3:]
            img_name = img_name[0] + '/' + img_name[1] + '/' + img_name[2]
            idx = item.strip().split(',')[1].replace(')', '').replace(' ', '')
            image_name2features_index[img_name] = idx

        # BBOXES LOADING FOR TEXT FEATURES
        # Load BBOXES of Scene Text
        with open(data_path + '/Drink_Bottle/google_ocr/bboxes/Drink_Bottle_bboxes.json', 'r') as fp:
            text_bboxes = json.load(fp)

        # Load BBOXES of Local Visual Features
        with open(data_path + '/Drink_Bottle/bottles_bboxes.npy', 'rb') as fp:
            local_bboxes = np.load(fp, encoding='bytes')


    embedding_size = get_embedding_size(args.embedding)
    print('Loading Model')
    net = load_model(args, num_classes, embedding_size)
    checkpoint = load_checkpoint(weight_file)
    net.load_state_dict(checkpoint)

    print('Checking CUDA')
    if args.cuda and args.ngpu > 1:
        print('\t* Data Parallel **NOT TESTED**')
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    if args.cuda:
        print('\t* CUDA ENABLED!')
        net = net.cuda()


    print('\n*** TEST ***\n')
    test(args, net, args.cuda, num_classes, gt_annotations, text_embedding, local_feats, image_name2features_index, text_bboxes, local_bboxes)
    print('*** Feature Extraction Completed ***')
    sys.exit()

if __name__ == '__main__':
    # Parse options
    args = Options_Test().parse()
    print('Parameters:\t' + str(args))

    # Check cuda & Set random seed
    args.cuda = args.ngpu > 0 and torch.cuda.is_available()
    main()