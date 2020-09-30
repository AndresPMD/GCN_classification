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
from sklearn.metrics import average_precision_score

import pdb
import sys
from tqdm import tqdm
import pickle

# Own modules

from utils import *
from options import *
from data.data_generator import *
from models.models import load_model
from custom_optim import *

# __author__ = "Andres Mafla Delgado;
# __email__ = "amafla@cvc.uab.cat;

# READS A LIST OF IMAGES BELONGING TO CON-TEXT OR BOTTLES AND EXTRACTS THE FEATURES AND PROBS - blur experiments

def test(args, net, cuda, num_classes, gt_annotations, text_embedding, local_feats, image_name2features_index, text_bboxes, local_bboxes, images_to_process):

    processed_imgs = 0
    # Switch to evaluation mode
    net.eval()

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    with torch.no_grad():
    	images_to_proc = images_to_process.keys()
    	gt_annotations_test = gt_annotations['test'].keys()
    	image_list = [i for i in images_to_proc if i in gt_annotations_test]
    	print('Processing %d images\n' %(len(image_list)))
    	# For Precision Metric
    	precision_per_class = [0.0] * num_classes
    	class_total = [0.00001] * num_classes

    	for image in tqdm(image_list):
            sample_size = 1
            if args.dataset == 'context':
            	img_path = args.base_dir + 'Context/data/JPEGImages/' + image
            	relative_path = '/Context/old_fisher_vectors/'
            	label = np.zeros(28)
            else:
            	img_path = args.base_dir + 'Drink_Bottle/' + image
            	relative_path = 'Drink_Bottle/old_fisher_vectors/'
            	label = np.zeros(20)

            img = Image.open(img_path).convert('RGB')
            data = test_transform(img)
            data = data.view(-1, 3, 224, 224)

            # Labels
            img_class = images_to_process[image]
            label[int(img_class) - 1] = 1
            
            # Textual Features
            if args.blur == 'none' or args.blur == 'visual':
            	if args.ocr =='yolo_phoc' and args.embedding == 'fisher':
            		with open ( args.base_dir + relative_path + image.replace('images/','')[:-3] +'json', 'r') as fp:
            			fisher_vector = json.load(fp)
            			text_features = np.resize(fisher_vector, (1, 38400))

            	else:
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
            if args.embedding == 'fisher':
            	text_features = text_features.view(sample_size, 38400)
            else:
            	text_features = text_features.view(sample_size, 15, 300)

            # SCENE TEXT BBOXES ONLY FOR GOOGLE OCR
            if args.blur == 'none' or args.blur =='visual':
                text_bboxes_sample = np.asarray(text_bboxes[image.replace('images/','')])
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
            	if args.dataset == 'bottles':
            		image = 'images/'+ image
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
                data, text_features, local_feats_sample, text_bboxes_features, local_bboxes_features = data.cuda(), text_features.cuda(),\
                                                                                             local_feats_sample.cuda(), text_bboxes_features.cuda(), local_bboxes_features.cuda()
            data = Variable(data)

            output, attn_mask, affinity_matrix = net(data, text_features, sample_size, local_feats_sample, text_bboxes_features,
                                                     local_bboxes_features)

            class_total[int(img_class) - 1] += 1
            # Precision
            softmax = nn.Softmax(dim=1)
            predicted = softmax(output)
            predicted = predicted.data.cpu().numpy()
            y_true = label
            precision_per_class[int(img_class) - 1] += average_precision_score(y_true, predicted.reshape(num_classes, ))
            processed_imgs += 1

    total_precision = [0.0] * num_classes
    for ix, value in enumerate (precision_per_class):
        total_precision[ix] = precision_per_class[ix]/class_total[ix]
        #print ('Average Precision for %d class: %.4f' % (ix + 1, total_precision[ix] ))
    total_mAP =sum(total_precision) / num_classes
    print('Mean Average Precision (mAP) is: %.4f' % (100 * total_mAP))
    print ('Process Completed - %d Processed Images' %(processed_imgs))
    return

def main():
    print('Preparing data')
    data_path = args.base_dir

    if args.dataset == 'context':
        num_classes = 28
        if args.embedding == 'fasttext':
        	weight_file = '/SSD/GCN_classification/best/context_fullGCN_bboxes_fasttext_google_ocr_concat_mean_split1/checkpoint_context.weights'
        elif args.embedding == 'fisher':
        	weight_file = '/SSD/GCN_classification/backup/context_orig_fisherNet_fisher_yolo_phoc_concat_mean/checkpoint_context.weights'
        elif args.embedding == 'glove':
        	weight_file = '/SSD/GCN_classification/backup/context_lenet_glove_e2e_mlt_concat_mean/checkpoint_context.weights'
        else:
        	print('Embedding not implemented for Performance eval')

        with open(data_path + '/Context/data/split_1.json', 'r') as fp:
            gt_annotations = json.load(fp)

        if args.embedding != 'glove':
        	with open(data_path + '/Context/google_ocr/text_embeddings/Context_fasttext.json','r') as fp:
        		text_embedding = json.load(fp)
        else:
        	with open(data_path + '/Context/' + args.ocr + '/text_embeddings/Context_' + args.embedding + '.pickle','rb') as fp:
        		text_embedding = pickle.load(fp)

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

        # Images with and without text
        with open (data_path + '/Context/data/images_with_text.json','r') as fp:
        	images_with_text = json.load(fp)
        with open (data_path + '/Context/data/images_no_text.json','r') as fp:
        	images_no_text = json.load(fp)

    else:
        num_classes = 20
        if args.embedding == 'fasttext':
        	weight_file = '/SSD/GCN_classification/best/bottles_fullGCN_bboxes_fasttext_google_ocr_concat_mean_split2/checkpoint_bottles.weights'
        elif args.embedding == 'fisher':
        	weight_file = '/SSD/GCN_classification/backup/bottles_orig_fisherNet_fisher_yolo_phoc_concat_mean/checkpoint_bottles.weights'
        elif args.embedding == 'glove':
        	weight_file = '/SSD/GCN_classification/backup/bottles_lenet_glove_e2e_mlt_concat_mean/checkpoint_bottles.weights'
        else:
        	print('Embedding not implemented for Performance eval')

        with open(data_path + '/Drink_Bottle/split_2.json', 'r') as fp:
            gt_annotations = json.load(fp)

        if args.embedding != 'glove':
        	with open(data_path + '/Drink_Bottle/google_ocr/text_embeddings/Drink_Bottle_fasttext.json','r') as fp:
        		text_embedding = json.load(fp)
        else:
        	with open(data_path + '/Drink_Bottle/' + args.ocr + '/text_embeddings/Drink_Bottle_' + args.embedding + '.pickle','rb') as fp:
        		text_embedding = pickle.load(fp)

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

        # Images with and without text
        with open (data_path + '/Drink_Bottle/images_with_text.json','r') as fp:
        	images_with_text = json.load(fp)
        with open (data_path + '/Drink_Bottle/images_no_text.json','r') as fp:
        	images_no_text = json.load(fp)


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
    print('\n*** Evaluating Performance on Images WITH Scene Text ***\n')
    test(args, net, args.cuda, num_classes, gt_annotations, text_embedding, local_feats, image_name2features_index, text_bboxes, local_bboxes, images_with_text)
    print('\n*** Evaluating Performance on Images WITHOUT Scene Text ***\n')
    test(args, net, args.cuda, num_classes, gt_annotations, text_embedding, local_feats, image_name2features_index, text_bboxes, local_bboxes, images_no_text)
    print('\nProcess Completed!')
    sys.exit()

if __name__ == '__main__':
    # Parse options
    args = Options_Test().parse()
    print('Parameters:\t' + str(args))

    # Check cuda & Set random seed
    args.cuda = args.ngpu > 0 and torch.cuda.is_available()
    main()