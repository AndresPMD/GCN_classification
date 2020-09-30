# -*- coding: utf-8 -*-
"""
    Parse input arguments
"""

import argparse


class Options():

    def __init__(self, test=False):
        # MODEL SETTINGS
        parser = argparse.ArgumentParser(description='Fine-grained Classification based on textual cues + GCN',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # Positional arguments
        parser.add_argument('dataset', type=str, choices=['context','bottles'],
                            help='Choose the Data set to use (context or bottles).')
        # Model
        parser.add_argument('--embedding',type=str, default = 'fasttext', choices=['w2vec','fasttext','glove', 'phoc', 'fisher'],
                            help='Choose between embeddings.')

        parser.add_argument('--ocr', type=str, default = 'google_ocr', choices=['textspotter','deeptextspotter','e2e_mlt', 'yolo_phoc', 'google_ocr'],
                            help='Choose between OCRs.')

        parser.add_argument('--model', type=str, default='fullGCN_bboxes', choices=['visualNet', 'lenet', 'baseNet', 'fisherNet', 'orig_fisherNet', 'TextNet',
                                                                             'globalNet', 'baseGCN', 'textGCN', 'fullGCN', 'fullGCN_attn', 'dualGCN',
                                                                             'fullGCN_bboxes'],
                            help='Choose between models.')

        parser.add_argument('--attn', action='store_true', help='Attention module')
        parser.add_argument('--plot', action='store_true', help='Qualitative results')
        parser.add_argument('--max_textual', type = int, default=15, help='Size of the text matrix.')
        parser.add_argument('--max_visual', type=int, default=36, help='Size of the local visual matrix from FRCNN.')

        # Splits
        parser.add_argument('--split', '-sp', type=int, default=2, help='Train/Test splits to use.')
        parser.add_argument('--test', type=str, default='False', help='Train/Test mode')

        # Data Path
        parser.add_argument('--data_path', type=str, default='/SSD/Datasets', help='Write the datasets path.')
        # Fusion Strategy

        parser.add_argument('--projection_layer', type=str, default='mean',
                            help='Options to project visual features after GCN, includes GRU, Fully Connected, Mean Pooling, Attention on Global feats',
                            choices=['mean', 'gru', 'fc', 'attention'])

        parser.add_argument('--fusion', type=str, default='concat', choices=['concat', 'dot', 'attention', 'block', 'mlb'],
                            help='Choose between last layer fusion strategies.')
        # MULTIMODAL DIM
        parser.add_argument('--mmdim', type=int, default=1600, help='Size of the Inner Multimodal Embedding')

        # VISUALIZATION

        parser.add_argument('--outimg', type=str, default=False, help='Visualization of attn True/False.')
        parser.add_argument('--outimg_path', type=str, default='../visualization/', help='Write the attn visual path.')

        # STORE
        parser.add_argument('--save_weights', type=str, default='True', help='Store training weights True/False')
        parser.add_argument('--save_img_feats', type=str, default='False', help='Store affinity matrix True/False')

        # Optimization options
        parser.add_argument('--epochs', '-e', type=int, default=60, help='Number of epochs to train.')
        parser.add_argument('--batch_size', '-b', type=int, default=64, help='Batch size.')
        parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3, help='The Learning Rate.')
        parser.add_argument('--optim', '-o', type=str, default='radam', help='Optimizers: sgd, adam, radam(with lookAhead)')
        parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
        parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
        parser.add_argument('--schedule', type=list, nargs='+', default=[15,30,45,48,50],
                            help='Decrease learning rate at these epochs.')

        parser.add_argument('--gamma', type=float, default=0.1, help='blocktuckR is multiplied by gamma on schedule.')
        parser.add_argument('--seed', type=int, default=42, help='Random seed.')
        parser.add_argument('--save', '-s', type=str, default='../backup/',
                            help='Folder to save checkpoints.')
        parser.add_argument('--load', '-l', type=str, default=None, help='Checkpoint path to resume / test.')
        parser.add_argument('--early_stop', '-es', type=int, default=10, help='Early stopping epochs.')
        parser.add_argument('--grad_clip', type=float, default=0.0, help='Gradient Clipping float value.')

        # Acceleration
        parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU, 1 = CUDA, 1 < DataParallel')
        parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
        # i/o
        parser.add_argument('--log', type=str, default='../results/',
                            help='Log folder.')
        parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                            help='How many batches to wait before logging training status')

        self.parser = parser

    def parse(self):
        return self.parser.parse_args()

class Options_Test():

    def __init__(self):
        # MODEL SETTINGS
        parser = argparse.ArgumentParser(description='Fine-grained Classification based on textual cues',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # Positional arguments
        parser.add_argument('dataset', type=str, choices=['context','bottles'],
                            help='Choose the Data set to use (context or bottles).')
        # Model
        parser.add_argument('--embedding',type=str, default = 'fasttext', choices=['w2vec','fasttext','glove', 'bert', 'phoc', 'fisher'],
                            help='Choose between embeddings.')

        parser.add_argument('--ocr', type=str, default = 'google_ocr', choices=['textspotter','deeptextspotter','e2e_mlt', 'yolo_phoc', 'google_ocr'],
                            help='Choose between OCRs.')

        parser.add_argument('--model', type=str, default='fullGCN_bboxes', choices=['visualNet', 'lenet', 'baseNet', 'fisherNet', 'orig_fisherNet',
                                                                                    'TextNet', 'fullGCN_bboxes'],
                            help='Choose between models.')

        parser.add_argument('--projection_layer', type=str, default='mean',
                            help='Options to project visual features after GCN, includes GRU, Fully Connected, Mean Pooling, Attention on Global feats',
                            choices=['mean', 'gru', 'fc', 'attention'])

        parser.add_argument('--fusion', type=str, default='concat', choices=['concat', 'dot', 'attention', 'block', 'mlb'],
                            help='Choose between last layer fusion strategies.')

        parser.add_argument('--attn', action='store_true', help='Attention module')
        parser.add_argument('--plot', action='store_true', help='Qualitative results')
        parser.add_argument('--max_textual', type = int, default=15, help='Size of the text matrix.')
        parser.add_argument('--max_visual', type=int, default=36, help='Size of the local visual matrix from FRCNN.')

        # Data Path
        parser.add_argument('--base_dir', type=str, default='/SSD/Datasets/', help='Write the base path to read Data related features')

        # Input images folder
        parser.add_argument('--input_dir', type=str, default='/home/amafla/Desktop/wacv2021/samples/inverse/',
                            help='Path to read images from')

        # MULTIMODAL DIM
        parser.add_argument('--blur', type=str, default='none', help='Input image is blurred: visual, text, none')

        parser.add_argument('--load', '-l', type=str, default=None, help='Checkpoint path to resume / test.')

        # Acceleration
        parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU, 1 = CUDA, 1 < DataParallel')
        parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')

        self.parser = parser

    def parse(self):
        return self.parser.parse_args()
