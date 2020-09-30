# -*- coding: utf-8 -*-
from __future__ import print_function, division
import sys
sys.path.insert(0,'.')

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import pdb

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# Custom fusion modules
from .fusion import *
from .Rs_GCN import *

"""
Visual Encoder model
"""

def load_model(args, classes_number, embedding_size):

    if args.model == 'visualNet':
        return Resnet_CNN(args=args, num_classes=classes_number, embedding_size=embedding_size)
    elif args.model == 'lenet':
        return Lenet_CNN(args = args, num_classes= classes_number, embedding_size=embedding_size)
    elif args.model == 'baseNet':
        return BaseNet(args = args, num_classes= classes_number, embedding_size=embedding_size)
    elif args.model == 'fisherNet':
        return FisherNet(args = args, num_classes= classes_number, max_textual = 1, embedding_size=embedding_size, reduced_size = 512)
    elif args.model == 'orig_fisherNet':
        return Orig_FisherNet(args = args, num_classes= classes_number, max_textual = 1, embedding_size=embedding_size, reduced_size = 512)
    elif args.model == 'TextNet':
        return TextNet(args = args, num_classes= classes_number, embedding_size=embedding_size, reduced_size = 512)
    elif args.model == 'globalNet':
        return globalNet(args=args, num_classes=classes_number, embedding_size=embedding_size)
    elif args.model == 'baseGCN':
        return baseGCN(args=args, num_classes=classes_number, embedding_size=embedding_size)
    elif args.model == 'textGCN':
        return textGCN(args=args, num_classes=classes_number, embedding_size=embedding_size)
    elif args.model == 'fullGCN':
        return fullGCN(args=args, num_classes=classes_number, embedding_size=embedding_size)
    elif args.model == 'fullGCN_attn':
        return fullGCN_attn(args=args, num_classes=classes_number, embedding_size=embedding_size)
    elif args.model == 'dualGCN':
        return dualGCN(args=args, num_classes=classes_number, embedding_size=embedding_size)
    elif args.model == 'fullGCN_bboxes':
        return fullGCN_bboxes(args=args, num_classes=classes_number, embedding_size=embedding_size)
    else:
        raise NameError(args.model + ' not implemented!')


class AttentionModel(nn.Module):
    def __init__(self, hidden_layer=380):
        super(AttentionModel, self).__init__()

        self.attn_hidden_layer = hidden_layer
        self.net = nn.Sequential(nn.Conv2d(2048, self.attn_hidden_layer, kernel_size=1),
                                 nn.Conv2d(self.attn_hidden_layer, 1, kernel_size=1))

    def forward(self, x):
        attn_mask = self.net(x) # Shape BS 1x7x7
        attn_mask = attn_mask.view(attn_mask.size(0), -1)
        attn_mask = nn.Softmax(dim=1)(attn_mask)
        attn_mask = attn_mask.view(attn_mask.size(0), 1, x.size(2), x.size(3))
        x_attn = x * attn_mask
        x = x + x_attn
        return x, attn_mask


class Lenet_CNN(nn.Module):
    def __init__(self, args, num_classes, embedding_size, pretrained=True):
        super(Lenet_CNN, self).__init__()
        self.args = args
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.pretrained = pretrained
        lenet = models.googlenet(pretrained)

        for name, child in lenet.named_children():
            for param in child.parameters():
                param.requires_grad = False

        self.cnn_features = nn.Sequential(*list(lenet.children())[:-1])
        # Initial Vf
        self.bn_vf = nn.BatchNorm1d(1024)
        self.fc_vf = nn.Linear(1024, 1024)

        # Initial Tf
        self.bn_tf = nn.BatchNorm1d(15)
        self.fc_tf = nn.Linear(300, 300)

        # Semantic Attention Weights
        self.bn_w = nn.BatchNorm1d(1024)
        self.fc_w = nn.Linear(1024, 300, bias=False)

        # Reshape Visual Features Before Concat
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512)

        # LAST LAYERS
        self.bn_clf = nn.BatchNorm1d(512 + 300)
        self.fc_clf = nn.Linear(512 + 300, num_classes)


    def forward(self, im, textual_features, sample_size, local_features, text_bboxes, local_bboxes):
        vf = self.cnn_features(im)  # Size (BS x 2048 x 7 x 7)
        vf = vf.view(sample_size, 1024)
        vf = F.leaky_relu(self.fc_vf(self.bn_vf(vf)))

        textual_features = F.leaky_relu(self.fc_tf(self.bn_tf(textual_features)))

        wi = self.fc_w(self.bn_w(vf))
        wi = torch.bmm(wi.view(sample_size, 1, 300), textual_features.permute(0, 2, 1))
        wi = torch.tanh(wi)
        wi = F.softmax(wi, dim=2)
        # Attention over textual features
        textual_features = torch.bmm(wi, textual_features)
        # Reshape vf before concat
        vf = self.bn1(vf)
        vf = F.leaky_relu(self.fc1(vf))

        x = torch.cat((textual_features[:, 0, :], vf), 1)
        x = F.dropout(self.fc_clf(self.bn_clf(x)), p=0.3, training=self.training)
        
        return x, 0, 0


class Resnet_CNN(nn.Module):
    def __init__(self, args , num_classes, embedding_size, pretrained=True, attention=True):
        super(Resnet_CNN, self).__init__()
        self.args = args
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.pretrained = pretrained
        resnet152 = models.resnet152(pretrained)

        for name, child in resnet152.named_children():
            if name not in ['layer4']:
                # print(name + ' is frozen')
                for param in child.parameters():
                    param.requires_grad = False
            else:
                # print(name + ' is not frozen')
                for param in child.parameters():
                    param.requires_grad = True

        self.cnn_features = nn.Sequential(*list(resnet152.children())[:-2])
        #
        #
        # for param in self.cnn_features.parameters():
        #     param.requires_grad = False

        # Attention model
        self.attention = attention
        self.attn = AttentionModel()
        self.attn_bn = nn.BatchNorm2d(2048)

        # OUTPUT OF CNN BS X 2048 X 7 X 7 =  100352
        self.fc1_bn = nn.BatchNorm1d(2048*7*7)
        self.fc1 = nn.Linear(2048*7*7, num_classes)


    def forward(self, im, textual_features, sample_size, local_features):
        x = self.cnn_features(im)  # Size (BS x 2048 x 7 x 7)

        if self.attention:
            x, attn_mask = self.attn(x)  # Size (BS x 2048)
        x = self.attn_bn(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(self.fc1_bn(x)))

        return x, attn_mask


class BaseNet(nn.Module):
    def __init__(self, args, num_classes, embedding_size = 300, pretrained=True, attention=True):
        super(BaseNet, self).__init__()
        self.args = args
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.embedding_size = embedding_size

        if self.args.fusion == 'block':
            self.fusion = Block([embedding_size, 1024], 1024+embedding_size, mm_dim= self.args.mmdim)
        elif self.args.fusion == 'blocktucker':
            self.fusion = BlockTucker([embedding_size, 1024], 1024+embedding_size, mm_dim= self.args.mmdim)
        elif self.args.fusion == 'tucker':
            self.fusion = Tucker ([embedding_size, 1024], 1024+embedding_size, mm_dim= self.args.mmdim)
        elif self.args.fusion == 'mutan':
            self.fusion = Mutan([embedding_size, 1024], 1024+embedding_size, mm_dim= self.args.mmdim)
        elif self.args.fusion == 'mlb':
            self.fusion = MLB([embedding_size, 1024], 1024+embedding_size, mm_dim= self.args.mmdim)
        elif self.args.fusion == 'mfb':
            self.fusion = MFB([embedding_size, 1024], 1024+embedding_size, mm_dim= self.args.mmdim)
        elif self.args.fusion == 'mfh':
            self.fusion = MFH([embedding_size, 1024], 1024+embedding_size, mm_dim= self.args.mmdim)

        # models.densenet169()
        resnet152 = models.resnet152(pretrained)

        for name, child in resnet152.named_children():
            if name not in ['layer4']:
                #print(name + ' is frozen')
                for param in child.parameters():
                    param.requires_grad = False
            else:
                #print(name + ' is not frozen')
                for param in child.parameters():
                    param.requires_grad = True

        self.cnn_features = nn.Sequential(*list(resnet152.children())[:-2])

        # Attention model
        self.attention = attention
        self.attn = AttentionModel()
        self.attn_bn = nn.BatchNorm2d(2048)

        self.fc1 = nn.Linear(100352, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)

        # Semantic Attention Weights
        self.fc_w = nn.Linear(1024, self.embedding_size, bias=False)

        # LAST LAYERS
        self.bn3 = nn.BatchNorm1d(1024 + self.embedding_size)
        self.fc3 = nn.Linear(1024 + self.embedding_size, num_classes)


        '''
        self.fc3 = nn.Linear(1024 + self.embedding_size, 300)

        # CLASSIF LAYER
        self.bn4 = nn.BatchNorm1d(300)
        self.fc4 = nn.Linear(300, num_classes)
        '''

    def forward(self, im, textual_features, sample_size):
        x = self.cnn_features(im)  # Size (BS x 2048 x 7 x 7)

        if self.attention:
            x, attn_mask = self.attn(x)  # Size (BS x 2048)

        x = self.attn_bn(x)
        x = x.view(x.size(0), -1)
        visual_features = F.relu(self.fc1_bn(self.fc1(x))) # Visual Features BS x 1024
        x = self.fc_w(visual_features) # BS x 300 or (embedding size)

        x = torch.bmm(x.view(sample_size, 1, self.embedding_size), textual_features.permute(0, 2, 1))
        x = torch.tanh(x)
        x = F.softmax(x, dim=2)
        # Attention over textual features
        x = torch.bmm(x, textual_features)
        # Reshape visual features before fusion
        # Fuse
        if self.args.fusion != 'concat':
            x = self.fusion([x.view(sample_size, -1),visual_features])
        else:
            x = torch.cat((x[:, 0, :], visual_features), 1)
        '''
        ranking_vector = F.relu(self.fc3(self.bn3(x)))
        x = F.dropout(self.fc4(self.bn4(ranking_vector)), p=0.3, training=self.training)
        '''
        x = F.dropout(self.fc3(self.bn3(x)), p=0.3, training=self.training)

        return x, attn_mask


class FisherNet(nn.Module):
    def __init__(self, args, num_classes, max_textual = 20, embedding_size = 38400, reduced_size = 512, pretrained=True, attention=True):
        super(FisherNet, self).__init__()
        self.args = args
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.embedding_size = embedding_size
        self.reduced_size = reduced_size
        self.max_textual = max_textual

        if self.args.fusion == 'block':
            self.fusion = Block([reduced_size, 1024], 1024+reduced_size, mm_dim= self.args.mmdim)
        elif self.args.fusion == 'blocktucker':
            self.fusion = BlockTucker([reduced_size, 1024], 1024+reduced_size, mm_dim= self.args.mmdim)
        elif self.args.fusion == 'tucker':
            self.fusion = Tucker ([reduced_size, 1024], 1024+reduced_size, mm_dim= self.args.mmdim)
        elif self.args.fusion == 'mutan':
            self.fusion = Mutan([reduced_size, 1024], 1024+reduced_size, mm_dim= self.args.mmdim)
        elif self.args.fusion == 'mlb':
            self.fusion = MLB([reduced_size, 1024], 1024+reduced_size, mm_dim= self.args.mmdim)
        elif self.args.fusion == 'mfb':
            self.fusion = MFB([reduced_size, 1024], 1024+reduced_size, mm_dim= self.args.mmdim)
        elif self.args.fusion == 'mfh':
            self.fusion = MFH([reduced_size, 1024], 1024+reduced_size, mm_dim= self.args.mmdim)

        resnet152 = models.resnet152(pretrained)

        for name, child in resnet152.named_children():
            if name not in ['layer4']:
                #print(name + ' is frozen')
                for param in child.parameters():
                    param.requires_grad = False
            else:
                #print(name + ' is not frozen')
                for param in child.parameters():
                    param.requires_grad = True

        self.cnn_features = nn.Sequential(*list(resnet152.children())[:-2])

        # Reduce Dimensionality of Fisher Vectors
        self.FV_bn1 = nn.BatchNorm1d(embedding_size)
        self.FV_fc1 = nn.Linear(embedding_size, 4096)
        self.FV_bn2 = nn.BatchNorm1d(4096)
        self.FV_fc2 = nn.Linear(4096, reduced_size)

        # Attention model
        self.attention = attention
        self.attn = AttentionModel()
        self.attn_bn = nn.BatchNorm2d(2048)

        self.fc1 = nn.Linear(100352, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)

        # Semantic Attention Weights
        self.fc_w = nn.Linear(1024, self.reduced_size, bias=False)

        # LAST LAYERS
        self.bn3 = nn.BatchNorm1d(1024 + self.reduced_size)
        self.fc3 = nn.Linear(1024 + self.reduced_size, num_classes)

    def forward(self, im, textual_features, sample_size):
        x = self.cnn_features(im)  # Size (BS x 2048 x 7 x 7)

        if self.attention:
            x, attn_mask = self.attn(x)  # Size (BS x 2048)

        x = self.attn_bn(x)
        x = x.view(x.size(0), -1)
        visual_features = F.relu(self.fc1_bn(self.fc1(x))) # Visual Features BS x 1024
        x = self.fc_w(visual_features) # BS x 300 or (embedding size)

        textual_features = F.relu(self.FV_fc1(self.FV_bn1(textual_features.view(sample_size, -1))))
        textual_features = F.dropout(F.relu(self.FV_fc2(self.FV_bn2(textual_features))), p=0.5, training=self.training)

        x = torch.mul(x, textual_features)
        x = torch.tanh(x)
        x = torch.mul(x, textual_features)
        # Reshape visual features before fusion
        # Fuse
        if self.args.fusion != 'concat':
            x = self.fusion([x.view(sample_size, -1),visual_features])
        else:
            x = torch.cat((x, visual_features), 1)

        x = F.dropout(self.fc3(self.bn3(x)), p=0.5, training=self.training)

        return x, attn_mask


class Orig_FisherNet(nn.Module):
    def __init__(self, args, num_classes, max_textual = 20, embedding_size = 38400, reduced_size = 512, pretrained=True, attention=True):
        super(Orig_FisherNet, self).__init__()
        self.args = args
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.embedding_size = embedding_size
        self.reduced_size = reduced_size
        self.max_textual = max_textual

        if self.args.fusion == 'block':
            self.fusion = Block([reduced_size, 1024], 1024+reduced_size, mm_dim= self.args.mmdim)
        elif self.args.fusion == 'blocktucker':
            self.fusion = BlockTucker([reduced_size, 1024], 1024+reduced_size, mm_dim= self.args.mmdim)
        elif self.args.fusion == 'tucker':
            self.fusion = Tucker ([reduced_size, 1024], 1024+reduced_size, mm_dim= self.args.mmdim)
        elif self.args.fusion == 'mutan':
            self.fusion = Mutan([reduced_size, 1024], 1024+reduced_size, mm_dim= self.args.mmdim)
        elif self.args.fusion == 'mlb':
            self.fusion = MLB([reduced_size, 1024], 1024+reduced_size, mm_dim= self.args.mmdim)
        elif self.args.fusion == 'mfb':
            self.fusion = MFB([reduced_size, 1024], 1024+reduced_size, mm_dim= self.args.mmdim)
        elif self.args.fusion == 'mfh':
            self.fusion = MFH([reduced_size, 1024], 1024+reduced_size, mm_dim= self.args.mmdim)

        resnet152 = models.resnet152(pretrained)

        for name, child in resnet152.named_children():
            if name not in ['layer4']:
                #print(name + ' is frozen')
                for param in child.parameters():
                    param.requires_grad = False
            else:
                #print(name + ' is not frozen')
                for param in child.parameters():
                    param.requires_grad = True

        self.cnn_features = nn.Sequential(*list(resnet152.children())[:-2])

        # Reduce Dimensionality of Fisher Vectors
        self.FV_bn1 = nn.BatchNorm1d(embedding_size)
        self.FV_fc1 = nn.Linear(embedding_size, 4096)
        self.FV_bn2 = nn.BatchNorm1d(4096)
        self.FV_fc2 = nn.Linear(4096, reduced_size)

        # Attention model
        self.attention = attention
        self.attn = AttentionModel()
        self.attn_bn = nn.BatchNorm2d(2048)

        self.fc1 = nn.Linear(100352, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)

        # Semantic Attention Weights
        self.fc_w = nn.Linear(1024, self.reduced_size, bias=False)

        # LAST LAYERS
        self.bn3 = nn.BatchNorm1d(1024 + self.reduced_size)
        self.fc3 = nn.Linear(1024 + self.reduced_size, num_classes)

    def forward(self, im, textual_features, sample_size, local_features, text_bboxes, local_bboxes):
        x = self.cnn_features(im)  # Size (BS x 2048 x 7 x 7)

        if self.attention:
            x, attn_mask = self.attn(x)  # Size (BS x 2048)

        x = self.attn_bn(x)
        x = x.view(x.size(0), -1)
        visual_features = F.relu(self.fc1_bn(self.fc1(x))) # Visual Features BS x 1024
        x = self.fc_w(visual_features) # BS x 300 or (embedding size)

        # FISHER FEATURES
        textual_features = F.relu(self.FV_fc1(self.FV_bn1(textual_features.view(sample_size, -1))))
        #textual_features = F.dropout(F.relu(self.FV_fc2(self.FV_bn2(textual_features))), p=0.5, training=self.training)
        textual_features = F.dropout(self.FV_fc2(self.FV_bn2(textual_features)), p=0.5, training=self.training)


        x = torch.mul(x, textual_features)
        x = torch.tanh(x)
        x = torch.mul(x, textual_features)
        # Reshape visual features before fusion
        # Fuse
        if self.args.fusion != 'concat':
            x = self.fusion([x.view(sample_size, -1),visual_features])
        else:
            x = torch.cat((x, visual_features), 1)

        x = F.dropout(self.fc3(self.bn3(x)), p=0.5, training=self.training)

        return x, attn_mask, 0

class TextNet(nn.Module):
    def __init__(self, args, num_classes, embedding_size = 300, reduced_size=512, pretrained=True, attention=True):
        super(TextNet, self).__init__()
        self.args = args
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.embedding_size = embedding_size
        self.reduced_size = reduced_size

        if self.args.fusion == 'block':
            self.fusion = Block([embedding_size, 1024], 1024+embedding_size, mm_dim= self.args.mmdim)
        elif self.args.fusion == 'blocktucker':
            self.fusion = BlockTucker([embedding_size, 1024], 1024+embedding_size, mm_dim= self.args.mmdim)
        elif self.args.fusion == 'tucker':
            self.fusion = Tucker ([embedding_size, 1024], 1024+embedding_size, mm_dim= self.args.mmdim)
        elif self.args.fusion == 'mutan':
            self.fusion = Mutan([embedding_size, 1024], 1024+embedding_size, mm_dim= self.args.mmdim)
        elif self.args.fusion == 'mlb':
            self.fusion = MLB([embedding_size, 1024], 1024+embedding_size, mm_dim= self.args.mmdim)
        elif self.args.fusion == 'mfb':
            self.fusion = MFB([embedding_size, 1024], 1024+embedding_size, mm_dim= self.args.mmdim)
        elif self.args.fusion == 'mfh':
            self.fusion = MFH([embedding_size, 1024], 1024+embedding_size, mm_dim= self.args.mmdim)

        resnet152 = models.resnet152(pretrained)

        for name, child in resnet152.named_children():
            if name not in ['layer4']:
                #print(name + ' is frozen')
                for param in child.parameters():
                    param.requires_grad = False
            else:
                #print(name + ' is not frozen')
                for param in child.parameters():
                    param.requires_grad = True

        self.cnn_features = nn.Sequential(*list(resnet152.children())[:-2])

        # Attention model
        self.attention = attention
        self.attn = AttentionModel()
        self.attn_bn = nn.BatchNorm2d(2048)

        self.fc1 = nn.Linear(100352, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)

        # Semantic Attention Weights
        self.fc_w = nn.Linear(1024, self.reduced_size, bias=False)

        # LAST LAYERS
        self.bn3 = nn.BatchNorm1d(1024 + self.reduced_size)
        self.fc3 = nn.Linear(1024 + self.reduced_size, num_classes)

        # ADDITIONAL LAYERS TO TEST SELF LEARNING OF MORPHOLOGY
        self.bn_text1 = nn.BatchNorm1d(self.args.max_textual)
        self.fc_text1 = nn.Linear(self.embedding_size, 550)

        self.bn_text2 = nn.BatchNorm1d(self.args.max_textual)
        self.fc_text2 = nn.Linear(550, self.reduced_size)

    def forward(self, im, textual_features, sample_size, local_features, text_bboxes, local_bboxes):

        x = self.cnn_features(im)  # Size (BS x 2048 x 7 x 7)

        if self.attention:
            x, attn_mask = self.attn(x)  # Size (BS x 2048)

        x = self.attn_bn(x)
        x = x.view(x.size(0), -1)
        visual_features = F.relu(self.fc1_bn(self.fc1(x))) # Visual Features BS x 1024
        x = self.fc_w(visual_features) # BS x 300 or (embedding size)


        # SELF LEARNING?
        textual_features = self.bn_text1(textual_features)
        textual_features = F.leaky_relu(self.fc_text1(textual_features))

        textual_features = self.bn_text2(textual_features)
        textual_features = F.leaky_relu(self.fc_text2(textual_features))


        # USUAL PIPELINE
        x = torch.bmm(x.view(sample_size, 1, self.reduced_size), textual_features.permute(0, 2, 1))
        x = torch.tanh(x)
        x = F.softmax(x, dim=2)
        # Attention over textual features
        x = torch.bmm(x, textual_features)
        # Reshape visual features before fusion
        # Fuse
        if self.args.fusion != 'concat':
            x = self.fusion([x.view(sample_size, -1),visual_features])
        else:
            x = torch.cat((x[:, 0, :], visual_features), 1)
        '''
        ranking_vector = F.relu(self.fc3(self.bn3(x)))
        x = F.dropout(self.fc4(self.bn4(ranking_vector)), p=0.3, training=self.training)
        '''
        x = F.dropout(self.fc3(self.bn3(x)), p=0.3, training=self.training)

        return x, attn_mask, 0


def normalize(x):
    return x / x.norm(dim=1, keepdim=True)

def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X

class globalNet(nn.Module):
    # Network that uses global (Resnet) and local (Faster RCNN VG features)
    def __init__(self, args, num_classes, embedding_size, pretrained=True, attention=True):
        super(globalNet, self).__init__()
        self.args = args
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.pretrained = pretrained
        resnet152 = models.resnet152(pretrained)

        for name, child in resnet152.named_children():
            if name not in ['layer4']:
                # print(name + ' is frozen')
                for param in child.parameters():
                    param.requires_grad = False
            else:
                # print(name + ' is not frozen')
                for param in child.parameters():
                    param.requires_grad = True

        self.cnn_features = nn.Sequential(*list(resnet152.children())[:-2])
        #
        #
        # for param in self.cnn_features.parameters():
        #     param.requires_grad = False

        # Attention model
        self.attention = attention
        self.attn = AttentionModel()
        self.attn_bn = nn.BatchNorm2d(2048)

        # OUTPUT OF CNN BS X 2048 X 7 X 7 =  100352
        self.fc1_bn = nn.BatchNorm1d(2048 * 7 * 7)
        self.fc1 = nn.Linear(2048 * 7 * 7, 2048)

        # LOCAL FEATURES N X 36 X 2048
        self.fc2_bn = nn.BatchNorm1d(36)
        self.fc2 = nn.Linear (2048,2048)

        # FINAL LAYER
        self.fc3_bn = nn.BatchNorm1d(2*2048)
        self.fc3 = nn.Linear(2*2048, num_classes)


    def forward(self, im, textual_features, sample_size, local_features, text_bboxes, local_bboxes):
        x = self.cnn_features(im)  # Size (BS x 2048 x 7 x 7)

        if self.attention:
            x, attn_mask = self.attn(x)  # Size (BS x 2048)
        x = self.attn_bn(x)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(self.fc1_bn(x)))

        v = F.leaky_relu(self.fc2(self.fc2_bn(local_features)))
        v = torch.mean(v, dim =1)

        x = torch.cat((x, v), 1)
        x = F.dropout(self.fc3(self.fc3_bn(x)), p=0.3, training=self.training)

        return x, attn_mask


class baseGCN(nn.Module):
    # Network that uses global (Resnet) and local (Faster RCNN VG features)
    def __init__(self, args, num_classes, embedding_size, pretrained=True, attention=True):
        super(baseGCN, self).__init__()
        self.args = args
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.pretrained = pretrained
        resnet152 = models.resnet152(pretrained)

        for name, child in resnet152.named_children():
            if name not in ['layer4']:
                # print(name + ' is frozen')
                for param in child.parameters():
                    param.requires_grad = False
            else:
                # print(name + ' is not frozen')
                for param in child.parameters():
                    param.requires_grad = True

        self.cnn_features = nn.Sequential(*list(resnet152.children())[:-2])

        # GCN reasoning
        self.Rs_GCN_1 = Rs_GCN(in_channels=2048, inter_channels=2048)
        self.Rs_GCN_2 = Rs_GCN(in_channels=2048, inter_channels=2048)
        self.Rs_GCN_3 = Rs_GCN(in_channels=2048, inter_channels=2048)
        self.Rs_GCN_4 = Rs_GCN(in_channels=2048, inter_channels=2048)


        # Attention model
        self.attention = attention
        self.attn = AttentionModel()
        self.attn_bn = nn.BatchNorm2d(2048)

        # OUTPUT OF CNN BS X 2048 X 7 X 7 =  100352
        self.fc1_bn = nn.BatchNorm1d(2048 * 7 * 7)
        self.fc1 = nn.Linear(2048 * 7 * 7, 2048)

        # LOCAL FEATURES N X 36 X 2048
        self.fc2_bn = nn.BatchNorm1d(self.args.max_visual)
        self.fc2 = nn.Linear (2048,2048)


        # FINAL LAYER
        self.fc3_bn = nn.BatchNorm1d(2*2048)
        self.fc3 = nn.Linear(2*2048, num_classes)


    def forward(self, im, textual_features, sample_size, local_features, text_bboxes, local_bboxes):
        x = self.cnn_features(im)  # Size (BS x 2048 x 7 x 7)

        if self.attention:
            x, attn_mask = self.attn(x)  # Size (BS x 2048)
        x = self.attn_bn(x)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(self.fc1_bn(x)))

        v = F.leaky_relu(self.fc2(self.fc2_bn(local_features)))


        # GCN reasoning
        # -> B,D,N
        GCN_img_emd = v.permute(0, 2, 1)
        GCN_img_emd = self.Rs_GCN_1(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_2(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_3(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_4(GCN_img_emd)
        # -> B,N,D
        GCN_img_emd = GCN_img_emd.permute(0, 2, 1)

        GCN_img_emd = l2norm(GCN_img_emd)

        GCN_img_emd = torch.mean(GCN_img_emd, dim =1)

        x = torch.cat((x, GCN_img_emd), 1)
        x = F.dropout(self.fc3(self.fc3_bn(x)), p=0.3, training=self.training)

        return x, attn_mask


class textGCN(nn.Module):
    # Network that uses global (Resnet) and local (Faster RCNN VG features)
    def __init__(self, args, num_classes, embedding_size, pretrained=True, attention=True):
        super(textGCN, self).__init__()
        self.args = args
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.pretrained = pretrained
        resnet152 = models.resnet152(pretrained)

        for name, child in resnet152.named_children():
            if name not in ['layer4']:
                # print(name + ' is frozen')
                for param in child.parameters():
                    param.requires_grad = False
            else:
                # print(name + ' is not frozen')
                for param in child.parameters():
                    param.requires_grad = True

        self.cnn_features = nn.Sequential(*list(resnet152.children())[:-2])

        # GCN reasoning
        self.Rs_GCN_1 = Rs_GCN(in_channels=2048, inter_channels=2048)
        self.Rs_GCN_2 = Rs_GCN(in_channels=2048, inter_channels=2048)
        self.Rs_GCN_3 = Rs_GCN(in_channels=2048, inter_channels=2048)
        self.Rs_GCN_4 = Rs_GCN(in_channels=2048, inter_channels=2048)

        # Attention model
        self.attention = attention
        self.attn = AttentionModel()
        self.attn_bn = nn.BatchNorm2d(2048)

        # OUTPUT OF CNN BS X 2048 X 7 X 7 =  100352
        self.fc1_bn = nn.BatchNorm1d(2048 * 7 * 7)
        self.fc1 = nn.Linear(2048 * 7 * 7, 2048)

        # LOCAL FEATURES N X 36 X 2048
        self.fc2_bn = nn.BatchNorm1d(self.args.max_visual)
        self.fc2 = nn.Linear(2048, 2048)

        # Final Visual Features projection
        self.fc_visual_bn = nn.BatchNorm1d(4096)
        self.fc_visual = nn.Linear(4096, 2048)

        # TEXTUAL FEATURES N X 36 X 2048
        self.bn_text1 = nn.BatchNorm1d(self.args.max_textual)
        self.fc_text1 = nn.Linear(self.embedding_size, 1024)
        self.bn_text2 = nn.BatchNorm1d(self.args.max_textual)
        self.fc_text2 = nn.Linear(1024, 2048)

        # FINAL LAYER
        self.fc3_bn = nn.BatchNorm1d(2 * 2048)
        self.fc3 = nn.Linear(2 * 2048, num_classes)

    def forward(self, im, textual_features, sample_size, local_features, text_bboxes, local_bboxes):
        x = self.cnn_features(im)  # Size (BS x 2048 x 7 x 7)

        if self.attention:
            x, attn_mask = self.attn(x)  # Size (BS x 2048)
        x = self.attn_bn(x)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(self.fc1_bn(x)))
        # FC for LOCAL Features
        GCN_img_emd = F.leaky_relu(self.fc2(self.fc2_bn(local_features)))

        # GCN reasoning
        # -> B,D,N
        GCN_img_emd = GCN_img_emd.permute(0, 2, 1)
        GCN_img_emd = self.Rs_GCN_1(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_2(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_3(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_4(GCN_img_emd)
        # -> B,N,D
        GCN_img_emd = GCN_img_emd.permute(0, 2, 1)

        GCN_img_emd = l2norm(GCN_img_emd)

        GCN_img_emd = torch.mean(GCN_img_emd, dim=1)
        # Concatenate Global and Local visual feats
        vf = torch.cat((x, GCN_img_emd), 1)

        vf = F.leaky_relu(self.fc_visual(self.fc_visual_bn(vf)))


        # Textual Features SHAPE: N X MAX_TEXTUAL X 300 (DEFAULT EMB SIZE)
        textual_features = self.bn_text1(textual_features)
        textual_features = F.leaky_relu(self.fc_text1(textual_features))

        textual_features = self.bn_text2(textual_features)
        textual_features = F.leaky_relu(self.fc_text2(textual_features)) # SHAPE: N X MAX_TEXTUAL X 2048

        # ATTENTION USUAL PIPELINE
        x = torch.bmm(vf.view(sample_size, 1, 2048), textual_features.permute(0, 2, 1))
        x = torch.tanh(x)
        x = F.softmax(x, dim=2)
        # Attention over textual features
        x = torch.bmm(x, textual_features)
        # Reshape visual features before fusion
        # Fuse
        if self.args.fusion != 'concat':
            # x = self.fusion([x.view(sample_size, -1), visual_features])
            print('Error FUSION Not implemented')
        else:
            x = torch.cat((x[:, 0, :], vf), 1)


        x = F.dropout(self.fc3(self.fc3_bn(x)), p=0.3, training=self.training)

        return x, attn_mask

class fullGCN(nn.Module):
    # Network that uses global (Resnet) and local (Faster RCNN VG features)
    def __init__(self, args, num_classes, embedding_size, pretrained=True, attention=True):
        super(fullGCN, self).__init__()
        self.args = args
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.pretrained = pretrained
        resnet152 = models.resnet152(pretrained)

        for name, child in resnet152.named_children():
            if name not in ['layer4']:
                # print(name + ' is frozen')
                for param in child.parameters():
                    param.requires_grad = False
            else:
                # print(name + ' is not frozen')
                for param in child.parameters():
                    param.requires_grad = True

        self.cnn_features = nn.Sequential(*list(resnet152.children())[:-2])

        # GCN reasoning
        self.Rs_GCN_1 = Rs_GCN(in_channels=2048, inter_channels=2048)
        self.Rs_GCN_2 = Rs_GCN(in_channels=2048, inter_channels=2048)
        self.Rs_GCN_3 = Rs_GCN(in_channels=2048, inter_channels=2048)
        self.Rs_GCN_4 = Rs_GCN(in_channels=2048, inter_channels=2048)
        # self.Rs_GCN_5 = Rs_GCN(in_channels=2048, inter_channels=2048)
        # self.Rs_GCN_6 = Rs_GCN(in_channels=2048, inter_channels=2048)
        # self.Rs_GCN_7 = Rs_GCN(in_channels=2048, inter_channels=2048)

        # Attention model
        self.attention = attention
        self.attn = AttentionModel()
        self.attn_bn = nn.BatchNorm2d(2048)

        # OUTPUT OF CNN BS X 2048 X 7 X 7 =  100352
        self.fc1_bn = nn.BatchNorm1d(2048 * 7 * 7)
        self.fc1 = nn.Linear(2048 * 7 * 7, 2048)

        # LOCAL FEATURES N X 36 X 2048
        self.fc2_bn = nn.BatchNorm1d(self.args.max_visual)
        self.fc2 = nn.Linear(2048, 2048)

        # TEXTUAL FEATURES N X 36 X 2048
        self.bn_text1 = nn.BatchNorm1d(self.args.max_textual)
        self.fc_text1 = nn.Linear(self.embedding_size, 1024)
        self.bn_text2 = nn.BatchNorm1d(self.args.max_textual)
        self.fc_text2 = nn.Linear(1024, 2048)

        # FINAL LAYER
        self.fc3_bn = nn.BatchNorm1d(2 * 2048)
        self.fc3 = nn.Linear(2 * 2048, num_classes)

    def forward(self, im, textual_features, sample_size, local_features, text_bboxes, local_bboxes):
        x = self.cnn_features(im)  # Size (BS x 2048 x 7 x 7)

        if self.attention:
            x, attn_mask = self.attn(x)  # Size (BS x 2048)
        x = self.attn_bn(x)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(self.fc1_bn(x)))
        # FC for LOCAL Features
        GCN_img_emd = F.leaky_relu(self.fc2(self.fc2_bn(local_features)))

        # Textual Features SHAPE: N X MAX_TEXTUAL X 300 (DEFAULT EMB SIZE)
        textual_features = self.bn_text1(textual_features)
        textual_features = F.leaky_relu(self.fc_text1(textual_features))

        textual_features = self.bn_text2(textual_features)
        textual_features = F.leaky_relu(self.fc_text2(textual_features))  # SHAPE: N X MAX_TEXTUAL X 2048


        # GCN reasoning LOCAL VISUAL + TEXTUAL FEATURES
        GCN_img_emd = torch.cat((GCN_img_emd, textual_features), dim=1)
        # -> B,D,N
        GCN_img_emd = GCN_img_emd.permute(0, 2, 1)
        GCN_img_emd = self.Rs_GCN_1(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_2(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_3(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_4(GCN_img_emd)
        # GCN_img_emd = self.Rs_GCN_5(GCN_img_emd)
        # GCN_img_emd = self.Rs_GCN_6(GCN_img_emd)
        # GCN_img_emd = self.Rs_GCN_7(GCN_img_emd)
        # -> B,N,D
        GCN_img_emd = GCN_img_emd.permute(0, 2, 1)

        GCN_img_emd = l2norm(GCN_img_emd)

        GCN_img_emd = torch.mean(GCN_img_emd, dim=1)
        # Concatenate Global and Local visual feats
        x = torch.cat((x, GCN_img_emd), dim=1)

        x = F.dropout(self.fc3(self.fc3_bn(x)), p=0.3, training=self.training)

        return x, attn_mask


class dualGCN(nn.Module):
    # Projection of fasttext into FasterRCNN space (No initial FC)
    def __init__(self, args, num_classes, embedding_size, pretrained=True, attention=True):
        super(dualGCN, self).__init__()
        self.args = args
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.pretrained = pretrained
        resnet152 = models.resnet152(pretrained)

        for name, child in resnet152.named_children():
            if name not in ['layer4']:
                # print(name + ' is frozen')
                for param in child.parameters():
                    param.requires_grad = False
            else:
                # print(name + ' is not frozen')
                for param in child.parameters():
                    param.requires_grad = True

        self.cnn_features = nn.Sequential(*list(resnet152.children())[:-2])

        # GCN reasoning
        self.Rs_GCN_1 = Rs_GCN(in_channels=2048, inter_channels=2048)
        self.Rs_GCN_2 = Rs_GCN(in_channels=2048, inter_channels=2048)
        self.Rs_GCN_3 = Rs_GCN(in_channels=2048, inter_channels=2048)
        self.Rs_GCN_4 = Rs_GCN(in_channels=2048, inter_channels=2048)

        self.Rs_GCN_5 = Rs_GCN(in_channels=300, inter_channels=300)
        self.Rs_GCN_6 = Rs_GCN(in_channels=300, inter_channels=300)
        self.Rs_GCN_7 = Rs_GCN(in_channels=300, inter_channels=300)
        self.Rs_GCN_8 = Rs_GCN(in_channels=300, inter_channels=300)

        # Attention model
        self.attention = attention
        self.attn = AttentionModel()
        self.attn_bn = nn.BatchNorm2d(2048)

        # OUTPUT OF CNN BS X 2048 X 7 X 7 =  100352
        self.fc1_bn = nn.BatchNorm1d(2048 * 7 * 7)
        self.fc1 = nn.Linear(2048 * 7 * 7, 2048)

        # LOCAL FEATURES N X 36 X 2048
        self.fc2_bn = nn.BatchNorm1d(self.args.max_visual)
        self.fc2 = nn.Linear(2048, 300)

        # TEXTUAL FEATURES TO N X 36 X 2048
        self.bn_text1 = nn.BatchNorm1d(self.args.max_textual)
        self.fc_text1 = nn.Linear(self.embedding_size, 2048)

        # FC TO FIT SPACES
        self.bn_output_gcn = nn.BatchNorm1d(300)
        self.fc_output_gcn = nn.Linear(300, 2048)


        # FINAL FUSION BEFORE CLASSIFICATION
        self.final_bn = nn.BatchNorm1d(3*2048)
        self.final_fc = nn.Linear(3*2048, num_classes)


    def forward(self, im, textual_features, sample_size, local_features, text_bboxes, local_bboxes):
        x = self.cnn_features(im)  # Size (BS x 2048 x 7 x 7)

        if self.attention:
            x, attn_mask = self.attn(x)  # Size (BS x 2048)
        x = self.attn_bn(x)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(self.fc1_bn(x)))

        # Textual Features SHAPE: N X MAX_TEXTUAL X 300 (DEFAULT EMB SIZE) TO 2048
        textual_features_2048 = self.bn_text1(textual_features)
        textual_features_2048 = F.leaky_relu(self.fc_text1(textual_features_2048))

        # FC for LOCAL Features TO 300
        local_features_300 = F.leaky_relu(self.fc2(self.fc2_bn(local_features)))

        # GCN reasoning LOCAL VISUAL + TEXTUAL FEATURES_2048
        GCN_img_emd = torch.cat((local_features, textual_features_2048), dim=1)
        # -> B,D,N
        GCN_img_emd = GCN_img_emd.permute(0, 2, 1)
        GCN_img_emd = self.Rs_GCN_1(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_2(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_3(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_4(GCN_img_emd)
        # -> B,N,D
        GCN_img_emd = GCN_img_emd.permute(0, 2, 1)
        GCN_img_emd = l2norm(GCN_img_emd)
        GCN_img_emd = torch.mean(GCN_img_emd, dim=1)

        # GCN reasoning TEXTUAL FEATURES + LOCAL VISUAL_300
        GCN_text_emd = torch.cat((textual_features, local_features_300), dim=1)
        # -> B,D,N
        GCN_text_emd = GCN_text_emd.permute(0, 2, 1)
        GCN_text_emd = self.Rs_GCN_5(GCN_text_emd)
        GCN_text_emd = self.Rs_GCN_6(GCN_text_emd)
        GCN_text_emd = self.Rs_GCN_7(GCN_text_emd)
        GCN_text_emd = self.Rs_GCN_8(GCN_text_emd)
        # -> B,N,D
        GCN_text_emd = GCN_text_emd.permute(0, 2, 1)
        GCN_text_emd = l2norm(GCN_text_emd)
        GCN_text_emd = torch.mean(GCN_text_emd, dim=1)
        # PROJECT FINAL FEATURES (OUTPUT FROM GCN) TO A SPACE DIM: 1 X 2048
        GCN_text_emd = F.leaky_relu(self.fc_output_gcn(self.bn_output_gcn(GCN_text_emd)))

        x = torch.cat((x, GCN_img_emd, GCN_text_emd), dim=1)

        x = F.dropout(self.final_fc(self.final_bn(x)), p=0.3, training=self.training)

        return x, attn_mask


class fullGCN_attn(nn.Module):
    # Network that uses global (Resnet) and local (Faster RCNN VG features)
    def __init__(self, args, num_classes, embedding_size, pretrained=True, attention=True):
        super(fullGCN_attn, self).__init__()
        self.args = args
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.pretrained = pretrained
        resnet152 = models.resnet152(pretrained)

        for name, child in resnet152.named_children():
            if name not in ['layer4']:
                # print(name + ' is frozen')
                for param in child.parameters():
                    param.requires_grad = False
            else:
                # print(name + ' is not frozen')
                for param in child.parameters():
                    param.requires_grad = True

        self.cnn_features = nn.Sequential(*list(resnet152.children())[:-2])

        # GCN reasoning
        self.Rs_GCN_1 = Rs_GCN(in_channels=2048, inter_channels=2048)
        self.Rs_GCN_2 = Rs_GCN(in_channels=2048, inter_channels=2048)
        self.Rs_GCN_3 = Rs_GCN(in_channels=2048, inter_channels=2048)
        self.Rs_GCN_4 = Rs_GCN(in_channels=2048, inter_channels=2048)
        self.Rs_GCN_5 = Rs_GCN(in_channels=2048, inter_channels=2048)
        self.Rs_GCN_6 = Rs_GCN(in_channels=2048, inter_channels=2048)
        self.Rs_GCN_7 = Rs_GCN(in_channels=2048, inter_channels=2048)
        self.Rs_GCN_8 = Rs_GCN(in_channels=2048, inter_channels=2048)

        # Attention model
        self.attention = attention
        self.attn = AttentionModel()
        self.attn_bn = nn.BatchNorm2d(2048)

        # OUTPUT OF CNN BS X 2048 X 7 X 7 =  100352
        self.fc1_bn = nn.BatchNorm1d(2048 * 7 * 7)
        self.fc1 = nn.Linear(2048 * 7 * 7, 2048)

        # # LOCAL FEATURES N X 36 X 2048
        self.fc2_bn = nn.BatchNorm1d(self.args.max_visual)
        self.fc2 = nn.Linear(2048, 2048)

        # TEXTUAL FEATURES N X 36 X 2048
        self.bn_text1 = nn.BatchNorm1d(self.args.max_textual)
        self.fc_text1 = nn.Linear(self.embedding_size, 1024)
        self.bn_text2 = nn.BatchNorm1d(self.args.max_textual)
        self.fc_text2 = nn.Linear(1024, 2048)

        # PROJECTION LAYER
        if self.args.projection_layer == 'gru':
            # GRU VISUAL+TEXTUAL UNDERSTANDING
            self.gru_local = nn.GRU(2048, 2048, 1, batch_first=True)

        elif self.args.projection_layer == 'fc' or self.args.projection_layer == 'attention':
        # FULLY CONNECTED OR ATTENTION
            self.bn_projection = nn.BatchNorm1d((self.args.max_textual + self.args.max_visual) * 2048)
            self.fc_projection = nn.Linear( (self.args.max_textual + self.args.max_visual) * 2048, 2048)

        # FINAL FUSION BEFORE CLASSIFICATION
        if self.args.fusion == 'block':
            self.fusion = Block([2048, 2048], 2048, mm_dim= self.args.mmdim)
            self.final_bn = nn.BatchNorm1d(2048)
            self.final_fc = nn.Linear(2048, num_classes)

        elif self.args.fusion == 'mlb':
            self.fusion = MLB([2048, 2048], 2048, mm_dim= self.args.mmdim)
            self.final_bn = nn.BatchNorm1d(2048)
            self.final_fc = nn.Linear(2048, num_classes)

        elif self.args.fusion == 'attention' or self.args.fusion == 'dot':
            # ATTENTION or DOT PRODUCT AS FUSION
            self.final_bn = nn.BatchNorm1d(2048)
            self.final_fc = nn.Linear(2048, num_classes)

        elif self.args.fusion == 'concat':
            # CONCATENATION AS FUSION
            self.final_bn = nn.BatchNorm1d(2*2048)
            self.final_fc = nn.Linear(2*2048, num_classes)
        else:
            print("Error: Last Layer Fusion selected not implemented")

    def forward(self, im, textual_features, sample_size, local_features,text_bboxes, local_bboxes):
        x = self.cnn_features(im)  # Size (BS x 2048 x 7 x 7)

        if self.attention:
            x, attn_mask = self.attn(x)  # Size (BS x 2048)
        x = self.attn_bn(x)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(self.fc1_bn(x)))

        # Textual Features SHAPE: N X MAX_TEXTUAL X 300 (DEFAULT EMB SIZE)
        textual_features = self.bn_text1(textual_features)
        textual_features = F.leaky_relu(self.fc_text1(textual_features))

        textual_features = self.bn_text2(textual_features)
        textual_features = F.leaky_relu(self.fc_text2(textual_features))  # SHAPE: N X MAX_TEXTUAL X 2048

        # FC for LOCAL Features
        GCN_img_emd = F.leaky_relu(self.fc2(self.fc2_bn(local_features)))
        GCN_img_emd = torch.cat((local_features, textual_features), dim=1)

        # GCN reasoning LOCAL VISUAL + TEXTUAL FEATURES
        # GCN_img_emd = torch.cat((GCN_img_emd, textual_features), dim=1)
        # -> B,D,N
        GCN_img_emd = GCN_img_emd.permute(0, 2, 1)
        GCN_img_emd = self.Rs_GCN_1(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_2(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_3(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_4(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_5(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_6(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_7(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_8(GCN_img_emd)
        # -> B,N,D
        GCN_img_emd = GCN_img_emd.permute(0, 2, 1)

        GCN_img_emd = l2norm(GCN_img_emd)

        # PROJECT FINAL VISUAL FEATURES (OUTPUT FROM GCN) TO A SPACE DIM: 1 X 2048

        if self.args.projection_layer == 'gru':
            # GRU VISUAL+TEXTUAL UNDERSTANDING
            rnn_img, hidden_state = self.gru_local(GCN_img_emd)
            GCN_img_emd = hidden_state[0] # Hidden state of last time step of i layer (in this case only one layer)

        elif self.args.projection_layer == 'fc':
            # FULLY CONNECTED... NOT ENOUGH GPU RAM
            GCN_img_emd = torch.reshape(GCN_img_emd,(sample_size, -1))
            GCN_img_emd= F.leaky_relu(self.fc_projection(self.bn_projection(GCN_img_emd)))

        elif self.args.projection_layer == 'attention':
            # ATTENTION
            visual_atnn = torch.bmm(x.reshape(sample_size,1,2048), GCN_img_emd.permute(0,2,1))
            visual_atnn = torch.tanh(visual_atnn)
            visual_atnn = F.softmax(visual_atnn, dim=1)
            # Attention over Global Visual Features
            GCN_img_emd = torch.bmm(visual_atnn, GCN_img_emd).reshape(sample_size, -1)

        elif self.args.projection_layer == 'mean':
            # MEAN VECTOR:
            GCN_img_emd = torch.mean(GCN_img_emd, dim=1)
        else:
            print("Forward pass Error in Projection Layer")

        # FINAL CONSTRUCTION OF VECTOR BEFORE CLASSIFICATION
        if self.args.fusion == 'attention':
            # ATTENTION AS FUSION
            visual_atnn = x * GCN_img_emd  # Elem-wise mult - Shape: N x 2048
            visual_atnn = torch.tanh(visual_atnn)
            visual_atnn = F.softmax(visual_atnn, dim=1)
            # Attention over Global Visual Features
            x = visual_atnn * GCN_img_emd

        elif self.args.fusion == 'mlb' or self.args.fusion =='block':
            x = self.fusion([GCN_img_emd.view(sample_size, -1), x])

        elif self.args.fusion == 'dot':
            # DOT PRODUCT AS FUSION
            x = x * GCN_img_emd  # Elem-wise mult - Shape: N x 2048
        elif self.args.fusion == 'concat':
            # CONCAT AS FUSION
            # Concatenate Global and Local visual feats
            x = torch.cat((x, GCN_img_emd), dim=1)
        else:
            print('Error on forward pass fusion')

        x = F.dropout(self.final_fc(self.final_bn(x)), p=0.3, training=self.training)

        return x, attn_mask

class fullGCN_bboxes(nn.Module):
    # Network that uses global (Resnet) and local (Faster RCNN VG features)
    def __init__(self, args, num_classes, embedding_size, pretrained=True, attention=True):
        super(fullGCN_bboxes, self).__init__()
        self.args = args
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.pretrained = pretrained
        resnet152 = models.resnet152(pretrained)

        for name, child in resnet152.named_children():
            if name not in ['layer4']:
                # print(name + ' is frozen')
                for param in child.parameters():
                    param.requires_grad = False
            else:
                # print(name + ' is not frozen')
                for param in child.parameters():
                    param.requires_grad = True

        self.cnn_features = nn.Sequential(*list(resnet152.children())[:-2])

        # GCN reasoning
        gcn_dim_size = 2048
        self.Rs_GCN_1 = Rs_GCN(in_channels=gcn_dim_size, inter_channels=gcn_dim_size)
        self.Rs_GCN_2 = Rs_GCN(in_channels=gcn_dim_size, inter_channels=gcn_dim_size)
        self.Rs_GCN_3 = Rs_GCN(in_channels=gcn_dim_size, inter_channels=gcn_dim_size)
        self.Rs_GCN_4 = Rs_GCN(in_channels=gcn_dim_size, inter_channels=gcn_dim_size)
        self.Rs_GCN_5 = Rs_GCN(in_channels=gcn_dim_size, inter_channels=gcn_dim_size)
        self.Rs_GCN_6 = Rs_GCN(in_channels=gcn_dim_size, inter_channels=gcn_dim_size)
        self.Rs_GCN_7 = Rs_GCN(in_channels=gcn_dim_size, inter_channels=gcn_dim_size)
        self.Rs_GCN_8 = Rs_GCN(in_channels=gcn_dim_size, inter_channels=gcn_dim_size)

        # Attention model
        self.attention = attention
        self.attn = AttentionModel()
        self.attn_bn = nn.BatchNorm2d(2048)

        # OUTPUT OF CNN BS X 2048 X 7 X 7 =  100352
        self.fc1_bn = nn.BatchNorm1d(2048 * 7 * 7)
        self.fc1 = nn.Linear(2048 * 7 * 7, 2048)

        # LOCAL FEATURES N X 36 X 2048
        self.fc2_bn = nn.BatchNorm1d(self.args.max_visual)
        self.fc2 = nn.Linear(2048, 1920)

        # TEXTUAL FEATURES N X 36 X 2048
        self.bn_text1 = nn.BatchNorm1d(self.args.max_textual)
        self.fc_text1 = nn.Linear(self.embedding_size, 1024)
        self.bn_text2 = nn.BatchNorm1d(self.args.max_textual)
        self.fc_text2 = nn.Linear(1024, 1920)

        # BBOX POSITIONAL ENCODING OF LOCAL FEATURES AND TEXT
        self.bn_encod_bboxes = nn.BatchNorm1d(self.args.max_visual + self.args.max_textual)
        self.fc_encod_bboxes = nn.Linear(4, 128)

        # # FC OUTPUT from GCN
        # self.bn_out_gcn = nn.BatchNorm1d(self.args.max_visual + self.args.max_textual)
        # self.fc_out_gcn = nn.Linear(gcn_dim_size, 2048)

        # PROJECTION LAYER
        if self.args.projection_layer == 'gru':
            # GRU VISUAL+TEXTUAL UNDERSTANDING
            self.gru_local = nn.GRU(2048, 2048, 1, batch_first=True)

        elif self.args.projection_layer == 'fc' or self.args.projection_layer == 'attention':
        # FULLY CONNECTED OR ATTENTION
            self.bn_projection = nn.BatchNorm1d((self.args.max_textual + self.args.max_visual) * 2048)
            self.fc_projection = nn.Linear( (self.args.max_textual + self.args.max_visual) * 2048, 2048)

        # FINAL FUSION BEFORE CLASSIFICATION
        if self.args.fusion == 'block':
            self.fusion = Block([2048, 2048], 2048, mm_dim= self.args.mmdim)
            self.final_bn = nn.BatchNorm1d(2048)
            self.final_fc = nn.Linear(2048, num_classes)

        elif self.args.fusion == 'mlb':
            self.fusion = MLB([2048, 2048], 2048, mm_dim= self.args.mmdim)
            self.final_bn = nn.BatchNorm1d(2048)
            self.final_fc = nn.Linear(2048, num_classes)

        elif self.args.fusion == 'attention' or self.args.fusion == 'dot':
            # ATTENTION or DOT PRODUCT AS FUSION
            self.final_bn = nn.BatchNorm1d(2048)
            self.final_fc = nn.Linear(2048, num_classes)

        elif self.args.fusion == 'concat':
            # CONCATENATION AS FUSION
            self.final_bn = nn.BatchNorm1d(2048 * 2)
            self.final_fc = nn.Linear(2048 * 2, num_classes)
        else:
            print("Error: Last Layer Fusion selected not implemented")

    def forward(self, im, textual_features, sample_size, local_features, text_bboxes, local_bboxes):
        x = self.cnn_features(im)  # Size (BS x 2048 x 7 x 7)

        if self.attention:
            x, attn_mask = self.attn(x)  # Size (BS x 2048)
        x = self.attn_bn(x)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(self.fc1_bn(x)))

        # Textual Features SHAPE: N X MAX_TEXTUAL X 300 (DEFAULT EMB SIZE)
        textual_features = self.bn_text1(textual_features)
        textual_features = F.leaky_relu(self.fc_text1(textual_features))

        textual_features = self.bn_text2(textual_features)
        textual_features = F.leaky_relu(self.fc_text2(textual_features))  # SHAPE: N X MAX_TEXTUAL X 1920

        # FC for LOCAL Features
        GCN_img_emd = F.leaky_relu(self.fc2(self.fc2_bn(local_features)))

        # FC for Visual and Textual BBOXES
        bboxes_feats = torch.cat((local_bboxes, text_bboxes), dim=1)
        bboxes_feats = self.bn_encod_bboxes(bboxes_feats)
        bboxes_feats = F.leaky_relu(self.fc_encod_bboxes(bboxes_feats))

        # CONCAT LOCAL FEATURES AND TEXTUAL FEATURES
        GCN_img_emd = torch.cat((GCN_img_emd, textual_features), dim=1)

        # CONCAT EACH BBOX AT THE LAST COLUMN OF TEXTUAL AND VISUAL FEATURES
        GCN_img_emd = torch.cat((GCN_img_emd, bboxes_feats), dim=2)

        # GCN reasoning LOCAL VISUAL + TEXTUAL FEATURES
        # GCN_img_emd = torch.cat((GCN_img_emd, textual_features), dim=1)
        # -> B,D,N
        GCN_img_emd = GCN_img_emd.permute(0, 2, 1)
        GCN_img_emd, __ = self.Rs_GCN_1(GCN_img_emd)
        GCN_img_emd, __ = self.Rs_GCN_2(GCN_img_emd)
        GCN_img_emd, __ = self.Rs_GCN_3(GCN_img_emd)
        GCN_img_emd, __ = self.Rs_GCN_4(GCN_img_emd)
        GCN_img_emd, __ = self.Rs_GCN_5(GCN_img_emd)
        GCN_img_emd, __ = self.Rs_GCN_6(GCN_img_emd)
        GCN_img_emd, __ = self.Rs_GCN_7(GCN_img_emd)
        GCN_img_emd, affinity_matrix = self.Rs_GCN_8(GCN_img_emd)
        # -> B,N,D
        GCN_img_emd = GCN_img_emd.permute(0, 2, 1)

        GCN_img_emd = l2norm(GCN_img_emd)

        # GCN_img_emd = self.bn_out_gcn(GCN_img_emd)
        # GCN_img_emd = F.leaky_relu(self.fc_out_gcn(GCN_img_emd))

        # PROJECT FINAL VISUAL FEATURES (OUTPUT FROM GCN) TO A SPACE DIM: 1 X 204

        if self.args.projection_layer == 'gru':
            # GRU VISUAL+TEXTUAL UNDERSTANDING
            rnn_img, hidden_state = self.gru_local(GCN_img_emd)
            GCN_img_emd = hidden_state[0] # Hidden state of last time step of i layer (in this case only one layer)

        elif self.args.projection_layer == 'fc':
            # FULLY CONNECTED... NOT ENOUGH GPU RAM
            GCN_img_emd = torch.reshape(GCN_img_emd,(sample_size, -1))
            GCN_img_emd= F.leaky_relu(self.fc_projection(self.bn_projection(GCN_img_emd)))

        elif self.args.projection_layer == 'attention':
            # ATTENTION
            visual_atnn = torch.bmm(x.reshape(sample_size,1,2048), GCN_img_emd.permute(0,2,1))
            # pdb.set_trace()
            # visual_atnn = torch.tanh(visual_atnn)
            visual_atnn = F.leaky_relu(visual_atnn)
            visual_atnn = F.softmax(visual_atnn, dim=2)
            # Attention over Global Visual Features
            GCN_img_emd = torch.bmm(visual_atnn, GCN_img_emd).reshape(sample_size, -1)

        elif self.args.projection_layer == 'mean':
            # MEAN VECTOR:
            GCN_img_emd = torch.mean(GCN_img_emd, dim=1)
        else:
            print("Forward pass Error in Projection Layer")

        # FINAL CONSTRUCTION OF VECTOR BEFORE CLASSIFICATION
        if self.args.fusion == 'attention':
            # ATTENTION AS FUSION
            visual_atnn = x * GCN_img_emd  # Elem-wise mult - Shape: N x 2048
            visual_atnn = torch.tanh(visual_atnn)
            visual_atnn = F.softmax(visual_atnn, dim=1)
            # Attention over Global Visual Features
            x = visual_atnn * GCN_img_emd

        elif self.args.fusion == 'mlb' or self.args.fusion =='block':
            x = self.fusion([GCN_img_emd.view(sample_size, -1), x])

        elif self.args.fusion == 'dot':
            # DOT PRODUCT AS FUSION
            x = x * GCN_img_emd  # Elem-wise mult - Shape: N x 2048
        elif self.args.fusion == 'concat':
            # CONCAT AS FUSION
            # Concatenate Global and Local visual feats
            x = torch.cat((x, GCN_img_emd), dim=1)
        else:
            print('Error on forward pass fusion')

        x = F.dropout(self.final_fc(self.final_bn(x)), p=0.3, training=self.training)

        return x, attn_mask, affinity_matrix

