import os
import json
'''
Generates three json files with annotations.
The annotations contains test - train splits and classes.
Dictionary of Dicts and each key under train or test contains the class of the image, NOT the Index! 

'''
# CLASSES
classes = []
with open('/SSD/Datasets/Context/classes.txt') as fp:
    lines = fp.readlines()
    fp.close()
for line in lines:
    classes.append(line.strip().replace('\n', ''))


# DEFINE SPLIT
splits = ['0','1','2']
for split in splits:


    # CREATE A DICT OF ALL TRAIN AND TEST IMAGES
    train_dict = {}
    test_dict = {}

    # TRAIN
    with open('/SSD/Datasets/Context/data/ImageSets/' + split + '/train.txt', 'r') as fp:
        lines = fp.readlines()
        fp.close()

    for line in lines:
        train_dict[line.strip() + '.jpg'] = 'x'

    files = os.listdir('/SSD/Datasets/Context/data/ImageSets/'+split+'/')
    for file in files:
        if file == 'all.txt' or file == 'train.txt' or file == 'test.txt': continue

        if file.split('_')[1] == 'train.txt':
            file_class = file.split('_')[0]
            with open('/SSD/Datasets/Context/data/ImageSets/' + split + '/' + file, 'r') as fp:
                lines = fp.readlines()
                fp.close()
            for line in lines:
                image, flag = line.strip().split('\t')
                if flag == '1':
                    train_dict[image + '.jpg'] = file_class

    # TEST
    lines = []
    with open('/SSD/Datasets/Context/data/ImageSets/' + split + '/test.txt', 'r') as fp:
        lines = fp.readlines()
        fp.close()
    for line in lines:
        test_dict[line.strip() + '.jpg'] = 'x'

    files = os.listdir('/SSD/Datasets/Context/data/ImageSets/'+split+'/')
    for file in files:
        if file == 'all.txt' or file == 'train.txt' or file == 'test.txt': continue

        if file.split('_')[1] == 'test.txt':
            file_class = file.split('_')[0]
            with open('/SSD/Datasets/Context/data/ImageSets/' + split + '/' + file, 'r') as fp:
                lines = fp.readlines()
                fp.close()
            for line in lines:
                image, flag = line.strip().split('\t')
                if flag == '1':
                    test_dict[image + '.jpg'] = file_class

    # # FINAL DICTIONARY TO BE SAVED
    dictionary = {'test':test_dict, 'train':train_dict, 'classes': classes}


    with open ('/SSD/Datasets/Context/data/split_'+ split +'.json','w') as fp:
        json.dump(dictionary, fp)

print ('JSON DICTIONARY COMPLETED..')