import os
import json
'''
Generates three json files with annotations.
The annotations contains test - train splits and classes.
Dictionary of Dicts and each key under train or test contains the class of the image, NOT the Index! 

'''
# CLASSES
classes = []
with open('/SSD/Datasets/Drink_Bottle/classes.txt') as fp:
    lines = fp.readlines()
    fp.close()
for line in lines:
    classes.append(line.strip().replace('\n', ''))


# DEFINE SPLIT
splits = ['1','2','3']
for split in splits:


    # CREATE A DICT OF ALL TRAIN AND TEST IMAGES
    train_dict = {}
    test_dict = {}

    # TRAIN
    with open('/SSD/Datasets/Drink_Bottle/split' + split + '/train.txt', 'r') as fp:
        lines = fp.readlines()
        fp.close()
    for line in lines:
        image_file, file_class = line.strip().split(' ')
        train_dict[image_file] = file_class

    # TEST
    lines = []
    with open('/SSD/Datasets/Drink_Bottle/split' + split + '/test.txt', 'r') as fp:
        lines = fp.readlines()
        fp.close()
    for line in lines:
        image_file, file_class = line.strip().split(' ')
        test_dict[image_file] = file_class

    # # FINAL DICTIONARY TO BE SAVED
    dictionary = {'test':test_dict, 'train':train_dict, 'classes': classes}


    with open ('/SSD/Datasets/Drink_Bottle/split_'+ str(int(split)-1) +'.json','w') as fp:

        json.dump(dictionary, fp)

print ('JSON DICTIONARY COMPLETED..')