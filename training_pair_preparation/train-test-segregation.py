import os
from collections import defaultdict
import spacy
from classes import Line, File, Pair
import random

def train_test_split(dirname, inp_json_path):
    train_path = '../key_ori/training_keys'
    test_path = '../key_ori/testing_keys'
    if not os.path.exists(train_path):
        os.makedirs(train_path)

    if not os.path.exists(test_path):
        os.makedirs(test_path)

    train_json_path = '../data/training_data'
    test_json_path = '../data/testing_data'
    if not os.path.exists(train_json_path):
        os.makedirs(train_json_path)

    if not os.path.exists(test_json_path):
        os.makedirs(test_json_path)

    for file in os.listdir(dirname):

        prob = random.uniform(0, 1)
        jsonfile = file.replace('key','inputs.json')
        if prob < 0.8:
            newfile = os.path.join(train_path, file)
            new_json_file = os.path.join(train_json_path,jsonfile)
        else:
            newfile = os.path.join(test_path, file)
            new_json_file = os.path.join(test_json_path,jsonfile)

        oldfile = os.path.join(dirname, file)
        old_json_file = os.path.join(inp_json_path, jsonfile)
        os.rename(oldfile, newfile)
        os.rename(old_json_file, new_json_file)
        print('{} and {}'.format(file, jsonfile))
train_test_split('../key_ori/keys','../data/Inputs')
