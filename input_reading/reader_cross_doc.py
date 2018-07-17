import json
import os
from collections import defaultdict
from .classes import Pair
import random

class Data_Reader:
    def __init__(self, dirname, labelfile , augmentation):
        self.fname_pair = defaultdict()
        self.dirname =dirname
        self.labelfile = labelfile
        self.aug = augmentation
        self.read_yyy()

        self.list_of_pairs =self.read_events_pair(self.aug)

    def read_yyy(self):
        data_list = list()
        fname_list = list()
        for jasonfile in os.listdir(self.dirname):
            #print('reading {}'.format(jasonfile))
            fname_list.append(jasonfile)
            jasonfile = os.path.join(self.dirname, jasonfile)
            with open(jasonfile) as json_data:
                data_list.append(json.load(json_data))

        #fname_pair = defaultdict()

        for f, data in zip(fname_list, data_list):
            no_of_node = len(data)
            list_of_pairs = list()

            for i in range(no_of_node):
                ev1 = data[i]
                key = f.replace('.inputs.json','')+' '+ ev1['event']['mentionid']
                #print(key)
                self.fname_pair[key] = ev1
        #return fname_pair


    def read_events_pair(self,augmentation =0):
        file = self.labelfile
        list_line = list()
        list_pairs = list()
        one_count =0
        zero_count=0
        aug_zero_count =0
        aug_one_count =0
        with open(file) as f:
            lines = f.readlines()
        print(len(lines))
        for i in range(len(lines)):
            line =  lines[i]
            tokens = line.split()
            fname1 = str(tokens[0])
            ev1 = str(tokens[1])
            ev2 = str(tokens[2])
            fname2 = str(tokens[3])
            same = int(tokens[4])

            if same >0:
                one_count+=1
            else:
                zero_count+=1
            key1 = fname1 + ' ' + ev1
            try:
                ev1 = self.fname_pair[key1]
            except KeyError:
                print('KeyError::{} is not present'.format(key1))
                raise

            key2 = fname2 + ' ' + ev2
            try:
                ev2 = self.fname_pair[key2]
            except KeyError:
                print('KeyError::{} is not present'.format(key2))
                raise
            newP = Pair(ev1,ev2,same)
            list_pairs.append(newP)

            if augmentation ==1:
            # data augmentation with associativity

                prob = random.uniform(0, 1)
                if prob > 0.6 and same==1:
                    newp = Pair(ev2,ev1,same)
                    aug_one_count = aug_one_count+1
                    list_pairs.append(newp)
                elif prob >0.9 and same==0:
                    newp = Pair(ev2,ev1,same)
                    aug_zero_count = aug_zero_count +1
                    list_pairs.append(newp)
            #data augmentation with reflex
                prob = random.uniform(0, 1)
                if prob > 0.9:
                    prob = random.uniform(0, 1)
                    if prob > 0.5:
                        newp = Pair(ev1,ev1,1)
                        aug_one_count = aug_one_count+1
                    else:
                        newp = Pair(ev2,ev2,1)
                        aug_zero_count = aug_zero_count +1
                    list_pairs.append(newp)
        print('oneC={}, zeroC={}, augOne={}, augZero={}'.format(one_count, zero_count, aug_one_count, aug_zero_count))
        return list_pairs
