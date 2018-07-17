import json
import os
from collections import defaultdict
from .classes import Pair

class Data:
    def read_jsons(self, dirname):
        data_list = list()
        fname_list = list()
        for jasonfile in os.listdir(dirname):
            print('reading {}'.format(jasonfile))
            fname_list.append(jasonfile)
            jasonfile = os.path.join(dirname, jasonfile)
            with open(jasonfile) as json_data:
                data_list.append(json.load(json_data))

        fname_pair = defaultdict()

        for f, data in zip(fname_list, data_list):
            no_of_node = len(data)
            list_of_pairs = list()
            if no_of_node >1:
                for i in range(no_of_node-1):
                    for j in range(i+1, no_of_node):
                        ev1 = data[i]
                        ev2 = data[j]
                        newp = Pair(ev1,ev2,-1)
                        list_of_pairs.append(newp)
            elif no_of_node ==1:
                ev1 = data[0]
                ev2 = data[0]
                newp = Pair(ev1,ev2,-1)
                list_of_pairs.append(newp)
                
            fname_pair[f.replace('.inputs.json','')]= list_of_pairs

        return fname_pair

    def read_jsons_list(self, fname_list, dirname):
        data_list = list()
        for jasonfile in fname_list:
            print('processing {}'.format(jasonfile))
            #fname_list.append(jasonfile)
            jasonfile = os.path.join(dirname, jasonfile)
            with open(jasonfile) as json_data:
                data_list.append(json.load(json_data))

            fname_pair = defaultdict()

        for f, data in zip(fname_list, data_list):
            no_of_node = len(data)
            list_of_pairs = list()
            for i in range(no_of_node-1):
                for j in range(i+1, no_of_node):
                    ev1 = data[i]
                    ev2 = data[j]
                    newp = Pair(ev1,ev2,-1)
                    list_of_pairs.append(newp)
            fname_pair[f.replace('.inputs.json','')]= list_of_pairs

        return fname_pair
