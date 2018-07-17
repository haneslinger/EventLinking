import os
from collections import defaultdict
import spacy
from classes import Line, File, Pair
import random

class read_file:
    def __init__(self, fileName):
        self.filename = fileName
        self.data = list()
        self.process_data= list()
        self.process_train_data = list()
        self.process_test_data= list()

    def readFiles(self):

        print('processing {}'.format(self.filename))
        file = self.filename
        list_line = list()
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
            same =0
            if str(tokens[4]).lower()== 'same':
                same = 1
            list_line.append([fname1,ev1,ev2,fname2, same])
        self.data = list_line

    def writeOP_train_test(self):
        print(len(self.data))
        index_shuf = list(range(len(self.data)))
        random.shuffle(index_shuf)
        for i in index_shuf:
            prob = random.uniform(0, 1)
            if prob <= 0.8:
                self.process_train_data.append(self.data[i])
            else:
                self.process_test_data.append(self.data[i])

        print('training length{} and test lemgth {}'.format(len(self.process_train_data),len(self.process_test_data)))
        file_train= os.path.join('../cluster','train_cross_doc.tsv')
        opfile = open(file_train,'w')
        for d in self.process_train_data:
            opfile.write('{}\t{}\t{}\t{}\t{}\n'.format(d[0],d[1],d[2],d[3],d[4]))

        file_test= os.path.join('../cluster','test_cross_doc.tsv')
        opfile = open(file_test,'w')
        for d in self.process_test_data:
            opfile.write('{}\t{}\t{}\t{}\t{}\n'.format(d[0],d[1],d[2],d[3],d[4]))

df = read_file('../cluster/all_xdoc_links.tsv')
df.readFiles()
df.writeOP_train_test()
