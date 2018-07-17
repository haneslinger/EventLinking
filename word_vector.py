import gensim
import numpy as np
import scipy
import string
import json
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from scipy import spatial
from timeit import default_timer as timer
from datetime import datetime
import random

class word_vec_wrapper:
    def __init__(self, fname,nlp):
        self.word_vectors = KeyedVectors.load_word2vec_format(fname, binary=False)
        self.nlp = nlp
        self.dim = self.word_vectors['the'].shape[0]
        self.unk = [random.gauss(0, 0.01) for _ in range(self.dim )]

    def similarity (self,d1,d2):
        sim = self.word_vectors.similarity(d1, d2)
        dis = self.word_vectors.distance(d1, d2)
        return sim
        #print('w1={} w2 = {} sim={}\tdis={}'.format(d1,d2,sim,dis))

    def similarity2(self, d1,d2):
        v1 = self.vector(d1)
        v2 = self.vector(d2)
        sim = 1-spatial.distance.cosine(v1,v2)
        return sim

    def vector(self, word):
        tokens = self.nlp(word)
        if len(tokens) == 1:
            if word.lower() in self.word_vectors.vocab:
                return self.word_vectors[word.lower()]
            else:
                return self.unk
        word_vecs = list()
        for tok in tokens:
            if str(tok).lower() in self.word_vectors.vocab:
                word_vecs.append(self.word_vectors[str(tok).lower()])
            else:
                word_vecs.append(self.unk)

        combined = [0]*len(word_vecs[0])
        for v in word_vecs:
            temp = [sum(x) for x in zip(combined, v)]
            combined = temp
        temp = [x/len(word_vecs) for x in combined]
        return temp


'''
s1 = datetime.now()#timer()
w2vec = word_vec_wrapper('/Users/abhipubali/Public/DropBox/sem2_s18/chenhao_course/word_embeddings_benchmarks/scripts/word2vec_wikipedia/wiki_w2v_300.txt')
e1 = datetime.now()#timer()
print('time taken to load(hh:mm:ss:ms):{}'.format(e1-s1))
s2 = datetime.now()#timer()
w2vec.similarity("kill","death")
w2vec.similarity("kill","die")
w2vec.similarity("correspondence","contact")
w2vec.similarity("remark","shit-talking")
e2 = datetime.now()#timer()
print('time taken to calculate(hh:mm:ss:ms):{}'.format(e2-s2))
#w2vec.similarity('man','woman')
'''
