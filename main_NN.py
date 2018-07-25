from featureExtraction.feature import Feature
from featureExtraction.model_arg_TD import My_Model
from ontology_processing.reo_handling import Reo
from word_vector import word_vec_wrapper
from training_pair_preparation.classes import Pair
import json
import spacy
import os
import random
import numpy as np
import tensorflow as tf
import configuration.config as cfg
from keras.utils import to_categorical

nlp = spacy.load('en')
#label_data_path = 'cluster/all.cluster'
data_folders= ['']
label_data_training = cfg.LABEL_data_TRAINING
label_data_testing = 'cluster/testing1.cluster'
testfileName='/Users/abhipubali/Public/DropBox/AIDA_Paper/work/data/010aaf594ae6ef20eb28e3ee26038375.rich_ere.xml.inputs.json'
#w2v = word_vec_wrapper('/Users/abhipubali/Public/DropBox/sem2_s18/chenhao_course/word_embeddings_benchmarks/scripts/word2vec_wikipedia/wiki_w2v_300.txt')
#w2v = word_vec_wrapper(cfg.W2V_PATH ,nlp)
w2v = None
def read_lable_data(file):
    list_line = list()
    list_pairs = list()
    with open(file) as f:
        lines = f.readlines()
    for i in range(len(lines)):
        line = lines[i]
        #tokens = nlp(line)
        tokens = line.split()
        #0 file namae 2-ev1 4-ev2 6-cluster
        #print(tokens[0])
        list_pairs.append(Pair(str(tokens[1]),str(tokens[2]),str(tokens[3]),str(tokens[0])))
        #list_pairs.append(Pair(str(tokens[2]),str(tokens[4]),str(tokens[6]),str(tokens[0])))
    return list_pairs

def read_events_pairs(filepath, list_of_pair, augmentation =0):
    event_pairs_ones = list()
    event_pairs_zeros = list()
    actual_event_pairs = list()
    one_count =0
    zero_count=0
    aug_one_count=0
    aug_zero_count =0
    for p in list_of_pair:
        fname = p.fname + '.inputs.json'
        fname = os.path.join(filepath, fname)
        with open(fname) as json_data:
            data= json.load(json_data)
        for d in data:
            if d['event']['mentionid'] == p.ev1:
                ev1 = d
            elif d['event']['mentionid'] == p.ev2:
                ev2 = d
        newp = Pair(ev1,ev2,p.same)
        actual_event_pairs.append(newp)
        #print(p.same)
        if p.same==1:
            one_count = one_count+1
            event_pairs_ones.append(newp)
        else:
            zero_count = zero_count+1
            event_pairs_zeros.append(newp)
    print('no of zero={}, num of ones ={}'.format(zero_count, one_count))
    return actual_event_pairs,one_count, zero_count# event_pairs_ones,event_pairs_zeros

def data_augmentation(list_pair):
    indices = [x for x in range(len(list_pair))]
    new_pair = list()
    random.shuffle(indices)
    aug_zero_count = 0
    aug_one_count = 0
    for i in indices:
        p = list_pair[i]
        prob = random.uniform(0, 1)
        if prob > 0.4 and p.same==1:
            newp = Pair(p.ev2,p.ev1,p.same)
            aug_one_count = aug_one_count+1
            new_pair.append(newp)
        elif prob >0.9 and p.same==0:
            newp = Pair(p.ev2,p.ev1,p.same)
            aug_zero_count = aug_zero_count +1
            new_pair.append(newp)
    #data augmentation with reflexive
        prob = random.uniform(0, 1)
        if prob > 0.6:
            prob = random.uniform(0, 1)
            if prob > 0.5:
                newp = Pair(p.ev1,p.ev1,1)
                aug_one_count = aug_one_count+1
                new_pair.append(newp)
            else:
                newp = Pair(p.ev2,p.ev2,1)
                aug_zero_count = aug_zero_count +1
                new_pair.append(newp)
    list_pair.extend(new_pair)
    print(' augOne={}, augZero={}'.format( aug_one_count, aug_zero_count))
    return list_pair, aug_one_count, aug_zero_count

def train_test_split(actual_event_pairs):
    indices = [i for i in range(len(actual_event_pairs))]
    train_set = list()
    test_set = list()
    random.shuffle(indices)
    for i in indices:
        train_prob = random.uniform(0, 1)
        if train_prob >.8:
            test_set.append(actual_event_pairs[i])
        else:
            train_set.append(actual_event_pairs[i])
    return train_set, test_set

def sampler(train_ones, train_zeros):
    print('number of ones{} number of zeros{}'.format(len(train_ones),len(train_zeros)))
    sample_len = min(len(train_ones),len(train_zeros))
    train_set = list()
    one_ind = [i for i in range(len(train_ones))]
    zero_ind = [i for i in range(len(train_zeros))]
    random.shuffle(one_ind)
    random.shuffle(zero_ind)
    if sample_len < len(train_zeros):
        a = [train_zeros[i] for i in zero_ind]
        train_set.extend(a[:sample_len])
        train_set.extend(train_ones)
        ind = [i for i in range(len(train_set))]
        random.shuffle(ind)
        train_set = [train_set[i] for i in ind]
        print('len of final data {}'.format(len(train_set)))
        return train_set
    elif sample_len < len(train_ones):
        a = [train_ones[i] for i in one_ind]
        train_set.extend(a[:sample_len])
        train_set.extend(train_zeros)
        ind = [i for i in range(len(train_set))]
        random.shuffle(ind)
        train_set = [train_set[i] for i in ind]
        return train_set

def sampler2(train_set, oC,zC):

    sample_len = min(oC,zC)
    print('number of ones: {} number of zeros: {} sample_len:{}'.format(oC,zC, sample_len))
    train_ret= list()
    ind = [i for i in range(len(train_set))]
    random.shuffle(ind)
    count1= 0
    count0 = 0
    for i in ind:
        if train_set[i].same == 1 and count1 <sample_len:
            train_ret.append(train_set[i])
            count1 = count1+1
        if train_set[i].same == 0 and count0 <sample_len:
            train_ret.append(train_set[i])
            count0 = count0 +1
        if count1 >sample_len and count0>sample_len :
            break

    print('{} {} {}'.format(len(train_set), count0, count1))
    ind = [i for i in range(len(train_ret))]
    random.shuffle(ind)
    train_ret = [train_ret[i] for i in ind]
    print('total len {}'.format(len(train_ret)))
    return train_ret


def feature_extraction_caller(event_pair_list, npa):
    ev_X1 = list()
    ev_X2 = list()
    arg_X1 = list()
    arg_X2 = list()
    S = list()
    Y = list()
    feat = Feature()

    for p in event_pair_list:
        Y.append(p.same)
        f1= feat.extract_feature(p.ev1, w2v)
        f2= feat.extract_feature(p.ev2, w2v)
        l1 = p.ev1['event']['lemma']
        l2 = p.ev2['event']['lemma']
        #sim =w2v.similarity2(l1,l2)
        arg_X1.append(f1[0])
        ev_X1.append(f1[1])
        arg_X2.append(f2[0])
        ev_X2.append(f2[1])
        #S.append(sim)
    if npa ==1:
        ev_X1 = np.array(ev_X1)
        ev_X2 = np.array(ev_X2)
        arg_X1 = np.array(arg_X1)
        arg_X2 = np.array(arg_X2)
        arg_X1 = np.expand_dims(arg_X1, axis=2)
        arg_X2 = np.expand_dims(arg_X2, axis=2)
        #S = np.array(S)
        Y = np.array(Y)
    return ev_X1, arg_X1, ev_X2, arg_X2, Y

def ontology_similarity(ev1,ev2,reo):
    ev1_ere = ev1['event']['ere']
    ev1_reo = ev1['event']['reo']
    if ev1_reo.find('unknown')>0:# make it case insensative
        ev1_reo = reo.findreo(ev1_ere)

    ev2_ere = ev2['event']['ere']
    ev2_reo = ev2['event']['reo']
    if ev2_reo.find('unknown')>0:# make it case insensative
        ev2_reo = reo.findreo(ev2_ere)

    vec =[-1]*5
    if ev1_reo == ev2_reo:
        vec[0]= 1
    else:
        a,c,dist = reo.get_ancestor_distance(ev1_reo, ev2_reo)
        s,cp,d1,d2 = reo.get_siblings(ev1_reo, ev2_reo)
        vec[1]= dist
        vec[2]= s
        vec[3]= d1
        vec[4]= d2
    return vec


if __name__ == '__main__':

    print('processing .cluster files for training')
    list_of_pair_train = read_lable_data(label_data_training )
    #print(list_of_pair[0].fname)
    actual_event_pairs,ones, zeros = read_events_pairs(cfg.TRAINING_json_DATA, list_of_pair_train)

    print('processing input files for training')
    #train_ones, train_zeros = read_events_pairs(cfg.TRAINING_json_DATA, list_of_pair_train)


    #train_ones = data_augmentation(train_ones)
    #train_zeors = data_augmentation(train_zeros)
    train_set = actual_event_pairs
    train_set,oC,zC =data_augmentation(train_set)# sampler(train_ones, train_zeros)
    train_set = sampler2(train_set,oC+ones, zC+zeros)
    print('extracting features for training')
    train_X1, arg_X1, train_X2, arg_X2, train_Y = feature_extraction_caller(train_set,1)
    #train_Y = to_categorical(train_Y)
    model = My_Model(train_X1.shape[1],arg_X1.shape[1], 50)
    model.train_model(train_X1,arg_X1, train_X2, arg_X2, train_Y,epch=cfg.epch)
    if not os.path.exists('trained_model'):
        os.makedirs('trained_model')
    saved_fname = model.save_model('trained_model')
    print('dimention of input is {} x {}'.format(train_X1.shape[1], 50))
    print('model saved as {}'.format(saved_fname))
    '''
    print('loading model')
    model1 =  My_Model(373, 50)
    model1.load_model('trained_model/model_2.h5')

    print('processing .cluster files for testing')
    list_of_pair_test = read_lable_data(label_data_testing )
    print('processing input files for testing')
    test_set = read_events_pairs('data/Inputs', list_of_pair_test)
    print('processing input files for testing')
    test_X1, test_X2, test_Y = feature_extraction_caller(test_set,1)


    print(model1.evaluate(test_X1, test_X2, test_Y ))
    '''

    #t1 = np.array(test_X1[0]).reshape(1,373)
    #t2 = np.array(test_X2[0]).reshape(1,373)
    #print(t1.shape)
    #p_y = model.predict(test_X1, test_X2)
    #print(p_y.shape)
    #print(test_Y.shape)
    #p= model.predict(t1, t2)
    #for i,p  in zip(test_Y, p_y):
        #print('{} vs {}'.format(p,i))
# augment 1 labeled data by - swapping ev1 and ev2
#augment i labeld data by putting ev1=ev2 from any label data
