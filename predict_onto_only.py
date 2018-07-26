from featureExtraction.feature_new import Feature
from featureExtraction.model_arg_onto_only import My_Model
from ontology_processing.reo_handling import Reo
from word_vector import word_vec_wrapper
from training_pair_preparation.classes import Pair
import json
import spacy
import os
import random
import numpy as np
import tensorflow as tf
from input_reading.input_reader import Data
import configuration.config as cfg
from postprocessing.key_generation import generate_key, write_key
nlp = spacy.load('en')
w2v = None#word_vec_wrapper(cfg.W2V_PATH ,nlp)

def feature_extraction_caller(event_pair_list, npa):
    ev_X1 = list()
    ev_X2 = list()
    arg_X1 = list()
    arg_X2 = list()
    onto_sim = list()
    S = list()
    Y = list()
    feat = Feature()
    reo = Reo(cfg.REO, cfg.ERE2REO)

    for p in event_pair_list:
        Y.append(p.same)
        f1= feat.extract_feature(p.ev1, w2v)
        f2= feat.extract_feature(p.ev2, w2v)
        l1 = p.ev1['event']['lemma']
        l2 = p.ev2['event']['lemma']
        onto = feat.ontology_similarity(p.ev1, p.ev2, reo)
        #sim =w2v.similarity2(l1,l2)
        arg_X1.append(f1[0])
        ev_X1.append(f1[1])
        arg_X2.append(f2[0])
        ev_X2.append(f2[1])
        onto_sim.append(onto)
        #S.append(sim)
    if npa ==1:
        ev_X1 = np.array(ev_X1)
        ev_X2 = np.array(ev_X2)
        arg_X1 = np.array(arg_X1)
        arg_X2 = np.array(arg_X2)
        onto_sim = np.array(onto_sim)
        arg_X1 = np.expand_dims(arg_X1, axis=2)
        arg_X2 = np.expand_dims(arg_X2, axis=2)
        #S = np.array(S)
        Y = np.array(Y)
    return ev_X1, arg_X1, ev_X2, arg_X2,onto_sim, Y


df = Data()

print('loading model {}'.format(cfg.IND_MODEL_PATH))
model1 =  My_Model( 20, 4, 50)#411 with ere event #373
model1.load_model(cfg.IND_MODEL_PATH)
print(model1.model.summary())
fname_pair = df.read_jsons(cfg.TESTING_json_DATA)
#model2 =  My_Model(411, 50)#411 with ere event #373
#model2.load_model(cfg.CROSS_MODEL_PATH)

fname_cluster_pair = list()

for fname in fname_pair:
    print('predicting {}'.format(fname))
    list_of_pairs = fname_pair[fname]
    test_X1, arg_X1, test_X2, arg_X2, onto_sim, test_Y = feature_extraction_caller(list_of_pairs,1)
    predicted_y = model1.predict( arg_X1, arg_X2, onto_sim)
    #predicted_y2 = model2.predict(test_X1, test_X2, test_S)
    #predicted_y = (predicted_y+predicted_y2)/2
    #print(predicted_y)
    #print('len of predicted = {}\n len of actual={} '.format(len(predicted_y),len(test_Y)))
    for i in range(len(predicted_y)):
        list_of_pairs[i].same = predicted_y[i]
        #print('{} vs {} = {}'.format(list_of_pairs[i].ev1['event']['mentionid'],list_of_pairs[i].ev2['event']['mentionid'],predicted_y[i]))
    fname, cluster = generate_key([fname,list_of_pairs])
    fname_cluster_pair.append([fname,cluster])

for ff in fname_cluster_pair:
    write_key(ff,cfg.OUTPUT_KEY)
