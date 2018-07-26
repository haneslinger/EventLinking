#import sys
#sys.path.append("..")
#from ..fileReading import Data
from .entity_dic import entity_dict
import gensim
import json
import os
import numpy as np
#import  .word_vector.word_vec_wrapper
#from ..ontology_processing.reo_handling import Reo
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

realismap ={'actual':0, 'generic':1, 'other':2, 'non-event':3}

nermap = {'sentence':0, 'commodity':1, 'time':2, 'crime':3, 'LOC':4, 'vehicle':5, 'PER':6, 'money':7, 'GPE':8, 'weapon':9, 'ORG':10, 'title':11, 'FAC':12}

arg_specificity = {'nonspecific': 1, 'specificGroup': 3, 'specific': 0, 'specificIndeterminate': 5, 'specificIndividual': 4, 'UNK': 2}

arg_ere= {'ere::Beneficiary': 46, 'adjudicator': 38, 'person': 8, 'ere::Giver': 43, 'ere::Agent': 24, 'destination': 15, 'audience': 17, 'place': 18, 'ere::Destination': 42, 'ere::Audience': 45, 'ere::Adjudicator': 48, 'giver': 36, 'ere::Person': 23, 'defendant': 11, 'ere::Origin': 41, 'ere::Org': 25, 'recipient': 20, 'ere::Defendant': 33, 'ere::Position': 39, 'beneficiary': 40, 'attacker': 6, 'instrument': 7, 'ere::Money': 44, 'ere::Target': 2, 'entity': 9, 'artifact': 28, 'ere::Place': 3, 'target': 14, 'money': 32, 'ere::Victim': 27, 'crime': 19, 'position': 26, 'ere::Crime': 35, 'origin': 22, 'thing': 21, 'victim': 0, 'ere::Entity': 12, 'ere::Thing': 31, 'time': 13, 'prosecutor': 10, 'ere::Prosecutor': 37, 'agent': 1, 'ere::Instrument': 5, 'ere::Attacker': 4, 'ere::Sentence': 34, 'ere::Artifact': 16, 'ere::Recipient': 30, 'ere::Time': 29, 'ere::Plaintiff': 47}

event_ere_map = {'transportperson': 6, 'correspondence': 18, 'acquit': 33, 'transferownership': 8, 'extradite': 37, 'fine': 22, 'injure': 11, 'trialhearing': 4, 'attack': 1, 'beborn': 14, 'mergeorg': 21, 'transportartifact': 16, 'sue': 34, 'endorg': 30, 'transaction': 29, 'releaseparole': 32, 'arrestjail': 2, 'transfermoney': 17, 'divorce': 19, 'execute': 31, 'contact': 3, 'pardon': 26, 'meet': 13, 'startorg': 10, 'die': 0, 'appeal': 35, 'broadcast': 5, 'nominate': 20, 'convict': 24, 'endposition': 27, 'artifact': 7, 'declarebankruptcy': 36, 'demonstrate': 28, 'chargeindict': 25, 'startposition': 12, 'sentence': 23, 'marry': 15, 'elect': 9}

event_reo_map = {'reo::UNK': 0, 'reo::Give': 1, 'reo::Send': 2, 'reo::Physical_attack': 3, 'reo::Death': 4, 'reo::Get': 5, 'reo::Accompanied_directed_intrinsic_change_location': 6, 'reo::Kill': 7, 'reo::Instrument_Communication': 8, False: 9, 'reo::Intrinsic_change_location': 10, 'reo::Statement': 11, 'reo::Buy_sell': 12, 'reo::Accompanied_directed_caused_change_location': 13, 'reo::Execute': 14, 'reo::Exchange': 15, 'reo::Meet': 16, 'reo::Conversing': 17, 'reo::Lecture': 18, 'reo::Intrinsic_reach': 19, 'reo::Intrinsic_leave': 20, 'reo::Arrest': 21, 'reo::Steal': 22, 'reo::End_position_with_organization': 23, 'reo::Hiring': 24, 'reo::Appeal': 25, 'reo::Firing': 26, 'reo::Prosecute': 27, 'reo::Discharge': 28, 'reo::Verdict': 29, 'reo::Begin_leadership_position': 30, 'reo::End_leadership_position': 31, 'reo::Loan_borrow': 32, 'reo::Experience_Injury': 33, 'reo::Change_possession': 34, 'reo::Charge': 35, 'reo::Caused_leave-Remove': 36, 'reo::Protest_conflict': 37, 'reo::Put': 38, 'reo::Declarational': 39, 'reo::Birth': 40, 'reo::Pardon': 41, 'reo::Attitudinal': 42, 'reo::Extradite': 43, 'reo::Trajectory_focused_intrinsic_change_location': 44, 'reo::Divorce': 45, 'reo::Verbal_Communication': 46, 'reo::Response': 47, 'reo::Question': 48, 'reo::Marry': 49}

arg_reo_map = {'UNK': 0, 'reo::Donor': 1, 'reo::Recipient': 2, 'reo::Agent': 3, 'reo::Theme': 4, 'reo::Attacker': 5, 'reo::Target': 6, 'reo::Victim': 7, 'reo::AccompaniedTheme': 8, 'reo::Destination': 9, 'reo::Mover': 10, 'reo::place': 11, 'reo::Place': 12, 'reo::Time': 13, 'reo::time': 14, 'reo::CrimeOrCause': 15, 'reo::Meeting_Parties': 16, 'reo::Audience': 17, 'reo::ModeOfTransportation': 18, 'reo::Initial_Location': 19, 'reo::Suspect': 20, 'reo::Employee': 21, 'reo::Employer': 22, 'reo::Defendant': 23, 'reo::Position': 24, 'reo::CostOrCotheme': 25, 'reo::Adjudicator': 26, 'reo::Captive': 27, 'reo::GovernedEntity': 28, 'reo::Leader': 29, 'reo::Cause': 30, 'reo::Protester': 31, 'reo::Child': 32, 'reo::Weapon': 33, 'reo::Authority': 34, 'reo::Instrument': 35, 'reo::Couple': 36, 'reo::Prosecutor': 37, 'reo::Artifact': 38}
class Feature:
    def __init__(self):
        key_max = max(arg_ere.keys(), key=(lambda k: arg_ere[k]))
        key_min = min(arg_ere.keys(), key=(lambda k: arg_ere[k]))
        self.arg_ere_max = arg_ere[key_max] + 1
        self.arg_ere_min = arg_ere[key_min]

        key_max = max(arg_specificity.keys(), key=(lambda k: arg_specificity[k]))
        key_min = min(arg_specificity.keys(), key=(lambda k: arg_specificity[k]))
        self.arg_specificity_max = arg_specificity[key_max] + 1
        self.arg_specificity_min = arg_specificity[key_min]

        key_max = max(arg_reo_map.keys(), key=(lambda k: arg_reo_map[k]))
        key_min = min(arg_reo_map.keys(), key=(lambda k: arg_reo_map[k]))
        self.arg_reo_map_max = arg_reo_map[key_max] + 1
        self.arg_reo_map_min = arg_reo_map[key_min]

        key_max = max(nermap.keys(), key=(lambda k: nermap[k]))
        key_min = min(nermap.keys(), key=(lambda k: nermap[k]))
        self.nermap_max = nermap[key_max] + 1
        self.nermap_min = nermap[key_min]

        key_max = max(event_ere_map.keys(), key=(lambda k:event_ere_map[k]))
        key_min = min(event_ere_map.keys(), key=(lambda k: event_ere_map[k]))
        self.event_ere_map_max = event_ere_map[key_max] + 1
        self.event_ere_map_min = event_ere_map[key_min]

        key_max = max(event_reo_map.keys(), key=(lambda k: event_reo_map[k]))
        key_min = min(event_reo_map.keys(), key=(lambda k: event_reo_map[k]))
        self.event_reo_map_max = event_reo_map[key_max] + 1
        self.event_reo_map_min = event_reo_map[key_min]

        key_max = max(realismap.keys(), key=(lambda k: realismap[k]))
        key_min = min(realismap.keys(), key=(lambda k: realismap[k]))
        self.realismap_max = realismap[key_max] + 1
        self.realismap_min = realismap[key_min]

        key_max = max(entity_dict.keys(), key=(lambda k: entity_dict[k]))
        key_min = min(entity_dict.keys(), key=(lambda k: entity_dict[k]))
        self.entity_dict_max = entity_dict[key_max] + 1
        self.entity_dict_min = entity_dict[key_min]

    def extract_feature(self, event,w2v):

        #if w2v != None:
            #word2vec_lemma = w2v.vector(event['event']['lemma'])#w2v[event['event']['lemma']]


        #----- argument features-----#
        args = event['arguments']
        no_of_args = len(args)

        #arg_entity_presence = [0]*len(entity_dict)
        i=0
        arg_feature = list()
        while i<5:

            if i < len(args):
                a = args[i]
                ere = a['ere']
                sp = a['entity-specificity']
                ner = a['entity-ner']
                entity = a['entity']
                areo = a['reo']

                aere_norm = (arg_ere[ere]+1 - self.arg_ere_min)/(self.arg_ere_max - self.arg_ere_min)
                areo_norm = (arg_reo_map[areo]+1 - self.arg_reo_map_min)/(self.arg_reo_map_max-self.arg_reo_map_min)
                asp_norm = (arg_specificity[sp]+1 - self.arg_specificity_min)/(self.arg_specificity_max - self.arg_specificity_min)
                aner_norm = (nermap[ner]+1 - self.nermap_min)/(self.nermap_max - self.nermap_min)
                aen_norm = (entity_dict[entity]+1 - self.entity_dict_min)/(self.entity_dict_max - self.entity_dict_min)
                ## save zero to build zero vector for padding. Hence adding one to all
                arg_feature.append(aere_norm)
                arg_feature.append(asp_norm )
                arg_feature.append(aner_norm)
                arg_feature.append(aen_norm )
                arg_feature.append(areo_norm )
            else:
                #zero vector padding
                arg_feature.append(0)
                arg_feature.append(0)
                arg_feature.append(0)
                arg_feature.append(0)
                arg_feature.append(0)
            i = i+1

        #--create the event feature---#
        ev_realis = event['event']['modality']
        ev_ere = event['event']['ere']
        ev_reo = event['event']['reo']

        realis_norm = ( realismap[ev_realis]+1 - self.realismap_min)/(self.realismap_max -self.realismap_min)
        ev_ere_norm = ( event_ere_map[ev_ere]+1 - self.event_ere_map_min)/( self.event_ere_map_max - self.event_ere_map_min)
        ev_reo_norm = (event_reo_map[ev_reo]+1 - self.event_reo_map_min)/(self.event_reo_map_max - self.event_reo_map_min)
        ev_feature = list()
        if no_of_args >5:
            no_of_args =5
        noarg_norm = no_of_args/5
        ev_feature.append(realis_norm)
        #if w2v != None:
            #ere_feature.extend(word2vec_lemma)
        ev_feature.append(ev_ere_norm)
        ev_feature.append(ev_reo_norm)
        ev_feature.append(no_of_args)
        feature = [arg_feature,ev_feature]
        #return np.array(feature)
        return [arg_feature,ev_feature]

    def ontology_similarity(self, ev1,ev2,reo):
        ev1_ere = ev1['event']['ere']
        ev1_reo = ev1['event']['reo']
        ev1_realis = realismap[ev1['event']['modality']]


        ev2_ere = ev2['event']['ere']
        ev2_reo = ev2['event']['reo']
        ev2_realis = realismap[ev2['event']['modality']]

        #print('finding relation between {} and {}'.format(ev1_reo,ev2_reo))
        if not ev2_reo:
            ev2_reo = 'reo::UNK'
        if not ev1_reo:
            ev1_reo = 'reo::UNK'

        if ev1_reo.find('unknown')>0 or ev1_reo.find('UNK')>0 :# make it case insensative
            ev1_reo = reo.findreo(ev1_ere)
        if ev2_reo.find('unknown')> 0 or ev2_reo.find('UNK')>0:# make it case insensative
            ev2_reo = reo.findreo(ev2_ere)

        #print('\t after change {} and {}'.format(ev1_reo,ev2_reo))

        r1 = max(ev1_realis, ev2_realis)
        r2 = min(ev1_realis, ev2_realis)
        r12 = 10*r1+r2

        vec =[-1]*5

        no_arg1 = len(ev1['arguments']) if len(ev1['arguments'])<5 else 5
        no_arg2 = len(ev2['arguments']) if len(ev2['arguments'])<5 else 5
        diff_val = (no_arg1 - no_arg2)/5
        if diff_val <0:
            diff_val = diff_val *-1
        vec.append(diff_val/5)

        if ev1_reo == ev2_reo:
            vec[0]= 1
        else:
            a,c,dist = reo.get_ancestor_distance(ev1_reo, ev2_reo)
            s,cp,d1,d2 = reo.get_siblings(ev1_reo, ev2_reo)
            vec[1]= (dist+1)/2 #normalize in 0,1
            vec[2]= (s+1)/2
            vec[3]= d1/12
            vec[4]= d2/12
        #vec.append(r12)
        return vec

if __name__ == '__main__':
    testfileName='/Users/abhipubali/Public/DropBox/AIDA_Paper/work/data/Inputs/010aaf594ae6ef20eb28e3ee26038375.rich_ere.xml.inputs.json'
    with open(testfileName) as json_data:
        data= json.load(json_data)
    ev= data[0]
    print(ev)
    feat = Feature()
    #w2v = KeyedVectors.load_word2vec_format('/Users/abhipubali/Public/DropBox/sem2_s18/chenhao_course/word_embeddings_benchmarks/scripts/word2vec_wikipedia/wiki_w2v_300.txt', binary=False)
    f= feat.extract_feature(ev, None)
    #print(f.shape)
    print(f)
