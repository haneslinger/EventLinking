
import os
import numpy as np
from collections import defaultdict

def generate_key(ff):
    filename = ff[0]
    pairs = ff[1]
    lines= list()
    cluster = defaultdict()
    c_num=0
    for p in pairs:
        ev1 = p.ev1['event']['mentionid']
        ev2 = p.ev2['event']['mentionid']
        same = p.same
        if same>0.49:
            if ev1 in cluster :#and ev2 not in cluster:
                cluster [ev2] = cluster[ev1]
            elif ev1 not in cluster and ev2 in cluster:
                cluster[ev1] = cluster[ev2]
            elif ev1 not in cluster and ev2 not in cluster:
                cluster[ev1] = c_num
                cluster[ev2] = c_num
                c_num = c_num + 1
            #elif ev1 in cluster and ev2 in cluster and cluster[ev1]!=cluster[ev2]:
                #print('\t think this as graph problem')
        else:
            if ev1 not in cluster:
                cluster[ev1] = c_num
                c_num = c_num + 1

            if ev2 not in cluster:
                cluster[ev2] = c_num
                c_num = c_num + 1
    return filename,cluster

def write_key(ff,key_path):
    cluster = ff[1]
    fname_to_write = ff[0].replace('.inputs.json','')
    fname = fname_to_write +'.key'
    #key_path = '../key'
    if not os.path.exists(key_path):
        os.makedirs(key_path)
    file= os.path.join(key_path ,fname)
    print('writing {}'.format(file))

    opfile = open(file,'w')
    opfile.write('#begin document ({}); part 000\n'.format(fname_to_write))

    for ev in cluster:
        opfile.write('{}\t{}\t({})\n'.format(fname_to_write, ev, cluster[ev]))
    opfile.write('#end document')
    opfile.close()
