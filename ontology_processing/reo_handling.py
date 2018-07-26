import json
import os
from collections import defaultdict
reo_jsonfile = 'reo.json'
ere2reo_json = 'ere2reo.json'

class Reo:
    def __init__(self, reo_json,ere2reo_json):
        with open(reo_json) as ontology:
            self.reo_ontology = json.load(ontology)
        with open(ere2reo_json) as mapping:
            self.ere2reo =  json.load(mapping)
        #self.reo_ontology = dict((str(k).lower(), str(v).lower()) for k,v in self.reo_ontology.items())
        #self.ere2reo = dict((str(k).lower(), str(v).lower()) for k,v in self.ere2reo.items())

    def findreo(self,ere):
        if ere.find("::")<0:
            ere = "ere::" + ere
        tokens = ere.split('::')
        ere = tokens[0] + '::' +tokens[1].title()
        if ere in self.ere2reo:
            return self.ere2reo[ere][0]
        return None

    def repair_key(self, target):
        ''' To make the first letter after '::' capital
        '''
        if target.find("::")<0:
            target = "reo::"+target
        tokens = target.split("::")
        a = tokens[1][0].upper() + tokens[1][1:]
        target = tokens[0] + '::' + a
        return target

    def get_ancestor(self, target):
        ''' find all ancestor of this target
        '''
        target = self.repair_key(target)
        return self.get_ancestor_helper(self.reo_ontology, target)

    def get_ancestor_helper(self, lookup_dic, target):
        ''' helper function for get_parent.
            Finds recursively all ancestor
        '''
        for element in lookup_dic:
            #print('{} vs {}'.format(element, target))
            if str(element) == target:
                return [element]
            else:
                if lookup_dic[element]:
                    check_child = self.get_ancestor_helper(lookup_dic[element],target)
                    if check_child:
                        return [element] + check_child



    def get_ancestor_distance(self, node1, node2):
        '''
        Given two nodes, returns parent, child and distance between them.
        else return None, None, -1
        '''
        if not node1 or not node2:
            return None, None, -1

        node1_rep = self.repair_key(node1)
        node2_rep = self.repair_key(node2)

        ancesstor_node1 = self.get_ancestor(node1_rep)
        if node2_rep in ancesstor_node1:
            ancestor = node2
            child = node1
            a_idx = ancesstor_node1.index(node2_rep)
            c_idx = len(ancesstor_node1)-1
            dist = c_idx - a_idx
            return ancestor, child, dist

        ancesstor_node2 = self.get_ancestor(node2_rep)
        if node1_rep in ancesstor_node2:
            ancestor = node1
            child = node2
            a_idx = ancesstor_node1.index(node1_rep)
            c_idx = len(ancesstor_node2)-1
            dist = c_idx - a_idx
            return ancestor, child, dist

        return None, None, -1



    def get_siblings(self, node1, node2):
        '''
        Given two nodes returns whether they are siblings, common parent and distance
        to the common parent
        '''
        if not node1 or not node2:
            return -1, None, -1, -1
        node1_rep = self.repair_key(node1)
        node2_rep = self.repair_key(node2)

        ancesstor_node1 = self.get_ancestor(node1_rep)
        ancesstor_node2 = self.get_ancestor(node2_rep)

        ancesstor_node1 = ancesstor_node1[:-1]
        ancesstor_node2 = ancesstor_node2[:-1]
        siblings = -1
        if ancesstor_node1 == ancesstor_node2:
            siblings = 1

        cp= None
        #if siblings < 0:
        for i,j in zip(ancesstor_node1, ancesstor_node2):
            if i==j:
                cp = i
            else:
                break
        dist1= -1
        dist2= -1
        if cp:
            p_idx1 = ancesstor_node1.index(cp)
            c_idx1 = len(ancesstor_node1)#don't substract 1, as the last element has been removed
            dist1 = c_idx1 - p_idx1
            p_idx2 = ancesstor_node2.index(cp)
            c_idx2 = len(ancesstor_node2)#don't substract 1, as the last element has been removed
            dist2 = c_idx2 - p_idx2

        return siblings, cp, dist1, dist2

    '''
    def get_parent(json_tree, target_id):
    for element in json_tree:
        if element['id'] == target_id:
            return [element['id']]
        else:
            if element['child']:
                check_child = get_parent(element['child'], target_id)
                if check_child:
                    return [element['id']] + check_child
    '''
    def get_depth_reo(self):
        return self.get_depth_onto(self.reo_ontology)

    def get_depth_onto(self,root):
        if root is None:
            return 0
        if isinstance(root, list):
            return max(map(self.get_depth_onto, root)) + 1
        if not isinstance(root, dict):
            return 1
        if root:
            return max(self.get_depth_onto(v) for k, v in root.items()) + 1
        else:
            return 0
if __name__ == '__main__':
    reo = Reo(reo_jsonfile,ere2reo_json)
    depth =reo.get_depth_reo()
    print('depth ={}'.format(depth))
    #print(reo.findreo("elect"))
    #print(reo.reo_ontology["reo::perdurant_entity"])
    #aaa = reo.get_parent("reo::extradite")
    #print(aaa)
    a,c,d = reo.get_ancestor_distance(None, None)
    print('ancestor = {}, child= {}, dist= {}'.format(a,c,d))
    s,cp,d1,d2 = reo.get_siblings(None,None)
    print('sibling = {}, cp= {}, dist1= {}, dist2= {}'.format(s,cp,d1,d2))
