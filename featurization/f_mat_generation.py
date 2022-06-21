# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 14:00:49 2022

@author: Sameitos
"""

import re
import networkx as nx
import numpy as np
from .creating_paths import *


def get_n2v(file_name):
    
    embeds = {}
    with open(file_name) as f:
        for row in f:
            row = re.split(' ',row.strip('\n'))
            embeds[row[0]] = list(np.array(re.split(',',row[1]),dtype = float))
    return embeds


def path_explorer(src, target, G, cutoff):
    
    simple_paths = nx.all_simple_paths(G,source = src, target = target, cutoff = cutoff)
    path_dict = {}
    for p in simple_paths:
        sim = 1
        for n in range(len(p)-1):
            sim*=G[p[n]][p[n+1]]['weight']
        
        if len(p)-1 not in path_dict.keys():
            path_dict[len(p)-1]=sim
        else:
            path_dict[len(p)-1]+=sim
    
    path_vector = []
    for i in range(1, cutoff+1):# in path_dict.keys():
        if i in path_dict.keys():
            path_vector.append(path_dict[i])
        else:
            path_vector.append(0)
    return path_vector


def get_path(src, target, G, cutoff = 3):
        
    path_vector = []
    if src in G.nodes and target in G.nodes:
        #if G.has_edge(src, target):
        path_vector = path_explorer(src, target, G, cutoff)
        #else:
        #    for i in range(cutoff):
        #        path_vector.append(0)
    else:
        for i in range(cutoff):
            path_vector.append(float(0))
    return path_vector
    
def gen_matrix(file_embeds, names, G, labels_list,fx_name,fy_name, cutoff = 3):

    fx = open(fx_name,'w')
    fy = open(fy_name,'w')
    embeds = get_n2v(file_name = file_embeds)

    for k,src in enumerate(names):
        for t, target in enumerate(names):
 
            if src in embeds.keys() and target in embeds.keys():
                src_node = embeds[src]
                target_node = embeds[target]
                    
                path_features = get_path(src, target, G, cutoff = cutoff)
                
                x_vector = np.array(src_node + target_node + path_features,dtype = str)
                
                if (src,target) in labels_list:label = 1
                else: label = 0
                
                write_vector = ' '.join(x_vector)

                fx.write(f'{write_vector}\n')
                fy.write(f'{label}\n')


    fx.close()
    fy.close()