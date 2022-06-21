# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 03:58:24 2022

@author: Sameitos
"""

import os
import numpy as np
import networkx as nx
from tqdm import tqdm

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


def get_path(src, target, G, cutoff):
        
    path_vector = []
    if src in G.nodes and target in G.nodes:
        
        path_vector = path_explorer(src, target, G, cutoff)
        
    else:
        for i in range(cutoff):
            path_vector.append(float(0))
    
    return path_vector


def find_paths(output_filename, G, cutoff = 3):
    
    if not os.path.isfile(output_filename):
        with open(output_filename,'w') as f:
            for src in tqdm(G.nodes):
                for target in G.nodes:
                    path_vector = get_path(src,target,G,cutoff)
                    write_path = ' '.join(np.array(path_vector,dtype = str))
                    f.write(f'{src},{target},{write_path}\n')
    else:
        print('The path file is already exist')











