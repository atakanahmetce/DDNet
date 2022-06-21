# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 21:44:16 2022

@author: Sameitos
"""

import networkx as nx
import numpy as np
from tqdm import tqdm
import re, os
import matplotlib.pyplot as plt


def get_names(file_name,idx = None):
    
    return np.loadtxt(file_name, dtype = str)

def get_sim_mat(file_name):

    return np.loadtxt(file_name)

def create_edgelist(file_name, names,sim_lim, similarity_matrix):
    
    if not os.path.isfile(file_name):        
        fused = similarity_matrix
        f = open(file_name,'w')
        for k in range(len(fused)):
            for t in range(len(fused)):
                weight = fused[k,t]
                if weight > sim_lim:
                    f.write(f'{names[k]} {names[t]} {weight}\n')
        f.close()
    else:
        print('The edgelist file is already exist')

def get_G(file_name):
    
    return nx.read_edgelist(file_name, nodetype = str, data = (('weight', float),))
    
def find_all_paths(source,target,G):
    
    return nx.all_simple_paths(G = G,source = source, target = target, cutoff = 8)

def get_label(file_name, names):
    
    labels = set()
    with open(file_name) as f:
        for row in f:
            row = re.split(',',row.strip('\n'))
            if row[0] in names and row[1] in names:
                labels.add((row[0],row[1]))
    
    return labels


























