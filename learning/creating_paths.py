# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 21:44:16 2022

@author: Sameitos
"""

import networkx as nx
import numpy as np
from tqdm import tqdm
import re, os, sys
import matplotlib.pyplot as plt


def get_samples(file_name):
    sample = []
    with open(file_name) as f:
        for row in f:
            sample.append(int(row.strip('\n')))
    return np.sort(sample)


def get_names(file_name,idx = None):
    names = []
    with open(file_name) as f:
        if idx is None:
            for row in f:
                names.append(row.strip('\n'))
        if idx is not None:
            for k,row in enumerate(f):
                
                if k in idx:names.append(row.strip('\n'))
    return names


def cos_str_sim(cos_file, str_file):
    
    cos_arr = np.loadtxt(cos_file)

    str_arr = np.loadtxt(str_file)

    return cos_arr, str_arr


def create_edgelist(file_name, names,sim_lim, idx, cos_file = None, str_file = None, fuse_name = None):

    if fuse_name is not None:
        fused = np.loadtxt(fuse_name) #linearly fused similarity matrices
    else:
        cos_arr, str_arr = cos_str_sim(cos_file, str_file)    
        fused = (cos_arr+str_arr)/2
    
    f = open(file_name,'w')
    addressed = set()
    for k in range(len(fused)):
        for t in range(len(fused)):
            if k in idx and t in idx:
                weight = fused[k,t]
                if weight > sim_lim:
                    f.write(f'{names[k]} {names[t]} {weight}\n')
        
    # for k,i in enumerate(names):
    #     for t,j in enumerate(names):
    #         weight = fused[k,t]
    #         if weight > sim_lim:
    #             f.write(f'{i} {j} {weight}\n')
    # return fused

def get_G(file_name):
    
    return nx.read_edgelist(file_name, nodetype = str, data = (('weight', float),))
    

def find_all_paths(source,target,G):
    
    return nx.all_simple_paths(G = G,source = source, target = target, cutoff = 8)


def draw_G(dG):
    
    f = plt.figure(figsize = (50,20))
    nx.draw(dG, with_labels = True)
    f.savefig('../figs/dG.svg', bbox_inches='tight')



def get_label(file_name, names):
    
    labels = set()
    with open(file_name) as f:
        for row in f:
            row = re.split(',',row.strip('\n'))
            if row[0] in names and row[1] in names:
                labels.add((row[0],row[1]))
    
    return labels


























