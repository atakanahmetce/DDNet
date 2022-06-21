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


def get_paths(file_name):
    
    paths = {}
    with open(file_name) as f:
        for row in f:
            row = re.split(',',row.strip('\n'))
            embeds[(row[0], row[1])] = list(np.array(re.split(',',row[-1]),dtype = float))
    return paths
    
def gen_matrix(file_embeds, file_paths, names, G, labels_list,fx_name,fy_name, cutoff = 3):

    fx = open(fx_name,'w')
    fy = open(fy_name,'w')
    embeds = get_n2v(file_name = file_embeds)
    paths = get_paths(file_name = file_paths)
    for k,src in enumerate(names):
        for t, target in enumerate(names):
 
            if src in embeds.keys() and target in embeds.keys():
                
                src_node = embeds[src]
                target_node = embeds[target]
                path_vector = paths[(src,target)]
                
                x_vector = np.array(src_node + target_node + path_features,dtype = str)
                
                if (src,target) in labels_list:label = 1
                else: label = 0
                
                write_vector = ' '.join(x_vector)

                fx.write(f'{write_vector}\n')
                fy.write(f'{label}\n')


    fx.close()
    fy.close()