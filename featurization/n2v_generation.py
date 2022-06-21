# -*- coding: utf-8 -*-
"""
Created on Wed May 25 19:48:09 2022

@author: Sameitos
"""
import os 
import numpy as np
import networkx as nx
from node2vec import Node2Vec
import itertools
from .creating_paths import *
from tqdm import tqdm


def generate_subs(node, G, hop):
    
    if hop == 1:
        sub = G.subgraph(list(G.neighbors(node)) + [node])

    elif hop == 2:
        first = list(G.neighbors(node)) + [node]
        second = [[i for i in G.neighbors(j)] for j in first]
        sub = G.subgraph(first + list(itertools.chain(*second)))

    return sub


def n2v_embedding(center, subgraph, dimensions = 32, walk_length = 30,
                  num_walks = 100, workers = 1,
                  p = 1, q = 1):
    
    n2v = Node2Vec(subgraph,p = p , q = q, dimensions = dimensions,
                    walk_length = walk_length,
                    num_walks=num_walks, 
                    workers = workers)
    
    model = n2v.fit(window = 10, min_count = 1, batch_words=4)
    
    return list(model.wv[center])



def gen_n2v(output_filename, G, hop = 1,
    dimensions = 32, num_walks = 100, walk_length = 30,
    p = 1, q = 1, workers = 4):
    
    if not os.path.isfile(output_filename):
        with open(output_filename,'w') as f:
            for j,k in enumerate(G.nodes):
                #if k in G.nodes:
                src_sub = generate_subs(k, G,hop = hop)
                src_node = n2v_embedding(k, src_sub, dimensions = dimensions, num_walks = num_walks, workers = workers,
                                        p = p, q = q, walk_length = walk_length)
                write_src = ','.join(np.array(src_node, dtype = str))
                f.write(f'{k} {write_src}\n')

    else:
        print('The embedding file is already exist')







