# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 14:00:49 2022

@author: Sameitos
"""
import itertools
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from creating_paths import *
from node2vec import Node2Vec

'''
for i in range(1,11):

    sample_idx = get_samples(file_name = '../data_folder/samples/sample_idx_'+ str(i) + '.txt')
    names = get_names(file_name = '../data_folder/drug_names.txt', idx = None)
    
    edge_list_file = '../data_folder/edgelists/listSim05_' + str(i) + '.edgelist'
    
    create_edgelist(file_name = edge_list_file,
                    fuse_name = '../data_folder/lin_fused_sim.txt',
                    cos_file = '../data_folder/cos_arr.txt',
                    str_file = '../data_folder/str_arr.txt', names = names,
                    idx = sample_idx,
                    sim_lim = 0.5)
'''


def generate_subs(node, G, hop = 1):
    
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
    
    n2v = Node2Vec(G,p = p , q = q, dimensions = dimensions,
                    walk_length = walk_length,
                    num_walks=num_walks, 
                    workers = workers)
    
    model = n2v.fit(window = 10, min_count = 1, batch_words=4)
    
    return list(model.wv[center])
    

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
        if G.has_edge(src, target):
            path_vector = path_explorer(src, target, G, cutoff)
    return path_vector


def half_main(src,target,G,X,y,cutoff):
    
    src_sub = generate_subs(src, G,hop = 1)
    target_sub = generate_subs(target, G,hop = 1)
  
    src_node = n2v_embedding(src, src_sub, dimensions = 32, num_walks = 100, workers = 4)
    target_node = n2v_embedding(target, target_sub, dimensions = 32, num_walks = 100, workers = 1)
    
    path_features = path_explorer(src, target, G, cutoff = cutoff)
    
    x_vector = np.array(src_node + target_node + path_features)
    
    if (src,target) in labels_list:label = 1
    else: label = 0
    
    X.append(x_vector)
    y.append(label)
    
    
def main(names, G, labels_list):
    X = []
    y = []
    cutoff = 7
    c = 0
    for k,src in enumerate(names):
        for t, target in enumerate(names):
            #if (src,target) in labels_list:
            
            half_main(src,target,G,X,y,cutoff)
            c+=1
            print('\n',c,'\n')
            #break

        #break
    #print(c)
    #if c == 0:
    #    return c,'noooo'
    # ran_neg = np.random.randint(0,2*len(names), c*3)
    # for k,src in enumerate(names):
    #     for t, target in enumerate(names):
    #         if (k+t) in ran_neg and (src,target) not in labels_list:
    #             half_main(src,target,G,X,y,cutoff)
    #             ran_neg = ran_neg[ran_neg !=(k+t)]
            
    #         if len(ran_neg) == 0:
    #             return np.array(X),np.array(y)
    
    return np.array(X),np.array(y, dtype = int)

for i in tqdm(range(1,11)):
    
    sample_idx = get_samples(file_name = '../data_folder/samples/sample_idx_'+ str(i) + '.txt')
    
    edge_list_file = '../data_folder/edgelists/listSim05_' + str(i) + '.edgelist'
    if False:
        names = get_names(file_name = '../data_folder/drug_names.txt', idx = None)
        create_edgelist(file_name = edge_list_file,
                        fuse_name = '../data_folder/lin_fused_sim.txt',
                        cos_file = '../data_folder/cos_arr.txt',
                        str_file = '../data_folder/str_arr.txt', names = names,
                        idx = sample_idx,
                        sim_lim = 0.5)
    else:
        names = get_names(file_name = '../data_folder/drug_names.txt', idx = set(sample_idx))
        G = get_G(file_name = edge_list_file)
        labels_list = get_label(file_name = '../data_folder/dd_labels.csv', names = names)
        
        print(f'stage: {i}, {len(G.nodes)}, {len(G.edges)}, {len(labels_list)}')
        X,y = main(names, G, labels_list)
        
        if len(X) == 0: continue
        np.savetxt('../data_folder/training_data/X_'+  str(i) + '.txt',X)
        np.savetxt('../data_folder/training_data/y_' + str(i) + '.txt',y)
    
        
    
