# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 12:24:35 2022

@author: Sameitos
"""

from featurization.creating_paths import *
from featurization import find_paths
from featurization import gen_n2v, gen_matrix

#creating graph from similarity matrices
names = get_names(file_name = 'data/datasets/test_data/names.txt')
sim_mat = get_sim_mat('data/datasets/test_data/sim_arr.txt')

edgelist_filename = 'data/edgelists/test_data.edgelist'
create_edgelist(file_name = edgelist_filename,
               similarity_matrix = sim_mat,
               names = names,
               sim_lim = 0.3)
G = get_G(file_name = edgelist_filename)

#generate paths and save them to file between all drugs found in graph:
meta_path_file = 'data/meta_paths/test_paths.txt'
find_paths(output_filename = meta_path_file, G = G, cutoff = 3)


print('graph done')
#form embeddings and save to file
embed_filename = 'data/n2v_embeddings/test_embeds.txt'
gen_n2v(output_filename = embed_filename, G = G,
       hop = 1, workers = 1)

print('paths done')
#feature (reconstruction) and label matrix generation
labels_list = get_label(file_name = 'data/datasets/data_mp/interactions.txt', names = names)

fx = 'data/training_data/X.txt' #feature matrix file name
fy = 'data/training_data/y.txt' #label matrix file name

print('matrix construction done')
gen_matrix(embed_filename, meta_path_file, labels_list,fx,fy)






