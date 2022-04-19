# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 23:16:51 2022

@author: Sameitos
"""


from scipy.spatial.distance import rogerstanimoto
from tqdm import tqdm
import numpy as np
import re

def loader(file_dFP):
    
    drug_feature = {}
    with open(file_dFP) as f:
        for row in tqdm(f):
            row = re.split('\t',row.strip('\n'))
            drug_feature[row[0]] = np.array(list(re.split(' ',row[-1])[:-1]),dtype = int)
    
    return drug_feature


def drug_similarity_matrix(drug_feature):
    
    d_sim = np.zeros((len(drug_feature.keys()),len(drug_feature.keys())))
    name = []
    for k,i in enumerate(tqdm(drug_feature.keys())):
        name.append(i)
        for t,j in enumerate(drug_feature.keys()):
            
            d_sim[k,t] = 1 - rogerstanimoto(drug_feature[i], drug_feature[j])
            
    f = open('../data_drug/d_sim_mat.txt','w')
    name_write = "\t".join(name)
    f.write(f'name\t{name_write}\n')
    print('writing_d_sim')
    for k,row in enumerate(tqdm(d_sim)):
        row = '\t'.join(np.array(row, dtype = str))
        f.write(f'{name[k]}\t{row}\n')
    f.close()

drug_feature = loader('../data_drug/drug_feature_mat.txt')
drug_similarity_matrix(drug_feature=drug_feature)
