# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 17:25:07 2022

@author: Sameitos
"""
from tqdm import tqdm
import json
import re
import numpy as np
from scipy.spatial.distance import cosine as cdt
from scipy.spatial.distance import rogerstanimoto
def loader():
    
    file_dt = '../data/drugs.json'
    file_dFP = '../data/drugFP.tsv'
    file_tCT = '../data/protein_1/fined_paac_matrix_1.txt'
    
    with open(file_dt) as f:
    	data = json.load(f)
    
    prots = set()
    drugs = {}
    for i in data:
        if i['accessions']:
            drugs[i['id']] = set(i['accessions'])
        for p in i['accessions']:
            prots.add(p)
    
    drug_feature = {}
    with open(file_dFP) as f:
        for row in tqdm(f):
            row = re.split('\t',row.strip('\n'))
            #print(re.split(' ',row[-1]))
            if row[0] in drugs.keys():
                drug_feature[row[0]] = np.array(list(re.split(' ',row[-1])[:-1]),dtype = int)
            
            
    prot_feature = {}
    with open(file_tCT) as f:
        for row in tqdm(f):
            row = re.split('\t',row.strip('\n'))
            if row[0] in prots:
                prot_feature[row[0]] = np.array(list(row[1:]),dtype = float)
    
    return prots, drugs, drug_feature, prot_feature


def load_interaction(prots,drugs):
    
    file_ppi = '../data/ppi_intact.txt'
    file_ddi = '../data/gokhans/drug-drug_interaction_Drugbank_v2.csv'

    ppi_data = {}    
    with open(file_ppi) as f:
        for row in tqdm(f):
            row = re.split(',',row.strip('\n'))
            if row[0] in prots and row[1] in prots:
                ppi_data[(row[0],row[1])] = float(row[-1])
    
    ddi_data = set()
    with open(file_ddi) as f:
        for row in tqdm(f):
            row = re.split(',',row.strip('\n'))
            if row[0] in drugs.keys() and row[1] in drugs.keys():
                ddi_data.add(tuple(row))
            
    return ppi_data,ddi_data


def create_ddi_matrix(drugs, ddi_data, eps):
    
    ddi_matrix = np.zeros((len(drugs.keys()),len(drugs.keys())))
    name = []
    for k,i in enumerate(tqdm(drugs.keys())):
        for t,j in enumerate(drugs.keys()):
            if (i,j) in ddi_data:
                ddi_matrix[k,t] = 1+eps
            elif i == j and (i,j) in ddi_data:
                ddi_matrix[k,t] = 1+eps
            else:
                ddi_matrix[k,t] = 0+eps
        name.append(i)
    
    f = open('../data/protein_1/ddi_mat.txt','w')
    name_write = "\t".join(name)
    f.write(f'name\t{name_write}\n')
    print('writing_ddi_matrix')
    for k,row in enumerate(tqdm(ddi_matrix)):
        row = '\t'.join(np.array(row, dtype = str))
        f.write(f'{name[k]}\t{row}\n')
    f.close()
        

def create_ppi_matrix(prots,ppi_data):
    
    ppi_matrix = np.zeros((len(prots),len(prots)))
    name = []
    for k,i in enumerate(tqdm(prots)):
        for t,j in enumerate(prots):
            try:
                if i != j:
                
                    ppi_matrix[k,t] = ppi_data[(i,j)]
                
                else:
                    ppi_matrix[k,t] = 1
            except:
                pass
        name.append(i)

    
    f = open('../data/protein_1/ppi_mat.txt','w')
    name_write = "\t".join(name)
    f.write(f'name\t{name_write}\n')
    print('writing_ppi_matrix')
    for k,row in enumerate(tqdm(ppi_matrix)):
        row = '\t'.join(np.array(row, dtype = str))
        f.write(f'{name[k]}\t{row}\n')
    f.close()
    

def drug_similarity_matrix(drug_feature):
    
    d_sim = np.zeros((len(drug_feature.keys()),len(drug_feature.keys())))
    name = []
    for k,i in enumerate(tqdm(drug_feature.keys())):
        name.append(i)
        for t,j in enumerate(drug_feature.keys()):
            
            d_sim[k,t] = 1 - rogerstanimoto(drug_feature[i], drug_feature[j])
            # joint = 0
            # for bit in range(len(drug_feature[i])):
            #     if drug_feature[i][bit] == drug_feature[j][bit]:
            #         joint+=1
            # d_sim[k,t] = joint/len(drug_feature[i])
            
            
    f = open('../data/protein_1/d_sim_mat.txt','w')
    name_write = "\t".join(name)
    f.write(f'name\t{name_write}\n')
    print('writing_d_sim')
    for k,row in enumerate(tqdm(d_sim)):
        row = '\t'.join(np.array(row, dtype = str))
        f.write(f'{name[k]}\t{row}\n')
    f.close()
    

def prot_similarity_matrix(prot_feature):
    
    p_sim = np.zeros((len(prot_feature.keys()),len(prot_feature.keys())))
    name = []
    for k,i in enumerate(tqdm(prot_feature.keys())):
        name.append(i)
        for t,j in enumerate(prot_feature.keys()):
            p_sim[k,t] = 1-cdt(prot_feature[i],prot_feature[j])
            #np.linalg.norm(prot_feature[i]-prot_feature[j])
        
    
    f = open('../data/protein_1/t_sim_mat.txt','w')
    name_write = "\t".join(name)
    print('writing_t_sim')
    f.write(f'name\t{name_write}\n')
    for k,row in enumerate(tqdm(p_sim)):
        row = '\t'.join(np.array(row, dtype = str))
        f.write(f'{name[k]}\t{row}\n')
    f.close()


# construct 'label_'list
def create_label_matrix(drugs,prots,prot_feature):
    f = open('../data/protein_1/label_list.txt','w')
    for d in tqdm(drugs.keys()):
        if d in drug_feature.keys():
            for p in prots:
                if p in prot_feature.keys():
                    if p in drugs[d]:f.write(f'{d}\t{p}\t{1}\n')
                    else:f.write(f'{d}\t{p}\t{0}\n')
    f.close()


prots, drugs, drug_feature, prot_feature = loader()
ppi_data,ddi_data = load_interaction(prots, drugs)

eps = 10**-3  
create_ddi_matrix(drugs, ddi_data, eps)
create_ppi_matrix(prots, ppi_data)
    
prot_similarity_matrix(prot_feature)
drug_similarity_matrix(drug_feature)

    
create_label_matrix(drugs,prots,prot_feature)


















