# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 21:58:08 2022

@author: Sameitos
"""

import json,re
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from tqdm import tqdm

def load_json():
    drug_file = '../data/drugs.json'
    with open(drug_file) as f:
    	data = json.load(f)
    
    drugs = {}
    for i in data:
        drugs[i['id']] = i['smiles']
    return drugs

def loader(drugs):

    data_int = set()
    with open('../data/gokhans/drug-drug_interaction_Drugbank_v2.csv') as f:
        for row in f:
            row = re.split(',',row.strip('\n'))
            if row[0] in drugs.keys() and row[1] in drugs.keys():
                data_int.add((row[0],row[1]))
                
    return data_int

def featurizer(drugs):
    
    non_converts = []
    feature_matrix = {}
    for d in tqdm(drugs.keys()):
        try:
            mol = Chem.MolFromSmiles(drugs[d])
            fp = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits = 4096)
            feature_matrix[d] = DataStructs.cDataStructs.BitVectToText(fp)
        except:
            non_converts.append(d)

    return feature_matrix

def write(data):
    

    #write feature data
    with open('../data_drug/drug_feature_mat.txt', 'w') as f:
        for d in data.keys():
            row = ' '.join(data[d])
            f.write(f'{d}\t{row}\n')
        
    # #write interaction data
    # with open('../data_drug/drug_drug_interactions.txt', 'w') as f:
    #     for i in tqdm(data.keys()):
    #         for j in data.keys():
    #             if (i,j) in data_int:f.write(f'{i},{j},{1}\n')
    #             else: f.write(f'{i},{j},{0}\n')



    # drug_name = []
    # with open('../data/gokhans/drugs_final_v7.txt') as df:
    #     for row in df:
    #         row = row.strip('\n')
    #         drug_name.append(row)
            
    
    # drug_feature = []
    # with open('../data/gokhans/featureS_drugs_final_v7_2.txt') as dff:
    #     for row in dff:
    #         row = row.strip('\n')
    #         drug_feature.append(row)
            
    
    
    
    
    #return #drug_feature, drug_name


drugs = load_json()
#interactions = loader(drugs)
drug_feature = featurizer(drugs)
write(data = drug_feature)#, data_int = interactions)









































































































