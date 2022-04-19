# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 21:54:25 2022

@author: Sameitos
"""

import json
import re
import os



drug_file = '../data/drugs.json'
with open(drug_file) as f:
	data = json.load(f)




#Protein fasta
'''
fasta_protein = '../data/gokhans/uniprot_human_review.fasta'
fasta_data = {}
with open(fasta_protein) as f:
    seq = []
    for k,row in enumerate(f):
        if k == 0:
            row = re.split(' ',row.strip('\n'))[0]
            pr =''
            for i in range(4,len(row)):    
                if row[i] == '|':break
                pr+=row[i]
        if k != 0 and re.search('>',row):
            
            
            fasta_data[pr] = ''.join(seq)
            seq = []
            row = re.split(' ',row.strip('\n'))[0]
            pr = ''
            for i in range(4,len(row)):    
                if row[i] == '|':break
                pr+=row[i]

            
        else:
            
            seq.append(row.strip('\n'))
            
    fasta_data[pr] = seq
            

tf = open('../data/prot_fasta.fa','w')
for i in data:
    for p in i['accessions']:
        if p in fasta_data.keys():
            if len(fasta_data[p])>31 and len(re.sub('X','',fasta_data[p]))>31:
                tf.write('%s\n' % ''.join(['>' + p + '|' + str(1) + '|' + 'training']))
                tf.write('%s\n' % fasta_data[p])
tf.close()
'''


#Drugs
'''
smiles = '../data/gokhans/drug_smiles_Drugbank.txt'
smiles_data = {}
with open(smiles) as f:
 	for row in f:
          row = re.split(',', row.strip('\n'))
          smiles_data[row[0]] = row[-1]


drug_feature = []
with open('../data/gokhans/featureS_drugs_final_v7_2.txt') as dff:
    for row in dff:
        row = row.strip('\n')
        drug_feature.append(row)
        
drug_name = []
with open('../data/gokhans/drugs_final_v7.txt') as df:
    for row in df:
        row = row.strip('\n')
        drug_name.append(row)

data_dict = {}
for i in range(len(drug_feature)):
    data_dict[drug_name[i]] = drug_feature[i]


f = open('../data/true_smiles.tsv','w')
for i in data:
      if i['id'] in data_dict.keys():
          f.write(f'{i["id"]}\t{data_dict[i["id"]]}\n')
f.close()

'''





