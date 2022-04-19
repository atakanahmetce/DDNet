# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 08:19:59 2022

@author: Sameitos
"""

#fusing function

import snf
import re, os
import numpy as np


def loader(file, file_name = None):
    
    data = []
    name = None
    with open(file) as f:
        for row in f:
            row = re.split('\t',row.strip('\n'))
            if row[0] != 'name':
                data.append(np.array(row[1:], dtype = float))
            else:
                name = np.array(row[1:])
        
    if file_name is not None and not os.path.isfile(file_name):
        with open(file_name,'w') as f:
            for n in name:
                f.write(f'{n}\n')
    return np.array(data),name
    
def fuser_writer(data_list,name,sim_file_name):
    print('to snf')
    fused_network = snf.snf(data_list, K=20)
    print('finish owwwwwwww yeah')

    print('write to file')
    sim_f = open(sim_file_name,'w')

    write_name = '\t'.join(name)
    sim_f.write(f'name\t{write_name}\n')
    for k,row in enumerate(fused_network):
        row = '\t'.join(row.astype('str'))
        sim_f.write(f'{name[k]}\t{row}\n')

    sim_f.close()

def main_prots():    
    t_sim_file = '../data/t_sim_mat.txt'
    ppi_file = '../data/ppi_mat.txt'
    
    tsim_data, name= loader(file_name = '../data/protein_names.txt',file = t_sim_file)
    ppi_data, name = loader(ppi_file)
    
    prot_fused_sim = '../data/fused_protein_network.txt'
    
    fuser_writer([tsim_data, ppi_data], name,prot_fused_sim) 

def main_drugs():
    d_sim_file = '../data/d_sim_mat.txt'
    ddi_file = '../data/ddi_mat.txt'
    
    dsim_data, name= loader(file_name = '../data/drug_names.txt',file = d_sim_file)
    ddi_data, name = loader(ddi_file)
    
    drug_fused_sim = '../data/fused_drug_network.txt'
    
    fuser_writer([dsim_data, ddi_data], name,drug_fused_sim) 



#main_drugs()
main_prots()


























