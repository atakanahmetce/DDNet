# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 13:28:29 2022

@author: Sameitos
"""

import re
import numpy as np

place = 'protein_0'

def loader(file,eps = 0):
    
    data ={}
    with open(file) as f:
        for row in f:
            row = re.split('\t',row.strip('\n'))
            if row[0] != 'name':
                data[row[0]] = np.array(row[1:], dtype = float) + eps
            else:
                data[row[0]] = np.array(row[1:])
    return data

def idxer(joints, names):
    
    joint_idx = []
    for i in range(len(names)):
        if names[i] in joints:
            joint_idx.append(i)

    return joint_idx

def builder(idx,sim_data):
    new_sim_data = {}
    for i in idx:
        if sim_data['name'][i] in sim_data.keys():
            new_sim_data[sim_data['name'][i]] = sim_data[sim_data['name'][i]][idx]
    new_sim_data['name'] = sim_data['name'][idx]
    
    return new_sim_data

def equalar(sim, inc):
    
    inc_list = {}
    for k,i in enumerate(inc["name"]):
        for t,j in enumerate(inc['name']):
            inc_list[(i,j)] = inc[i][t]
    
    eq_inc = np.zeros((len(sim)-1,len(sim)-1))
    for k, i in enumerate(sim['name']):
        for t, j in enumerate(sim['name']):
            eq_inc[k,t] = inc_list[(i,j)]
    
    new_interaction = {}
    for k,row in enumerate(eq_inc):
        new_interaction[sim['name'][k]] = row

    new_interaction['name'] = sim['name']
    return new_interaction

def write(f,data):
    
    names = '\t'.join(data['name'])
    f.write(f'name\t{names}\n')
    for n in data['name']:
        row = '\t'.join(np.array(data[n], dtype = str))
        f.write(f'{n}\t{row}\n')

def main_prots():
    
    tsim_file = '../data_backup/' + place + '/' + 't_sim_mat.txt'
    ppi_file = '../data_backup/' + place + '/' + 'ppi_mat.txt'
    
    tsim_data = loader(tsim_file)
    ppi_data = loader(ppi_file, eps = 0.001)
    
    t_prot = tsim_data.keys()
    p_prot = ppi_data.keys()
    
    joint_prots = set(p_prot).intersection(set(t_prot))
    
    Tjoint_idx = idxer(joint_prots, tsim_data['name'])
    Pjoint_idx = idxer(joint_prots, ppi_data['name'])
    
    new_tsim_data = builder(Tjoint_idx,tsim_data) 
    new_ppi_data = builder(Pjoint_idx,ppi_data)     
    
    new2_ppi_data = equalar(new_tsim_data, new_ppi_data)
    tf = open('../data/' + place + '/' + 't_sim_mat.txt','w')
    pf = open('../data/' + place + '/' + 'ppi_mat.txt','w')
    write(tf,new_tsim_data)
    write(pf,new2_ppi_data)
    
def main_drugs():
    
    dsim_file = '../data_backup/' + place + '/' + 'd_sim_mat.txt'
    ddi_file = '../data_backup/' + place + '/' + 'ddi_mat.txt'
    
    dsim_data = loader(dsim_file)
    ddi_data = loader(ddi_file)
    
    d_drug = dsim_data.keys()
    dd_drug = ddi_data.keys()
    
    joint_drugs = set(d_drug).intersection(set(dd_drug))
    
    Djoint_idx = idxer(joint_drugs, dsim_data['name'])
    DDjoint_idx = idxer(joint_drugs, ddi_data['name'])
    
    new_dsim_data = builder(Djoint_idx,dsim_data) 
    new_ddi_data = builder(DDjoint_idx,ddi_data)     
    
    new2_ddi_data = equalar(new_dsim_data, new_ddi_data)
    df = open('../data/' + place + '/' + 'd_sim_mat.txt','w')
    ddf = open('../data/' + place + '/' + 'ddi_mat.txt','w')
    write(df,new_dsim_data)
    write(ddf,new2_ddi_data)


main_drugs()
main_prots()






