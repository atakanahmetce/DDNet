# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 10:45:40 2022

@author: Sameitos
"""

#This file to select proteins and drugs that have 0.5 or higher similarity value. Other values 
#are to be sacked and not to be used in interaction prediction

import re, os
import numpy as np


def loader(sim_file):
    
    data = {}
    name = None
    with open(sim_file) as f:
        for row in f:
            row = re.split('\t',row.strip('\n'))
            if row[0] != 'name':
                data[row[0]] = row[1:]
            else:
                name = row[1:]

    return data,name

def filter(data,name,sim_lim):
    f = open('test_drug.txt','w')
    dict_data = {}

    for k,i in enumerate(name):
        for t,j in enumerate(name):
            f.write(f'{i},{j}\t{float(data[i][t])}\n')
            if float(data[i][t])>=0.1:
                dict_data[(i,j)] = float(data[i][t])
    
    f.close()
    return dict_data



# ttSim_file = '../data/fused_protein_network.txt'
# ttSim,tname = loader(ttSim_file)
# tt05 = filter(ttSim, name = tname, sim_lim = 0.5)

ddSim_file = '../data/fused_drug_network.txt'
ddSim,dname = loader(ddSim_file)
dd05 = filter(ddSim, name = dname, sim_lim = 0.5)



