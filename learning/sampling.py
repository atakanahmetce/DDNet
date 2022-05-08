# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 22:03:02 2022

@author: Sameitos
"""

#This file is to create samples to decrease complexity. {0,1} 120x120 interaction 
from tqdm import tqdm
import numpy as np
import re
import random
def get_names(file_name):
    names = []
    with open(file_name) as f:
        for row in f:
            names.append(row.strip('\n'))
    return names,set(names)

names,s_names = get_names('../data_folder/drug_names.txt')

def get_dd():
    
    aaa = set()
    data = []
    with open('../data_folder/dd_labels.csv') as f:
        for row in f:
            row = re.split(',',row.strip('\n'))
            data.append((row[0],row[1]))
            aaa.add(row[0])
            aaa.add(row[1])
    return set(data), aaa

label,aaa = get_dd()

dd = 0
for i in names:
    for j in names:
        if (i,j) in label:
    #if i in aaa:
            dd+=1
        
print(dd)

def sampler(names,s_names, label, sample = 10):
    

    
    for s in tqdm(range(sample)):
        c = 0
        f = open(f'../data_folder/samples/sample_idx_{s+1}.txt','w')
        addressed = set()    
        while c<50:
            idx = random.randint(0,len(label))
            it = label[idx]
            
            #for it in label:
            if it[0] in s_names and it[1] in s_names:
                if it[0] not in addressed and it[1] not in addressed:
                    addressed.add(names.index(it[0]))
                    addressed.add(names.index(it[1]))
                # f.write(f'{names.index(it[0])}\n')
                # f.write(f'{names.index(it[1])}\n')
                    c+=2
        for i in addressed:
            f.write(f'{i}\n')
    
        #init_len = len(addressed)
        # while c<25:
            
        #     idx_1 = random.randint(0,1155)
        #     idx_2 = random.randint(0,1155)
        #     #addressed.add(idx_1)
        #     #if c >= 25:
        #     #    if len(addressed)>init_len+20:
        #     #        f.write(f'{idx_1}\n')                
        #     #if idx_1 not in addressed and idx_2 not in addressed:
        #     if (idx_1,idx_2) in label:
        #         f.write(f'{idx_1}\n')
        #         f.write(f'{idx_2}\n')
        #         c+=1
        #         print(c)

        f.close()
#sampler(names,s_names, label)


