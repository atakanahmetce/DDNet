# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 22:30:01 2022

@author: Sameitos
"""

import re

def loader(file_name):
    
    data = {}
    with open(file_name) as f:
        for row in f:
            row = re.split('\t',row.strip('\n'))
            data[(row[0],row[-1])] = row[1]
    
    return data


def writer(file_name):
    
    with open()