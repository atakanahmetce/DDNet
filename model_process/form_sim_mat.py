# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 11:56:25 2022

@author: Sameitos
"""

import numpy as np
from scipy.spatial.distance import rogerstanimoto, jaccard, cosine

#form similarity matrix by drug binary data

def vec2sim(data, function = 'cos', name = False):
    
    '''
    Description:
        change smiles data to bit data
        
    Parameters:
        data: {dict,list}, hold the SMILES data of drugs. If data is dict,
            then drug names are key and SMILES are value. If list, first column
            is name of drug or not respect to "name" and second column is 
            SMILES strings
        name: {bool}, default = False, if True, first column of data is names of 
            drug
        function: {str}, {cos, jaccard, tanimoto}, default = "cos", similarity matrix 
            function
    Return:
        sim_arr: {numpy array}, hold the SMILES data of drugs
        names: {numpy array}, if name = True or data is dict, given in another numpy array
    '''
    
    if function == 'cos':
        f = cosine
    elif function == 'jaccard':
        f = jaccard
    elif function == 'tanimoto':
        f = rogerstanimoto
    
    if type(data) == dict:
        names = []
        sim_arr = np.zeros((len(data.keys()),len(data.keys())))
        for k,i in enumerate(data.keys()):
            for t,j in enumerate(data.keys()):
                sim_arr[k,t] = 1 - f(data[i], data[j])
            names.append(i)
        return sim_arr, np.array(names)
    else:
        
        if name:
            names = []
            sim_arr = np.zeros((len(data),len(data)))
            for k,i in enumerate(data):
                for t,j in enumerate(data):
                    sim_arr[k,t] = 1 - f(data[i][1:], data[j][1:])
                names.append(i[0])
            return sim_arr, np.array(names)
        
        else:
            
        
            sim_arr = np.zeros((len(data.keys()),len(data.keys())))
            for k,i in enumerate(data):
                for t,j in enumerate(data):
                    sim_arr[k,t] = 1 - f(data[i], data[j])
                
            return sim_arr
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
