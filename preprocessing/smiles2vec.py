# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 11:52:18 2022

@author: Sameitos
"""

#convert smiles to binary data by rdkit
import json,re
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from tqdm import tqdm



def featurizer(data, name = False):
    
    '''
    Description:
        change smiles data to bit data
        
    Parameters:
        data: {dict,list}, hold the SMILES data of drugs. If data is dict,
            then drug names are key and SMILES are value. If list, first column
            is name of drug or not respect to "name" and second column is 
            SMILES strings.
        name: {bool}, default = False, if True, first column of data is names of 
            drug.
    Return:
        bit_data: {dict,list}, hold the SMILES data of drugs. If data is dict,
            then drug names are key and SMILES are value of bit_data. If data is list,
            first column is name of drug or not respect to "name"
    '''
    
    
    
    if type(data) == dict:
        
        non_converts = []
        bit_data = {}
        for d in tqdm(data.keys()):
            try:
                mol = Chem.MolFromSmiles(data[d])
                fp = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits = 1024)
                bit_data[d] = DataStructs.cDataStructs.BitVectToText(fp)
            except:
                print('Not all SMILES are converted to bit data: ', d)
        return bit_data
    
    else:
        
        if name:
            non_converts = []
            bit_data = []
            for d in tqdm(data):
                try:
                    mol = Chem.MolFromSmiles(d[1])
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits = 1024)
                    bit_data.append([d,DataStructs.cDataStructs.BitVectToText(fp)])
                except:
                    print('Not all SMILES are converted to bit data: ', d[0])
                    non_converts.append(d[0])
            return bit_data
        
        else:
            non_converts = []
            bit_data = []
            for d in tqdm(data):
                try:
                    mol = Chem.MolFromSmiles(d)
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits = 1024)
                    bit_data.append(DataStructs.cDataStructs.BitVectToText(fp))
                except:
                    print('Not all SMILES are converted to bit data: ', d)
                    non_converts.append(d)
            return bit_data
    
    
    