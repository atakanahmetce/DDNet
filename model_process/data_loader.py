# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 15:11:23 2022

@author: Sameitos
"""


import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.metrics import confusion_matrix

from .data_clean import pca,selectK



def loader(file_X, file_y):
    
    X, y = np.loadtxt(file_X), np.loadtxt(file_y)
    
    return X,y

def splitter(X,y, ratio):
        
    return train_test_split(X,y, test_size = ratio)

def norm(X_train, X_test):
    
    scaler = Normalizer().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test


def get_data(file_X,file_y, ratio = 0.2, normal = True, feat_ext = None,
             threshold = 0.95,
             k = 100):
    
    """
    Description:
        Function loads feature and label matrix and splitting to train and 
        test set. Preprocessing (normalization, feature extraction) can be 
        done respect to parameters.
    Parameters:
        file_X: {str}, file name of feature matrix
        file_y: {str}. file name of label matrix
        ratio: {float}, default = 0.2, ratio of test size to whole data size
        normal: {bool}, default = True, If True, feature based normalization
            is done
        feat_ext: {str}, {'pca', 'chi2', 'f_classif', 'mutual_info_classif'},
            default = None, If None, no extraction is done.
        threshold: {float}, default = 0.95, threshold variance ratio of PCA 
            matrix. Until limit is satisfied, dimensions are returned.
        k: {int}, default = 100, select best k features according to values.
    Return:
        X_train: {numpy array}, train feature matrix
        X_test: {numpy array}, test feature matrix
        y_train: {numpy array}, train label matrix
        y_test: {numpy array}, test label matrix
    
    """
    
    X,y = loader(file_X,file_y)
    
    if feat_ext is not None:
        if feat_ext == 'pca':
            X = pca(X,threshold = threshold)
        else:
            X = selectK(X, y, k = k, method = feat_ext)
    
    X_train,X_test,y_train, y_test = splitter(X,y, ratio)
    
    if normal:
        X_train,X_test = norm(X_train, X_test)
    
    return X_train,X_test,y_train, y_test 
        
    
