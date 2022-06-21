# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 11:53:01 2022

@author: Sameitos
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from scipy import stats

from sklearn.cluster import Birch
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.feature_selection import SelectKBest


#after node2vec inputs to clean outliers and get clusters to increase similarity

#feature selection based on chi2 and f_classification with labels from BIRCH clustering
#if clustering Silhouette score is low, then apply clustering can causes misleading

def pca(X,y = None):
    
    '''
    apply pca to data plot data if labels are active and always 
    return first 2 column
    '''
    
    pca = PCA(n_components = 2)
    X_2 = pca.fit_transform(X)
    if y is not None:
        fig = plt.figure(figsize = (8,8))
        plt.scatter(X_2[:,0], X_2[:,1] , c = y)
    return X_2[:,0], X_2[:,1]

def apply_birch(data,n):
    
    '''
    If data is too big, it can be handiful
    '''
    model = Birch(n_clusters = n)
    prediction = model.fit_predict(data)
    
    return prediction

def apply_kmeans(data,n):
    
    model = KMeans(n_clusters = n)
    prediction = model.fit_predict(data)
    
    return prediction

def selectK(method,X,y,k = 256):
    
    '''
    Select best k feature by the stats values and return matrix and selected
    feature column indices.
    
    '''
    model = SelectKBest(method, k = k)
    new_arr = model.fit_transform(X,y)
    selected_idx = model.get_support(indices = True)
    
    return new_arr, selected_idx

def common_indices(X,y,k):
    
    '''
    Apply chi2 and f_classification stats and select best k
    '''
    X_chi2,chi_idx = selectK(chi2, X, y, k)
    X_fclass,f_idx = selectK(f_classif, X, y, k)

    commons = []
    for i in range(k):
        if chi_idx[i] == f_idx[i]:
            commons.append(chi_idx[i])
    return commons

def outlier_detection(data, function = 'z_score',threshold = 3.0):
    
    '''
    Description:
        Detect outliers in data according to z score or quantile range
    Paramters:
        data: {numpy array}, parameter holds data
        function: {str}, {"z_score", "quantile"}, function for
            outlier detection
        threshold: {float}, threshold to hold samples give z_score under it 
    Return:
        new_data: {numpy array}, data after cleaning from outliers
        outliers: {numpy array}, outliers
    '''
    if function == 'quantile':
        q25 = data.quantile(0.25)
        q75 = data.quantile(0.75)
        intr_qr = q75-q25
        
        max_range = q75 + 2*intr_qr
        min_range = q25 - 1.25*intr_qr
        
        return data[~((data<min_range) | (data>max_range)).any(axis = 1)]



    elif function == 'z_score':
        z_score = np.abs(stats.zscore(data))
        return data[(z_score<threshold).all(axis = 1)]
    




