# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 11:53:01 2022

@author: Sameitos
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.cluster import Birch
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.feature_selection import SelectKBest

from tqdm import tqdm

#after node2vec inputs to clean outliers and get clusters to increase similarity

#feature selection based on chi2 and f_classification with labels from BIRCH clustering
#if clustering Silhouette score is low, then apply clustering can causes misleading

def pca(X,y = None,threshold = 0.95):
    
    '''
    Descrition:
        apply pca to data plot data if labels are active and always return first 2 column
    Parameters:
        X: {numpy array, list}, feature matrix
        y: {numpy array, list}, label matrix, if it is not None, plot is drawn
        threshold: {float}, minimum threshold that PCA fitted X holds variance ratio of 
            original X
    Return:
        new_X: Fitted data holds variance ratio of X as threshold allows
    '''
    
    pca = PCA(n_components = len(X[0]))
    X_2 = pca.fit_transform(X)
    if y is not None:
        fig = plt.figure(figsize = (8,8))
        plt.scatter(X_2[:,0], X_2[:,1] , c = y)
    variance = pca.explained_variance_ratio_
    
    total_var = 0
    last_idx = 2
    for k,v in enumerate(variance):
        total_var += v
        if total_var <= threshold:
            last_idx = k+1
            break   
    new_X = X_2[:,[i for i in range(last_idx)]]
    

    return new_X

def selectK(X,y,k = 256, method = 'chi2'):
    
    '''
    Select best k feature by the stats values and return matrix and selected
    feature column indices.
    
    '''

    if method == 'chi2':
        model = SelectKBest(chi2, k = k)

    elif method == 'f_classif':
        model = SelectKBest(f_classif, k = k)

    elif method == 'mutual_classif':
        model = SelectKBest(mutual_info_classif, k = k)

    new_arr = model.fit_transform(X,y)
    #selected_idx = model.get_support(indices = True)
    
    return new_arr

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

    data = pd.DataFrame(data)
    if function == 'quantile':
        q25 = data.quantile(0.25)
        q75 = data.quantile(0.75)
        intr_qr = q75-q25
        
        max_range = q75 + 2*intr_qr
        min_range = q25 - 1.25*intr_qr
        
        return data[~((data<min_range) | (data>max_range)).any(axis = 1)].to_numpy()



    elif function == 'z_score':
        z_score = np.abs(stats.zscore(data))
        return data[(z_score<threshold).all(axis = 1)].to_numpy()
    
def apply_birch(data,n):
    
    '''
    If data is too big, it can be handiful
    '''
    model = Birch(n_clusters = n)
    prediction = model.fit_predict(data)
    
    return prediction

def apply_kmeans(data,n = 2):
    
    model = KMeans(n_clusters = n)
    prediction = model.fit_predict(data)
    
    return prediction



