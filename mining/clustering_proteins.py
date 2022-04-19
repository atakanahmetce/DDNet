# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 12:11:19 2022

@author: Sameitos
"""

import re, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.preprocessing import normalize as norm
from sklearn.decomposition import PCA

from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering as ACL
from sklearn.metrics import silhouette_score

feature = 'ctriad'
#loading data
def loader(file_name):
    
    data = []
    addressed = set()
    name = []
    with open(file_name) as f:
        for row in f:
            row = re.split('\t',row.strip('\n'))
            if not row[0] in addressed:
                addressed.add(row[0])
                data.append(np.array(row[1:],dtype = float))
                name.append(row[0])

    data = pd.DataFrame(data)
    data['name'] = name
    data = data.set_index('name')
    print(data.shape)
    return data,name

data,name = loader('../data_backup/target_' + feature + '.txt')

#draw plot
def draw_FL(X,y,c_type):
    
    pca = PCA(n_components = 2)
    X_2 = pca.fit_transform(X)
    print(pca.explained_variance_ratio_)
    
    fig = plt.figure(figsize = (8,8))
    plt.scatter(X_2[:,0], X_2[:,1] , c = y)
    fig.savefig('../figs/' + c_type + '_' + feature + '_plot.svg',bbox_inches='tight')

#eleminate outliers qr method
def range_outlier(data):
    q25 = data.quantile(0.25)
    q75 = data.quantile(0.75)
    intr_qr = q75-q25
    
    max_range = q75 + 2*intr_qr
    min_range = q25 - 1.25*intr_qr
    
    return data[~((data<min_range) | (data>max_range)).any(axis = 1)]

'''qr_data = range_outlier(data)'''

#eleminate z-score

def z_outlier(data,threshold):
    z_score = np.abs(stats.zscore(data))
    return data[(z_score<threshold).all(axis = 1)]

data = z_outlier(data, threshold = 3)

#Normalizing
normal_data = norm(data, norm = 'l2', axis = 0)

def calcs(data,model):
    
    preds = model.fit_predict(data)
    return silhouette_score(data, preds), preds

def calculate_silhouette(data, n_clusters, random_state = 36, func = 'kmeans'):
    scores,preds = [],[]
    for n in n_clusters:
        if func == 'kmeans':
            score, pred = calcs(data,KMeans(n_clusters = n, random_state = random_state))
            draw_FL(X = data, y = pred, c_type = 'kmeans_' + str(n))
        elif func == 'gmm':
            score, pred = calcs(data,GMM(n_components=n, random_state = random_state))
            draw_FL(X = data, y = pred, c_type = 'gmm_' + str(n))
        scores.append(score)
        preds.append(pred)
        
    best_n, best_preds = n_clusters[scores.index(max(scores))], preds[scores.index(max(scores))]
    fig = plt.figure(figsize = (8,8))
    plt.plot(n_clusters, scores)
    plt.ylabel('silhouette score')
    plt.xlabel('# of clusters')
    fig.savefig('../figs/' + func + '_' + str(best_n) + '_' + feature + '_silhouette.svg',bbox_inches='tight')
    
    return max(scores),best_preds

def write(file_name, data, pred):
    num_data = data.to_numpy()
    
    for k,i in enumerate(data.index):
        f = open(file_name[:-4] + '_' + str(pred[k]) + '.txt','a')
        row = '\t'.join(np.array(num_data[k], dtype = str))
        f.write(f'{i}\t{row}\n')
    f.close()

best_score_kmeans, kmeans_preds = calculate_silhouette(data, n_clusters = [2,3,4,5,6,7,8])
best_score_gmm,gmm_preds = calculate_silhouette(data, n_clusters = [2,3,4,5,6,7,8], func = 'gmm')

if best_score_kmeans>best_score_gmm:
    best_pred = kmeans_preds
else:
    best_pred = gmm_preds


write(file_name = '../data/fined_' + feature + '_matrix.txt',data = data, pred = best_pred)



''' 
Samples

#kmeans for proteins
#y_Kpred = KMeans(n_clusters = 4, random_state = 56).fit_predict(normal_data)
#draw_FL(X = data, y = y_Kpred, c_type = 'kmeans_4')

#gmm clustering
#y_Gpred = GMM(n_components=4, random_state=56).fit_predict(normal_data)
#draw_FL(X = data, y = y_Gpred, c_type = 'gmm_4')
'''

'''--Will not be considered--
#hierarchical clustering
y_Hpred = ACL(n_clusters = 4, linkage = 'single').fit_predict(normal_data)
draw_FL(X = data, y = y_Hpred, c_type = 'acl_single_4')

'''
