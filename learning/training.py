# -*- coding: utf-8 -*-
"""
Created on Wed May  4 19:35:04 2022

@author: Sameitos
"""

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
import numpy as np
from evaluation_functions import  evaluate_score
def loader():
    
    file_X = '../data_folder/training_data/X_1.txt'
    file_y = '../data_folder/training_data/y_1.txt'
    X, y = np.loadtxt(file_X), np.loadtxt(file_y)
    
    return X,y

def splitter(X,y):
        
    return train_test_split(X,y, test_size = 0.2)

def norm(X_train, X_test):
    
    scaler = Normalizer().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test

def train(X_train, X_test, y_train, y_test, ml_type = 'rf'):
    
    if ml_type == 'rf':
        model = RandomForestClassifier(n_estimators=100 ,n_jobs=-1,
                                    class_weight='balanced',
                                    criterion='gini')   
    elif ml_type == 'gb':
        model = GradientBoostingClassifier()
    
    model.fit(X_train,y_train)
    return model 


X,y = loader()
X_train,X_test,y_train, y_test = splitter(X,y)
X_train,X_test = norm(X_train, X_test)
model_rf = train(X_train, X_test, y_train, y_test)
score_train_rf =  evaluate_score(model_rf, X_train, y_train)
score_test_rf =  evaluate_score(model_rf, X_test, y_test)

model_gb = train(X_train, X_test, y_train, y_test, ml_type='gb')
score_train_gb =  evaluate_score(model_gb, X_train, y_train)
score_test_gb =  evaluate_score(model_gb, X_test, y_test)













