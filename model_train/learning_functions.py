# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 04:13:50 2022

@author: Sameitos
"""


import torch
import torch.nn as nn

from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV,RepeatedKFold,RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.metrics import confusion_matrix

import numpy as np

def apply_cv(model, parameters, X_train, y_train):
    
    cv = RepeatedStratifiedKFold(n_splits=5,n_repeats = 2,random_state= 56)
    clf = RandomizedSearchCV(model, parameters,n_iter = 30, n_jobs = -1, cv = cv, scoring = 'f1',
                             random_state = 56)
    clf.fit(X_train,y_train)
    
    return clf.best_estimator_


class Net(nn.Module):
    
    def __init__(self, in_size, hid_size, layer_size, p, out_size = 1,):
        super(Net, self).__init__()
        
        self.layer_size = layer_size
        
        self.lf = nn.Linear(in_size, hid_size)
        self.lm = nn.Linear(hid_size,hid_size)
        self.lo = nn.Linear(hid_size, out_size)
        
        self.dropout = nn.Dropout(p)
        self.softmax = nn.Softmax(dim = 0)
        self.leaky = nn.LeakyReLU()
        
        
        
    def forward(self,X):
        
        first_layer = self.dropout(self.leaky(self.lf(X)))
        
        layer = self.leaky(self.lm(first_layer))
        for i in range(3,self.layer_size):
            layer = self.leaky(self.lm(layer))
        
        pre_act = self.lo(self.dropout(layer))
        out_layer = self.softmax(pre_act)

        return out_layer

def deep_model(X, y, deep_kwargs, isValid = False):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#Device
    print(torch.cuda.get_device_name(0))
    
    lr = deep_kwargs['lr'] # learning rate
    eps = deep_kwargs['eps'] # epsilon
    epochs = deep_kwargs['epochs'] # # of epoch
    p = deep_kwargs['p'] # dropout_fraction
    layer_size = deep_kwargs['n_layer'] # # of layer data proceeds. First and last layers should be counted. 
    hid_size = deep_kwargs['hidden_layer_size'] # # of dimensions in hidden layer sizez.
    
    #Change type of matrices
    if isinstance(X,np.ndarray):
        X = torch.from_numpy(X).to(device)
        y = torch.from_numpy(y).reshape(len(X),1).to(device)

    #Set objects
    model = Net(in_size = X.shape[1], hid_size = hid_size, layer_size = layer_size, p = p).to(device)
    criterion = nn.BCELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=eps)
    
    #training
    
    if isValid:
        
        n_splits = 10
        rkf = RepeatedKFold(n_splits = n_splits, n_repeats = 2, random_state = 10000)
        min_valid_loss = float('inf')
        returned_model = None
        for epoch in tqdm(range(epochs)):
            train_loss = 0.0
            valid_loss = 0
            for train_idx,test_idx in rkf.split(X):
    
                model.train()
                optimizer.zero_grad()
                
                prediction = model(X[train_idx].float())            
                loss = criterion(prediction, y[train_idx].float())
                loss.backward()
                optimizer.step()
                train_loss += loss.item()           
                
                model.eval()
                with torch.no_grad():
                    test_pred = model(X[test_idx].float())
                    loss = criterion(test_pred,y[test_idx].float())
                    valid_loss += loss.item()
                    
            cm_predict = np.where(prediction.cpu().detach().numpy()<0.5/len(X[train_idx]),0,1)
            TN, FP, FN, TP = confusion_matrix(y[train_idx].cpu().detach().numpy(),cm_predict).ravel()    
            
            if min_valid_loss>valid_loss/(n_splits-1):
                min_valid_loss = valid_loss/(n_splits-1)
                
                test_pred = np.where(test_pred.cpu().detach().numpy()<0.5/len(X[test_idx]),0,1)
                TN, FP, FN, TP = confusion_matrix(y[test_idx].cpu().detach().numpy(),
                                                    test_pred).ravel()
                
                
            
                returned_model = model        
        
        optimizer.zero_grad()        
        return returned_model
    
    else:
        
        for epoch in tqdm(range(epochs)):
    
            optimizer.zero_grad()
            
            prediction = model(X.float())            
            loss = criterion(prediction, y.float())
            loss.backward()
            optimizer.step()
            
            cm_predict = np.where(prediction.cpu().detach().numpy()<0.5/len(X),0,1)
            TN, FP, FN, TP = confusion_matrix(y.cpu().detach().numpy(),cm_predict).ravel()    
            
            
        optimizer.zero_grad()        
        return model




def train(X_train, y_train, ml_type = 'rf', isValid = False, 
          epochs = 200, # # of epochs
          p = 0.4,# dropout fraction
          lr = 0.0007,# learning rate
          eps = 10**-5,# epsilon
          n_layer = 5,# # of layer
          hidden_layer_size = 128, #dimensions of hidded layers
          ):
    
    if ml_type == 'rf':
        if isValid:
            model = RandomForestClassifier()
            parameters = dict(
                n_estimators = [int(i) for i in np.linspace(10,50,num=5)],
                max_features = ["auto","sqrt","log2"],
                bootstrap = [True, False],
                min_samples_split = np.linspace(0.05, 1.0, 10, endpoint=True)
                )
            return apply_cv(model,parameters, X_train, y_train)
        else:
            model = RandomForestClassifier(n_estimators=100 ,n_jobs=-1,
                                    class_weight='balanced',
                                    criterion='gini')
            model.fit(X_train,y_train)
            return model
    elif ml_type == 'gb':
        if isValid:
            model = GradientBoostingClassifier()
            parameters = dict(
                loss = ['deviance','exponential'],
                learning_rate = np.linspace(0.01,0.15,num = 5),
                criterion = ['mse','friedman_mse'],
                #max_depth = np.arange(3,10),
                )
            return apply_cv(model,parameters, X_train, y_train)
        else:
            model = GradientBoostingClassifier()
            model.fit(X_train,y_train)
            return model 
                    

    elif ml_type == 'deep':
        deep_param = dict(epochs = epochs,
                        p = p,
                        lr = lr,
                        eps = eps,
                        n_layer = n_layer,
                        hidden_layer_size = hidden_layer_size)
        
        return deep_model(X_train, y_train, deep_kwargs = deep_param, isValid = False)
    





























































