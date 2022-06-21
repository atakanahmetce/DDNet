# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 12:25:49 2022

@author: Sameitos
"""

from model_train import get_data, train, evaluate_score

#loading and preprocessing training data
fx = 'data/training_data/X.txt' #feature matrix file name
fy = 'data/training_data/y.txt' #label matrix file name
X_train,X_test,y_train,y_test = get_data(fx,fy, ratio = 0.2)

#train a model
model = train(X_train,y_train, ml_type = 'deep', epochs = 50)

#get test scores
score_test = evaluate_score(model, X_test, y_test, isDeep = True)

#to save score 
import pandas as pd
test_scores = [score_test]
data = pd.DataFrame.from_dict(test_scores)
data.to_csv('../scores/test_scores.csv',index = False, header = True)