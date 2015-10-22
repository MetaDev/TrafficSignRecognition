# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 13:09:49 2015

@author: Rian
"""
import numpy
from sklearn import cross_validation

def logloss(xtrain, ytrain, xtest, ytest, model):
    classes = numpy.unique(ytrain)
    model.fit(xtrain,ytrain)
    probabilities =  model.predict_log_proba(xtest)
    total = 0
    for i in range(len(ytest)):
        for j in range(len(classes)):
            if classes[j] == ytest[i]:
                total += probabilities[i, j]
    return - total / len(ytest)
   
def loglossKFold(x, y, model, n_folds = 8):
    kfold = cross_validation.KFold(len(x), n_folds = n_folds, shuffle = True)
    scores = []
    for train_index, test_index in kfold:
        trainFeatures = [x[i] for i in train_index]
        trainClasses  = [y[i] for i in train_index]
        testFeatures  = [x[i] for i in test_index]
        testClasses   = [y[i] for i in test_index]
        scores.append(logloss(trainFeatures, trainClasses, testFeatures, testClasses, model))
    return scores