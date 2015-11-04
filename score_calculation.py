# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 13:09:49 2015

@author: Rian
"""
import numpy
from sklearn import cross_validation

limit = 10 ** -15

def logloss(xtrain, ytrain, xtest, ytest, model):
    classes = numpy.unique(ytrain)
    model.fit(xtrain,ytrain)
    probabilities =  model.predict_proba(xtest)
    total = 0
    for i in range(len(ytest)):
        found = False
        for j in range(len(classes)):
            if classes[j] == ytest[i]:
                found = True
                if(probabilities[i, j] < limit): probabilities[i, j] = limit
                total += numpy.log(probabilities[i, j])
        if not found:
            total += numpy.log(limit)
    return - total / len(ytest)
   
def loglossKFold(x, y, model, n_folds = 8, given_kfold = False):
    if given_kfold:
        kfold = n_folds
    else:
        kfold = cross_validation.KFold(len(x), n_folds = n_folds, shuffle = True)
    scores = []
    for train_index, test_index in kfold:
        trainFeatures = [x[i] for i in train_index]
        trainClasses  = [y[i] for i in train_index]
        testFeatures  = [x[i] for i in test_index]
        testClasses   = [y[i] for i in test_index]
        scores.append(logloss(trainFeatures, trainClasses, testFeatures, testClasses, model))
    return scores