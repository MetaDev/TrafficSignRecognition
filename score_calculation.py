# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 13:09:49 2015

@author: Rian
"""
import numpy
from sklearn import cross_validation
from sklearn import metrics
import sys

limit = 10 ** -15

def logloss(xtrain, ytrain, xtest, ytest, model):
    classes = numpy.unique(ytrain)
    model.fit(xtrain,ytrain)
    probabilities =  model.predict_proba(xtest)
    total = 0
    for i in range(len(ytest)):
        found = False
        class_probabilities = probabilities[i, :]
        class_probabilities = class_probabilities / numpy.linalg.norm(class_probabilities)
        for j in range(len(classes)):
            if classes[j] == ytest[i]:
                found = True
                if(class_probabilities[j] < limit): class_probabilities[j] = limit
                if(class_probabilities[j] > 1-limit): class_probabilities[j] = 1-limit
                total += numpy.log(class_probabilities[j])
        if not found:
            total += numpy.log(limit)
    return - total / len(ytest)
   
def loglossKFold(x, y, model, n_folds = 8, given_kfold = False, verbose = False):
    if given_kfold:
        kfold = n_folds
    else:
        kfold = cross_validation.KFold(len(x), n_folds = n_folds, shuffle = True)
    scores = []
    index = 0
    for train_index, test_index in kfold:
        if verbose:        
            index += 1
            sys.stdout.write("\r%d" % index)
            sys.stdout.flush()
        trainFeatures = [x[i] for i in train_index]
        trainClasses  = [y[i] for i in train_index]
        testFeatures  = [x[i] for i in test_index]
        testClasses   = [y[i] for i in test_index]
        scores.append(logloss(trainFeatures, trainClasses, testFeatures, testClasses, model))
    return scores