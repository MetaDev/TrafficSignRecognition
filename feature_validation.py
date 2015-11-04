# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 15:50:00 2015

@author: Rian
"""
import sklearn.cross_validation as cv
import numpy
from sklearn import linear_model
from sklearn import cross_validation
import score_calculation

def validate_feature_linear(features, classes, n_folds = 5, print_folds = True):
    amount = len(features)
    kfold = cv.KFold(amount, n_folds = n_folds, shuffle = True)
    model = linear_model.LogisticRegression(class_weight='auto')
    score = cross_validation.cross_val_score(model, features, classes, cv=kfold)
    print("absolute scores")
    if print_folds: print("\tfolds:", score)
    print("\tmean:", score.mean(), "std:", numpy.std(score))

    model = linear_model.LogisticRegression(class_weight='auto')
    scores = score_calculation.loglossKFold(features, classes, model, kfold, given_kfold = True)
    print("logloss scores")
    if print_folds: print("\tfolds",scores)
    print("\tmean:", numpy.mean(scores), "std:", numpy.std(scores))