# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 15:50:00 2015

@author: Rian
"""
import sklearn.cross_validation as cv
import numpy
from sklearn import lda
from sklearn import cross_validation
import score_calculation

def validate_feature(features, labels, classes, model, n_folds = 5, print_folds = True, print_absolute = True, print_logloss = True, verbose = False):
    kfold = cv.LabelKFold(labels, n_folds)
    if print_absolute: 
        score = cross_validation.cross_val_score(model, features, classes, cv=kfold)
        print("absolute scores")
        if print_folds: print("\tfolds:", score)
        print("\tmean:", score.mean(), "std:", numpy.std(score))
    if print_logloss: 
        scores = score_calculation.loglossKFold(features, classes, model, kfold, given_kfold = True, verbose = verbose)
        print("logloss scores")
        if print_folds: print("\tfolds",scores)
        print("\tmean:", numpy.mean(scores), "std:", numpy.std(scores))

def validate_feature_linear(features, labels, classes, n_folds = 5, print_folds = True, print_absolute = True, print_logloss = True):
    kfold = cv.LabelKFold(labels, n_folds)
    model = lda.LDA()
    if print_absolute: score = cross_validation.cross_val_score(model, features, classes, cv=kfold)
    if print_absolute: print("absolute scores")
    if print_folds: print("\tfolds:", score)
    if print_absolute: print("\tmean:", score.mean(), "std:", numpy.std(score))

    scores = score_calculation.loglossKFold(features, classes, model, kfold, given_kfold = True)
    if print_logloss: print("logloss scores")
    if print_folds: print("\tfolds",scores)
    if print_logloss: print("\tmean:", numpy.mean(scores), "std:", numpy.std(scores))