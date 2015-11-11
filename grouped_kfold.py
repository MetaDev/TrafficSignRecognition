# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 19:04:54 2015

@author: Rian
"""
from sklearn import cross_validation as cv

def grouped_kfold(amount, n_folds):
    kfold = cv.KFold(amount, n_folds = n_folds, shuffle = True)
    cv.KFold()