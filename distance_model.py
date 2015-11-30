# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 15:35:18 2015

@author: Rian
"""

import numpy
from numpy import linalg
from scipy import stats
import sys

class DistanceModel: 
    
    def __init__(self, verbose = False):
        self.verbose = verbose
    
    def fit(self, X, Y):
        X = numpy.array(X)
        Y = numpy.array(Y)
        self.classes = numpy.unique(Y)
        nr_of_features = numpy.shape(X)[1]
        nr_of_classes = len(self.classes)
        self.total_means = numpy.mean(X, 0)
        self.total_stds = numpy.std(X, 0)
        self.total_skews = stats.skew(X, 0)
        self.inside_means = numpy.zeros((nr_of_classes, nr_of_features))
        self.inside_stds = numpy.zeros((nr_of_classes, nr_of_features))
        self.inside_skews = numpy.zeros((nr_of_classes, nr_of_features))
        self.class_probabilities = numpy.zeros((nr_of_classes))
        for i in range(len(self.classes)):
            c = self.classes[i]
            inside = X[Y == c]
            self.inside_means[i,:] = numpy.mean(inside, 0)
            self.inside_stds[i,:] = numpy.std(inside, 0)
            self.inside_skews[i,:] = stats.skew(inside, 0)
            self.class_probabilities[i] = len(inside) / len(Y)
            
    def predict_proba(self, X):
        X = numpy.array(X)
        probabilities = numpy.zeros((len(X), len(self.classes)))
        for i in range(len(X)):
            if self.verbose:
                sys.stdout.write("\rpredicting %d / %d" % (i, len(X)))
                sys.stdout.flush()
            for j in range(len(self.classes)):
                value_for_c_proba = stats.pearson3.pdf(X[i,j], self.inside_skews[j], self.inside_means[j], self.inside_stds[j] * 3)
                value_for_all_proba = stats.pearson3.pdf(X[i,j], self.total_skews[j], self.total_means[j], self.total_stds[j] * 3)
                class_proba = self.class_probabilities[j]
                probabilities[i,j] = numpy.sqrt(numpy.sum(numpy.square(value_for_c_proba * class_proba / value_for_all_proba)))
            probabilities[i, :] = probabilities[i, :] / (linalg.norm(probabilities[i, :]))
        return probabilities
                
    def predict(self, X):
        probabilities = self.predict_proba(X)
        predictions = numpy.empty((len(probabilities)), dtype = numpy.str)
        for i in range(len(probabilities)):
            predictions[i] = self.classes[numpy.argmax(probabilities[i, :])]
        return predictions
            
            