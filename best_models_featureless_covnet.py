# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 16:02:41 2015

@author: Rian
"""

import feature_validation as validation
import feature_extraction as extraction
import image_operations as operations
import data_loading as loader
import util
import numpy
import csv_output
from sklearn import lda, svm
from skimage import feature, color, exposure
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier


def flatten(resized, pixelsize):
    size1_F = len(resized[0])
    size2_F = len(resized[0][0])
    size3_F = pixelsize
    #♥print("size:",size1_F,",",size2_F)
    array = numpy.array(resized[0])
    reshaped = numpy.reshape(array,(size1_F*size2_F*size3_F))
    reshaped = [reshaped]
    for i in range(1,amount):
        size1 = len(resized[i])
        size2 = len(resized[i][0])
        size3 = pixelsize
        if(size1*size2*size3 != size1_F*size2_F*size3_F):print("size:",size1,",",size2)
        if(i%100 == 0):print(i,"/",amount)
        array = numpy.array(resized[i])
        a = numpy.reshape(array,(size1*size2*size3))
        reshaped = numpy.concatenate((reshaped,[a]),0)
    return reshaped
    
   
print("loading data...")
size = 32
images, labels, classes = loader.loadTrainingImagesPoleNumbersAndClasses()
amount = len(images)

print("resizing...")
resized = util.loading_map(lambda x : operations.cropAndResize(x, 0, size), images)
    
print("hsv...")
hsv = util.loading_map(color.rgb2hsv, resized)
hsv = flatten(hsv,3)

from sklearn import random_projection
print("model")
model2 = Pipeline([
    ("Multi-layer Perceptron", MLPClassifier(algorithm='sgd', hidden_layer_sizes=(200), random_state=1,learning_rate='constant',max_iter=300))
    ])
    
n_folds = 5

validation.validate_feature(hsv, labels, classes, model2, n_folds, False, False, True, True)

print('\a')
