# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 15:25:10 2015

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
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn import random_projection
from sklearn.ensemble import RandomForestClassifier
from sklearn import feature_selection
   
#preloading
print("loading data...")
size = 100
images, labels, classes = loader.loadTrainingImagesPoleNumbersAndClasses()
amount = len(images)

print("resizing...")
resized = util.loading_map(lambda x : operations.cropAndResize(x, 0, size), images)
print("grayscaling...")
grayscaled = util.loading_map(color.rgb2gray, resized)

n_folds = 10

model = Pipeline([
    ("standard scaler", StandardScaler()),   
    ("logistic regression", LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial'))
    ])

for cpb in range(2, 11):
    ppc = int(100 / cpb)
    print("hog_8_", ppc, "_", cpb, " features")
    hog = util.loading_map(lambda x: feature.hog(x, orientations = 8, pixels_per_cell = (ppc, ppc), cells_per_block=(cpb, cpb), normalise = True), grayscaled)
    print(numpy.shape(hog))
    validation.validate_feature(hog, labels, classes, model, n_folds, False, False, True, True)

#hog_8_12_8 is optimal, optimizing orientations:
for orientations in range(10, 11):
    print("hog_", orientations, "_12_8 features")
    hog = util.loading_map(lambda x: feature.hog(x, orientations = orientations, pixels_per_cell = (12,12), cells_per_block=(8,8), normalise = True), grayscaled)
    print(numpy.shape(hog))
    validation.validate_feature(hog, labels, classes, model, n_folds, False, False, True, True)

#retry optimizing cpb using 6 orientations
for cpb in range(6, 11):
    ppc = int(100 / cpb)
    print("hog_6_", ppc, "_", cpb, " features")
    hog = util.loading_map(lambda x: feature.hog(x, orientations = 6, pixels_per_cell = (ppc, ppc), cells_per_block=(cpb, cpb), normalise = True), grayscaled)
    print(numpy.shape(hog))
    validation.validate_feature(hog, labels, classes, model, n_folds, False, False, True, True)
  
#no better result, ok!
print('\a')