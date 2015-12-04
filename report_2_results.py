# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 00:37:43 2015

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
print("hsv...")
hsv = util.loading_map(color.rgb2hsv, resized)
print("luv...")
luv = util.loading_map(color.rgb2luv, resized)
print("grayscaling...")
grayscaled = util.loading_map(color.rgb2gray, resized)
print("edges...")
edges = util.loading_map(feature.canny, grayscaled)

print("brightness features")
brightness = util.loading_map(extraction.calculateDarktoBrightRatio, resized)
print("luv features")
luv_features = util.loading_map(lambda x: extraction.pixel_features(x, 11), luv)
print("hog features")
hog = util.loading_map(lambda x: feature.hog(x, orientations = 6, pixels_per_cell = (12,12), cells_per_block=(8,8), normalise = True), grayscaled)

hog_l_b = numpy.concatenate((hog, luv_features, brightness), 1)

n_folds = 10

LR = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial')
SSLR = Pipeline([
    ("standard scaler", StandardScaler()),   
    ("logistic regression", LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial'))
    ])
RSLR = Pipeline([
    ("robust scaler", RobustScaler()),   
    ("logistic regression", LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial'))
    ])
SSPCALR = Pipeline([
    ("standard scaler", StandardScaler()),
    ("pca", PCA(n_components = 350))
    ("logistic regression", LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial'))
    ])
SSLDALR = Pipeline([
    ("standard scaler", StandardScaler()),
    ("lda", lda.LDA(n_components = 80))
    ("logistic regression", LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial'))
    ])
    
for (model, model_name) in (LR, "LR"),(SSLR, "SSLR"), (RSLR, "RSLR"), (SSPCALR, "SSPCALR"), (SSLDALR, "SSLDALR"):
    print("----------")    
    print(model_name)    
    print("----------")    
    for (features, feature_name) in (luv_features, "LUV"), (hog, "HOG"), (hog_l_b, "HOG + LUV + B"):
        print("Evaluating", feature_name)
        validation.validate_feature(features, labels, classes, model, n_folds, False, False, True, True)
print('\a')