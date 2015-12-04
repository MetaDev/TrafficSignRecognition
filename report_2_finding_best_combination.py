# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 23:26:40 2015

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
print("hsv 11 + std")
hsv_11_std = util.loading_map(lambda x: extraction.split_image_features(
    lambda y : extraction.color_features(y, mean = True, std = True), 11, x), hsv)
print("luv features")
luv_features = util.loading_map(lambda x: extraction.pixel_features(x, 11), luv)
print("hog features")
hog = util.loading_map(lambda x: feature.hog(x, orientations = 6, pixels_per_cell = (12,12), cells_per_block=(8,8), normalise = True), grayscaled)
print("corner features")
corners = util.loading_map(lambda x: extraction.pixel_features(numpy.sqrt(feature.corner_shi_tomasi(x, sigma=8)), 8), grayscaled)


n_folds = 10

model = Pipeline([
    ("standard scaler", StandardScaler()),   
    ("logistic regression", LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial'))
    ])

print("Evaluating brightness features")
validation.validate_feature(brightness, labels, classes, model, n_folds, False, False, True, True)
print("Evaluating luv features")
validation.validate_feature(luv_features, labels, classes, model, n_folds, False, False, True, True)
print("Evaluating hog features")
validation.validate_feature(hog, labels, classes, model, n_folds, False, False, True, True)
print("Evaluating corner features")
validation.validate_feature(corners, labels, classes, model, n_folds, False, False, True, True)
print("Evaluating HSV_11_std features")
validation.validate_feature(numpy.array(hsv_11_std), labels, classes, model, n_folds, False, False, True, True)

#we combine the 2 best ones
hog_hs = numpy.concatenate((hog, hsv_11_std), 1)
print("Evaluating HOG+HS features")
print(numpy.shape(hog_hs))
validation.validate_feature(hog_hs, labels, classes, model, n_folds, False, False, True, True)
#we add the 3rd best one
hog_l = numpy.concatenate((hog, luv_features), 1)
print("Evaluating HOG+L features")
print(numpy.shape(hog_l))
validation.validate_feature(hog_l, labels, classes, model, n_folds, False, False, True, True)
#we add the 4th best one
hog_hs_b = numpy.concatenate((hog_hs, brightness), 1)
print("Evaluating HOG+HS+B features")
print(numpy.shape(hog_hs_b))
validation.validate_feature(hog_hs_b, labels, classes, model, n_folds, False, False, True, True)
#we add the 5th best one
hog_hs_b_c = numpy.concatenate((hog_hs_b, brightness), 1)
print("Evaluating HOG+HS+B+C features")
print(numpy.shape(hog_hs_b_c))
validation.validate_feature(hog_hs_b_c, labels, classes, model, n_folds, False, False, True, True)
#we add the 4th best one
hog_l_b = numpy.concatenate((hog_l, brightness), 1)
print("Evaluating HOG+L+B features")
print(numpy.shape(hog_l_b))
validation.validate_feature(hog_l_b, labels, classes, model, n_folds, False, False, True, True)
#we add the 5th best one
hog_l_b_c = numpy.concatenate((hog_l_b, brightness), 1)
print("Evaluating HOG+L+B+C features")
print(numpy.shape(hog_l_b_c))
validation.validate_feature(hog_l_b_c, labels, classes, model, n_folds, False, False, True, True)
print('\a')
