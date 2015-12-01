# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 20:17:20 2015

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

model = Pipeline([
    ("standard scaler", StandardScaler()),
    ("logistic regression", LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial'))
    ])
    
n_folds = 10

print("rgb 1")
rgb_1 = util.loading_map(lambda x: extraction.pixel_features(x, 1), resized)
validation.validate_feature(rgb_1, labels, classes, model, n_folds, False, False, True, True)
print("rgb 3")
rgb_3 = util.loading_map(lambda x: extraction.pixel_features(x, 3), resized)
validation.validate_feature(rgb_3, labels, classes, model, n_folds, False, False, True, True)
print("rgb 5")
rgb_5 = util.loading_map(lambda x: extraction.pixel_features(x, 5), resized)
validation.validate_feature(rgb_5, labels, classes, model, n_folds, False, False, True, True)
print("rgb 7")
rgb_7 = util.loading_map(lambda x: extraction.pixel_features(x, 7), resized)
validation.validate_feature(rgb_7, labels, classes, model, n_folds, False, False, True, True)
print("rgb 9")
rgb_9 = util.loading_map(lambda x: extraction.pixel_features(x, 9), resized)
validation.validate_feature(rgb_9, labels, classes, model, n_folds, False, False, True, True)
print("rgb 7 + std")
rgb_7_std = util.loading_map(lambda x: extraction.split_image_features(
    lambda y : extraction.color_features(y, mean = True, std = True), 8, x), resized)
validation.validate_feature(rgb_7_std, labels, classes, model, n_folds, False, False, True, True)
print("rgb 7 + std + skew")
rgb_7_std_skew = util.loading_map(lambda x: extraction.split_image_features(
    lambda y : extraction.color_features(y, mean = True, std = True, skew = True), 8, x), resized)
validation.validate_feature(rgb_7_std_skew, labels, classes, model, n_folds, False, False, True, True)
print("rgb 7 + std + skew + kurtosis")
rgb_7_std_skew_kurtosis = util.loading_map(lambda x: extraction.split_image_features(
    lambda y : extraction.color_features(y, mean = True, std = True, skew = True, kurtosis = True), 8, x), resized)
validation.validate_feature(rgb_7_std_skew_kurtosis, labels, classes, model, n_folds, False, False, True, True)

print('\a')