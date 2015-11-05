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

#preloading
print("loading data...")
images, classes = loader.loadUniqueTrainingAndClasses()
amount = len(images)
print("resizing...")
resized = util.loading_map(lambda x : operations.cropAndResize(x, 0, 50), images)
print("normalizing...")
normalized = util.loading_map(operations.normalizeImage, resized)
print("reducing color space...")
reduced = util.loading_map(operations.reduceColorSpace, resized)

#feature extraction
#print("color features...")
#color_features                  = util.loading_map(extraction.color_features, resized)
#print("normalized color features...")
#normalized_color_features       = util.loading_map(extraction.color_features, normalized)
print("split normalized color features...")
split_color_features            = util.loading_map(lambda x: extraction.split_image_features(extraction.color_features, 5, x), resized)
#print("split normalized color features...")
#split_normalized_color_features = util.loading_map(lambda x: extraction.split_image_features(extraction.color_features, 3, x), normalized)
#print("angle features")
#angle_features         = util.loading_map(lambda x : extraction.calculateAngleFeatures(x, 7), resized)
print("weighted angle features")
weighted_angle_features         = util.loading_map(lambda x : extraction.weightedAngleFeatures(x, 7), resized)

combined_features = split_color_features.copy()
for i in range(amount):
    combined_features[i] = numpy.append(combined_features[i], weighted_angle_features[i])

n_folds = 8

#model evaluation
print("Evaluating color features")
#validation.validate_feature_linear(color_features,                  classes, n_folds, False, False, True)
print("Evaluating normalized color features")
#validation.validate_feature_linear(normalized_color_features,       classes, n_folds, False, False, True)
print("Evaluating split color features")
validation.validate_feature_linear(split_color_features,            classes, n_folds, False, False, True)
print("Evaluating split normalized color features")
#validation.validate_feature_linear(split_normalized_color_features, classes, n_folds, False, False, True)
print("Evaluating angle features")
#validation.validate_feature_linear(angle_features,                   classes, n_folds, False, False, True)
print("Evaluating weighted angle features")
validation.validate_feature_linear(weighted_angle_features,         classes, n_folds, False, False, True)
#print("Evaluating haralds features")
#validation.validate_feature_linear(harald_features,                 classes, n_folds, False, True, True)
print("Evaluating combined features")
validation.validate_feature_linear(combined_features,               classes, n_folds, False, True, True)