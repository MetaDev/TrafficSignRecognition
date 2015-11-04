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

#preloading
print("loading data...")
images, classes = loader.loadUniqueTrainingAndClasses()
print("resizing...")
resized = util.loading_map(lambda x : operations.cropAndResize(x, 0.10, 50), images)
print("normalizing...")
normalized = util.loading_map(operations.normalizeImage, resized)

#feature extraction
print("color features...")
color_features = util.loading_map(extraction.color_features, resized)
print("normalized color features...")
normalized_color_features = util.loading_map(extraction.color_features, normalized)
print("split normalized color features...")
split_normalized_color_features = util.loading_map(lambda x: extraction.split_image_features(extraction.color_features, 3, x), normalized)

#model evaluation
print("Evaluating color features")
validation.validate_feature_linear(color_features, classes, 5, False)
print("Evaluating normalized color features")
validation.validate_feature_linear(normalized_color_features, classes, 5, False)
print("Evaluating split normalized color features")
validation.validate_feature_linear(split_normalized_color_features, classes, 5, False)