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

#preloading
print("loading data...")
images, classes = loader.loadTrainingAndClasses()
amount = len(images)
print("resizing...")
resized = util.loading_map(lambda x : operations.cropAndResize(x, 0, 50), images)

#feature extraction
print("split normalized color features...")
split_color_features            = util.loading_map(lambda x: extraction.split_image_features(extraction.color_features, 3, x), resized)

n_folds = 8

#model evaluation
print("Evaluating split color features")
validation.validate_feature_linear(split_color_features,    classes, n_folds, False, False, True)
print("---")

test_data = loader.loadTest()
test_resized = util.loading_map(lambda x : operations.cropAndResize(x, 0, 50), test_data)
test_features = util.loading_map(lambda x: extraction.split_image_features(extraction.color_features, 3, x), test_resized)

csv_output.generate(split_color_features, classes, test_features, "lolwut.csv")