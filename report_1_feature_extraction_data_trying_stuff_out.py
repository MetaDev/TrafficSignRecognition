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
from sklearn import lda
from skimage import feature
from skimage import color

def rian(image):
    keypoints = feature.corner_peaks(feature.corner_harris(image))
    return numpy.array([keypoints.shape[0]])

#preloading
print("loading data...")
images, classes = loader.loadTrainingAndClasses()
amount = len(images)
print("resizing...")
resized = util.loading_map(lambda x : operations.cropAndResize(x, 0, 50), images)
print("grayscaling...")
grayscaled = util.loading_map(color.rgb2gray, resized)

#feature extraction
#print("split color features...")
#split_color_features            = util.loading_map(lambda x: extraction.split_image_features(extraction.color_features, 3, x), resized)
print("corner count")
corner_features                    = util.loading_map(lambda x: rian(x), grayscaled)

n_folds = 5

#model evaluation
#print("Evaluating split color features")
#validation.validate_feature_linear(split_color_features,    classes, n_folds, False, False, True)
#print("---")
print("Evaluating corner features")
validation.validate_feature_linear(corner_features,    classes, n_folds, True, True, True)
print("---")

#test_data = loader.loadTest()
#test_resized = util.loading_map(lambda x : operations.cropAndResize(x, 0, 50), test_data)
#test_features = util.loading_map(lambda x: extraction.split_image_features(extraction.color_features, 3, x), test_resized)

#model = lda.LDA()
#csv_output.generate(split_color_features, classes, test_features, model, "lolwut.csv")