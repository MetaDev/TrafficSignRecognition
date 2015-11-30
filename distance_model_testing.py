# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 15:43:03 2015

@author: Rian
"""

from distance_model import DistanceModel
import data_loading
import util
import image_operations as operations
import feature_extraction as extraction
from skimage import feature, color, exposure
import feature_validation as validation

training_images, training_labels, training_classes = data_loading.loadTrainingImagesPoleNumbersAndClasses()
size = 100

print("resizing...")
resized = util.loading_map(lambda x : operations.cropAndResize(x, 0, size), training_images)
print("hsv...")
hsv = util.loading_map(color.rgb2hsv, resized)
print("grayscaling...")
grayscaled = util.loading_map(color.rgb2gray, resized)

print("colors")
colors = util.loading_map(lambda x: extraction.split_image_features(extraction.calculateColorFeatures, 7, x), hsv)

n_folds = 5

print("evaluating colors")
model = DistanceModel()
#from sklearn.linear_model import LogisticRegression
#model = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial')
validation.validate_feature(colors, training_labels, training_classes, model, n_folds, False, False, True, True)
print('\a')