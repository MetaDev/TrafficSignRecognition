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
from skimage import color

#preloading
print("loading data...")
#images, classes = loader.loadUniqueTrainingAndClasses()
images, classes = loader.loadTrainingAndClasses()
amount = len(images)
print("resizing...")
resized = util.loading_map(lambda x : operations.cropAndResize(x, 0, 50), images)
print("normalizing...")
normalized = util.loading_map(operations.normalizeImage, resized)
print("grayscaling...")
grayscale = util.loading_map(color.rgb2gray, resized)
#print("reducing color space...")
#reduced = util.loading_map(operations.reduceColorSpace, resized)

#feature extraction
#print("color features...")
#color_features                  = util.loading_map(extraction.color_features, resized)
#print("normalized color features...")
#normalized_color_features       = util.loading_map(extraction.color_features, normalized)
print("mean channels features...")
split_color_features            = util.loading_map(lambda x: extraction.split_image_features(extraction.color_features, 3, x), resized)
print("moment channel features...")
moment_channel_features            = util.loading_map(lambda x: extraction.split_image_features(lambda y: extraction.color_features(y,True,True,True,True), 3, x), resized)
#print("split normalized color features...")
#split_normalized_color_features = util.loading_map(lambda x: extraction.split_image_features(extraction.color_features, 3, x), normalized)
#print("angle features")
#angle_features         = util.loading_map(lambda x : extraction.calculateAngleFeatures(x, 7), resized)
print("weighted angle features")
weighted_angle_features         = util.loading_map(lambda x : extraction.weightedAngleFeatures(x, 7), resized)
print("Perceived brightness features")
brightness_features             = util.loading_map(extraction.calculateDarktoBrightRatio, resized)
print("High Pass Frequency features")
frequency_features              = util.loading_map(lambda x : extraction.frequencyFeatures(x,selectedclasses=[22,23])[1::8], grayscale)


combined_features1 = []
for i in range(amount):
    combined_feature = numpy.array([])
    combined_feature = numpy.append(combined_feature, split_color_features[i])
    combined_feature = numpy.append(combined_feature, weighted_angle_features[i])
    combined_features1.append(combined_feature)

combined_features2 = []
for i in range(amount):
    combined_feature = numpy.array([])
    combined_feature = numpy.append(combined_feature, split_color_features[i])
    combined_feature = numpy.append(combined_feature, brightness_features[i])
    combined_features2.append(combined_feature)
    
combined_features3 = []
for i in range(amount):
    combined_feature = numpy.array([])
    combined_feature = numpy.append(combined_feature, split_color_features[i])
    combined_feature = numpy.append(combined_feature, brightness_features[i])
    combined_feature = numpy.append(combined_feature, frequency_features[i])
    combined_features3.append(combined_feature)
    
combined_features4 = []
for i in range(amount):
    combined_feature = numpy.array([])
    combined_feature = numpy.append(combined_feature, split_color_features[i])
    combined_feature = numpy.append(combined_feature, weighted_angle_features[i])
    combined_feature = numpy.append(combined_feature, brightness_features[i])
    combined_features4.append(combined_feature)
    
combined_features5 = []
for i in range(amount):
    combined_feature = numpy.array([])
    combined_feature = numpy.append(combined_feature, split_color_features[i])
    combined_feature = numpy.append(combined_feature, weighted_angle_features[i])
    combined_feature = numpy.append(combined_feature, brightness_features[i])
    combined_feature = numpy.append(combined_feature, frequency_features[i])
    combined_features5.append(combined_feature)
    
    

n_folds = 5

#model evaluation
print("Evaluating mean channels features")
validation.validate_feature_linear(split_color_features,    classes, n_folds, False, False, True)
print("---")
print("Evaluating moment channels features")
validation.validate_feature_linear(moment_channel_features,    classes, n_folds, False, False, True)
print("---")
print("Evaluating weighted angle features")
validation.validate_feature_linear(weighted_angle_features, classes, n_folds, False, False, True)
print("---")
print("Evaluating perceived brightness features")
validation.validate_feature_linear(brightness_features,     classes, n_folds, False, False, True)
print("---")
print("Evaluating High pass frequency features")
validation.validate_feature_linear(frequency_features,      classes, n_folds, False, False, True)
print("---")
print("Evaluating combined features: mean channels & weigthed angle")
validation.validate_feature_linear(combined_features1,       classes, n_folds, False, False, True)
print("Evaluating combined features: mean channels & perceived brightness")
validation.validate_feature_linear(combined_features2,       classes, n_folds, False, False, True)
print("Evaluating combined features: mean channels & perceived brightness & frequency")
validation.validate_feature_linear(combined_features3,       classes, n_folds, False, False, True)
print("Evaluating combined features: mean channels & perceived brightness & weighted angle")
validation.validate_feature_linear(combined_features4,       classes, n_folds, False, False, True)
print("Evaluating combined features: mean channels & perceived brightness & weighted angle & frequency")
validation.validate_feature_linear(combined_features5,       classes, n_folds, False, False, True)

#lower performing features
#print("Evaluating color features")
#validation.validate_feature_linear(color_features,                  classes, n_folds, False, False, True)
#print("Evaluating normalized color features")
#validation.validate_feature_linear(normalized_color_features,       classes, n_folds, False, False, True)
#print("Evaluating split normalized color features")
#validation.validate_feature_linear(split_normalized_color_features, classes, n_folds, False, False, True)
#print("Evaluating angle features")
#validation.validate_feature_linear(angle_features,                   classes, n_folds, False, False, True)