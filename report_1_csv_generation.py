# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 15:58:59 2015

@author: Pieter
"""

import datetime
import feature_extraction as extraction
import image_operations as operations
import data_loading as loader
import util
import numpy
from skimage import color
import csv_output
from sklearn import lda
import time

#preloading
print("loading train data...")
#images, classes = loader.loadUniqueTrainingAndClasses()
images, classes = loader.loadTrainingAndClasses()
amount = len(images)
print("resizing...")
resized = util.loading_map(lambda x : operations.cropAndResize(x, 0, 50), images)
print("normalizing...")
normalized = util.loading_map(operations.normalizeImage, resized)
print("grayscaling...")
grayscale = util.loading_map(color.rgb2gray, resized)

print("mean channels features...")
split_color_features            = util.loading_map(lambda x: extraction.split_image_features(extraction.color_features, 3, x), resized)

print("weighted angle features")
weighted_angle_features         = util.loading_map(lambda x : extraction.weightedAngleFeatures(x, 7), resized)
print("Perceived brightness features")
brightness_features             = util.loading_map(extraction.calculateDarktoBrightRatio, resized)
print("High Pass Frequency features")
frequency_features              = util.loading_map(lambda x : extraction.frequencyFeatures(x,selectedclasses=[22,23])[1::8], grayscale)




combined_features_train = []
for i in range(amount):
    combined_feature = numpy.array([])
    combined_feature = numpy.append(combined_feature, split_color_features[i])
    combined_feature = numpy.append(combined_feature, weighted_angle_features[i])
    combined_feature = numpy.append(combined_feature, brightness_features[i])
    combined_feature = numpy.append(combined_feature, frequency_features[i])
    combined_features_train.append(combined_feature)


print("loading test data...")

test_data = loader.loadTest()
test_resized = util.loading_map(lambda x : operations.cropAndResize(x, 0, 50), test_data)
test_grayscale = util.loading_map(color.rgb2gray, test_resized)

print("mean channels features...")
split_color_features_t            = util.loading_map(lambda x: extraction.split_image_features(extraction.color_features, 3, x), test_resized)
print("weighted angle features")
weighted_angle_features_t         = util.loading_map(lambda x : extraction.weightedAngleFeatures(x, 7), test_resized)
print("Perceived brightness features")
brightness_features_t             = util.loading_map(extraction.calculateDarktoBrightRatio, test_resized)
print("High Pass Frequency features")
frequency_features_t              = util.loading_map(lambda x : extraction.frequencyFeatures(x,selectedclasses=[22,23])[1::8], test_grayscale)

combined_features_test = []
for i in range(len(test_data)):
    combined_feature = numpy.array([])
    combined_feature = numpy.append(combined_feature, split_color_features_t[i])
    combined_feature = numpy.append(combined_feature, weighted_angle_features_t[i])
    combined_feature = numpy.append(combined_feature, brightness_features_t[i])
    combined_feature = numpy.append(combined_feature, frequency_features_t[i])
    combined_features_test.append(combined_feature)
    
model = lda.LDA()

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')
csv_output.generate(combined_features_train, classes, combined_features_test, model, "csv/group10_"+st+".csv")