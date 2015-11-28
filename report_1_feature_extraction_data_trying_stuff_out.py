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

def corner_count(image):
    keypoints = feature.corner_peaks(feature.corner_harris(image))
    return numpy.array([keypoints.shape[0]])
    
def expanded_orb_descriptors(image):
    descriptor_extractor = feature.ORB(n_keypoints=200)
    descriptor_extractor.detect_and_extract(image)
    descriptors = descriptor_extractor.descriptors
    expanded = numpy.append(descriptors, numpy.zeros((200-len(descriptors), 256)) - 1, 0)
    return expanded.flatten()
    
def daisy_features(image):
    features = feature.daisy(image, step = 8, radius = 20, rings = 2, histograms = 4, orientations = 4)
    return features.flatten()
    
def reduce_features(features, number, classes):
    partial_features = numpy.array(features)[:,1::10]
    partial_classes = numpy.array(classes)[:,1::10]
    model = lda.LDA(n_components=number)
    model.fit_transform(partial_features,partial_classes)
    return model.transform(features)

#preloading
print("loading data...")
size = 70
images, labels, classes = loader.loadTrainingImagesPoleNumbersAndClasses()
amount = len(images)
print("resizing...")
resized = util.loading_map(lambda x : operations.cropAndResize(x, 0, size), images)
print("grayscaling...")
grayscaled = util.loading_map(color.rgb2gray, resized)

#feature extraction
#print("split color features...")
#split_color_features            = util.loading_map(lambda x: extraction.split_image_features(extraction.color_features, 3, x), resized)
#print("corner count")
#corner_features                    = util.loading_map(lambda x: corner_count(x), grayscaled)
print("daisy features")
daisy = util.loading_map(lambda x: feature.daisy(x, step = 8, radius = 20, rings = 2, histograms = 4, orientations = 4).flatten(), grayscaled)

n_folds = 5

model = lda.LDA()
#model evaluation
#print("Evaluating split color features")
#validation.validate_feature(split_color_features, labels, classes, model, n_folds, True, True, True)
#print("---")
#print("Evaluating corner features")
#validation.validate_feature(corner_features, labels, classes, model, n_folds, True, True, True)
#print("---")
print("Evaluating daisy features")
validation.validate_feature(daisy, labels, classes, model, n_folds, True, True, True)
print("---")

"""test_data = loader.loadTest()
test_resized = util.loading_map(lambda x : operations.cropAndResize(x, 0, size), test_data)
test_grayscaled = util.loading_map(color.rgb2gray, test_resized)
test_features = util.loading_map(lambda x: feature.daisy(x, step = 8, radius = 20, rings = 2, histograms = 4, orientations = 4).flatten(), test_grayscaled)

csv_output.generate(daisy, classes, test_features, model, "daisy.csv")"""