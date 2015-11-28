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
#from sklearn import lda, svm
from skimage import feature, color, exposure
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA


def calcHOG(image,orient=8,nr_of_cells_per_image=6,nr_of_cells_per_block=2, normalise = False):
   height = len(image)
   width = len(image[0]) 
   ppc=(height/nr_of_cells_per_image,width/nr_of_cells_per_image)
   cpb=(nr_of_cells_per_block, nr_of_cells_per_block)
   fd = feature.hog(image, orientations=orient, pixels_per_cell=ppc,cells_per_block=cpb, normalise = normalise)
   return numpy.array(fd).flatten()
   
#preloading
print("loading data...")
size = 100
images, labels, classes = loader.loadTrainingImagesPoleNumbersAndClasses()
amount = len(images)

print("resizing...")
resized = util.loading_map(lambda x : operations.cropAndResize(x, 0, size), images)
print("hsv...")
hsv = util.loading_map(color.rgb2hsv, resized)
print("grayscaling...")
grayscaled = util.loading_map(color.rgb2gray, resized)
print("edges...")
edges = util.loading_map(feature.canny, grayscaled)

print("brightness features")
brightness = util.loading_map(extraction.calculateDarktoBrightRatio, resized)
print("color features")
colors = util.loading_map(lambda x: extraction.split_image_features(extraction.calculateColorFeatures, 7, x), hsv)
print("hog features")
hog = util.loading_map(lambda x: calcHOG(x,orient=6,nr_of_cells_per_image=6,nr_of_cells_per_block=2, normalise = True), grayscaled)

combined_bc = numpy.concatenate((brightness, colors), 1)
combined_bh = numpy.concatenate((brightness, hog), 1)
combined_ch = numpy.concatenate((colors, hog), 1)
combined_bch = numpy.concatenate((brightness, colors, hog), 1)

n_folds = 5

#model = svm.SVC(kernel='linear', probability = True)

model = Pipeline([
    ("standard scaler", StandardScaler()),   
    #("principal component analysis", PCA(n_components = 300)), <- appears to reduce efficiency
    ("logistic regression", LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial'))
    ])

print("Evaluating brightness features")
validation.validate_feature(brightness, labels, classes, model, n_folds, False, False, True)
print("Evaluating color features")
validation.validate_feature(colors, labels, classes, model, n_folds, False, False, True, True)
print("Evaluating HOG features")
validation.validate_feature(hog, labels, classes, model, n_folds, False, False, True, True)
print("Evaluating combined bc features")
validation.validate_feature(combined_bc, labels, classes, model, n_folds, False, False, True)
print("Evaluating combined bh features")
validation.validate_feature(combined_bh, labels, classes, model, n_folds, False, False, True)
print("Evaluating combined ch features")
validation.validate_feature(combined_ch, labels, classes, model, n_folds, False, False, True)
print("Evaluating combined bch features")
validation.validate_feature(combined_bch, labels, classes, model, n_folds, False, False, True, True)
print('\a')

#generate CSV
"""
test_data = loader.loadTest()
test_resized = util.loading_map(lambda x : operations.cropAndResize(x, 0, size), test_data)
test_grayscaled = util.loading_map(color.rgb2gray, test_resized)
test_hsv = util.loading_map(color.rgb2hsv, test_resized)
print("test brightness features")
test_brightness = util.loading_map(extraction.calculateDarktoBrightRatio, test_resized)
print("test color features")
test_colors = util.loading_map(lambda x: extraction.split_image_features(extraction.calculateColorFeatures, 7, x), test_hsv)
print("test hog features")
test_hog = util.loading_map(lambda x: calcHOG(x,orient=6,nr_of_cells_per_image=6,nr_of_cells_per_block=2, normalise = True), test_grayscaled)

#test_features = util.loading_map(lambda x: feature.daisy(x, step = 8, radius = 20, rings = 2, histograms = 4, orientations = 4).flatten(), test_grayscaled)
test_features = numpy.concatenate((test_brightness, test_colors, test_hog), 1)

csv_output.generate(combined_bch, classes, test_features, model, "ultimate.csv")#"""
print('\a')