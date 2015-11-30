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
from sklearn import lda, svm
from skimage import feature, color, exposure
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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
print("luv...")
luv = util.loading_map(color.rgb2luv, resized)
print("hed...")
hed = util.loading_map(color.rgb2hed, resized)
print("rgbcie...")
cie = util.loading_map(color.rgb2rgbcie, resized)
print("grayscaling...")
grayscaled = util.loading_map(color.rgb2gray, resized)
#print("edges...")
#edges = util.loading_map(feature.canny, grayscaled)

print("brightness features")
brightness = util.loading_map(extraction.calculateDarktoBrightRatio, resized)
print("hsv features")
hsv_features = util.loading_map(lambda x: extraction.split_image_features(extraction.calculateColorFeatures, 8, x), hsv)
print("luv features")
luv_features = util.loading_map(lambda x: extraction.split_image_features(
    lambda y : extraction.color_features(y, mean = True, std = True), 7, x), luv)
print('\a')
print("hed features")
hed_features = util.loading_map(lambda x: extraction.split_image_features(
    lambda y : extraction.color_features(y, mean = True, std = True), 8, x), hed)
print("rgbcie features")
cie_features = util.loading_map(lambda x: extraction.split_image_features(extraction.calculateColorFeatures, 8, x), cie)
print("hog features")
hog = util.loading_map(lambda x: calcHOG(x,orient=6,nr_of_cells_per_image=6,nr_of_cells_per_block=2, normalise = True), grayscaled)
#print("daisy features")
#daisy = util.loading_map(lambda x: feature.daisy(x, step = 32, radius = 30, rings = 2, histograms = 7, orientations = 7).flatten(), grayscaled)
#print('\a')

hybrid_hsv_luv     = numpy.concatenate((hsv_features, luv_features), 1)
hybrid_hog_luv     = numpy.concatenate((hog, luv_features), 1)
hybrid_hog_hsv     = numpy.concatenate((hog, hsv_features), 1)
hybrid_hog_hsv_luv = numpy.concatenate((hog, hsv_features, luv_features), 1)
hybrid_bright_hog_hsv_luv = numpy.concatenate((brightness,hog, hsv_features, luv_features), 1)
hybrid_bright_hed_hog_luv = numpy.concatenate((brightness, hed_features, hog, luv_features), 1)
hybrid_bright_hog_luv = numpy.concatenate((brightness, hog, luv_features), 1)

n_folds = 5

#model = svm.SVC(kernel='linear', probability = True)
from sklearn import random_projection
model = Pipeline([
    ("standard scaler", StandardScaler()),   
    ("principal component analysis", PCA(500)), #<- appears to reduce efficiency
    #("lda projection", lda.LDA(n_components = 80)),
    #("gaussian random projection", random_projection.GaussianRandomProjection(n_components = 100)),
    #("sparse random projection", random_projection.SparseRandomProjection(n_components = 500)),
    ("logistic regression", LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial'))
    #("svm", svm.SVC(kernel = "sigmoid", C = 1000, gamma = 0.0001, probability = True))
    ])

#print("Evaluating brightness features")
#validation.validate_feature(brightness, labels, classes, model, n_folds, False, False, True)
print("Evaluating hsv features")
validation.validate_feature(hsv_features, labels, classes, model, n_folds, False, False, True, True)
print("Evaluating luv features")
validation.validate_feature(luv_features, labels, classes, model, n_folds, False, False, True, True)
print('\a')
print("Evaluating hed features")
validation.validate_feature(hed_features, labels, classes, model, n_folds, False, False, True, True)
print("Evaluating cie features")
validation.validate_feature(cie_features, labels, classes, model, n_folds, False, False, True, True)
print("Evaluating HOG features")
validation.validate_feature(hog, labels, classes, model, n_folds, False, False, True, True)
#print("Evaluating daisy features")
#validation.validate_feature(daisy, labels, classes, model, n_folds, False, False, True, True)
print("Evaluating hsv+luv")
validation.validate_feature(hybrid_hsv_luv, labels, classes, model, n_folds, False, False, True)
print("Evaluating hog+luv")
validation.validate_feature(hybrid_hog_luv, labels, classes, model, n_folds, False, False, True)
print("Evaluating hog+hsv")
validation.validate_feature(hybrid_hog_hsv, labels, classes, model, n_folds, False, False, True)
print("Evaluating hog+hsv+luv")
validation.validate_feature(hybrid_hog_hsv_luv, labels, classes, model, n_folds, False, False, True)
print("Evaluating brightness+hog+hsv+luv")
validation.validate_feature(hybrid_bright_hog_hsv_luv, labels, classes, model, n_folds, False, False, True, True)
#print("Evaluating combined bcdh features")
#validation.validate_feature(combined_bcdh, labels, classes, model, n_folds, False, True, True, True)
print("Evaluating brightness+hed+hog+luv")
validation.validate_feature(hybrid_bright_hed_hog_luv, labels, classes, model, n_folds, False, False, True, True)
print("Evaluating brightness+hog+luv")
validation.validate_feature(hybrid_bright_hog_luv, labels, classes, model, n_folds, False, False, True, True)
print('\a')

#generate CSV
"""
"""
test_data = loader.loadTest()
test_resized = util.loading_map(lambda x : operations.cropAndResize(x, 0, size), test_data)
test_grayscaled = util.loading_map(color.rgb2gray, test_resized)
test_luv = util.loading_map(color.rgb2luv, test_resized)
print("test brightness features")
test_brightness = util.loading_map(extraction.calculateDarktoBrightRatio, test_resized)
print("test luv features")
test_luv_features = util.loading_map(lambda x: extraction.split_image_features(
    lambda y : extraction.color_features(y, mean = True, std = True), 7, x), test_luv)
print("test hog features")
test_hog = util.loading_map(lambda x: calcHOG(x,orient=6,nr_of_cells_per_image=6,nr_of_cells_per_block=2, normalise = True), test_grayscaled)

#test_features = util.loading_map(lambda x: feature.daisy(x, step = 8, radius = 20, rings = 2, histograms = 4, orientations = 4).flatten(), test_grayscaled)
test_features = numpy.concatenate((test_brightness, test_hog, test_luv_features), 1)

csv_output.generate(hybrid_bright_hog_luv, classes, test_features, model, "ultimate2.csv")#"""
print('\a')