# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 20:17:20 2015

@author: Rian
"""
import feature_validation as validation
import feature_extraction as extraction
import image_operations as operations
import data_loading as loader
import util
import numpy
from skimage import color, exposure
from sklearn import lda
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plot

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

model = Pipeline([
    ("standard scaler", StandardScaler()),
    ("logistic regression", LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial'))
    ])
    
n_folds = 10

for i in range(1, 14, 2):
    print("rgb", i)
    rgb_f = util.loading_map(lambda x: extraction.pixel_features(x, i), resized)
    validation.validate_feature(rgb_f, labels, classes, model, n_folds, False, False, True, True)

for i in range(1, 14, 2):
    print("hsv", i)
    hsv_f = util.loading_map(lambda x: extraction.pixel_features(x, i), hsv)
    validation.validate_feature(hsv_f, labels, classes, model, n_folds, False, False, True, True)
    
for i in range(1, 14, 2):
    print("luv", i)
    luv_f = util.loading_map(lambda x: extraction.pixel_features(x, i), luv)
    validation.validate_feature(luv_f, labels, classes, model, n_folds, False, False, True, True)

#With STD
print("rgb 11 + std")
rgb_11_std = util.loading_map(lambda x: extraction.split_image_features(
    lambda y : extraction.color_features(y, mean = True, std = True), 11, x), resized)
#validation.validate_feature(rgb_11_std, labels, classes, model, n_folds, False, False, True, True)
print("hsv 11 + std")
hsv_11_std = util.loading_map(lambda x: extraction.split_image_features(
    lambda y : extraction.color_features(y, mean = True, std = True), 11, x), hsv)
#validation.validate_feature(hsv_11_std, labels, classes, model, n_folds, False, False, True, True)
print('\a')
print("luv 11 + std")
luv_11_std = util.loading_map(lambda x: extraction.split_image_features(
    lambda y : extraction.color_features(y, mean = True, std = True), 11, x), luv)
#validation.validate_feature(luv_11_std, labels, classes, model, n_folds, False, False, True, True)

#Using LDA to plot features
class_a = 'F50' #blue square
class_b = 'B9' #diamond
class_c = 'C35' #red circle
class_array = numpy.array(classes)
a = class_array == class_a
b = class_array == class_b
c = class_array == class_c
projector = lda.LDA(n_components = 2)

figure = plot.figure(1)
for (feature, name, position) in (rgb_11_std, "rgb", 131), (hsv_11_std, "hsv", 132), (luv_11_std, "luv",133):
    projector.fit(numpy.array(feature)[0::2,:], classes[0::2])
    a_results = projector.transform(numpy.array(feature)[a][1::2,:])
    b_results = projector.transform(numpy.array(feature)[b][1::2,:])
    c_results = projector.transform(numpy.array(feature)[c][1::2,:])

    a_plot = plot.scatter(a_results[:,0], a_results[:,1], c = 'b')
    b_plot = plot.scatter(b_results[:,0], b_results[:,1], c = 'g')
    c_plot = plot.scatter(c_results[:,0], c_results[:,1], c = 'r')

    plot.subplot(position)    
    plot.legend((a_plot, b_plot, c_plot), ("F50", "B9", "C35"), title = name)
    