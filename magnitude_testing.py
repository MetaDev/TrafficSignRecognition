# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 20:42:21 2015

@author: Rian
"""

import data_loading as loader
import image_operations as operations
import feature_extraction as extractor
import sklearn.cross_validation as cv
import numpy
import csv
from scipy import misc
from scipy import stats
from sklearn import neighbors
from sklearn import cross_validation

def magnitude(image, splits = 1):
    angles, magnitudes = operations.calculatePixelAngleAndMagnitude(image)
    features = numpy.zeros(splits * splits * 2)
    width = len(image)
    height = len(image[0])
    for i in range(splits):
        for j in range(splits):
            index = (i*splits + j) * 2
            subMagnitude = magnitudes[width/splits*i:width/splits*(i+1), height/splits*j:height/splits*(j+1)]
            features[index] = subMagnitude.mean()
            features[index + 1] = subMagnitude.std()
    return features

def angleCount(image, angleThreshold = 0.01, magnitudeThreshold = 100):
    angles, magnitudes = operations.calculatePixelAngleAndMagnitude(image)
    barrier = magnitudes > magnitudeThreshold
    usableAngles = (angles[barrier] / angleThreshold).astype(int)
    _, counts = numpy.unique(usableAngles, return_counts = True)
    if len(counts) == 0:
        mean = 0
    else:
        mean = counts.mean()
    return (counts > mean).sum()

images, classes = loader.loadTrainingAndClasses()

thumbsize = 50
thumbs = list(map(lambda x: misc.imresize(x, (thumbsize, thumbsize)), images))

features = numpy.zeros((0, 1))
for i in range(len(thumbs)):
    if i % 100 == 0: print(i)
    features = numpy.concatenate((features, [[angleCount(thumbs[i])]]))

model = neighbors.KNeighborsClassifier(n_neighbors = 5)
score = cross_validation.cross_val_score(model, features, classes, cv = 5)
print(score)
print('\a')