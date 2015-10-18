# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 15:01:46 2015

@author: Rian
"""

import data_loading as loader
import feature_extraction as extractor
import sklearn.cross_validation as cv
import numpy
import csv
from scipy import misc
from sklearn import svm

def distance(a,b):
    return numpy.sum(numpy.square(numpy.add(a, numpy.multiply(b, -1))))

def nearestNeighbour(xs, ys, x):
    bestIndex = 0
    bestDistance = distance(xs[0],x)
    for i in range(len(xs)):
        d = distance(xs[i], x)
        if d < bestDistance:
            bestIndex = i
            bestDistance = d
    return ys[bestIndex]

print("Loading images")
trainingImages, trainingClasses = loader.loadTrainingAndClasses()
testImages = loader.loadTest()
trainingAmount = len(trainingImages)
testAmount = len(testImages)

print("Making thumbnails")
size = 50
trainingThumbs = list(map(lambda x: misc.imresize(x,(size,size)), trainingImages))
testThumbs = list(map(lambda x: misc.imresize(x,(size,size)), testImages))

print("Calculating features")
#features = list(map(extractor.calculateNormalizedColorFeatures, images))
angleClasses = 7
trainingFeatures = numpy.zeros([trainingAmount, 4 * angleClasses + 3])
testFeatures = numpy.zeros([testAmount, 4 * angleClasses + 3])
for i in range(trainingAmount):
    print(i, "/", trainingAmount)
    trainingFeatures[i] = extractor.colorQuadrantAngleFeatures(trainingThumbs[i], angleClasses, 100, 160)
for i in range(testAmount):
    print(i, "/", testAmount)
    testFeatures[i] = extractor.colorQuadrantAngleFeatures(testThumbs[i], angleClasses, 100, 160)
    
print("Predicting Testdata")
model = svm.SVC(kernel = 'poly', degree = 15, probability = True)
model.fit(trainingFeatures, trainingClasses)

with open('color_anglequadrant_csv_classification_1.csv', 'w', newline = '') as csvfile:
    classes = numpy.unique(trainingClasses)
    fieldnames = numpy.insert(classes, 0, 'Id')
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for i in range(testAmount):
        labels = classes
        labels = numpy.insert(labels, 0, 'Id')
        values = model.predict_proba(testFeatures[i])
        values = numpy.insert(values,0, int(i + 1))
        dictionary = dict(zip(labels, values))
        writer.writerow(dictionary)
print('\a')