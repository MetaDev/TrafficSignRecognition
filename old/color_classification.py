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

def distance(a,b):
    return numpy.sum(numpy.square(numpy.add(a, numpy.multiply(b, -1))))
    
def errorRate(a,b):
    return numpy.sum(numpy.array(a) == numpy.array(b)) / len(a)

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
size = 25
trainingThumbs = list(map(lambda x: misc.imresize(x,(size,size)), trainingImages))
testThumbs = list(map(lambda x: misc.imresize(x,(size,size)), testImages))

print("Calculating features")
#features = list(map(extractor.calculateNormalizedColorFeatures, images))
trainingFeatures = numpy.zeros([trainingAmount, 3])
testFeatures = numpy.zeros([testAmount, 3])
for i in range(trainingAmount):
    print(i, "/", trainingAmount)
    trainingFeatures[i] = extractor.calculateNormalizedColorFeatures(trainingThumbs[i])
for i in range(testAmount):
    print(i, "/", testAmount)
    testFeatures[i] = extractor.calculateNormalizedColorFeatures(testThumbs[i])  
    
print("Predicting Testdata")
testClasses = [nearestNeighbour(trainingFeatures, trainingClasses, x) for x in testFeatures]

with open('color_classification_1.csv', 'w', newline = '') as csvfile:
    classes = numpy.unique(trainingClasses)
    fieldnames = numpy.insert(classes, 0, 'Id')
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for i in range(testAmount):
        values = {'Id': (i+1)}
        for c in classes: 
            if testClasses[i] == c :
                values[c] = 1
            else:
                values[c] = 0
        writer.writerow(values)