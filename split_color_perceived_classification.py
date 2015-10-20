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
from sklearn import neighbors

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
splits = 5
trainingFeatures = numpy.zeros([trainingAmount, splits * splits * 3 + 100])
testFeatures = numpy.zeros([testAmount, splits * splits * 3 + 100])
for i in range(trainingAmount):
    print(i, "/", trainingAmount)
    harald = extractor.calculateDarktoBrightRatio(trainingThumbs[i])
    rian = extractor.splitColorFeatures(trainingThumbs[i], splits)
    trainingFeatures[i] = numpy.append(harald, rian)
for i in range(testAmount):
    print(i, "/", testAmount)
    harald = extractor.calculateDarktoBrightRatio(testThumbs[i])
    rian = extractor.splitColorFeatures(testThumbs[i], splits)
    testFeatures[i] = numpy.append(harald, rian)
    
print("Predicting Testdata")
model = neighbors.KNeighborsClassifier(n_neighbors = 1) #svm.SVC(kernel = 'linear', probability = True)
model.fit(trainingFeatures, trainingClasses)

"""with open('split_color_perceived_classification_1.csv', 'w', newline = '') as csvfile:
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
        dictionary['Id'] = int(dictionary['Id'])
        writer.writerow(dictionary)"""
        
testClasses = model.predict(testFeatures)
        
errorRate = 0.02
with open('split_color_perceived_classification_2.csv', 'w', newline = '') as csvfile:
    classes = numpy.unique(trainingClasses)
    fieldnames = numpy.insert(classes, 0, 'Id')
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for i in range(testAmount):
        values = {'Id': (i+1)}
        for c in classes: 
            if testClasses[i] == c :
                values[c] = 1 - errorRate
            else:
                values[c] = errorRate / (len(classes) - 1)
        writer.writerow(values)
print('\a')