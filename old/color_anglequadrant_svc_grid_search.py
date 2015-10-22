# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 15:01:46 2015

@author: Rian
"""

import data_loading as loader
import feature_extraction as extractor
import sklearn.cross_validation as cv
import numpy
from scipy import misc
from sklearn import svm, grid_search

def errorRate(a,b):
    return numpy.sum(numpy.array(a) != numpy.array(b)) / len(a)

print("Loading images")
images, classes = loader.loadTrainingAndClasses()
amount = len(images)

print("Making thumbnails")
    
thumbsize = 25
thumbs = [misc.imresize(x,(thumbsize, thumbsize)) for x in images]

print("Calculating features")
#features = list(map(extractor.calculateNormalizedColorFeatures, images))
angleClasses = 7
features = numpy.zeros([len(images), 4 * angleClasses + 3])
for i in range(amount):
    print(i, "/", amount)
    features[i] = extractor.colorQuadrantAngleFeatures(thumbs[i], angleClasses, 100, 160)

#parameters = {'C': [1, 10, 100, 1000, 10000], 'gamma': [0.01, 0.001, 0.0001], 'kernel': ['rbf']}
#classifier = grid_search.GridSearchCV(svm.SVC(), parameters)
#classifier.fit(features, classes)  

    
print("Producing KFold indexes")
kfold = cv.KFold(amount, n_folds = 10, shuffle = True)

print("Evaluating model with KFold")
counter = 0
errors  = numpy.zeros(len(kfold))
for train_index, test_index in kfold:
    print(counter)
    trainFeatures = [features[i] for i in train_index]
    trainClasses  = [classes[i] for i in train_index]
    testFeatures  = [features[i] for i in test_index]
    testClasses   = [classes[i] for i in test_index]
    
    #parameters = {'C': [1, 10, 100, 1000, 10000], 'gamma': [0.01, 0.001, 0.0001], 'kernel': ['rbf']}
    classifier = svm.SVC(C=1000, gamma = 0.01)#grid_search.GridSearchCV(svm.SVC(), parameters)
    classifier.fit(trainFeatures, trainClasses)  
    
    predictedClasses = classifier.predict(testFeatures)
    errors[counter-1] = errorRate(testClasses, predictedClasses)
    print(errors[counter-1])
    counter = counter + 1
    
print("mean error ", errors.mean())
print('\a')