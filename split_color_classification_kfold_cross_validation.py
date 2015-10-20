# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 15:01:46 2015

@author: Rian
"""

import data_loading as loader
import feature_extraction as extractor
import image_operations as operations
import sklearn.cross_validation as cv
import numpy
from scipy import misc
from scipy import stats
from sklearn import neighbors
from sklearn import cross_validation

def distance(a,b):
    return numpy.sum(numpy.square(numpy.add(a, numpy.multiply(b, -1))))
    
def errorRate(a,b):
    return numpy.sum(numpy.array(a) != numpy.array(b)) / len(a)

def nearestNeighbour(xs, ys, x):
    bestIndex = 0
    bestDistance = distance(xs[0],x)
    for i in range(len(xs)):
        d = distance(xs[i], x)
        if d < bestDistance:
            bestIndex = i
            bestDistance = d
    return ys[bestIndex]
    
def kNearestNeighbour(k, xs, ys, x):
    distances = [distance(x, i) for i in xs] 
    indexes = numpy.argsort(distances)
    classes = [ys[indexes[i]] for i in range(k)]
    return stats.mode(classes)[0][0]

print("Loading images")
images, classes = loader.loadTrainingAndClasses()
amount = len(images)

print("Making thumbnails")
def resizeProper(image, maxPixels):
    ratio = len(image) / len(image[0])
    height = int(numpy.sqrt(maxPixels / ratio))
    width = int(ratio * height)
    return misc.imresize(image, (width, height))
    
thumbsize = 50
thumbs = [misc.imresize(x,(thumbsize, thumbsize)) for x in images]

print("Calculating features")
#features = list(map(extractor.calculateNormalizedColorFeatures, images))
splits = 5
features = numpy.zeros([len(images), splits*splits*3])
for i in range(amount):
    print(i, "/", amount)
    features[i] = extractor.splitColorFeatures(thumbs[i],splits)
    
#model = grid_search.GridSearchCV(svm.SVC(),{'kernel' : ['poly'], 'C' : [1, 10, 100, 1000], 'degree' : [4,7,10], 'shrinking' : [True, False]})    
#model.fit(features, classes)    
#print(model.best_estimator_)
#print('\a')


print("Producing KFold indexes")
kfold = cv.KFold(amount, n_folds = 10, shuffle = True)
model = neighbors.KNeighborsClassifier(n_neighbors = 1)
score = cross_validation.cross_val_score(model, features, classes, cv=kfold)
print(score)
print(score.mean())

predictions = cross_validation.cross_val_predict(model, features, classes, cv = kfold)
wrongIndexes = numpy.nonzero(predictions != classes)
wrongs = numpy.unique(numpy.concatenate((predictions[[wrongIndexes]], numpy.array(classes)[[wrongIndexes]])))

print('\a')