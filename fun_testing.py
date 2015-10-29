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
from sklearn import neighbors
from sklearn import lda
from sklearn import svm
from sklearn import cross_validation
import score_calculation
import image_operations

#images, classes = loader.loadProblematicImagesAndClasses()
images, classes = loader.loadUniqueTrainingAndClasses()
amount = len(images)
thumbsize = 50
thumbs = [misc.imresize(x,(thumbsize, thumbsize)) for x in images]
features = []
for i in range(amount):
    if(i%10 ==0):print(i, "/", amount)
    colors = extractor.normalizedColorFeatures(thumbs[i][5:45,5:45,:])
    #angles = extractor.weightedAngleFeatures(thumbs[i][5:45,5:45,:], 11)
    #differences = extractor.pixelDifferences(thumbs[i][15:45,15:45,:])
    #feature = numpy.concatenate((colors, angles, differences))
    features.append(colors)

kfold = cv.KFold(amount, n_folds = 4, shuffle = True)
#model = neighbors.KNeighborsClassifier(n_neighbors = 1)
model = lda.LDA()
score = cross_validation.cross_val_score(model, features, classes, cv=kfold)
print("scores ", score)
print("mean score ", score.mean())
print("std score ", numpy.std(score))

#model = svm.SVC(kernel = 'linear', probability = True)
#model = neighbors.KNeighborsClassifier(n_neighbors = 1)
model = lda.LDA()
scores = score_calculation.loglossKFold(features, classes, model, 4)
print("logloss scores ", scores)
print("logloss score mean ", numpy.mean(scores), " ", numpy.std(scores))

predictions = cross_validation.cross_val_predict(model, numpy.array(features), numpy.array(classes), cv = kfold)
wrongIndexes = numpy.nonzero(predictions != classes)
uniqueWrongs, counts = numpy.unique(numpy.append(predictions[[wrongIndexes]], numpy.array(classes)[[wrongIndexes]]), return_counts = True)
wrongs = uniqueWrongs[counts > 10]

print('\a')