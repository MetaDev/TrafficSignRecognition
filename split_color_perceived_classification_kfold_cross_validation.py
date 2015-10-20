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

print("Loading images")
images, classes = loader.loadProblematicImagesAndClasses()
#images, classes = loader.loadTrainingAndClasses()
amount = len(images)

print("Making thumbnails")

thumbsize = 50
thumbs = [misc.imresize(x,(thumbsize, thumbsize)) for x in images]

print("Calculating features")
#features = list(map(extractor.calculateNormalizedColorFeatures, images))
splits = 5
features = numpy.zeros([len(images), 100 + splits * splits * 3])
for i in range(amount):
    print(i, "/", amount)
    #features[i] = extractor.splitColorFeatures(thumbs[i],splits)
    harald = extractor.calculateDarktoBrightRatio(thumbs[i])
    rian = extractor.splitColorFeatures(thumbs[i], splits)
    features[i] = numpy.append(harald, rian)
    
#model = grid_search.GridSearchCV(svm.SVC(),{'kernel' : ['poly'], 'C' : [1, 10, 100, 1000], 'degree' : [4,7,10], 'shrinking' : [True, False]})    
#model.fit(features, classes)    
#print(model.best_estimator_)
#print('\a')

print("Producing KFold indexes")
kfold = cv.KFold(amount, n_folds = 5, shuffle = True)
model = neighbors.KNeighborsClassifier(n_neighbors = 1)
score = cross_validation.cross_val_score(model, features, classes, cv=kfold)
print(score)
print(score.mean())

predictions = cross_validation.cross_val_predict(model, features, classes, cv = kfold)
wrongIndexes = numpy.nonzero(predictions != classes)
uniqueWrongs, counts = numpy.unique(numpy.append(predictions[[wrongIndexes]], numpy.array(classes)[[wrongIndexes]]), return_counts = True)
wrongs = uniqueWrongs[counts > 10]

print('\a')