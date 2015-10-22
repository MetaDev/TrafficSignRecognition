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
from sklearn import lda
from sklearn import qda
from sklearn import svm
from sklearn import cross_validation
from sklearn import mixture
import score_calculation

print("Loading images")
#images, classes = loader.loadProblematicImagesAndClasses()
images, classes = loader.loadTrainingAndClasses()
amount = len(images)

print("Making thumbnails")

thumbsize = 50
thumbs = [misc.imresize(x,(thumbsize, thumbsize)) for x in images]

print("Calculating features")
#features = list(map(extractor.calculateNormalizedColorFeatures, images))
splits = 5
features = []
for i in range(amount):
    if(i%10 ==0):print(i, "/", amount)
    #features[i] = extractor.splitColorFeatures(thumbs[i],splits)
    harald = extractor.calculateDarktoBrightRatio(thumbs[i])[0::4]
    rian = extractor.splitColorFeatures(thumbs[i], splits)[0::4]
    features.append(numpy.append(harald, rian))
    
#model = grid_search.GridSearchCV(svm.SVC(),{'kernel' : ['poly'], 'C' : [1, 10, 100, 1000], 'degree' : [4,7,10], 'shrinking' : [True, False]})    
#model.fit(features, classes)    
#print(model.best_estimator_)
#print('\a')

print("Producing KFold indexes")
kfold = cv.KFold(amount, n_folds = 8, shuffle = True)
#model = neighbors.KNeighborsClassifier(n_neighbors = 1)
model = svm.SVC(kernel = 'linear')
#model = qda.QDA()
score = cross_validation.cross_val_score(model, features, classes, cv=kfold)
print("scores ", score)
print("mean score ", score.mean())

model = svm.SVC(kernel = 'linear', probability = True)
#model = neighbors.KNeighborsClassifier(n_neighbors = 1)
scores = score_calculation.loglossKFold(features, classes, model, 8)
print("logloss scores ", scores)
print("logloss score mean ", numpy.mean(scores), " ", numpy.std(scores))

#predictions = cross_validation.cross_val_predict(model, features, classes, cv = kfold)
#wrongIndexes = numpy.nonzero(predictions != classes)
#uniqueWrongs, counts = numpy.unique(numpy.append(predictions[[wrongIndexes]], numpy.array(classes)[[wrongIndexes]]), return_counts = True)
#wrongs = uniqueWrongs[counts > 10]

print('\a')