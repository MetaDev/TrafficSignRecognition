# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 11:01:46 2015

@author: Pieter
"""

import data_loading as loader
import feature_extraction as extractor
import image_operations as operations
import sklearn.cross_validation as cv
import numpy
import copy
import math
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

def rgb2gray(rgb):
    return numpy.dot(rgb[...,:3],[0.299, 0.587, 0.144])

def mask_frequency(fshift, thumbsize, frequencyclasses, fc):
    stepsize = (thumbsize/2)/frequencyclasses
    m = copy.copy(fshift)
    middle = (thumbsize/2)-1;
    if fc != 0:
        #set inside of ring to 0 (if fc=0 we pick the innermost block)
        m[middle-stepsize*(fc-1):middle+stepsize*(fc)+1,middle-stepsize*(fc-1):middle+stepsize*(fc)+1] = 0
    if fc != frequencyclasses-1:
        #set outer ring to 0
        m[0:middle-stepsize*(fc),:] = 0
        m[middle+stepsize*(fc+1)+1:thumbsize,:] = 0
        m[:,0:middle-stepsize*(fc),] = 0
        m[:,middle+stepsize*(fc+1)+1:thumbsize] = 0
    return m

    
def frequencyFeatures(thumbs, frequencyclasses = 25, subsect_v = 10, subsect_h=10, selectedclasses = [22,23,24]):
    features = numpy.ones([len(images), len(selectedclasses)*subsect_v*subsect_h])     #to save feature class frequencies
    for i in range(amount):
        if i%100==0:
            print("freq:", i, "/", amount)
        for subsection in range(subsect_v*subsect_h):
            horizontal = subsection % subsect_h
            vertical = math.floor(subsection/subsect_v)
            h_size = thumbsize/subsect_h
            v_size = thumbsize/subsect_v
            subthumb = thumbs[i][horizontal*h_size:(horizontal+1)*h_size,vertical*v_size:(vertical+1)*v_size]
            fthumb = numpy.fft.fft2(subthumb)  #fourier transform
            fshift = numpy.fft.fftshift(fthumb) #shift 0 frequency to center
            index = 0
            for fc in range(frequencyclasses):
                if fc in selectedclasses:
                    m = mask_frequency(fshift,thumbsize,frequencyclasses,fc) #select frequency components of this class
                    f_ishift = numpy.fft.ifftshift(m)                        #inverse shift
                    img_back = numpy.fft.ifft2(f_ishift)                     #inverse transform
                    img_back = numpy.abs(img_back)
                    features[i][subsection*len(selectedclasses)+index] = sum(sum(img_back))/(h_size*v_size)   #last multiplication is so there is more weight on high frequencies (edges)
                    index += 1
    return features


print("Loading images")
images, classes = loader.loadTrainingAndClasses()
amount = len(images)

print("Making thumbnails")
def resizeProper(image, maxPixels):
    ratio = len(image) / len(image[0])
    height = int(numpy.sqrt(maxPixels / ratio))
    width = int(ratio * height)
    return misc.imresize(image, (width, height))
    
thumbsize = 50;
thumbs = [misc.imresize(x,(thumbsize,thumbsize)) for x in images]   #resize !!
thumbs = [rgb2gray(x) for x in thumbs]                              #grayscale !!

print("Calculating features")

features = frequencyFeatures(thumbs)


print("Evaluating model with KFold (1)")
model = neighbors.KNeighborsClassifier(n_neighbors = 1)
kfold = cv.KFold(amount, n_folds = 10, shuffle = True)
score = cross_validation.cross_val_score(model, features, classes, cv = kfold)
print(score)


print("Evaluating model with KFold (2)")
counter = 0
errors  = numpy.zeros(len(kfold))


for train_index, test_index in kfold:
    print(counter)
    trainFeatures = [features[i] for i in train_index]
    trainClasses  = [classes[i] for i in train_index]
    testFeatures  = [features[i] for i in test_index]
    testClasses   = [classes[i] for i in test_index]
    
    model = neighbors.KNeighborsClassifier(n_neighbors = 1)
    model.fit(trainFeatures, trainClasses)    
    
    predictedClasses = model.predict(testFeatures)
    #predictedClasses = [kNearestNeighbour(10, trainFeatures, trainClasses, x) for x in testFeatures]
    errors[counter-1] = errorRate(testClasses, predictedClasses)
    print(errors[counter-1])
    counter = counter + 1
    

    
print("mean error ", errors.mean())
print('\a')