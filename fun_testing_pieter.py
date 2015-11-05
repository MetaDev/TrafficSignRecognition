# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 15:01:46 2015

@author: Rian
"""
import data_loading as loader
import feature_extraction as extractor
import sklearn.cross_validation as cv
import numpy
import copy
from scipy import misc
from sklearn import neighbors
from sklearn import lda
from sklearn import svm
from sklearn import cross_validation
import score_calculation
import image_operations as iop

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

def frequencyFeaturesImage(image, frequencyclasses = 25, subsect_v = 10, subsect_h=10, selectedclasses = [22,23,24]):
    features = numpy.zeros([len(selectedclasses)*subsect_v*subsect_h])     #to save feature class frequencies
    thumbsize = len(image)
    transformed = numpy.zeros(len(image),len(image[0]))    
    for subsection in range(subsect_v*subsect_h):
        horizontal = subsection % subsect_h
        vertical = math.floor(subsection/subsect_v)
        h_size = thumbsize/subsect_h
        v_size = thumbsize/subsect_v
        subthumb = image[horizontal*h_size:(horizontal+1)*h_size,vertical*v_size:(vertical+1)*v_size]
        fthumb = numpy.fft.fft2(subthumb)  #fourier transform
        fshift = numpy.fft.fftshift(fthumb) #shift 0 frequency to center
        index = 0
        for fc in range(frequencyclasses):
            if fc in selectedclasses:
                m = mask_frequency(fshift,thumbsize,frequencyclasses,fc) #select frequency components of this class
                f_ishift = numpy.fft.ifftshift(m)                        #inverse shift
                img_back = numpy.fft.ifft2(f_ishift)                     #inverse transform
                img_back = numpy.abs(img_back)
                features[subsection*len(selectedclasses)+index] = sum(sum(img_back))/(subsect_h*subsect_v)   #last multiplication is so there is more weight on high frequencies (edges)
                index += 1
        transformed [horizontal*h_size:(horizontal+1)*h_size,vertical*v_size:(vertical+1)*v_size] = img_back
    return transformed

def rgb2gray(rgb):
    return numpy.dot(rgb[...,:3],[0.299, 0.587, 0.144])


print("Loading Images")
#images, classes = loader.loadProblematicImagesAndClasses()
images, classes = loader.loadUniqueTrainingAndClasses()
amount = len(images)

print("Preprocessing thumbs")
thumbsize = 50
thumbs = [misc.imresize(x,(thumbsize, thumbsize)) for x in images]
#thumbs = [iop.normalizeImage(x) for x in thumbs]
thumbs = [rgb2gray(x) for x in thumbs]   


print("Feature Extraction")
features = []
for i in range(amount):
    if(i%10 ==0):print(i, "/", amount)
    #colors = extractor.normalizedColorFeatures(thumbs[i][5:45,5:45,:])
    #angles = extractor.weightedAngleFeatures(thumbs[i][5:45,5:45,:], 11)
    #differences = extractor.pixelDifferences(thumbs[i][15:45,15:45,:])
    #feature = numpy.concatenate((colors, angles, differences))
    frequency = extractor.frequencyFeatures(thumbs[i],selectedclasses=[22,23])[1::8]
    features.append(frequency)


print("Evaluating")

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