# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 16:32:02 2015

@author: Rian
"""

from scipy import ndimage
from scipy import signal
import numpy
from matplotlib import pyplot
import data_loading as loader
import feature_extraction as extractor
import image_operations as operations
import sklearn.cross_validation as cv
import numpy
from scipy import misc
from scipy import stats
from sklearn import neighbors
from sklearn import svm
from sklearn import cross_validation

rotatedImage = ndimage.imread("train/rectangles_up/F59/00283_01126.png")
improvedImage = ndimage.rotate(rotatedImage, 180)
#pyplot.imshow()
verticalKernel = numpy.matrix([[-1,-1,-1],[0,0,0],[1,1,1]])
horizontalKernel = numpy.matrix([[-1,0,1],[-1,0,1],[-1,0,1]])

def meanAngle(image):
    grayImage = numpy.mean(image, 2)
    verticalDifference = signal.convolve2d(grayImage, verticalKernel, boundary = 'symm')
    horizontalDifference = signal.convolve2d(grayImage, horizontalKernel, boundary = 'symm')
    vertical = verticalDifference.mean()
    horizontal = horizontalDifference.mean()
    angle = numpy.arctan2(vertical, horizontal)
    if angle < 0: angle += numpy.pi
    magnitude = numpy.sqrt(vertical ** 2 + horizontal ** 2)
    return angle, magnitude
    
def splitMeanAngle(image, splits = 2, threshold = 10):
    width = len(image)
    height = len(image[0])
    result = numpy.zeros([splits, splits])
    for i in range(splits):
        for j in range(splits):
            subImage = image[i * (width / splits) : (i + 1) * (width / splits), j * (height / splits) : (j+1) * (height / splits), :]
            angle, magnitude = meanAngle(subImage)
            result[i, j] = angle
            if magnitude < threshold:
                result[i,j] = -1
    return result
    
def splitMeanAngleFeatures(image, splits = 2, threshold = 10):
    result = splitMeanAngle(image, splits, threshold).flatten()
    return result

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
features = numpy.zeros([len(images), splits * splits])
for i in range(amount):
    if(i%10 ==0):print(i, "/", amount)
    features[i] = splitMeanAngleFeatures(thumbs[i], splits)
    
print("Producing KFold indexes")
kfold = cv.KFold(amount, n_folds = 5, shuffle = True)
model = neighbors.KNeighborsClassifier(n_neighbors = 1)
score = cross_validation.cross_val_score(model, features, classes, cv=kfold)
print(score)
print(score.mean())
    