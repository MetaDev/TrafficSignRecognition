# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 23:09:33 2015

@author: Rian
"""

import numpy as np
from scipy import ndimage
from scipy import signal
from scipy import stats
from glob import glob
from pathlib import Path

def extract(filename):
    image = ndimage.imread(filename)
    category = Path(filename).parent.name
    superCategory = Path(filename).parent.parent.name
    return (image, superCategory, category)
    
def calculatePixelAngleAndMagnitude(image):
    verticalKernel = np.matrix([[-1,-1,-1],[0,0,0],[1,1,1]])
    verticalR = signal.convolve2d(image[:,:,0], verticalKernel, boundary = 'symm')
    verticalG = signal.convolve2d(image[:,:,1], verticalKernel, boundary = 'symm')
    verticalB = signal.convolve2d(image[:,:,2], verticalKernel, boundary = 'symm')
    verticalDifference = np.mean(np.dstack([verticalR, verticalG, verticalB]),2)
    
    horizontalKernel = np.matrix([[-1,0,1],[-1,0,1],[-1,0,1]])
    horizontalR = signal.convolve2d(image[:,:,0], horizontalKernel, boundary = 'symm')
    horizontalG = signal.convolve2d(image[:,:,1], horizontalKernel, boundary = 'symm')
    horizontalB = signal.convolve2d(image[:,:,2], horizontalKernel, boundary = 'symm')
    horizontalDifference = np.mean(np.dstack([horizontalR, horizontalG, horizontalB]),2)

    angle = np.vectorize(np.arctan2)(verticalDifference, horizontalDifference)    
    magnitude = np.vectorize(lambda x,y: np.sqrt(x**2 + y**2))(verticalDifference, horizontalDifference)    
     
    return (angle, magnitude)
    
def calculateAngleFeatures(image, angleDetail = 4):
    height = len(image)
    width = len(image[0])
    angles, magnitudes = calculatePixelAngleAndMagnitude(image)
    features = np.zeros(angleDetail + 1)
    for i in range(height):
        for j in range(width):
            angleClass = angles[i,j]
            if angleClass < 0:
                angleClass = np.pi + angleClass
            angleClass = np.floor(angleClass / (np.pi / angleDetail))
            pixelAngle = angleClass % angleDetail
            if(magnitudes[i,j] > 128):
                features[pixelAngle] += 1
            else:
                features[angleDetail] += 1
    return features / (width * height)
    
def calculateNormalizedColorFeatures(image):
    pixels = image.size
    features = np.zeros(3)
    for i in range(len(image)):
        for j in range(len(image[0])):
            length = np.sqrt(image[i,j,0] ** 2 + image[i,j,1] ** 2 + image[i,j,2] ** 2)
            features[0] += image[i,j,0] / (length + 1)
            features[1] += image[i,j,1] / (length + 1)
            features[2] += image[i,j,2] / (length + 1)
    return features / pixels
    
def calculateColorFeatures(image):
    pixels = image.size
    features = np.zeros(3)
    for i in range(len(image)):
        for j in range(len(image[0])):
            features[0] += image[i,j,0]
            features[1] += image[i,j,1]
            features[2] += image[i,j,2]
    return features / pixels
    
def calculateAngleMoments(image):
    angles, magnitudes = calculatePixelAngleAndMagnitude(image)
    vectors = np.vectorize(lambda a, m: (m * np.cos(a), m * np.sin(a)))(angles, magnitudes)
    vectors = np.zeros([len(image), len(image[0]), 2])
    for i in range(len(image)):
        for j in range(len(image[0])):
            vectors[i,j,0] = magnitudes[i,j] * np.cos(angles[i,j])
            vectors[i,j,1] = magnitudes[i,j] * np.sin(angles[i,j])
    vectors = vectors.reshape([-1,2])
    meanVector = np.mean(vectors,0)
    sdVector = np.std(vectors,0)
    skewVector = stats.skew(vectors,0)
    kurtosisVector = stats.kurtosis(vectors,0)
    mean = np.arctan2(meanVector[1], meanVector[0])
    sd = np.arctan2(sdVector[1], sdVector[0])
    skew = np.arctan2(skewVector[1], skewVector[0])
    kurtosis = np.arctan2(kurtosisVector[1], kurtosisVector[0])
    return (mean, sd, skew, kurtosis)

def normalizeImage(image):
    normalized = image / 1.0
    for i in range(len(image)):
        for j in range(len(image[0])):
            length = np.sqrt(image[i,j,0] ** 2 + image[i,j,1] ** 2 + image[i,j,2] ** 2)
            normalized[i,j,:] = [float(normalized[i,j,0]), float(normalized[i,j,1]), float(normalized[i,j,2])] / length
    return normalized

def loadTrainingImages():
    imagePaths = glob("train/*/*/*.png")
    return list(map(extract, imagePaths))
    

