# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 21:51:48 2015

@author: Rian
"""

import image_operations as op
import numpy as np
from scipy import stats

def calculateAngleFeatures(image, angleDetail = 4):
    height = len(image)
    width = len(image[0])
    angles, magnitudes = op.calculatePixelAngleAndMagnitude(image)
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
    return calculateColorFeatures(op.normalizeImage(image))
    
def calculateSpecialColorFeatures(image):
    return calculateColorFeatures(op.reduceColorSpace(image))
    
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
    angles, magnitudes = op.calculatePixelAngleAndMagnitude(image)
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