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
    
def angleFeatures(image, classAmount = 3, threshold = 100):
    angles, magnitudes = op.calculatePixelAngleAndMagnitude(image)
    classes = np.zeros(classAmount)
    total = 1
    for i in range(len(angles)):
        for j in range(len(angles[0])):
            angle = angles[i,j]
            if angle < 0:
                angle += np.pi
            angle = (angle + np.pi/4) % np.pi
            if magnitudes[i,j] > threshold:
                angleClass = angle / (np.pi / classAmount)
                if angleClass == classAmount:
                    angleClass -= 1
                classes[angleClass] += 1
                total += 1                
    return classes / total 
    
def angleColorFeatures(image, angleClassAmount = 3, angleMagnitudeThreshold = 100, colorScale = 1.0):
    return np.concatenate((angleFeatures(image, angleClassAmount, angleMagnitudeThreshold), calculateNormalizedColorFeatures(image) * colorScale / (255 * angleClassAmount)))
    
#Should probably be inside image_operations as this produces an image
def angleClasses(image, classAmount = 4, threshold = 100):
    angles, magnitudes = op.calculatePixelAngleAndMagnitude(image)
    result = angles.copy()
    for i in range(len(angles)):
        for j in range(len(angles[0])):
            angle = angles[i,j]
            if angle < 0:
                angle += np.pi
            angle = (angle + np.pi/4) % np.pi
            if magnitudes[i,j] > threshold:
                angleClass = angle / (np.pi / classAmount)
                if angleClass == classAmount:
                    angleClass -= 1
                result[i,j] = int(angleClass)
            else:
                result[i,j] = -1
    return result