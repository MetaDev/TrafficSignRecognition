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
    features = np.zeros(3)
    features[0] = 3 * image[:,:,0].mean()
    features[1] = 3 * image[:,:,1].mean()
    features[2] = 3 * image[:,:,2].mean()
    return features
    
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
    concreteAngles = (angles + (angles < 0) * np.pi + np.pi/4) % np.pi
    angleClasses = (concreteAngles / (np.pi / classAmount)).astype(int)
    barrier = magnitudes > threshold
    filteredAngleClasses = (angleClasses + 1) * barrier
    importantAngleClasses = filteredAngleClasses[np.nonzero(filteredAngleClasses)] - 1
    counted = np.bincount(importantAngleClasses) / len(importantAngleClasses)
    return np.append(counted, np.zeros(classAmount - len(counted)))
    
def splitColorFeatures(image, splits = 3):
    features = np.zeros(3 * splits * splits)
    normalized = op.normalizeImage(image)
    width = len(normalized)
    height = len(normalized[0])
    for i in range(splits):
        for j in range(splits):
            index = (i*splits + j) * 3
            subImage = normalized[height/splits*i:height/splits*(i+1), width/splits*j:width/splits*(j+1), :]
            subFeatures = calculateColorFeatures(subImage)
            features[index] = subFeatures[0]
            features[index+1] = subFeatures[1]
            features[index+2] = subFeatures[2]
    return features
    
#Van Pieter
def quadrantAngleFeatures(image, angleclasses = 4, angleMagnitudeThreshold = 100):
    features = np.zeros(angleclasses*4)
    for quadrant in range(4):
        horizontal = quadrant % 2       #0 or 1 for which horizontal quadrant
        vertical = (quadrant / 2 ) >= 1 #0 or 1 for which vertical quadrant
        size = len(image)/2
        subthumb = image[horizontal*size:(horizontal+1)*size,vertical*size:(vertical+1)*size,:]
        features[quadrant*angleclasses:(quadrant+1)*angleclasses] = angleFeatures(subthumb, angleclasses, angleMagnitudeThreshold)  
    return features
    
#Combined features
def angleColorFeatures(image, angleClassAmount = 3, angleMagnitudeThreshold = 100, colorScale = 1.0):
    return np.concatenate((angleFeatures(image, angleClassAmount, angleMagnitudeThreshold), calculateNormalizedColorFeatures(image) * colorScale / (255 * angleClassAmount)))
    
def angleQuadrantAngleFeatures(image, angleClassAmount = 3, angleMagnitudeThreshold = 100):
    return np.concatenate((angleFeatures(image, angleClassAmount, angleMagnitudeThreshold), quadrantAngleFeatures(image, angleClassAmount, angleMagnitudeThreshold)))

def colorQuadrantAngleFeatures(image, angleClassAmount = 3, angleMagnitudeThreshold = 100, colorScale = 1.0):
    return np.concatenate((quadrantAngleFeatures(image, angleClassAmount, angleMagnitudeThreshold), calculateNormalizedColorFeatures(image) * colorScale / (255 * angleClassAmount)))
    
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