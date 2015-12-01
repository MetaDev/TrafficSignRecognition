# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 21:51:48 2015

@author: Rian
"""

import image_operations as op
import numpy as np
import numpy
import copy
import math
import scipy
from scipy import stats
from enum import Enum
from scipy import signal
from scipy import misc
from skimage import exposure, color
from skimage.feature import hog

def pixel_features(image, size):
    return misc.imresize(image, (size, size)).flatten()

def split_image_features(feature, splits, image):
    features = []
    width = len(image)
    height = len(image[0])
    for i in range(splits):
        for j in range(splits):
            subImage = image[height/splits*i:height/splits*(i+1), width/splits*j:width/splits*(j+1), :]
            subFeatures = feature(subImage)
            features.append(subFeatures)
    return numpy.array(features).flatten()

def color_features(image, mean = True, std = False, skew = False, kurtosis = False):
    reds   = image[:,:,0].flatten()
    greens = image[:,:,1].flatten()
    blues  = image[:,:,2].flatten()
    features = numpy.array([])
    if mean:
        features = numpy.append(features, [reds.mean(), greens.mean(), blues.mean()])
    if std:
        features = numpy.append(features, [numpy.std(reds), numpy.std(greens), numpy.std(blues)])
    if skew:
        features = numpy.append(features, [stats.skew(reds), stats.skew(greens), stats.skew(blues)])
    if kurtosis:
        features = numpy.append(features, [stats.kurtosis(reds), stats.kurtosis(greens), stats.kurtosis(blues)])
    return features

def normalizedColorFeatures(image, mean = True, std = True, skew = True, kurtosis = True):
    return color_features(op.normalizeImage(image), mean, std, skew, kurtosis)
      
def colorDistances(image, r_steps = 2, g_steps = 2, b_steps = 2, std = False):
    features = []
    for r in range(r_steps):
        for g in range(g_steps):
            for b in range(b_steps):
                color = numpy.array([(r + 1) / r_steps,(g + 1) / g_steps,(b + 1) / b_steps]) * 255
                distances = numpy.linalg.norm(image - color, axis = 2)
                features.append(distances.mean())
                if std: features.append(numpy.std(distances))
    return numpy.array(features)
                  
def normalizedColorDistances(image, r_steps = 2, g_steps = 2, b_steps = 2):
    normalized = op.normalizeImage(image)
    features = []
    for r in range(r_steps):
        for g in range(g_steps):
            for b in range(b_steps):
                color = numpy.array([r,g,b]) / (numpy.linalg.norm([r,g,b]) + 0.01) * 255
                distances = numpy.linalg.norm(normalized - color, axis = 2)
                features.append(distances.mean())
                features.append(numpy.std(distances))
    return numpy.array(features)

def weightedAngleFeatures(image, classAmount = 3):
    angles, magnitudes = op.calculatePixelAngleAndMagnitude(image)
    concreteAngles = (angles + (angles < 0) * np.pi + np.pi/4) % np.pi
    angleClasses = (concreteAngles / (np.pi / classAmount)).astype(int)
    counted = numpy.zeros(classAmount)
    for i in range(classAmount):
        counted[i] = numpy.sum(numpy.multiply(angleClasses == i, magnitudes))
    return counted

verticalKernel = numpy.matrix([[-1,-1,-1],[0,0,0],[1,1,1]])
horizontalKernel = numpy.matrix([[-1,0,1],[-1,0,1],[-1,0,1]])
def pixelDifferences(image):
    verticalR = signal.convolve2d(image[:,:,0], verticalKernel, boundary = 'symm')
    verticalG = signal.convolve2d(image[:,:,1], verticalKernel, boundary = 'symm')
    verticalB = signal.convolve2d(image[:,:,2], verticalKernel, boundary = 'symm')
    
    horizontalR = signal.convolve2d(image[:,:,0], horizontalKernel, boundary = 'symm')
    horizontalG = signal.convolve2d(image[:,:,1], horizontalKernel, boundary = 'symm')
    horizontalB = signal.convolve2d(image[:,:,2], horizontalKernel, boundary = 'symm')
    return numpy.array([verticalR.mean(), numpy.std(verticalR), stats.skew(verticalR.flatten()), stats.kurtosis(verticalR.flatten()),
                        verticalG.mean(), numpy.std(verticalG), stats.skew(verticalG.flatten()), stats.kurtosis(verticalG.flatten()),
                        verticalB.mean(), numpy.std(verticalB), stats.skew(verticalB.flatten()), stats.kurtosis(verticalB.flatten()),
                        horizontalR.mean(), numpy.std(horizontalR), stats.skew(horizontalR.flatten()), stats.kurtosis(horizontalR.flatten()),
                        horizontalG.mean(), numpy.std(horizontalG), stats.skew(horizontalG.flatten()), stats.kurtosis(horizontalG.flatten()),
                        horizontalB.mean(), numpy.std(horizontalB), stats.skew(horizontalB.flatten()), stats.kurtosis(horizontalB.flatten())])

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
    
def splitAngleFeatures(image, splits = 3, angleClasses = 4, angleMagnitudeThreshold = 100):
    features = np.zeros(angleClasses * splits * splits)
    width = len(image)
    height = len(image[0])
    for i in range(splits):
        for j in range(splits):
            index = (i*splits + j) * angleClasses
            subImage = image[width/splits*i:width/splits*(i+1), height/splits*j:height/splits*(j+1), :]
            subFeatures = angleFeatures(subImage, angleClasses, angleMagnitudeThreshold)
            for k in range(angleClasses):
                features[index + k] = subFeatures[k]
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
    
def frequencyFeatures(image, frequencyclasses = 25, subsect_v = 4, subsect_h=4, selectedclasses = [22,23,24]):
    features = numpy.zeros([len(selectedclasses)*subsect_v*subsect_h])     #to save feature class frequencies
    thumbsize = len(image)    
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
    return features
    
#source: http://alienryderflex.com/hsp.html
def calcPixelBrightness(r,g,b,rWeight=0.299,gWeight=0.587,bWeight=0.114):
    return rWeight*math.pow(r,2)+gWeight*math.pow(g,2) + bWeight*math.pow(b,2)
    
class Interpolation(Enum):
    nearest = 0
    bilinear = 1
    bicubic = 2
    cubic = 3
    
def calculateDarktoBrightRatio(image, nrOfBlocks=6, interpolation=2, trimBorderFraction=0.1):
    height = len(image)
    width = len(image[0]) 
    #trim borders of the image 
    image=image[height*(trimBorderFraction): height-height*(trimBorderFraction), width*(trimBorderFraction): width-width*(trimBorderFraction), :]

    height = len(image)
    width = len(image[0]) 
    
    imageBrightness = numpy.zeros((height,width))
    #first calculate brightness for each pixel than resize array
    for i in range(height):
        for j in range(width):
           
            r = image[i, j, 0]
            g = image[i, j, 1]
            b = image[i, j, 2]
            #convert rgb to brightness
            imageBrightness[height-i-1][j]= calcPixelBrightness(r,g,b)
           
    #normalise feature using its histogram
    imageBrightness=exposure.equalize_hist(imageBrightness)

    #reduce resolution     
    #use scyppy image resize to create blocks   
    reducedImageBrightness=scipy.misc.imresize(imageBrightness,(nrOfBlocks,nrOfBlocks),Interpolation(interpolation).name)   
   
    #normalize and flatten
    return numpy.array(reducedImageBrightness/255.0).flatten()
    
#Combined features
def angleColorFeatures(image, angleClassAmount = 3, angleMagnitudeThreshold = 100, colorScale = 1.0):
    return np.concatenate((angleFeatures(image, angleClassAmount, angleMagnitudeThreshold), calculateNormalizedColorFeatures(image) * colorScale / (255 * angleClassAmount)))
    
def angleQuadrantAngleFeatures(image, angleClassAmount = 3, angleMagnitudeThreshold = 100):
    return np.concatenate((angleFeatures(image, angleClassAmount, angleMagnitudeThreshold), quadrantAngleFeatures(image, angleClassAmount, angleMagnitudeThreshold)))

def colorQuadrantAngleFeatures(image, angleClassAmount = 3, angleMagnitudeThreshold = 100, colorScale = 1.0):
    return np.concatenate((quadrantAngleFeatures(image, angleClassAmount, angleMagnitudeThreshold), calculateNormalizedColorFeatures(image) * colorScale / (255 * angleClassAmount)))
    
def splitAngleSplitColorFeatures(image, angleSplit = 3, angleClassAmount = 3, angleMagnitudeThreshold = 100, colorSplit = 3, colorScale = 1.0):
    return np.concatenate((splitAngleFeatures(image, angleSplit, angleClassAmount, angleMagnitudeThreshold), splitColorFeatures(image, colorSplit) * colorScale))    
    
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
    
def rgb2gray(rgb):
    return numpy.dot(rgb[...,:3],[0.299, 0.587, 0.144])
    
def calcHOG(image,orient=8,nr_of_cells_per_image=6,nr_of_cells_per_block=2,normalise=True):
    #image should be resized to square
   imageGray = color.rgb2gray(image)
   #normalize
   imageGray=exposure.equalize_hist(imageGray)
   
   height = len(image)
   width = len(image[0]) 
   ppc=(height/nr_of_cells_per_image,width/nr_of_cells_per_image)
   cpb=(nr_of_cells_per_block, nr_of_cells_per_block)
   fd = hog(imageGray, orientations=orient, pixels_per_cell=ppc,cells_per_block=cpb)
   return numpy.array(fd).flatten()

def calcHOGWrapper(image_single_channel,orient=8,pixel_per_cell=5,nr_of_cells_per_block=2,normalise=True):
    #normalize
    image_single_channel=exposure.equalize_hist(image_single_channel)
    ppc=(pixel_per_cell,pixel_per_cell)
    cpb=(nr_of_cells_per_block, nr_of_cells_per_block)
    fd = hog(image_single_channel, orientations=orient, pixels_per_cell=ppc,cells_per_block=cpb,normalise=normalise)
    return numpy.array(fd).flatten()
