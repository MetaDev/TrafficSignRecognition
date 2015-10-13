# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 23:09:33 2015

@author: Rian
"""

import numpy as np
from matplotlib import pyplot as plot
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
from scipy import signal
from glob import glob
from pathlib import Path
from sklearn import cross_validation

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
    
def calculateColorFeatures(image):
    pixels = image.size
    features = np.zeros(3)
    for i in range(len(image)):
        for j in range(len(image[0])):
            length = np.sqrt(image[i,j,0] ** 2 + image[i,j,1] ** 2 + image[i,j,2] ** 2)
            features[0] += image[i,j,0] / (length + 1)
            features[1] += image[i,j,1] / (length + 1)
            features[2] += image[i,j,2] / (length + 1)
    return features / pixels
    
def normalize(image):
    normalized = image / 1.0
    for i in range(len(image)):
        for j in range(len(image[0])):
            length = np.sqrt(image[i,j,0] ** 2 + image[i,j,1] ** 2 + image[i,j,2] ** 2)
            normalized[i,j,:] = [float(normalized[i,j,0]), float(normalized[i,j,1]), float(normalized[i,j,2])] / length
    return normalized

testimage = ndimage.imread("train/blue_circles/D1a/00062_04919.png")
wut = ndimage.gaussian_filter(testimage, 2)

#plot.imshow(wut)

imagePaths = glob("train/*/*/*.png")
images = list(map(extract, imagePaths))
xs = list(map(lambda x: x[0], images))
ys = list(map(lambda x: x[1], images))

#xTrain, xTest, yTrain, yTest = cross_validation.train_test_split(xs, ys, test_size=0.4)

classA = list(filter(lambda i: i[1] == "stop", images))
classB = list(filter(lambda i: i[1] == "diamonds", images))
classC = list(filter(lambda i: i[1] == "reversed_triangles", images))

fig = plot.figure()
axes = fig.add_subplot(111, projection = '3d')

def plotFeatures(data, amount, color, marker):  
    counter = 0
    for image in data[0:amount]:
        print(counter)
        counter+=1
        angleFeatures = calculateAngleFeatures(image[0])
        colorFeatures = calculateColorFeatures(image[0])
        axes.scatter(colorFeatures[0], angleFeatures[2], colorFeatures[2], c=color,marker=marker)

#plotFeatures(classA, 10, 'r', 'o')
plotFeatures(classB, 50, 'b', '^')
plotFeatures(classC, 50, 'g', 'x')