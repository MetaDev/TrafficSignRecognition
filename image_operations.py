# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 21:51:11 2015

@author: Rian
"""

import numpy
from scipy import signal

def normalizeImage(image):
    normalized = image / 1.0
    for i in range(len(image)):
        for j in range(len(image[0])):
            length = numpy.sqrt(image[i,j,0] ** 2 + image[i,j,1] ** 2 + image[i,j,2] ** 2)
            normalized[i,j,:] = [float(normalized[i,j,0]), float(normalized[i,j,1]), float(normalized[i,j,2])] / length
    return normalized
    
def calculatePixelAngleAndMagnitude(image):
    verticalKernel = numpy.matrix([[-1,-1,-1],[0,0,0],[1,1,1]])
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