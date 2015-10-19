# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 21:51:11 2015

@author: Rian
"""

import numpy
from scipy import signal
from skimage import exposure

def normalizeImage(image):
    reds = image[:,:,0].astype(float)
    greens = image[:,:,1].astype(float)
    blues = image[:,:,2].astype(float)
    lengths = numpy.sqrt(numpy.square(reds) + numpy.square(greens) + numpy.square(blues)) + 0.01
    return (numpy.dstack((reds/ lengths, greens/lengths, blues/lengths)) * 255).astype(numpy.uint8)
    
def correctWhiteBalance(image):
    reds   = image[:,:,0]
    greens = image[:,:,1]
    blues  = image[:,:,2]
    #balancedReds   = ((reds - reds.min()) / (reds.max() - reds.min() + 1) * 255).astype(numpy.uint8)
    #balancedGreens = ((greens - greens.min()) / (greens.max() - greens.min() + 1) * 255).astype(numpy.uint8)
    #balancedBlues  = ((blues - blues.min()) / (blues.max() - blues.min() + 1) * 255).astype(numpy.uint8)
    balancedReds   = exposure.equalize_hist(reds)
    balancedGreens = exposure.equalize_hist(greens)
    balancedBlues  = exposure.equalize_hist(blues)
    return (numpy.dstack([balancedReds, balancedGreens, balancedBlues]) * 255).astype(numpy.uint8)
    
def reduceColorSpace(image, threshold = 128):
    reduced = image.copy()
    for i in range(len(image)):
        for j in range(len(image[0])):
            if image[i,j,0] < threshold:
                reduced[i,j,0] = 0
            else:
                reduced[i,j,0] = 255
            if image[i,j,1] < threshold:
                reduced[i,j,1] = 0
            else:
                reduced[i,j,1] = 255
            if image[i,j,2] < threshold:
                reduced[i,j,2] = 0
            else:
                reduced[i,j,2] = 255
    return reduced
    
def calculatePixelAngleAndMagnitude(image):
    verticalKernel = numpy.matrix([[-1,-1,-1],[0,0,0],[1,1,1]])
    verticalR = signal.convolve2d(image[:,:,0], verticalKernel, boundary = 'symm')
    verticalG = signal.convolve2d(image[:,:,1], verticalKernel, boundary = 'symm')
    verticalB = signal.convolve2d(image[:,:,2], verticalKernel, boundary = 'symm')
    verticalDifference = numpy.mean(numpy.dstack([verticalR, verticalG, verticalB]),2)
    
    horizontalKernel = numpy.matrix([[-1,0,1],[-1,0,1],[-1,0,1]])
    horizontalR = signal.convolve2d(image[:,:,0], horizontalKernel, boundary = 'symm')
    horizontalG = signal.convolve2d(image[:,:,1], horizontalKernel, boundary = 'symm')
    horizontalB = signal.convolve2d(image[:,:,2], horizontalKernel, boundary = 'symm')
    horizontalDifference = numpy.mean(numpy.dstack([horizontalR, horizontalG, horizontalB]),2)

    angle = numpy.vectorize(numpy.arctan2)(verticalDifference, horizontalDifference)    
    magnitude = numpy.vectorize(lambda x,y: numpy.sqrt(x**2 + y**2))(verticalDifference, horizontalDifference)    
     
    return (angle, magnitude)