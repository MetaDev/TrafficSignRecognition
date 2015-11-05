# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 21:51:11 2015

@author: Rian
"""
import sys
import numpy
from scipy import signal
from scipy import misc
from skimage import exposure
from matplotlib import pyplot as plot

def cropAndResize(image, crop_percentage, size):
    big_size = int(size * (1 + 2*crop_percentage))
    crop_size = int(crop_percentage * size)
    resized = misc.imresize(image, (big_size, big_size), interp = 'nearest')
    cropped = resized[crop_size : crop_size + size, crop_size : crop_size + size, :]
    return cropped

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
    
    
def kMeanColors(image,K=4):
    
    max_iterations = 50    
    
    difference = 1
    iterations = 0
    
    rows = len(image)
    cols = len(image[0])
    
    N = rows*cols #dit is het aantal pixels in de thumbnail
    S = len(image[0][0]) #dit is in principe 3 (voor rgb), maar ge weet nooit
    
    x = numpy.reshape(image,(N,S))

    mu = numpy.zeros((K,S))
    for k in range(K):
        row = numpy.random.randint(0,rows)
        col = numpy.random.randint(0,cols)
        mu[k] = image[row][col]
      
    
    previous = sys.maxsize
    
    r = calcR(mu,x)
    
    while difference>0 and iterations < max_iterations:
        print("iteration:",iterations)
        mu = calcMu(r,x)        
        r = calcR(mu,x)
        current = objectiveFunction(mu,r,x)
        #print("current",current)
        difference = previous - current
        print("current",current)
        previous = current
        iterations += 1

        result = numpy.zeros((rows,cols,S))    
    
        for row in range(rows):
            for col in range(cols):
                for k in range(K):
                    if r[row*cols+col][k]:
                        result[row][col] = mu[k]    
        plot.imshow(result.astype(int))
    
    return result
    
    
def calcMu(r,x):
    N = len(x)
    S = len(x[0])
    K = len(r[0])
    mu = numpy.zeros((K,S))
    for k in range(K):
        teller = numpy.zeros(3)
        noemer = 0
        for n in range(N):
            for s in range(S):
                teller[s] = teller[s] + x[n][s]*r[n][k]
            noemer = noemer + r[n][k]
        print("teller",teller)
        print("noemer",noemer)
        mu[k] = numpy.array(teller)/noemer
    print("mu",mu)
    return mu
    
def calcR(mu,x):
    K = len(mu)
    N = len(x)
    r = numpy.zeros((N,K))
    for n in range(N):
        index = -1
        D_min = 0
        D = 0
        for k in range(K):
            #print("x",x[n])
            #print("mu",mu[k])
            sub = numpy.subtract(x[n],mu[k])
            
            sub = numpy.square(sub)
            D = numpy.sum(sub)
            
            if index == -1 or D < D_min:
                D_min = D
                index = k
        r[n][index] = 1
    return r
            
def objectiveFunction(mu,r,x):
    K = len(mu)
    N = len(x)
    S = len(x[0])
    J = 0
    for n in range(N):
        for k in range(K):
            #print("xn (3)",x[n])
            #print("mu (3)",mu[k])
            sub = numpy.subtract(x[n],mu[k])
            
            sub = numpy.square(sub)
            dist = numpy.sum(sub)
            #print("dist:",dist)
            dist = dist*r[n][k]
            J += dist
    return J
        
    
    
    
    
    
    
    
    
    
    
    
    
    