# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 12:36:35 2015

@author: Rian
"""

from scipy import ndimage
from scipy import misc
import image_operations as op
import data_loading as loader
import feature_extraction as extractor
from matplotlib import pyplot as plot
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy


wantedClasses = 5
wantedImages = 5

numpy.random.seed(1234)

images, classes = loader.loadUniqueTrainingAndClasses()

classAmount = len(numpy.unique(classes))

relevantClasses = numpy.unique(classes)[[numpy.random.permutation(range(classAmount))[0:wantedClasses]]]
relevantImages = numpy.array(list(map(lambda c: images[classes == c][0:wantedImages], relevantClasses)))

blue = [40, 100, 160]
red = [150, 30, 25]
white = [255,255,255]
black = [0,0,0]
colors = numpy.array([blue, red, white, black])
def coolbeans(image):
    newImage = image.copy()
    for i in range(len(newImage)):
        for j in range(len(newImage[i])):
            distances = numpy.apply_along_axis(numpy.linalg.norm,1,image[i,j] - colors)
            newImage[i,j] = colors[numpy.argmin(distances)]
    return newImage
    
def normalize_vector(vector):
    minimum = numpy.min(vector)
    maximum = numpy.max(vector)
    return numpy.array([(v - minimum) * 255 / (maximum - minimum) for v in vector])
    
def normalize(image):
    reds = image[:,:,0]
    greens = image[:,:,1]
    blues = image[:,:,2]
    newReds = normalize_vector(reds) * 255
    newGreens = normalize_vector(greens) * 255
    newBlues = normalize_vector(blues) * 255
    return numpy.dstack((newReds, newGreens, newBlues)).astype(numpy.uint8)
    
def transform(image):
    resized = misc.imresize(image, (60,60), interp = 'nearest')[5:55,5:55,:]
    return op.normalizeImage(resized) #op.normalizeImage(resized)
    #return misc.imresize(image, (50,50))

def plotTransformation(transformation):
    imagePlot = plot.figure()
    for i in range(wantedClasses):
        for j in range(len(relevantImages[i])):
            singlePlot = imagePlot.add_subplot(wantedClasses, wantedImages, 1 + i*wantedImages + j)
            singlePlot.imshow(transformation(relevantImages[i][j]))