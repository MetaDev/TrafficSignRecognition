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

numpy.random.seed(123)

images, classes = loader.loadUniqueTrainingAndClasses()

classAmount = len(numpy.unique(classes))

relevantClasses = numpy.unique(classes)[[numpy.random.permutation(range(classAmount))[0:wantedClasses]]]
relevantImages = numpy.array(list(map(lambda c: images[classes == c][0:wantedImages], relevantClasses)))

def transform(image):
    resized = ndimage.imres
    return op.normalizeImage(image)

imagePlot = plot.figure()
for i in range(wantedClasses):
    for j in range(len(relevantImages[i])):
        singlePlot = imagePlot.add_subplot(wantedClasses, wantedImages, i + j*wantedClasses)
        singlePlot.imshow(transform(relevantImages[i][j]))