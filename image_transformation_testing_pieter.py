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
import copy
import math

wantedClasses = 5
wantedImages = 5

#numpy.random.seed(55)

images, classes = loader.loadUniqueTrainingAndClasses()

classAmount = len(numpy.unique(classes))

relevantClasses = numpy.unique(classes)[[numpy.random.permutation(range(classAmount))[0:wantedClasses]]]
relevantImages = numpy.array(list(map(lambda c: images[classes == c][0:wantedImages], relevantClasses)))

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

def frequencyFeaturesImage(image, frequencyclasses = 25, subsect_v = 10, subsect_h=10, selectedclass = 24):
    thumbsize = len(image)
    transformed = numpy.zeros(( len(image) , len(image[0]) ))    
    for subsection in range(subsect_v*subsect_h):
        horizontal = subsection % subsect_h
        vertical = math.floor(subsection/subsect_v)
        h_size = thumbsize/subsect_h
        v_size = thumbsize/subsect_v
        subthumb = image[horizontal*h_size:(horizontal+1)*h_size,vertical*v_size:(vertical+1)*v_size]
        fthumb = numpy.fft.fft2(subthumb)  #fourier transform
        fshift = numpy.fft.fftshift(fthumb) #shift 0 frequency to center
        index = 0
        fc = selectedclass
        m = mask_frequency(fshift,thumbsize,frequencyclasses,fc) #select frequency components of this class
        f_ishift = numpy.fft.ifftshift(m)                        #inverse shift
        img_back = numpy.fft.ifft2(f_ishift)                     #inverse transform
        img_back = numpy.abs(img_back)
        index += 1
        transformed [horizontal*h_size:(horizontal+1)*h_size,vertical*v_size:(vertical+1)*v_size] = img_back
    return transformed

def rgb2gray(rgb):
    return numpy.dot(rgb[...,:3],[0.299, 0.587, 0.144])

def transform(image):
    thumbsize = 50
    thumb = misc.imresize(image,(thumbsize, thumbsize),interp='nearest')
    #thumb = op.normalizeImage(thumb)
    thumb = rgb2gray(thumb)
    
    return thumb


imagePlot = plot.figure()


singlePlot = imagePlot.add_subplot(1, 6, 1)
singlePlot.imshow(relevantImages[0][0])

transformed = transform(relevantImages[0][0])

singlePlot3 = imagePlot.add_subplot(1, 6, 2)
frequ = frequencyFeaturesImage(transformed,selectedclass = 24)
singlePlot3.imshow(frequ, cmap='gray')

singlePlot4 = imagePlot.add_subplot(1, 6, 3)
frequ = frequencyFeaturesImage(transformed,selectedclass = 23)
singlePlot4.imshow(frequ, cmap='gray')

singlePlot5 = imagePlot.add_subplot(1, 6, 4)
frequ = frequencyFeaturesImage(transformed,selectedclass = 22)
singlePlot5.imshow(frequ, cmap='gray')

singlePlot5 = imagePlot.add_subplot(1, 6, 5)
frequ = frequencyFeaturesImage(transformed,selectedclass = 21)
singlePlot5.imshow(frequ, cmap='gray')

singlePlot5 = imagePlot.add_subplot(1, 6, 6)
frequ = frequencyFeaturesImage(transformed,selectedclass = 20)
singlePlot5.imshow(frequ, cmap='gray')

#for i in range(wantedClasses):
#    for j in range(len(relevantImages[i])):
#        singlePlot = imagePlot.add_subplot(wantedClasses, wantedImages, j + i*wantedClasses +1)
#        transformedImage = transform(relevantImages[i][j])
#        singlePlot.imshow(transformedImage, cmap='gray')