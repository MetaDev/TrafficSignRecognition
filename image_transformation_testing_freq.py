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


wantedClasses = 10
wantedImages = 5

numpy.random.seed(1234)

images, classes = loader.loadTrainingAndClasses()

classAmount = len(numpy.unique(classes))

relevantClasses = numpy.unique(classes)[[numpy.random.permutation(range(classAmount))[0:wantedClasses]]]
relevantImages = numpy.array(list(map(lambda c: images[classes == c][0:wantedImages], relevantClasses)))

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
    thumb = op.normalizeImage(thumb)
    thumb = rgb2gray(thumb)
    transf = frequencyFeaturesImage(thumb,selectedclass = 21)
    return transf
    #return misc.imresize(image, (50,50))

imagePlot = plot.figure()
for i in range(wantedClasses):
    for j in range(len(relevantImages[i])):
        singlePlot = imagePlot.add_subplot(wantedClasses, wantedImages, 1 + i*wantedImages + j)
        singlePlot.imshow(transform(relevantImages[i,j]),cmap='gray')