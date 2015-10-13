# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 23:58:29 2015

@author: Rian

Shows why normalizing colors before using them is interesting.
It helps for images taken in different lighting conditions.
First the script shows a dark and normal image without normalization
and with normalization. Then it shows that normalization produces better
clustering when used as features of the data set.

TODO: name some more stuff
"""

from test import *
from matplotlib import pyplot as plot
from mpl_toolkits.mplot3d import Axes3D

darkImage = ndimage.imread("train/diamonds/B9/00585_09701.png")
normalImage = ndimage.imread("train/diamonds/B9/00042_00233.png")

comparison = plot.figure()
standardDark = comparison.add_subplot(2,2,1)
standardDark.imshow(darkImage)
standardNormal = comparison.add_subplot(2,2,2)
standardNormal.imshow(normalImage)

normalDark = comparison.add_subplot(2,2,3)
normalDark.imshow(normalizeImage(darkImage))
normalNormal = comparison.add_subplot(2,2,4)
normalNormal.imshow(normalizeImage(normalImage))

images = loadTrainingImages()

classA = list(filter(lambda i: i[2] == "D1a", images))
classB = list(filter(lambda i: i[2] == "B9", images))
classC = list(filter(lambda i: i[2] == "B5", images))

fig = plot.figure()
axes = fig.add_subplot(1,2,1, projection = '3d', title = 'Using Colors', xlabel = 'Red Mean', ylabel = 'Green Mean', zlabel = 'Blue Mean')
axes2 = fig.add_subplot(1,2,2, projection = '3d', title = 'Using Normalized Colors', xlabel = 'Normalized Red Mean', ylabel = 'Normalized Green Mean', zlabel = 'Normalized Blue Mean')

def plotFeatures(data, amount, color, marker, label = ''):  
    counter = 0
    for image in data[0:amount]:
        print(counter)
        counter+=1
        #angleFeatures = calculateAngleFeatures(image[0])
        colorFeatures = calculateColorFeatures(image[0])
        #angleMoments = calculateAngleMoments(image[0])
        axes.scatter(colorFeatures[0],colorFeatures[1],colorFeatures[2], c=color, marker=marker)
    axes.plot([], [], marker, c=color, label=label)

plotFeatures(classA, -1, 'r', 'o', 'D1a')
plotFeatures(classB, -1, 'b', '^', 'B9')
plotFeatures(classC, -1, 'g', 'x', 'B5')
axes.legend(loc = 0, scatterpoints = 1)

def plotFeatures(data, amount, color, marker, label = ''):  
    counter = 0
    for image in data[0:amount]:
        print(counter)
        counter+=1
        #angleFeatures = calculateAngleFeatures(image[0])
        colorFeatures = calculateNormalizedColorFeatures(image[0])
        #angleMoments = calculateAngleMoments(image[0])
        axes2.scatter(colorFeatures[0],colorFeatures[1],colorFeatures[2], c=color, marker=marker)
    axes2.plot([], [], marker, c=color, label=label)

plotFeatures(classA, -1, 'r', 'o', 'D1a')
plotFeatures(classB, -1, 'b', '^', 'B9')
plotFeatures(classC, -1, 'g', 'x', 'B5')
axes2.legend(loc = 0, scatterpoints = 1)