# -*- coding: utf-8 -*-
"""

calculate black density on white
image is triple D array, last array [r,g,b]
"""
import numpy 
import math
import scipy 
from scipy import stats

from skimage import exposure

import data_loading as loader
from sklearn import neighbors
from sklearn import cross_validation
from enum import Enum
import sklearn.cross_validation as cv
import matplotlib.image as mpimg
from matplotlib import pyplot as plot

#mode 0
#brightness sqrt( 0.299*R^2 + 0.587*G^2 + 0.114*B^2 )
#source: http://alienryderflex.com/hsp.html
#mode 1
#special mode for traffic signs

def calcPixelBrightness(r,g,b,mode):
    if mode == 0:
        return 0.299*math.pow(r,2)+0.587*math.pow(g,2) + 0.114*math.pow(b,2)
    if mode == 1:
        return 0.299*math.pow(r,2)+ 0.114*math.pow(b,2)

"""more imporvements
calculate average brightness of image to remove the lighting effect and calcualte brightness and darkness 
thershhold based on that (the brightness distr could also be calculated and used)
to only calculate the most important pixels the image could also be trimmed around the borders to redude environment influence
better yet wuld be edge detection on the sign

what we calculate is actually a filtered and reduced image (binary) of the images brightness, for the purpose of differentiating 
sign with similar form but different figure on it

visualise this

use image resize to calculate blocks

"""

class Interpolation(Enum):
    nearest = 0
    bilinear = 1
    bicubic = 2
    cubic = 3
#the white threshhold is propably more important as white is more susceptible to different lighting
#Interpolation to use for re-sizing (‘nearest’, ‘bilinear’, ‘bicubic’ or ‘cubic’).
def calculateDarktoBrightRatio(image,brightnessMode=0, maxDarkLevel=0.1,minBrightLevel=0.9, nrOfBlocks=1, interpolation=2, trimBorderFraction=0):

  
    height = len(image)
    width = len(image[0]) 
    #TODO trim corners, rarely in image there
    #trim borders of the image 
    image=image[height*(trimBorderFraction): height-height*(trimBorderFraction), width*(trimBorderFraction): width-width*(trimBorderFraction), :]

   
    
    height = len(image)
    width = len(image[0]) 
    
    imageBrightness = numpy.zeros((height,width))
    
    #first calculate brightness for each pixel than resize array
    # TODO accelerate with numpy
    for i in range(height):
        for j in range(width):
           
            r = image[i, j, 0]
            g = image[i, j, 1]
            b = image[i, j, 2]
            #convert rgb to brightness
            imageBrightness[height-i-1][j]= calcPixelBrightness(r,g,b,brightnessMode)
           
           
    #normalise feature using its histogram
    imageBrightness=exposure.equalize_hist(imageBrightness)
    
    #TODO use edge detection (high difference in brightness) to calculate trim and to calculate the mask for the histogram only covering the sign and not environment
    #imageBrightnessHist, imageBrightnessBinEdges = numpy.histogram(imageBrightness.flatten(),bins=nrOfBrightnessLevels,density=True)    
    
    #TODO, filter, only count very dark and bright pixels (threshhold) 
  
    #reduce resolution     
    #use scyppy image resize to create blocks   
    reducedImageBrightness=scipy.misc.imresize(imageBrightness,(nrOfBlocks,nrOfBlocks),Interpolation(interpolation).name)   
   
    #flatten
    return reducedImageBrightness.flatten()
 
def filterClassFromImages(images,classes,className):
     return[images[i] for i in range(len(images)) if classes[i] == className]
            
print("Loading images")
#images, classes = loader.loadProblematicImagesAndClasses()
images, classes = loader.loadTrainingAndClasses()

amount = len(images)

"""
print("Plot feature")
plot.close("all")
nrOfBlocks=8
brightThreshhold=0.8
darkTreshhold=0.2
classA = filterClassFromImages(images,classes,"B1")
classB = filterClassFromImages(images,classes,"B3")

feature = calculateDarktoBrightRatio(classA[0],brightThreshhold,darkTreshhold,nrOfBlocks)
fig, (ax1, ax2) = plot.subplots(1,2)

imgplot = ax1.pcolor(feature)
imgplot = ax2.imshow(classA[0])
feature1 = calculateDarktoBrightRatio(classB[1],brightThreshhold,darkTreshhold,nrOfBlocks)

fig1, (ax11, ax12) = plot.subplots(1,2)
imgplot = ax11.pcolor(feature1)
plot.colorbar(imgplot)
imgplot = ax12.imshow(classB[1])
plot.show()

"""
print("Making thumbnails")
def resizeProper(image, maxPixels):
    ratio = len(image) / len(image[0])
    height = int(numpy.sqrt(maxPixels / ratio))
    width = int(ratio * height)
    return scipy.misc.imresize(image, (width, height))
    
thumbs = [resizeProper(x, 200) for x in images]

print("Calculating features")
nrOfBlocks=8
brightnessMode=1
brightThreshhold=0.8
darkTreshhold=0.1
interp=3
border=0.17
reducedFeatureRatio=1
features = []
for i in range(amount):
    print(i, "/", amount)
    ratio=calculateDarktoBrightRatio(thumbs[i],brightnessMode,brightThreshhold,darkTreshhold,nrOfBlocks,interpolation=interp,trimBorderFraction=border)[1::reducedFeatureRatio]
    features.append(ratio)
  
features=numpy.array(features)

print("Producing KFold indexes")
kfold = cv.KFold(amount, n_folds = 5, shuffle = True)
model = neighbors.KNeighborsClassifier(n_neighbors = 1)
score = cross_validation.cross_val_score(model, features, classes, cv=kfold)
print(score)
print(score.mean())

predictions = cross_validation.cross_val_predict(model, features, classes, cv = kfold)
wrongIndexes = numpy.nonzero(predictions != classes)
uniqueWrongs, counts = numpy.unique(numpy.append(predictions[[wrongIndexes]], numpy.array(classes)[[wrongIndexes]]), return_counts = True)
wrongs = uniqueWrongs[counts > 10]
print(wrongs)

print('\a')
