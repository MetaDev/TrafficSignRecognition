# -*- coding: utf-8 -*-
"""

calculate black density on white
image is triple D array, last array [r,g,b]
"""
import numpy 
import math
import scipy 
from scipy import stats

from enum import Enum
import sklearn.cross_validation as cv
import matplotlib.image as mpimg
from matplotlib import pyplot as plot

# Rian code
from glob import glob
from pathlib import Path

def distance(a,b):
    return numpy.sum(numpy.square(numpy.add(a, numpy.multiply(b, -1))))
    
def errorRate(a,b):
    return numpy.sum(numpy.array(a) != numpy.array(b)) / len(a)


    
def kNearestNeighbour(k, xs, ys, x):
    distances = [distance(x, i) for i in xs] 
    indexes = numpy.argsort(distances)
    classes = [ys[indexes[i]] for i in range(k)]
    return stats.mode(classes)[0][0]
    
def extract(filename):
    image = mpimg.imread(filename)
    category = Path(filename).parent.name
    superCategory = Path(filename).parent.parent.name
    return (image, superCategory, category)
    

#reqad images as floats 0-1
    
def loadTrainingAndClasses():
    imagePaths = glob("train/*/*/*.png")
    return [mpimg.imread(x) for x in imagePaths], [Path(x).parent.name for x in imagePaths]
    
def loadTest():
    imagePaths = glob("test/*.png")
    return [mpimg.imread(x) for x in imagePaths]
# end code


#brightness sqrt( 0.299*R^2 + 0.587*G^2 + 0.114*B^2 )
#source: http://alienryderflex.com/hsp.html
#takes R,G,B in 0-1 range
def calcPixelBrightness(r,g,b):
    return 0.299*math.pow(r,2)+0.587*math.pow(g,2) + 0.114*math.pow(b,2)

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
def calculateDarktoBrightRatio(image, brightThreshhold, darkThreshhold, nrOfBlocks=1, interpolation=2):

    #TODO trim   
   
    height = len(image)
    width = len(image[0]) 

    #first calculate brightness for each pixel than resize array

    imageBrightness = numpy.zeros((height,width))
    # a possible improvement would be to check if we are calculating the density inside the sign or not
    # we want to ignore the environments influence on the density
    # TODO accelerate with numpy
    for i in range(height):
        for j in range(width):
           
            r = image[i, j, 0]
            g = image[i, j, 1]
            b = image[i, j, 2]
            #convert rgb to brightness
            imageBrightness[height-i-1][j]= calcPixelBrightness(r,g,b)
           
    #TODO, filter, only count very dark and bright pixels (threshhold) 
    #set everything to bright (1), dont consider pixels in the corners, ther'll rarely be a figure
           
    #use scyppy image resize to create blocks   
    reducedImageBrightness=scipy.misc.imresize(imageBrightness,(nrOfBlocks,nrOfBlocks),Interpolation(interpolation).name)/255   
    #reducedImageBrightness=scipy.ndimage.interpolation.zoom(imageBrightness,(nrOfBlocks/width,nrOfBlocks/height),order=interpolation)  

    return reducedImageBrightness
 
def filterClassFromImages(images,classes,className):
     return[images[i] for i in range(len(images)) if classes[i] == className]
            
print("Loading images")
images, classes = loadTrainingAndClasses()
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
    
thumbs = [scipy.misc.imresize(x,(25,25)) for x in images]

print("Calculating features")
nrOfBlocks=8
brightThreshhold=0.8
darkTreshhold=0.2
features = numpy.zeros((amount,nrOfBlocks,nrOfBlocks))
for i in range(amount):
    print(i, "/", amount)
    features[i] = calculateDarktoBrightRatio(thumbs[i],brightThreshhold,darkTreshhold,nrOfBlocks)
  


print("Producing KFold indexes")
kfold = cv.KFold(amount, n_folds = 10, shuffle = True)

print("Evaluating model with KFold")
counter = 0
errors  = numpy.zeros(len(kfold))
for train_index, test_index in kfold:
    print(counter)
    trainFeatures = [features[i] for i in train_index]
    trainClasses  = [classes[i] for i in train_index]
    testFeatures  = [features[i] for i in test_index]
    testClasses   = [classes[i] for i in test_index]
    
    predictedClasses = [kNearestNeighbour(10, trainFeatures, trainClasses, x) for x in testFeatures]
    errors[counter-1] = errorRate(testClasses, predictedClasses)
    print(errors[counter-1])
    counter = counter + 1
    
print("mean error ", errors.mean())
