# -*- coding: utf-8 -*-
"""

calculate black density on white
image is triple D array, last array [r,g,b]
"""
import numpy 
import math
from scipy import misc
import sklearn.cross_validation as cv

#from matplotlib import pyplot as plot
import data_loading as loader

# Rian code k-fold
def distance(a,b):
    return numpy.sum(numpy.square(numpy.add(a, numpy.multiply(b, -1))))
    
def errorRate(a,b):
    return numpy.sum(numpy.array(a) != numpy.array(b)) / len(a)

def nearestNeighbour(xs, ys, x):
    bestIndex = 0
    bestDistance = distance(xs[0],x)
    for i in range(len(xs)):
        d = distance(xs[i], x)
        if d < bestDistance:
            bestIndex = i
            bestDistance = d
    return ys[bestIndex]
    
def kNearestNeighbour(k, xs, ys, x):
    distances = [distance(x, i) for i in xs] 
    indexes = numpy.argsort(distances)
    classes = [ys[indexes[i]] for i in range(k)]
    return stats.mode(classes)[0][0]
# end code


#brightness sqrt( 0.299*R^2 + 0.587*G^2 + 0.114*B^2 )
#source: http://alienryderflex.com/hsp.html
#takes R,G,B in 0-1 range
def calcPixelBrightness(r,g,b):
    return 0.299*math.pow(r,2)+0.587*math.pow(g,2) + 0.114*math.pow(b,2)

#the white threshhold is propably more important as white is more susceptible to different lighting
def calculateDarktoBrightRatio(image, brightThreshhold, darkThreshhold, nrOfBlocks=1):
    height = len(image)
    width = len(image[0])
    brightInBlock = 0
    darkInBlock = 0
    previousBlockNrI=0
    previousBlockNrJ=0
    blockNrI=0
    blockNrJ=0
    densityBlocks = np.zeros((nrOfBlocks,nrOfBlocks))
    # a possible improvement would be to check if we are calculating the density inside the sign or not
    # we want to ignore the environments influence on the density
    # TODO accelerate with numpy
    for i in range(height):
        for j in range(width):
            #entering new block
            #save desnity in feature and reset counters
            blockNrI = math.floor((i/height)*nrOfBlocks)
            blockNrJ = math.floor((j/width)*nrOfBlocks)
            if (previousBlockNrJ!=blockNrI or previousBlockNrI!=blockNrJ):
                densityBlocks[previousBlockNrI,previousBlockNrJ]=darkInBlock/(brightInBlock+1)
                darkInBlock=0
                brightInBlock=0
           
            r = image[i, j, 0]/255
            g = image[i, j, 1]/255
            b = image[i, j, 2]/255
            #convert rgb to brightness
            brightness = calcPixelBrightness(r,g,b);
            #count the bright and dark pixels
            if(brightness>brightThreshhold):
                brightInBlock+=1
            elif(brightness<darkThreshhold):
                darkInBlock+=1
            #save previous block index    
            previousBlockNrI=blockNrI
            previousBlockNrJ=blockNrJ
           
                
                
    #normalise features over blocksize
    return densityBlocks/((height*width)/math.pow(nrOfBlocks,2))
    
print("Loading images")
images, classes = loader.loadTrainingAndClasses()
amount = len(images)

print("Making thumbnails")
def resizeProper(image, maxPixels):
    ratio = len(image) / len(image[0])
    height = int(numpy.sqrt(maxPixels / ratio))
    width = int(ratio * height)
    return misc.imresize(image, (width, height))
    
thumbs = [misc.imresize(x,(25,25)) for x in images]

print("Calculating features")
nrOfBlocks=6
brightThreshhold=0.8
darkTreshhold=0.2
features = numpy.zeros((amount,nrOfBlocks,nrOfBlocks))
for i in range(amount):
    print(i, "/", amount)
    features[i] = calculateDarktoBrightRatio(thumbs[i],brightThreshhold,darkTreshhold,nrOfBlocks)
#visualise stuff TODO
  
print(features[0])


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
