# -*- coding: utf-8 -*-
"""

calculate black density on white
image is triple D array, last array [r,g,b]
"""
import numpy 
import math
import scipy 
from scipy import stats

import score_calculation

from skimage import exposure
from sklearn import lda
import data_loading as loader
from sklearn import svm

from sklearn import cross_validation
from enum import Enum
import sklearn.cross_validation as cv
import matplotlib.image as mpimg
from matplotlib import pyplot as plot
import feature_extraction
import image_operations

#mode 0
#brightness sqrt( 0.299*R^2 + 0.587*G^2 + 0.114*B^2 )
#source: http://alienryderflex.com/hsp.html
#mode 1
#special mode for traffic signs

def calcPixelBrightness(r,g,b,rWeight=0.299,gWeight=0.587,bWeight=0.114):
    return rWeight*math.pow(r,2)+gWeight*math.pow(g,2) + bWeight*math.pow(b,2)
   

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
def calcEdgeSum(image):
    angle, magn = image_operations.calculatePixelAngleAndMagnitude(image)
    print(angle)
    plot.imshow(angle)

    
    
class Interpolation(Enum):
    nearest = 0
    bilinear = 1
    bicubic = 2
    cubic = 3
#the white threshhold is propably more important as white is more susceptible to different lighting
#Interpolation to use for re-sizing (‘nearest’, ‘bilinear’, ‘bicubic’ or ‘cubic’).
def calculateDarktoBrightRatio(image, maxDarkLevel=0.1,minBrightLevel=0.9, nrOfBlocks=1, interpolation=2, trimBorderFraction=0):

  
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
            imageBrightness[height-i-1][j]= calcPixelBrightness(r,g,b)
           
           
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


print("Making thumbnails")


size=50
thumbs = [scipy.misc.imresize(img, (size, size)) for img in images]

calcEdgeSum(thumbs[0])
plot.imshow(thumbs[0])
#print("Calculating features")
#nrOfBlocks=8
#brightThreshhold=0.8
#darkTreshhold=0.1
#interp=3
#border=0.17
#reducedFeatureRatio=1
#features = []
#for i in range(amount):
#    #print(i, "/", amount)
#    #ratio = calculateDarktoBrightRatio(thumbs[i],brightThreshhold,darkTreshhold,nrOfBlocks,interpolation=interp,trimBorderFraction=border)[1::reducedFeatureRatio]
#    #color =  feature_extraction.splitColorFeatures(thumbs[i],5)
#    
#    feature=numpy.append(ratio,color)
#    features.append(feature)
#  
#features=numpy.array(features)
#
#print("Producing KFold indexes")
#kfold = cv.KFold(amount, n_folds = 5, shuffle = True)
#model = svm.SVC(kernel='linear')
#
#
#print("K-fold prediction score")
#score = cross_validation.cross_val_score(model, features, classes, cv=kfold)
#print(score)
#print(score.mean())
#
#print("K-fold log loss prediction score")
#model = svm.SVC(kernel='poly',degree=2,probability=True)
#scores = score_calculation.loglossKFold(features,classes,model,8)
#print(scores)
#print(numpy.mean(scores),' ' ,numpy.std(scores))
#
#predictions = cross_validation.cross_val_predict(model, features, classes, cv = kfold)
#wrongIndexes = numpy.nonzero(predictions != classes)
#uniqueWrongs, counts = numpy.unique(numpy.append(predictions[[wrongIndexes]], numpy.array(classes)[[wrongIndexes]]), return_counts = True)
#print('average misclassifications, std misclassifications')
#print(numpy.mean(uniqueWrongs),' ' ,numpy.std(uniqueWrongs))
#print('Classes with more than 5 misclassifications')
#wrongs = uniqueWrongs[counts > 5]
#print(wrongs)
#
#print('\a')
