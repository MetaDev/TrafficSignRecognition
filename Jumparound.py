# -*- coding: utf-8 -*-
"""
test file
"""
import numpy 
import math
import scipy 
from scipy import stats

import score_calculation

from skimage import exposure

import data_loading as loader
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.lda import LDA

from sklearn import cross_validation
from enum import Enum
import sklearn.cross_validation as cv
import matplotlib.image as mpimg
from matplotlib import pyplot as plot
import feature_extraction
import image_operations
from skimage.feature import hog
from skimage import data, color, exposure
from pybrain.structure import FeedForwardNetwork

#mode 0
#brightness sqrt( 0.299*R^2 + 0.587*G^2 + 0.114*B^2 )
#source: http://alienryderflex.com/hsp.html
#mode 1
#special mode for traffic signs

def calcPixelBrightness(r,g,b,rWeight=0.299,gWeight=0.587,bWeight=0.114):
    return rWeight*math.pow(r,2)+gWeight*math.pow(g,2) + bWeight*math.pow(b,2)
   
#TODO: http://pybrain.org/docs/tutorial/intro.html
   
def calcHOG(image,orient=8,nr_of_cells_per_image=6,nr_of_cells_per_block=2):
    #image should be resized to square
   imageGray = color.rgb2gray(image)
   #normalize
   imageGray=exposure.equalize_hist(imageGray)
   
   height = len(image)
   width = len(image[0]) 
   ppc=(height/nr_of_cells_per_image,width/nr_of_cells_per_image)
   cpb=(nr_of_cells_per_block, nr_of_cells_per_block)
   fd = hog(imageGray, orientations=orient, pixels_per_cell=ppc,cells_per_block=cpb)
   return numpy.array(fd).flatten()


class Interpolation(Enum):
    nearest = 0
    bilinear = 1
    bicubic = 2
    cubic = 3
#the white threshhold is propably more important as white is more susceptible to different lighting
#Interpolation to use for re-sizing (‘nearest’, ‘bilinear’, ‘bicubic’ or ‘cubic’).
def calculateDarktoBrightRatio(image, nrOfBlocks=6, interpolation=2, trimBorderFraction=0.1):

  
    height = len(image)
    width = len(image[0]) 
    #trim borders of the image 
    image=image[height*(trimBorderFraction): height-height*(trimBorderFraction), width*(trimBorderFraction): width-width*(trimBorderFraction), :]

    height = len(image)
    width = len(image[0]) 
    
    imageBrightness = numpy.zeros((height,width))
    #first calculate brightness for each pixel than resize array
    for i in range(height):
        for j in range(width):
           
            r = image[i, j, 0]
            g = image[i, j, 1]
            b = image[i, j, 2]
            #convert rgb to brightness
            imageBrightness[height-i-1][j]= calcPixelBrightness(r,g,b)
           
    #normalise feature using its histogram
    imageBrightness=exposure.equalize_hist(imageBrightness)

    #reduce resolution     
    #use scyppy image resize to create blocks   
    reducedImageBrightness=scipy.misc.imresize(imageBrightness,(nrOfBlocks,nrOfBlocks),Interpolation(interpolation).name)   
   
    #normalize and flatten
    return numpy.array(reducedImageBrightness/255.0).flatten()
 
def filterClassFromImages(images,classes,className):
     return[images[i] for i in range(len(images)) if classes[i] == className]
            
print("Loading images")
#images, classes = loader.loadProblematicImagesAndClasses()
images, classes = loader.loadUniqueTrainingAndClasses()

amount = len(images)


print("Making thumbnails")


size=50
thumbs = [scipy.misc.imresize(img, (size, size)) for img in images]


print("Calculating features")
features=[]
for i in range(amount):
    print (str(i)+'/'+str(amount))
    HOG=calcHOG(thumbs[i])    
    bright=calculateDarktoBrightRatio(thumbs[i])
    feature=numpy.concatenate((HOG,bright))
    features.append(feature)
  
features=numpy.array(features)

#reduce feature dimensionality
nr_of_features=25
#LDA gives best results but warning: Variables are collinear
reduction_type=1
if(reduction_type==0):
    # Principal Components
    pca = PCA(n_components=nr_of_features)
    features=pca.fit_transform(features)
elif(reduction_type==1):
    # Linear Discriminant Analysis
    lda = LDA(n_components=nr_of_features)
    features=lda.fit_transform(features,classes)


print("Producing KFold indexes")
kfold = cv.KFold(amount, n_folds = 5, shuffle = True)
model = svm.SVC(kernel='linear')


print("K-fold prediction score")
score = cross_validation.cross_val_score(model, features, classes, cv=kfold)
print(score)
print(numpy.mean(score),numpy.std(score))

print("K-fold log loss prediction score")
model = svm.SVC(kernel='poly',degree=2,probability=True)
scores = score_calculation.loglossKFold(features,classes,model,8)
print(scores)
print(numpy.mean(scores),' ' ,numpy.std(scores))



print('\a')
