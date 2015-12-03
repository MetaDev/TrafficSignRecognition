# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 16:02:41 2015

@author: Rian
"""

import feature_validation as validation
import feature_extraction as extraction
import image_operations as operations
import data_loading as loader
import util
import numpy
import csv_output
from sklearn import lda, svm
from skimage import feature, color, exposure
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier


def calcHOG(image,orient=8,nr_of_cells_per_image=6,nr_of_cells_per_block=2, normalise = False):
   height = len(image)
   width = len(image[0]) 
   ppc=(height/nr_of_cells_per_image,width/nr_of_cells_per_image)
   cpb=(nr_of_cells_per_block, nr_of_cells_per_block)
   fd = feature.hog(image, orientations=orient, pixels_per_cell=ppc,cells_per_block=cpb, normalise = normalise)
   return numpy.array(fd).flatten()
   
print("loading data...")
size = 100
images, labels, classes = loader.loadTrainingImagesPoleNumbersAndClasses()
amount = len(images)

print("resizing...")
resized = util.loading_map(lambda x : operations.cropAndResize(x, 0, size), images)

print("grayscaling...")
resized = util.loading_map(color.rgb2gray, resized)

print("reshaping to array...")
size1_F = len(resized[0])
size2_F = len(resized[0][0])
#♥print("size:",size1_F,",",size2_F)
array = numpy.array(resized[0])
reshaped = numpy.reshape(array,(size1_F*size2_F))
reshaped = [reshaped]
for i in range(1,amount):
    size1 = len(resized[i])
    size2 = len(resized[i][0])
    if(size1*size2 != size1_F*size2_F):print("size:",size1,",",size2)
    if(i%100 == 0):print(i,"/",amount)
    array = numpy.array(resized[i])
    a = numpy.reshape(array,(size1*size2))
    reshaped = numpy.concatenate((reshaped,[a]),0)
    
#♠print("dimensions reshaped:",len(reshaped),",",len(reshaped[0]),",",len(reshaped[0][0]))
    

#model = svm.SVC(kernel='linear', probability = True)
from sklearn import random_projection
print("model")
model = Pipeline([
    #("standard scaler", StandardScaler()),
    #MinMaxScaler((-1,1)),
    #("principal component analysis", PCA(192)), #<- appears to reduce efficiency
    #("lda projection", lda.LDA(n_components = 80)),
    #("gaussian random projection", random_projection.GaussianRandomProjection(n_components = 150)),
    #("sparse random projection", random_projection.SparseRandomProjection(n_components = 350)),
    ("Multi-layer Perceptron", MLPClassifier(algorithm='l-bfgs', hidden_layer_sizes=(108,108), random_state=1,learning_rate='constant'))
    #("svm", svm.SVC(kernel = "sigmoid", C = 1000, gamma = 0.0001, probability = True))
    ])
    
n_folds = 5
print("Evaluating featureless covnet")
validation.validate_feature(reshaped, labels, classes, model, n_folds, False, False, True, True)
print('\a')
