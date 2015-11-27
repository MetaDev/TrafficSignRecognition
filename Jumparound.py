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
from skimage import exposure, color

from matplotlib import pyplot as plot
from sknn.mlp import Classifier, Convolution, Layer
import image_operations
      

#to install sknn: pip install scikit-neuralnetwork      
#docs: http://scikit-neuralnetwork.readthedocs.org/en/latest/module_mlp.html
print("Loading images")
#images, classes = loader.loadProblematicImagesAndClasses()
images, classes = loader.loadUniqueTrainingAndClasses()

amount = len(images)


print("Making thumbnails")


size=50
thumbs = [scipy.misc.imresize(img, (size, size)) for img in images]
thumbs= [image_operations.normalizeImage(img)/255 for img in thumbs]
images=[image_operations.normalizeImage(img)/255 for img in images]
thumbs=numpy.array(thumbs)
images=numpy.array(images)
#save classes as ints 
#create map from class->index 
unique_classes = list(set(classes))
int_unique_classes={unique_classes[i]:i for i in range(len(unique_classes))}

#convert classes with map
int_classes=[int_unique_classes[cl] for cl in classes]
int_classes=numpy.array(int_classes)

#normalize image
##image should be resized to square
#imageGray = color.rgb2gray(image)
##normalize
#imageGray=exposure.equalize_hist(imageGray)
print("Calculating neural net")

nn = Classifier(
    layers=[
        Convolution("Rectifier", channels=8, kernel_shape=(3,3,3)),
        Layer("Softmax")],
    learning_rate=0.02,
    n_iter=20)
nn.fit(thumbs, int_classes)


y_example = nn.predict(thumbs)

print(numpy.unique(y_example))

#print("Producing KFold indexes")
#kfold = cv.KFold(amount, n_folds = 5, shuffle = True)
#
#print("K-fold prediction score")
#score = cross_validation.cross_val_score(nn, thumbs, int_classes, cv=3)
#print(score)
#print(numpy.mean(score),numpy.std(score))
#
#print("K-fold log loss prediction score")
#scores = score_calculation.loglossKFold(thumbs,int_classes,nn,8)
#print(scores)
#print(numpy.mean(scores),' ' ,numpy.std(scores))



print('\a')
