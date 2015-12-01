# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 15:29:44 2015

@author: Harald
"""

from sklearn.ensemble import RandomForestClassifier
import numpy 
import scipy 
import data_loading as loader
from skimage import exposure, color

import feature_extraction
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import lda
import image_operations
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import feature_validation as validation
print("Loading images")
images, labels, classes = loader.loadTrainingImagesPoleNumbersAndClasses()

amount = len(images)


print("Making thumbnails")

size=50
thumbs = [image_operations.cropAndResize(img, 0.1,size) for img in images]

print("Extract features")



#the method needs square single channel images

thumbsGray = [color.rgb2gray(img)for img in thumbs]


HOGGray = [feature_extraction.calcHOGWrapper(img,orient=8,pixel_per_cell=5,nr_of_cells_per_block=2) for img in thumbsGray]

features = HOGGray

    
print("Build random forest with gray features")

n_folds = 5
#iterate over multiple tree numbers 100->400
for nr_of_trees in range(100,400,50):
    print(nr_of_trees)
    model = Pipeline([
         ("Random forest classifier", RandomForestClassifier(n_estimators=nr_of_trees,class_weight='balanced',n_jobs=4))
         ])
    print("Validating Features")
    validation.validate_feature(features, labels, classes, model, n_folds, True, True, True, True)

    model = Pipeline([
     ("Extreme forest classifier", ExtraTreesClassifier(n_estimators=nr_of_trees,class_weight='balanced',n_jobs=4))
    ])
    print("Validating Features")
    validation.validate_feature(features, labels, classes, model, n_folds, True, True, True, True)

