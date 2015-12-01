from sklearn.ensemble import RandomForestClassifier
import numpy 
import scipy 
import data_loading as loader
from skimage import exposure, color

import feature_extraction
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import lda
import util

import image_operations
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import feature_validation as validation
import feature_extraction as extraction
import image_operations as operations
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

def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = numpy.arange(0, size, 1, float)
    y = x[:,numpy.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return numpy.exp(-4*numpy.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

   
#preloading
print("loading data...")
size = 50
images, labels, classes = loader.loadTrainingImagesPoleNumbersAndClasses()
amount = len(images)

print("resizing...")
resized = [image_operations.cropAndResize(img, 0.1,size) for img in images]

print("luv...")
luv =  [color.rgb2luv(img) for img in resized]
print("hed...")
hed =  [color.rgb2hed(img) for img in resized]

print("grayscaling...")
grayscaled =  [color.rgb2gray(img) for img in resized]
#print("edges...")

print("brightness features")
brightness = util.loading_map(extraction.calculateDarktoBrightRatio, resized)

print("luv features")

luv_features = util.loading_map(lambda x: extraction.split_image_features(
    lambda y : extraction.color_features(y, mean = True, std = True), 7, x), luv)
print('\a')
print("hed features")
hed_features = util.loading_map(lambda x: extraction.split_image_features(
    lambda y : extraction.color_features(y, mean = True, std = True), 8, x), hed)

print("hog features")
hog = util.loading_map(lambda x: feature_extraction.calcHOGWrapper(x), grayscaled)



hybrid_bright_hed_hog_luv = numpy.concatenate((brightness, hed_features, hog, luv_features), 1)


n_folds=5
print("Build random forest")
nr_of_trees=500
model = Pipeline([
    ("standard scaler", StandardScaler()),   
     ("Random forest classifier", RandomForestClassifier(n_estimators=nr_of_trees,class_weight='balanced',n_jobs=4))
    ])

print("Evaluating brightness+hed+hog+luv")
validation.validate_feature(hybrid_bright_hed_hog_luv, labels, classes, model, n_folds, False, False, True, True)

print('\a')
n_folds=5
print("Build extreme forest")
nr_of_trees=500
model = Pipeline([
    ("standard scaler", StandardScaler()),   
     ("Random forest classifier", ExtraTreesClassifier(n_estimators=nr_of_trees,class_weight='balanced',n_jobs=4))
    ])

print("Evaluating brightness+hed+hog+luv")
validation.validate_feature(hybrid_bright_hed_hog_luv, labels, classes, model, n_folds, False, False, True, True)

print('\a')


