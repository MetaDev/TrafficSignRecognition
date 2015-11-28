# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 13:45:31 2015

@author: Pieter
"""
import data_loading as loader

import numpy
from skimage import color
from skimage import exposure
from scipy import misc

import matplotlib.pyplot as plt
import numpy as np

from lasagne.layers import InputLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.layers import DenseLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax

from nolearn.lasagne import objective
from lasagne.layers import get_all_params

from nolearn.lasagne import NeuralNet
from nolearn.lasagne import TrainSplit
from lasagne.updates import adam


# FUNCTIONS

# Loading and processing data
def load_data():

    #save classes as ints 
    #create map from class->index 
    print("Loading images and classes")
    images, classes = loader.loadTrainingAndClasses()
    
    unique_classes = list(set(classes))
    int_unique_classes={unique_classes[i]:i for i in range(len(unique_classes))}
    
    #convert classes with map
    int_classes=[int_unique_classes[cl] for cl in classes]
    int_classes=numpy.array(int_classes)
    
    #preprocess images
    print("Pre-processing images")
    thumbsize = 28;
    thumbs = [misc.imresize(x,(thumbsize,thumbsize)) for x in images] 
    grays = [color.rgb2gray(x) for x in thumbs] 
    normies = [exposure.equalize_hist(x) for x in grays] 
    normies = np.array(normies)
    
    return normies, int_classes
    
X, y =load_data();


# L2 regularization
def regularization_objective(layers, lambda1=0., lambda2=0., *args, **kwargs):
    # default loss
    losses = objective(layers, *args, **kwargs)
    # get the layers' weights, but only those that should be regularized
    # (i.e. not the biases)
    weights = get_all_params(layers[-1], regularizable=True)
    # sum of absolute weights for L1
    sum_abs_weights = sum([abs(w).sum() for w in weights])
    # sum of squared weights for L2
    sum_squared_weights = sum([(w ** 2).sum() for w in weights])
    # add weights to regular loss
    losses += lambda1 * sum_abs_weights + lambda2 * sum_squared_weights
    return losses


if(1):
    print("Plotting some processed images")
    figs, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i in range(4):
        for j in range(4):
            axes[i, j].imshow(-X[i + 4 * j].reshape(28, 28), cmap='gray', interpolation='none')
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            axes[i, j].set_title("Label: {}".format(y[i + 4 * j]))
            axes[i, j].axis('off')


print("Defining layers")
num_classes = len(set(y))
layers0 = [
    # layer dealing with the input data
    (InputLayer, {'shape': (None, X.shape[1], X.shape[2], 1)}),

    # first stage of our convolutional layers
    (Conv2DLayer, {'num_filters': 96, 'filter_size': 5}),
    (Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),
    (Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),
    (Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),
    (Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),
    (MaxPool2DLayer, {'pool_size': 2}),

    # second stage of our convolutional layers
    (Conv2DLayer, {'num_filters': 128, 'filter_size': 3}),
    (Conv2DLayer, {'num_filters': 128, 'filter_size': 3}),
    (Conv2DLayer, {'num_filters': 128, 'filter_size': 3}),
    (MaxPool2DLayer, {'pool_size': 2}),

    # two dense layers with dropout
    (DenseLayer, {'num_units': 64}),
    (DropoutLayer, {}), #used to regularize neural networks
    (DenseLayer, {'num_units': 64}),

    # the output layer
    (DenseLayer, {'num_units': num_classes, 'nonlinearity': softmax}),
]

print("Defining Neural Network")
net0 = NeuralNet(
    layers=layers0,
    max_epochs=10,

    update=adam,
    update_learning_rate=0.0002,

    objective=regularization_objective,
    objective_lambda2=0.0025,

    train_split=TrainSplit(eval_size=0.25),
    verbose=1,
)
print("Training Neural Network")
net0.fit(X, y)