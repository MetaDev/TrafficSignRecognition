# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 23:09:33 2015

@author: Rian
"""

from scipy import ndimage
from glob import glob
from pathlib import Path
import numpy

def extract(filename):
    image = ndimage.imread(filename)
    category = Path(filename).parent.name
    superCategory = Path(filename).parent.parent.name
    return (image, superCategory, category)
    


def loadTrainingImages():
    imagePaths = glob("train/*/*/*.png")
    return list(map(extract, imagePaths))
    
def loadTrainingAndClasses():
    imagePaths = glob("train/*/*/*.png")
    return [ndimage.imread(x) for x in imagePaths], [Path(x).parent.name for x in imagePaths]
    
def loadProblematicImagesAndClasses():
    problemClasses = ['A51', 'B17', 'begin', 'end', 'lang']
    images, classes = loadTrainingAndClasses()
    filteredImageClasses = numpy.array(list(filter(lambda x: x[1] in problemClasses, zip(images,classes))))
    return filteredImageClasses[:,0], filteredImageClasses[:, 1]
    
def loadTest():
    imagePaths = glob("test/*.png")
    return [ndimage.imread(x) for x in imagePaths]