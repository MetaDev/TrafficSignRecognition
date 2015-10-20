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
    problemClasses = ['A14', 'A1AB', 'A1CD', 'A23', 'A31', 'A51', 'A7A', 'A7B', 'B15A',
       'B17', 'C21', 'C3', 'C43', 'D7', 'D9', 'F1', 'F4b', 'begin', 'end',
       'lang', 'm']
    images, classes = loadTrainingAndClasses()
    filteredImageClasses = numpy.array(list(filter(lambda x: x[1] in problemClasses, zip(images,classes))))
    return filteredImageClasses[:,0], filteredImageClasses[:, 1]
    
def loadTest():
    imagePaths = glob("test/*.png")
    return [ndimage.imread(x) for x in imagePaths]