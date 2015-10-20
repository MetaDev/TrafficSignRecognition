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
    problemClasses = ['A13', 'A14', 'A15', 'A1AB', 'A1CD', 'A23', 'A23_yellow', 'A25',
       'A29', 'A31', 'A51', 'A7A', 'A7B', 'B1', 'B15A', 'B17', 'C1', 'C11',
       'C21', 'C23', 'C29', 'C3', 'C31', 'C35', 'C37', 'C43', 'D1a', 'D1e',
       'D5', 'D7', 'D9', 'E1', 'E3', 'E5', 'E7', 'E9a', 'E9a_miva', 'E9b',
       'E9cd', 'E9e', 'F1', 'F12a', 'F19', 'F1a_h', 'F23A', 'F29',
       'F33_34', 'F3a_h', 'F45', 'F47', 'F49', 'F4b', 'F50', 'F59', 'F87',
       'Handic', 'X', 'begin', 'e0c', 'end', 'lang', 'm']
    images, classes = loadTrainingAndClasses()
    filteredImageClasses = numpy.array(list(filter(lambda x: x[1] in problemClasses, zip(images,classes))))
    return filteredImageClasses[:,0], filteredImageClasses[:, 1]
    
def loadTest():
    imagePaths = glob("test/*.png")
    return [ndimage.imread(x) for x in imagePaths]