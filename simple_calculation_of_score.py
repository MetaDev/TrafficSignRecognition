# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 20:18:33 2015

@author: Rian
"""
import numpy

def simpleScore(classes, percent, actual = -1):
    if actual == -1: actual = percent
    wrongPredictions = numpy.log((1-percent) / (classes - 1))
    rightPredictions = numpy.log(percent)
    return - (actual * rightPredictions + (1-actual) * wrongPredictions)
    