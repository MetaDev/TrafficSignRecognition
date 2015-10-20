# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 20:18:33 2015

@author: Rian
"""
import numpy

def score(classes, percent):
    wrongPredictions = numpy.log10((1-percent) / (classes - 1))
    rightPredictions = numpy.log10(percent)
    return - (percent * rightPredictions + (1-percent) * wrongPredictions)