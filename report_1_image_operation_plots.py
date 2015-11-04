# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 15:03:01 2015

@author: Rian
"""

import image_transformation_testing as plotter
import image_operations as op

#regular images
plotter.plotTransformation(lambda x : x)

#resized and cropped images
plotter.plotTransformation(lambda x : op.cropAndResize(x, 0.10, 50))

#normalized images
plotter.plotTransformation(lambda x : op.normalizeImage(op.cropAndResize(x, 0.10, 50)))