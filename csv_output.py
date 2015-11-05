# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 15:23:18 2015

@author: Rian
"""
import numpy
import csv

def generate(x_train, y_train, x_test, model, csv_file_name):
    with open(csv_file_name, 'w', newline = '') as csvfile:
        classes = numpy.unique(y_train)
        fieldnames = numpy.insert(classes, 0, 'Id')
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)
        for i in range(len(x_test)):
            labels = classes
            labels = numpy.insert(labels, 0, 'Id')
            values = numpy.insert(probabilities[i],0, int(i + 1))
            dictionary = dict(zip(labels, values))
            dictionary['Id'] = int(dictionary['Id'])
            writer.writerow(dictionary)