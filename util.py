# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 16:51:51 2015

@author: Rian
"""
import sys

def loading_map(operation, data):
    counter = 0
    total = len(data)
    result = []
    for i in range(total):
        counter += 1
        sys.stdout.write("\r%d / %d" % (counter, total))
        sys.stdout.flush()
        result.append(operation(data[i]))
    print()
    return result