# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 21:22:52 2020

@author: Mattia
"""
from scipy.stats import pearsonr
from numpy import diff
from scipy.signal import correlate

def advcorr(X,Y):
    """
        Gradient based correlation function  
            (aka a delicately crafted correlation algorithm through dumb parametric studies.)
    """

    # Calculate gradients
    dX = diff(X)
    dY = diff(Y)

    # See the correlation
    o1 = correlate(dX, dX)
    o2 = correlate(dX, dY)
    o = pearsonr(o1, o2)

    return o[0]