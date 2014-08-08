# -*- coding: utf-8 -*-
"""
Created on Wed Aug 06 17:55:22 2014

@author: hmendoza

odeOrbitTransfer
"""

from pylab import *

def odeOrbitTransfer(x, u):
    A = zeros((4,4))
    B = zeros((4,2))
    mu = 20.
    A[0,2] = 1; A[1,3] = 1; A[2,0] = mu/(x[3]**3); A[2,3] = x[1]*x[3];
    A[3,2] = -(2.*x[3])/x[1];
    B[2,0] = 1; B[3,0] = 1./x[1]

    result = mul(A*x) + mul(B*u)    
    
    return result