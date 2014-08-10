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
    G = 6.67384e-11
    m1 = 0 # Planet's mass
    m2 = 0 # Satellite mass
    #G * (m1 + m2) # Gravitational Constant - 6.67384e-11 m^3 kg^-1 s^-2
    mu =  3.986012e5 
    A[0,2] = 1; A[1,3] = 1; A[2,0] = mu/(x[0]**3); A[2,3] = x[0]*x[3];
    A[3,2] = -(2.*x[3])/x[0];
    B[2,0] = 1; B[3,0] = 1./x[0]

    result = dot(A,x) + dot(B,u)    
    
    return result