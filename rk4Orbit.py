# -*- coding: utf-8 -*-
"""
Created on Tue Aug 05 12:28:06 2014

@author: hmendoza
"""
from pylab import *

def rkOrbit(x0, u, T, N):
    Nx = float(N)
    step = T / Nx
    x = x0
    sim = zeros((4,N))
    
    for i in range(N):
        k1 = odeOrbitTransfer(x, u)
        k2 = odeOrbitTransfer(x + 0.5*step*k1, u)
        k3 = odeOrbitTransfer(x + 0.5*step*k2, u)
        k4 = odeOrbitTransfer(x + step*k3, u)
        sim[:,i] = x + step*(1./6.)*(k1 + 2*k2 + 2*k3 + k4)
        x = sim[:,i]
    return sim