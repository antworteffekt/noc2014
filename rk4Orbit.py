# -*- coding: utf-8 -*-
"""
Created on Tue Aug 05 12:28:06 2014

@author: hmendoza
"""
from pylab import *

def rkOrbit(x0, u, T, N):
    Nx = float(N)
    sS = T / Nx
    x = x0
    sim = zeros((4,N))
    
    for i in range(N):
        k1 = odeOrbitTransfer(x, u)
        k2 = odeOrbitTransfer(x + 0.5*sS*k1, u)
        k3 = odeOrbitTransfer(x + 0.5*sS*k2, u)
        k4 = odeOrbitTransfer(x + sS*k3, u)
        sim[:,i] = x + sS*(1./6.)*(k1 + 2*k2 + 2*k3 + k4)
        x = sim[:,i]
    return sim