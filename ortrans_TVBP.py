# -*- coding: utf-8 -*-
"""
Created on Wed Aug 06 17:38:34 2014

@author: hmendoza, jli

TBVP for simple orbit transfer
"""

from casadi import *
from pylab import *

#x = SX.sym("x",4)  # x = [r, 0 ,r_dot, 0_dot]
#u = SX.sym("u",2)

N = 30
Nx = 4
T = 20

x0 = array([4,2,1,5])
u0 = array([2,2])

orbitSim = zeros((Nx,N))
orbitSim = rkOrbit(x0, u0, T, N)