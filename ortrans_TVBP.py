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

N = 100
nx = 4 # State size
T = 25000

x0 = array([7000,0,1,1.0781e-3]) # Initial State
u0 = array([0,0])

Ss = T/float(N) #Size Step
nx = alen(x0)
orbitSim = zeros((nx,N))
orbitSim = rkOrbit(x0, u0, T, N)

#----- Plot of Simulation ---------#
r = orbitSim[0,:]
theta = orbitSim[1,:]
px = r*cos(theta)
py = r*sin(theta)

t = arange(0,T,Ss)
fig1 = plt.figure(1)
plot(px, py ,'ro-')
#plt.plot(t,p_rk4,'bo-')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Orbit Simulation')
plt.grid(True)

# ---- NPL Formulation ---- #