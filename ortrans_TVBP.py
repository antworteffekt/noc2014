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

N = 200
nx = 4 # State size
T = 2.5

x0 = array([3500,0,0,1]) # Initial State
u0 = array([0,0])

Ss = T/float(N) #Size Step
nx = alen(x0)
orbitSim = zeros((nx,N))
orbitSim = rkOrbit(x0, u0, T, N)

#----- Plot ---------#
px_plot = orbitSim[0,:]
py_plot = orbitSim[1,:]
t = arange(0,T,Ss)
fig1 = plt.figure(1)
plot(t,px_plot,'ro-')
#plt.plot(t,p_rk4,'bo-')
plt.xlabel('t [s]')
plt.ylabel('p [m]')
plt.title('Orbit Simulation')
plt.grid(True)