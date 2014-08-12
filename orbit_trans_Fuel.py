# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 13:04:55 2014

@author: hmendoza, jl
"""

from casadi import *
from casadi.tools import *
from pylab import *
import matplotlib.pyplot as plt


N = 60
nx = 5 # State size - Added mass
nu = 2 # Control Size
tguess = 3000.0

u = SX.sym("u",nu) # Thrust Controls
x = SX.sym("x",nx) # States [x,theta, x_dot, theta_dot, mass]

# Create Initial Point Value as equal bounds
fuel0 = 40
x0 = array([7000, 0, 0 ,1.0781e-3, fuel0]) # Initial State
u0 = array([0.0,0.006])
xN = array([8000, pi, 0, 8.82337e-4,10000]) # Final State

#System dynamics
G = 6.67384e-11
mu =  3.986012e5
#m1 = 0 # Planet's mass
#m2 = 0 # Satellite mass
r = x[0]
theta = x[1]
rdot = x[2]
thetadot = x[3]
m = x[4]

rdotdot = -mu/(r**2) + u[0]/m + r*(thetadot**2)

thetadotdot =  u[1]/(r*m) - (2*thetadot*rdot)/r

us = 1e0
mdot = -1*(sqrt(1 + (u[0]**2 + u[1]**2)*us**2) - 1)/us;
#mdot = -1/(9.81*450) * sqrt (u[0]**2 + u[1]**2 + 1e-10)

xdot = vertcat([rdot, thetadot, rdotdot, thetadotdot, mdot])
qdot = u[0]**2 + u[1]**2

orbODE = SXFunction([x,u],[xdot,qdot])
orbODE.setOption("name","orbODE")
orbODE.init()

# Define NLP variables
W = struct_symMX([
      (
        entry("X",shape=(nx,1),repeat=N+1),
        entry("U",shape=(nu,1),repeat=N)
      ),
      entry("T")
])

#Simulation of Orbit with r = R1
U = MX.sym("U",nu)
X0 = MX.sym("X",nx)
T0 = MX.sym("T")
# ????? Because these are Steps
# We believe the Integrator is sensitive to
# these values
#RK4 with M Steps
XF = X0
M = 1;# DT = float(Tf)/float((N*M))
DT = T0/M
QF = 0
R_terms = []
for j in range(M):
    [k1, k1q] = orbODE([XF,             U])
    [k2, k2q] = orbODE([XF + DT/2 * k1, U])
    [k3, k3q] = orbODE([XF + DT/2 * k2, U])
    [k4, k4q] = orbODE([XF + DT   * k3, U])
    XF += DT/6*(k1   + 2*k2   + 2*k3   + k4)
    QF += DT/6*(k1q  + 2*k2q  + 2*k3q  + k4q)
    #R_terms.append(XF)
    #R_terms.append(U)
    
#R_terms = vertcat(R_terms) # Concatenate terms
#orbSim = MXFunction([X0,U],[XF,QF])
orbSim = MXFunction([X0,U,T0],[XF,QF])
orbSim.setOption("name","orbSim")
orbSim.init()


# NLP constraints
g = []

J = 0

# Terms in the Gauss-Newton objective
#R = []

# Build up a graph of integrator calls
for k in range(N):
    # Call the integrator
    [x_next_k, qF] = orbSim([W["X",k], W["U",k], W["T"]/float(N)])

    # Append continuity constraints
    g.append(x_next_k - W["X",k+1])

    # Append Terminal Cost
    #J += qF
    # Append Gauss-Newton objective terms
    #R.append(R_terms)

# Concatenate constraints
g = vertcat(g)

# Create "Upper and Lower" Boundaries for Constraints
gmin = zeros(g.shape)
gmax = zeros(g.shape)

# Concatenate terms in Gauss-Newton objective
#R = vertcat(R)

# Objective function
#obj = mul(R.T,R)/2

# Initial Guess - Simulation
w0 = W(0); w0["X",0] = x0; w0["U",0] = u0
w0["T"] = tguess
orbSim.setInput(w0["U",0],1)
orbSim.setInput(w0["T"]/float(N),2)
for l in range(N):
    orbSim.setInput(w0["X",l],0)
    orbSim.evaluate()
    w0["X",l+1] = orbSim.getOutput(0)
    #print w0["X",l+1]

r_0 = w0["X",:,0]
theta_0 = w0["X",:,1]
px_0 = r_0 * cos(theta_0)
py_0 = r_0 * sin(theta_0)
figure(6)
plot(px_0,py_0)
figure(7)
plot(linspace(0,tguess,N+1),w0["X",:,4])
#Cost Optimality
f = W["X",-1,4]

# Construct and populate the vectors with
# upper and lower simple bounds
w_min = W(-inf)
w_max = W(inf)

# Control bounds
#w_min["U",:] = array([-1800,-2)
#w_max["U",:] = array([2, 2])

w_min["X",:,0] = x0[0] - 100
w_max["X",:,0] = xN[0] + 100
w_min["X",:,2] = -0.001
w_min["X",:,4] = 1
w_max["X",:,4] = fuel0 + 1
w_min["T"] = 30.0
w_max["T"] = 10000.0

# Initial Conditions
w_min["X",0] = w_max["X",0] = x0

# Terminal Conditions
w_min["X",-1] = w_max["X",-1] = xN
w_min["X",-1,1] = 0
w_max["X",-1,1] = 1.5*pi
w_min["X",-1,4] = 10
w_max["X",-1,4] = fuel0 + 1
#w_max["X",-1,4] = 4000

# Create an NLP solver object
nlp = MXFunction(nlpIn(x=W),nlpOut(f=-f,g=g))
nlp_solver = NlpSolver("ipopt", nlp)
#nlp_solver.setOption("linear_solver", "mumps")
nlp_solver.setOption("max_iter",500)
nlp_solver.init()
nlp_solver.setInput(w0,"x0")
#nlp_solver.setInput(woo,"x0")
nlp_solver.setInput(w_min,"lbx")
nlp_solver.setInput(w_max,"ubx")
nlp_solver.setInput(gmin,"lbg")
nlp_solver.setInput(gmax,"ubg")

# Solve the Problem
nlp_solver.evaluate()

# Retrieve to plot
sol_W = W(nlp_solver.getOutput("x"))
Tf = sol_W["T"]
u_opt_r = sol_W["U",:,0]
u_opt_t = sol_W["U",:,1]
r_opt = sol_W["X",:,0]
rdot_opt = sol_W["X",:,2]
theta_opt = sol_W["X",:,1]
m_opt = sol_W["X",:,4]
px_opt = r_opt * cos(theta_opt)
py_opt = r_opt * sin(theta_opt)
print max(u_opt_r), max(u_opt_t)
fig1 = plt.figure(1)
plt.clf()
plt.step(linspace(0,Tf,N),u_opt_r,'b-.',linspace(0,Tf,N),u_opt_t,'r-.')
plt.title("Orbit Transfer Control - multiple shooting")
plt.xlabel('time')
plt.legend(['u_r','u_t'])
plt.grid()
plt.show()
fig1.savefig('Orbit_Control_F.jpg')
fig2 = plt.figure(2)
plt.plot(linspace(0,Tf,N+1),r_opt,'r.-')
plt.title("Orbit - Trajectory")
plt.xlabel('time')
plt.ylabel('r [m]')
plt.grid()
plt.show()
fig2.savefig('Orbit_Radius_F.jpg')
fig5 = plt.figure(5)
plt.plot(linspace(0,Tf,N+1),rdot_opt,'r.-')
plt.title("Orbit - Trajectory")
plt.xlabel('time')
plt.ylabel('rdot [m]')
plt.grid()
plt.show()
fig5.savefig('Orbit_Radiusdot_F.jpg')
fig3 = plt.figure(3)
plt.plot(px_opt, py_opt,'g.-')
plt.title("Orbit - Trajectory in x vs y")
plt.xlabel('Position in x [m]')
plt.ylabel('Position in y [m]')
plt.grid()
plt.show()
fig3.savefig('Orbit_XY_F.jpg')
fig4 = plt.figure(4)
plt.plot(linspace(0,Tf,N+1),m_opt,'y.-')
plt.title("Orbit - Mass Consumption")
plt.xlabel('time')
plt.ylabel('m [kg]')
plt.grid()
plt.show()
fig4.savefig('Orbit_Mass_F.jpg')