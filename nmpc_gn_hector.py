from casadi import *
from casadi.tools import *
from pylab import *

from plotter import *

N = 20      # Control discretization
T = 10.0    # End time

# Which QP_solver to use
qp_solver_class = "ipopt"
#qp_solver_class = "qpoases"

# Declare variables (use scalar graph)
u  = SX.sym("u")    # control
x  = SX.sym("x",2)  # states

# System dynamics
xdot = vertcat( [(1 - x[1]**2)*x[0] - x[1] + u, x[0]] )
f = SXFunction([x,u],[xdot])
f.setOption("name","f")
f.init()

# RK4 with M steps
# also outputs contributions to Gauss-Newton objective
U = MX.sym("U")
X0 = MX.sym("X0",2)
M = 10; DT = T/(N*M)
XF = X0
QF = 0
R_terms = [] # Terms in the Gauss-Newton objective
for j in range(M):
    [k1] = f([XF,             U])
    [k2] = f([XF + DT/2 * k1, U])
    [k3] = f([XF + DT/2 * k2, U])
    [k4] = f([XF + DT   * k3, U])
    XF += DT/6*(k1   + 2*k2   + 2*k3   + k4)
    R_terms.append(XF)
    R_terms.append(U)
    
R_terms = vertcat(R_terms) # Concatenate terms
F = MXFunction([X0,U],[XF,R_terms])
F.setOption("name","F")
F.init()

# Define NLP variables
W = struct_symMX([
      (
        entry("X",shape=(2,1),repeat=N+1),
        entry("U",shape=(1,1),repeat=N)
      )
])

# NLP constraints
g = []

# Terms in the Gauss-Newton objective
R = []

# Build up a graph of integrator calls
for k in range(N):
    # Call the integrator
    [x_next_k, R_terms] = F([ W["X",k], W["U",k] ])

    # Append continuity constraints
    g.append(x_next_k - W["X",k+1])

    # Append Gauss-Newton objective terms
    R.append(R_terms)

# Concatenate constraints
g = vertcat(g)

# Concatenate terms in Gauss-Newton objective
R = vertcat(R)

# Functions for calculating g and R
g_fcn = MXFunction([W],[g]);  g_fcn.init()
R_fcn = MXFunction([W],[R]);  R_fcn.init()

# Functions for calculating the Jacobians of g and R
jac_g_fcn = g_fcn.jacobian(); jac_g_fcn.init()
jac_R_fcn = R_fcn.jacobian(); jac_R_fcn.init()

# To allocate a QP solver in CasADi, we need to know the sparsity patterns of H and A
A_sparsity = jac_g_fcn.getOutput(0).sparsity()
jac_R_sparsity = jac_R_fcn.getOutput(0).sparsity()
H_sparsity = mul(jac_R_sparsity.T,jac_R_sparsity)  # Get sparsity pattern of multiplication
qp = qpStruct(h=H_sparsity,a=A_sparsity)

# Allocate a QP solver
if qp_solver_class!="ipopt":
    qp_solver = QpSolver(qp_solver_class,qp)
else: # Solve a QP using an NLP solver (IPOPT)
    qp_solver = QpSolver("nlp",qp) 
    qp_solver.setOption("nlp_solver","ipopt")    
    qp_solver.setOption("nlp_solver_options",{"linear_solver":"mumps"})
qp_solver.init()

# Construct and populate the vectors with
# upper and lower simple bounds
w_min = W(-inf)
w_max = W( inf)

# Control bounds
w_min["U",:] = -1
w_max["U",:] = 1

w_k = W(0)
ts = linspace(0,T,N+1)
plotter = Plotter(ts)
t = 0
x_current = array([1,0])
while True:
    w_min["X",0] = x_current
    w_max["X",0] = x_current
    
    # Evaluation of the Jacobians of g and R
    jac_g_fcn.setInput(w_k)
    jac_R_fcn.setInput(w_k)
    
    jac_R_fcn.evaluate()    
    jac_g_fcn.evaluate()
    
    jac_R = jac_R_fcn.getOutput(0)
    Rq = jac_R_fcn.getOutput(1)
    
    g_grad = jac_g_fcn.getOutput(0)
    geq = jac_g_fcn.getOutput(1)
    
    h_qp = mul(jac_R.T,jac_R); # Hessian of QP
    R_grad = mul(jac_R.T,Rq) # Gradient of the Objective
    
    dw_max = DMatrix(w_max) - DMatrix(w_k)
    dw_min = DMatrix(w_min) - DMatrix(w_k)
    
    qp_solver.setInput(h_qp, "h")
    qp_solver.setInput(R_grad,"g") # Objective part
    qp_solver.setInput(g_grad,"a") # Equalities function
    qp_solver.setInput(dw_max, "ubx")
    qp_solver.setInput(dw_min, "lbx")
    qp_solver.setInput(-geq, "uba")
    qp_solver.setInput(-geq, "lba")    
    
    qp_solver.evaluate();
    dw =  qp_solver.getOutput("x");
    
    w_new = w_k + dw # New w after the step

    # Extract from the solution the first control
    sol = W(w_new)
    u_nmpc = sol["U",0]

    # Plot the solution
    plotter.show(t,x_current,sol)
    import sys
    sys.stdout.write('Waiting for your input (<enter>, "quit|clip|clear", or numbers ):')
    wait = raw_input()
    if "quit" in wait:
        break
    if "clear" in wait:
        plotter.clear()
    if "clip" in wait:
        plotter.toggleClipping()
    try:  # Easier to Ask Forgiveness than Permission
        x_current[:] = array(map(float,wait.split(" ")))
    except:
        pass
    
    # Simulate the system with this control
    F.setInput( x_current,0)
    F.setInput( u_nmpc ,1)
    F.evaluate()
  
    # Update the current state
    x_current = F.getOutput(0)
  
    t += T/N
    # Shift the time to have a better initial guess
    # For the next time horizon
    w_k["X",:-1] = sol["X",1:]
    w_k["U",:-1] = sol["U",1:]
    w_k["X",-1] = sol["X",-1]
    w_k["U",-1] = sol["U",-1]