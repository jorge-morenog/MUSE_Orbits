from numpy import array, zeros, linspace
import matplotlib.pyplot as plt
from ODES.Temporal_schemes import Euler, RK2, RK4, AB2, Inv_Euler, CN, LF, Embedded_RK, GBS_NL_fixed, GBS_variable_NL
from Physics.Kepler import Kepler
from Physics.Linear_Oscilator import Lin_Osc
from ODES.Cauchy_problem import Cauchy


###########################################################
#                          DATA                           #
###########################################################
# Select the physical problem you want to solve (from the ones implemented in the "Physics" library)
problem = Kepler

# Selected the temporal schemes to use (from the ones implemented in the "ODES.Temporal_schemes" library)
scheme_1 = Euler
scheme_2 = Inv_Euler
scheme_3 = CN
scheme_4 = RK4

# Select the initial conditions
if problem == Lin_Osc: 
    # Linear Oscillator
    x0  = 1
    vx0 = 0
    U0 = array( [x0,vx0] )
elif problem == Kepler: 
    # Kepler's orbit
    x0  = 1
    y0  = 0
    vx0 = 0
    vy0 = 1
    U0 = array( [x0,y0,vx0,vy0] )

# Initial and final instants
t0 = 0
tf = 20

# Number of intervals for numerical integration
N = 3000

###########################################################
#                          CODE                           #
###########################################################

# Initialize time vector
t  = zeros(N+1)
t  = linspace(t0,tf,N+1)
dt = (tf-t0)/N

# Calculate solution for defined temporal schemes
print("\n-- IMPORTANT: -------------------------------------------------------")
print("If trying to solve with an implicit scheme and solver exceeds the\nmaximum number of iterations, try giving a larger value for N.")
print("---------------------------------------------------------------------\n")

U_1 = Cauchy(scheme_1, problem, U0, t,)
U_2 = Cauchy(scheme_2, problem, U0, t)
U_3 = Cauchy(scheme_3, problem, U0, t)
U_4 = Cauchy(scheme_4, problem, U0, t)


###########################################################
#                         PLOTS                           #
###########################################################

plt.figure(figsize=(13, 7))
plt.axis("equal")


if problem == Lin_Osc: 
    # Plot for Linear Oscillator
    plt.plot( t[:], U_1[:,0] , '-b'  , lw = 1.0, label =r"{}".format(scheme_1.__name__) )
    plt.plot( t[:], U_2[:,0] , '-r'  , lw = 1.0, label =r"{}".format(scheme_2.__name__) )
    plt.plot( t[:], U_3[:,0] , '--y' , lw = 2.0, label =r"{}".format(scheme_3.__name__) )
    plt.plot( t[:], U_4[:,0] , '-c' , lw = 1.0, label =r"{}".format(scheme_4.__name__) )
    plt.xlabel('t')
    plt.ylabel('x')
elif problem == Kepler:
    # Plot for Kepler's orbit
    plt.plot( U_1[:, 0], U_1[:,1] , '-b'  , lw = 1.0, label =r"{}".format(scheme_1.__name__) )
    plt.plot( U_2[:, 0], U_2[:,1] , '-r'  , lw = 1.0, label =r"{}".format(scheme_2.__name__) )
    plt.plot( U_3[:, 0], U_3[:,1] , '--y' , lw = 2.0, label =r"{}".format(scheme_3.__name__) )
    plt.plot( U_4[:, 0], U_4[:,1] , '-c' , lw = 1.0, label =r"{}".format(scheme_4.__name__) )
    plt.xlabel('x')
    plt.ylabel('y')

plt.legend()
plt.title( r'{} solved with $\Delta$t={}'.format( problem.__name__, round(dt, 5) ) )
plt.grid()
plt.show()


