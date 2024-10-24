# The idea is to obtain the error of our numeric solution at any time
# using the Richardson's extrapolation: 
#
#           E(t_n) = ( U1_n - U2_n ) / ( 1 - 2^(-q) ),              [1]
#
# where U1 is the solution for dt (coarse mesh) and U2 is the one 
# obtained for dt/2 (fine mesh). 
# 
# Every temporal scheme has an order "q" (DATA). 
# The convergence rate, AKA how great a temporal scheme is, is
# determined by:  
#
#               log( |E(_tn)| ) = C - q·log(N),                     [2]
#
# Notice that, as we can see from [1], the position occupied by the dt/2 
# grid is twice the positivon of the dt grid. 
# 
# Example: ---------------------------------------------------------------
# 
#  t1 = [0    1    2    3    4 ]
#  t2 = [0   1/2   1   3/2   2   5/2   4   7/2   8] 
#
#   -> t = 1s: t1[1] or t2[2]
#   -> t = 2s: t1[2] or t2[4]
#   -> t = 3s: t1[3] or t2[6]


###########################################################
#                       LIBRARIES                         #
########################################################### 
from numpy import array, zeros, linspace
import matplotlib.pyplot as plt
from ODES.Temporal_schemes import Euler, RK2, RK4, AB2, Inv_Euler, CN
from Cauchy_Error.Richardson import Cauchy_Error
from Physics.Kepler import Kepler
from Physics.Linear_Oscilator import Lin_Osc


###########################################################
#                         DATA                            #
###########################################################
# Select the problem you want to solve
problema = Kepler

# Select the temporal scheme with wHich you want to calculate
# the numerical solution
esquema1 = Euler
esquema2 = RK4
esquema3 = Inv_Euler
esquema4 = CN

# Initial conditions
x0  = 1
y0  = 0
vx0 = 0
vy0 = 1

# Create the initial conditions vector accroding to the 
# dimension of the solution vector of your problem
U0 = array( [x0,y0,vx0,vy0] )

# Initial and final instants
t0 = 0
tf = 20

# Number of intervals for the coarse  mesh
N = 2000 # give a larger value for Inverse Euler scheme ( dt < 0.033 )

###########################################################
#                         CODE                            #
###########################################################

# Initialize the temporal instants vectors
t  = zeros(N+1)
t  = linspace(t0,tf,N+1)
dt = (t[-1]-t[0])/N

# Compute the error with Richardson's extrapolation
U1, error1 = Cauchy_Error(problema, esquema1, U0, t)
U2, error2 = Cauchy_Error(problema, esquema2, U0, t)
U3, error3 = Cauchy_Error(problema, esquema3, U0, t)
U4, error4 = Cauchy_Error(problema, esquema4, U0, t)

###########################################################
#                         GRÁFICAS                        #
###########################################################
# Plots the solution with "esquema1" temporal scheme
plt.figure(figsize=(13, 7))
plt.axis("equal")

plt.plot( U1[:,0], U1[:,1] , '-b' , lw = 1.0)
plt.xlabel('x')
plt.ylabel('y')
plt.title( r'{} problem solved with {} scheme (coarse mesh with $\Delta$t={})'.format( problema.__name__, esquema1.__name__, round(dt, 2) ) )
plt.grid()
plt.show()

# Plots the error for componets 0 and 1 of the of the solution vector for 
# the "esquema1" temporal scheme.
plt.figure(figsize=(13, 7))

plt.plot( t[:], error1[:,0] , '-b' , lw = 1.0, label ="Error en x" )
plt.plot( t[:], error1[:,1] , '-r' , lw = 1.0, label ="Error en y" )
plt.xlabel('t')
plt.ylabel('Error')
plt.legend()
plt.title( r'Error of {} problem solved with {} scheme (coarse mesh with $\Delta$t={})'.format( problema.__name__, esquema1.__name__, round(dt, 5) ) )
plt.grid()
plt.show()

# Compares the component 0 of the solution vector given by different temproal schemes
plt.figure(figsize=(13, 7))

plt.plot( t[:], error1[:,0] , '-b' , lw = 1.0, label =r"{}".format(esquema1.__name__) )
plt.plot( t[:], error2[:,0] , '-r' , lw = 1.0, label =r"{}".format(esquema2.__name__) )
plt.plot( t[:], error3[:,0] , '-g' , lw = 1.0, label =r"{}".format(esquema3.__name__) )
plt.plot( t[:], error4[:,0] , '-m' , lw = 1.0, label =r"{}".format(esquema4.__name__) )
plt.xlabel('t')
plt.ylabel('Error')
plt.legend()
plt.title( r'Error in x component of {} problem (coarse mesh with $\Delta$t={})'.format( problema.__name__, round(dt, 2) ) )
plt.grid()
plt.show()
