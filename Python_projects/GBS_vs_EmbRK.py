import time
from numpy import array, zeros, linspace, sqrt
import matplotlib.pyplot as plt
from ODES.Temporal_schemes import Embedded_RK, GBS_variable_NL, GBS_NL_fixed
from ODES.Cauchy_problem import Cauchy
from numpy.linalg import norm
from Physics.Linear_Oscilator import Lin_Osc
from Physics.Kepler import Kepler
from Physics.Arenstorf import Arenstorf

########################################
def F_with_counter(U, t, *args):
    global call_count, problem
    call_count += 1
    return problem(U, t)  # Changes depending on the select physics problem

###########################################################
#                          DATA                           #
###########################################################
# Select the physics problem
problem = Arenstorf

# Select the temporal scheemes with which you want to calulate the solution
scheme_1 = GBS_variable_NL
scheme_2 = Embedded_RK

# Initial conditions (vary depending on the physics problem)
x0  = 0.994
y0  = 0
vx0 = 0
vy0 = -2.001585106
U0  = array( [x0,y0,vx0,vy0] )

# x0  = 1
# y0  = 0
# vx0 = 0
# vy0 = 1
# U0  = array( [x0,y0,vx0,vy0] )

# x0  = 1
# vx0 = 0
# U0  = array( [x0,vx0] )

# Initial and final instants
t0 = 0
tf = 17.5

# Numbers of intervals for numerical integration
N = 1750

###########################################################
#                          CODE                           #
###########################################################

# Initialize time vector
t  = zeros(N+1)
t  = linspace(t0,tf,N+1)
dt = (tf-t0)/N

# Calculate the solution in every time instant
U_1 = Cauchy(scheme_1, problem, U0, t, 13, 1.E-4)
U_2 = Cauchy(scheme_2, problem, U0, t, 2, 3E-6)

print("\n---------------------- COMPUTATIONAL EFFORTS -----------------------------")
print("\nNOTE: Computational effort estimation only counts F evaluations, not total operations.")
# Estimated computational effot of scheme scheme_1
call_count = 0  
start_time1 = time.time()
U_1_count = Cauchy(scheme_1, F_with_counter, U0, t, 13, 1.E-4)
execution_time1 = time.time() - start_time1
cost_1 = call_count / execution_time1 *10**(-6)
print(f"\nTotal number of evalutations of F function: {call_count}")
print(f"Total running time of {scheme_1.__name__} scheme: {execution_time1:.4f} seconds")
print(f"Estimated computational effort of {scheme_1.__name__} scheme: {cost_1:.4f} MFLOPS\n")

# Estimated computational effot of scheme scheme_2
call_count = 0 
start_time2 = time.time()
U_2_count = Cauchy(scheme_2, F_with_counter, U0, t, 2, 3E-6)
execution_time2 = time.time() - start_time2
cost_2 = call_count / execution_time2 *10**(-6)
print(rf"Total number of evalutations of F function: {call_count}")
print(f"Total running time of {scheme_2.__name__} scheme: {execution_time2:.4f} seconds")
print(f"Estimated computational effort of {scheme_2.__name__} scheme: {cost_2:.4f} MFLOPS")
print("--------------------------------------------------------------------------\n")

###########################################################
#                           PLOTS                         #
###########################################################
plt.figure(figsize=(13, 7))
plt.axis("equal")

# plt.plot( t, U_1[:,0] , '--r' , lw = 2.0, label =r"{}".format(scheme_1.__name__) )
# plt.plot( t, U_2[:,0] , '-b' , lw = 1.0, label =r"{}".format(scheme_2.__name__) )
# plt.xlabel('t')
# plt.ylabel('x')

plt.plot( U_1[:,0], U_1[:,1] , '--r' , lw = 2.0, label =r"{}".format(scheme_1.__name__) )
plt.plot( U_2[:,0], U_2[:,1] , '-b' , lw = 1.0, label =r"{}".format(scheme_2.__name__) )
plt.xlabel('x')
plt.ylabel('y')

plt.legend()
plt.title( r'{} solved with $\Delta$t={}'.format( problem.__name__, round(dt, 5) ) )
plt.grid()
plt.show()


