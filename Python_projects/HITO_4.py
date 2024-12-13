# Integration of linear oscilator, and Regions of Absolute Stability 
from numpy import array, linspace
import matplotlib.pyplot as plt
from ODES.Temporal_schemes import Euler, RK2, RK4, AB2, Inv_Euler, CN, LF, Embedded_RK, GBS_NL_fixed, GBS_variable_NL
from ODES.Cauchy_problem import Cauchy
from Physics.Kepler import Kepler
from Physics.Linear_Oscilator import Lin_Osc
from Stability.StabilityRegions import StabilityRegion
from Stability.Jacobian import eigenvalues_jacobian

#######################################################################
#                               DATA                                  #
#######################################################################
# Select the physical problem you want to solve numerically
problem = Lin_Osc

# Select the Absolute Stability Region (REA) you want to see plotted
REA_scheme = CN

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

# Number of intervals for the time 
N   = 2000
# Limits of the interval where you want to compute the solution
t0  = 0
tf  = 20

# Select the numerical scheme with wich you want to integrate 
scheme1 = Euler
scheme2 = Inv_Euler
scheme3 = LF
scheme4 = CN
scheme5 = RK4

# Define the nummber and value of the rho iso-lines to plot in the REA
isolines = linspace(0, 1, 11)

# Dimensions of the imaginary plane where you want to plot the Absolute Stability Region
x = linspace(-5,5,101)
y = linspace(-5,5,101)

#######################################################################
#                               CODE                                  #
#######################################################################

# Create the vector of temporal instants
t = linspace(t0, tf, N+1)
dt = (tf-t0)/N


# Obtain the solution in each time step 
sol_1 = Cauchy(scheme1, problem, U0, t)
sol_2 = Cauchy(scheme2, problem, U0, t)
sol_3 = Cauchy(scheme3, problem, U0, t)
sol_4 = Cauchy(scheme4, problem, U0, t)
sol_5 = Cauchy(scheme5, problem, U0, t)

rho = StabilityRegion(x, y, REA_scheme)

# Obtain the eigenvalues at the final instant
eigenvalues = eigenvalues_jacobian(problem, sol_1[-1,:], t[-1])
# print("eigenvalues = ", eigenvalues.real)


#######################################################################
#                              PLOTS                                  #
#######################################################################
# Plot of the results for the armonic oscilator
plt.figure(figsize=(10, 6))
plt.plot(t,sol_1[:,0],'-b' ,label=r"{}".format(scheme1.__name__))
plt.plot(t,sol_2[:,0],'-r' ,label=r"{}".format(scheme2.__name__))
plt.plot(t,sol_3[:,0],'-g' ,label=r"{}".format(scheme3.__name__))
plt.plot(t,sol_4[:,0],'-c' ,label=r"{}".format(scheme4.__name__))
plt.plot(t,sol_5[:,0],'-m' ,label=r"{}".format(scheme5.__name__))

plt.xlabel("t  [non dim.]")
plt.ylabel("Position  [non dim.]")
plt.title(r"Position of the 1D armonic oscilator (with no dumping or friction), with $\Delta t ={}$".format((tf-t0)/N))
plt.grid()
plt.legend()
plt.show()

# Plot of the regions for every numerical scheme used in the script
plt.figure(figsize=(6, 5))
plt.axis("equal")

if REA_scheme == LF: 
    CS = plt.contour(x, y, rho, levels = [0.99999] )                         # Draws isolines for different rho values
else: 
    CSF = plt.contourf(x, y, rho, levels = [0, 1],  colors=['#C0C0C0'] )    # colours the inside of the stability region
    CS = plt.contour(x, y, rho, levels = isolines )                         # Draws isolines for different rho values

# Plot eigenvalue * dt
plt.plot(  eigenvalues.real*dt, eigenvalues.imag*dt, 'o', label = r"Roots of the problem ($\Delta t={}$)".format(dt)   ) 

plt.clabel(CS, inline=1, fontsize=6.5, inline_spacing=0.5)

plt.xlim([x[0], x[-1]])
plt.ylim([y[0], y[-1]])

plt.xlabel("Re(|r|)")
plt.ylabel("Im(|r|)")
plt.title(r'Absolute Stability Region of {} scheme (current $\Delta t = {}$)'.format(REA_scheme.__name__, (tf-t0)/N) )

plt.rcParams.update({'font.size': 9}) # Adjust legend fontsize
plt.legend(loc = "lower left")

plt.grid()
plt.show()
