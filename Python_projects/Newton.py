from numpy import array, zeros, linspace, exp
import matplotlib.pyplot as plt
from ODES.Temporal_schemes import Euler, RK2, RK4, AB2, Inv_Euler, CN
from Cauchy_Error.Richardson import Cauchy_Error
from Physics.Kepler import Kepler
from Physics.Linear_Oscilator import Lin_Osc

def Newton(F,x0,dF_dx = None, tolerance = 1E-8, maxiter=50): # When an equality is given in an argument, means that if none value is given, it takes the value of the equality
    def dFx(x): 
        if dF_dx == None: 
            delta = 1E-4
            return ( F(x+delta) - F(x-delta) ) / (2*delta)
        else: 
            return dF_dx(x)
        
    
    xn  = x0 
    error = 1.
    iter = 0
    while error > tolerance and iter<maxiter:
        xn1 = xn - F(xn)/dFx(xn)
        error = abs(xn1-xn)
        # print("xn = ", xn, "xn+1-xn = ", xn1-xn)
        xn  = xn1
        iter += 1

    print("iteraciones = ", iter)        
    return xn1

# Function which Newton will found zeros
def function(x):
    
    return exp(x) - 2*x - 2

def derivative(x):
    
    return exp(x) - 2


cero1 = Newton(function, -2, derivative)
print("cero1 = ", cero1)
print("residuo1 = ", function(cero1))

cero2 = Newton(function, -2)
print("cero2 = ", cero2)
print("residuo2 = ", function(cero2))

x = linspace(-2,2,100)
grafica = function(x)

plt.figure(figsize=(13, 7))
plt.axis("equal")

plt.plot( x[:], grafica[:] , '-b' , lw = 1.0)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title("Grafica de f(x)")
plt.grid()
plt.show()

