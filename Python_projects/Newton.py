from numpy import array, zeros, linspace, exp
import matplotlib.pyplot as plt
from ODES.Temporal_schemes import Euler, RK2, RK4, AB2, Inv_Euler, CN
from Cauchy_Error.Richardson import Cauchy_Error
from Physics.Kepler import Kepler
from Physics.Linear_Oscilator import Lin_Osc

def Newton(F,x0,dF_dx, tolerance = 1E-8): # When an equality is given in an argument, means that if none value is given, it takes the value of the equality
        
    xn  = x0 
    error = 1.
    while error > tolerance:
            xn1 = xn - F(xn)/dF_dx(xn)
            error = abs(xn1-xn)
            print("xn = ", xn, "xn+1-xn = ", xn1-xn)
            xn  = xn1
            
    return

# Function which Newton will found zeros
def function(x):
    
    return exp(x) - 2*x - 2

def derivative(x):
    
    return exp(x) - 2


cero1 = Newton(function, -1, derivative)
cero2 = Newton(function, 2, derivative)
print("cero1 = ", cero1)
print("cero2 = ", cero2)

x = linspace(0,1,100)
grafica = function(x)

plt.figure(figsize=(13, 7))
plt.axis("equal")

plt.plot( x[:], grafica[:] , '-b' , lw = 1.0)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title("Grafica de f(x)")
plt.grid()
plt.show()

