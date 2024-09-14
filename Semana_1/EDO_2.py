# y1' =  y1 + y2^2 + 3x
# y2' = 3y1 + y2^3 - cos(x)
#   y1(0) = 0
#   y2(0) = 0

from numpy import cos, linspace
from scipy.integrate import odeint

###########################################################
#                         DATOS                           #
###########################################################

# Intervalos de cálculo de la solución
x0 = 0
xf = 1
N  = 100

# Condiciones iniciales
y10 = 0
y20 = 0

###########################################################
#                         CÓDIGO                          #
###########################################################

# # Incremento espacial
# dx = (xf-x0)/N

# Vector solucion en la condicion inicial
S0 = (y10, y20)

# Funcion de la derecha de la igualdad de las EDOs
def dSdx(S,x): 
    y1, y2 = S
    S1 =   y1 + y2**2 + 3*x
    S2 = 3*y1 + y2**3 - cos(x)
    
    F = [S1, S2]
    return F

# Malla espacial equiespaciada
x = linspace(x0, xf, N)

# Calculo de la solucion con funcion de sicpy
sol = odeint(dSdx, S0, x)

# Establecemos la solucion de cada variable
y1 = sol.T[0]
y2 = sol.T[1]

# Graficamos
import matplotlib.pyplot as plt
plt.plot(x, y1, '-b' , lw=1.0 , label='y1')
plt.plot(x, y2, '--r', lw=1.0 , label='y2')
plt.legend(loc='best')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solución EDO')
plt.grid()
plt.show()