from numpy import linspace
from scipy.integrate import odeint
###############################################################
#                           DATOS                             #
###############################################################

# Instante inicial
t0=0
# Instante final
tf=10
# Numero de intervalos
N=10
# Condicion inicial
v0 = 0

###############################################################
#                          CÓDIGO                             #
###############################################################

# Incremento temporal
dt=(tf-t0)/N

# Funcion F(U;t)
def dvdt(t,v): 
    return 2*v**2 - 5

# Tiempos en los que se calcula la solución discreta
t=linspace(t0,tf,N+1)

# Solucion con solver en FORTRAN de la libreria NUMPY
sol_m1 = odeint(dvdt,y0=v0, t=t, tfirst=True)

print("tiempo =", t)