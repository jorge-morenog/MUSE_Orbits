from numpy import array, zeros, linspace
import matplotlib.pyplot as plt
from ODES.Temporal_schemes import Euler, RK2, RK4, AB2, Inv_Euler, CN
from ODES.Cauchy_problem import Cauchy
from Physics.Kepler import Kepler

###########################################################
#                          DATA                           #
###########################################################
# Selecciona el problema que quieres resolver (de los implementados en las funciones)
problema = Kepler

# Selecciona los esquemas numéricos con los que quieres calcular la solución
esquema_1 = Euler
esquema_2 = RK4
esquema_3 = Inv_Euler
esquema_4 = CN

# Condiciones iniciales
x0  = 1
y0  = 0
vx0 = 0
vy0 = 1

# Creamos vector de condiciones iniciales
U0 = array( [x0,y0,vx0,vy0] )

# Instantes inicial y final
t0 = 0
tf = 20

# Número de intervalos (=nº de instantes de tiempo - 1)
N = 200

###########################################################
#                         CÓDIGO                          #
###########################################################

# Inicializamos y creamos el vector de instantes de tiempo
t  = zeros(N+1)
t  = linspace(t0,tf,N+1)
dt = (tf-t0)/N

# Obtenemos la solución del problema de Cauchy que hemos especificado
print("\n-- IMPORTANT: -------------------------------------------------------")
print("If trying to solve with an implicit scheme and solver exceeds the\nmaximum number of iterations, try giving a larger value for N.")
print("---------------------------------------------------------------------\n")

# U_1 = Cauchy(esquema_1, problema, U0, t)
U_2 = Cauchy(esquema_2, problema, U0, t)
# U_3 = Cauchy(esquema_3, problema, U0, t)
# U_4 = Cauchy(esquema_4, problema, U0, t)


###########################################################
#                         GRÁFICAS                        #
###########################################################

plt.figure(figsize=(13, 7))
plt.axis("equal")

# plt.plot( U_1[:, 0], U_1[:,1] , '-b' , lw = 1.0, label =r"{}".format(esquema_1.__name__) )
plt.plot( U_2[:, 0], U_2[:,1] , '-r' , lw = 1.0, label =r"{}".format(esquema_2.__name__) )
# plt.plot( U_3[:, 0], U_3[:,1] , '--y' , lw = 1.0, label =r"{}".format(esquema_3.__name__) )
# plt.plot( U_4[:, 0], U_4[:,1] , '--c' , lw = 0.4, label =r"{}".format(esquema_4.__name__) )
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title( r'{} solved with $\Delta$t={}'.format( problema.__name__, round(dt, 5) ) )
plt.grid()
plt.show()


