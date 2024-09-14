from numpy import zeros, transpose
##########################################################
#                         DATOS                          #
##########################################################
# Condición inicial de posición
x0 = 1
y0 = 0
# Condición inicial de velocidad
vx0 = 0
vy0 = 1
# Instantes inicial y final
t0 = 0
tf = 20
# Número de inervalos en la malla temporal
N = 500

##########################################################
#                        CÓDIGO                          #
##########################################################
# Función: problema de Kepler
def Kepler(U): 
    x, y, x_dot, y_dot = U

    F1 = x_dot
    F2 = y_dot
    F3 = -x /(x**2+y**2)**(3/2)
    F4 = -y /(x**2+y**2)**(3/2)

    return [F1,F2,F3,F4]

# Función: esquema Euler explícito
def Euler(U0,N,dt): 
    U=zeros((len(U0),N+1))
    U[:,0]=U0
    
    i=0
    for i in range(0,N,1):
        # print("------------------------i=", i)
        U[:,i+1]= U[:,i] + transpose(Kepler(U[:,i]))*dt
        # print("U=", U)
        i=+1
    return U

# Incremento temporal
dt = (tf-t0)/N
# Solución en el instante inicial
U0 = [x0,y0,vx0,vy0]
# Solución numérica con el método de Euler explícito
sol_euler = Euler(U0,N,dt)
x_sol = sol_euler[0,:]
y_sol = sol_euler[1,:]

##########################################################
#                        GRÁFICA                         #
##########################################################
import matplotlib.pyplot as plt
plt.plot(x_sol, y_sol, 'b', lw = 1.0)
plt.xlabel('x')
plt.ylabel('y')
plt.title(r'Órbita de Kepler con esquema Euler explícito ($\Delta$t={}s)'.format(dt))
plt.grid()
plt.show()