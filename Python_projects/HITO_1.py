from numpy import array, zeros, linspace, concatenate
from numpy.linalg import norm
import matplotlib.pyplot as plt

def Kepler(U, t): # U = vector

    r = U[0:2]
    rdot = U[2:4]
    F = concatenate( (rdot,-r/norm(r)**3), axis=0 )

    return F

def Oscilador(U, t):

    x = U[0]
    xdot = U[1]
    F = array( [xdot, -x] )

    return F

# Esquema EULER explícito
def Euler(F, U, dt, t):

    return U + dt * F(U,t)


def RK4(F, U, dt, t):

    k1 = F(U,t)
    k2 = F( U + k1 * dt/2, t + dt/2)
    k3 = F( U + k2 * dt/2, t + dt/2)
    k4 = F( U + k3 * dt  , t + dt  )
    
    return U + dt/6 * ( k1 + 2*k2 + 2*k3 + k4)

# Obtener la solución de un problema dU/dt = F (ccii), dada una CI
# y un esquema temporal
# Inputs : 
#           Esquema temporal
#           Funcion F(U,t)
#           Condición inicial
#           Partición temporal
# Output: 
#           Solución para todo "t" y toda componente
def Cauchy(Esquema, F, U0, t):
    
    N = len(t)-1
    U = zeros((N+1, len(U0)))

    U[0,:] = U0
    for n in range(0,N):

        U[n+1,:] = Esquema( F, U[n,:], t[n+1]-t[n], t[n] )

    return U 




#################################### DATOS ####################################

# Condiciones iniciales
x0  = 1
y0  = 0
vx0 = 0
vy0 = 1
# Condiciones iniciales OSCILADOR
x0_osc  = 1
vx0_osc = 0
# Instantes inicial y final
t0 = 0
tf = 20
# Número de intervalos (=nº de instantes de tiempo - 1)
N = 200

################################### CÓDIGO ####################################

# Creamos vector de condiciones iniciales
U0 = array([x0,y0,vx0,vy0])
U0_osc = array([x0_osc, vx0_osc])

# Inicializamos matrices de soluciones
U_euler = zeros([N+1,len(U0)])
U_rk2   = zeros([N+1,len(U0)])
U_rk4   = zeros([N+1,len(U0)])
U_ab2   = zeros([N+1,len(U0)])

# Inicializamos lado derecho de EDO para Euler y AB2
F_euler = zeros([N+1,len(U0)])
F_ab2   = zeros([N+1,len(U0)])

k1_rk2 = zeros([N+1,len(U0)])
k2_rk2 = zeros([N+1,len(U0)])

k1_rk4 = zeros([N+1,len(U0)])
k2_rk4 = zeros([N+1,len(U0)])
k3_rk4 = zeros([N+1,len(U0)])
k4_rk4 = zeros([N+1,len(U0)])


# Inicializamos vector de instantes temporales en los que se calcula la solcuión
t = zeros(N+1)
# print(isinstance(U, list))  # Esto devolverá True si es una lista

# Creamos vector de instantes temporales
t = linspace(t0,tf,N+1)
# Como hemos usado linspace, definimos el dt (al usar linspace es constante)
dt = (tf-t0)/N

# Calculamos la solución para esquema EULER EXPLÍCITO
# U_euler[0,:] = U0
# for n in range(0,N):

    # F_euler[n,0] = U_euler[n,2]
    # F_euler[n,1] = U_euler[n,3]
    # F_euler[n,2] = - U_euler[n,0]/(U_euler[n,0]**2 + U_euler[n,1]**2)**(3/2)
    # F_euler[n,3] = - U_euler[n,1]/(U_euler[n,0]**2 + U_euler[n,1]**2)**(3/2)

    # F_euler[n,:] = Kepler(U_euler[n,:])

    # U_euler[n+1,0] = U_euler[n,0] + (t[n+1]-t[n])*F_euler[n,0]
    # U_euler[n+1,1] = U_euler[n,1] + (t[n+1]-t[n])*F_euler[n,1]
    # U_euler[n+1,2] = U_euler[n,2] + (t[n+1]-t[n])*F_euler[n,2]
    # U_euler[n+1,3] = U_euler[n,3] + (t[n+1]-t[n])*F_euler[n,3]

    # U_euler[n+1,:] = U_euler[n,:] + (t[n+1]-t[n])*F_euler[n,:]

    # U_euler[n+1,:] = Euler( Kepler, U_euler[n,:], dt, t[n] )
    # U_euler[n+1,:] = Euler( F =  Kepler, U = U_euler[n,:], dt = dt, t = t[n] )

# U_euler = Cauchy(Euler, Kepler, U0, t)

# U_rk4   = Cauchy(RK4  , Kepler, U0, t)

U_osc_euler = Cauchy(Euler, Oscilador, U0_osc, t )
U_osc_rk4 = Cauchy(RK4, Oscilador, U0_osc, t )
# Calculamos la solución para esquema RUNGE-KUTTA DE 2 ETAPAS
U_rk2[0,:] = U0
for n in range(0,N):

    # k1 = F(Un,tn)
    k1_rk2[n,0] = U_rk2[n,2]
    k1_rk2[n,1] = U_rk2[n,3]
    k1_rk2[n,2] = - U_rk2[n,0]/(U_rk2[n,0]**2 + U_rk2[n,1]**2)**(3/2)
    k1_rk2[n,3] = - U_rk2[n,1]/(U_rk2[n,0]**2 + U_rk2[n,1]**2)**(3/2)

    # k2 = F(Un + k1·dt, tn+dt)
    k2_rk2[n,0]= U_rk2[n,2] + k1_rk2[n,2]*(t[n+1]-t[n])
    k2_rk2[n,1]= U_rk2[n,3] + k1_rk2[n,3]*(t[n+1]-t[n])
    k2_rk2[n,2] = - ( U_rk2[n,0] + k1_rk2[n,0]*(t[n+1]-t[n]) ) / ( ( U_rk2[n,0] + k1_rk2[n,0]*(t[n+1]-t[n]) )**2 + ( U_rk2[n,1] + k1_rk2[n,1]*(t[n+1]-t[n]) )**2 )**(3/2)
    k2_rk2[n,3] = - ( U_rk2[n,1] + k1_rk2[n,1]*(t[n+1]-t[n]) ) / ( ( U_rk2[n,0] + k1_rk2[n,0]*(t[n+1]-t[n]) )**2 + ( U_rk2[n,1] + k1_rk2[n,1]*(t[n+1]-t[n]) )**2 )**(3/2)

    # U(n+1)=U(n) + (dt/2)*( k1+k2 )
    U_rk2[n+1,0] = U_rk2[n,0] + (t[n+1]-t[n])/2 *(k1_rk2[n,0]+k2_rk2[n,0])
    U_rk2[n+1,1] = U_rk2[n,1] + (t[n+1]-t[n])/2 *(k1_rk2[n,1]+k2_rk2[n,1])
    U_rk2[n+1,2] = U_rk2[n,2] + (t[n+1]-t[n])/2 *(k1_rk2[n,2]+k2_rk2[n,2])
    U_rk2[n+1,3] = U_rk2[n,3] + (t[n+1]-t[n])/2 *(k1_rk2[n,3]+k2_rk2[n,3])

# Calculamos la solución para esquema RUNGE-KUTTA DE 4 ETAPAS
# U_rk4[0,:] = U0
# for n in range(0,N):

#     # k1 = F(Un,tn)
#     k1_rk4[n,0] = U_rk4[n,2]
#     k1_rk4[n,1] = U_rk4[n,3]
#     k1_rk4[n,2] = - U_rk4[n,0]/(U_rk4[n,0]**2 + U_rk4[n,1]**2)**(3/2)
#     k1_rk4[n,3] = - U_rk4[n,1]/(U_rk4[n,0]**2 + U_rk4[n,1]**2)**(3/2)

#     # k2 = F(Un + k1·(dt/2), tn + dt/2)
#     k2_rk4[n,0]= U_rk4[n,2] + k1_rk4[n,2]*(t[n+1]-t[n])/2
#     k2_rk4[n,1]= U_rk4[n,3] + k1_rk4[n,3]*(t[n+1]-t[n])/2
#     k2_rk4[n,2] = - ( U_rk4[n,0] + k1_rk4[n,0]*(t[n+1]-t[n])/2 ) / ( ( U_rk4[n,0] + k1_rk4[n,0]*(t[n+1]-t[n])/2 )**2 + ( U_rk4[n,1] + k1_rk4[n,1]*(t[n+1]-t[n])/2 )**2 )**(3/2)
#     k2_rk4[n,3] = - ( U_rk4[n,1] + k1_rk4[n,1]*(t[n+1]-t[n])/2 ) / ( ( U_rk4[n,0] + k1_rk4[n,0]*(t[n+1]-t[n])/2 )**2 + ( U_rk4[n,1] + k1_rk4[n,1]*(t[n+1]-t[n])/2 )**2 )**(3/2)
    
#     # k3 = F(Un + k2·(dt/2), tn + dt/2)
#     k3_rk4[n,0]= U_rk4[n,2] + k2_rk4[n,2]*(t[n+1]-t[n])/2
#     k3_rk4[n,1]= U_rk4[n,3] + k2_rk4[n,3]*(t[n+1]-t[n])/2
#     k3_rk4[n,2] = - ( U_rk4[n,0] + k2_rk4[n,0]*(t[n+1]-t[n])/2 ) / ( ( U_rk4[n,0] + k2_rk4[n,0]*(t[n+1]-t[n])/2 )**2 + ( U_rk4[n,1] + k2_rk4[n,1]*(t[n+1]-t[n])/2 )**2 )**(3/2)
#     k3_rk4[n,3] = - ( U_rk4[n,1] + k2_rk4[n,1]*(t[n+1]-t[n])/2 ) / ( ( U_rk4[n,0] + k2_rk4[n,0]*(t[n+1]-t[n])/2 )**2 + ( U_rk4[n,1] + k2_rk4[n,1]*(t[n+1]-t[n])/2 )**2 )**(3/2)
    
#     # k2 = F(Un + k3·dt, tn + dt)
#     k4_rk4[n,0]= U_rk4[n,2] + k3_rk4[n,2]*(t[n+1]-t[n])
#     k4_rk4[n,1]= U_rk4[n,3] + k3_rk4[n,3]*(t[n+1]-t[n])
#     k4_rk4[n,2] = - ( U_rk4[n,0] + k3_rk4[n,0]*(t[n+1]-t[n]) ) / ( ( U_rk4[n,0] + k3_rk4[n,0]*(t[n+1]-t[n]) )**2 + ( U_rk4[n,1] + k3_rk4[n,1]*(t[n+1]-t[n]) )**2 )**(3/2)
#     k4_rk4[n,3] = - ( U_rk4[n,1] + k3_rk4[n,1]*(t[n+1]-t[n]) ) / ( ( U_rk4[n,0] + k3_rk4[n,0]*(t[n+1]-t[n]) )**2 + ( U_rk4[n,1] + k3_rk4[n,1]*(t[n+1]-t[n]) )**2 )**(3/2)
    
#     # U(n+1)=U(n) + (dt/2)*( k1+k2 )
#     U_rk4[n+1,0] = U_rk4[n,0] + (t[n+1]-t[n])/6 *( k1_rk4[n,0] + 2*k2_rk4[n,0] +2*k3_rk4[n,0] + k4_rk4[n,0])
#     U_rk4[n+1,1] = U_rk4[n,1] + (t[n+1]-t[n])/6 *( k1_rk4[n,1] + 2*k2_rk4[n,1] +2*k3_rk4[n,1] + k4_rk4[n,1])
#     U_rk4[n+1,2] = U_rk4[n,2] + (t[n+1]-t[n])/6 *( k1_rk4[n,2] + 2*k2_rk4[n,2] +2*k3_rk4[n,2] + k4_rk4[n,2])
#     U_rk4[n+1,3] = U_rk4[n,3] + (t[n+1]-t[n])/6 *( k1_rk4[n,3] + 2*k2_rk4[n,3] +2*k3_rk4[n,3] + k4_rk4[n,3])

# Calculamos la solución para esquema ADAMS-BASHFORTH DE 2 PASOS
# U_ab2[0,:] = U0
# for n in range(0,N-1):

#     F_ab2[n,0] = U_ab2[n,2]
#     F_ab2[n,1] = U_ab2[n,3]
#     F_ab2[n,2] = - U_ab2[n,0]/(U_ab2[n,0]**2 + U_ab2[n,1]**2)**(3/2)
#     F_ab2[n,3] = - U_ab2[n,1]/(U_ab2[n,0]**2 + U_ab2[n,1]**2)**(3/2)

#     # U(n+1) = U(n) + (dt/2)*( 3F(n) - F(n-1) )
#     if n==0: # El valor en el instante dt lo tomamos del Euler
#         U_ab2[n+1,0] = U_ab2[n,0] + (t[n+1]-t[n])*F_ab2[n,0]
#         U_ab2[n+1,1] = U_ab2[n,1] + (t[n+1]-t[n])*F_ab2[n,1]
#         U_ab2[n+1,2] = U_ab2[n,2] + (t[n+1]-t[n])*F_ab2[n,2]
#         U_ab2[n+1,3] = U_ab2[n,3] + (t[n+1]-t[n])*F_ab2[n,3]

#         U_ab2[n+2,0] = U_ab2[n+1,0] + (t[n+2]-t[n-1])/2 *(3*F_ab2[n+1,0] - F_ab2[n,0])
#         U_ab2[n+2,1] = U_ab2[n+1,1] + (t[n+2]-t[n-1])/2 *(3*F_ab2[n+1,1] - F_ab2[n,1])
#         U_ab2[n+2,2] = U_ab2[n+1,2] + (t[n+2]-t[n-1])/2 *(3*F_ab2[n+1,2] - F_ab2[n,2])
#         U_ab2[n+2,3] = U_ab2[n+1,3] + (t[n+2]-t[n-1])/2 *(3*F_ab2[n+1,3] - F_ab2[n,3])

#     else:
#         U_ab2[n+2,0] = U_ab2[n+1,0] + (t[n+2]-t[n-1])/2 *(3*F_ab2[n+1,0] - F_ab2[n,0])
#         U_ab2[n+2,1] = U_ab2[n+1,1] + (t[n+2]-t[n-1])/2 *(3*F_ab2[n+1,1] - F_ab2[n,1])
#         U_ab2[n+2,2] = U_ab2[n+1,2] + (t[n+2]-t[n-1])/2 *(3*F_ab2[n+1,2] - F_ab2[n,2])
#         U_ab2[n+2,3] = U_ab2[n+1,3] + (t[n+2]-t[n-1])/2 *(3*F_ab2[n+1,3] - F_ab2[n,3])

# plt.figure(figsize=(13, 7))
# plt.axis("equal")

# plt.plot( U_euler[:, 0], U_euler[:,1] , '-b' , lw = 1.0, label ="Euler explícito" )
# plt.plot( U_rk2[:, 0]  , U_rk2[:,1]   , '--g', lw = 1.0, label ="Runge-Kutta 2" )
# plt.plot( U_rk4[:, 0]  , U_rk4[:,1]   , '-r' , lw = 1.0, label ="Runge-Kutta 4" )
# # plt.plot( U_ab2[:, 0]  , U_ab2[:,1]   , '--k', lw = 1.0, label ="Adams-Bashforth 2" )

# plt.legend()
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title(r'Órbita con distintos esquemas ($\Delta$t={})'.format(round(dt,2)))
# plt.grid()
# plt.show()

plt.figure(figsize=(13, 7))
plt.axis("equal")

plt.plot( t, U_osc_euler[:,0] , '-b' , lw = 1.0, label ="Euler explícito" )
plt.plot( t, U_osc_rk4[:,0] , '-r' , lw = 1.0, label ="rk4" )

plt.legend()
plt.xlabel('t')
plt.ylabel('x')
plt.title(r'Oscilador ($\Delta$t={})'.format(round(dt,2)))
plt.grid()
plt.show()

print("U_osc =", U_osc_euler)