from numpy import array, zeros, linspace, concatenate
from numpy.linalg import norm
import matplotlib.pyplot as plt
import inspect

###########################################################
#                       PROBLEMAS                         #
###########################################################

# Problema de Kepler: rvec = -rvec / norm(rvec)^3
def Kepler(U, t): # U = vector

    r = U[0:2]
    rdot = U[2:4]
    F = concatenate( (rdot,-r/norm(r)**3), axis=0 )

    return F

# Problema de Oscilador armónico: xdot2 + x = 0 
def Oscilador(U, t):

    x = U[0]
    xdot = U[1]
    F = array( [xdot, -x] )

    return F

###########################################################
#                   ESQUEMAS NUMÉRICOS                    #
###########################################################

# Esquema EULER explícito
def Euler(F, U, dt, t):

    return U + dt * F(U,t)

# Adams-Bashforth de 2 pasos
def AB2(F, U2, U1, dt, t):

    return U2  + (dt/2)*( 3*F(U2,t) - F(U1,t-dt) )

# Runge-Kutta de 2 etapas
def RK2(F, U, dt, t):

    k1 = F( U         , t      )
    k2 = F( U + k1*dt , t + dt )

    return U  + (dt/2)*(k1 + k2)

# Runge-Kutta de 4 etapas
def RK4(F, U, dt, t):

    k1 = F(U,t)
    k2 = F( U + k1 * dt/2, t + dt/2)
    k3 = F( U + k2 * dt/2, t + dt/2)
    k4 = F( U + k3 * dt  , t + dt  )
    
    return U + dt/6 * ( k1 + 2*k2 + 2*k3 + k4)

###########################################################
#                   PROBLEMA DE CAUCHY                    #
###########################################################
#
# Obtener la solución de un problema dU/dt = F (ccii), dada una CI
# y un esquema temporal
#
# Inputs : 
#          -> Esquema temporal
#          -> Funcion F(U,t)
#          -> Condición inicial
#          -> Partición temporal
#
# Output :  
#          -> Solución para todo "t" y toda componente
#
def Cauchy(Esquema, F, U0, t):
    
    N = len(t)-1
    U = zeros((N+1, len(U0)))

    U[0,:] = U0
    if Esquema == AB2:
        for n in range(0,N-1):   # empieza en 0 y acaba en (N-1)-1 = N-2 -> n toma N-1 valores ( desde 0 hasta N-2 )
            if n==0:
                U[n+1,:] = Euler(F, U[n,:], t[n+1]-t[n], t[n])
                U[n+2,:] = Esquema( F, U[n+1,:], U[n,:], t[n+2]-t[n+1], t[n+1] )
            else: 
                U[n+2,:] = Esquema( F, U[n+1,:], U[n,:], t[n+2]-t[n+1], t[n+1] )
    else:
        for n in range(0,N):
            U[n+1,:] = Esquema( F, U[n,:], t[n+1]-t[n], t[n] )

    return U 

###########################################################
#                          DATOS                          #
###########################################################
# Selecciona el problema que quieres resolver (de los implementados en las funciones)
problema = Kepler

# Condiciones iniciales
x0_kep  = 1
y0_kep  = 0
vx0_kep = 0
vy0_kep = 1

# Condiciones iniciales OSCILADOR
x0_osc  = 1
vx0_osc = 0

# Instantes inicial y final
t0 = 0
tf = 20

# Número de intervalos (=nº de instantes de tiempo - 1)
N = 200

###########################################################
#                         CÓDIGO                          #
###########################################################

# Creamos vector de condiciones iniciales
if   problema == Kepler:
    U0 = array( [x0_kep,y0_kep,vx0_kep,vy0_kep] )

elif problema == Oscilador: 
    U0 = array( [x0_osc, vx0_osc] )

else:
    print(f"\n ¡¡¡¡ Especifique U0 metiendo un elif más cerca de la línea de código {inspect.currentframe().f_lineno} !!!! \n")

# Inicializamos y creamos el vector de instantes de tiempo
t  = zeros(N+1)
t  = linspace(t0,tf,N+1)
dt = (tf-t0)/N

# Obtenemos la solución del problema de Cauchy que hemos especificado
U_euler = Cauchy(Euler, problema, U0, t)
U_rk2   = Cauchy(RK2  , problema, U0, t)
U_rk4   = Cauchy(RK4  , problema, U0, t)
U_ab2   = Cauchy(AB2  , problema, U0, t)

###########################################################
#                         GRÁFICAS                        #
###########################################################

plt.figure(figsize=(13, 7))
plt.axis("equal")
if problema == Kepler: 
    plt.plot( U_euler[:, 0], U_euler[:,1] , '-b' , lw = 1.0, label ="Euler explícito" )
    plt.plot( U_rk2[:, 0]  , U_rk2[:,1]   , '--g', lw = 1.0, label ="Runge-Kutta 2" )
    plt.plot( U_rk4[:, 0]  , U_rk4[:,1]   , '-r' , lw = 1.0, label ="Runge-Kutta 4" )
    plt.plot( U_ab2[:, 0]  , U_ab2[:,1]   , '--k', lw = 1.0, label ="Adams-Bashforth 2" )
    plt.xlabel('x')
    plt.ylabel('y')

elif problema == Oscilador:  
    plt.plot( t, U_euler[:,0] , '-b' , lw = 1.0, label ="Euler explícito" )
    plt.plot( t, U_rk2[:,0]   , '--g', lw = 1.0, label ="Runge-Kutta 2" )
    plt.plot( t, U_rk4[:,0]   , '-r' , lw = 1.0, label ="Runge-Kutta 4" )
    plt.plot( t, U_ab2[:,0]   , '--k', lw = 1.0, label ="Adams-Bashforth 2" )
    plt.xlabel('t')
    plt.ylabel('x')

else:
    print(f"\n ¡¡¡¡ Especifique gráfica metiendo un elif más cerca de la línea de código {inspect.currentframe().f_lineno} !!!! \n")


plt.legend()
plt.title( r'{} resuelto con $\Delta$t={}'.format( problema.__name__, round(dt, 2) ) )
plt.grid()
plt.show()
