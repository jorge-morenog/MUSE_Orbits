from numpy import array, zeros, linspace
import matplotlib.pyplot as plt 

#####################################################################
#                            FUNCIONES                              #
#####################################################################

##############################################
# FUNCIÓN DEL PROBLEMA DE KEPLER
##############################

def Kepler(U,t):  # El U que le llega es un vector
    
    F1 = U[2]
    F2 = U[3]
    F3 = - U[0]/(U[0]**2 + U[1]**2)**(3/2)
    F4 = - U[1]/(U[0]**2 + U[1]**2)**(3/2)
    
    F = array([F1,F2,F3,F4])
    
    return F

##############################################
# FUNCIONES DE LOS ESQUEMAS NUMÉRICOS
##############################

# Euler explícito 
def Euler(U, t, dt, F):  # U es un vector, t es un instante de tiempo 

    return U + dt * F(U, t)

# Adams-Bashforth de 2 pasos
def AB2(U2, U1, t, dt, F):

    return U2  + (dt/2)*( 3*F(U2,t) - F(U1,t-dt) )

# Runge-Kutta de 2 etapas
def RK2(U, t, dt, F):

    k1 = F( U         , t      )
    k2 = F( U + k1*dt , t + dt )

    return U  + (dt/2)*(k1 + k2)

# Runge-Kutta de 4 etapas
def RK4(U, t, dt, F):

    k1 = F( U           , t       )
    k2 = F( U + k1*dt/2 , t + dt/2)
    k3 = F( U + k2*dt/2 , t + dt/2)
    k4 = F( U + k3*dt   , t + dt  )

    return U  + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

##############################################
# FUNCIÓN PARA EL PROBLEMA DE CAUCHY
##############################
def Cauchy(F, t, U0, Esquema):

    N = len(t) - 1
    U = array( zeros( [len(t), len(U0)] ) ) # len(t) filas -> índices desde el 0 hasta el len(t)-1 // len(U0) columnas -> índices desde el 0 hasta len(U0)-1 
    U[0,:] = U0

    if Esquema == AB2:
        for n in range(0,N-1):   # empieza en 0 y acaba en (N-1)-1 = N-2 -> n toma N-1 valores ( desde 0 hasta N-2 )
            if n==0:
                U[n+1,:] = Euler(U[n,:], t[n], t[n+1]-t[n], F)
                U[n+2,:] = Esquema( U[n+1,:], U[n,:], t[n+1], t[n+2]-t[n+1], F )
            else: 
                U[n+2,:] = Esquema( U[n+1,:], U[n,:], t[n+1], t[n+2]-t[n+1], F )

    else: 
        for n in range(0,N):   # empieza en 0 y acaba en (N)-1 = N-1 -> n toma N valores ( desde 0 hasta N-1 )
            U[n+1,:] = Esquema( U[n,:], t[n], t[n+1]-t[n], F )

    return U


#####################################################################
#              DATOS (solo tocar esta parte del script)             #
#####################################################################
# Instante inicial
t0 = 0
# Instante final
tf = 20
# Número de intervalos
N = 200
# Condiciones iniciales
x0  = 1
y0  = 0
vx0 = 0
vy0 = 1
# Paso temporal constante o no 
paso = "si"

#####################################################################
#                             CÓDIGO                                #
#####################################################################

# Vector solución en el instante inicial
U0 = array( [x0,y0,vx0,vy0] )

# Instantes temporales en los que se evalúa la solución
if paso == "si":
    t = array( zeros([N+1]) )  # El vector t tiene N+1 componentes (índices desde 0 hasta N)
    dt = (tf-t0)/N
    # Vector de instantes temporales en los que se calcula la solución... linspace = paso temporal constante
    t = linspace(t0,tf,N+1)  # t0 dentro -> N+1 

# Solución con esquema Euler explícito
U_euler = Cauchy( Kepler, t, U0, Euler )
# Solución con esquema Adams-Bashforth de 2 pasos
U_ab2   = Cauchy( Kepler, t, U0, AB2   )
# Solución con esquema Runge-Kutta de 2 etapas
U_rk2   = Cauchy( Kepler, t, U0, RK2   )
# Solución con esquema Runge-Kutta de 4 etapas
U_rk4   = Cauchy( Kepler, t, U0, RK4   )

#####################################################################
#                             GRÁFICA                               #
#####################################################################
plt.figure(figsize=(13, 7))
plt.axis("equal")

plt.plot( U_euler[:, 0], U_euler[:,1] , '-b' , lw = 1.0, label ="Euler explícito" )
plt.plot( U_ab2[:, 0]  , U_ab2[:,1]   , '--k', lw = 1.0, label ="Adams-Bashforth 2" )
plt.plot( U_rk2[:, 0]  , U_rk2[:,1]   , '--g', lw = 1.0, label ="Runge-Kutta 2" )
plt.plot( U_rk4[:, 0]  , U_rk4[:,1]   , '-r' , lw = 1.0, label ="Runge-Kutta 4" )

plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title(r'Órbita con distintos esquemas ($\Delta$t={})'.format(round(dt,2)))
plt.grid()
plt.show()
