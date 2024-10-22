from scipy.optimize import newton

#################################################################
# Definición de los distintos esquemas temporales
#################################################################

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


# Esquema EULER inverso
def Inv_Euler(F, U, dt, t):

    def G(X):
        return X - U - dt*F(X,t)
    
    return newton(G, U)  # Utiliza como punto inicial el valor de la solución en el instante anterior

# Esquema Crank-Nicolson
def CN(F, U, dt, t):

    def G(X):
        return X - U - dt/2* ( F(U,t) + F(X,t) )
    
    return newton(G, U)  # Utiliza como punto inicial el valor de la solución en el instante anterior