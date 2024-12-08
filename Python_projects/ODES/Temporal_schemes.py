from scipy.optimize import newton

#################################################################
# Definición de los distintos esquemas temporales
#################################################################

# Explicit Euler scheme
def Euler(F, U, dt, t):

    return U + dt * F(U,t)


# Adams-Bashforth de 2 pasos
def AB2(F, U2, U1, dt, t):

    return U2  + (dt/2)*( 3*F(U2,t) - F(U1,t-dt) )


# 2nd order Runge-Kutta scheme 
def RK2(F, U, dt, t):

    k1 = F( U         , t      )
    k2 = F( U + k1*dt , t + dt )

    return U  + (dt/2)*(k1 + k2)


# 4th order Runge-Kutta scheme
def RK4(F, U, dt, t):

    k1 = F(U,t)
    k2 = F( U + k1 * dt/2, t + dt/2)
    k3 = F( U + k2 * dt/2, t + dt/2)
    k4 = F( U + k3 * dt  , t + dt  )
    
    return U + dt/6 * ( k1 + 2*k2 + 2*k3 + k4)


# Inverse Euler scheme
def Inv_Euler(F, U, dt, t):

    def G(X):
        return X - U - dt*F(X,t)
    
    return newton(G, U, maxiter = 5000)  # Utiliza como punto inicial el valor de la solución en el instante anterior


# Crank-Nicolson scheme
def CN(F, U, dt, t):

    def G(X):
        return X - U - dt/2* ( F(U,t) + F(X,t) )
    
    return newton(G, U, maxiter = 5000)  # Uses as initial point the one obtained in the previous step 


# Leap-Frog scheme
def LF(F, U2, U1, dt, t):

    return U1 + (2*dt)*F(U2,t)
