#################################################################
# Definición del problema de Cauchy
#################################################################
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
from numpy import zeros
# from numpy.linalg import norm
from ODES.Temporal_schemes import Euler, AB2

##################################
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