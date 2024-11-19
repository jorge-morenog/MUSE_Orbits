from numpy import zeros, linspace, log10
from numpy.linalg import norm
from ODES.Cauchy_problem import Cauchy
# || E[N,:]|| = ||k|| dt^q ; N·dt = T, dt = T/N
# log( ||E|| ) = log( ||k|| ) + q·log( dt ) = log( ||k|| ) + q·( log(T) - log(N) ) = log( ||k|| ) + q·log(T) - q·log(N)
# Calculamos un multiplo del error, que es como trasladar la recta paralelamente: 
# multE = U2-U1 ; log( ||E|| ) = log( U2-U1 ) - log(1-1/2^q) -> ese ultimmo termino traslada la recta
# Hago gráfica log(multE)-log(N). 


# Inputs : Funcion del PVI (F), condicion inicial (U0), t (instantes de tiempo en los que se calcula la solución) y esquema numérico (scheme)
# Outputs: orden del esquema numérico
def Conv(F, scheme, U0, t, ptsgraf, Nf):  
    
    N = len(t) - 1   # Number of intervals for the first N-value

    interval = (Nf-N) / ( ptsgraf - 1 )

    Ngraf = zeros(ptsgraf, dtype = int)
    Ngraf[0] = N

    U1graf = zeros( (ptsgraf, len(U0)) )
    U1graf[0,:] = Cauchy(scheme, F, U0, t)[-1,:]

    U2graf = zeros( (ptsgraf, len(U0)) )
    U2graf[0,:] = Cauchy(scheme, F, U0, linspace(t[0],t[-1],2*N+1))[-1,:]
    
    logmultE = zeros( (ptsgraf, len(U0)) )
    logmultE[0,:] = log10( norm( U2graf[0,:] - U1graf[0,:] ) )

    for i in range(1, ptsgraf): 

        # Ngraf[i] = (2**i) * N      # Number of intervals for the "i" N-value
        Ngraf[i] = N + i*interval   # Number of intervals for the "i" N-value
        print("N = ", Ngraf[i])

        t1 = zeros(Ngraf[i]+1)
        t2 = zeros(2*Ngraf[i]+1)

        t1 = linspace(t[0], t[-1], Ngraf[i]+1)     # dt
        t2 = linspace(t[0], t[-1], 2*Ngraf[i]+1)   # dt/2

        U1graf[i,:] = Cauchy(scheme, F, U0, t1)[-1,:]  # Solution for t   at final instant for Ni
        U2graf[i,:] = Cauchy(scheme, F, U0, t2)[-1,:]  # Solution for t/2 at final instant for Ni

        logmultE[i,:] = log10( norm( U2graf[i,:] - U1graf[i,:] ) )

    return log10( Ngraf ), logmultE