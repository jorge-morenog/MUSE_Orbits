from scipy.optimize import newton
from numpy import zeros, float64, dot, copy, ndarray, size, linspace
from numpy.linalg import norm
# from Cauchy_Error.Richardson import Cauchy_Error

#################################################################
# Definition of diferent temporal schemes
#################################################################

# Explicit Euler scheme
def Euler(F, U, dt, t, *args):

    return U + dt * F(U,t, *args)


# Adams-Bashforth de 2 pasos
def AB2(F, U2, U1, dt, t, *args):

    return U2  + (dt/2)*( 3*F(U2,t, *args) - F(U1,t-dt, *args) )


# 2nd order Runge-Kutta scheme 
def RK2(F, U, dt, t, *args):

    k1 = F( U         , t      , *args)
    k2 = F( U + k1*dt , t + dt , *args)

    return U  + (dt/2)*(k1 + k2)


# 4th order Runge-Kutta scheme
def RK4(F, U, dt, t, *args):

    k1 = F( U            , t       , *args)
    k2 = F( U + k1 * dt/2, t + dt/2, *args)
    k3 = F( U + k2 * dt/2, t + dt/2, *args)
    k4 = F( U + k3 * dt  , t + dt  , *args)
    
    return U + dt/6 * ( k1 + 2*k2 + 2*k3 + k4)


# Inverse Euler scheme
def Inv_Euler(F, U, dt, t, *args):

    def G(X, *args):
        return X - U - dt*F(X , t, *args)
    
    return newton(G, U, maxiter = 5000)  # Uses as initial value the one obtained in the previous step


# Crank-Nicolson scheme
def CN(F, U, dt, t, *args):

    def G(X, *args):
        return X - U - dt/2* ( F(U, t, *args) + F(X, t, *args) )
    
    return newton(G, U, maxiter = 5000)  # Uses as initial point the one obtained in the previous step 


# Leap-Frog scheme
def LF(F, U2, U1, dt, t, *args):

    return U1 + (2*dt)*F(U2, t, *args)

# Embedded Runge-Kutta
def Embedded_RK(F, U, dt, t, q, Tolerance, *args): 

    #(a, b, bs, c) = Butcher_array(q)
    #a, b, bs, c = Butcher_array(q)

    N_stages = { 2:2, 3:4, 8:13  }
    Ns = N_stages[q]
    a = zeros( (Ns, Ns), dtype=float64) 
    b = zeros(Ns); bs = zeros(Ns); c = zeros(Ns) 

    if Ns==2: 
        a[0,:] = [ 0, 0 ]
        a[1,:] = [ 1, 0 ] 
        b[:]  = [ 1/2, 1/2 ] 
        bs[:] = [ 1, 0 ] 
        c[:]  = [ 0, 1]  

    elif Ns==13: 
        c[:] = [ 0., 2./27, 1./9, 1./6, 5./12, 1./2, 5./6, 1./6, 2./3 , 1./3,   1., 0., 1.]
        
        a[0,:]  = [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0] 
        a[1,:]  = [ 2./27, 0., 0., 0., 0., 0., 0.,  0., 0., 0., 0., 0., 0] 
        a[2,:]  = [ 1./36 , 1./12, 0., 0., 0., 0., 0.,  0.,0., 0., 0., 0., 0] 
        a[3,:]  = [ 1./24 , 0., 1./8 , 0., 0., 0., 0., 0., 0., 0., 0., 0., 0] 
        a[4,:]  = [ 5./12, 0., -25./16, 25./16., 0., 0., 0., 0., 0., 0., 0., 0., 0]
        a[5,:]  = [ 1./20, 0., 0., 1./4, 1./5, 0., 0.,0., 0., 0., 0., 0., 0] 
        a[6,:]  = [-25./108, 0., 0., 125./108, -65./27, 125./54, 0., 0., 0., 0., 0., 0., 0] 
        a[7,:]  = [ 31./300, 0., 0., 0., 61./225, -2./9, 13./900, 0., 0., 0., 0., 0., 0] 
        a[8,:]  = [ 2., 0., 0., -53./6, 704./45, -107./9, 67./90, 3., 0., 0., 0., 0., 0] 
        a[9,:]  = [-91./108, 0., 0., 23./108, -976./135, 311./54, -19./60, 17./6, -1./12, 0., 0., 0., 0] 
        a[10,:] = [ 2383./4100, 0., 0., -341./164, 4496./1025, -301./82, 2133./4100, 45./82, 45./164, 18./41, 0., 0., 0] 
        a[11,:] = [ 3./205, 0., 0., 0., 0., -6./41, -3./205, -3./41, 3./41, 6./41, 0., 0., 0]
        a[12,:] = [ -1777./4100, 0., 0., -341./164, 4496./1025, -289./82, 2193./4100, 51./82, 33./164, 19./41, 0.,  1., 0]

        b[:]  = [ 41./840, 0., 0., 0., 0., 34./105, 9./35, 9./35, 9./280, 9./280, 41./840, 0., 0.] 
        bs[:] = [ 0., 0., 0., 0., 0., 34./105, 9./35, 9./35, 9./280, 9./280, 0., 41./840, 41./840]     


    
    k = RK_stages( F, U, t, dt, a, c ) 
    Error = dot( b-bs, k )

    dt_min = min( dt, dt * ( Tolerance / norm(Error) ) **(1/q) )
    N = int( dt/dt_min  ) + 1
    h = dt / N
    Uh = U.copy()

    for i in range(0, N): 

        k = RK_stages( F, Uh, t + h*i, h, a, c ) 
        Uh += h * dot( b, k )

    return Uh

def RK_stages( F, U, t, dt, a, c ): 

    k = zeros( (len(c), len(U)), dtype=float64 )

    for i in range(len(c)): 
        
        for  j in range(len(c)-1): 
          Up = U + dt * dot( a[i, :], k)

        k[i, :] = F( Up, t + c[i] * dt ) 

    return k 

def Butcher_array(q): 

    N_stages = { 2:2, 3:4, 8:13  }

    N =  N_stages[q]
    a = zeros((N, N), dtype = float64); 
    b = zeros((N)); bs = zeros((N)); c = zeros((N)) 
    
    if q==2: 

        a[0,:] = [ 0, 0 ]
        a[1,:] = [ 1, 0 ] 
        b[:]  = [ 1/2, 1/2 ] 
        bs[:] = [ 1, 0 ] 
        c[:]  = [ 0, 1]  

    elif q==3: 

        c[:] = [ 0., 1./2, 3./4, 1. ]

        a[0,:] = [  0., 0., 0.,  0        ]
        a[1,:] = [ 1./2, 0., 0., 0        ]
        a[2,:] = [ 0.,	3./4, 0., 0    	]
        a[3,:] = [ 2./9,	1./3,	4./9, 0	]

        b[:]  = [ 2./9,	1./3,	4./9,	0. ]
        bs[:] = [ 7./24,	1./4,	1./3,	1./8 ]
   
    elif q==8:
        
        c[:] = [ 0., 2./27, 1./9, 1./6, 5./12, 1./2, 5./6, 1./6, 2./3 , 1./3,   1., 0., 1.]

        a[0,:]  = [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0] 
        a[1,:]  = [ 2./27, 0., 0., 0., 0., 0., 0.,  0., 0., 0., 0., 0., 0] 
        a[2,:]  = [ 1./36 , 1./12, 0., 0., 0., 0., 0.,  0.,0., 0., 0., 0., 0] 
        a[3,:]  = [ 1./24 , 0., 1./8 , 0., 0., 0., 0., 0., 0., 0., 0., 0., 0] 
        a[4,:]  = [ 5./12, 0., -25./16, 25./16., 0., 0., 0., 0., 0., 0., 0., 0., 0]
        a[5,:]  = [ 1./20, 0., 0., 1./4, 1./5, 0., 0.,0., 0., 0., 0., 0., 0] 
        a[6,:]  = [-25./108, 0., 0., 125./108, -65./27, 125./54, 0., 0., 0., 0., 0., 0., 0] 
        a[7,:]  = [ 31./300, 0., 0., 0., 61./225, -2./9, 13./900, 0., 0., 0., 0., 0., 0] 
        a[8,:]  = [ 2., 0., 0., -53./6, 704./45, -107./9, 67./90, 3., 0., 0., 0., 0., 0] 
        a[9,:]  = [-91./108, 0., 0., 23./108, -976./135, 311./54, -19./60, 17./6, -1./12, 0., 0., 0., 0] 
        a[10,:] = [ 2383./4100, 0., 0., -341./164, 4496./1025, -301./82, 2133./4100, 45./82, 45./164, 18./41, 0., 0., 0] 
        a[11,:] = [ 3./205, 0., 0., 0., 0., -6./41, -3./205, -3./41, 3./41, 6./41, 0., 0., 0]
        a[12,:] = [ -1777./4100, 0., 0., -341./164, 4496./1025, -289./82, 2193./4100, 51./82, 33./164, 19./41, 0.,  1., 0]

        b[:]  = [ 41./840, 0., 0., 0., 0., 34./105, 9./35, 9./35, 9./280, 9./280, 41./840, 0., 0.] 
        bs[:] = [ 0., 0., 0., 0., 0., 34./105, 9./35, 9./35, 9./280, 9./280, 0., 41./840, 41./840]     
        
    else:
        print("Butcher array  not avialale for order =", q)
        exit()

    #return (a, b, bs, c) 
    return a, b, bs, c 

# Gragg-Bulirsh-Stoer scheme
def lagrange_interpolant(x, xp, f_values):
    """
    Calculates the value for the complete Largange polynomial at the xp value specified for the x nodes.
    INPUTS : x (nodes), f_values (value, at the x nodes, of the function to interpolate), xp (point where 
            we want to evaluate the Lagrange interpolant).
    OUTPUTS: result (value of the interpolant at the xp point specified).
    """
    N = len(x) - 1
    result = 0.0

    # Loop para calcular la combinación lineal de los polinomios de Lagrange
    for j in range(N + 1):
        L_j_xp = 1.0
        for r in range(N + 1):
            if r != j:
                L_j_xp *= (xp - x[r]) / (x[j] - x[r])
        
        result += f_values[j] * L_j_xp  # Sumar el término correspondiente

    return result

def GBS_NL_fixed(F, U1, dt, t1, NL_fixed, *args):

    Nv = len(U1)  # Number of variables of the problem

    hl = zeros(NL_fixed)  # vector that saves temporal step for every level of the GBS scheme
    U_filt = zeros((NL_fixed, Nv))
    U2 = zeros(Nv)

    for i in range(1, NL_fixed+1): 
            
            hl[i-1] = dt/(2*i) 
            U_lev = zeros((2*i+2,Nv))

            # Obtain the solution in every step of the submesh
            for j in range(0, 2*i + 2):   # For every level, We must have the solution in one more step than the final instant in order to filter

                if j == 0: 
                    U_lev[j,:] = U1
                elif j ==1: 
                    U_lev[j,:] = Euler(F, U_lev[j-1], hl[i-1], t1, *args)
                else: 
                    U_lev[j,:] = LF(F, U_lev[j-1], U_lev[j-2], hl[i-1], t1, *args)
                # Obtain the filtered solution at the final instant for the level "i"
                if j == (2*i+2)-1: 
                    U_filt[i-1,:] = (1/4)*( U_lev[j,:] + 2*U_lev[j-1,:] + U_lev[j-2,:]) 
    
    for k in range(0,Nv):
        U2[k] = lagrange_interpolant(hl[:], 0, U_filt[:,k])
    
    return U2

def GBS_variable_NL(F, U1, dt, t1, max_NL, tolerance, *args):
    """
    Gragg-Bulirsh-Stoer method with variable NL based on a tolerance.

    INPUTS:
    F          : Function defining the system of ODEs.
    U1         : Initial condition.
    dt         : Time step.
    t1         : Initial time.
    tolerance  : Tolerance to decide convergence.
    max_NL     : Maximum allowed number of levels.
    *args      : Additional arguments for the function F.

    OUTPUTS:
    U2         : Solution vector at the next time step.
    NL_used    : Number of levels used to achieve the desired tolerance.
    """
    Nv = len(U1)  # Number of variables of the problem

    hl = zeros(max_NL)  # Vector that saves time step for every level
    U_filt = zeros((max_NL, Nv))
    U2 = zeros(Nv)

    # Initialize variables for dynamic NL determination
    NL = 1  # Start with the first level

    while NL <= max_NL:
        hl[NL - 1] = dt / (2 * NL)
        U_lev = zeros((2 * NL + 2, Nv))

        # Obtain the solution for every step in the submesh
        for j in range(0, 2 * NL + 2):
            if j == 0:
                U_lev[j, :] = U1
            elif j == 1:
                U_lev[j, :] = Euler(F, U_lev[j - 1], hl[NL - 1], t1, *args)
            else:
                U_lev[j, :] = LF(F, U_lev[j - 1], U_lev[j - 2], hl[NL - 1], t1, *args)

            # Obtain the filtered solution at the final instant for the current level
            if j == (2 * NL + 2) - 1:
                U_filt[NL - 1, :] = (1 / 4) * (U_lev[j, :] + 2 * U_lev[j - 1, :] + U_lev[j - 2, :])

        # If we are beyond the first level, check the tolerance condition
        if NL > 1:
            # Calculate the difference between the interpolants for NL and NL-1
            interp_NL = zeros(Nv)
            interp_NL_minus_1 = zeros(Nv)

            for k in range(0, Nv):
                interp_NL[k] = lagrange_interpolant(hl[:NL], 0, U_filt[:NL, k])
                interp_NL_minus_1[k] = lagrange_interpolant(hl[:NL - 1], 0, U_filt[:NL - 1, k])

            norm_diff = norm(interp_NL - interp_NL_minus_1)

            # Check if the norm of the difference is within the tolerance
            if norm_diff < tolerance:
                U2 = interp_NL  # Solution corresponds to the current NL level
                # print("\nNL = ", NL)
                return U2

        # Increment NL to try the next level
        NL += 1

    # If max_NL is reached without meeting tolerance, use the last level
    if NL >= max_NL + 1: 
        print("\n----------------------- ERROR IN SOLUTION CONVERGENCE -----------------------------------")
        print(f"Convergence was not achieved for specified values of: tolerance = {tolerance}, max_NL = {max_NL}.\nTry giving a lower tolerance value or bigger max_NL.")
        print("-----------------------------------------------------------------------------------------\n")
        raise RuntimeError("Simulation stopped because the tolerance criterion was not met.")