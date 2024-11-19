# Gives the solution given by the specified 
from numpy import zeros
from ODES.Temporal_schemes import RK4, CN
from ODES.Cauchy_problem import Cauchy

# # Function for the Richardson's extrapolation to compute the numerical error of the Cauchy 
# # problem with a concrete scheme
# def Cauchy_Error(F, esquema, U0, t):
    
#     N = len(t) - 1
#     # Dada la partición t1 con N+1 puntos, obtiene la partición t2 con 2*N+1 puntos. 
#     # Los nodos pared de t2 
#     t1 = t
#     def refine_mesh(t1): 

#         t2 = zeros(2*N+1)
#         for i in range(0,N): 
#             t2[2*i] = t1[i]
#             t2[2*i+1] = ( t1[i+1] + t[i] ) / 2
        
#         t2[2*N] = t1[N]

#         return t2

#     # Particion equiespaciada de un segmento de la recta real, en N trozos, entre a y b
#     # def particion(a, b, N): 
#     #     t = zeros(N+1)
#     #     for i in range(0,N+1): 
#     #         t[i] = a + (b-a)/N *i
#     #     return t


#     t2 = refine_mesh(t1)
    
#     # Specification of the scheme order
#     if esquema==RK4: 
#         q=4
#     elif esquema==CN: 
#         q=2
#     else:
#         q=1

#     # Computing the solution with the different meshes
#     U1 = Cauchy(esquema, F, U0, t1)   # t1 = original mesh = coarse mesh
#     U2 = Cauchy(esquema, F, U0, t2)   # t2 = refined mesh of t1 
    
#     E = zeros( (N+1,len(U0)) )
#     for i in range(0,N):

#         E[i+1,:] = ( U1[i,:] - U2[2*i,:] ) / ( 1 - 2**q )

#     return U1, E

# Function for the Richardson's extrapolation to compute the numerical error of the Cauchy 
# problem with a concrete scheme
def Cauchy_Error(F, esquema, U0, t):
    
    N = len(t) - 1
    # Dada la partición t1 con N+1 puntos, obtiene la partición t2 con 2*N+1 puntos. 
    # Los nodos pared de t2 
    t1 = t
    def refine_mesh(t1): 

        t2 = zeros(2*N+1)
        for i in range(0,N): 
            t2[2*i] = t1[i]
            t2[2*i+1] = ( t1[i+1] + t[i] ) / 2
        
        t2[2*N] = t1[N]

        return t2


    t2 = refine_mesh(t1)
    
    # Specification of the scheme order
    if esquema==RK4: 
        q=4
    elif esquema==CN: 
        q=2
    else:
        q=1

    # Computing the solution with the different meshes
    U1 = Cauchy(esquema, F, U0, t1)   # t1 = original mesh = coarse mesh
    U2 = Cauchy(esquema, F, U0, t2)   # t2 = refined mesh of t1 
    
    E = zeros( (N+1,len(U0)) )
    for i in range(0,N):

        E[i+1,:] = ( U1[i,:] - U2[2*i,:] ) / ( 1 - 2**q )

    # return U1, E
    return E