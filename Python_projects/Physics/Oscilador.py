# Problema de Oscilador armónico: xdot2 + x = 0. 
def Oscilador(U, t):

    x = U[0]
    xdot = U[1]
    F = array( [xdot, -x] )

    return F