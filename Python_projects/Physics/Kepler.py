from numpy import concatenate
from numpy.linalg import norm

# Problema de Kepler: rvec = -rvec / norm(rvec)^3.
def Kepler(U, t): # U = vector

    r = U[0:2]
    rdot = U[2:4]
    F = concatenate( (rdot,-r/norm(r)**3), axis=0 )

    return F