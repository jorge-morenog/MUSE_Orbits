from numpy import array, concatenate
from numpy.linalg import norm

U = array([1,2,3,4])

print("U=", U[0:1])


r = U[0:2]
rdot = U[2:4]

F = concatenate((rdot,-r/norm(r)**3), axis=1)


print("F =", F)