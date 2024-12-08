from numpy import meshgrid, array, zeros, size, absolute, sqrt, sin, where, nan, zeros_like
from ODES.Temporal_schemes import Euler, Inv_Euler, RK2, RK4, CN, LF, AB2
import math

def StabilityPolynomial(scheme, w):
    if scheme == Euler: 
        r = 1 + w
    elif scheme == Inv_Euler: 
        r = 1 / (1 - w)
    elif scheme == RK2:
        r = 1 + w + (w**2)/2
    elif scheme == RK4:
        r = 1 + w + (w**2)/2 + (w**3)/(3*2) + (w**4)/(4*3*2)
    elif scheme == CN:
        r = (1 + w/2) / (1 - w/2)
    elif scheme == LF: # Leap-Frog has REA along the imaginary axis between -i and +i
        r = 1 + 0 * w  # Dummy to fit structure; we handle LF separately later (watch StabilityRegion function)
    elif scheme == AB2:
        r = 1/2 + 3/4 * w + sqrt(1/4 - 1/2 * w + 9/16 * w**2)

    return r

def StabilityRegion(x, y, scheme):
    Z = array([[complex(re, im) for re in x] for im in y])
    R = abs(StabilityPolynomial(scheme, Z))

    if scheme == LF:
        # For LF, explicitly enforce REA on the imaginary axis between -i and +i
        R = zeros_like(Z, dtype=float)
        for i, im in enumerate(y):
            for j, re in enumerate(x):
                if re == 0 and -1 <= im <= 1:
                    R[i, j] = 1
    return R