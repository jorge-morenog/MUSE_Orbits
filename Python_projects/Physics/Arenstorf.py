from numpy import array, sqrt
from numpy.linalg import norm

def Arenstorf(U, t):
    
    mu = 0.012277471  # Value for Earth-Moon
    
    x , y, vx, vy  = U
    
    D1 = sqrt( (x+mu)**2 + y**2 )**3
    D2 = sqrt( (x-(1-mu))**2 + y**2 )**3
    
    dxdt = vx
    dydt = vy
    dvxdt = x + 2*vy - (1-mu)*( x + mu )/D1 - mu*(x-(1-mu))/D2
    dvydt = y - 2*vx - (1-mu) * y/D1- mu * y/D2
    
    F = array([ dxdt, dydt, dvxdt, dvydt ])
    
    return F
