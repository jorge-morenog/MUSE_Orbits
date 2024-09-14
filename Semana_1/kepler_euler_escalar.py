### Problema de Kepler (EDO 2º orden): Problema de Cauchy (V.I) ###
#
#        d^2r_vec/dt^2 = r_vec / |r_vec|^3
#
# donde |r_vec|^3 = (x^2 + y^2)^(3/2).
#
# De esta ecuación vectorial se sacan 2 ecuaciones escalares (una en x y otra en y): 
#
#        d^x/dt^2 = x / (x^2 + y^2)^(3/2)  ;  x(0)=1, dx/dt(0)=0 .
#        d^y/dt^2 = y / (x^2 + y^2)^(3/2)  ;  y(0)=0, dx/dt(0)=1 .
#
# PASO 1: Tomar nuevas variables para reducir el orden de las EDOs
#        dx/dt = v_x            # Introduce una EDO adicional!!
#        dy/dt = v_y            # Introduce una EDO adicional!!
#
# PASO 2: Planteamos el sistema de EDOs que nos queda
#   (1)  dx/dt = v_x
#   (2)  dy/dt = v_y
#   (3)  dz_x/dt = x / (x^2 + y^2)^(3/2)
#   (4)  dz_y/dt = y / (x^2 + y^2)^(3/2)
#
# PASO 3: Planteamos el esquema que seguiremos en el script
#
#        dU/dt = F(U;t)
#
#        U = [x, y, dx/dt, dy/dt]    
#        F = [dx/dt, dy/dt, x/(x^2+y^2)^(3/2), y/(x^2+y^2)^(3/2)]
#
# es decir: 
#
#        U(0) = x ,     U(1) = y ,     U(2) = dx/dt,     U(3) = dy/dt
#
#        F = [U(2),    U(3),    U(0)/(U(0)**2 + U(1)**2)**(3/2),    U(1)/(U(0)**2 + U(1)**2)**(3/2) ]
#

# Instante inicial
t0 = 0
# Instante final
tf = 3
# Numero de intervalos
N = 50
# Incremento de tiempo
delta_t = (tf-t0)/N
# Vector de condiciones iniciales
U0 = [1,0,0,1]


def F(x,y,x_dot,y_dot):
    F1 = x_dot
    F2 = y_dot
    F3 = - x / (x**2 + y**2)**(3/2)
    F4 = - y / (x**2 + y**2)**(3/2)

    F=(F1,F2,F3,F4)
    return F

