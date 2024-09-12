#######################################################
# Orbita de Kepler con esquema Euler explicito
#####################

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

