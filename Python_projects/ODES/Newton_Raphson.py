#####################################################################
# Script que implementa el método de Newton-Raphson
#####################################################################
#
# NOTA: Esto lo usaremos para los esquemas numéricos implícitos, 
# por ejemplo, Euler inverso, Crank-Nicolson etc. 
#####################################################################
# Problema: Obtener los ceros de la función x^3 - 2x - 5 = 0. 
#
# Paso 1) Construir función implícita para buscar sus ceros
#
#                       F(x) = x^3 - 2x - 5 = 0
# 
# Paso 2) Calcular la derivada de la función implícita
#
#                       F'(x) = 3x^2 -2
#
# Paso 3) Selecciono punto de inicio y resuelvo con el Método
#
#                       x_inicio = x0
#                       x1 = x0 + f(x0)/f'(x0)
#                       G(x1) <= tolerancia => x1 = solución
#
#                       G(x1) > tolerancia 
#                       x2 = x1 + f(x1)/f'(x1)
#                       G(x2) <= tolerancia => x2 = solución
#                       etc.
#
# NOTA: Hay que establecer un número máximo de iteraciones. Si la 
# solución no se encuentra dentro de ese número de iteraciones es 
# porque tenemos que dar otro valor al punto inicial. 
#
#
#
#####################################################################
#                            LIBRERÍAS                              #
#####################################################################
from numpy import sin, cos, log, exp, linspace, array, zeros, transpose
import matplotlib.pyplot as plt 

#####################################################################
#                               DATOS                               #
#####################################################################
x0 = -10
max_iter = 500
epsilon = 10**(-9)

#####################################################################
#                            FUNCIONES                              #
#####################################################################

# Funciones modificables: F(x) y F'(x)
def funcion(x): 
    # F = x**3 -2*x -5 
    # F = sin(x) - x/2 - log(x+1)
    # F = exp(x) - 2*x 
    F = cos(x) - x**2
    return F

def derivada(x): 
    # dFdx = 3*x**2 - 2
    # dFdx = cos(x) - 1/2 -1/(x+1)
    # dFdx = exp(x) - 2
    dFdx = -sin(x) - 2*x
    return dFdx

# Función NO modificable: cálculo del nuevo punto
def punto(x):
    xnew = x - funcion(x)/derivada(x)
    return xnew
#####################################################################
#                        CÓDIGO (no tocar)                          #
#####################################################################

x_eval = x0
for i in range(1,max_iter+2):

    cero = funcion(x_eval)

    if i == max_iter: 
        print("\n----------- SOLUCIÓN NO ENCONTRADA -----------")
        print("x_inicial      =", x0)
        print("tolerancia     =", epsilon)
        print("nº iteraciones = nº máximo =", i)
        print("x_ultimo       =", x_eval)
        print("cero           =", cero)
        print("\nNOTA: Probar con otro x0 o aumentar nº de iteraciones máximas")
        print("------------------------------------------\n")
        break
    else: 
        if abs(cero) <= epsilon:
            print("\n---------------- SOLUCIÓN ----------------")
            print("x_inicial      =", x0)
            print("tolerancia     =", epsilon)
            print("nº iteraciones =", i)
            print("x_sol          =", x_eval)
            print("------------------------------------------\n")
            break
        else:
            x_eval = punto(x_eval)

x = linspace(0, 1, 100)  # Más puntos para una gráfica más suave
print("x=", x)
F_plot = funcion(x)  # Calcula la función en los puntos
print("F=", F_plot)



plt.axis("equal")
plt.plot(x, F_plot, label='F(x) = e^x - 2x', color='blue')
plt.title("Gráfica de la función")
plt.xlabel("x")
plt.ylabel("F(x)")
plt.legend()
plt.grid()
plt.show()
