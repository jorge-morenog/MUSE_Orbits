# Importamos los paquetes que queramos
import Pkg 
Pkg.add("Plots")
Pkg.add("OffsetArrays")
# Pkg.add("Latexify")
Pkg.add("LaTeXStrings")

# Decimos cuáles de esos paquetes vamos a usar
using Plots
using OffsetArrays: Origin
# using Latexify
using LaTeXStrings

#####################################################################
#                            FUNCIONES                              #
#####################################################################

##############################################
# FUNCIÓN DEL PROBLEMA DE KEPLER
##############################
function Kepler(U,t)
    x, y, x_dot, y_dot = U[1], U[2], U[3], U[4]

    F1 = x_dot
    F2 = y_dot
    F3 = -x /(x^2+y^2)^(3/2)
    F4 = -y /(x^2+y^2)^(3/2)

    return [F1, F2, F3, F4]
end

##############################################
# FUNCIONES DE LOS ESQUEMAS NUMÉRICOS
##############################

# Euler explícito 
function Euler(U, t, dt, F)  # U es un vector, t es un instante de tiempo 
    return U + dt * F(U, t)
end
# Adams-Bashforth de 2 pasos
function AB2(U2, U1, t, dt, F)

    return U2  + (dt/2)*( 3*F(U2,t) - F(U1,t-dt) )
end

# Runge-Kutta de 2 etapas
function RK2(U, t, dt, F)

    k1 = F( U         , t      )
    k2 = F( U + k1*dt , t + dt )

    return U  + (dt/2)*(k1 + k2)
end

# Runge-Kutta de 4 etapas
function RK4(U, t, dt, F)

    k1 = F( U           , t       )
    k2 = F( U + k1*dt/2 , t + dt/2)
    k3 = F( U + k2*dt/2 , t + dt/2)
    k4 = F( U + k3*dt   , t + dt  )

    return U  + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
end

##############################################
# FUNCIÓN PARA EL PROBLEMA DE CAUCHY
##############################
function Cauchy(F, t, U0, Esquema)

    N = length(t) - 1
    U = zeros( length(t), length(U0) )  # length(t) filas -> índices desde el 0 hasta el length(t)-1 // length(U0) columnas -> índices desde el 0 hasta length(U0)-1 
    U = Origin(0, 1)(U)              # (a,b) -> a = primer índice para las filas, b = primer índice para las columnas
    U[0,:] = U0

    if Esquema == AB2  # el valor de U1, necesario para U2, se calcula con Euler explícito (solo ese valor)
        for n in 0:N-2   # empieza en 0 y acaba en (N)-1 = N-1 -> n toma N valores ( desde 0 hasta N-1 )
            if n==0
                U[n+1,:] = Euler(U[n,:], t[n], t[n+1]-t[n], F)
                U[n+2,:] = Esquema( U[n+1,:], U[n,:], t[n+1], t[n+2]-t[n+1], F )
            else
                U[n+2,:] = Esquema( U[n+1,:], U[n,:], t[n+1], t[n+2]-t[n+1], F )
            end
        end
    else 
        for n in 0:N-1   # empieza en 0 y acaba en (N)-1 = N-1 -> n toma N valores ( desde 0 hasta N-1 )
            U[n+1,:] = Esquema( U[n,:], t[n], t[n+1]-t[n], F )
        end
    end

    return U
end

#####################################################################
#                               DATOS                               #
#####################################################################

# Instante inicial
t0 = 0.
# Instante final
tf = 20.
# Número de intervalos temporales
N  = 200
# Condiciones iniciales
x0  = 1
y0  = 0
vx0 = 0
vy0 = 1

#####################################################################
#                             CÓDIGO                                #
#####################################################################

# vector solución en el instante inicial (ccii)
U0 = [x0,y0,vx0,vy0]

# Generamos espacio para el vector de tiempos (N intervalos = N+1 instantes de tiempo)
t = zeros(N+1)  

# Creamos el vector de instantes temporales
t = Vector( range(t0, tf, N+1) )             # range de Julia = linspace de Python (ptos equeispaciados)
# Paso temporal
dt = (tf-t0)/N

# Decimos qué valor queremos como primer índice
t = Origin(0)(t)

# Solución con esquema Euler explícito
U_euler = Cauchy( Kepler, t, U0, Euler )
# Solución con esquema Adams-Bashforth de 2 pasos
U_ab2   = Cauchy( Kepler, t, U0, AB2   )
# Solución con esquema Runge-Kutta de 2 etapas
U_rk2   = Cauchy( Kepler, t, U0, RK2   )
# Solución con esquema Runge-Kutta de 4 etapas
U_rk4   = Cauchy( Kepler, t, U0, RK4   )

#####################################################################
#                             GRÁFICAS                              #
#####################################################################

# Estilos de lineas: :solid, :dash, :dot  , :dashdot, :dashdotdot
# Colores de lineas: :red  , :blue, :green, :yellow , :purple    , :orange, :pink, :brown, :black, :gray

# Plots de datos Euler y RK4
plot( U_euler[:, 1], U_euler[:, 2], label="Euler explícito"    , line = (:solid, :blue ), lw=1.0)
plot!(U_ab2[:, 1]  , U_ab2[:, 2]  , label="Adams-Bashforth 2"  , line = (:dash , :black), lw=1.0)
plot!(U_rk2[:, 1]  , U_rk2[:, 2]  , label="Runge-Kutta 2"      , line = (:dash , :green), lw=1.0)
plot!(U_rk4[:, 1]  , U_rk4[:, 2]  , label="Runge-Kutta 4"      , line = (:solid, :red  ), lw=1.0)

# Título y etiquetas
title!("Órbita de Kepler (Δt = $dt)")
xlabel!("x")
ylabel!("y")
# Tamaño de la gráfica en pantalla
plot!(size=(700, 600))

# Opcional: Activando esta línea desaparece la leyenda 
# legend!(false)

# Aspecto de la gráfica (igualar la relación de aspecto)
plot!(aspect_ratio=:equal)
