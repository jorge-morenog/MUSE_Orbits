from numpy import array, zeros, zeros_like
from scipy.linalg import eig

def system_matrix(F, U0, t, eps=1e-6):
        """
        Calcula la matriz linealizada del operador F(U, t) en el punto U0.

        Parameters:
            F (callable): Operador diferencial (función que toma U y t como entrada).
            U0 (array-like): Punto de evaluación (vector).
            t (float): Tiempo de evaluación.
            eps (float): Perturbación para aproximación de diferencias finitas.

        Returns:
            A (2D numpy array): Matriz linealizada del operador.
        """
        U0  = array(U0)
        N   = len(U0)  # Dimensión del sistema
        A   = zeros((N, N))  # Inicialización de la matriz resultante

        for j in range(N):
            delta       = zeros_like(U0)    # Vector perturbación inicializado en cero
            delta[j]    = eps               # Perturba solo la componente j-ésima

            # Aproximación por diferencias finitas centrales
            A[:, j] = (F(U0 + delta, t) - F(U0 - delta, t)) / (2 * eps)

        return A
    
def eigenvalues_jacobian(differential_operator, U0, t0):
    """
    Calcula los valores propios de la matriz jacobiana de un sistema ODE.
    
    Parameters:
        differential_operator (callable): Operador diferencial del sistema (F).
        U0 (array-like): Estado inicial.
        t0 (float): Tiempo inicial.
    
    Returns:
        lambda_ (1D array): Valores propios del jacobiano.
    """
    U0 = array(U0)
    N = len(U0)  # Dimensión del sistema

    # Calcula la matriz jacobiana usando una función auxiliar
    A = system_matrix(differential_operator, U0, t0)
    
    # Calcula los valores propios de A
    lambda_ = eig(A, left=False, right=False)
    
    return lambda_