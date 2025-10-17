
# ===================================================================
#               Matrices de transferencia de rayos
# ===================================================================

import numpy as np
import matplotlib.pyplot as plt

# ===================================================================
#       Definicion de matrices de difernetes interacciones
# ===================================================================

def matriz_refraccion_curvas(n1, n2, R):
    
    #   Calcula la matriz de transferencia de rayos para una refraccion
    #    n1: indice de refraccion del medio incidente
    #    n2: indice de refraccion del medio transmitido
    #    R: radio de curvatura de la superficie reflectante
    
    a = (n1 - n2) / (n2 * R)  
    b =  n1 / n2
    Ra_curve = np.array([[1, 0], [a, b]])

    return Ra_curve    

def matriz_propagacion(d):
    #   Calcula la matriz de transferencia de rayos para una propagacion
    #    d: distancia de propagacion
    
    Prop = np.array([[1, d], [0, 1]])
    return Prop

def matriz_refraccion(n1, n2):
    
    #   Calcula la matriz de transferencia de rayos para una refraccion
    #    n1: indice de refraccion del medio incidente
    #    n2: indice de refraccion del medio transmitido
    
    a =  n1 / n2
    Ra = np.array([[1, 0], [0, a]])

    return Ra   

def matriz_reflexion_curvas(R):
    
    #   Calcula la matriz de transferencia de rayos para una reflexion
    #    R: radio de curvatura de la superficie reflectante
    
    a = 2 / R
    Re_curve = np.array([[1, 0], [a, -1]])

    return Re_curve   

def lente_delgada(f):
    
    #   Calcula la matriz de transferencia de rayos para una lente delgada
    #    f: distancia focal de la lente
    
    a = -1 / f
    Lente = np.array([[1, 0], [a, 1]])

    return Lente

def matriz_del_sistema(matrices):
    
    #   Calcula la matriz de transferencia de rayos total del sistema
    #    matrices: lista de matrices de transferencia de rayos de cada elemento del sistema
    
    M_total = np.eye(2)  # Matriz identidad 2x2
    for M in matrices:
        M_total = np.dot(M, M_total)  # Multiplicacion de matrices en orden

    return M_total

# ===================================================================
#                Calculo de la trayectoria 01
# ===================================================================

#Toca dividir la trayectoria en partes de S
# Datos del sistema todos en [mm]

f = 500     # Focal de la lente
D_L1 = 100  # Diametro de la lente L1
l = 50      # Grosor BS

M_prop1 = matriz_propagacion(f)                    # Propagacion hasta L1
Lente_1 = lente_delgada(f)                         # Pasa por L1
M_prop2 = matriz_propagacion(l)                    # Propagacion hasta M1
M_reflexion_M1 = matriz_reflexion_curvas(np.inf)   # Reflexion en M1
M_prop3 = matriz_propagacion(l)                    # Propagacion hasta L1
Lente_2 = lente_delgada(f)                         # Pasa por L1
M_prop4 = matriz_propagacion(f)                    # Propagacion hasta el plano imagen CAM1

trayectoria_01 = [M_prop4, Lente_2, M_prop3, M_reflexion_M1, M_prop2, Lente_1, M_prop1]
M_total_01 = matriz_del_sistema(trayectoria_01)     # Matriz total del sistema trayectoria 01
#print("Matriz total del sistema trayectoria 01:\n", M_total_01)

# ===================================================================
#                Añadir efectos difractivos
# ===================================================================

A = M_total_01[0,0]
B = M_total_01[0,1]
C = M_total_01[1,0]
D = M_total_01[1,1]
lam = 0.000633  # Longitud de onda en mm (633 nm)
k = 2 * np.pi / lam  # Numero de onda

def funcion_transferencia(A,B,C,D,y1,y2,lo):
    #   Calcula la funcion de transferencia del sistema optico
    #    A,B,C,D: elementos de la matriz de transferencia de rayos del sistema optico

    h = np.exp(1j*k*lo) * np.exp((1j*k/ (2*B)) * (A*y1**2 - 2*y1*y2 + D*y2**2))

    return h

def transmitancia_entrada(y0):
    
    # Parámetros de la simulación
    N = 1024              # Número de muestras (píxeles) en cada dimensión
    L = 10.0              # Tamaño físico de la rejilla de simulación en mm

    # Distancia de propagación
    z = 200.0             # Distancia del plano objeto al plano de observación en mm
    
    # Espaciado de muestreo en el plano objeto (ξ, η)
    d_xi = L / N
    xi_vec = np.arange(-N/2, N/2) * d_xi
    xi, eta = np.meshgrid(xi_vec, xi_vec)

    # Espaciado de muestreo en el plano de frecuencias (fx, fy)
    dfx = 1 / L
    fx_vec = np.arange(-N/2, N/2) * dfx
    fx, fy = np.meshgrid(fx_vec, fx_vec)

    # --- Ejemplo: Una apertura circular de radio R ---
    R = 1.0  # Radio de la apertura en mm

    # Inicializamos la matriz S como una matriz de ceros complejos
    S = np.zeros((N, N), dtype=complex)

    # Definimos la apertura: transmitancia 1 dentro del círculo, 0 fuera
    aperture_condition = (xi**2 + eta**2) < R**2
    S[aperture_condition] = 1 + 0j
    
    # La onda plana incidente con incidencia normal tiene amplitud 1 y fase 0
    # En este caso, el campo justo después del objeto es igual a su transmitancia.
    U_out_obj = S

    return U_out_obj
