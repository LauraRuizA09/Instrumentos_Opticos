# ===================================================================
#                    Funciones a utilizar
# ===================================================================

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
from PIL import Image

# ===================================================================
#               Definicion Objeto de entrada del sistema
# ===================================================================

#------------------------------MUESTREO------------------------------

#----- Horizontal-------
Nx = 1080 # Número de muestras (píxeles)
Lx = 10 # Tamaño físico de la ventana (mm)
dx = Lx / Nx # Paso espacial Δx
dfx = 1 / Lx # Paso en frecuencia Δfx

#----- Vertical-------
Ny = 1080 # Número de muestras (píxeles)
Ly = 10 # Tamaño físico de la ventana (mm)
dy = Ly / Ny # Paso espacial Δy
dfy = 1 / Ly # Paso en frecuencia Δfy

# ===================================================================
#                  Coordenadas espaciales (ξ, η)
# ===================================================================

# Usamos xi (ξ) y eta (η) para definir S(ξ,η)
n = np.arange(Nx) - Nx//2 # Contadores centrados
m = np.arange(Ny) - Ny//2
xi_vec = n * dx 
eta_vec = m * dy
xi, eta = np.meshgrid(xi_vec, eta_vec) 

# ===================================================================
#                  Coordenadas de frecuencia (fx, fy)
# ===================================================================

p = np.arange(Nx) - Nx//2 # Contadores centrados
q = np.arange(Ny) - Ny//2
fx_vec = p * dfx 
fy_vec = q * dfy
fx, fy = np.meshgrid(fx_vec, fy_vec) 

# ===================================================================
#              Definición de la Transmitancia del Objeto
# ===================================================================

def transmitancia_entrada(tipo_de_objeto):

 if tipo_de_objeto == 'rectangular':
  
  # Rendija rectangular de ancho a y altura b
    a = 2 # Ancho de la rendija en mm
    b = 1 # Altura de la rendija en mm

    aperture = (abs(xi) <= a/2)*(abs(eta) <= b/2)
    U0 = aperture.astype(np.complex128) #Como la funcion anterior retorna valores booleanos debemos pasarla a valores complejos para poder tener términos de fase.

 elif tipo_de_objeto == 'circular':

    # Apertura circular de radio r
    r = 1 # Radio de la apertura en mm
    aperture = (xi**2 + eta**2 <= r**2)
    U0 = aperture.astype(np.complex128)

 elif tipo_de_objeto == 'imagen':

    # Cargar una imagen en escala de grises y normalizarla
    imagen = Image.open('Practicas/Practica_03/Punto_02/Imagenes Test USAF/T-20-final-rev-1-400x400.jpg').convert('L')
    imagen = imagen.resize((Nx, Ny)) # Redimensionar la imagen al tamaño Nx x Ny
    U0 = np.array(imagen) / 255.0 # Normalizar a [0, 1]
    U0 = U0.astype(np.complex128) # Convertir a tipo complejo para incluir fase si es necesario

 else:
  raise ValueError("Tipo de objeto no reconocido. Use 'rectangular', 'circular' o 'imagen'.")

 return U0, (Lx,Ly)



# ===================================================================
#         Definicion de matrices de diferentes interacciones
# ===================================================================

def matriz_refraccion_curvas(n1, n2, R):
 
 # Calcula la matriz de transferencia de rayos para una refraccion
 # n1: indice de refraccion del medio incidente
 # n2: indice de refraccion del medio transmitido
 # R: radio de curvatura de la superficie reflectante
 
 a = (n1 - n2) / (n2 * R) 
 b = n1 / n2
 Ra_curve = np.array([[1, 0], [a, b]])

 return Ra_curve 

def matriz_propagacion(d):
 # Calcula la matriz de transferencia de rayos para una propagacion
 # d: distancia de propagacion
 
 Prop = np.array([[1, d], [0, 1]])
 return Prop

def matriz_refraccion(n1, n2):
 
 # Calcula la matriz de transferencia de rayos para una refraccion
 # n1: indice de refraccion del medio incidente
 # n2: indice de refraccion del medio transmitido
 
 a = n1 / n2
 Ra = np.array([[1, 0], [0, a]])

 return Ra 

def matriz_reflexion_curvas(R):
 
 # Calcula la matriz de transferencia de rayos para una reflexion
 # R: radio de curvatura de la superficie reflectante
 
 a = 2 / R
 Re_curve = np.array([[1, 0], [a, 1]])

 return Re_curve 

def lente_delgada(f):
 
 # Calcula la matriz de transferencia de rayos para una lente delgada
 # f: distancia focal de la lente
 
 a = -1 / f
 Lente = np.array([[1, 0], [a, 1]])

 return Lente

def matriz_del_sistema(matrices):
 
 # Calcula la matriz de transferencia de rayos total del sistema
 # matrices: lista de matrices de transferencia de rayos de cada elemento del sistema
 
 M_total = np.eye(2) # Matriz identidad 2x2

 #for M in reversed(matrices):
  #M_total = np.dot(M_total, M) # Multiplicacion de matrices en orden
 for M in matrices:
  M_total = M_total @ M  # Multiplicacion de matrices en orden

 return M_total


# ===================================================================
#         Función de Propagación de Campo Complejo (ABCD)
# ===================================================================

def propagar_campo_ABCD(U_in, L_in_mm, lam_mm, M_system, Equal, L_diferente):

    # Extraer parámetros del sistema y del campo
    A = M_system[0, 0]
    B = M_system[0, 1]
    C = M_system[1, 0]
    D = M_system[1, 1]
    
      
    # Si B=0, es un sistema de imagen pura, no de difracción.
    if np.isclose(B, 0):

        print(f"Advertencia: B=0 (Sistema de Imagen). Magnificación = {A:.2f}")

        Ny, Nx = U_in.shape
        Lx, Ly = L_in_mm
        
        L_out_mm = (Lx * np.abs(A), Ly * np.abs(A))
        dx_out, dy_out = L_out_mm[0] / Nx, L_out_mm[1] / Ny

        x_vec = (np.arange(Nx) - Nx // 2) * dx_out
        y_vec = (np.arange(Ny) - Ny // 2) * dy_out

        x_out, y_out = np.meshgrid(x_vec, y_vec, indexing='xy')

        return U_in, L_out_mm, (x_out, y_out)

    # --- Propagación por FFT (B != 0) ---
    Ny, Nx = U_in.shape
    Lx, Ly = L_in_mm
    Lx_out, Ly_out = L_diferente
    k = 2 * np.pi / lam_mm

    if Equal == "NO":
      
      Lx_out = Lx
      Ly_out = Ly

    # Coordenadas de entrada (Plano 1: y1)
    dx1 = Lx / Nx
    dy1 = Ly / Ny
    x1_vec = (np.arange(Nx) - Nx // 2) * dx1
    y1_vec = (np.arange(Ny) - Ny // 2) * dy1
    x1_mesh, y1_mesh = np.meshgrid(x1_vec, y1_vec, indexing='xy')

    # --- 1. "Fase parabólica plano apertura" ---
    phase_in = (k * A / (2 * B)) * (x1_mesh**2 + y1_mesh**2)
    U_mid1 = U_in * np.exp(1j * phase_in)

    # --- 2. "Kernel Fourier" (Implementado con FFT) ---
    # Usamos fftshift/ifftshift para centrar el origen
    U_mid2 = fftshift(fft2(ifftshift(U_mid1))) * dx1 * dy1

    # --- 3. Coordenadas de salida (Plano 2: y2) ---
    # Relación de escalado: y2 = lambda * B * fy
    dfx = 1 / Lx_out
    dfy = 1 / Ly_out
    fx_vec = (np.arange(Nx) - Nx // 2) * dfx
    fy_vec = (np.arange(Ny) - Ny // 2) * dfy
    
    x2_vec = lam_mm * B * fx_vec
    y2_vec = lam_mm * B * fy_vec
    
    # Tamaño físico de la ventana de salida
    Lx_out = np.abs(x2_vec[-1] - x2_vec[0]) + np.abs(x2_vec[1] - x2_vec[0])
    Ly_out = np.abs(y2_vec[-1] - y2_vec[0]) + np.abs(y2_vec[1] - y2_vec[0])
    L_out_mm = (Lx_out, Ly_out)
    
    x2_mesh, y2_mesh = np.meshgrid(x2_vec, y2_vec, indexing='xy')

    # --- 4. "Fase parabólica plano observación" ---
    phase_out = (k * D / (2 * B)) * (x2_mesh**2 + y2_mesh**2)
    
    # --- 5. Prefactor (El 1/(i*lambda*B)) ---
    pre_factor = 1 / (1j * lam_mm * B)
    
    U_out = pre_factor * U_mid2 * np.exp(1j * phase_out)
    
    return U_out, L_out_mm, (x2_mesh, y2_mesh)

# ===================================================================
#                       Definimos la pupila
# ===================================================================

def plano_pupila(lamda,Sen,F_MO, Le, Ln):

    Ne=np.shape(Sen)[1]
    Nn=np.shape(Sen)[0]

    dx=lamda*F_MO/(Le)
    dy=lamda*F_MO/(Ln)

    x=np.arange(-Ne/2,Ne/2)*dx
    y=np.arange(-Nn/2,Nn/2)*dy

    X,Y=np.meshgrid(x,y)

    return(X,Y)