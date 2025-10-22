
# ===================================================================
#               Matrices de transferencia de rayos
# ===================================================================

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ===================================================================
#       Definicion de matrices de diferentes interacciones
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

    for M in reversed(matrices):
        M_total = np.dot(M_total, M)  # Multiplicacion de matrices en orden

    return M_total

# ===================================================================
#               Calculo de la trayectoria 01 parte 1
# ===================================================================

#Toca dividir la trayectoria en partes de S
# Datos del sistema todos en [mm]

f = 500     # Focal de la lente
D_L1 = 100  # Diametro de la lente L1
l = 50      # Grosor BS

M_prop1 = matriz_propagacion(f)                    # Propagacion hasta L1
Lente_1 = lente_delgada(f)                         # Pasa por L1
M_prop2 = matriz_propagacion(f)                    # Propagacion hasta M1

trayectoria_pt_01 = [M_prop1, Lente_1, M_prop2]
M_total_01 = matriz_del_sistema(trayectoria_pt_01)     # Matriz total del sistema trayectoria 01

#Aproximar a cero los valores muy pequeños
threshold = 1e-15 
M_total_01[np.abs(M_total_01) < threshold] = 0.0

print("Matriz total del sistema trayectoria 01:\n", M_total_01)

# ===================================================================
#              Calculo de la trayectoria 01 parte 2
# ===================================================================

#Toca dividir la trayectoria en partes de S
# Datos del sistema todos en [mm]

f = 500     # Focal de la lente
D_L1 = 100  # Diametro de la lente L1
l = 50      # Grosor BS

M_reflexion_M1 = matriz_reflexion_curvas(np.inf)   # Reflexion en M1
M_prop3 = matriz_propagacion(f)                    # Propagacion hasta L1
Lente_2 = lente_delgada(f)                         # Pasa por L1
M_prop4 = matriz_propagacion(f)                    # Propagacion hasta el plano imagen CAM1

trayectoria_pt_02 = [M_reflexion_M1, M_prop3, Lente_2, M_prop4]
M_total_02 = matriz_del_sistema(trayectoria_pt_02)     # Matriz total del sistema trayectoria 01

#Aproximar a cero los valores muy pequeños
threshold = 1e-15 
M_total_02[np.abs(M_total_02) < threshold] = 0.0

print("Matriz total del sistema trayectoria 02:\n", M_total_02)

# ===================================================================
#                Añadir efectos difractivos
# ===================================================================

# Matriz total trayectoria 01
A_1 = M_total_01[0,0]
B_1 = M_total_01[0,1]
C_1 = M_total_01[1,0]
D_1 = M_total_01[1,1]

# Matriz total trayectoria 02
A_2 = M_total_02[0,0]
B_2 = M_total_02[0,1]
C_2 = M_total_02[1,0]
D_2 = M_total_02[1,1]

lam = 0.000633  # Longitud de onda en mm (633 nm)
k = 2 * np.pi / lam  # Numero de onda


#-----Muestreo Horizontal-------
Nx = 1024              # Número de muestras (píxeles)
Lx = 10                # Tamaño físico de la ventana (mm)
dx = Lx / Nx           # Paso espacial Δx
dfx = 1 / Lx           # Paso en frecuencia Δfx

#-----Muestreo Vertical-------
Ny = 1024              # Número de muestras (píxeles)
Ly = 10                # Tamaño físico de la ventana (mm)
dy = Ly / Ny           # Paso espacial Δy
dfy = 1 / Ly           # Paso en frecuencia Δfy

# ---------- Coordenadas espaciales (ξ, η) ----------
#     Usamos xi (ξ) y eta (η) para definir S(ξ,η)

n = np.arange(Nx) - Nx//2      # Contadores centrados
m = np.arange(Ny) - Ny//2
xi_vec = n * dx                
eta_vec = m * dy
xi, eta = np.meshgrid(xi_vec, eta_vec) 

# ---------- Coordenadas de frecuencia (fx, fy) ----------
p = np.arange(Nx) - Nx//2      # Contadores centrados
q = np.arange(Ny) - Ny//2
fx_vec = p * dfx               
fy_vec = q * dfy
fx, fy = np.meshgrid(fx_vec, fy_vec) 

def transmitancia_entrada(tipo_de_objeto):

    # ===================================================================
    #                Definición de la Transmitancia del Objeto
    # ===================================================================

    if tipo_de_objeto == 'rectangular':
        
        # Rendija rectangular de ancho a y altura b
        a = 2  # Ancho de la rendija en mm
        b = 1  # Altura de la rendija en mm

        aperture = (abs(xi) <= a/2)*(abs(eta) <= b/2)
        U0 = aperture.astype(np.complex128) #Como la funcion anterior retorna valores booleanos debemos pasarla a valores
                                            #complejos para poder tener términos de fase.
    elif tipo_de_objeto ==  'circular':
        # Apertura circular de radio r
        r = 1  # Radio de la apertura en mm

        aperture = (xi**2 + eta**2 <= r**2)
        U0 = aperture.astype(np.complex128)
    
    elif tipo_de_objeto == 'imagen':
        # Cargar una imagen en escala de grises y normalizarla
        imagen = Image.open('Practicas/Practica_02/Actividad_1/Noise images/Noise (18).png').convert('L')
        imagen = imagen.resize((Nx, Ny))  # Redimensionar la imagen al tamaño Nx x Ny
        U0 = np.array(imagen) / 255.0  # Normalizar a [0, 1]
        U0 = U0.astype(np.complex128)  # Convertir a tipo complejo para incluir fase si es necesario

    else:
        raise ValueError("Tipo de objeto no reconocido. Use 'rectangular', 'circular' o 'imagen'.")

    return U0

# Obtener el campo de entrada

campo_entrada_1 = transmitancia_entrada('imagen')   #  S(ξ,η)

def propagar_ABCD(U1, A, B, C, D, lam, dx1, dy1):

    Ny, Nx = U1.shape
    k = 2 * np.pi / lam
    Lx = Nx * dx1
    Ly = Ny * dy1

    # --- Coordenadas de entrada ---
    n = np.arange(Nx) - Nx//2
    m = np.arange(Ny) - Ny//2
    xi_vec = n * dx1
    eta_vec = m * dy1
    xi_mesh, eta_mesh = np.meshgrid(xi_vec, eta_vec, indexing='ij')

    # 1. Fase cuadrática de entrada (dependiente de A)
    phase1 = (k / (2 * B)) * A * (xi_mesh**2 + eta_mesh**2)
    U_intermediate1 = U1 * np.exp(1j * phase1)

    # 2. Transformada de Fourier vía FFT
    U_shifted = np.fft.ifftshift(U_intermediate1)
    U_fft_unscaled = np.fft.fft2(U_shifted)
    U_intermediate2 = np.fft.fftshift(U_fft_unscaled)

    # 3. Coordenadas del plano de salida (x, y)
    dfx = 1 / Lx
    dfy = 1 / Ly
    fx_vec = (np.arange(Nx) - Nx//2) * dfx
    fy_vec = (np.arange(Ny) - Ny//2) * dfy
    x_vec = fx_vec * lam * B
    y_vec = fy_vec * lam * B
    dx2 = np.abs(x_vec[1] - x_vec[0]) # Paso de muestreo en salida
    dy2 = np.abs(y_vec[1] - y_vec[0])
    x_mesh, y_mesh = np.meshgrid(x_vec, y_vec, indexing='ij')

    # 4. Fase cuadrática de salida (dependiente de D)
    phase2 = (k / (2 * B)) * D * (x_mesh**2 + y_mesh**2)
    U_intermediate3 = U_intermediate2 * np.exp(1j * phase2)

    # 5. Factores globales de escala y fase
    # Omitimos exp(ikL0) por ser fase global constante
    pre_factor = (dx1 * dy1) / (1j * lam * B)
    U2 = pre_factor * U_intermediate3

    return U2, x_mesh, y_mesh, dx2, dy2


S1_campo, S1_x_mesh, S1_y_mesh, S1_dx, S1_dy = propagar_ABCD(campo_entrada_1, A_1, B_1, C_1, D_1, lam, dx, dy)
# se debe multiplicar por la transmitaancia t(x,y) pero en este caso es 1

#Definir los campos de entrada para la segund parte de la trayectoria 01

campo_entrada_2 =  S1_campo 

S2_campo, S2_x_mesh, S2_y_mesh, S2_dx, S2_dy = propagar_ABCD(campo_entrada_2, A_2, B_2, C_2, D_2, lam, dx, dy)



# ===================================================================
#               GRAFICAR CAMPO DE ENTRADA Y SALIDA 
# ===================================================================

fig, axes = plt.subplots(1, 3, figsize=(21, 6)) # Ajusta figsize si es necesario (más ancho ahora)

# --- Graficar del campo de entrada---
intensity = np.abs(campo_entrada_1)**2
extent_in = [-Lx/2, Lx/2, -Ly/2, Ly/2] # Renombrado para claridad
im_int = axes[0].imshow(intensity, cmap='gray', extent=extent_in, origin='upper', aspect='equal')
fig.colorbar(im_int, ax=axes[0], label='Intensidad Transmitida |U0|^2', shrink=0.8)
axes[0].set_xlabel('ξ (mm)')
axes[0].set_ylabel('η (mm)')
axes[0].set_title('Campo de Entrada S(ξ,η)')
axes[0].grid(False)

# --- Graficar Intensidad de S1_campo---
intensity_S1 = np.abs(S1_campo)**2

# Límites del espejo M1 
ancho_m1_mm = 10.4
alto_m1_mm = 5.8

# Usas las dimensiones de la cámara para definir el extent directamente
extent_cam1 = [-ancho_m1_mm/2, ancho_m1_mm/2, -alto_m1_mm/2, alto_m1_mm/2]

# Calcular el extent correcto para S1_campo usando sus coordenadas
extent_S1 = [S1_x_mesh.min(), S1_x_mesh.max(), S1_y_mesh.min(), S1_y_mesh.max()]
im_s1 = axes[1].imshow(intensity_S1, cmap='viridis', extent=extent_in, origin='lower', aspect='equal')
fig.colorbar(im_s1, ax=axes[1], label='Intensidad |S1|^2', shrink=0.8)
axes[1].set_xlabel('x (mm)') # Coordenadas del plano M1
axes[1].set_ylabel('y (mm)')
axes[1].set_title('Campo en M1 (Trayectoria 1)') # Título más descriptivo
axes[1].grid(False)

# --- Graficar Intensidad de S2_campo ---
intensity_S2 = np.abs(S2_campo)**2 # Renombrado para claridad

# Límites de la cámara CAM1
ancho_cam1_mm = 4640 * 3.8e-3 # Ancho físico en mm
alto_cam1_mm = 3506 * 3.8e-3  # Alto físico en mm

# Usas las dimensiones de la cámara para definir el extent directamente
extent_cam1 = [-ancho_cam1_mm/2, ancho_cam1_mm/2, -alto_cam1_mm/2, alto_cam1_mm/2]

im_s2 = axes[2].imshow(intensity_S2, cmap='viridis', extent=extent_cam1, origin='lower', aspect='equal') # Usando extent_cam1
fig.colorbar(im_s2, ax=axes[2], label='Intensidad |S2|^2', shrink=0.8) # Cambiado a intensity_S2, im_s2
axes[2].set_xlabel('u (mm)') # Coordenadas del plano Cam1
axes[2].set_ylabel('v (mm)')
axes[2].set_title('Campo en Cam1 (Trayectoria 2)') # Título más descriptivo
axes[2].grid(False)


# --- Ajustar espaciado y mostrar la figura completa ---
plt.tight_layout() # Ajusta el espaciado para evitar superposiciones
plt.show() # Muestra la figura con los tres subplots
