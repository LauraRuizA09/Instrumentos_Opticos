# ===================================================================
#               Matrices de transferencia de rayos
# ===================================================================

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.widgets import Slider
from scipy.signal import find_peaks 

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
 Re_curve = np.array([[1, 0], [a, -1]])

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

 for M in reversed(matrices):
  M_total = np.dot(M_total, M) # Multiplicacion de matrices en orden
 return M_total

# ===================================================================
#           Calculo de la trayectoria 01 parte 1
# ===================================================================

#Toca dividir la trayectoria en partes
# Datos del sistema todos en [mm]

f = 500 # Focal de la lente
D_L1 = 100 # Diametro de la lente L1
l = 50 # Grosor BS

M_prop1 = matriz_propagacion(f) # Propagacion hasta L1
Lente_1 = lente_delgada(f) # Pasa por L1
M_prop2 = matriz_propagacion(f) # Propagacion hasta M1

trayectoria_pt_01 = [M_prop1, Lente_1, M_prop2]
M_total_01 = matriz_del_sistema(trayectoria_pt_01) # Matriz total del sistema trayectoria 01 pt 1

#Aproximar a cero los valores muy pequeños
threshold = 1e-15 
M_total_01[np.abs(M_total_01) < threshold] = 0.0

print("Matriz total del sistema trayectoria 01:\n", M_total_01)

# ===================================================================
#               Calculo de la trayectoria 01 parte 2
# ===================================================================

M_reflexion_M1 = matriz_reflexion_curvas(np.inf) # Reflexion en M1
M_prop3 = matriz_propagacion(f) # Propagacion hasta L1
Lente_2 = lente_delgada(f) # Pasa por L1
M_prop4 = matriz_propagacion(f) # Propagacion hasta el plano imagen CAM1

trayectoria_pt_02 = [M_reflexion_M1, M_prop3, Lente_2, M_prop4]
M_total_02 = matriz_del_sistema(trayectoria_pt_02) # Matriz total del sistema trayectoria 01 pt 2

#Aproximar a cero los valores muy pequeños
threshold = 1e-15 
M_total_02[np.abs(M_total_02) < threshold] = 0.0

print("Matriz total del sistema trayectoria 02:\n", M_total_02)

# ===================================================================
#                   Calculo de la trayectoria 02
# ===================================================================

d = -f             # analizando para que en CAM2 se forme la TF de la imagen se requieren estas distancias
w = f/2

M_prop5 = matriz_propagacion(d)
M_reflexion_M2 = matriz_reflexion_curvas(np.inf) # Reflexion en M2
M_prop3 = matriz_propagacion(w) # Propagacion hasta L2
Lente_3 = lente_delgada(f) # Pasa por L2
M_prop4 = matriz_propagacion(f) # Propagacion hasta el plano imagen CAM2

trayectoria_pt_03 = [M_reflexion_M1, M_prop3, Lente_2, M_prop4]
M_total_03 = matriz_del_sistema(trayectoria_pt_02) # Matriz total del sistema trayectoria 02

#Aproximar a cero los valores muy pequeños
threshold = 1e-15 
M_total_03[np.abs(M_total_02) < threshold] = 0.0

print("Matriz total del sistema trayectoria 03:\n", M_total_03)

# ===================================================================
#                   Añadir efectos difractivos
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

# Matriz total trayectoria 03
A_3 = M_total_03[0,0]
B_3 = M_total_03[0,1]
C_3 = M_total_03[1,0]
D_3 = M_total_03[1,1]

lam = 0.000633 # Longitud de onda en mm (633 nm)
k = 2 * np.pi / lam # Numero de onda


#-----Muestreo Horizontal-------
Nx = 1024 # Número de muestras (píxeles)
Lx = 10 # Tamaño físico de la ventana (mm)
dx = Lx / Nx # Paso espacial Δx
dfx = 1 / Lx # Paso en frecuencia Δfx

#-----Muestreo Vertical-------
Ny = 1024 # Número de muestras (píxeles)
Ly = 10 # Tamaño físico de la ventana (mm)
dy = Ly / Ny # Paso espacial Δy
dfy = 1 / Ly # Paso en frecuencia Δfy

# ---------- Coordenadas espaciales (ξ, η) ----------
# Usamos xi (ξ) y eta (η) para definir S(ξ,η)

n = np.arange(Nx) - Nx//2 # Contadores centrados
m = np.arange(Ny) - Ny//2
xi_vec = n * dx 
eta_vec = m * dy
xi, eta = np.meshgrid(xi_vec, eta_vec) 

# ---------- Coordenadas de frecuencia (fx, fy) ----------
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
    imagen = Image.open('Practicas/Practica_02/Actividad_1/Noise images/Noise (3).png').convert('L')
    imagen = imagen.resize((Nx, Ny)) # Redimensionar la imagen al tamaño Nx x Ny
    U0 = np.array(imagen) / 255.0 # Normalizar a [0, 1]
    U0 = U0.astype(np.complex128) # Convertir a tipo complejo para incluir fase si es necesario

 else:
  raise ValueError("Tipo de objeto no reconocido. Use 'rectangular', 'circular' o 'imagen'.")

 return U0

# ===================================================================
#              Obtener el campo de entrada
# ===================================================================

campo_entrada_1 = transmitancia_entrada('imagen') # S(ξ,η)

# ===================================================================
#             Definición de la función de propagación ABCD
# ===================================================================


def propagar_ABCD_(U1, A, B, C, D, lam):

 #Caso en el que estamos:
 #salida_ (t(x,y)) = (np.exp(1j*k*f) / 1j*lam*f) * (np.fft.fft2(campo_entrada_1)* dx * dy)
 #salida_t (O(u,v)) = (np.exp(-1j*k*f) / -1j*lam*f) * (np.fft.ifft2(salida_) * (Nx * Ny) * dfx * dfy)

 # --- Coordenadas de entrada ---
 n = np.arange(Nx) - Nx//2
 m = np.arange(Ny) - Ny//2
 xi_vec = n * dx
 eta_vec = m * dy
 xi_mesh, eta_mesh = np.meshgrid(xi_vec, eta_vec, indexing='xy')

 # --- Cálculo de la Integral ---

 # Fase cuadrática de entrada (dependiente de A)
 phase1 = (k / (2 * B)) * A * (xi_mesh**2 + eta_mesh**2)
 U_intermediate1 = U1 * np.exp(1j * phase1)

 if B > 0:
 # El kernel corresponde a una TF
    U_fft_unscaled = np.fft.fft2(U1) * dx * dy
 
 else: # B < 0
 # El kernel corresponde a una TF inversa.
    U_fft_unscaled = np.fft.ifft2(U1) * (Nx * Ny) * dfx * dfy


 # Coordenadas espaciales de salida: x2 = lambda*B*fx, y2 = lambda*B*fy.
 # El signo de B afecta correctamente la escala y posible inversión del eje.
 x_vec = fx_vec * lam * B
 y_vec = fy_vec * lam * B
 # Paso de muestreo en la salida. Usamos abs porque el paso siempre es positivo.
 dx2 = np.abs(x_vec[1] - x_vec[0]) if Nx > 1 else 0
 dy2 = np.abs(y_vec[1] - y_vec[0]) if Ny > 1 else 0
 # Mallas de coordenadas 2D de salida. Usamos 'xy' para consistencia.
 x_mesh, y_mesh = np.meshgrid(x_vec, y_vec, indexing='xy')

 # Fase cuadrática de salida (dependiente de D)
 phase2 = (k / (2 * B)) * D * (x_mesh**2 + y_mesh**2)
 U_intermediate3 = U_fft_unscaled * np.exp(1j * phase2)

 # Factores globales de escala y fase
 # Omitimos exp(ikL0) por ser fase constante.
 # El pre_factor se combina con el fft_scale_factor que depende de si usamos FFT o IFFT.
 pre_factor_integral = 1 / (1j * lam * B)
 U2 = pre_factor_integral * U_intermediate3 

 # Devuelve el campo y las mallas/pasos de salida.
 return U2, x_mesh, y_mesh, dx2, dy2


# ===================================================================
#              Propagación de los campos
# ===================================================================

#Transmitancia del espejo M1
T_xy = 1 

# Calculamos la propagacion de la primera trayectoria donde veremos el campo t(x,y) en el plano del espejo M1
S1_campo, S1_x_mesh, S1_y_mesh, S1_dx, S1_dy = propagar_ABCD_(campo_entrada_1, A_1, B_1, C_1, D_1, lam) * T_xy

#Definir los campos de entrada para la segunda parte de la trayectoria 01 pt 02
campo_entrada_2 = S1_campo 

#Calculamos la propagacion de la segunda trayectoria donde veremos el campo O(u,v) en el plano de la camara CAM1
S2_campo, S2_x_mesh, S2_y_mesh, S2_dx, S2_dy = propagar_ABCD_(campo_entrada_2, A_2, B_2, C_2, D_2, lam)

# Calculamos la propagacion de la segunda trayectoria donde veremos el campo U(x,y) en el plano de la camara CAM2
S3_campo, S3_x_mesh, S3_y_mesh, S3_dx, S3_dy = propagar_ABCD_(campo_entrada_1, A_2, B_2, C_2, D_2, lam)


# ===================================================================
#           Graficar campos de entrada y salida
# ===================================================================

#Graficamos el campo O(u,v) en CAM1 y el campo S(ξ,η) de entrada

fig, axes = plt.subplots(1, 2, figsize=(21, 6)) # Ajusta figsize si es necesario (más ancho ahora)

intensity = np.abs(campo_entrada_1)**2
extent_in = [-Lx/2, Lx/2, -Ly/2, Ly/2]
im_int = axes[0].imshow(intensity, cmap='gray', extent=extent_in, origin='upper', aspect='equal')
fig.colorbar(im_int, ax=axes[0], label='Intensidad Transmitida |U0|^2', shrink=0.8)
axes[0].set_xlabel('ξ (mm)')
axes[0].set_ylabel('η (mm)')
axes[0].set_title('Campo de Entrada $S(ξ,η)$')
axes[0].grid(False)

# --- Intensidad de S1_campo t(x,y) ---
intensity_S1 = np.abs(S1_campo)**2

# Límites del espejo M1 
ancho_m1_mm = 10.4
alto_m1_mm = 5.8

# Usas las dimensiones del espejo para definir el extent directamente
extent_m1 = [-ancho_m1_mm/2, ancho_m1_mm/2, -alto_m1_mm/2, alto_m1_mm/2]
#extent_S1 = [S1_x_mesh.min(), S1_x_mesh.max(), S1_y_mesh.min(), S1_y_mesh.max()]
#im_s1 = axes[1].imshow(intensity_S1, cmap='gray', extent=extent_m1, origin='lower', aspect='equal')
#fig.colorbar(im_s1, ax=axes[1], label='Intensidad |S1|^2', shrink=0.8)
#axes[1].set_xlabel('x (mm)') # Coordenadas del plano M1
#axes[1].set_ylabel('y (mm)')
#axes[1].set_title('Campo en M1 $t(x,y)$') # Título más descriptivo
#axes[1].grid(False)
#axes[1].set_xlim(-0.05,0.05)
#axes[1].set_ylim(-0.05,0.05)

# --- Intensidad de S2_campo O(u,v) ---
intensity_S2 = np.abs(S2_campo)**2

# Límites de la cámara CAM1
ancho_cam1_mm = 4640 * 3.8e-3 # Ancho físico en mm
alto_cam1_mm = 3506 * 3.8e-3 # Alto físico en mm

# Usas las dimensiones de la cámara para definir el extent directamente
extent_cam1 = [-ancho_cam1_mm/2, ancho_cam1_mm/2, -alto_cam1_mm/2, alto_cam1_mm/2]

im_s2 = axes[1].imshow(intensity_S2, cmap='gray', extent=extent_cam1, origin='lower', aspect='equal')
fig.colorbar(im_s2, ax=axes[1], label='Intensidad |S2|^2', shrink=0.8)
axes[1].set_xlabel('u (mm)') # Coordenadas del plano Cam1
axes[1].set_ylabel('v (mm)')
axes[1].set_title('Campo en CAM1 $O(u,v)$') # Título más descriptivo
axes[1].grid(False)


plt.tight_layout() # Ajusta el espaciado para evitar superposiciones
plt.show()


# ===================================================================
#      Eliminar ruido de la imagen en CAM1 (mascara rectangular)
# ===================================================================

# Campo con ruido (resultado de la propagación)
campo_ruidoso = S2_campo # EL que llega a O(u,v)
rows, cols = campo_ruidoso.shape # Obtener dimensiones aquí

# --- Función para aplicar el filtro ---
def filtrar_imagenes(imagen_noisy, cutoff_val):
 
 # Calcular la FFT 2D de la imagen ruidosa para estar en el plano de frecuencias y hacer el filtro
 fft_image = np.fft.fft2(imagen_noisy)
 
 # Crea coordenadas de malla para el filtro
 center_x_local, center_y_local = cols // 2, rows // 2 
 x_mask_coords, y_mask_coords = np.meshgrid(np.arange(cols), np.arange(rows))

 # Crea la máscara rectangular: True (o 1) dentro de los límites, False (o 0) fuera
 lpf_mask_local = (np.abs(x_mask_coords - center_x_local) <= cutoff_val) & \
 (np.abs(y_mask_coords - center_y_local) <= cutoff_val)

 # Aplicar el Filtro en el Dominio de la Frecuencia
 fft_filtered_shifted_local = fft_image * lpf_mask_local

 # Calcula la FFT inversa para obtener la imagen filtrada
 image_filtered_local = np.fft.ifft2(fft_filtered_shifted_local) # Usar fft_filtered_local

 # Toma la magnitud (la IFFT puede tener componentes imaginarias muy pequeñas)
 return np.abs(image_filtered_local), fft_image


# ===================================================================
#   Analisis del espectro de frecuencias para identificar picos
# ===================================================================

#Para aplicar la mascara correctamente debemos identificar cuales son las frecuencias que queremos eliminar
# por ende analizamos el espectro de fourier para ver que tipo de mascara debemos utilizar

filtro, TF = filtrar_imagenes(campo_ruidoso, 450)   #Calculamos la TF del campo ruidoso 

# --- Preparar TF para Graficar ---
TF_shifted = np.fft.fftshift(TF)                # Centrar la TF
# Es importante aclarar que la TF en este caso cumple que es la imagen que se ve en CAM2 para etsoe ntonces 
# utilizamos la progacion de ABCD para obtener la TF de la imagen y a partir de ahi analizamos sus frecuencias
# para poder fiktrar  el ruido correctamente.
TF_shifted = np.fft.fftshift(S3_campo)          # Aca utilizamos la imagen generada en CAM2
TF_magnitude = np.abs(TF_shifted)               # Magnitud de la TF
TF_log_magnitude = np.log1p(TF_magnitude)       # Usar escala logarítmica para mejor visualización (log(1+x) para evitar log(0))

# --- Calcular extensión espacial y de frecuencia ---
# Extensión espacial ya definida como extent_cam1
# Calcular frecuencias para los ejes (usando las dimensiones de salida S2)
dx_out = S2_dx # Paso espacial en la salida
dy_out = S2_dy

# Usaremos las frecuencias originales fx_vec, fy_vec ya que la máscara se define allí
fx = fx_vec 
fy = fy_vec 
extent_freq = [fx.min(), fx.max(), fy.min(), fy.max()] 

# --- Graficar ---
fig_tf, axes_tf = plt.subplots(1, 2, figsize=(15, 6)) 

# --- Graficar del campo de estudio (CAM1, ruidoso) ---
intensity_in = np.abs(campo_ruidoso)**2
im_int = axes_tf[0].imshow(intensity_in, cmap='gray', extent=extent_cam1, origin='lower', aspect='equal') # Usando extent_cam1
fig_tf.colorbar(im_int, ax=axes_tf[0], label='Intensidad $|O(u,v)|^2$', shrink=0.8) # Usando O(u,v)
axes_tf[0].set_xlabel('u (mm)')
axes_tf[0].set_ylabel('v (mm)')
axes_tf[0].set_title('Campo en CAM1 Ruidoso $O(u,v$')
axes_tf[0].grid(False)

# --- Graficar la Magnitud de la Transformada de Fourier (escala log) ---
im_tf_plot = axes_tf[1].imshow(TF_log_magnitude, cmap='viridis', extent=extent_freq, origin='lower', aspect='auto') 
fig_tf.colorbar(im_tf_plot, ax=axes_tf[1], label='Log Magnitud $|\\mathcal{F}\\{O(u,v)\\}|$', shrink=0.8) # Usando O(u,v)
axes_tf[1].set_xlabel('$f_x$ (1/mm)') 
axes_tf[1].set_ylabel('$f_y$ (1/mm)') 
axes_tf[1].set_title('Espectro de Frecuencias del Campo en $CAM1$')
axes_tf[1].grid(False)

plt.tight_layout() 
#plt.savefig('campo_cam1_y_transformada.png') 
plt.show()


# ===================================================================
#       Identificar frecuencias exactas para realizar la mascara
# ===================================================================

# --- Encontrar picos en el eje fx (líneas verticales brillantes) ---
# Sumar la magnitud a lo largo del eje y (axis=0) para obtener el perfil horizontal
sum_along_fy = np.sum(TF_magnitude, axis=0)
center_x_index = cols // 2

# Encontrar picos , Prominence ayuda a encontrar picos que sobresalen
peaks_fx_indices, properties_fx = find_peaks(sum_along_fy, prominence=np.max(sum_along_fy)/10) # prominencia > 10% del max

# Filtrar el pico central (DC) si está presente
peaks_fx_indices = peaks_fx_indices[peaks_fx_indices != center_x_index]

# Obtener las frecuencias fx correspondientes
high_fx_values = fx[peaks_fx_indices]

print("Altas frecuencias detectadas en fx (líneas verticales):")
if len(high_fx_values) > 0:
 for f_val, intensity_sum in zip(high_fx_values, sum_along_fy[peaks_fx_indices]):
  print(f" fx ≈ {f_val:.2f} (1/mm), Suma de Magnitud ≈ {intensity_sum:.2e}")
else:
 print(" No se detectaron picos prominentes fuera del centro.")

# --- Encontrar picos en el eje fy (líneas horizontales brillantes) ---
# Sumar la magnitud a lo largo del eje x (axis=1) para obtener el perfil vertical
sum_along_fx = np.sum(TF_magnitude, axis=1)
center_y_index = rows // 2

# Encontrar picos. Ajusta 'prominence' o 'height'
peaks_fy_indices, properties_fy = find_peaks(sum_along_fx, prominence=np.max(sum_along_fx)/10)

# Filtrar el pico central (DC)
peaks_fy_indices = peaks_fy_indices[peaks_fy_indices != center_y_index]

# Obtener las frecuencias fy correspondientes
high_fy_values = fy[peaks_fy_indices]

print("\nAltas frecuencias detectadas en fy (líneas horizontales):")
if len(high_fy_values) > 0:
 for f_val, intensity_sum in zip(high_fy_values, sum_along_fx[peaks_fy_indices]):
  print(f" fy ≈ {f_val:.2f} (1/mm), Suma de Magnitud ≈ {intensity_sum:.2e}")
else:
 print(" No se detectaron picos prominentes fuera del centro.")

# --- Graficamos los perfiles y los picos encontrados ---
fig_peaks, axes_peaks = plt.subplots(2, 1, figsize=(10, 8))

axes_peaks[0].plot(fx, sum_along_fy)
axes_peaks[0].plot(high_fx_values, sum_along_fy[peaks_fx_indices], "x", color='red', label='Picos Detectados (Altas Freq.)')
axes_peaks[0].set_title('Suma de Magnitud a lo largo de $f_y$ (Detecta picos en $f_x$)')
axes_peaks[0].set_xlabel('$f_x$ (1/mm)')
axes_peaks[0].set_ylabel('Suma de Magnitud')
axes_peaks[0].legend()
axes_peaks[0].grid(True)

axes_peaks[1].plot(fy, sum_along_fx)
axes_peaks[1].plot(high_fy_values, sum_along_fx[peaks_fy_indices], "x", color='red', label='Picos Detectados (Altas Freq.)')
axes_peaks[1].set_title('Suma de Magnitud a lo largo de $f_x$ (Detecta picos en $f_y$)')
axes_peaks[1].set_xlabel('$f_y$ (1/mm)')
axes_peaks[1].set_ylabel('Suma de Magnitud')
axes_peaks[1].legend()
axes_peaks[1].grid(True)

plt.tight_layout()
#plt.savefig('perfiles_frecuencia_picos.png')
plt.show() 

# ===================================================================
#                FILTRAR con máscara Gaussiana
# ===================================================================
def create_gaussian_notch_mask(TF_shape, fx_coords, fy_coords, noise_freq_coords, sigma_alrededor, sigma):
    
    # Utilizamos este metodo que calcula como son las frecuenicas segun los vecinos de cada pixel o punto
    # ya que esto nos da mayor precision al momento de crear la mascara

    # --- Crear matrices 2D con los valores de frecuencia fx y fy ---
    FX, FY = np.meshgrid(fx_coords, fy_coords) 

    # --- Inicializar la máscara ---
    gaussian_notch_mask = np.ones(TF_shape, dtype=float)

    # --- Crear y aplicar mascarasa Gaussianas para el RUIDO ---
    if not noise_freq_coords:
        print("  No se especificaron coordenadas de ruido.")
    else:
        for (center_fx, center_fy) in noise_freq_coords:
            distance_sq_noise = (FX - center_fx)**2 + (FY - center_fy)**2
            gaussian_dip = 1.0 - np.exp(-distance_sq_noise / (2 * sigma_alrededor**2))
            gaussian_notch_mask *= gaussian_dip
            #print(f"  Muesca Gaussiana aplicada centrada en ({center_fx:.2f}, {center_fy:.2f})")

    # --- Crear y aplicar mascara Gaussiana para el centro (si sigma > 0) ---
    if sigma > 0:
        distance_sq_ = FX**2 + FY**2
        dip = 1.0 - np.exp(-distance_sq_ / (2 * sigma**2))
        gaussian_notch_mask *= dip
        print(f"  Mascara  Gaussiana aplicada.")
    else:
        print("\nSigma es cero, no se aplicará mascara en el centro.")

    print("\n--- Máscara Gaussiana creada ---")
    return gaussian_notch_mask

# ===================================================================
#  Aplicacion la mascara GAUSSIANA para eliminar las frecuencias no deseadas
# ===================================================================

# --- Frecuencias base detectadas en los ejes ---
base_target_fx = high_fx_values
base_target_fy = high_fy_values

# --- Generar coordenadas (fx, fy) únicas objetivo (igual que antes) ---
target_coords_freq = []
for fy_val in base_target_fy: target_coords_freq.extend([(0, fy_val), (0, -fy_val)])
for fx_val in base_target_fx: target_coords_freq.extend([(fx_val, 0), (-fx_val, 0)])
for fx_val in base_target_fx:
    for fy_val in base_target_fy:
        target_coords_freq.extend([(fx_val, fy_val), (-fx_val, fy_val), (fx_val, -fy_val), (-fx_val, -fy_val)])
target_coords_freq_set = set(target_coords_freq)
target_coords_freq_set.discard((0, 0))
unique_target_coords_freq = list(target_coords_freq_set)

#print("\nCoordenadas (fx, fy) únicas objetivo para las mascaras de ruido:")
#for fx_val, fy_val in sorted(unique_target_coords_freq):
#    print(f"  ({fx_val:.2f}, {fy_val:.2f})")

# --- Parámetros de la máscara GAUSSIANA ---

sigma_noise_freq = 0.2 # Sigma para las masacaras de alrededor del centro  (1/mm)
sigma_freq = 0   # Sigma para la masacara en el centro (1/mm)

# --- Crear la máscara GAUSSIANA que no deja pasar las frecuencias (altas) ---
gaussian_mask_10 = create_gaussian_notch_mask(TF_shifted.shape, fx, fy, unique_target_coords_freq, sigma_noise_freq, sigma_freq)

# --- Mascara que solo deja pasar las frecuencias encontradas(altas) ---
gaussian_mask_01 = 1.0 - gaussian_mask_10

# --- Graficar la forma de la mascara ---
plt.figure(figsize=(7, 6))
plt.imshow(gaussian_mask_01, cmap='gray', extent=[fx.min(), fx.max(), fy.min(), fy.max()], origin='lower', aspect='equal', vmin=0, vmax=1)
plt.colorbar(label='Transmitancia de la Máscara Gaussiana')
plt.title(f'Máscara de Muesca Gaussiana (Sigma Alrededor={sigma_noise_freq:.2f}, Sigma={sigma_freq:.2f} 1/mm)')
plt.xlabel('$f_x$ (1/mm)')
plt.ylabel('$f_y$ (1/mm)')

# plt.savefig('mascara_gaussiana_2D_DC_creada.png')
plt.show()

# ===================================================================
#                   Función para aplicar el filtro
# ===================================================================

def filtrar_imagenes_mask(imagen_noisy, mask):

 # Calcular la TF de la imagen ruidosa y centrarla
 fft_image_shifted = np.fft.fftshift(np.fft.fft2(imagen_noisy))

 # Aplicar la mascara
 fft_filtered_shifted_local = fft_image_shifted * mask 

 # Volver a centrar la TF
 fft_filtered_local = np.fft.ifftshift(fft_filtered_shifted_local)

 # Calcula la TF inversa para obtener la imagen filtrada
 image_filtered_local = np.fft.ifft2(fft_filtered_local)

 # Toma la magnitud de la TF
 return np.abs(image_filtered_local), fft_image_shifted 

# ===================================================================
#           Graficar campos de entrada y salida corregidos
# ===================================================================

fig_final, axes_final = plt.subplots(1, 2, figsize=(21, 6)) 
intensity = np.abs(campo_ruidoso)**2
# Usaremos extent_cam1 que ya calculaste para el campo ruidoso
im_int = axes_final[0].imshow(intensity, cmap='gray', extent=extent_cam1, origin='lower', aspect='equal') 
fig_final.colorbar(im_int, ax=axes_final[0], label='Intensidad |O(u,v)|^2', shrink=0.8) 
axes_final[0].set_xlabel('u (mm)')
axes_final[0].set_ylabel('v (mm)')
axes_final[0].set_title('Campo en CAM1 Ruidoso $O(u,v)$') 
axes_final[0].grid(False)

# --- Intensidad de campo filtrado con máscara GAUSSIANA ---
campo_filtrado, TF_noisy_shifted = filtrar_imagenes_mask(campo_ruidoso, gaussian_mask_10)
intensity_filtered = np.abs(campo_filtrado)**2
im_s2 = axes_final[1].imshow(intensity_filtered, cmap='gray', extent=extent_cam1, origin='lower', aspect='equal') 
fig_final.colorbar(im_s2, ax=axes_final[1], label='Intensidad Filtrada', shrink=0.8) 
axes_final[1].set_xlabel('u (mm)')
axes_final[1].set_ylabel('v (mm)')
axes_final[1].set_title('Campo en CAM1 Filtrado') 
axes_final[1].grid(False)

plt.tight_layout()
#plt.savefig('Imagen_Filtrada_Gaussiana.png')
plt.show()