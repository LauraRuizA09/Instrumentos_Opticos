
# ===================================================================
#               Matrices de transferencia de rayos
# ===================================================================

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.widgets import Slider

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
        imagen = Image.open('Practicas/Practica_02/Actividad_1/Noise images/Noise (16).png').convert('L')
        imagen = imagen.resize((Nx, Ny))  # Redimensionar la imagen al tamaño Nx x Ny
        U0 = np.array(imagen) / 255.0  # Normalizar a [0, 1]
        U0 = U0.astype(np.complex128)  # Convertir a tipo complejo para incluir fase si es necesario

    else:
        raise ValueError("Tipo de objeto no reconocido. Use 'rectangular', 'circular' o 'imagen'.")

    return U0

# Obtener el campo de entrada

campo_entrada_1 = transmitancia_entrada('imagen')   #  S(ξ,η)


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


S1_campo, S1_x_mesh, S1_y_mesh, S1_dx, S1_dy = propagar_ABCD_(campo_entrada_1, A_1, B_1, C_1, D_1, lam)
# se debe multiplicar por la transmitaancia t(x,y) pero en este caso es 1

#Definir los campos de entrada para la segund parte de la trayectoria 01

campo_entrada_2 =  S1_campo 

S2_campo, S2_x_mesh, S2_y_mesh, S2_dx, S2_dy = propagar_ABCD_(campo_entrada_2, A_2, B_2, C_2, D_2, lam)


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
axes[0].set_title('Campo de Entrada $S(ξ,η)$')
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
axes[1].set_title('Campo en M1 $t(x,y)$') # Título más descriptivo
axes[1].grid(False)
axes[1].set_xlim(-0.05,0.05)
axes[1].set_ylim(-0.05,0.05)

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
axes[2].set_title('Campo en CAM1 $O(u,v)$') # Título más descriptivo
axes[2].grid(False)


# --- Ajustar espaciado y mostrar la figura completa ---
plt.tight_layout() # Ajusta el espaciado para evitar superposiciones
plt.show() # Muestra la figura con los tres subplots


# ===================================================================
#              ELIMINAR RUIDO DE LA IMAGEN (CON SLIDERS)
# ===================================================================

# Campo con ruido (resultado de la propagación)
campo_ruidoso = S2_campo  #  O(u,v)
rows, cols = campo_ruidoso.shape # Obtener dimensiones aquí

# --- Función para aplicar el filtro  ---
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
    return np.abs(image_filtered_local)

# --- Configuración inicial para el gráfico interactivo ---
# Valor inicial para el único slider
initial_cutoff_slider = 450

# Ajustar el máximo del slider. Limitado por la dimensión más pequeña
max_cutoff = min(600, cols // 2, rows // 2)
# Asegurar que el valor inicial no exceda el máximo
initial_cutoff_slider = min(initial_cutoff_slider, max_cutoff)

fig_slider, axs_slider = plt.subplots(1, 2, figsize=(12, 6))
# Ajustar espacio inferior para un solo slider
plt.subplots_adjust(bottom=0.2) # Menos espacio necesario ahora

# Mostrar imagen original ruidosa (intensidad)
im_noisy_slider = axs_slider[0].imshow(np.abs(campo_ruidoso)**2, cmap='gray')
axs_slider[0].set_title('Imagen Original (Ruidosa)')
axs_slider[0].axis('off')

# Aplicar filtro inicial y mostrar
filtered_image_slider_initial = filtrar_imagenes(campo_ruidoso, initial_cutoff_slider)
im_filtered_slider = axs_slider[1].imshow(filtered_image_slider_initial, cmap='gray')
# Actualizar título para reflejar un solo valor de corte
axs_slider[1].set_title(f'Imagen Filtrada (Corte X=Y={initial_cutoff_slider})')
axs_slider[1].axis('off')

# --- Crear eje para el slider ---
axcolor_slider = 'lightgoldenrodyellow'
# Un solo eje ahora, ligeramente más arriba
ax_cutoff_slider = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor=axcolor_slider)

# --- Crear el slider ---
slider_cutoff_widget = Slider( # Renombrado para claridad
    ax=ax_cutoff_slider,
    label='Corte X=Y', # Etiqueta actualizada
    valmin=10,
    valmax=max_cutoff, # Usar el máximo calculado
    valinit=initial_cutoff_slider,
    valstep=10
)

# --- Función de actualización para el slider ---
def update_single_slider(val):
    # Leer el valor del único slider
    current_cutoff_slider = int(slider_cutoff_widget.val)
    # Aplicar el filtro usando este valor para ambos ejes
    filtered_image_new_slider = filtrar_imagenes(campo_ruidoso, current_cutoff_slider)
    im_filtered_slider.set_data(filtered_image_new_slider)
    # Actualizar título
    axs_slider[1].set_title(f'Imagen Filtrada (Corte X=Y={current_cutoff_slider})')
    fig_slider.canvas.draw_idle()

# --- Conectar slider a la función de actualización ---
slider_cutoff_widget.on_changed(update_single_slider) # Conectar el único slider

# --- Mostrar el gráfico interactivo ---
print("\nMostrando gráfico interactivo para filtrado simétrico. Mueve el slider.")
plt.show()

print(f"\nProceso completado.")