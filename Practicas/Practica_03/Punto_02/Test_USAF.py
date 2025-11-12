import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import cv2
from PIL import Image



ruta_del_resultado = "Practicas/Practica_03/Punto_01/resultado_microscopio.npy"
I_final_camara = np.load(ruta_del_resultado)

#plt.imshow(I_final_camara, cmap='gray')
#plt.title("Imagen Final (Cargada desde Archivo)")
#plt.show()

# --- 1. PARÁMETROS DEL SISTEMA (Reemplaza los valores necesarios) ---

# Parámetros Ópticos (dados por ti)
f_TL = 200            # Longitud focal de la lente de tubo (mm)
M = 20                 # Magnificación (ej: 20x)
lam = 533e-6             # Longitud de onda de la luz (mm) [533 nm]
NA = 0.25               # Apertura Numérica (NA) del objetivo (MO)
# -----------------------------------------------------------------

# Parámetros Derivados (calculados)
f_MO = f_TL / M          # Longitud focal del objetivo (mm)

# Parámetros del Detector (Cámara Alvium 1800 U-811m con Sony IMX546)
p_s = 0.00274            # Tamaño del píxel (mm) [2.74 µm]

print(f"--- Parámetros del Sistema ---")
print(f"  Longitud de onda (λ):   {lam * 1e6:.1f} nm")
print(f"  Magnificación (M):      {M:.1f}x")
print(f"  Apertura Numérica (NA): {NA:.2f}")
print(f"  Tamaño de píxel (p_s):  {p_s * 1e3:.2f} µm")
print(f"  f_MO (calculada):       {f_MO:.2f} mm")
print(f"  f_TL (dada):          {f_TL:.2f} mm")
print("-" * 30)

# --- 2. CÁLCULO DE LÍMITES TEÓRICOS (en lp/mm) ---

# Límite 1: Resolución Óptica (Límite de Difracción de Abbe)
# Es la máxima frecuencia espacial (lp/mm) que la óptica puede pasar.
SF_optico = NA / lam
print(f"  Límite Óptico (Abbe):     {SF_optico:.2f} lp/mm")

# Límite 2: Resolución del Detector (Límite de Nyquist)
# Es la máxima frecuencia que la cámara puede "ver", referida al plano del objeto.
SF_camara_objeto = M / (2 * p_s)
print(f"  Límite Detector (Nyquist): {SF_camara_objeto:.2f} lp/mm")

print("-" * 30)

# --- 3. LÍMITE TEÓRICO TOTAL DEL SISTEMA ---
# El sistema estará limitado por el valor MÁS BAJO de los dos.
SF_teorica_total = min(SF_optico, SF_camara_objeto)

if SF_optico < SF_camara_objeto:
    print(f"  Sistema LIMITADO POR DIFRACCIÓN (la óptica es el límite).")
else:
    print(f"  Sistema LIMITADO POR DETECTOR (los píxeles son el límite).")

print(f"\n  RESOLUCIÓN TEÓRICA TOTAL: {SF_teorica_total:.2f} lp/mm")


# --- Fórmula USAF (la usaremos mucho) ---
def get_sf_usaf(G, E):
    """
    Calcula la frecuencia espacial (lp/mm) para un Grupo (G) y Elemento (E)
    de la mira USAF 1951.
    """
    return 2**(G + (E - 1) / 6.0)

# --- Calibración Teórica (en mm) ---
G_calib = 0
E_calib = 1

# 1. Frecuencia espacial de nuestro elemento de calibración
SF_calib = get_sf_usaf(G_calib, E_calib)
print(f"Elemento de calibración: Grupo {G_calib}, Elemento {E_calib}")
print(f"  Frecuencia teórica (SF_calib): {SF_calib:.2f} lp/mm")

# 2. Periodo (cuántos mm ocupa un par línea-espacio)
P_calib = 1.0 / SF_calib
print(f"  Periodo teórico (P_calib):   {P_calib:.2f} mm/lp")

# 3. Ancho total del patrón
# Un patrón USAF (3 barras, 2 espacios) equivale a 2.5 periodos.
ancho_teorico_mm = 2.5 * P_calib
print(f"  Ancho total teórico del patrón: {ancho_teorico_mm:.2f} mm")


import numpy as np
import matplotlib.pyplot as plt

# (Asegúrate de tener también tu función 'calcular_contraste')
def calcular_contraste(I_max, I_min):
    if I_max + I_min == 0: return 0.0
    return (I_max - I_min) / (I_max + I_min)

def analizar_contraste_region(I_imagen_final, x_vec, y_vec, 
                              x_centro_mm, y_centro_mm, 
                              ancho_mm, alto_mm, 
                              nombre_elemento="Elemento"):
    """
    Analiza el contraste en una región rectangular (ROI) de la imagen final.
    
    Argumentos:
    I_imagen_final: Tu array 2D con la imagen resultado.
    x_vec, y_vec: Tus vectores de coordenadas en mm.
    x_centro_mm, y_centro_mm: Coordenadas (en mm) del CENTRO de la caja.
    ancho_mm, alto_mm: Tamaño (en mm) de la caja de análisis.
    nombre_elemento: String para el título del gráfico.
    """
    
    print(f"\n--- Analizando Región: {nombre_elemento} ---")
    
    # --- 1. Encontrar los límites de la caja en mm ---
    x_min_mm = x_centro_mm - ancho_mm / 2
    x_max_mm = x_centro_mm + ancho_mm / 2
    y_min_mm = y_centro_mm - alto_mm / 2
    y_max_mm = y_centro_mm + alto_mm / 2

    # --- 2. Encontrar los índices de píxeles correspondientes ---
    try:
        idx_x_min = np.argmin(np.abs(x_vec - x_min_mm))
        idx_x_max = np.argmin(np.abs(x_vec - x_max_mm))
        idx_y_min = np.argmin(np.abs(y_vec - y_min_mm))
        idx_y_max = np.argmin(np.abs(y_vec - y_max_mm))
        
        # Asegurarnos de que min < max
        if idx_x_min > idx_x_max: idx_x_min, idx_x_max = idx_x_max, idx_x_min
        if idx_y_min > idx_y_max: idx_y_min, idx_y_max = idx_y_max, idx_y_min
        
    except Exception as e:
        print(f"Error encontrando índices: {e}")
        return

    # --- 3. Recortar la región de interés (ROI) ---
    region_recortada = I_imagen_final[idx_y_min : idx_y_max, 
                                      idx_x_min : idx_x_max]

    if region_recortada.size == 0:
        print("Error: La región recortada está vacía. Verifica coordenadas.")
        return

    # --- 4. Calcular I_max e I_min (Método Estadístico) ---
    # Usamos percentiles (95 y 5) en lugar de max/min.
    # Esto es mucho más robusto contra píxeles ruidosos.
    I_max = np.percentile(region_recortada, 95)
    I_min = np.percentile(region_recortada, 5)

    # --- 5. Calcular Contraste ---
    C = calcular_contraste(I_max, I_min)

    print(f"  I_max (Percentil 95): {I_max:.4f}")
    print(f"  I_min (Percentil 5):  {I_min:.4f}")
    print(f"  CONTRASTE MEDIDO (C): {C:.4f}")

    # --- 6. La Prueba Visual: El Histograma ---
    plt.figure(figsize=(10, 5))
    
    # Gráfico de la región recortada
    plt.subplot(1, 2, 1)
    plt.imshow(region_recortada, cmap='gray', aspect='equal')
    plt.title(f"Región Recortada: {nombre_elemento}")
    
    # Gráfico del histograma
    plt.subplot(1, 2, 2)
    plt.hist(region_recortada.ravel(), bins=50, color='blue', alpha=0.7)
    plt.axvline(I_min, color='g', linestyle='--', label=f'I_min (P5) = {I_min:.2f}')
    plt.axvline(I_max, color='r', linestyle='--', label=f'I_max (P95) = {I_max:.2f}')
    plt.title(f"Histograma (Contraste = {C:.3f})")
    plt.xlabel("Intensidad de Píxel")
    plt.ylabel("Conteo")
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return C
    
N_puntos = I_final_camara.shape[0]  # Obtener N desde la imagen cargada
L_simulacion = 10  # ¡¡REEMPLAZA ESTO!! con tu ancho real en mm

x_vec = np.linspace(-L_simulacion / 2, L_simulacion / 2, N_puntos)
y_vec = np.linspace(-L_simulacion / 2, L_simulacion / 2, N_puntos)

# Habilitar modo interactivo de Matplotlib
# %matplotlib qt  # (Si estás en un notebook/iPython)
plt.figure(figsize=(10, 8))
plt.pcolormesh(x_vec, y_vec, I_final_camara, cmap='gray', shading='auto')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.title('Imagen Resultado - Haz Zoom para encontrar G8 y G9')
plt.axis('equal')
plt.colorbar()
plt.show()
   
# --- Carga tus datos ---
# I_imagen_final = np.load(...)
# x_vec = ...
# y_vec = ...

# --- ANÁLISIS G8, E6 (Resuelto) ---
# (Reemplaza estos valores con tu medición del gráfico)
x_centro_g8 = -0.04   # (mm) - EJEMPLO
y_centro_g8 = -0.15   # (mm) - EJEMPLO
ancho_g8    = 0.005 # (mm) - EJEMPLO
alto_g8     = 0.015 # (mm) - EJEMPLO (para barras verticales)

C_G8E6 = analizar_contraste_region(I_final_camara, x_vec, y_vec,
                                   x_centro_g8, y_centro_g8,
                                   ancho_g8, alto_g8,
                                   nombre_elemento="Grupo 8, Elemento 6")

# --- ANÁLISIS G9, E1 (No resuelto) ---
# (Reemplaza estos valores con tu medición del gráfico)
x_centro_g9 = -0.02   # (mm) - EJEMPLO
y_centro_g9 = -0.13   # (mm) - EJEMPLO
ancho_g9    = 0.004 # (mm) - EJEMPLO
alto_g9     = 0.012 # (mm) - EJEMPLO

C_G9E1 = analizar_contraste_region(I_final_camara, x_vec, y_vec,
                                   x_centro_g9, y_centro_g9,
                                   ancho_g9, alto_g9,
                                   nombre_elemento="Grupo 9, Elemento 1")


# ===================================================================
#                  Análisis Cuantitativo de la PSF
# ===================================================================

# I_final_camara ya fue calculada por tu simulación
# L_camara_salida y mesh_camara también

# 1. Extraer el perfil de la PSF
print("\n--- Análisis de Resolución por PSF ---")
try:
    # Obtener el vector de coordenadas X (en mm) del plano de la cámara
    # (Asumiendo que mesh_camara es (X_cam, Y_cam))
    x_cam_vec = mesh_camara[0][0, :] 
    
    # Obtener el perfil central (fila del medio)
    centro_y_idx = I_final_camara.shape[0] // 2
    perfil_psf = I_final_camara[centro_y_idx, :]
    
    # Normalizar el perfil
    perfil_psf = perfil_psf / np.max(perfil_psf)

    # 2. Calcular el FWHM (Ancho a la Mitad del Máximo)
    
    # Encontrar todos los píxeles que están por encima del 50% (0.5)
    indices_fwhm = np.where(perfil_psf >= 0.5)[0]
    
    if indices_fwhm.size > 0:
        idx_inicio = indices_fwhm[0]
        idx_fin = indices_fwhm[-1]
        
        # Ancho en píxeles
        ancho_en_pixeles = idx_fin - idx_inicio
        
        # Convertir píxeles a mm
        dx_camara = L_camara_salida[0] / I_final_camara.shape[1]
        FWHM_medido_camara = ancho_en_pixeles * dx_camara
        
        # 3. Referir la medida al plano del OBJETO
        # (Dividimos por la magnificación del sistema)
        FWHM_medido_objeto = FWHM_medido_camara / Mx # Mx = 20
        
        print(f"  FWHM medido (en cámara):   {FWHM_medido_camara * 1000:.2f} µm")
        print(f"  Magnificación (M):        {Mx:.1f}x")
        print(f"  FWHM MEDIDO (en objeto):  {FWHM_medido_objeto * 1000:.2f} µm")

        # 4. Calcular el FWHM Teórico
        # Para un disco de Airy (la PSF teórica), FWHM = 0.51 * λ / NA
        FWHM_teorico = (0.51 * lam) / NA
        
        print(f"  FWHM TEÓRICO (0.51*λ/NA): {FWHM_teorico * 1000:.2f} µm")

        # 5. Calcular el Error
        error_psf = np.abs((FWHM_teorico - FWHM_medido_objeto) / FWHM_teorico) * 100
        print(f"\n  Error entre Medición y Teoría: {error_psf:.1f}%")

        # 6. Graficar el perfil de la PSF
        plt.figure()
        plt.plot(x_cam_vec, perfil_psf)
        plt.axhline(0.5, color='r', linestyle='--', label='50% (FWHM)')
        plt.axvline(x_cam_vec[idx_inicio], color='g', linestyle='--')
        plt.axvline(x_cam_vec[idx_fin], color='g', linestyle='--', label=f'FWHM = {FWHM_medido_camara*1000:.2f} µm')
        plt.title("Perfil de la PSF Medida (en el plano de la cámara)")
        plt.xlabel("Posición u (mm)")
        plt.ylabel("Intensidad Normalizada")
        plt.legend()
        plt.show()

    else:
        print("  Error: No se pudo encontrar el FWHM. El perfil es demasiado estrecho.")

except Exception as e:
    print(f"Error durante el análisis de PSF: {e}")
    print("Asegúrate de que 'mesh_camara' está definido correctamente.")