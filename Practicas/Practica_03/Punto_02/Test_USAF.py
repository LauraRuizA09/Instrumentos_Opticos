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
SF_optico = 2 * NA / lam
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

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# --- 1. FUNCIONES AUXILIARES (Necesarias para el análisis) ---

def calcular_contraste(I_max, I_min):
    """Calcula el Contraste de Michelson."""
    if I_max + I_min == 0: 
        return 0.0
    return (I_max - I_min) / (I_max + I_min)

def extraer_perfil(I_imagen_final, x_vec, y_vec, p1_mm, p2_mm, num_puntos=100):
    """
    Extrae un perfil de intensidad de una imagen 2D a lo largo de una línea.
    
    Argumentos:
    I_imagen_final : El array 2D de tu imagen resultado.
    x_vec          : El vector de coordenadas X (en mm) (eje 1 de I_imagen_final).
    y_vec          : El vector de coordenadas Y (en mm) (eje 0 de I_imagen_final).
    p1_mm          : Tupla (x, y) de inicio del perfil (en mm).
    p2_mm          : Tupla (x, y) de fin del perfil (en mm).
    """
    
    # Crear la función de interpolación.
    # Asume que 'y_vec' corresponde al eje 0 y 'x_vec' al eje 1
    try:
        # Nota: Los ejes para RegularGridInterpolator deben estar en orden (y, x)
        interpolador = RegularGridInterpolator((y_vec, x_vec), I_imagen_final)
    except ValueError as e:
        print(f"Error al crear interpolador: {e}")
        print(f"Asegúrate que I_final ({I_imagen_final.shape}) coincide con y_vec ({y_vec.size}) y x_vec ({x_vec.size}).")
        return None, None
    except Exception as e:
        print(f"Error inesperado en RegularGridInterpolator: {e}")
        return None, None

    # Crear los puntos (x, y) a lo largo de la línea
    x_perfil = np.linspace(p1_mm[0], p2_mm[0], num_puntos)
    y_perfil = np.linspace(p1_mm[1], p2_mm[1], num_puntos)
    
    # Crear el eje de distancia para la gráfica
    distancia_total = np.sqrt((p2_mm[0] - p1_mm[0])**2 + (p2_mm[1] - p1_mm[1])**2)
    eje_distancia = np.linspace(0, distancia_total, num_puntos)
    
    # Muestrear los puntos. El interpolador espera (y, x).
    puntos_a_muestrear = np.vstack((y_perfil, x_perfil)).T
    
    try:
        perfil = interpolador(puntos_a_muestrear)
        return eje_distancia, perfil
    except Exception as e:
        print(f"Error al interpolar puntos (¿puntos fuera de rango?): {e}")
        return None, None

# --- 2. CARGA DE DATOS Y DEFINICIÓN DE COORDENADAS ---
print("Cargando datos...")
try:
    ruta_del_resultado = "Practicas/Practica_03/Punto_01/resultado_microscopio.npy"
    I_final_camara = np.load(ruta_del_resultado) 
except FileNotFoundError:
    print(f"ERROR: No se encontró el archivo en {ruta_del_resultado}")
    I_final_camara = np.random.rand(100,100) # Placeholder para evitar crash
    print("Usando datos aleatorios. ¡Revisa la ruta del archivo .npy!")

# Usamos Lx=1.0, como confirmaste
N_puntos = I_final_camara.shape[0]
L_simulacion = 1.0  

x_vec = np.linspace(-L_simulacion / 2, L_simulacion / 2, N_puntos)
y_vec = np.linspace(-L_simulacion / 2, L_simulacion / 2, N_puntos)
print(f"Datos cargados. Tamaño: {I_final_camara.shape}, L={L_simulacion} mm")


# --- 3. VISUALIZACIÓN INTERACTIVA (Tu Tarea) ---
print("\n--- TAREA MANUAL ---")
print("Se abrirá la imagen. Haz zoom para encontrar G9-E6 y G10-E1.")
print("Encuentra los puntos p1 y p2 (en mm) para una LÍNEA CORTA y PERPENDICULAR a las barras.")
print("Cierra la ventana y edita las variables en el Paso 4.")

plt.figure(figsize=(10, 8))
plt.pcolormesh(x_vec, y_vec, I_final_camara, cmap='gray', shading='auto')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.title('Imagen Resultado - Busca G9-E6 y G10-E1')
plt.axis('equal')
plt.colorbar()
plt.show()

# --- 4. ANÁLISIS DE CONTRASTE (Método de Línea) ---

# --- ANÁLISIS G9, E6 (912.2 lp/mm) ---
# (¡REEMPLAZA ESTOS VALORES!)
p1_G9E6 = (0.104, -0.535)  # (x_inicio, y_inicio) en mm
p2_G9E6 = (0.180, -0.694) # (x_fin, y_fin) en mm

dist_G9E6, perfil_G9E6 = extraer_perfil(I_final_camara, x_vec, y_vec, 
                                        p1_G9E6, p2_G9E6, num_puntos=100)

if perfil_G9E6 is not None:
    I_max = np.max(perfil_G9E6)
    I_min = np.min(perfil_G9E6)
    C_G9E6 = calcular_contraste(I_max, I_min)
    
    print(f"\n--- Resultados del Análisis (G9, E6) ---")
    print(f"  Frecuencia (G9, E6): 912.2 lp/mm")
    print(f"  I_max medida: {I_max:.4f}")
    print(f"  I_min medida: {I_min:.4f}")
    print(f"  CONTRASTE MEDIDO (C): {C_G9E6:.4f}")
else:
    print("Error al extraer el perfil para G9-E6. Revisa tus coordenadas p1/p2.")

# --- ANÁLISIS G10, E1 (1024 lp/mm) ---
# (¡REEMPLAZA ESTOS VALORES!)
p1_G10E1 = (-1.051, -0.474)  # (x_inicio, y_inicio) en mm
p2_G10E1 = (-1.011, -0.586) # (x_fin, y_fin) en mm

dist_G10E1, perfil_G10E1 = extraer_perfil(I_final_camara, x_vec, y_vec, 
                                          p1_G10E1, p2_G10E1, num_puntos=100)

if perfil_G10E1 is not None:
    I_max = np.max(perfil_G10E1)
    I_min = np.min(perfil_G10E1)
    C_G10E1 = calcular_contraste(I_max, I_min)
    
    print(f"\n--- Resultados del Análisis (G10, E1) ---")
    print(f"  Frecuencia (G10, E1): 1024 lp/mm")
    print(f"  I_max medida: {I_max:.4f}")
    print(f"  I_min medida: {I_min:.4f}")
    print(f"  CONTRASTE MEDIDO (C): {C_G10E1:.4f}")
else:
    print("Error al extraer el perfil para G10-E1. Revisa tus coordenadas p1/p2.")