import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import cv2
from PIL import Image
from scipy.ndimage import gaussian_filter1d

ruta_del_resultado = "Practicas/Practica_03/Punto_01/resultado_microscopio.npy"
I_final_camara = np.load(ruta_del_resultado)

# --- Líneas para graficar ---

# 1. Crea una figura
plt.figure(figsize=(8, 6))

# 2. Muestra la imagen en escala de grises
plt.imshow(I_final_camara, cmap='gray')

# 3. Añade la barra de color (recomendado)
plt.colorbar(label='Intensidad')

# 4. Añade títulos
plt.title("Imagen Resultante en la Cámara")
plt.xlabel("Píxeles (x)")
plt.ylabel("Píxeles (y)")

# 5. Muestra el gráfico
plt.show()


# ---  PARÁMETROS DEL SISTEMA ---

# Parámetros Ópticos 
f_TL = 200            # Longitud focal de la lente de tubo (mm)
M = 20                 # Magnificación (ej: 20x)
lam = 533e-6             # Longitud de onda de la luz (mm) [533 nm]
NA = 0.25               # Apertura Numérica (NA) del objetivo (MO)
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




# --- PARTE 1: Función de Cálculo Teórico ---
# (Sin cambios)
def calcular_datos_usaf(group, element):
    sf_lpmm = 2**(group + (element - 1) / 6.0)
    return sf_lpmm

# --- PARTE 2: Función de Análisis de Imagen (LÓGICA MEJORADA) ---

def analizar_perfil_robusto(nombre_archivo_imagen):
    """
    Calcula el perfil, ratio y uniformidad de una imagen pre-recortada
    ASUMIENDO QUE CONTIENE BARRAS HORIZONTALES.
    """
    img_color = cv2.imread(nombre_archivo_imagen)

    imagen_gris = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    h, w = imagen_gris.shape
    
    tipo_barras = "Horizontales (Asumido)"
    perfil_crudo = np.mean(imagen_gris, axis=1) # Perfil vertical
    
    # 1. Suavizar el perfil
    sigma_suavizado = 15
    perfil_suave = gaussian_filter1d(perfil_crudo, sigma=sigma_suavizado)
    
    # 2. Encontrar picos y valles
    rango_dinamico = np.percentile(perfil_suave, 95) - np.percentile(perfil_suave, 5)
    prominencia_minima = max(rango_dinamico * 0.05, 3.0) 
    distancia_minima_pixeles = h / 6.0 
        
    indices_picos, _ = find_peaks(perfil_suave, 
                                  prominence=prominencia_minima, 
                                  distance=distancia_minima_pixeles)
    indices_valles, _ = find_peaks(-perfil_suave, 
                                   prominence=prominencia_minima, 
                                   distance=distancia_minima_pixeles)
    
    # 3. Calcular Métricas
    # SI, Y SOLO SI, encontramos al menos 3 picos y 2 valles
    if len(indices_picos) >= 3 and len(indices_valles) >= 2:
        
        intensidades_picos = perfil_suave[indices_picos]
        intensidades_valles = perfil_suave[indices_valles]
        
        i_max_promedio = np.mean(intensidades_picos)
        i_min_promedio = np.mean(intensidades_valles)
        
        # Calcular Ratio de Rayleigh
        ratio_rayleigh = i_min_promedio / i_max_promedio if i_max_promedio > 0 else 1.0
            
        # Calcular Uniformidad de Picos (Tu criterio)
        std_picos = np.std(intensidades_picos)
        uniformidad_picos_ratio = std_picos / i_max_promedio if i_max_promedio > 0 else 1.0
            
        print(f"  Picos/Valles encontrados: {len(indices_picos)} picos, {len(indices_valles)} valles.")
        print(f"  I_max_prom={i_max_promedio:.1f}, I_min_prom={i_min_promedio:.1f}")
            
    else:
        # ¡No Resuelto! (El "parche gris")
        print(f"  No se detectaron 3 picos y 2 valles (Encontrados: {len(indices_picos)} picos, {len(indices_valles)} valles).")
        
        # Forzamos los ratios para que fallen
        ratio_rayleigh = 1.0
        uniformidad_picos_ratio = 1.0
        i_max_promedio = np.mean(perfil_suave)
        i_min_promedio = i_max_promedio
    
    return perfil_crudo, perfil_suave, i_max_promedio, i_min_promedio, ratio_rayleigh, uniformidad_picos_ratio, indices_picos, indices_valles

# --- PARTE 3: LÓGICA PRINCIPAL (Estilo "Script") ---

print("--- Analizador de Resolución Robusto (Rayleigh + Uniformidad) ---")
print("--- MODO: BARRAS HORIZONTALES FIJO ---")

# --- LAURA: DEFINE AQUÍ TUS IMÁGENES Y SU NÚMERO DE ELEMENTO ---
imagenes_y_elementos = [
("Practicas/Practica_03/Punto_02/LImites Resolucion (Cualit.)/G1E2.png", 2),
("Practicas/Practica_03/Punto_02/LImites Resolucion (Cualit.)/G1E3.png", 3),
("Practicas/Practica_03/Punto_02/LImites Resolucion (Cualit.)/G1E4.png", 4),
("Practicas/Practica_03/Punto_02/LImites Resolucion (Cualit.)/G1E5.png", 5),
("Practicas/Practica_03/Punto_02/LImites Resolucion (Cualit.)/G1E6.png", 6)
]
# -----------------------------------------------------------------

group_num = 1 # Fijo para este ejemplo

resultados_finales = [] 
umbral_rayleigh = 0.735 # 73.5%
umbral_uniformidad = 0.20 # 20% (Umbral para tu criterio. Ajusta si es necesario)

# Iterar, analizar y MOSTRAR CADA GRÁFICO
for filename, element_num in imagenes_y_elementos:
    
    print(f"\n--- Procesando: '{filename}' (Grupo {group_num}, Elemento {element_num}) ---")
    
    sf_lpmm = calcular_datos_usaf(group_num, element_num)
    
    resultado = analizar_perfil_robusto(filename)
    if resultado[0] is None:
        print("  Saltando esta imagen.")
        continue 
        
    perfil_crudo, perfil_suave, i_max, i_min, ratio_r, ratio_u, idx_picos, idx_valles = resultado
    
    print(f"  Frecuencia Teórica: {sf_lpmm:.3f} lp/mm")
    print(f"  Ratio Rayleigh (I_min/I_max): {ratio_r:.4f} (Límite: <= {umbral_rayleigh})")
    print(f"  Ratio Uniformidad (Std/Mean): {ratio_u:.4f} (Límite: <= {umbral_uniformidad})")
    
    resultados_finales.append((element_num, sf_lpmm, ratio_r, ratio_u))
    
    # --- Graficar el perfil ---
    plt.figure(figsize=(8, 6))
    
    plt.plot(perfil_crudo, range(len(perfil_crudo)), 'k-', alpha=0.3, label='Perfil Crudo')
    plt.plot(perfil_suave, range(len(perfil_suave)), 'c-', lw=2, label='Perfil Suavizado')
    plt.plot(perfil_suave[idx_picos], idx_picos, 'r^', markersize=8, label='Picos Detectados')
    plt.plot(perfil_suave[idx_valles], idx_valles, 'bv', markersize=8, label='Valles Detectados')

    plt.xlabel("Intensidad Promediada (0-255)")
    plt.ylabel("Posición Y (píxeles)")
    
    plt.axvline(i_max, color='r', linestyle='--', label=f'I_max_prom ~ {i_max:.1f}')
    plt.axvline(i_min, color='b', linestyle='--', label=f'I_min_prom ~ {i_min:.1f}')
    plt.axvline(i_max * umbral_rayleigh, color='g', linestyle=':', label=f'Umbral Rayleigh ({i_max * umbral_rayleigh:.1f})')
    
    plt.title(f"Perfil: G={group_num}, E={element_num} (Ratio_R={ratio_r:.3f}, Ratio_U={ratio_u:.3f})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    print("  Mostrando gráfico de perfil... Cierra la ventana para continuar.")
    plt.show() 

# --- PARTE 4: REPORTE FINAL ---
if resultados_finales:
    resultados_finales.sort(key=lambda item: item[1]) # Ordenar por frecuencia
    
    print("\n" + "="*70)
    print(f"--- REPORTE DE RESOLUCIÓN (Criterio Doble, G={group_num}) ---")
    print("  Freq (lp/mm) | Elem | Ratio Rayleigh | Uniform. Picos | Veredicto")
    print("  ----------------------------------------------------------------------")
    
    for (elem, freq, ratio_r, ratio_u) in resultados_finales:
        # Veredicto basado en AMBOS criterios
        if ratio_r <= umbral_rayleigh and ratio_u <= umbral_uniformidad:
            veredicto = "RESUELTO"
        else:
            veredicto = "NO RESUELTO"
            
        print(f"     {freq:7.3f}   |  {elem}   |    {ratio_r:8.4f}    |     {ratio_u:8.4f}   | {veredicto}")
    
    print("="*70)
else:
    print("\nNo se pudo analizar ninguna imagen.")