
# ===================================================================
#                    Funciones para el test USAF
# ===================================================================

import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter1d 

# ===================================================================
#               Calculo numerico de la resolución
# ===================================================================4

def calcular_resolucion_usaf():

    print("CÁLCULO DE LA RESOLUCIÓN TEÓRICA (LÍMITE DE ABBE)")
    print("---------------------------------------------------------")

    # --- Parámetros Fijos del Sistema---
    # Unidades en metros m
    
    # Longitud de onda 533 nm
    lambda_m = 533e-9
    
    # Apertura Numérica 
    NA = 0.25
    
    # Magnificación del objetivo 
    M = 20
    
    # Parámetros del sensor (Alvium 1800 U-811m) 
    delta_sensor_m = 2.74e-6            # Tamaño de pixel sensor Sony IMX546
    N = 2848                            # Número de píxeles  Nx = Ny
    
    # Tamaño estándar del test USAF T20-CPG
    # link: https://www.appliedimage.com/product-category/test-targets-and-charts/usaf-targets/usaf-1951-standard-resolution-target-t-20/
    # 10.2 cm = 0.102 m   ------> Este test utilizado mide 10.2cm X 10.2cm
    L_m = 0.102 
    
    # 1.1: Calcular Límite de Abbe
    #  R_Abbe = lambda / (2 * NA)
    r_abbe_m = lambda_m / (2 * NA)
    
    print(f"Parámetros del sistema:")
    print(f"  Longitud de onda (lambda): {lambda_m * 1e9:.1f} nm")
    print(f"  Apertura Numérica (NA): {NA}")
    print(f"Resultado (Teórico):")
    print(f"  Resolución de Abbe (R_Abbe) = {r_abbe_m * 1e6:.3f} µm\n")

    print("---------------------------------------------------------")
    print("CÁLCULO DE LA RESOLUCIÓN MEDIDA (USAF ESCALADO)")
    print("---------------------------------------------------------")

    #  Registrar los datos de cual es el ultimo G y E que se logra ver con claridad
    try:
        grupo = int(input("Ingresa el número de GRUPO observable: "))
        elemento = int(input("Ingresa el número de ELEMENTO observable: "))
    except ValueError:
        print("Error: Debes ingresar números enteros.")
        return

    # Calcular la Resolución (R_sis) para ese (G, E)
    # Fórmula estándar USAF 1951 (en pares de líneas por mm)
    # R_lp_mm = 2^(Grupo + (Elemento - 1) / 6)
    r_lp_mm = math.pow(2, (grupo + (elemento - 1) / 6))
    
    # Convertir a pares de líneas por metro (lp/m)
    r_lp_m = r_lp_mm * 1000
    
    # Convertir a tamaño de característica (resolución) en metros
    # R_Std = 1 / (lp/m)
    r_std_m = 1.0 / r_lp_m

    print(f"Cálculo de R_sis para (Grupo {grupo}, Elemento {elemento}):")
    print(f"  Frecuencia espacial = {r_lp_mm:.3f} lp/mm")
    print(f"  Resolución (R_sis) = {r_std_m * 1e6:.3f} µm")

    # Calcular el tamaño de píxel en el plano objeto 
    # Delta_Objeto = Delta_Sensor / M
    delta_objeto_m = delta_sensor_m / M

    print(f"\nCálculo de Escalado (Simulación):")
    print(f"  Tamaño de píxel del sensor (Delta_Sensor) = {delta_sensor_m * 1e6:.3f} µm")
    print(f"  Tamaño de píxel del objeto (Delta_Objeto) = {delta_objeto_m * 1e6:.3f} µm")

    # Calcular el número total de píxeles que tendría el test estándar 
    # N_test = L_Std / Delta_Objeto
    N_test = L_m / delta_objeto_m
    print(f"  Píxeles en el test estándar = {N_test:.0f}")

    # Usar la regla de tres para encontrar la resolución real (Z) 
    # N_test / R_sis = N / R_real  =>  R_real = (N * R_sis) / Y
    R_real = (N * r_std_m) / N_test
    
    print(f"  Píxeles en la simulación = {N:.0f}")
    print(f"Resultado (Medido):")
    print(f"  Resolución Medida = {R_real * 1e6:.3f} µm\n")
    
    print("---------------------------------------------------------")
    print(" COMPARACIÓN DE RESULTADOS")
    print("---------------------------------------------------------")
    
    error_abs_m = (abs(R_real - r_abbe_m)/r_abbe_m) * 100
    
    print(f"  Resolución Teórica (Abbe): {r_abbe_m * 1e6:.3f} µm")
    print(f"  Resolución Medida (USAF):  {R_real * 1e6:.3f} µm")
    print(f"  Error % = {error_abs_m:.3f} %")
    print("---------------------------------------------------------")


# ===================================================================
#               Perfiles de intensidad del test USAF
# ===================================================================

# --- La idea es graficar los elementos de algunos grupos para saber hasta ---
# --- que punto se identifican las lineas horizontales haciendo el uso ---
# --- del criterio de resolucion de rayleight donde se evidencia en los ---
# --- perfiles de inetensidad hasta que punto se distinguen dos lineas en ---
# --- nuestro caso particular, basandonos en estos perifles podemos identificar ---
# --- que elementos si se resuelven o no, o hasta cual elemento lo hace bien --


def graficar_perfil_horizontal(nombre_archivo_imagen):

    img_color = cv2.imread(nombre_archivo_imagen)
    imagen_gris = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    
    # Promediamos horizontalmente (axis=1) para obtener un perfil vertical (vs. y)
    perfil_crudo = np.mean(imagen_gris, axis=1) 
    
    # ---  Suavizar el perfil ---
    sigma_suavizado = 20        #entre mayor es el numero mayor es el suavizado
    perfil_suave = gaussian_filter1d(perfil_crudo, sigma=sigma_suavizado)
    
    return perfil_crudo, perfil_suave, sigma_suavizado