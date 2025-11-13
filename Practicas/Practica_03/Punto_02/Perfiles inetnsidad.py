import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter1d # <--- LIBRERÍA DE SUAVIZADO

# --- Cargar la imagen .npy principal ---
ruta_del_resultado = "Practicas/Practica_03/Punto_01/resultado_microscopio.npy"
I_final_camara = np.load(ruta_del_resultado)

# --- 1. Visualización de la imagen de entrada completa ---
plt.figure(figsize=(8, 6))
plt.imshow(I_final_camara, cmap='gray')
plt.colorbar(label='Intensidad')
plt.title("Imagen Resultante en la Cámara")
plt.xlabel("Píxeles (x)")
plt.ylabel("Píxeles (y)")
print("Mostrando imagen de entrada principal... Cierra la ventana para continuar.")
plt.show()


# --- PARTE 2: Función de Análisis y Graficación de Perfil ---

def graficar_perfil_horizontal(nombre_archivo_imagen):
    """
    Carga una imagen, calcula el perfil de intensidad horizontal
    (promediando filas) y devuelve ambos perfiles (crudo y suave).
    """
    img_color = cv2.imread(nombre_archivo_imagen)
    imagen_gris = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    
    # --- Lógica de Perfil para LÍNEAS HORIZONTALES ---
    # Promediamos horizontalmente (axis=1) para obtener un perfil vertical (vs. y)
    perfil_crudo = np.mean(imagen_gris, axis=1) 
    
    # --- 1. Suavizar el perfil ---
    # Puedes hacer este número MÁS GRANDE para MÁS SUAVIZADO
    sigma_suavizado = 15
    perfil_suave = gaussian_filter1d(perfil_crudo, sigma=sigma_suavizado)
    
    return perfil_crudo, perfil_suave

# --- PARTE 3: LÓGICA PRINCIPAL (Iterar y Graficar) ---

print("\n" + "="*70)
print("--- Generador de Perfiles de Intensidad para LÍNEAS HORIZONTALES ---")

# --- LAURA: DEFINE AQUÍ TUS IMÁGENES Y SU NÚMERO DE ELEMENTO ---
imagenes_y_elementos = [
    ("Practicas/Practica_03/Punto_02/LImites Resolucion (Cualit.)/G0E4.png", 4),
    ("Practicas/Practica_03/Punto_02/LImites Resolucion (Cualit.)/G0E5.png", 5),
    ("Practicas/Practica_03/Punto_02/LImites Resolucion (Cualit.)/G0E6.png", 6),
    ("Practicas/Practica_03/Punto_02/LImites Resolucion (Cualit.)/G1E1.png", 1),
    ("Practicas/Practica_03/Punto_02/LImites Resolucion (Cualit.)/G1E2.png", 2),
]
# -----------------------------------------------------------------

# Iterar, analizar y MOSTRAR CADA GRÁFICO
for filename, element_num in imagenes_y_elementos:
    
    print(f"\n--- Procesando: '{filename}' (Elemento {element_num}) ---")
    
    # 1. Calcular perfiles
    perfil_crudo, perfil_suave = graficar_perfil_horizontal(filename)
    
    # 2. Graficar el perfil (verticalmente)
    plt.figure(figsize=(8, 6)) # Más alto para un perfil vertical
    
    # El eje Y es la posición en píxeles, el eje X es la intensidad
    eje_y = range(len(perfil_crudo))
    
    # Graficamos ambos perfiles
    plt.plot(perfil_crudo, eje_y, 'k-', alpha=0.3, label='Perfil Crudo (Promedio Horizontal)')
    plt.plot(perfil_suave, eje_y, 'c-', lw=2, label=f'Perfil Suavizado (sigma={3.0})')

    plt.xlabel("Intensidad Promediada (0-255)")
    plt.ylabel("Posición Y (píxeles)")
    
    plt.title(f"Perfil de Intensidad (para Líneas Horizontales) - Elemento {element_num}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Invertir el eje Y para que coincida con la visualización de la imagen (opcional)
    # plt.gca().invert_yaxis() 
    
    print("  Mostrando gráfico de perfil... Cierra la ventana para continuar.")
    plt.show() 

print("\n" + "="*70)
print("Análisis de perfiles completado.")
print("="*70)