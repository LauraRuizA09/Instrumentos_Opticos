import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
# Eliminadas las importaciones de skimage que ya no se usan

# --- Configuración ---
NOISY_IMG_PATH = 'Practicas/Practica_02/Actividad_1/Noise images/Noise (9).png'

# Define aquí TODAS las imágenes filtradas que quieres comparar
# Es un diccionario que mapea la etiqueta (el sigma) a la ruta del archivo
FILTERED_IMAGES = {
    "σ=0.02": 'Practicas/Practica_02/Actividad_1/Variacion Sigma/sigma_0.02.png',
    "σ=0.10": 'Practicas/Practica_02/Actividad_1/Variacion Sigma/sigma_0.1.png',
    "σ=0.30": 'Practicas/Practica_02/Actividad_1/Variacion Sigma/CORE_FILTR.png',
    "σ=0.50": 'Practicas/Practica_02/Actividad_1/Variacion Sigma/sigma_0.5.png',
    "σ=0.70": 'Practicas/Practica_02/Actividad_1/Variacion Sigma/sigma_0.7.png'
    # NOTA: Puedes añadir las de 0.90 y 1.20 si las tienes
}

# Fracción del espectro que consideramos "altas frecuencias"
HF_FREQUENCY_RATIO = 0.25
# ---------------------

def load_image_or_exit(image_path):
    """Carga una imagen en escala de grises o termina el script si no la encuentra."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: No se pudo cargar la imagen en: {image_path}")
        print("Por favor, asegúrate de que el archivo exista.")
        sys.exit(1)
    return image

def get_fft_spectrum(image):
    """Calcula el espectro de magnitud 2D centrado."""
    image_float = image.astype(np.float32)
    f = np.fft.fft2(image_float)
    fshift = np.fft.fftshift(f)
    return fshift

def calculate_hf_energy(fshift, hf_ratio):
    """Calcula la energía total en las altas frecuencias del espectro."""
    rows, cols = fshift.shape
    crow, ccol = rows // 2, cols // 2
    
    radius = int(min(crow, ccol) * hf_ratio)
    
    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    
    # Máscara de paso-alto (1 en altas frec, 0 en bajas frec)
    mask = x*x + y*y > radius*radius
    
    # Calcular la energía de alta frecuencia (Suma de |F(u,v)|^2)
    energy = np.sum(np.abs(fshift[mask])**2)
    return energy

# --- 1. Cargar Imagen Ruidosa y Analizarla (Solo una vez) ---
print("--- ANÁLISIS CUANTITATIVO DE FILTROS GAUSSIANOS ---")
noisy_img = load_image_or_exit(NOISY_IMG_PATH)
noisy_fshift = get_fft_spectrum(noisy_img)

# Calcular la energía de referencia de la imagen ruidosa
hf_energy_noisy = calculate_hf_energy(noisy_fshift, HF_FREQUENCY_RATIO)
print(f"  > Energía HF (Ruido) Base: {hf_energy_noisy:10.2e}")


# Listas para guardar los resultados para las gráficas
results_hf_reduction = []
result_labels = [] # Para las etiquetas de las barras (e.g., "σ=0.10")

for sigma_label, img_path in FILTERED_IMAGES.items():
    
    filtered_img = load_image_or_exit(img_path)
    filtered_fshift = get_fft_spectrum(filtered_img)
    
    # Calcular métricas para la imagen filtrada
    hf_energy_filtered = calculate_hf_energy(filtered_fshift, HF_FREQUENCY_RATIO)
    
    # Calcular porcentajes
    hf_reduction_pct = (1 - hf_energy_filtered / hf_energy_noisy) * 100
    
    # Almacenar resultados
    results_hf_reduction.append(hf_reduction_pct)
    result_labels.append(sigma_label)

# --- Impresión de Tabla de Resumen ---
print("\n--- ANÁLISIS ---")
print(f"{'Filtro':<8} | {'Porcentaje de reducción de ruido':<18}")
print("-" * 30)
for i, label in enumerate(result_labels):
    print(f"{label:<8} | {results_hf_reduction[i]:<18.1f}")
print("---------------------------------")


# --- 3. Visualización (Solo la gráfica de barras de Reducción de Ruido) ---

# Crear 1 sola gráfica
fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
fig.suptitle(f'Efectividad del Filtro', fontsize=16)

# Generar colores para las barras
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(result_labels)))

# Gráfica: Reducción de Ruido (HF)
ax.bar(result_labels, results_hf_reduction, color=colors, width=0.6)
ax.set_title('Reducción de Ruido (Energía de Alta Frecuencia)')
ax.set_ylabel('% de Energía Eliminada ')
ax.set_xlabel('Filtro Aplicado (Sigma)')
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Añadir el texto del porcentaje sobre cada barra
for i, val in enumerate(results_hf_reduction):
    # Ajustar la posición vertical del texto para que no se salga
    y_pos = max(val - 5, 0) if val < 0 else val + 1 
    ax.text(i, y_pos, f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.savefig('Practicas/Practica_02/Actividad_1/Correlaciones/sigma_mascara.png')
plt.show()