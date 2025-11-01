import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import estimate_sigma
from skimage.measure import shannon_entropy
import sys

# ===================================================================
#                Cargar datos e imágenes
# ===================================================================

NOISY_IMG_PATH = 'Practicas/Practica_02/Actividad_1/Datos Experimentales/Imagen SIN Filtrar.png'
FILTERED_IMG_PATH = 'Practicas/Practica_02/Actividad_1/Datos Experimentales/Imagen Filtrada.png'

# Fracción del espectro que consideramos "altas frecuencias"
# 0.15 = el 15% central es baja frecuencia, el resto es alta.
HF_FREQUENCY_RATIO = 0.5

# ===================================================================
#                Funciones a utilizar
# ===================================================================

def load_image_or_exit(image_path):

    #Carga una imagen en escala de grises o termina el script si no la encuentra
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: No se pudo cargar la imagen en: {image_path}")
        print("Por favor, asegúrate de que el archivo exista en la misma carpeta que el script.")
        sys.exit(1)
    return image

def get_fft_spectrum(image):

    # Calcula el espectro de magnitud 2D centrado y en escala logarítmica
    # Convertir a float para la FFT
    image_float = image.astype(np.float32)
    
    # Aplicar FFT y centrar (shift) 
    f = np.fft.fft2(image_float)
    fshift = np.fft.fftshift(f)
    
    # Calcular el espectro de magnitud para visualización (logarítmico)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1) # +1 para evitar log(0)
    
    return fshift, magnitude_spectrum

def calculate_hf_energy(fshift, hf_ratio):

    #Calcula la energía total en las altas frecuencias del espectro
    rows, cols = fshift.shape
    crow, ccol = rows // 2, cols // 2
    
    # Definir el radio de la banda de bloqueo (bajas frecuencias)
    radius = int(min(crow, ccol) * hf_ratio)
    
    # Crear una máscara circular (1 en altas frec, 0 en bajas frec)
    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    mask = x*x + y*y > radius*radius
    
    # Calcular la energía de alta frecuencia (Suma de |F(u,v)|^2)
    hf_energy = np.sum(np.abs(fshift[mask])**2)
    return hf_energy

# ===================================================================
#                     Cargar imágenes
# ===================================================================

noisy_img = load_image_or_exit(NOISY_IMG_PATH)
filtered_img = load_image_or_exit(FILTERED_IMG_PATH)

# ===================================================================
#                Análisis Cuantitativo
# ===================================================================

#Análisis Espectral (FFT)
print("\n[MÉTODO: Energía de Alta Frecuencia (Global)]")
noisy_fshift, noisy_spec_img = get_fft_spectrum(noisy_img)
filtered_fshift, filtered_spec_img = get_fft_spectrum(filtered_img)

hf_energy_noisy = calculate_hf_energy(noisy_fshift, HF_FREQUENCY_RATIO)
hf_energy_filtered = calculate_hf_energy(filtered_fshift, HF_FREQUENCY_RATIO)
hf_reduction = (1 - hf_energy_filtered / hf_energy_noisy) * 100
print(f"  > Energía HF (Ruidosa):   {hf_energy_noisy:10.2e}")
print(f"  > Energía HF (Filtrada):  {hf_energy_filtered:10.2e}")
print(f"  > Reducción de energía HF:  {hf_reduction:10.1f}%")



# ===================================================================
#                Graficar Resultados Comparativos
# ===================================================================

fig = plt.figure(figsize=(16, 12)) # Ajustar tamaño (más alto)
gs = fig.add_gridspec(2, 4)
fig.suptitle('Comparación Cuantitativa de Reducción de Ruido', fontsize=20, y=0.98)

ax1 = fig.add_subplot(gs[0, 1:3]) 
ax2 = fig.add_subplot(gs[1, 0:2])
ax3 = fig.add_subplot(gs[1, 2:4])


# ¿Energía de Alta Frecuencia (FFT) - va en ax1
ax1.bar(
    ['Imagen Ruidosa', 'Imagen Filtrada'], 
    [hf_energy_noisy, hf_energy_filtered], 
    color=["#1D20D3", "#0EB31C"],
    width=0.6
)
ax1.set_title('Método: Energía de Alta Frecuencia ', fontsize=14)
ax1.set_ylabel('Energía Total (Escala Log)', fontsize=12)
ax1.set_yscale('log') # La escala logarítmica es clave aquí
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax1.text(1, hf_energy_filtered, f'{hf_reduction:.1f}%\nReducción', 
               ha='center', va='bottom', fontsize=12, color='black', fontweight='bold')

#Espectro FFT (Ruidosa) 
im1 = ax2.imshow(noisy_spec_img, cmap='hot', 
                       vmax=max(noisy_spec_img.max(), filtered_spec_img.max()),
                       vmin=min(noisy_spec_img.min(), filtered_spec_img.min()))
ax2.set_title('Espectro de Frecuencias $S(ξ,η)$', fontsize=14)
ax2.set_xticks([])
ax2.set_yticks([])
fig.colorbar(im1, ax=ax2, orientation='horizontal', pad=0.1, fraction=0.046)

# Dibujar el círculo que separa altas/bajas frecuencias
radius = int(min(noisy_img.shape[0]//2, noisy_img.shape[1]//2) * HF_FREQUENCY_RATIO)
circle = plt.Circle((noisy_img.shape[1]//2, noisy_img.shape[0]//2), radius, color='black', 
                    fill=False, linestyle='--', linewidth=2, alpha=0.7)
ax2.add_patch(circle)


# Espectro FFT (Filtrada)
im2 = ax3.imshow(filtered_spec_img, cmap='hot',
                       vmax=max(noisy_spec_img.max(), filtered_spec_img.max()),
                       vmin=min(noisy_spec_img.min(), filtered_spec_img.min()))
ax3.set_title('Espectro de Frecuencias ($U(x´,y´)$)', fontsize=14)
ax3.set_xticks([])
ax3.set_yticks([])
fig.colorbar(im2, ax=ax3, orientation='horizontal', pad=0.1, fraction=0.046)

# Dibujar el círculo
radius = int(min(filtered_img.shape[0]//2, filtered_img.shape[1]//2) * HF_FREQUENCY_RATIO)
circle = plt.Circle((filtered_img.shape[1]//2, filtered_img.shape[0]//2), radius, color='black', 
                    fill=False, linestyle='--', linewidth=2, alpha=0.7)
ax3.add_patch(circle)

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajustar para el supertítulo


plt.savefig('Practicas/Practica_02/Actividad_1/Correlaciones/filtrado_experimental.png')
plt.show()