import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import shannon_entropy
import sys

# --- Configuración ---
NOISY_IMG_PATH = 'Practicas/Practica_02/Actividad_1/Noise images/Noise (9).png'

# ¡Aquí defines los sigmas que quieres probar!
# He usado los de tu imagen de ejemplo.
SIGMAS_TO_TEST = [0.10, 0.30, 0.50, 0.70, 0.90, 1.20]

# Fracción del espectro que consideramos "altas frecuencias"
HF_FREQUENCY_RATIO = 0.25
# ---------------------

def load_image_or_exit(image_path):
    """Carga una imagen en escala de grises o termina el script si no la encuentra."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: No se pudo cargar la imagen en: {image_path}")
        print("Por favor, asegúrate de que el archivo exista en la misma carpeta que el script.")
        sys.exit(1)
    return image

def get_fft_spectrum(image):
    """Calcula el espectro de magnitud 2D centrado."""
    image_float = image.astype(np.float32)
    f = np.fft.fft2(image_float)
    fshift = np.fft.fftshift(f)
    # No necesitamos la imagen logarítmica, solo los datos complejos
    return fshift

def calculate_energy(fshift, hf_ratio, mode='high'):
    """Calcula la energía total en las altas o bajas frecuencias del espectro."""
    rows, cols = fshift.shape
    crow, ccol = rows // 2, cols // 2
    
    radius = int(min(crow, ccol) * hf_ratio)
    
    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    
    if mode == 'high':
        # Máscara de paso-alto (1 en altas frec, 0 en bajas frec)
        mask = x*x + y*y > radius*radius
    else: # mode == 'low'
        # Máscara de paso-bajo (0 en altas frec, 1 en bajas frec)
        mask = x*x + y*y <= radius*radius
    
    # Calcular la energía (Suma de |F(u,v)|^2)
    energy = np.sum(np.abs(fshift[mask])**2)
    return energy

# --- 1. Cargar Imagen Ruidosa y Analizarla ---
print("Analizando la imagen ruidosa base...")
noisy_img = load_image_or_exit(NOISY_IMG_PATH)
noisy_fshift = get_fft_spectrum(noisy_img)

# Calcular la energía de referencia de la imagen ruidosa
hf_energy_noisy = calculate_energy(noisy_fshift, HF_FREQUENCY_RATIO, mode='high')
lf_energy_noisy = calculate_energy(noisy_fshift, HF_FREQUENCY_RATIO, mode='low')
entropy_noisy = shannon_entropy(noisy_img)

print(f"  > Energía HF (Ruido) Base: {hf_energy_noisy:10.2e}")
print(f"  > Energía LF (Señal) Base: {lf_energy_noisy:10.2e}")
print(f"  > Entropía Base:           {entropy_noisy:10.2f}")

# --- 2. Bucle de Análisis de Filtros ---
print("\n--- Analizando Filtros Gaussianos ---")
hf_reductions = []
lf_losses = []
entropies = []
scores = []
sigma_labels = []

for sigma in SIGMAS_TO_TEST:
    print(f"Probando sigma = {sigma:.2f}...")
    
    # Aplicar filtro Gaussiano. (0, 0) deja que OpenCV elija el tamaño del kernel.
    filtered_img = cv2.GaussianBlur(noisy_img, (0, 0), sigma)
    
    # Calcular métricas para la imagen filtrada
    filtered_fshift = get_fft_spectrum(filtered_img)
    hf_energy_filtered = calculate_energy(filtered_fshift, HF_FREQUENCY_RATIO, mode='high')
    lf_energy_filtered = calculate_energy(filtered_fshift, HF_FREQUENCY_RATIO, mode='low')
    entropy_filtered = shannon_entropy(filtered_img)
    
    # Calcular porcentajes
    hf_reduction_pct = (1 - hf_energy_filtered / hf_energy_noisy) * 100
    lf_loss_pct = (1 - lf_energy_filtered / lf_energy_noisy) * 100
    
    # Calcular el "Puntaje de Eficiencia"
    # Queremos alta reducción de ruido (HF) y baja pérdida de señal (LF)
    # Por lo tanto, un buen puntaje es (Ruido Eliminado / Señal Perdida)
    # Se añade 1e-9 para evitar división por cero si la pérdida de LF es 0%
    score = hf_reduction_pct / (lf_loss_pct + 1e-9) 

    # Almacenar resultados
    hf_reductions.append(hf_reduction_pct)
    lf_losses.append(lf_loss_pct)
    entropies.append(entropy_filtered)
    scores.append(score)
    sigma_labels.append(f"σ={sigma:.2f}")

# Imprimir un resumen en la consola
print("\n--- RESUMEN DEL ANÁLISIS ---")
print(f"{'Sigma':<8} | {'% Red. Ruido (HF)':<18} | {'% Pérdida Señal (LF)':<20} | {'Entropía':<10} | {'Puntaje (Red/Pérdida)':<22}")
print("-" * 80)
for i, sigma in enumerate(SIGMAS_TO_TEST):
    print(f"{sigma:<8.2f} | {hf_reductions[i]:<18.1f} | {lf_losses[i]:<20.1f} | {entropies[i]:<10.2f} | {scores[i]:<22.2f}")

# Encontrar el mejor sigma según el puntaje
best_index = np.argmax(scores)
best_sigma = SIGMAS_TO_TEST[best_index]
print(f"\nMEJOR BALANCE (Puntaje más alto): sigma = {best_sigma:.2f}")
print("---------------------------------------------------------")

# --- 3. Visualización (Gráficas de Barras) ---

fig, axs = plt.subplots(3, 1, figsize=(12, 18), constrained_layout=True)
fig.suptitle(f'Análisis de Trade-Off: Filtro Gaussiano vs. Sigma (sobre {NOISY_IMG_PATH})', fontsize=16)
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sigma_labels)))

# Gráfica 1: Reducción de Ruido (HF)
axs[0].bar(sigma_labels, hf_reductions, color=colors)
axs[0].set_title('Reducción de Ruido (Energía de Alta Frecuencia)')
axs[0].set_ylabel('% de Energía HF Eliminada (Más es mejor)')
axs[0].set_xlabel('Valor de Sigma del Filtro')
axs[0].grid(axis='y', linestyle='--', alpha=0.7)
for i, val in enumerate(hf_reductions):
    axs[0].text(i, val + 1, f'{val:.1f}%', ha='center')

# Gráfica 2: Pérdida de Señal (LF)
axs[1].bar(sigma_labels, lf_losses, color=colors)
axs[1].set_title('Pérdida de Información (Energía de Baja Frecuencia)')
axs[1].set_ylabel('% de Energía LF Perdida (Menos es mejor)')
axs[1].set_xlabel('Valor de Sigma del Filtro')
axs[1].grid(axis='y', linestyle='--', alpha=0.7)
for i, val in enumerate(lf_losses):
    axs[1].text(i, val + 0.5, f'{val:.1f}%', ha='center')

# Gráfica 3: Puntaje de Eficiencia (Trade-off)
bar_colors = colors.copy() 
bar_colors = [tuple(c) for c in colors]
bar_colors[best_index] = 'red' # Resaltar la mejor barra en rojo

axs[2].bar(sigma_labels, scores, color=bar_colors)
axs[2].set_title('Puntaje de Eficiencia (Reducción de Ruido / Pérdida de Señal)')
axs[2].set_ylabel('Puntaje (Más es mejor)')
axs[2].set_xlabel('Valor de Sigma del Filtro')
axs[2].grid(axis='y', linestyle='--', alpha=0.7)
axs[2].text(best_index, scores[best_index] + 0.5, 'MEJOR', ha='center', fontweight='bold', color='red')

plt.show()