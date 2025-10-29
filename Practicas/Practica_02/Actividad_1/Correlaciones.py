import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 1. Cargar imágenes (asumimos que 'f' es la imagen y 'h' es la plantilla)
# (Aquí usamos datos de ejemplo)
f = np.array(Image.open("ruta/a/tu/imagen_principal.png").convert('L'))
h = np.array(Image.open("ruta/a/tu/plantilla.png").convert('L'))

# 2. Asegurar el mismo tamaño (Padding)
shape_f = f.shape
shape_h = h.shape

# Crear una matriz 'h' del tamaño de 'f' con la plantilla en la esquina
h_padded = np.zeros_like(f, dtype=float)
h_padded[0:shape_h[0], 0:shape_h[1]] = h
# Es importante centrar la plantilla si su origen importa, 
# pero para la detección de picos simple, esto funciona.

# 3. Calcular FFTs
F = np.fft.fft2(f)
H = np.fft.fft2(h_padded) # Usar la versión con padding

# 4. Calcular el filtro acoplado (conjugado)
H_conj = np.conj(H)

# 5. Multiplicar en el dominio de Fourier
R = F * H_conj

# 6. IFFT para obtener el plano de correlación
r = np.fft.ifft2(R)

# 7. El resultado 'r' es complejo. La intensidad de correlación es la magnitud al cuadrado.
# A menudo usamos fftshift para centrar el pico de autocorrelación (lag=0) en el medio.
correlation_plane = np.fft.fftshift(np.abs(r)**2)

# 8. Encontrar la ubicación del pico
y, x = np.unravel_index(np.argmax(correlation_plane), correlation_plane.shape)
peak_value = correlation_plane[y, x]

print(f"Pico de correlación encontrado en (y, x): ({y}, {x}) con valor {peak_value:.2e}")

# --- Graficar ---
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

ax[0].imshow(f, cmap='gray')
ax[0].set_title('Imagen $f(x, y)$')

ax[1].imshow(h, cmap='gray')
ax[1].set_title('Plantilla $h(x, y)$')

im = ax[2].imshow(correlation_plane, cmap='hot')
ax[2].plot(x, y, 'g+', markersize=10) # Marcar el pico
ax[2].set_title('Plano de Correlación $|r(x, y)|^2$')
fig.colorbar(im, ax=ax[2], label='Intensidad de Correlación')

plt.tight_layout()
plt.show()