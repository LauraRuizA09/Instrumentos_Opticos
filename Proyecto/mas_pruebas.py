import numpy as np
import matplotlib.pyplot as plt

# 1. Configuración del espacio (La "Zona de Prueba")
width = 0.1  # Ancho en metros (ej. 10 cm)
height = 0.1 # Alto en metros
resolution = 500 # Píxeles por lado (calidad de la simulación)

x = np.linspace(-width/2, width/2, resolution)
y = np.linspace(-height/2, height/2, resolution)
X, Y = np.meshgrid(x, y)

# 2. Parámetros físicos del aire y el sonido
n0 = 1.00029  # Índice de refracción base del aire
lambda_sound = 0.02 # Longitud de onda del sonido (metros). 2cm aprox = 17kHz
amplitude = 0.00001 # Cambio en el índice de refracción (dn). Es muy pequeño en la realidad.

# 3. Modelado de la Onda de Sonido
# Teoría: mu(x) = mu0 - Delta_mu * sin(2*pi*x / lambda) 
# Vamos a crear una onda esférica emitida desde el centro (0,0)
r = np.sqrt(X**2 + Y**2) # Distancia al centro
k = 2 * np.pi / lambda_sound # Número de onda

# Creamos el campo de índices de refracción n(x,y)
# Usamos sin(k*r) para ondas concéntricas
n_field = n0 + amplitude * np.sin(k * r)

# 4. Visualización del campo de densidad (Lo que realmente está pasando físicamente)
plt.figure(figsize=(8, 6))
plt.imshow(n_field, extent=[-width/2, width/2, -height/2, height/2], cmap='viridis')
plt.colorbar(label='Índice de Refracción n(x,y)')
plt.title('Paso 1: Modelado de la Presión del Aire (Onda de Sonido)')
plt.xlabel('Metros')
plt.ylabel('Metros')
plt.show()