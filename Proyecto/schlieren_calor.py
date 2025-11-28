import matplotlib.pyplot as plt
import numpy as np
from Funciones import mapeo, generar_ruido_fractal, calcular_gradientes, plot_simulacion, generar_columna_calor
from Funciones import simular_camino_completo, simular_corte_cuchilla, calcular_desviacion_angular


# ===================================================================
#                   Parametros Fisicos
# ===================================================================

# Intensidad del calor
# 80°C sobre el ambiente 
delta_temp_K = 80.0  

# Ancho de la fuente de calor
ancho_fuente = 0.02  # 2 cm de ancho

# Configuración del espacio (Espejo de 30 cm)
resolucion = 800
tamano_fisico_m = 0.3

# ===================================================================
#                   Simular columna de calor
# ===================================================================

X, Y, dx = mapeo(resolucion, tamano_fisico_m)
n_map = generar_columna_calor(X, Y, delta_temp_K, ancho_fuente)
dndx, dndy = calcular_gradientes(n_map, dx)
eps_x, eps_y = calcular_desviacion_angular(dndx, dndy, espesor_z=0.1) # Asumimos flama de 10cm prof.

# ===================================================================
#                   Parametros distancias
# ===================================================================

focal = 1.0
posicion_objeto = 0.5 # Objeto a 0.5m del espejo
desp_x, desp_y, factor_S = simular_camino_completo(eps_x, eps_y, posicion_objeto, focal)

# ===================================================================
#                       Visualización
# ===================================================================

radio_focal_val = 0.001 # 1mm (El calor es fuerte, bajamos sensibilidad subiendo el radio)

img_vert = simular_corte_cuchilla(desp_x, desp_y, "vertical", radio_focal_val)
img_horiz = simular_corte_cuchilla(desp_x, desp_y, "horizontal", radio_focal_val)

fig, axes = plt.subplots(1, 3, figsize=(15,8))

axes[0].imshow(n_map, cmap='gray', extent=[-15,15,-15,15])
axes[0].set_title("Campo de Temperatura")

axes[1].imshow(img_vert, cmap='gray', extent=[-15,15,-15,15], vmin=0, vmax=1)
axes[1].set_title("Simulación Schlieren (Cuchilla Vertical)")

axes[2].imshow(img_horiz, cmap='gray', extent=[-15,15,-15,15], vmin=0, vmax=1)
axes[2].set_title("Simulación Schlieren (Cuchilla Horizontal)")


plt.show()