import matplotlib.pyplot as plt
import numpy as np
from Funciones import mapeo, generar_onda_sonora, calcular_gradientes, plot_simulacion
from Funciones import simular_camino_completo, simular_corte_cuchilla, calcular_desviacion_angular, simular_z_type_dos_espejos

# ===================================================================
#            Generamos la onda de sonido
# ===================================================================

#-------Datos del sistema fisico-------

# - 20000 Hz (20 kHz) = Ultrasonido bajo (límite auditivo humano)
# - 40000 Hz (40 kHz) = Ultrasonido estándar (transductores comunes)
# - 1000 Hz  (1 kHz)  = Sonido agudo audible (se verán pocas ondas muy grandes)

frecuencia_generador_hz = 40000
velocidad_sonido = 343    # m/s (en aire a 20°C)

# Calcular Longitud de Onda (lambda = v / f)
longitud_onda = velocidad_sonido / frecuencia_generador_hz

# Calcular Frecuencia Espacial  (1 / lambda)
f_onda = 1.0 / longitud_onda

f_onda = 80                 # Frecuencia visual
Amplitud = 0.005             # Intensidad de la onda, me dice que tanto esta cambiando n
                            # dejamos este valor que es exagerado para una mejor visualizacion el real es mucho mas bajo
                            # 10e-6 seria le cmabio del indice de refraccion


X, Y, dx = mapeo(resolucion=800, tamano_fisico=0.3)
n_map = generar_onda_sonora(X, Y, f_onda, Amplitud)
dndx, dndy = calcular_gradientes(n_map, dx)
eps_x, eps_y = calcular_desviacion_angular(dndx, dndy, espesor_z=0.3)


# ===================================================================
#          Simulacion de la configuración Z-type
# ===================================================================

focal_m2 = 1.5           # Espejo 2 de enfoque (f=1.5m)
distancia_objeto = 1     # Objeto a 1 metro del espejo 2
radio_focal = 0.0005  #sensibilidad de que tanto esta dispersada la luz 

# Calculamos trayectoria
desp_x_z, desp_y_z, factor_S_z = simular_z_type_dos_espejos(eps_x, eps_y, distancia_objeto, focal_m2)

print(f"Sensibilidad Z-Type: {factor_S_z:.4f} m/rad")
print(f"En un sistema Z ideal, la sensibilidad debería ser igual a la focal ({focal_m2} m).")

# Generar imagen
img_ztype = simular_corte_cuchilla(desp_x_z, desp_y_z, "horizontal", radio_focal)

plot_simulacion(img_ztype, f"Schlieren Z-Type (Dos Espejos f={focal_m2}m)", cmap='inferno')


# ===================================================================
#                   Comparación cuchillas
# ===================================================================

img_horizontal = simular_corte_cuchilla(desp_x_z, desp_y_z, "horizontal", radio_focal)
img_vertical   = simular_corte_cuchilla(desp_x_z, desp_y_z, "vertical", radio_focal)
img_circular   = simular_corte_cuchilla(desp_x_z, desp_y_z, "circular", radio_focal)

# Creamos la figura
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
plt.subplots_adjust(hspace=0.3, wspace=0.1)

# [0,0] Horizontal (Gradientes en Y)
axes[0, 0].imshow(img_horizontal, cmap='inferno', extent=[-15,15,-15,15])
axes[0, 0].set_title("Cuchilla Horizontal")
axes[0, 0].set_ylabel("Imagen Schlieren", fontsize=12, fontweight='bold')

# [0,1] Vertical (Gradientes en X)
axes[0, 1].imshow(img_vertical, cmap='inferno', extent=[-15,15,-15,15])
axes[0, 1].set_title("Cuchilla Vertical")

# [0,2] Circular (Campo Oscuro)
# Usamos 'inferno' para que resalte como en la imagen de referencia
axes[0, 2].imshow(img_circular, cmap='inferno', extent=[-15,15,-15,15])
axes[0, 2].set_title("Filtro Circular")


# Creamos arrays simples de 100x100 para dibujar los cuadrados blanco/negro

# -- Esquema Horizontal --
esquema_h = np.ones((100, 100))
esquema_h[50:, :] = 0 # Mitad de abajo negra
axes[1, 0].imshow(esquema_h, cmap='gray', vmin=0, vmax=1)
axes[1, 0].scatter([50], [50], c='red', s=30) # Punto rojo = Foco ideal
axes[1, 0].set_title("Forma del Filtro")
axes[1, 0].set_ylabel("Esquema Físico", fontsize=12, fontweight='bold')

# -- Esquema Vertical --
esquema_v = np.ones((100, 100))
esquema_v[:, :50] = 0 # Mitad izquierda negra
axes[1, 1].imshow(esquema_v, cmap='gray', vmin=0, vmax=1)
axes[1, 1].scatter([50], [50], c='red', s=30) # Punto rojo = Foco ideal
axes[1, 1].set_title("Forma del Filtro")

# -- Esquema Circular --
esquema_c = np.ones((100, 100))
y_grid, x_grid = np.ogrid[:100, :100]
centro = (50, 50)
radio_esquema = 15
mascara_circulo = (x_grid - centro[0])**2 + (y_grid - centro[1])**2 <= radio_esquema**2
esquema_c[mascara_circulo] = 0 # Centro negro
axes[1, 2].imshow(esquema_c, cmap='gray', vmin=0, vmax=1)
axes[1, 2].set_title("Forma del Filtro")


plt.suptitle("Comparación de Filtros (Knife) Schlieren ($2$ $Espejos$ $Z-type$)", fontsize=16)
plt.show()