import numpy as np
import matplotlib.pyplot as plt
from Funciones import N_0, generar_onda_plana, plot_simulacion
from Funciones import onda_sonora_campo, schlieren_1M, generar_onda_esferica

# ===================================================================
#           Constantes fisicas y parametros de muestreo
# ===================================================================

# Parámetros de prueba
L_z = 0.3               # Espesor de la zona de prueba (30 cm)
lam = 633e-9        # Longitud de onda (633 nm)

# Parámetros de la simualcion de muestreo
L_x = 0.2            # 20 cm de ventana
L_y = 0.2
Nx = 1024            # Resolución
Ny = 1024
f = 1.2
R = 2 * f
# ==*=================================================================
#             Generar onda sonora como un campo con fase
# ===================================================================

S_campo, fase, n_map = onda_sonora_campo()

# Definir Onda Plana
#U_0 = generar_onda_plana(Nx,Ny)

#Onda esferica
U_0 = generar_onda_esferica(Nx,Ny,lam,R,L_x,L_y)

# ===================================================================
#    Visualización del cambio de fase y el indice de refraccion n
# ===================================================================

plt.figure(figsize=(12, 5))

# Gráfico 1: El índice de refracción (Física del aire)
plt.subplot(1, 2, 1)
plt.title(f"Índice de Refracción $n(x,y)$\nBase: {N_0:.6f}")
plt.imshow(n_map, cmap='gray')
plt.colorbar(label="n")

# Gráfico 2: La Fase (Óptica de Fourier)
plt.subplot(1, 2, 2)
plt.title(r"Carga de Fase $\phi(x,y)$ (Radianes)")
plt.imshow(fase, cmap='gray') 
plt.colorbar(label="Rad")

plt.tight_layout()
plt.show()

print(f"Campo S generado. Tipo: {S_campo.dtype}")
print(f"Fase máxima acumulada: {np.max(fase):.2f} radianes")

# ===================================================================
#                   Propagación sistema óptico
# ===================================================================

Imagen_sch_1M_V = schlieren_1M(U_0, lam, S_campo, "vertical")
Imagen_sch_1M_H = schlieren_1M(U_0, lam, S_campo, "horizontal")
Imagen_sch_1M_C = schlieren_1M(U_0, lam, S_campo, "circular")

# ===================================================================
#                         Resultados
# ===================================================================

#plot_simulacion(Imagen_sch_1M_V, "Imagen $SCHLIEREN$ 1 Espejo ($Sonido$) \n Knife Vertical", "gray")
#plot_simulacion(Imagen_sch_1M_H, "Imagen $SCHLIEREN$ 1 Espejo ($Sonido$) \n Knife Horizontal", "gray")
#plot_simulacion(Imagen_sch_1M_C, "Imagen $SCHLIEREN$ 1 Espejo ($Sonido$) \n Knife Circular", "gray")

# ===================================================================
#              Comparación difernetes cuchillas 
# ===================================================================

fig, axs = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle("Comparación de imagenes Schlieren (1 Espejo) con diferentes $Knifes$", fontsize=16)

# Datos comunes para recrear las máscaras visualmente
Nx, Ny = Imagen_sch_1M_V.shape
cx, cy = Nx // 2, Ny // 2

# ---------------- COLUMNA 1: CORTE VERTICAL ----------------
axs[0, 0].imshow(Imagen_sch_1M_V, cmap='gray')
axs[0, 0].set_title("Schlieren Vertical")
axs[0, 0].axis('off')

mask_v = np.ones((Nx, Ny))
mask_v[:, :cx] = 0 # Bloqueamos izquierda (0 = Negro)
axs[1, 0].imshow(mask_v, cmap='gray', vmin=0, vmax=1)
axs[1, 0].set_title("Filtro: Vertical")
axs[1, 0].axis('off')
# Añadimos borde negro fino para que se note el cuadro blanco
for spine in axs[1,0].spines.values(): spine.set_edgecolor('black'); spine.set_linewidth(1)


# ---------------- COLUMNA 2: CORTE HORIZONTAL ----------------
axs[0, 1].imshow(Imagen_sch_1M_H, cmap='gray')
axs[0, 1].set_title("Schlieren Horizontal")
axs[0, 1].axis('off')

mask_h = np.ones((Nx, Ny))
mask_h[:cy, :] = 0 # Bloqueamos abajo (0 = Negro)
axs[1, 1].imshow(mask_h, cmap='gray', vmin=0, vmax=1)
axs[1, 1].set_title("Filtro: Horizontal")
axs[1, 1].axis('off')


# ---------------- COLUMNA 3: CORTE CIRCULAR ----------------
axs[0, 2].imshow(Imagen_sch_1M_C, cmap='gray')
axs[0, 2].set_title("Schlieren Circular")
axs[0, 2].axis('off')

mask_c = np.ones((Nx, Ny))
y_g, x_g = np.ogrid[:Nx, :Ny]
mask_c[(x_g - cx)**2 + (y_g - cy)**2 < 20**2] = 0 # Bloqueamos punto central

# Hacemos zoom al centro porque el punto es muy pequeño
zoom = 100 
mask_c_zoom = mask_c[cx-zoom:cx+zoom, cy-zoom:cy+zoom]

axs[1, 2].imshow(mask_c_zoom, cmap='gray', vmin=0, vmax=1)
axs[1, 2].set_title("Filtro: Circular")
axs[1, 2].axis('off')

plt.tight_layout()
plt.show()

















