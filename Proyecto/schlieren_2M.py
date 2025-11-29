import numpy as np
import matplotlib.pyplot as plt
from Funciones import mapeo, generar_campo_entrada_S, generar_onda_sonora, N_0, generar_onda_plana, plot_simulacion, propagar_ABCD_
from Funciones import aplicar_filtro_cuchilla

# ===================================================================
#           Constantes fisicas y parametros de muestreo
# ===================================================================

# Parámetros de prueba
L_z = 0.3           # Espesor de la zona de prueba (30 cm)
lam = 633e-9        # Longitud de onda (633 nm)

# Parámetros de la simualcion de muestreo
L_x = 0.2           # 20 cm de ventana
L_y = 0.2
Nx = 1024           # Resolución
Ny = 1024

# ===================================================================
#                    Generamos la onda de sonido
# ===================================================================

frecuencia_generador_hz = 40000
velocidad_sonido = 343    # m/s (en aire a 20°C)

# Calcular Longitud de Onda
longitud_onda = velocidad_sonido / frecuencia_generador_hz

# Calcular Frecuencia Espacial
f_onda = 1.0 / longitud_onda

f_onda = 80                 # Frecuencia visual ajustada para ver detalles
Amplitud = 0.005            # Intensidad de la onda (cambio de n)

X, Y, dx, dy = mapeo(Nx, Ny, L_x)
n_map = generar_onda_sonora(X, Y, f_onda, Amplitud)

# ===================================================================
#             Generar onda sonora como un campo con fase
# ===================================================================

S_campo, fase = generar_campo_entrada_S(n_map, lam, L_z)

# Definir Onda Plana
# EN SISTEMA Z-TYPE: U_0 representa la luz que YA salió del primer espejo.
# Es decir, U_0 es el haz colimado (paralelo) que entra a la zona de prueba.
U_0 = generar_onda_plana(Nx,Ny)

# ===================================================================
#    Visualización del cambio de fase y el indice de refraccion n
# ===================================================================

plt.figure(figsize=(12, 5))

# Gráfico 1: El índice de refracción
plt.subplot(1, 2, 1)
plt.title(f"Índice de Refracción $n(x,y)$\nBase: {N_0:.6f}")
plt.imshow(n_map, cmap='gray')
plt.colorbar(label="n")

# Gráfico 2: La Fase
plt.subplot(1, 2, 2)
plt.title(r"Carga de Fase $\phi(x,y)$ (Radianes)")
plt.imshow(fase, cmap='gray') 
plt.colorbar(label="Rad")

plt.tight_layout()
plt.show()

print(f"Campo S generado. Tipo: {S_campo.dtype}")
print(f"Fase máxima acumulada: {np.max(fase):.2f} radianes")

# ===================================================================
#             Propagación sistema óptico (CONFIGURACIÓN 2 ESPEJOS)
# ===================================================================

# Datos fisicos del segundo espejo (El de enfoque)
f2 = 1.0              # Distancia focal del Espejo 2
d_foco = f2           # Distancia al plano de la cuchilla
R2 = 2 * f2           # Radio de curvatura del Espejo 2
k = 2 * np.pi / lam   # Numero de onda 

# --- Definición del recorrido (Z-Type) ---

# PASO 1: Zona de Prueba (Haz Paralelo)
# En el sistema Z, la luz viaja paralela entre el Espejo 1 y el Espejo 2.
# El objeto (S_campo) perturba esta luz plana directamente.
# Nota: Ignoramos la difracción en el espacio libre dentro de la zona de prueba 
# por ser pequeña (L_z) comparada con f2, asumiendo "Objeto Delgado".

camp_zona_prueba = U_0 * S_campo

# PASO 2: Interacción con el Espejo 2 (Enfoque)
# El haz perturbado golpea el segundo espejo parabólico.
# Esto añade la curvatura necesaria para converger la luz.
# Usamos "espejo" con R2.
camp_espejo2, _, _, _, _ = propagar_ABCD_(camp_zona_prueba, "espejo", 0, R2, lam, k)

# PASO 3: Propagación al Foco (Cuchilla)
# La luz viaja desde el Espejo 2 hasta su punto focal (distancia f2).
# Aquí el haz se reduce a un punto (transformada de Fourier).
camp_plano_focal, _, _, _, _ = propagar_ABCD_(camp_espejo2, "propagar", d_foco, 0, lam, k)

# PASO 4: Aplicamos el filtro de la cuchilla
# Cortamos en el plano de Fourier.
S_filtred = aplicar_filtro_cuchilla(camp_plano_focal, "horizontal")

# PASO 5: Cámara (Transformada Inversa)
# La lente de la cámara reconstruye la imagen final desde el plano filtrado.
campo_en_sensor = np.fft.fftshift(np.fft.ifft2(S_filtred))

# Calculamos la intensidad 
Imagen_Schlieren = np.abs(campo_en_sensor)**2

# Visualización Final
plot_simulacion(Imagen_Schlieren, "Imagen $SCHLIEREN$ Z-Type (2 Espejos)", "gray")