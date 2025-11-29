import numpy as np
import matplotlib.pyplot as plt
from Funciones import mapeo, generar_campo_entrada_S, generar_onda_sonora, N_0, generar_onda_plana, plot_simulacion, propagar_ABCD_
from Funciones import aplicar_filtro_cuchilla

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

# ===================================================================
#                    Generamos la onda de sonido
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


X, Y, dx, dy = mapeo(Nx, Ny, L_x)
n_map = generar_onda_sonora(X, Y, f_onda, Amplitud)


# ===================================================================
#             Generar onda sonora como un campo con fase
# ===================================================================

S_campo, fase = generar_campo_entrada_S(n_map, lam, L_z)

# Definir Onda Plana
U_0 = generar_onda_plana(Nx,Ny)

# ===================================================================
#             Propagación sistema óptico
# ===================================================================

# Datos fisicos del sistema en m
f = 1               #distancia focal del espejo
d = f             #distancia de propagacion
R = 2*f
k = 2 * np.pi / lam # Numero de onda 

#Definición del recorrido

#De la fuente -> espejo
#S1_campo, S1_x_mesh, S1_y_mesh, S1_dx, S1_dy = propagar_ABCD_(U_0,"propagar", d, 0, lam,k)

#Interaccion con el espejo
S3_campo, S3_x_mesh, S3_y_mesh, S3_dx, S3_dy = propagar_ABCD_(U_0, "espejo", 0, R, lam,k)

#Multiplicoc por el objeto como si fuera una trasnmitancia
camp1 = S3_campo * S_campo

#Del objeto -> cuchilla
S5_campo, S5_x_mesh, S5_y_mesh, S5_dx, S5_dy = propagar_ABCD_(camp1, "propagar", d, 0, lam,k)

plot_simulacion(np.abs(S5_campo)**2, "Campo", "gray")

#Aplicamos el filtro de la cuchilla
S_filtred = aplicar_filtro_cuchilla(S5_campo,"horizontal")

plot_simulacion(np.abs(S_filtred)**2, "Cuchilla", "gray")


# Aplicamos la Transformada Inversa (La lente formadora de imagen)
campo_en_sensor = np.fft.fftshift(np.fft.ifft2(S_filtred))

# Calculamos la intensidad (lo que ve el ojo/cámara)
Imagen_Schlieren = np.abs(campo_en_sensor)**2

plot_simulacion(Imagen_Schlieren, "Imagen $SCHLIEREN$ 1 Espejo ($Sonido$)", "gray")
























# --- Visualización ---
plt.figure(figsize=(12, 5))

# Gráfico 1: El índice de refracción (Física del aire)
plt.subplot(1, 2, 1)
plt.title(f"Índice de Refracción $n(x,y)$\nBase: {N_0:.6f}")
plt.imshow(n_map, cmap='gray')
plt.colorbar(label="n")

# Gráfico 2: La Fase (Óptica de Fourier)
# Esto es lo que 'verá' la transformada de Fourier en el siguiente paso
plt.subplot(1, 2, 2)
plt.title(r"Carga de Fase $\phi(x,y)$ (Radianes)")
plt.imshow(fase, cmap='gray') # RdBu ayuda a ver compresiones (rojo) vs rarefacciones (azul)
plt.colorbar(label="Rad")

plt.tight_layout()
plt.show()

print(f"Campo S generado. Tipo: {S_campo.dtype}")
print(f"Fase máxima acumulada: {np.max(fase):.2f} radianes")