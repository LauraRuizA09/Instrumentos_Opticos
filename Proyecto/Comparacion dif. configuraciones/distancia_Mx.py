import sys
import os

# 1. Obtenemos la ruta absoluta de la carpeta donde está ESTE archivo nuevo
directorio_actual = os.path.dirname(os.path.abspath(__file__))

# 2. Obtenemos la ruta del "directorio padre" (la carpeta de atrás/arriba)
directorio_padre = os.path.dirname(directorio_actual)

# 3. Agregamos el directorio padre a la lista de rutas donde Python busca módulos
sys.path.append(directorio_padre)


import numpy as np
import matplotlib.pyplot as plt
from Funciones import mapeo, generar_campo_entrada_S, generar_onda_sonora, N_0, generar_onda_plana, plot_simulacion, propagar_ABCD_
from Funciones import aplicar_filtro_cuchilla
from matplotlib.widgets import Slider

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
#Como es una onda plana entonces es la misma si la propagamos en el espacio libre
#S1_campo, S1_x_mesh, S1_y_mesh, S1_dx, S1_dy = propagar_ABCD_(U_0,"propagar", d, 0, lam,k)

#Interaccion con el espejo
S3_campo, S3_x_mesh, S3_y_mesh, S3_dx, S3_dy = propagar_ABCD_(U_0, "espejo", 0, R, lam,k)

#Multiplicamos por el objeto como si fuera una trasnmitancia
camp1 = S3_campo * S_campo

#Del objeto -> cuchilla
S5_campo, S5_x_mesh, S5_y_mesh, S5_dx, S5_dy = propagar_ABCD_(camp1, "propagar", d, 0, lam,k)

#Aplicamos el filtro de la cuchilla
S_filtred = aplicar_filtro_cuchilla(S5_campo,"horizontal")

# Aplicamos la Transformada Inversa (La lente formadora de imagen) que seria la camara o sensor a utilizar
campo_en_sensor = np.fft.fftshift(np.fft.ifft2(S_filtred))

# Calculamos la intensidad 
Imagen_Schlieren = np.abs(campo_en_sensor)**2

plot_simulacion(Imagen_Schlieren, "Imagen $SCHLIEREN$ 1 Espejo ($Sonido$)", "gray")

# ===================================================================
#           Configuración del Gráfico Interactivo
# ===================================================================

fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(bottom=0.25) # Espacio para el slider

# Mostrar imagen inicial
im = ax.imshow(Imagen_Schlieren, cmap='gray')
ax.set_title(f"Simulación Schlieren - Distancia d = {d:.2f} m")
ax.axis('off')

# Crear el Slider
ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03]) # Posición [x, y, ancho, alto]
slider_d = Slider(
    ax=ax_slider,
    label='Distancia d (m)',
    valmin=0.5,
    valmax= f+0.5,
    valinit=0.1,
)

# Función de actualización
def update(val):
    d_nuevo = slider_d.val
    
    # Recalcular solo la propagación final
    S5_new, _, _, _, _ = propagar_ABCD_(camp1, "propagar", d_nuevo, 0, lam, k)
    
    # Aplicar cuchilla y reconstruir
    S_filt_new = aplicar_filtro_cuchilla(S5_new, "horizontal")
    sensor_new = np.fft.fftshift(np.fft.ifft2(S_filt_new))
    Img_new = np.abs(sensor_new)**2
    
    # Actualizar gráfico
    im.set_data(Img_new)
    ax.set_title(f"Simulación Schlieren - Distancia d = {d_nuevo:.4f} m")
    fig.canvas.draw_idle()

slider_d.on_changed(update)

plt.show()