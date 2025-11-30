import sys
import os

directorio_actual = os.path.dirname(os.path.abspath(__file__))
directorio_padre = os.path.dirname(directorio_actual)
sys.path.append(directorio_padre)

import numpy as np
import matplotlib.pyplot as plt
from Funciones import mapeo, generar_campo_entrada_S, generar_onda_sonora, N_0, generar_onda_plana, plot_simulacion, propagar_ABCD_
from Funciones import aplicar_filtro_cuchilla


# ===================================================================
#           Función Local para inyectar Aberración (Coma)
# ===================================================================

def agregar_coma(campo, X, Y, magnitud):
    k_luz = 2 * np.pi / 633e-9
    # Normalizar coordenadas al radio máximo
    R_max = np.max(X)
    rho = np.sqrt(X**2 + Y**2) / R_max
    theta = np.arctan2(Y, X)
    
    # Polinomio de Zernike para Coma
    W = magnitud * (3*rho**3 - 2*rho) * np.sin(theta)
    
    fase_error = np.exp(1j * k_luz * W)
    return campo * fase_error

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
d = f               #distancia de propagacion
R = 2*f
k = 2 * np.pi / lam # Numero de onda 

#Definición del recorrido

#De la fuente -> espejo
#Como es una onda plana entonces es la misma si la propagamos en el espacio libre
#S1_campo, S1_x_mesh, S1_y_mesh, S1_dx, S1_dy = propagar_ABCD_(U_0,"propagar", d, 0, lam,k)

#Interaccion con el espejo
S3_campo, S3_x_mesh, S3_y_mesh, S3_dx, S3_dy = propagar_ABCD_(U_0, "espejo", 0, R, lam,k)

# Agregamos Coma al espejo para simular el defecto físico
magnitud_coma = 1.5e-6  # Magnitud del defecto (ajustable)
S3_ab = agregar_coma(S3_campo, X, Y, magnitud_coma)

#Multiplicamos por el objeto como si fuera una trasnmitancia
camp1 = S3_campo * S_campo
camp1_ = S3_ab * S_campo 

#Del objeto -> cuchilla
S5_campo, S5_x_mesh, S5_y_mesh, S5_dx, S5_dy = propagar_ABCD_(camp1, "propagar", d, 0, lam,k)
S5_campo_, S5_x_mesh_, S5_y_mesh_, S5_dx_, S5_dy_ = propagar_ABCD_(camp1_, "propagar", d, 0, lam,k)

#Aplicamos el filtro de la cuchilla
S_filtred = aplicar_filtro_cuchilla(S5_campo,"horizontal")
S_filtred_ = aplicar_filtro_cuchilla(S5_campo_,"horizontal")

# Aplicamos la Transformada Inversa (La lente formadora de imagen) que seria la camara o sensor a utilizar
campo_en_sensor = np.fft.fftshift(np.fft.ifft2(S_filtred))
campo_en_sensor_ = np.fft.fftshift(np.fft.ifft2(S_filtred_))

# Calculamos la intensidad 
Img_ideal = np.abs(campo_en_sensor)**2
Img_ab = np.abs(campo_en_sensor_)**2

# ===================================================================
#           Visualización Comparativa
# ===================================================================

fig, ax = plt.subplots(1, 2, figsize=(14, 6))

ax[0].imshow(Img_ideal, cmap='gray')
ax[0].set_title("Sistema Ideal (Z-Type)\nSin Aberraciones")
ax[0].axis('off')

ax[1].imshow(Img_ab, cmap='gray')
ax[1].set_title("Sistema 1 Espejo (Fuera de Eje)\nCon Aberración de Coma")
ax[1].axis('off')

plt.suptitle("Impacto de las Aberraciones Ópticas en Schlieren", fontsize=16)
plt.tight_layout()
plt.show()