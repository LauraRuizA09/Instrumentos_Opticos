import numpy as np
import matplotlib.pyplot as plt
from Funciones import mapeo, generar_campo_entrada_S, N_0, generar_onda_plana, plot_simulacion, propagar_ABCD_
from Funciones import aplicar_filtro_cuchilla, generar_columna_calor

# ===================================================================
#                   Parametros Fisicos y de muestreo
# ===================================================================

# Intensidad del calor
# 80°C sobre el ambiente 
delta_temp_K = 80

# Ancho de la fuente de calor
ancho_fuente = 0.02  # 2 cm de ancho

# Configuración del espacio (Espejo de 30 cm)
resolucion = 800
tamano_fisico_m = 0.3

# Parámetros de prueba
L_z = 0.3               # Espesor de la zona de prueba (30 cm)
lam = 633e-9        # Longitud de onda (633 nm)

# Parámetros de la simualcion de muestreo
L_x = 0.2            # 20 cm de ventana
L_y = 0.2
Nx = 1024            # Resolución
Ny = 1024

# ===================================================================
#                   Simular columna de calor
# ===================================================================

X, Y, dx, dy = mapeo(Nx, Ny, tamano_fisico_m)
n_map = generar_columna_calor(X, Y, delta_temp_K, ancho_fuente)


# ===================================================================
#         Generar onda de calor como un campo con fase
# ===================================================================

S_campo, fase = generar_campo_entrada_S(n_map, lam, L_z)

# Definir Onda Plana
U_0 = generar_onda_plana(Nx,Ny)

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