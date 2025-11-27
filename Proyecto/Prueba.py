import matplotlib.pyplot as plt
import numpy as np
from Funciones import mapeo, generar_onda_sonora, calcular_gradientes, calcular_desviacion_angular
from Funciones import calcular_sensibilidad_un_espejo, simular_cuchilla, calcular_sensibilidad_z_type
from Funciones import generar_pluma_termica, simular_cuchilla_esquina

# 1. CONFIGURACIÓN DEL SISTEMA (Experimento Virtual)
# ---------------------------------------------------
FOCAL_ESPEJO = 1.0       # Espejo de 1 metro de focal (Radio = 2m)
DISTANCIA_OBJETO = 0.1   # Objeto pegado al espejo (distancia relativa pequeña)
FRECUENCIA_ONDA = 80     # Frecuencia visual
AMPLITUD_RHO = 0.05      # Intensidad de la onda

# 2. GENERAR FÍSICA (Módulo 1)
# ---------------------------------------------------
X, Y, dx = mapeo(resolucion=800, tamano_fisico=0.3)
n_map = generar_onda_sonora(X, Y, FRECUENCIA_ONDA, AMPLITUD_RHO)
dndx, dndy = calcular_gradientes(n_map, dx)

# Nota: En sistema de 1 espejo, la luz pasa 2 veces. 
# Espesor efectivo = espesor real * 2
eps_x, eps_y = calcular_desviacion_angular(dndx, dndy, espesor_z=0.1 * 2) 

# 3. SIMULAR ÓPTICA (Módulo 2)
# ---------------------------------------------------
# Calculamos cuánto amplifica el espejo la desviación
sensibilidad = calcular_sensibilidad_un_espejo(FOCAL_ESPEJO, DISTANCIA_OBJETO)

# Generamos la imagen final (Schlieren Image)
# Probamos con Cuchilla VERTICAL (sensible a gradientes horizontales X)
imagen_schlieren = simular_cuchilla(eps_x, eps_y, sensibilidad, tipo='vertical', corte=50)

# 4. VISUALIZACIÓN FINAL
# ---------------------------------------------------
plt.figure(figsize=(10, 8))
# Usamos mapa de grises porque así se ve en la cámara real
plt.imshow(imagen_schlieren, cmap='gray', vmin=0, vmax=1)
plt.title(f"Simulación Schlieren (1 Espejo)\nCuchilla Vertical - Corte 50%")
plt.colorbar(label="Intensidad en Cámara (Norm)")
plt.axis('off')
plt.show()


# 1. PARAMETROS DEL MONTAJE Z-TYPE (Dos Espejos)
# ---------------------------------------------------
FOCAL_ESPEJO_2 = 1.0     # Focal del segundo espejo (ej. 1 metro)
# Nota: La focal del espejo 1 no afecta la sensibilidad, solo colima la luz.

# Parámetros de la Onda
FRECUENCIA_ONDA = 60
AMPLITUD_RHO = 0.08      # Un poco más fuerte para compensar que solo pasa 1 vez
ESPESOR_REAL_ONDA = 0.1  # 10 cm de ancho de la onda

# 2. GENERAR EL OBJETO (FÍSICA)
# ---------------------------------------------------
X, Y, dx = mapeo(resolucion=800, tamano_fisico=0.3)
n_map = generar_onda_sonora(X, Y, FRECUENCIA_ONDA, AMPLITUD_RHO)
dndx, dndy = calcular_gradientes(n_map, dx)

# OJO AQUÍ: En Z-Type la luz pasa 1 SOLA VEZ.
# Usamos el espesor real, sin multiplicar por 2.
eps_x, eps_y = calcular_desviacion_angular(dndx, dndy, espesor_z=ESPESOR_REAL_ONDA)

# 3. SIMULACIÓN ÓPTICA
# ---------------------------------------------------
# Calculamos sensibilidad para configuración Z
sensibilidad = calcular_sensibilidad_z_type(FOCAL_ESPEJO_2)

# Generamos la imagen
# Probamos con Cuchilla HORIZONTAL esta vez (sensible a gradientes verticales Y)
# Esto hará que la parte de arriba y abajo de los anillos brille.
imagen_z = simular_cuchilla(eps_x, eps_y, sensibilidad, tipo='horizontal', corte=50)

# 4. VISUALIZACIÓN COMPARATIVA
# ---------------------------------------------------
plt.figure(figsize=(10, 8))
plt.imshow(imagen_z, cmap='gray', vmin=0, vmax=1)
plt.title(f"Simulación Z-Type (2 Espejos)\nCuchilla Horizontal - Efecto Relieve")
plt.colorbar(label="Intensidad")
plt.axis('off')
plt.show()

# 1. CONFIGURACIÓN
# ---------------------------------------------------
FOCAL_ESPEJO = 1.5       # Espejo de 1.5m (típico telescopio aficionado)
DISTANCIA_OBJETO = 0.2   # Objeto a 20cm del espejo
ANCHO_PLUMA = 0.015      # 1.5 cm de grosor (como la llama de una vela)
INTENSIDAD = 0.00005     # Cambio de índice n (muy pequeño, el calor es sutil)

# 2. GENERAR EL FÓSFORO VIRTUAL
# ---------------------------------------------------
# Creamos una ventana de 15x15 cm
X, Y, dx = mapeo(resolucion=800, tamano_fisico=0.15)

# Generamos el mapa de refracción del calor
n_map = generar_pluma_termica(X, Y, intensidad_dn=INTENSIDAD, ancho=ANCHO_PLUMA)

# Calculamos gradientes
dndx, dndy = calcular_gradientes(n_map, dx)

# Luz pasa 2 veces (Sistema 1 espejo coincidente)
eps_x, eps_y = calcular_desviacion_angular(dndx, dndy, espesor_z=0.05 * 2) 

# 3. SIMULACIÓN ÓPTICA
# ---------------------------------------------------
sensibilidad = calcular_sensibilidad_un_espejo(FOCAL_ESPEJO, DISTANCIA_OBJETO)

# --- PRUEBA 1: CUCHILLA VERTICAL (Detecta cambios horizontales) ---
# Ideal para ver los bordes de la columna de calor
img_vertical = simular_cuchilla(eps_x, eps_y, sensibilidad, tipo='vertical', corte=50)

# --- PRUEBA 2: CUCHILLA HORIZONTAL (Detecta cambios verticales) ---
# Ideal para ver turbulencia subiendo
img_horizontal = simular_cuchilla(eps_x, eps_y, sensibilidad, tipo='horizontal', corte=50)

# 4. VISUALIZACIÓN
# ---------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(img_vertical, cmap='gray', vmin=0, vmax=1)
ax[0].set_title("Fósforo/Vela - Cuchilla Vertical\n(Resalta los lados)")
ax[0].axis('off')

ax[1].imshow(img_horizontal, cmap='gray', vmin=0, vmax=1)
ax[1].set_title("Fósforo/Vela - Cuchilla Horizontal\n(Resalta el ascenso)")
ax[1].axis('off')

plt.show()


# 1. CONFIGURACIÓN (Sistema 1 Espejo - Coincidente)
# ---------------------------------------------------
FOCAL = 1.0              # 1 metro de focal
DIST_OBJETO = 0.2        # Objeto cerca del espejo
INTENSIDAD_CALOR = 0.00008 # Un poco más fuerte para ver detalles

# 2. GENERAR FÍSICA (Calor / Pluma)
# ---------------------------------------------------
X, Y, dx = mapeo(800, 0.15) # 15cm de ventana
n_map = generar_pluma_termica(X, Y, intensidad_dn=INTENSIDAD_CALOR, ancho=0.015, turbulencia=0.6)
dndx, dndy = calcular_gradientes(n_map, dx)

# Luz pasa 2 veces (ida y vuelta)
eps_x, eps_y = calcular_desviacion_angular(dndx, dndy, espesor_z=0.03 * 2)

# 3. SIMULACIÓN ÓPTICA (Filtro de Esquina)
# ---------------------------------------------------
sensibilidad = calcular_sensibilidad_un_espejo(FOCAL, DIST_OBJETO)

# Usamos la nueva función con 50% de corte en ambos ejes
imagen_esquina = simular_cuchilla_esquina(eps_x, eps_y, sensibilidad, corte_x=50, corte_y=50)

# 4. VISUALIZACIÓN
# ---------------------------------------------------
plt.figure(figsize=(8, 8))
plt.imshow(imagen_esquina, cmap='gray', vmin=0, vmax=1)
plt.title("Simulación Schlieren: Filtro de Esquina (Rectangular)\n(Sensibilidad X e Y simultánea)")
plt.colorbar(label="Intensidad")
plt.axis('off')
plt.show()