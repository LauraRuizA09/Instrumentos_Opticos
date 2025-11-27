import matplotlib.pyplot as plt
import numpy as np
from Funciones import mapeo, generar_onda_sonora, calcular_gradientes, plot_simulacion
from Funciones import simular_camino_completo, simular_corte_cuchilla, calcular_desviacion_angular


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
#           Visualizacion de la onda y como cambia n
# ===================================================================

plot_simulacion(n_map, f"Campo de Índice de Refracción (n)\nFrecuencia: {frecuencia_generador_hz/1000:.1f} kHz", cmap='viridis')

plot_simulacion(dndx, "Gradiente Horizontal (dn/dx)\n(Base para Cuchilla Vertical)", cmap='seismic')

plot_simulacion(dndy, "Gradiente Vertical (dn/dy)\n(Base para Cuchilla Horizontal)", cmap='seismic')


# ===================================================================
#           Interaccion onda y camino optico completo
# ===================================================================

# 1. Configuración del Laboratorio
focal = 1.0           # Espejo f=1m (R=2m)
posicion_onda = 0.5   # La onda está a 0.5m del espejo (y a 1.5m de la cámara)

# 2. Ejecutar Simulación de Camino Completo
desp_x, desp_y, factor_S = simular_camino_completo(eps_x, eps_y, posicion_onda, focal)

print(f"--- Resultado de la Simulación Óptica ---")
print(f"Configuración: Objeto a {posicion_onda}m del espejo.")
print(f"Factor de Sensibilidad calculado: {factor_S:.4f} metros de desplazamiento por radián.")
print(f"(Esto significa que si la onda desvía la luz 1 mrad, la mancha se mueve {factor_S} mm en la cuchilla)")

# 3. Aplicar Cuchilla y Graficar
# Usamos un radio focal pequeño (0.5mm)
img_final_trayectoria = simular_corte_cuchilla(desp_x, desp_y, tipo="circular", radio_focal_mm=0.5)
plot_simulacion(img_final_trayectoria, 
                f"Schlieren Trayectoria Completa (Ida y Vuelta)\nObjeto a {posicion_onda}m del espejo", 
                cmap='inferno')