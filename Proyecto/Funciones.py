import numpy as np
import matplotlib.pyplot as plt


# ===================================================================
#               Variables fisicas a utilizar
# ===================================================================

RHO_0 = 1.225        # kg/m^3 (Densidad base)
K_GLADSTONE = 2.26e-4 # m^3/kg (Constante para luz visible)


# ===================================================================
#            Creacion de la onda sonora esferica
# ===================================================================

def mapeo(resolucion, tamano_fisico):

    x = np.linspace(-tamano_fisico/2, tamano_fisico/2, resolucion)
    y = np.linspace(-tamano_fisico/2, tamano_fisico/2, resolucion)
    X, Y = np.meshgrid(x, y)
    dx = tamano_fisico / resolucion
    return X, Y, dx

def generar_onda_sonora(X, Y, frecuencia_espacial, amplitud_rho):

   # Genera el campo de índice de refracción (n) para una onda sonora esférica
   # Basado en Gladstone-Dale: n = 1 + K * rho
    
    # Radio desde el centro
    R = np.sqrt(X**2 + Y**2)
    
    # Modelado de onda esférica (sinusoidal)
    perturbacion = amplitud_rho * np.sin(2 * np.pi * frecuencia_espacial * R)
    
    # Densidad total
    rho_total = RHO_0 + perturbacion
    
    # Índice de refracción
    n_field = 1.0 + (K_GLADSTONE * rho_total)
    
    return n_field

# ===================================================================
#           Definicion efectos difractivos segun el n
# ===================================================================

def calcular_gradientes(n_field, dx):

    # Calcula cuánto cambia 'n' en cada punto
    # Esto es lo que el sistema Schlieren detecta

    # n_field = Valores del indice de refraccion calculados de la onda sonora
    # dx = distancia entre pixeles

    # np.gradient usa diferencias centrales, es decir mira el vecino i-1 y i+1 y calcula la pendiente
    grad = np.gradient(n_field, dx) 

    dn_dy = grad[0] # Gradiente en eje vertical, cuanto se desvia el rayo de luz hacia arriba o hacia abajo
    dn_dx = grad[1] # Gradiente en eje horizontal, cuanto se desvia el rayo de luz hacia la derecha o hacia la izquierda
    
    return dn_dx, dn_dy

def calcular_desviacion_angular(dn_dx, dn_dy, espesor_z):

    # Calcula el ángulo epsilon que se desvía la luz
    # epsilon_x = (1/n) * Integral(dn/dx * dz)
    # Asumimos aproximación paraxial (n ~ 1) y perturbación constante en Z, es decir los rayos d eluz etsan muy cerca al eje z
    # Aqui asumimos que la onda esferica no cambai realmente a medida que aumenta la distancia, es decir una onda cilindrica por que si cambia en X y Y 

    n_promedio = 1.00029 # Aproximado para aire
    
    # Integración: gradiente * longitud de interacción
    epsilon_x = (1 / n_promedio) * dn_dx * espesor_z
    epsilon_y = (1 / n_promedio) * dn_dy * espesor_z
    
    return epsilon_x, epsilon_y

def plot_simulacion(campo, titulo, cmap='viridis'):

    plt.figure(figsize=(6, 5))
    plt.imshow(campo, cmap=cmap)
    plt.colorbar()
    plt.title(titulo)
    plt.axis('off')
    plt.show()


# ===================================================================
#         Funciones de propagación de la luz por matrices
# ===================================================================

def matriz_propagacion(d):
 # Calcula la matriz de transferencia de rayos para una propagacion
 # d: distancia de propagacion
 
 Prop = np.array([[1, d], [0, 1]])
 return Prop

def matriz_reflexion_curvas(R):
 
 # Calcula la matriz de transferencia de rayos para una reflexion
 # R: radio de curvatura de la superficie reflectante
 
 a = - 2 / R
 Re_curve = np.array([[1, 0], [a, 1]])

 return Re_curve 

# ===================================================================
#                 Configuración de un solo espejo 
# ===================================================================

def simular_camino_completo(eps_x, eps_y, distancia_espejo_onda, focal_espejo):

    # Recorrido: Fuente de luz -> Onda sonido -> Espejo -> Onda de sonido -> Cuchilla -> Sensor
    
    R_espejo = 2 * focal_espejo  # Radio de curvatura
    
    # En configuración coincidente, la fuente y cámara están en el centro de curvatura (2f)
    distancia_total = R_espejo 
    
    # Distancias parciales
    d1_fuente_a_onda = distancia_total - distancia_espejo_onda
    d2_onda_a_espejo = distancia_espejo_onda
    
    M_viaje_largo = matriz_propagacion(d1_fuente_a_onda) # Tramo Fuente-Onda
    M_viaje_corto = matriz_propagacion(d2_onda_a_espejo) # Tramo Onda-Espejo
    M_espejo      = matriz_reflexion_curvas(R_espejo)    
    
    # CÁLCULO DE SENSIBILIDAD (Trazado de un "Rayo Unitario")
    # Para no multiplicar matrices gigantes de 800x800 píxeles, calculamos qué le pasa
    # a un rayo "test" con desviación = 1 radian, y luego aplicamos ese factor a la imagen
    
    # Estado Inicial del Rayo (Desviación relativa = 0)
    # Vector: [Altura (y), Angulo (theta)]
    rayo_test = np.array([0.0, 0.0]) 
    
    # --- INICIO DEL VIAJE ---
    
    # Fuentede luz -> Onda de sonido 
    rayo_test = M_viaje_largo @ rayo_test
    
    # El rayo cruza la onda de sonido
    # Físicamente: El ángulo aumenta en 1 unidad (nuestro test)
    rayo_test[1] += 1.0 
    
    # Onda -> Espejo
    rayo_test = M_viaje_corto @ rayo_test
    
    # Reflexión en el Espejo
    rayo_test = M_espejo @ rayo_test
    
    # Espejo -> Onda 
    rayo_test = M_viaje_corto @ rayo_test
    
    # El rayo vuelve a cruzar la onda
    # Físicamente: Se vuelve a sumar la desviación (el sonido sigue ahí)
    rayo_test[1] += 1.0
    
    # Onda -> Cuchilla/Sensor (Viaje Final)
    rayo_test = M_viaje_largo @ rayo_test
    
    
    # El valor final rayo_test[0] es la altura (y) a la que llega el rayo.
    # Este valor es nuestro "Factor de Sensibilidad Total" del sistema.
    sensibilidad_total = rayo_test[0]
    
    # APLICAR AL CAMPO COMPLETO DE LA IMAGEN
    # Ahora que sabemos cuánto se mueve un rayo por cada radián de desviación,
    # multiplicamos por tus matrices de gradientes reales.
    
    desplazamiento_x_final = sensibilidad_total * eps_x
    desplazamiento_y_final = sensibilidad_total * eps_y
    
    return desplazamiento_x_final, desplazamiento_y_final, sensibilidad_total

# ===================================================================
#                 Creación de la cuchilla
# ===================================================================

def simular_corte_cuchilla(desp_x, desp_y, tipo, radio_focal_m):
    
    # Intensidad base (Fondo)
    # Si es circular (campo oscuro), el fondo es negro
    # Si es cuchilla recta, el fondo es gris
    
    I_base = 0.5
        
    if tipo == "vertical":

        # Normalizamos el desplazamiento respecto al tamaño del foco
        cambio_luz = (desp_x / radio_focal_m)
        imagen = I_base + cambio_luz
        
    elif tipo == "horizontal":

        cambio_luz = (desp_y / radio_focal_m)
        imagen = I_base + cambio_luz
        
    elif tipo == "circular":
        I_base = 0.0
        # Calculamos la magnitud total del desplazamiento (radio)
        magnitud_desp = np.sqrt(desp_x**2 + desp_y**2)
        
        # Cuanto más se aleja del centro bloqueado, más brillante es la imagen
        imagen = (magnitud_desp / radio_focal_m)
        
    else:
        # Por defecto devuelve negro
        return np.zeros_like(desp_x)

    # Limitar valores físicos
    return np.clip(imagen, 0, 1)

# ===================================================================
#              Configuración de dos espejos (Z-type)
# ===================================================================

def simular_z_type_dos_espejos(eps_x, eps_y, distancia_onda_espejo2, focal_espejo2):
    
    # Simula el recorrido Z-Type: Onda -> Espejo 2 (Enfoque) -> Cuchilla

    R_espejo2 = 2 * focal_espejo2
    
    # La cuchilla se coloca en el foco del Espejo 2
    distancia_espejo2_cuchilla = focal_espejo2
    
    # Tramo A: Desde la Onda hasta el Espejo 2
    M_viaje_onda_espejo = matriz_propagacion(distancia_onda_espejo2)
    
    # Tramo B: Reflexión en Espejo 2
    M_reflexion2 = matriz_reflexion_curvas(R_espejo2)
    
    # Tramo C: Desde Espejo 2 hasta Cuchilla (Distancia focal)
    M_viaje_espejo_cuchilla = matriz_propagacion(distancia_espejo2_cuchilla)
    
    # 3. CÁLCULO DE SENSIBILIDAD (Trazado de Rayo Unitario)
    # Rayo Test inicial en la Onda: [y=0, theta=0]
    rayo_test = np.array([0.0, 0.0])
    
    # --- INICIO DEL VIAJE ---

    # El rayo cruza la onda UNA sola vez
    rayo_test[1] += 1.0  # Sumamos 1 radian de desviación
    
    # Onda -> Espejo 2
    rayo_test = M_viaje_onda_espejo @ rayo_test
    
    # Reflexión en Espejo 2 
    rayo_test = M_reflexion2 @ rayo_test
    
    # Espejo 2 -> Cuchilla
    rayo_test = M_viaje_espejo_cuchilla @ rayo_test
    
    # --- FIN DEL VIAJE ---
    
    # El valor final rayo_test[0] es la altura (y) en la cuchilla.
    sensibilidad_total = rayo_test[0]
    
    desp_x = sensibilidad_total * eps_x
    desp_y = sensibilidad_total * eps_y
    
    return desp_x, desp_y, sensibilidad_total

# ===================================================================
#       Generar una simulaaciond de una columna de calor
# ===================================================================

def generar_ruido_fractal(X, Y, escala=10.0, complejidad=3):

    ruido = np.zeros_like(X)
    
    # Sumamos capas de "ruido" (Octavas)
    for i in range(1, complejidad + 1):
        frecuencia = 2**i  # Cada capa es más detallada
        amplitud = 1 / frecuencia
        # Desfases aleatorios fijos para que no parezca un patrón repetido
        fase_x = np.sin(i * 132.1) * 10 
        fase_y = np.cos(i * 54.3) * 10
        
        # El ruido se mueve principalmente en Y (el calor sube)
        ruido += amplitud * np.sin(X * escala * frecuencia + fase_x) * \
                            np.sin(Y * escala * frecuencia * 0.5 + fase_y)
                            
    return ruido

def generar_columna_calor(X, Y, temperatura_max_delta, ancho_columna_m):
    
    #Simula el índice de refracción de una columna de aire caliente (Vela/Soldador).
    
    #  Constantes físicas
    T_ambiente = 293.15 # Kelvin (20°C)
    n0_aire = 1.00029   # Indice base
    
    # Geometría
    # El calor sube, así que depende de Y
    # Queremos que la columna sea ancha abajo y se disperse arriba
    
    # Centro de la columna (con turbulencia añadida)
    # El 'ruido' hace que el centro oscile izquierda/derecha a medida que sube
    turbulencia = generar_ruido_fractal(X, Y, escala=20, complejidad=4)
    
    # La turbulencia afecta más arriba (Y alto) que abajo (Y bajo)
    # Normalizamos Y para que vaya de 0 (abajo) a 1 (arriba)
    y_norm = (Y - Y.min()) / (Y.max() - Y.min())
    desvio_centro = turbulencia * 0.02 * y_norm # 2cm de oscilación máxima arriba
    
    distancia_al_centro = np.abs(X - desvio_centro)
    
    # Perfil de Temperatura (Gaussiana)
    # T = T_amb + DeltaT * exp(-x^2 / sigma^2)
    # sigma (ancho) crece a medida que sube (difusión)
    ancho_variable = ancho_columna_m * (0.5 + 1.5 * y_norm) 
    
    perfil_gaussiano = np.exp(-(distancia_al_centro**2) / (2 * (ancho_variable/2)**2))
    
    # Aplicamos la temperatura
    # La temperatura decae con la altura (se enfría al subir)
    enfriamiento = 1.0 - (0.5 * y_norm) 
    T_campo = T_ambiente + (temperatura_max_delta * perfil_gaussiano * enfriamiento)
    
    # Convertir Temperatura a Índice de Refracción (Gladstone-Dale aproximado)
    # n - 1 es proporcional a la densidad, y densidad es inv. prop. a Temperatura
    # (n_nuevo - 1) = (n0 - 1) * (T_amb / T_nuevo)
    
    n_field = 1.0 + (n0_aire - 1.0) * (T_ambiente / T_campo)
    
    return n_field