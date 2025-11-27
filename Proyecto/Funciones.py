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

def simular_camino_completo(eps_x, eps_y, distancia_espejo_onda, focal_espejo):

    # Recorrido: Fuente de luz -> Onda sonido -> Espejo -> Onda de sonido -> Cuchilla -> Sensor
    
    R_espejo = 2 * focal_espejo  # Radio de curvatura
    
    # En configuración coincidente (Z-type), la fuente y cámara están en el centro de curvatura (2f)
    distancia_total = R_espejo 
    
    # Distancias parciales
    d1_fuente_a_onda = distancia_total - distancia_espejo_onda
    d2_onda_a_espejo = distancia_espejo_onda
    
    M_viaje_largo = matriz_propagacion(d1_fuente_a_onda) # Tramo Fuente-Onda
    M_viaje_corto = matriz_propagacion(d2_onda_a_espejo) # Tramo Onda-Espejo
    M_espejo      = matriz_reflexion_curvas(R_espejo)    
    
    # 3. CÁLCULO DE SENSIBILIDAD (Trazado de un "Rayo Unitario")
    # Para no multiplicar matrices gigantes de 800x800 píxeles, calculamos qué le pasa
    # a un rayo "test" con desviación = 1 radian, y luego aplicamos ese factor a tu imagen.
    
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


def simular_corte_cuchilla(desp_x, desp_y, tipo="circular", radio_focal_mm=0.5):

    # Convertir radio del foco a metros
    radio_focal_m = radio_focal_mm / 1000.0
    
    # Intensidad base (Fondo)
    # Si es circular (campo oscuro), el fondo es negro (0.0).
    # Si es cuchilla recta, el fondo es gris (0.5).
    if tipo == "circular":
        I_base = 0.0
    else:
        I_base = 0.5
        
    if tipo == "vertical":
        # Cuchilla Vertical
        # Normalizamos el desplazamiento respecto al tamaño del foco
        cambio_luz = (desp_x / radio_focal_m)
        imagen = I_base + cambio_luz
        
    elif tipo == "horizontal":
        # Cuchilla Horizontal
        cambio_luz = (desp_y / radio_focal_m)
        imagen = I_base + cambio_luz
        
    elif tipo == "circular":
        # Filtro Circular (Campo Oscuro)
        # Calculamos la magnitud total del desplazamiento (radio)
        magnitud_desp = np.sqrt(desp_x**2 + desp_y**2)
        
        # Cuanto más se aleja del centro bloqueado, más brillante es la imagen
        imagen = (magnitud_desp / radio_focal_m)
        
    else:
        # Por defecto devuelve negro si el tipo está mal escrito
        return np.zeros_like(desp_x)

    # Limitar valores físicos (Clip entre 0 y 1 para que sea una imagen válida)
    return np.clip(imagen, 0, 1)








def generar_pluma_termica(X, Y, intensidad_dn=0.0003, ancho=0.02, turbulencia=0.5):

    # Genera un mapa de índice de refracción simulando el aire caliente subiendo de un fósforo.
    
    # Definir la trayectoria de la columna (serpenteo)
    # Usamos seno para simular que el humo se mueve de lado a lado al subir
    # Y va de negativo (abajo) a positivo (arriba).
    
    # Frecuencia espacial del serpenteo
    k = 150
    
    # El centro de la columna se desplaza en X según la altura Y
    x_centro = (turbulencia * 0.01) * np.sin(k * Y) * np.exp(Y*2) # Se mueve más arriba
    
    # Perfil de Temperatura (Gaussiano invertido)
    # El aire es más caliente en el centro (x_centro) y se enfría hacia afuera.
    # Usamos una función Gaussiana: exp(-x^2)
    distancia_al_centro = X - x_centro
    perfil_calor = np.exp(- (distancia_al_centro**2) / (2 * ancho**2))
    
    # Disipación (El calor se disipa al subir)
    # Hacemos que la intensidad baje a medida que Y aumenta (se enfría arriba)
    # Normalizamos Y para que vaya de 0 a 1 aprox para la disipación
    y_norm = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))
    disipacion = 1.0 - (0.7 * y_norm) # Se mantiene al 30% arriba
    
    # Calcular el índice n
    # n_base es aprox 1.00029. 
    # El calor BAJA el índice, por eso RESTAMOS la perturbación.
    n_base = 1.00029
    perturbacion = intensidad_dn * perfil_calor * disipacion
    
    # Añadimos un poco de ruido aleatorio para simular micro-turbulencia
    ruido = np.random.normal(0, 0.05 * intensidad_dn, X.shape) * perfil_calor
    
    n_field = n_base - (perturbacion + ruido)
    
    return n_field

def simular_cuchilla_esquina(eps_x, eps_y, sensibilidad, corte_x=50, corte_y=50):
    """
    Simula una cuchilla rectangular (ESQUINA) que corta en X y en Y simultáneamente.
    Esto permite ver gradientes en todas las direcciones (efecto relieve 3D diagonal).
    
    Args:
        eps_x, eps_y: Mapas de desviación angular.
        sensibilidad: Factor B (metros/radián).
        corte_x, corte_y: Porcentaje de corte en cada eje (50% es el estándar).
        
    Returns:
        imagen (array 2D): Intensidad resultante.
    """
    radio_fuente = 0.0001 # 1mm de fuente
    
    # --- EJE X (Vertical Knife Edge) ---
    desplazamiento_x = eps_x * sensibilidad
    pos_cuchilla_x = (corte_x - 50) / 100 * radio_fuente * 2
    dist_borde_x = desplazamiento_x - pos_cuchilla_x
    # Transmisión de luz en X (0 a 1)
    transmision_x = 0.5 + (dist_borde_x / (2 * radio_fuente))
    transmision_x = np.clip(transmision_x, 0, 1)
    
    # --- EJE Y (Horizontal Knife Edge) ---
    desplazamiento_y = eps_y * sensibilidad
    pos_cuchilla_y = (corte_y - 50) / 100 * radio_fuente * 2
    dist_borde_y = desplazamiento_y - pos_cuchilla_y
    # Transmisión de luz en Y (0 a 1)
    transmision_y = 0.5 + (dist_borde_y / (2 * radio_fuente))
    transmision_y = np.clip(transmision_y, 0, 1)
    
    # --- COMBINACIÓN (INTERSECCIÓN) ---
    # Si la cuchilla es una esquina sólida que bloquea, por ejemplo, el cuadrante inferior izquierdo,
    # la luz solo pasa si logra superar AMBOS bordes.
    # Multiplicamos las transmisiones para simular que la luz debe sobrevivir a ambos cortes.
    imagen_final = transmision_x * transmision_y
    
    return imagen_final