import numpy as np
import matplotlib.pyplot as plt


# ===================================================================
#               Variables fisicas a utilizar
# ===================================================================

RHO_0 = 1.225                       # kg/m^3 (Densidad base)
K_GLADSTONE = 2.26e-4               # m^3/kg (Constante para luz visible)
N_0 = 1.0 + (K_GLADSTONE * RHO_0)   #indice de fase del aire riguroso
AMPLITUD_A0 = 1                     # Intensidad inicial de la onda


# ===================================================================
#            Creacion de la onda sonora esferica
# ===================================================================

def mapeo(resolucion_x, resolucion_y, tamano_fisico):

    x = np.linspace(-tamano_fisico/2, tamano_fisico/2, resolucion_x)
    y = np.linspace(-tamano_fisico/2, tamano_fisico/2, resolucion_y)
    X, Y = np.meshgrid(x, y)
    dx = tamano_fisico / resolucion_x
    dy = tamano_fisico / resolucion_y
    return X, Y, dx,dy

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

# ===================================================================
#             Construcción del Campo Complejo S(x,y)
# ===================================================================

def generar_campo_entrada_S(n_field, lambda_luz, espesor_z):

    # Calcular el número de onda
    k_luz = 2 * np.pi / lambda_luz
    
    # Calcular la variación respecto al aire en reposo (Delta n)
    # Esto es lo que causa el retraso
    delta_n = n_field - N_0
    
    # Calcular el Mapa de Fase phi(x,y)
    # phi = k * delta_n * L
    # Esto representa cuánto se atrasó la onda en cada punto (x,y)
    phi_map = k_luz * delta_n * espesor_z
    
    # Construir el campo complejo S(x,y)
    # S = A0 * exp(i * phi)
    S_field = AMPLITUD_A0 * np.exp(1j * phi_map)
    
    return S_field, phi_map


def plot_simulacion(campo, titulo, cmap):

    plt.figure(figsize=(6, 5))
    plt.imshow(campo, cmap=cmap)
    plt.colorbar()
    plt.title(titulo)
    plt.axis('off')
    plt.show()

def generar_onda_plana(resolucion_x,resolucion_y, amplitud=1.0):

    # U(x,y) = A * e^(i*0)
    campo = np.full((resolucion_x, resolucion_y), amplitud, dtype=np.complex128)
    return campo

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


def propagar_ABCD_(U1, tipo_propagacion, d, R, lam, k):
    
    # ===================================================================
    #                  Coordenadas espaciales (TU CÓDIGO INTACTO)
    # ===================================================================

    #-----Muestreo Horizontal-------
    Nx = np.shape(U1)[0] # Número de muestras (píxeles)
    Lx = 0.2 # Tamaño físico de la ventana (mm)
    dx = Lx / Nx # Paso espacial Δx
    dfx = 1 / Lx # Paso en frecuencia Δfx

    #-----Muestreo Vertical-------
    Ny = np.shape(U1)[1]  # Número de muestras (píxeles)
    Ly = 0.2 # Tamaño físico de la ventana (mm)
    dy = Ly / Ny # Paso espacial Δy
    dfy = 1 / Ly # Paso en frecuencia Δfy
    
    # --- Coordenadas de entrada ---
    n = np.arange(Nx) - Nx//2
    m = np.arange(Ny) - Ny//2
    xi_vec = n * dx
    eta_vec = m * dy
    xi_mesh, eta_mesh = np.meshgrid(xi_vec, eta_vec, indexing='xy')

    # ===================================================================
    #                  Coordenadas de frecuencia (fx, fy)
    # ===================================================================

    p = np.arange(Nx) - Nx//2 # Contadores centrados
    q = np.arange(Ny) - Ny//2
    fx_vec = p * dfx 
    fy_vec = q * dfy
    fx, fy = np.meshgrid(fx_vec, fy_vec) 

    # ===================================================================
    #                  LÓGICA CONDICIONAL (AQUÍ ESTÁ EL CAMBIO)
    # ===================================================================

    if tipo_propagacion == "propagar":
        # CASO 1: INTEGRAL DE COLLINS (Viaje en espacio libre)
        # Aquí B = d, por lo tanto B != 0. No hay división por cero.
        
        M = matriz_propagacion(d)
        A, B, C, D = M.ravel()

        # --- Cálculo de la Integral ---

        # Fase cuadrática de entrada (dependiente de A)
        phase1 = (k / (2 * B)) * A * (xi_mesh**2 + eta_mesh**2)
        U_intermediate1 = U1 * np.exp(1j * phase1)

        # Aplicamos shift para centrar, transformamos y regresamos el centro
        U_shifted = np.fft.ifftshift(U_intermediate1)

        if B > 0:
            # El kernel corresponde a una TF
            U_fft_raw = np.fft.fft2(U_shifted) * dx * dy
        else: 
            # El kernel corresponde a una TF inversa.
            U_fft_raw = np.fft.ifft2(U_shifted) * (Nx * Ny) * dfx * dfy
        
        U_fft_unscaled = np.fft.fftshift(U_fft_raw)

        # Coordenadas espaciales de salida: x2 = lambda*B*fx
        x_vec = fx_vec * lam * B
        y_vec = fy_vec * lam * B

        # Paso de muestreo en la salida.
        dx2 = np.abs(x_vec[1] - x_vec[0]) if Nx > 1 else 0
        dy2 = np.abs(y_vec[1] - y_vec[0]) if Ny > 1 else 0
        
        # Mallas de coordenadas 2D de salida.
        x_mesh, y_mesh = np.meshgrid(x_vec, y_vec, indexing='xy')

        # Fase cuadrática de salida (dependiente de D)
        phase2 = (k / (2 * B)) * D * (x_mesh**2 + y_mesh**2)
        
        # Factores globales
        pre_factor_integral = 1 / (1j * lam * B)
        U2 = pre_factor_integral * U_fft_unscaled * np.exp(1j * phase2) 

        return U2, x_mesh, y_mesh, dx2, dy2

    elif tipo_propagacion == "espejo":
        # CASO 2: ESPEJO CURVO (Elemento delgado)
        # Aquí B = 0. No usamos integral. Solo multiplicamos fase.
        # Matriz espejo: [[1, 0], [-2/R, 1]] -> C = -2/R
        
        C = -2 / R
        
        # Fórmula de fase para elemento delgado: exp( i * k/2 * C * r^2 )
        fase_espejo = (k / 2) * C * (xi_mesh**2 + eta_mesh**2)
        
        U2 = U1 * np.exp(1j * fase_espejo)
        
        # En un espejo/lente, las coordenadas de salida son IGUALES a las de entrada
        return U2, xi_mesh, eta_mesh, dx, dy
    
    
# ===================================================================
#                 Creación de la cuchilla
# ===================================================================

def aplicar_filtro_cuchilla(campo_fourier, tipo):

    Nx, Ny = campo_fourier.shape
    cx, cy = Nx // 2, Ny // 2  # Centro óptico (Frecuencia cero)
    porcentaje_corte=0.5
    
    # Crear la máscara (Todo transparente por defecto)
    mascara = np.ones((Nx, Ny))
    
    if tipo == "vertical":
        # Schlieren estándar: Cortar desde un lado (ej. izquierda)
        # Calculamos el índice de corte
        limite = int(Nx * porcentaje_corte)
        mascara[:, :limite] = 0
        
    elif tipo == "horizontal":
        # Cortar desde abajo/arriba
        limite = int(Ny * porcentaje_corte)
        mascara[:limite, :] = 0
        
    elif tipo == "circular":
        # Campo oscuro (Dark Field): Bloquea solo el punto central
        y, x = np.ogrid[:Nx, :Ny]
        # Radio del bloqueo (ajustable, ej. 20 pixeles)
        radio_bloqueo = 20 
        mascara_distancia = (x - cx)**2 + (y - cy)**2
        mascara[mascara_distancia < radio_bloqueo**2] = 0
        
    # Aplicamos el filtro: Campo * Máscara
    campo_filtrado = campo_fourier * mascara
    
    return campo_filtrado


# ===================================================================
#       Generar una simulacion de una columna de calor
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



def onda_sonora_campo():

    # ===================================================================
    #           Constantes fisicas y parametros de muestreo
    # ===================================================================

    # Parámetros de prueba
    L_z = 0.3           # Espesor de la zona de prueba (30 cm)
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

    return S_campo, fase, n_map


def schlieren_1M(U_0, lam, S_campo, tipo_cuchilla):

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

    #Multiplicamos por el objeto como si fuera una trasnmitancia
    camp1 = S3_campo * S_campo

    #Del objeto -> cuchilla
    S5_campo, S5_x_mesh, S5_y_mesh, S5_dx, S5_dy = propagar_ABCD_(camp1, "propagar", d, 0, lam,k)

    #Aplicamos el filtro de la cuchilla
    S_filtred = aplicar_filtro_cuchilla(S5_campo, tipo_cuchilla)

    # Aplicamos la Transformada Inversa (La lente formadora de imagen) que seria la camara o sensor a utilizar
    campo_en_sensor = np.fft.fftshift(np.fft.ifft2(S_filtred))

    # Calculamos la intensidad 
    Imagen_Schlieren = np.abs(campo_en_sensor)**2

    return Imagen_Schlieren

def schlieren_2M(U_0, lam, S_campo, tipo_cuchilla):

    # ===================================================================
    #             Propagación sistema óptico (CONFIGURACIÓN 2 ESPEJOS)
    # ===================================================================

    # Datos fisicos de los espejos (Asumimos simetría R1 = R2)
    f_espejo = 1.0        # Distancia focal 
    R_espejo = 2 * f_espejo # Radio de curvatura (R = 2f)
    d_foco = f_espejo     # Distancia del Espejo 2 al sensor (Foco)          
    k = 2 * np.pi / lam   # Numero de onda 

    # --- Definición del recorrido---

    # La luz se propaga hasta el primer espejo
    camp_m, _, _, _, _ = propagar_ABCD_(U_0, "propagar", f_espejo, 0, lam, k)

    # La luz rebota en el primer espejo.
    camp_m1, _, _, _, _ = propagar_ABCD_(camp_m, "espejo", 0, R_espejo, lam, k)

    # La luz viaja por el aire hasta llegar a donde está el sonido.
    camp_antes_obj, _, _, _, _ = propagar_ABCD_(camp_m1, "propagar", f_espejo/2, 0, lam, k)

    # La luz atraviesa la perturbación.
    camp_despues_obj = camp_antes_obj * S_campo

    # La luz sigue viajando por el aire hasta el segundo espejo.
    camp_antes_m2, _, _, _, _ = propagar_ABCD_(camp_despues_obj, "propagar", f_espejo/2, 0, lam, k)

    # El haz rebota en el segundo espejo y empieza a converger.
    camp_m2, _, _, _, _ = propagar_ABCD_(camp_antes_m2, "espejo", 0, R_espejo, lam, k)

    # La luz viaja hasta el plano focal donde está la cuchilla.
    camp_plano_focal, _, _, _, _ = propagar_ABCD_(camp_m2, "propagar", f_espejo, 0, lam, k)

    # Aplicamos el filtro de la cuchilla
    S_filtred = aplicar_filtro_cuchilla(camp_plano_focal, tipo_cuchilla)

    # PASO 8: Cámara (Transformada Inversa)
    campo_en_sensor = np.fft.fftshift(np.fft.ifft2(S_filtred))

    # Calculamos la intensidad 
    Imagen_Schlieren = np.abs(campo_en_sensor)**2

    return Imagen_Schlieren