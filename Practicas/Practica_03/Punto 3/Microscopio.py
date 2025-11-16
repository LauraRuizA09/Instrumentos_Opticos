import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
def plano_pupila(lamda,Sen,F_MO, Le, Ln): #Función para obtener las coordenadas del plano pupila
    Ne=np.shape(Sen)[1]# Se debe tener el mismo número de pixeles por largo y ancho que el campo de entrada
    Nn=np.shape(Sen)[0]# Para que sea compatible con el tamaño del campo en la pupila computado por FFT
    dx=lamda*F_MO/(Le) # Se calcula el cambio en el intervalo espacial 
    dy=lamda*F_MO/(Ln)
    x=np.arange(-Ne/2,Ne/2)*dx #Se crean los ejes
    y=np.arange(-Nn/2,Nn/2)*dy
    X,Y=np.meshgrid(x,y) #Se crea la rejilla de coordenadas
    escala=[-Nn*dx/2,Nn*dx/2,-Ne*dy/2,Ne*dy/2]
    return X,Y,escala
def microscopio(M,Sen,P,Le,Ln):
    #Campo a la entrada de la pupila ( coordenadas (x,y) )
    Sxy=np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Sen)))
    #Campo a la salida de la pupila ( coordenadas (x,y) )
    Sprimaxy=Sxy*P
    #Campo en el plano de observación (Coordenadas (u,v))
    Suv=np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Sprimaxy)))
    escala=[-Le*M/2,Le*M/2,-Ln*M/2,Ln*M/2] #Se calcula la ventana de las coordenadas U,V
    return Suv, escala

def complex_to_rgb_hsv(complex_matrix, saturation_value=1.0):
    """
    Convierte una matriz compleja en una matriz RGB utilizando el mapeo HSV.
    La fase se mapea al tono (color), y la amplitud se mapea al brillo (valor).

    Parámetros:
    -----------
    complex_matrix : np.ndarray
        La matriz 2D de números complejos a visualizar.
    saturation_value : float, opcional
        El valor de saturación (S) para el espacio HSV. 
        Debe estar entre 0.0 y 1.0. 
        Un valor de 1.0 produce colores puros. Por defecto es 1.0.

    Retorna:
    --------
    np.ndarray
        Una matriz 3D (M x N x 3) de valores RGB normalizados entre 0.0 y 1.0,
        lista para ser ploteada con matplotlib.imshow().
    """

    # 1. Obtener Fase y Amplitud de la matriz compleja
    phase = np.angle(complex_matrix)
    amplitude = np.abs(complex_matrix)

    # 2. Normalizar la fase y la amplitud para el espacio HSV (valores de 0 a 1)

    # H (Hue - Tono): Mapeamos la fase de [-pi, +pi] a [0, 1]
    H = (phase + np.pi) / (2 * np.pi)

    # S (Saturation - Saturación): Usamos el valor constante proporcionado
    # Aseguramos que esté dentro del rango [0, 1]
    S = np.full_like(H, np.clip(saturation_value, 0.0, 1.0))
    
    # V (Value - Brillo): Normalizamos la amplitud por su valor máximo
    # Evitar división por cero si la amplitud máxima es 0 (matriz de ceros)
    max_amplitude = np.max(amplitude)
    if max_amplitude == 0:
        V = np.zeros_like(H)
    else:
        V = amplitude / max_amplitude

    # 3. Crear la matriz HSV y convertirla a RGB
    hsv_matrix = np.stack([H, S, V], axis=-1)
    rgb_matrix = hsv_to_rgb(hsv_matrix)

    return rgb_matrix