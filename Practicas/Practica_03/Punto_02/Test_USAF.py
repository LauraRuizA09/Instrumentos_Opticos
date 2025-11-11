import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import cv2


ruta_del_resultado = "Practicas/Practica_03/Punto_01/resultado_microscopio.npy"
I_final_camara = np.load(ruta_del_resultado)

plt.imshow(I_final_camara, cmap='gray')
plt.title("Imagen Final (Cargada desde Archivo)")
plt.show()



