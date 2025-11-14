# ===================================================================
#                     Test USAF 1951 T20-CPG
# ===================================================================

import numpy as np
import matplotlib.pyplot as plt
from Funciones import calcular_resolucion_usaf
from Funciones import graficar_perfil_horizontal
import matplotlib.image as mpimg


# ===================================================================
#       Cargamos la imagen resultante de nuetsro sistema optico
# ===================================================================

ruta_del_resultado = "Practicas/Practica_03/Punto_01/resultado_microscopio.npy"
I_final_camara = np.load(ruta_del_resultado)

plt.figure(figsize=(8, 6))
plt.imshow(I_final_camara, cmap='gray')
plt.colorbar(label='Intensidad')
plt.title("Imagen Resultante en la Cámara")
plt.xlabel("Píxeles (x)")
plt.ylabel("Píxeles (y)")
plt.savefig("Practicas/Practica_03/Punto_02/Resultados/Test USAF bajo el microscopio.png")
plt.show()

# ===================================================================
#         Calculamos la resoluciones teoricas y experimentales
# ===================================================================

# Viendo la imagen anteriro podemos decir que grupo y que elemento son nuestro 
# limite de resolucion para asi calcular la resolucion del instrumento y compararla con la teorica

calcular_resolucion_usaf()


# ===================================================================
#                Analisis de perfiles de intensidad
# ===================================================================

# Aca calculamos los perfiles de inetensidad de las lineas horizontales de varios
# elementos que estan en el limite de la resolucion para ais comparar y saber de 
# manaera cuantitativa que elemnto y que grupo no se distingue bien y poder calcular
# la resolcuion definitiva dle sistema

imagenes_y_elementos = [
    ("Practicas/Practica_03/Punto_02/LImites Resolucion (Cualit.)/G0E4.png", 4,0),
    ("Practicas/Practica_03/Punto_02/LImites Resolucion (Cualit.)/G0E5.png", 5,0),
    ("Practicas/Practica_03/Punto_02/LImites Resolucion (Cualit.)/G0E6.png", 6,0),
    ("Practicas/Practica_03/Punto_02/LImites Resolucion (Cualit.)/G1E1.png", 1,1),
    ("Practicas/Practica_03/Punto_02/LImites Resolucion (Cualit.)/G1E2.png", 2,1),
    ("Practicas/Practica_03/Punto_02/LImites Resolucion (Cualit.)/G1E3.png", 3,1),
    ("Practicas/Practica_03/Punto_02/LImites Resolucion (Cualit.)/G1E4.png", 4,1),
    ("Practicas/Practica_03/Punto_02/LImites Resolucion (Cualit.)/G1E5.png", 5,1),
    ("Practicas/Practica_03/Punto_02/LImites Resolucion (Cualit.)/G1E6.png", 6,1)
]


# ===================================================================
#                   Mostrar los resultados
# ===================================================================

#Contamos cuántos gráficos vamos a hacer
N = len(imagenes_y_elementos)

# Decidimos la forma de la rejilla (Grid)
ncols = 3
nrows = (N + ncols - 1) // ncols 

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 8, nrows * 6))
axes_flat = axes.flatten()

fig.suptitle("Análisis de Perfiles Horizontales de Intensidad", fontsize=16)


colores = [
    ('k', 'c'),    # Colores para el Elemento 4 G0
    ('k', 'm'),    # Colores para el Elemento 5 G0
    ('k', 'y'),    # Colores para el Elemento 6 G0
    ('k', 'b'),    # Colores para el Elemento 1 G1
    ('k', 'r'),    # Colores para el Elemento 2 G1
    ('k', 'g'),    # Colores para el Elemento 3 G1
    ('k', 'm'),    # Colores para el Elemento 4 G1
    ('k', 'c'),    # Colores para el Elemento 5 G1
]

# ===================================================================
#                   Iterar y mostrar los resultados
# ===================================================================

for i, (filename, element_num, group) in enumerate(imagenes_y_elementos):

    ax = axes_flat[i]
    
    # Calculamos los perfiles
    perfil_crudo, perfil_suave, sigma = graficar_perfil_horizontal(filename)
    
    # El eje Y es la posición en píxeles, el eje X es la intensidad
    eje_y = range(len(perfil_crudo))
    
    # Asignar colores según el índice
    # Usamos % (módulo) para repetir colores si hay más elementos que colores definidos
    color_crudo, color_suave = colores[i % len(colores)]
    

    #ax.plot(perfil_crudo, eje_y, color=color_crudo, alpha=0.3, label=f'Elemento {element_num} Grupo {group}') 
    ax.plot(perfil_suave, eje_y, color=color_suave, lw=2, label=f'Elemento {element_num} Grupo {group}')
    ax.set_xlabel("Intensidad Promediada")
    ax.set_ylabel("Posición (píxeles)")
    ax.legend() 
    ax.grid(True)
    ax.set_xlim(0, 120)

# Ocultamos los ejes que no se usaron
for j in range(i + 1, len(axes_flat)):
    axes_flat[j].axis('off')

# Ajustamos el espaciado 
#fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # 'rect' deja espacio para el suptitle

# Guardamos el resultado para el informe
plt.savefig("Practicas/Practica_03/Punto_02/Resultados/Perfiles de Intensidad Completos.png")
plt.show()
