# ===================================================================
#                 Sistema óptico: Microscopio 
# ===================================================================

#-----Importamos las funciones para la trasnferencia de rayos ópticos------
from Propagacion import propagar_campo_ABCD
from Propagacion import matriz_propagacion
from Propagacion import lente_delgada
from Propagacion import matriz_del_sistema
from Propagacion import plano_pupila
from Propagacion import Lx, Ly, Nx, Ny 
from scipy.interpolate import interp2d

import matplotlib.pyplot as plt
import numpy as np
import os

# ===================================================================
#                 Definimos el campo de Entrada S(x,y)
# ===================================================================

ruta_script = os.path.dirname(os.path.abspath(__file__))

# --- Función para corregir 'i' por 'j' ---
def convertidor_complejo_i_a_j(bytes_string):
    # loadtxt pasa los datos como "bytes", por eso usamos .decode()
    string_limpio = bytes_string.decode().replace('i', 'j')
    return complex(string_limpio)

nombre_archivo = os.path.join('MuestrasBio', 'MuestraBio_E05.csv')

ruta_completa_archivo = os.path.join(ruta_script, nombre_archivo)

campo_entrada_1 = np.loadtxt(
        ruta_completa_archivo, 
        dtype=np.complex128, 
        delimiter=',',
        converters=convertidor_complejo_i_a_j # <-- ¡AQUÍ ESTÁ LA MAGIA!
    )

# ===================================================================
#                 Calculo de la trayectoria
# ===================================================================

# Datos del sistema todos en [mm]

f_TL = 200              # Focal de la lente
d = f_TL                # Propagacion de P(x,y) a TL
lam = 533 * 1e-6        # Longitud de onda en mm

# Magnificacion =  f_TL / f_MO
# f_MO = f_TL / Magnificacion

Mx = 20                 #20x
f_MO = f_TL / Mx        # Focal de la lente

M_prop1 = matriz_propagacion(f_MO) # Propagacion hasta MO
Lente_MO = lente_delgada(f_MO) # Pasa por M0
M_prop2 = matriz_propagacion(f_MO) # Propagacion hasta P(x,y)

# Los diafragmas no pueden representarse como una matriz ABCD.
# Por ende separamos la propagación en un calculo de dos campos
# Primero calculamos el cmapo hasta la pupila y calculamos el campo de salida 
# y ahi si propagamso ese cmapo de slaida hasta el de slaida final completo


Campo_1 = [M_prop2, Lente_MO, M_prop1]
M_1 = matriz_del_sistema(Campo_1)

#Aproximar a cero los valores muy pequeños
threshold = 1e-15 
M_1[np.abs(M_1) < threshold] = 0.0

print("Matriz hasta P(x,y):\n", M_1)

# ===================================================================
#                    Propagamos el primer campo 
# ===================================================================

L_equal = [0,0]
campo_prop_1, L_out, mesh = propagar_campo_ABCD(campo_entrada_1, L_in, lam, M_1, 'NO', L_equal )

#Multiplicamos el campo por la pupila y para eso debemos definirla


# --- Definición de la Pupila P(x,y) ---
NA = 0.25                # Apertura Numérica (NA) del objetivo (20x/0.25)
#
RPu = f_MO * NA          # Cálculo del radio (en mm) de la pupila P(x,y):
                           # 1. Definición de NA (aprox. paraxial): NA ≈ sin(θ_max) ≈ θ_max
                           # 2. Trazado de rayos: y_pupila = f_MO * θ_rayo
                           # 3. RPu (radio máx) = f_MO * θ_max = f_MO * NA

Le = .39
Ln = .39

Ne = 1080
Nn = 1080

de = Le/Ne
dn = Ln/Nn

Eje_Horizontal = np.arange(-Ne/2,Ne/2)
Eje_Vertical = np.arange(-Nn/2,Nn/2)

e,n = Eje_Horizontal*de,Eje_Vertical*dn
E,N = np.meshgrid(e,n)
Sen = np.sin(200*E*np.pi/Le).astype(np.complex128)

X,Y = plano_pupila(lam,Sen,f_MO, Le, Ln)
P = (X**2+Y**2<=RPu**2)


campo_prop_1_ = campo_prop_1 * P

# ===========================================================================
#  Calculo de la trayectoria : Ahora propagamos del diafragma hacia adelante 
# ===========================================================================

M_prop3 = matriz_propagacion(d) # Propagacion hasta TL
Lente_TL = lente_delgada(f_TL) # Pasa por TL
M_prop4 = matriz_propagacion(f_TL) # Propagacion hasta O(u,v) (Camara)

Campo_2 = [M_prop4, Lente_TL, M_prop3]
M_2 = matriz_del_sistema(Campo_2)

#Aproximar a cero los valores muy pequeños
threshold = 1e-15 
M_2[np.abs(M_2) < threshold] = 0.0

print("Matriz de O(u,v):\n", M_2)

# ===================================================================
#                    Propagamos el campo final O(u,v)
# ===================================================================

#--------------Calculamos el tamaño fisico del sensor de salida-------------

# Resolucion de 8.1 MP
Nx_cam = 2848   # H
Ny_cam = 2848   # V

pixel_ = 0.00274       #sensor (Sony IMX546) El tamaño de cada píxel es: Tamaño de píxel = 2.74 µm (0.00274 mm)

Lx_camara = Nx_cam * pixel_  
Ly_camara = Ny_cam * pixel_  

L_camara  = [Lx_camara, Ly_camara]

campo_prop_2, L_camara_salida, mesh_camara = propagar_campo_ABCD(campo_prop_1_, mesh, lam, M_2, 'YES', L_camara) 


# ===================================================================
#            NUEVO: Modelado de la Cámara Alvium U-811m
# ===================================================================
print("Modelando el muestreo del sensor de la cámara...")
# 1. Especificaciones de la cámara
Nx_cam = 2848 # Píxeles (H)
Ny_cam = 2848 # Píxeles (V)
pixel_ = 0.00274       #sensor (Sony IMX546) 2.74 µm 

Lx_camara = Nx_cam * pixel_  
Ly_camara = Ny_cam * pixel_  

# 2. Coordenadas (u,v) del sensor de la cámara
u_cam_vec = (np.arange(Nx_cam) - Nx_cam // 2) * pixel_
v_cam_vec = (np.arange(Ny_cam) - Ny_cam // 2) * pixel_

# 3. Intensidad simulada "continua"
Intensidad_continua = np.abs(campo_prop_2)**2

# 4. Interpolar la intensidad sobre los píxeles de la cámara
print("Interpolando la intensidad sobre los píxeles del sensor...")
f_interp = interp2d(
    mesh_camara[0][0, :],              # Coordenadas 'u' de la simulación
    mesh_camara[1][:, 0],              # Coordenadas 'v' de la simulación
    Intensidad_continua,    # Intensidad simulada
    kind='linear',          # Interpolación lineal
    fill_value=0            # Rellena con 0 si nos salimos de la ventana
)

# La imagen final tal como la ve la cámara
Imagen_final_muestreada = f_interp(u_cam_vec, v_cam_vec)
print("Muestreo de cámara completado.")

# ===================================================================
#                             GRÁFICAS 
# ===================================================================
print("Generando gráficos...")

# 1. Campo de Entrada (Muestra)
I_entrada = np.abs(campo_entrada_1)**2

# 2. Campo en la Pupila (Filtrado)
I_pupila_filtrada = np.abs(campo_prop_1_)**2

# 3. Campo Final (Imagen Muestreada)
I_final_camara = Imagen_final_muestreada # Usamos la imagen interpolada

# --- Crear la figura con 3 subplots ---
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# --- Gráfico 1: Campo de Entrada ---
ax1.imshow(I_entrada, 
           extent=[-L_in[0]/2, L_in[0]/2, -L_in[1]/2, L_in[1]/2], 
           cmap='gray')
ax1.set_title("1. Campo de Entrada $S(x,y)$")
ax1.set_xlabel(f"ξ (mm) - {Nx} píxeles")
ax1.set_ylabel(f"η (mm) - {Ny} píxeles")

# --- Gráfico 2: Campo Propagado 1 (En la Pupila) ---
ax2.imshow(np.log(I_pupila_filtrada + 1e-10), # Usamos log
           extent=[-L_out[0]/2, L_out[0]/2, -L_out[1]/2, L_out[1]/2], 
           cmap='afmhot')
ax2.set_title(f"2. Campo en Pupila $P(x,y)$ (Radio={RPu:.2f} mm)")
ax2.set_xlabel("x (mm)")
ax2.set_ylabel("y (mm)")

# --- Gráfico 3: Campo Propagado 2 (En la Cámara) ---
ax3.imshow(I_final_camara, 
           extent=[-Lx_camara/2, Lx_camara/2, -Ly_camara/2, Ly_camara/2], 
           cmap='gray')
ax3.set_title(f"3. Imagen Final en Cámara Alvium ({Nx_cam}x{Ny_cam})")
ax3.set_xlabel("u (mm)")
ax3.set_ylabel("v (mm)")

plt.tight_layout()
plt.show()