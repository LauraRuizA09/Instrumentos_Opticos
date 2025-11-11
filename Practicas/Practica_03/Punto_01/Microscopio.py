# ===================================================================
#                 Sistema óptico: Microscopio 
# ===================================================================

#-----Importamos las funciones para la trasnferencia de rayos ópticos------
from Propagacion import propagar_campo_ABCD
from Propagacion import matriz_propagacion
from Propagacion import lente_delgada
from Propagacion import matriz_del_sistema
from Propagacion import transmitancia_entrada
from Propagacion import plano_pupila
from Propagacion import Lx, Ly, Nx, Ny 

import matplotlib.pyplot as plt
import numpy as np

# ===================================================================
#                 Definimos el campo de Entrada S(x,y)
# ===================================================================

#Ejemplo
campo_entrada_1, L_in = transmitancia_entrada('imagen') # S(ξ,η)


# Creamos el impulso (delta de Dirac) en el centro para calcular la PSF del sistema
#campo_entrada_1 = np.zeros((Ny, Nx), dtype=complex)
#campo_entrada_1[Ny//2, Nx//2] = 1.0 # Impulso en el origen
#L_in = (Lx, Ly)

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

campo_prop_2, L_camara_salida, mesh_camara = propagar_campo_ABCD(campo_prop_1_, L_camara, lam, M_2, 'YES', L_camara) 



# ===================================================================
#                             GRÁFICAS 
# ===================================================================

# -----Intensidades------
I_entrada = np.abs(campo_entrada_1)**2
I_pupila_filtrada = np.abs(campo_prop_1_)**2
I_final_camara = np.abs(campo_prop_2)**2
I_final_camara_ = np.abs(campo_prop_2)**2
I_final_log = np.log(I_final_camara_ + 1e-10) # Usamos log(I) para ver los anillos débiles, Sumamos un bias para evitar log(0)

#---------Guardar informaaicon para el otro punto---------
np.save("Practicas/Practica_03/Punto_01/resultado_microscopio.npy", I_final_camara)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 6))

# --- Gráfico 1: Campo de Entrada ---
ax1.imshow(I_entrada, 
           extent=[-L_in[0]/2, L_in[0]/2, -L_in[1]/2, L_in[1]/2], 
           cmap='gray')
ax1.set_title("1. Campo de Entrada $S(x,y)$")
ax1.set_xlabel("ξ (mm)")
ax1.set_ylabel("η (mm)")

# --- Gráfico 2: Campo Propagado 1 (En la Pupila) ---
# (Usamos log(I) para ver mejor los patrones de difracción)
ax2.imshow(np.log(I_pupila_filtrada + 1e-10), 
           extent=[-L_out[0]/2, L_out[0]/2, -L_out[1]/2, L_out[1]/2], 
           cmap='afmhot')
ax2.set_title(f"2. Campo en Pupila $P(x,y)$ (Radio={RPu:.2f} mm)")
ax2.set_xlabel("x (mm)")
ax2.set_ylabel("y (mm)")

# --- Gráfico 3: Campo Propagado 2 (En la Cámara) ---
ax3.imshow(I_final_camara, 
           extent=[-L_camara_salida[0]/2, L_camara_salida[0]/2, -L_camara_salida[1]/2, L_camara_salida[1]/2], 
           cmap='gray')
ax3.set_title("3. Campo Final en Cámara (Imagen)")
ax3.set_xlabel("u (mm)")
ax3.set_ylabel("v (mm)")

# --- Gráfico 4: Campo Propagado 2 (En la Cámara con el patron de difracción) ---
ax4.imshow(I_final_log, 
           extent=[-L_camara_salida[0]/2, L_camara_salida[0]/2, -L_camara_salida[1]/2, L_camara_salida[1]/2], 
           cmap='afmhot') 
ax4.set_title("3. Campo Final en Cámara (Imagen) $O(u,v)$")
ax4.set_xlabel("u (mm)")
ax4.set_ylabel("v (mm)")


plt.tight_layout()
plt.show()















