import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ===================================================================
#      Simulación Espectro Angular para un campo U de una imagen
# ===================================================================


# ===================================================================
#                       Cargar la imagen                         
# ===================================================================

imagen=Image.open(r"/home/laura/Documentos/GitHub/Instrumentos_Opticos/Practicas/Practica_01/Punto 04/Mediciones Laboratorio/26.png").convert('L')         # Se carga en escala de grises
data=(np.flipud(np.array(imagen)))                     # Se convierte en un numpy array y se voltea para que los ejes de pixel del método imshow y espaciales coincidan                                                                                                  # Se calcula o se define el umbral para la binarización
data_bin=data/np.max(data)                             # Se aplica el umbral de binarización

# ===================================================================
#                       Metodo Espectro Angular
# ===================================================================

z = -66                              # Distancia de propagación en mm
lam_nm = 633                         # Longitud de onda en nanómetros
lam_mm = lam_nm * 1e-6               # Conversión a milímetros
k = 2 * np.pi / lam_mm               # Número de onda en mm^-1


#-----Muestreo Horizontal-------

Nx = np.shape(data)[1]               # número de muestras por eje (es mejor utilizar un numero mayor de muestras para mejorar el muestreo)
Lx= 6.6                            # tamaño físico de la ventana (mm) restringido por la camara 6.66
dx = Lx / Nx                         # paso espacial Δ
dfx = 1 / Lx                         # paso en frecuencia Δf

#-----Muestreo Vertical-------

Ny = np.shape(data)[0]               # número de muestras por eje (es mejor utilizar un numero mayor de muestras para mejorar el muestreo)
Ly = 5.32                           # tamaño físico de la ventana (mm) restringido por la camara 5.32
dy = Ly / Ny                         # paso espacial Δ
dfy = 1 / Ly                         # paso en frecuencia Δf

# ---------- Coordenadas espaciales ----------

n = np.arange(Nx) - Nx//2            # Contadores
m = np.arange(Ny) - Ny//2
x = n * dx                           # x = Δx * n
y = m * dx                           # y = Δy * m
X, Y = np.meshgrid(x, y)

# ---------- Coordenadas de frecuencia ----------

p = np.arange(Nx) - Nx//2            #Contadores
q = np.arange(Ny) - Ny//2
P,Q=np.meshgrid(p,q)
fx = p * dfx                         # fx = Δfx * p   
fy = q * dfy                         # fy = Δfy * q
FX, FY = np.meshgrid(fx, fy)

# ===================================================================
#              Definimos el metodo como una funcion 
# ===================================================================

def Funcion_de_transferencia(A, d, lam_mm): #Definimos la función de transferencia
    k = (2*np.pi) / lam_mm
    w= 1 - ((lam_mm)**2) * ((P*dfx)**2 + (Q*dfy)**2)
    m=1j*d*k*np.sqrt(w.astype(np.complex128))  # Permitimos resultados complejos en la raíz
    return A * np.exp(m)

def Espectro_Angular(U0,z,lam_mm):

    # ===================================================================
    #                           A[p,q,0] por FFT
    # ===================================================================

    # Acá solo debemos aplicar la transformada de fourier a la funció U[n,m,0]
    # y multiplicarla por Δx*Δy, ya que segun la simplificacion y el analisis
    # matematicos y fisico llegamos a esa relacion de aplicar FFT, computacionalmente
    # usamos fft2 debido a que estamos trabajando en dos dimensiones
    A0 = np.fft.fft2(U0) * ((dx*dy)) #Cálculo del espectro de entrada
    A0_ = np.fft.fftshift(A0)  # Organizamos el espectro de frecuencias para
                                # que la multiplicación por la función de transferencia
                                # se haga de manera organizada

    # ===================================================================
    #           A[p,q,z] por Función de Transferencia
    # ===================================================================
    Az = Funcion_de_transferencia(A0_,z,lam_mm)  # Propagamos el espectro de entrada y
                                                 # Propagamos el espectro de entrada y
    Az_ = np.fft.fftshift(Az)                      # De-centramos el espectro de frecuencias
                                               # por que la IFFT recibe el espectro "desorganizado"
    # ===================================================================
    #                      U[n,m,z] por IFFT
    # ===================================================================
    # Aca para obtener la funcion del campo a la salida U[n,m,z] debemos pasar
    # del espacio de frecuencias al espacio dimensional por eso utilizamos
    # la transformada inversa de fourier de la funcion del espectro angular
    Uz = (np.fft.ifft2(Az_)) * ((dfx*dfy))

    # ===================================================================
    #                   Re-ordenar el campo U[n,m,z]
    # ===================================================================

    # Por la teoria vista de analisis de muestreo vimos que realmente no estamos calculando 
    # el orden correcto de nuestro campo a la salida es decir que este, está en posiciones incorrectas
    # por ende debemos hacer algo para organizarlo, debemos shiftearlo para que ahora sí esté centrado.

    # Usamos fftshift para centrar el campo en el dominio espacial
    #Uz_shifted = np.fft.fftshift(Uz)

    return Uz


# ===================================================================
#                        Campo inicial U(x,y,0)
# ===================================================================

aperture = np.sqrt(data_bin)    #Le sacamos la raiz cuadrada a la imagen para obtener el campo 
                                # ya que lo que tenia inicialmente era la intensidad

U0 = aperture.astype(np.complex128) 

# ===================================================================
#                        Analisis del Campo U0
# ===================================================================
"""
extent = [-Lx/2, Lx/2, -Ly/2, Ly/2] #Dominio espacial
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))

fig.suptitle("Análisis del Campo $U(x,y)$", fontsize=16, fontweight='bold')

# Campo de salida parte real (ax1)
ax1.set_title(r"$\Re\{|U(x,y,250mm)|\}$")
ax1.set_xlabel("x (mm)")
ax1.set_ylabel("y (mm)")
im1 = ax1.imshow(U0.real, cmap="grey", extent=extent, origin='lower')

# Campo de salida parte imaginaria (ax2)
im2 = ax2.imshow(U0.imag, cmap="bone", extent=extent, origin='lower')
ax2.set_title(r"$\Im\{|U(x,y,250mm)|\}$")
ax2.set_xlabel("x (mm)")
ax2.set_ylabel("y (mm)")
fig.subplots_adjust(left=0.09, right=1.01)
fig.colorbar(im1, ax=ax1) # Es buena práctica asociar el colorbar al eje
fig.colorbar(im2, ax=ax2)


# Asegúrate de que la ruta exista. Usa el objeto 'fig' que contiene ambos plots.
fig.savefig(r"/home/laura/Documentos/GitHub/Instrumentos_Opticos/Practicas/Practica_01/Punto 04/Resultados/Resultados/Analisis U0.png", dpi=300, bbox_inches='tight')

# Finalmente, muestra el gráfico en pantalla
plt.show()
"""
#Aqui es donde notamos que el campo no tiene parte imaginaria debido a que partimos de la intensidad entonces el proceso para 
#obtener el campo a partir de la intensidad nos indica que hay una perdida de informacion en este caso de fase por ende 
# intentamos agregarle una fase esferica para ver si obtenemos un campo con parte imaginaria y parte real

# ===================================================================
#                      Recuperar fase perdida
# ===================================================================

# Fase esférica
R = 1000  # Radio de curvatura de la onda esférica en mm.
         # R > 0 para onda divergente, R < 0 para onda convergente.

# Creación del término de fase esférica
fase_esferica = np.exp(1j * k / (2 * R) * (X**2 + Y**2))

# El campo inicial ahora es la apertura multiplicada por la fase esférica
U0_sphere = aperture.astype(np.complex128) * fase_esferica

Uz = Espectro_Angular(U0_sphere,z,lam_mm)  # Campo a la salida 
Uz_no_sphere = Espectro_Angular(U0,z,lam_mm)    # Campo a la salida sin fase esférica


# ===================================================================
#                Graficas del campo con fase esférica
# ===================================================================

extent = [-Lx/2, Lx/2, -Ly/2, Ly/2] #Dominio espacial
fig1, axs1 = plt.subplots(1, 3, figsize=(18, 10))

fig1.suptitle("Análisis del Campo $Plane$ $Sphere$ $U_0(x,y)$", fontsize=16, fontweight='bold', y = 0.82)

# Campo de salida parte real (Posición: Fila 0, Columna 0)
axs1[0].set_title(r"$\Re\{U'(x,y,250mm)\}$")
axs1[0].set_xlabel("x (mm)")
axs1[0].set_ylabel("y (mm)")
im1 = axs1[0].imshow(Uz.real, cmap="grey", extent=extent, origin='lower')
fig1.colorbar(im1, ax=axs1[0], fraction=0.046, pad=0.04)

# Campo de salida parte imaginaria (Posición: Fila 0, Columna 1)
axs1[1].set_title(r"$\Im\{U'(x,y,250mm)\}$")
axs1[1].set_xlabel("x (mm)")
axs1[1].set_ylabel("y (mm)")
im2 = axs1[1].imshow(Uz.imag, cmap="bone", extent=extent, origin='lower')
fig1.colorbar(im2, ax=axs1[1], fraction=0.046, pad=0.04)

# Modulo cuadrado (Intensidad) (Posición: Fila 0, Columna 2)
intensidad = np.abs(Uz)**2
axs1[2].set_title(r"$|U'(x,y,250mm)|^2$")
axs1[2].set_xlabel("x (mm)")
axs1[2].set_ylabel("y (mm)")
im3 = axs1[2].imshow(intensidad, cmap="hot", extent=extent, origin='lower')
fig1.colorbar(im3, ax=axs1[2], fraction=0.046, pad=0.04)

# ===================================================================
#               Graficas del campo sin fase esférica
# ===================================================================

extent = [-Lx/2, Lx/2, -Ly/2, Ly/2] #Dominio espacial
fig2, axs2 = plt.subplots(1, 3, figsize=(18, 10))

fig2.suptitle("Análisis del Campo $Plane$ $Wave$ $U_0(x,y)$", fontsize=16, fontweight='bold', y = 0.82)

# Campo de salida parte real (Posición: Fila 1, Columna 0)
fase = np.angle(Uz)
axs2[0].set_title(r"$\Re\{U'(x,y,250mm)\}$")
axs2[0].set_xlabel("x (mm)")
axs2[0].set_ylabel("y (mm)")
im4 = axs2[0].imshow(Uz_no_sphere.real, cmap="grey", extent=extent, origin='lower')
fig2.colorbar(im4, ax=axs2[0], fraction=0.046, pad=0.04)

# Campo de salida parte imaginaria (Posición: Fila 1, Columna 1)
axs2[1].set_title(r"$\Im\{U'(x,y,250mm)\}$")
axs2[1].set_xlabel("x (mm)")
im5 = axs2[1].imshow(Uz_no_sphere.imag, cmap="bone", extent=extent, origin='lower')
fig2.colorbar(im5, ax=axs2[1], fraction=0.046, pad=0.04)

# Modulo cuadrado (Intensidad) (Posición: Fila 1, Columna 2)
intensidad_ = np.abs(Uz_no_sphere)**2
axs2[2].set_title(r"$|U'(x,y,250mm)|^2$")
axs2[2].set_xlabel("x (mm)")
im6 = axs2[2].imshow(intensidad_, cmap="hot", extent=extent, origin='lower')
fig2.colorbar(im6, ax=axs2[2], fraction=0.046, pad=0.04)

# Ajusta el espaciado para que no se solapen
fig1.tight_layout(rect=[0, 0, 1, 0.96]) # Ajustar rect para el suptitle
fig2.tight_layout(rect=[0, 0, 1, 0.96]) # Ajustar rect para el suptitle

# Guardar la figura completa en un archivo
fig1.savefig("/home/laura/Documentos/GitHub/Instrumentos_Opticos/Practicas/Practica_01/Punto 04/Resultados/Resultados/Campo PW.png", dpi=300, bbox_inches='tight')
fig2.savefig("/home/laura/Documentos/GitHub/Instrumentos_Opticos/Practicas/Practica_01/Punto 04/Resultados/Resultados/Campo SW.png", dpi=300, bbox_inches='tight')

plt.show()

# Aqui notamos que no es solo multiplicar por una fase esferica, sino que debemos conocer mas terminos
# de la fase para poder recuperar la parte imaginaria del campo, por ende
# continuaremos con el campo U0 sin fase esferica para ver que resultados obtenemos

# ===================================================================
#               Visualizar el resultado 
# ===================================================================

Uz_no_sphere = Espectro_Angular(U0,z,lam_mm)    # Campo a la salida sin fase esférica
I = np.abs(Uz_no_sphere)**2                     # Intensidad a la salida
I_norm = I / np.max(I)                          # Intensidad normalizada

extent = [-Lx/2, Lx/2, -Ly/2, Ly/2]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

fig.suptitle("Resultado Contrapropagación de $U(x,y)$", fontsize=16, fontweight='bold')

# Campo de entrada 
ax1.set_title(f"|$U(x,y,0)|^2$")
ax1.set_xlabel("x (mm)")
ax1.set_ylabel("y (mm)")
im1 = ax1.imshow(np.abs(U0)**2, cmap="grey", extent=extent, origin='lower')
fig.colorbar(im1, ax=ax1, label="Intensidad") # Barra de color para el primer plot

# Espectro angular 
ax2.set_title(f"|$U(x,y,z)|^2$ z = {z} mm")
ax2.set_xlabel("x (mm)")
ax2.set_ylabel("y (mm)")
im2 = ax2.imshow(I_norm, cmap="grey", extent=extent, origin='lower', vmin=0, vmax=1)
fig.colorbar(im2, ax=ax2, label="Intensidad Normalizada") # Barra de color para el segundo plot
fig.tight_layout(pad=1.5)

# Guardar la figura completa en un archivo 
fig.savefig("/home/laura/Documentos/GitHub/Instrumentos_Opticos/Practicas/Practica_01/Punto 04/Resultados/Resultados/Campo Final (Z=66).png", dpi=300, bbox_inches='tight')

plt.show()
