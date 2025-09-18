#Punto_01

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import fresnel

#Metodo Espectro Angular

# ---------- Parámetros Físicos ----------
a = 4  # Tamaño horizontal de la rendija en  mm
b = 1  # Tamaño vertical de la rendija en mm
z = 100   # Distancia de propagación en mm
lam_nm = 650         # Longitud de onda en nanómetros
lam_mm = lam_nm * 1e-6   # Conversión a milímetros

#-----Muestreo Horizontal-------
Nx = 1024  # número de muestras por eje (es mejor utilizar un numero mayor de muestras para mejorar el muestreo)
Lx= 10   # tamaño físico de la ventana (mm)
dx = Lx / Nx    # paso espacial Δ
dfx = 1 / Lx    # paso en frecuencia Δf

#-----Muestreo Vertical-------
Ny = 1024     # número de muestras por eje (es mejor utilizar un numero mayor de muestras para mejorar el muestreo)
Ly =10  # tamaño físico de la ventana (mm)
dy = Ly / Ny    # paso espacial Δ
dfy = 1 / Ly    # paso en frecuencia Δf

# ---------- Coordenadas espaciales ----------
n = np.arange(Nx) - Nx//2 # Contadores
m = np.arange(Ny) - Ny//2
x = n * dx                # x = Δx * n
y = m * dx                # y = Δy * m
X, Y = np.meshgrid(x, y)

# ---------- Coordenadas de frecuencia ----------
p = np.arange(Nx) - Nx//2 #Contadores
q = np.arange(Ny) - Ny//2
P,Q=np.meshgrid(p,q)
fx = p * dfx               # fx = Δfx * p   
fy = q * dfy               # fy = Δfy * q
FX, FY = np.meshgrid(fx, fy)

# ---------- Campo inicial U(x,y,0) ----------
# Rendija rectangular de ancho a y altura b
aperture = (abs(X) <= a/2)*(abs(Y) <= b/2)  
U0 = aperture.astype(np.complex128) #Como la funcion anterior retorna valores booleanos debemos pasarla a valores 
                                    #complejos para poder tener términos de fase.
# ---------- A[p,q,0] por FFT ----------

# Acá solo debemos aplicar la transformada de fourier a la funció U[n,m,0]
# y multiplicarla por Δx*Δy = Δ², ya que segun la simplificacion y el analisis 
# matematicos y fisico llegamos a esa relacion de aplicar FFT, computacionalmente
# usamos fft2 debido a que estamos trabajando en dos dimensiones

A0 = np.fft.fft2(U0) * ((dx*dy)) #Cálculo del espectro de entrada
A0_ = np.fft.fftshift(A0)  # Organizamos el espectro de frecuencias para
                           # que la multiplicación por la función de transferencia
                           # se haga de manera organizada
# ---------- A[p,q,z] ----------

def Funcion_de_transferencia(A, a, z, lam_mm): #Definimos la función de transferencia
    k = (2*np.pi) / lam_mm 
    w= 1 - ((lam_mm*dfx)**2) * (P**2 + Q**2)
    m=1j*z*k*np.sqrt(w.astype(np.complex128))  # Permitimos resultados complejos en la raíz
    return A * np.exp(m)                       


Az = Funcion_de_transferencia(A0_,a,z,lam_mm)  # Propagamos el espectro de entrada y
Az_ = np.fft.fftshift(Az)                      # De-centramos el espectro de frecuencias
                                               # por que la IFFT recibe el espectro "desorganizado"

# ---------- U[n,m,z] por IFFT ----------

# Aca para obtener la funcion del campo a la salida U[n,m,z] debemos pasar 
# del espacio de frecuencias al espacio dimensional por eso utilizamos
# la transformada inversa de fourier de la funcion del espectro angular

Uz = (np.fft.ifft2(Az_)) * ((dfx)**2) 

# ---------- Re-ordenar el campo U[n,m,z] ----------

# Por la teoria vista de analisis de muestreo vimos que realmente no estamos calculando 
# el orden correcto de nuestro campo a la salida es decir que este, está en posiciones incorrectas
# por ende debemos hacer algo para organizarlo, debemos shiftearlo para que ahora sí esté centrado.

# Usamos fftshift para centrar el campo en el dominio espacial
Uz_shifted = np.fft.fftshift(Uz)

# ---------- Intensidad I[x,y,z] ----------

I = np.abs(Uz)**2
I_norm = I / np.max(I)  # Normalizar

# ---------- Visualizar el resultado ----------
#Fresnel
def IF(p,lam_mm,z,X):      #Función para calcular la integral de fresnel alrededor del eje genérico "X"
    NF=(p/2)**2/(lam_mm*z)
    l=np.sqrt(2*NF)*(1-(2*X)/p)
    u=np.sqrt(2*NF)*(1+(2*X)/p)
    Su, Cu=fresnel(u)
    Sl, Cl=fresnel(l)
    return (1/np.sqrt(2))*(Cl+Cu)-(1j/np.sqrt(2))*(Sl+Su)
def amplitud(a,lam_mm,z,X,Y):
    k=2*np.pi/lam_mm
    amp=(np.exp(1j*(k*z))/1j)*IF(a,lam_mm,z,X)*IF(b,lam_mm,z,Y)
    amp_max=np.max(amp)
    return amp/amp_max
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
extent = [-Lx/2, Lx/2, -Ly/2, Ly/2] #Dominio espacial
# Expresión analítica (ax1)
ax1.set_title(f"|$U(x,y,z)|^2$ z = {z} mm (Expresión analítica)")
ax1.set_xlabel("x (mm)")
ax1.set_ylabel("y (mm)")
im1=ax1.imshow(abs(amplitud(a,lam_mm,z,X,Y))**2,cmap="grey", extent=extent, origin='lower')
# Espectro angular (ax2)
im2 = ax2.imshow(I_norm, cmap="grey", extent=extent, origin='lower')
ax2.set_title(f"|$U(x,y,z)|^2$ z = {z} mm (Espectro angular numérico)")
ax2.set_xlabel("x (mm)")
ax2.set_ylabel("y (mm)")
fig.subplots_adjust(left=0.09,right=1.01)
fig.colorbar(im2, ax=[ax1, ax2], label="Intensidad Normalizada")
plt.show()