#Punto_01

import numpy as np
import matplotlib.pyplot as plt

#Metodo Espectro Angular

# ---------- Parámetros ----------

#-----Muestreo Horizontal-------
Nx = 400        # número de muestras por eje (es mejor utilizar un numero mayor de muestras para mejorar el muestreo)
Lx = 10        # tamaño físico de la ventana (mm)
dx = Lx / Nx    # paso espacial Δ
dfx = 1 / Lx    # paso en frecuencia Δf

#-----Muestreo Vertical-------
Ny = 400        # número de muestras por eje (es mejor utilizar un numero mayor de muestras para mejorar el muestreo)
Ly = 10         # tamaño físico de la ventana (mm)
dy = Ly / Ny    # paso espacial Δ
dfy = 1 / Ly    # paso en frecuencia Δf

# ---------- Coordenadas espaciales ----------
n = np.arange(Nx)  # Contadores
m = np.arange(Ny)
#x = n * dx                # x = Δx * n
#y = m * dx                # y = Δy * m
#X, Y = np.meshgrid(x, y)

x = (np.arange(Nx) - Nx/2) * dx
y = (np.arange(Ny) - Ny/2) * dy
X, Y = np.meshgrid(x, y)

# ---------- Coordenadas de frecuencia ----------
p = np.arange(Nx) #Contadores
q = np.arange(Ny)
fx = p * dfx               # fx = Δfx * p   
fy = q * dfy               # fy = Δfy * q
FX, FY = np.meshgrid(fx, fy)

# ---------- Campo inicial U(x,y,0) ----------

# Abertura circular de radio r0 (amplitud 1 dentro)
r0 = 2 # mm
aperture = (X**2 + Y**2) <= r0**2  #Aqui lo que hacemos es comparar los valores de x y y si estan dentro o fuera del circulo
                                   #si estan dentro su valor es 1 (true) y estan fuera es 0 (false)

U0 = aperture.astype(np.complex128) #Como la funcion anteriro retorna valores booleanos debemso pasarla a valores 
                                    #complejos de matrices para poder hacer FFT


# ---------- A[p,q,0] por FFT ----------

# Aca solo debemos aplicar la trasnsformada de fouriere a la funcio U[n,m,0]
# y multiplicarla por Δx*Δy = Δ², ya que segun la simplificacion y el analisis 
# matematicos y fisico llegamos a esa relacion de aplicar FFT, computacionalmente
# usamos fft2 debido a que estamos trabajndo en dos dimensiones

A0 = np.fft.fft2(U0) * (dx*dy)

# ---------- A[p,q,z] ----------

lam_nm = 650.0           # longitud de onda en nanómetros
lam_mm = lam_nm * 1e-6   # conversión a milímetros
k = (2*np.pi) / lam_mm
z = 10                   # en mm
w = (lam_mm*dfx*dfy)**2
h = np.sqrt(1-((w)*(FX**2 + FY**2)))
Az = A0 * (np.exp(1j*z*k*h))


# ---------- U[n,m,z] por IFFT ----------

# Aca para obtener la funcion dle campo a la slaida U[n,m,z] debemos pasar 
# del espacio de frecuencias al espacio dimensional por eso utilizamos
# la transformada inversa de fourier de la funcion del espectro angular

Uz = np.fft.ifft2(Az) 

# Crear ejes de frecuencia
fx = np.fft.fftfreq(Nx, d=dx)
fy = np.fft.fftfreq(Ny, d=dy)
FX, FY = np.meshgrid(fx, fy)

# ---------- Graficar magnitud de Az centrada ----------
plt.figure(figsize=(6,6))
plt.imshow(np.fft.fftshift(np.abs(Uz)), 
           extent=[fx.min(), fx.max(), fy.min(), fy.max()],
           origin='lower', cmap='inferno')
plt.colorbar(label='|Az(fx,fy)|')
plt.title("Espectro de Fourier con propagación |Az|")
plt.xlabel("fx [1/mm]")
plt.ylabel("fy [1/mm]")

# ---------- Zoom (opcional) ----------
f0 = 1. # rango de frecuencia que quieres ver
plt.xlim(-f0, f0)
plt.ylim(-f0, f0)

plt.show()


# ---------- Re-ordenar el campo U[n,m,z] ----------

# Por la teoria vista de analisis de muestreo vimos que realmente no estamos calculando 
# el orden correcto de nuetsro campo a la salida es decir este esta en posiciones incorrectas
# por ende debemos hacer algo para organizarlo, debemos shiftearlo para que ahora si este centrado

# Usamos fftshift para centrar el campo en el dominio espacial
Uz_shifted = np.fft.fftshift(Uz)

# ---------- Intensidad I[x,y,z] ----------

I = np.abs(A0)**2
I = I / np.max(I)  # Normalizar

# ---------- Visualizar el resultado ----------
extent = [-Lx/2, Lx/2, -Ly/2, Ly/2]
plt.imshow(I, extent=extent, origin='lower')
plt.colorbar()
plt.title("Intensidad |$U(x,y,z)|^2$ a $z$=$10mm$")
plt.show()