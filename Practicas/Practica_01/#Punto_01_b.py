#Punto 01

import numpy as np
import matplotlib.pyplot as plt

#Metodo Transformada de Fresnel

# ---------- Parámetros ----------

#-----Muestreo Horizontal-------
Nx = 1024        # número de muestras por eje (es mejor utilizar un numero mayor de muestras para mejorar el muestreo)
Lx = 10        # tamaño físico de la ventana (mm)
Lx0= 10      # tamaño del plano de apertura
dx = Lx / Nx    # paso espacial Δ
dx0 = Lx0/Nx  # paso en frecuencia Δf

#-----Muestreo Vertical-------
Ny = 1024        # número de muestras por eje (es mejor utilizar un numero mayor de muestras para mejorar el muestreo)
Ly = 10         # tamaño físico de la ventana (mm)
Ly0 = 10    # tamaño del plano de apertura
dy = Ly / Ny    # paso espacial Δ
dy0 = Ly0/Ny    # paso en frecuencia Δf


# Contadores
n = np.arange(Nx) - Nx//2 
m = np.arange(Ny) - Ny//2

# ---------- Coordenadas espaciales en el plano de observacion ----------
x = n * dx                # x = Δx * n
y = m * dx                # y = Δy * m
X, Y = np.meshgrid(x, y)

#Contadores
n0 = np.arange(Nx) - Nx//2 
m0 = np.arange(Ny) - Ny//2

# ---------- Coordenadas en el plano de la apertura ----------
x0 = n0 * dx0                # x = Δx * n
y0 = m0 * dx0              # y = Δy * m
X0, Y0 = np.meshgrid(x0, y0)


# ---------- Campo inicial U(x,y,0) ----------

# Apertura cuadrada de lado L (amplitud 1 dentro)
L = 4  # mm
aperture = (np.abs(X) <= L/2) & (np.abs(Y) <= L/2)  
# Si (x,y) están dentro del cuadrado → True (1), si no → False (0)

U0 = aperture.astype(np.complex128) #Como la funcion anteriro retorna valores booleanos debemso pasarla a valores 
                                    #complejos de matrices para poder hacer FF


# ---------- Prepara U'[n,m,0]----------

lam_nm = 650.0           # longitud de onda en nanómetros
lam_mm = lam_nm * 1e-6   # conversión a milímetros
k = (2*np.pi) / lam_mm
z = 10                 # en mm
U0_prima = U0 * np.exp(1j * (k/(2*z)) * ((X0*dx0)**2 + (Y0*dy0)**2))

# ---------- U''[n,m,z] por FFT ----------

U0_prima2 = np.fft.fft2(U0_prima) * (dx0**2)
U0_prima2_ = np.fft.fftshift(U0_prima2)  # Centramos el espectro de frecuencias

# ---------- Visualización en coordenadas físicas ----------
plt.figure(figsize=(6,6))
plt.imshow(np.abs(U0_prima), extent=[x0.min(), x0.max(), y0.min(), y0.max()],
           cmap="gray", origin="lower")
plt.title("Apertura cuadrada en coordenadas físicas (mm)")
plt.xlabel("x [mm]")
plt.ylabel("y [mm]")
plt.colorbar(label="Amplitud")
plt.show()


# ---------- Escalar U[n,m,z] ----------

U_z = (np.exp(1j * k * z) / (1j * lam_mm * z)) * np.exp(1j * (k/(2*z)) * (X**2 + Y**2)) * U0_prima2_

# ---------- Re-ordenar el campo U_z ----------

U_z_ = np.fft.fftshift(U_z)

# ---------- Intensidad I[x,y,z] ----------

I = np.abs(U_z_)**2
I = I / np.max(I)  # Normalizar

# ---------- Graficar ----------
extent = [-Lx/2, Lx/2, -Ly/2, Ly/2]
plt.imshow(I, extent=extent, origin='lower')
plt.colorbar()
plt.title("Intensidad |$U(x,y,z)|^2$ a $z$=$10mm$")
plt.show()