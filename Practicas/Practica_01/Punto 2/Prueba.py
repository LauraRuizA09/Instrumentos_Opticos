import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Parámetros definidos en el paso 1

k=10  #Número de pasos por milímetro
T = 1/k # Periodo en mm
lam_nm = 633      # Longitud de onda en nanómetros
lam_mm = lam_nm * 1e-6  # Conversión a milímetros
Z=31.595
Lx=4 # Tamaño de ventana horizontal en milímetros
Ly=4 # Tamaño de ventana vertical en milímetros
Nx=2048 #Número de muestras horizontales
Ny=2048 #Número de muestras verticales

# ---------- Campo inicial U(x,y,0) ----------
x0_ =np.linspace(-Lx/2,Lx/2,Nx)
y0_ =np.linspace(-Ly/2,Ly/2,Ny)
X0_, Y0_ = np.meshgrid(x0_, y0_)
aperture = (np.mod(X0_, T) < T/2).astype(float)
U0 = aperture.astype(np.complex128)

# ---------- Campo final U(x,y,z) ----------
def fase_esf_parax(U, k, z, X_coord, Y_coord): #Función auxiliar para muestrear una fase esférica en aproximación paraxial
    e_arg = (k / (2 * z)) * (X_coord**2 + Y_coord**2)
    return U * np.exp(1j * e_arg)

def escala(U, k, z, X_coord, Y_coord): #Función auxiliar para escalar y muestrear el campo de salida de la transformada de Fresnel
    A = np.exp(1j * k * z) / (1j * lam_mm * z)
    return fase_esf_parax(U, k, z, X_coord, Y_coord) * A

def Transformada_de_Fresnel(Lx,Ly,U0,z,lam_mm):
    #-----Muestreo Horizontal-------
    Nx = np.shape(U0)[1]  # número de muestras por eje
    dx = Lx / Nx   # paso espacial Δx

    #-----Muestreo Vertical-------
    Ny = np.shape(U0)[0]    # número de muestras por eje
    dy = Ly / Ny   # paso espacial Δy


    # ---------- Coordenadas espaciales iniciales----------
    n0 = np.arange(Nx) - Nx//2 # Contadores
    m0 = np.arange(Ny) - Ny//2
    x0 = n0 * dx
    y0 = m0 * dy 
    X0, Y0 = np.meshgrid(x0, y0)

    # ---------- Coordenadas espaciales finales----------
    dx_ = lam_mm * z / (Nx * dx)
    dy_ = lam_mm * z / (Ny * dy) 
    n = n0
    m = m0
    x = n * dx_
    y = m * dy_ 
    X, Y = np.meshgrid(x, y)
    k = 2 * np.pi / lam_mm
    # 1. Aplicar fase de entrada
    Uprima = fase_esf_parax(U0, k, z, X0, Y0)

    # 2. Calcular FFT
    Udobleprima = (dx * dy) * np.fft.fft2(Uprima)

    # 3. Centrar el espectro
    Udobleprimaorga = np.fft.fftshift(Udobleprima)

    # 4. Escalar y aplicar fase de salida
    Usalida = escala(Udobleprimaorga, k, z, X, Y)
    return Usalida,X,Y

Uz,X,Y=Transformada_de_Fresnel(Lx,Ly,U0,Z,lam_mm)
I = np.abs(Uz)**2
I_norm = I/np.max(I)

# ---------- Visualización ----------
extent1= [-Lx/2, Lx/2, -Ly/2, Ly/2]
extent2 = [X[0][0], X[0][-1], Y[0][0], Y[-1][0]]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
#Expresión analítica
ax1.set_title(f"|$U(x,y,0)|^2$")
ax1.set_xlabel("x (mm)")
ax1.set_ylabel("y (mm)")
im1=ax1.imshow(abs(U0)**2,cmap="grey", extent=extent1, origin='lower')
# Transformada de Fresnel (ax2)
im2 = ax2.imshow(I_norm, cmap="grey", extent=extent2, origin='lower')
ax2.set_title(f"|$U(x,y,z)|^2$ z = {Z} mm (Transformada de Fresnel)")
ax2.set_xlabel("x (mm)")
ax2.set_ylabel("y (mm)")
ax2.set_xlim(-Lx/2, Lx/2)
ax2.set_ylim(-Ly/2, Ly/2)
fig.subplots_adjust(left=0.09,right=1.01)
fig.colorbar(im2, ax=[ax1, ax2], label="Intensidad Normalizada")
plt.show()
