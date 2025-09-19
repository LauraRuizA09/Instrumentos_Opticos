import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Parámetros definidos en el paso 1

k=10/1 
T = 1/k # Periodo en mm
lam_nm = 633      # Longitud de onda en nanómetros
lam_mm = lam_nm * 1e-6  # Conversión a milímetros
z=31.595
#-----Muestreo Horizontal-------
Nx = 2048 # muestras
Lx = 4  # mm
dx = Lx / Nx   # paso espacial Δx

#-----Muestreo Vertical-------
Ny = Nx    # número de muestras por eje
Ly = 4 # tamaño físico de la ventana (mm)
dy = Ly / Ny   # paso espacial Δy

# ---------- Coordenadas espaciales iniciales----------
n0 = np.arange(Nx) - Nx//2 # Contadores
m0 = np.arange(Ny) - Ny//2
x0 = n0 * dx
y0 = m0 * dy 
X0, Y0 = np.meshgrid(x0, y0)

k = 2 * np.pi / lam_mm
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
# ---------- Campo inicial U(x,y,0) ----------
aperture = (np.mod(X0, T) < T/2).astype(float)
U0 = aperture.astype(np.complex128)
# ---------- Campo final U(x,y,z) ----------
def fase_esf_parax(U, k, z, X_coord, Y_coord):
    e_arg = (k / (2 * z)) * (X_coord**2 + Y_coord**2)
    return U * np.exp(1j * e_arg)

def escala(U, k, z, X_coord, Y_coord):
    A = np.exp(1j * k * z) / (1j * lam_mm * z)
    return fase_esf_parax(U, k, z, X_coord, Y_coord) * A

# 1. Aplicar fase de entrada
Uprima = fase_esf_parax(U0, k, z, X0, Y0)

# 2. Calcular FFT
Udobleprima = (dx * dy) * np.fft.fft2(Uprima)

# 3. Centrar el espectro
Udobleprimaorga = np.fft.fftshift(Udobleprima)

# 4. Escalar y aplicar fase de salida
Usalida = escala(Udobleprimaorga, k, z, X, Y)
I = np.abs(Usalida)**2
I_norm = I/np.max(I)
# ---------- Visualización ----------
extent1= [-Lx/2, Lx/2, -Ly/2, Ly/2]
extent2 = [-Nx * dx_ / 2, Nx * dx_ / 2, -Ny * dy_ / 2, Ny * dy_ / 2]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
#Expresión analítica
ax1.set_title(f"|$U(x,y,0)|^2$")
ax1.set_xlabel("x (mm)")
ax1.set_ylabel("y (mm)")
im1=ax1.imshow(abs(U0)**2,cmap="grey", extent=extent1, origin='lower')
# Transformada de Fresnel (ax2)
im2 = ax2.imshow(I_norm, cmap="grey", extent=extent2, origin='lower')
ax2.set_title(f"|$U(x,y,z)|^2$ z = {z} mm (Transformada de Fresnel numérica)")
ax2.set_xlabel("x (mm)")
ax2.set_ylabel("y (mm)")
ax2.set_xlim(-Lx/2, Lx/2)
ax2.set_ylim(-Ly/2, Ly/2)
fig.subplots_adjust(left=0.09,right=1.01)
fig.colorbar(im2, ax=[ax1, ax2], label="Intensidad Normalizada")
plt.show()
