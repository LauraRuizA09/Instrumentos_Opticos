import numpy as np
import matplotlib.pyplot as plt

# --- Parámetros de la rejilla y la longitud de onda ---
k_grating = 10  
T = 1 / k_grating  # Periodo en mm
lam_nm = 633      # Longitud de onda en nanómetros
lam_mm = lam_nm * 1e-6  # Conversión a milímetros

# --- Parámetros de muestreo y ventana ---
Nx = 2048 # muestras
Lx = 4  # tamaño físico de la ventana (mm)
dx = Lx / Nx   # paso espacial Δx
Ny = Nx
Ly = 4 
dy = Ly / Ny  

# --- Función para calcular la distancia de Talbot ---
def talbot_distance(L, lam, N):
    return 2 * (L**2) / lam * N

# --- SOLICITAR N AL USUARIO ---
try:
    N = int(input("Ingresa un número entero N para la autoimagen de Talbot: "))
    if N <= 0:
        print("N debe ser un entero positivo.")
        exit()
except ValueError:
    print("Entrada no válida. Por favor, ingresa un número entero.")
    exit()

# --- Calcular la distancia de propagación z ---
z = talbot_distance(T, lam_mm, N)
print(f"Calculando patrón de difracción para z = {z:.4f} mm (N={N})")


# --- Coordenadas espaciales iniciales ---
n0 = np.arange(Nx) - Nx // 2
m0 = np.arange(Ny) - Ny // 2
x0 = n0 * dx
y0 = m0 * dy
X0, Y0 = np.meshgrid(x0, y0)

# --- Coordenadas espaciales finales (dependen de z) ---
dx_ = lam_mm * z / (Nx * dx)
dy_ = lam_mm * z / (Ny * dy)
n = n0
m = m0
x = n * dx_
y = m * dy_
X, Y = np.meshgrid(x, y)

# --- Campo inicial U(x,y,0) (Rejilla Ronchi) ---
aperture = (np.mod(X0, T) < T/2).astype(float)
U0 = aperture.astype(np.complex128)

# --- FUNCIONES DE PROPAGACIÓN ---
k = 2 * np.pi / lam_mm
def fase_esf_parax(U, k, z, X_coord, Y_coord):
    e_arg = (k / (2 * z)) * (X_coord**2 + Y_coord**2)
    return U * np.exp(1j * e_arg)

def escala(U, k, z, X_coord, Y_coord):
    A = np.exp(1j * k * z) / (1j * lam_mm * z)
    return fase_esf_parax(U, k, z, X_coord, Y_coord) * A

# --- PROPAGACIÓN ---
# 1. Aplicar fase de entrada
Uprima = fase_esf_parax(U0, k, z, X0, Y0)
# 2. Calcular FFT
Udobleprima = (dx * dy) * np.fft.fft2(Uprima)
# 3. Centrar el espectro
Udobleprimaorga = np.fft.fftshift(Udobleprima)
# 4. Escalar y aplicar fase de salida
Usalida = escala(Udobleprimaorga, k, z, X, Y)
I = np.abs(Usalida)**2
I_norm = I / np.max(I)

# --- VISUALIZACIÓN ---
extent1 = [-Lx/2, Lx/2, -Ly/2, Ly/2]
extent2 = [-Nx * dx_ / 2, Nx * dx_ / 2, -Ny * dy_ / 2, Ny * dy_ / 2]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.set_title(f"|$U(x,y,0)|^2$ (Entrada - Rejilla)")
ax1.set_xlabel("x (mm)")
ax1.set_ylabel("y (mm)")
im1 = ax1.imshow(abs(U0)**2, cmap="gray", extent=extent1, origin='lower')

im2 = ax2.imshow(I_norm, cmap="gray", extent=extent2, origin='lower')
ax2.set_title(f"|$U(x,y,z)|^2$ z = {z:.4f} mm (N={N})")
ax2.set_xlabel("x (mm)")
ax2.set_ylabel("y (mm)")
ax2.set_xlim(-Lx/2, Lx/2)
ax2.set_ylim(-Ly/2, Ly/2)

fig.subplots_adjust(left=0.09, right=1.01)
fig.colorbar(im2, ax=[ax1, ax2], label="Intensidad Normalizada")
plt.show()