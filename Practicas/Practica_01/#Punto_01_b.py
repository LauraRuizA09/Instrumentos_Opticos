#Punto 01

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import fresnel
from matplotlib.widgets import Slider


# ===================================================================
#           METODO TRANSFORMADA DE FRESNEL (FFT)
# ===================================================================

# ---------- Parámetros ----------
Nx = 1024                # número de muestras
Ny = 1024
lam_nm = 650             # longitud de onda en nanómetros
lam_mm = lam_nm * 1e-6   # conversión a milímetros
k = (2 * np.pi) / lam_mm
L = 4  # Lado de la apertura cuadrada en mm
a = L
b = L

# ---------- Muestreo y Coordenadas  ----------

# Se define el paso en el plano de observación (salida)
dx = 0.01  #  1/100 de mm   
dy = dx    # Δ = dy = dx

# Coordenadas en el plano de observación (salida)
n = np.arange(Nx) - Nx // 2
m = np.arange(Ny) - Ny // 2
x = n * dx
y = m * dy
X, Y = np.meshgrid(x, y)


# ===================================================================
#                CONFIGURACIÓN DE LA GRÁFICA INICIAL
# ===================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(left=0.1, bottom=0.25)

# Gráfica analítica
ax1.set_title("Solución Analítica")
ax1.set_xlabel("x (mm)")
ax1.set_ylabel("y (mm)")
extent = [-Nx * dx / 2, Nx * dx / 2, -Ny * dy / 2, Ny * dy / 2]
im1 = ax1.imshow(np.zeros((Nx, Ny)), cmap="gray", extent=extent, origin='lower', vmin=0, vmax=1)

# Gráfica numérica
ax2.set_title("Cálculo Numérico")
ax2.set_xlabel("x (mm)")
im2 = ax2.imshow(np.zeros((Nx, Ny)), cmap="gray", extent=extent, origin='lower', vmin=0, vmax=1)

# Barra de color
cbar_ax = fig.add_axes([0.92, 0.25, 0.02, 0.6])
fig.colorbar(im2, cax=cbar_ax, label="Intensidad Normalizada")

# ===================================================================
#                 FUNCIONES PARA SOLUCIÓN ANALÍTICA
# ===================================================================
def IF(p, lam_mm, z, X_coord):
    NF = (p / 2)**2 / (lam_mm * z)
    l = np.sqrt(2 * NF) * (1 - (2 * X_coord) / p)
    u = np.sqrt(2 * NF) * (1 + (2 * X_coord) / p)
    Su, Cu = fresnel(u)
    Sl, Cl = fresnel(l)
    return (1 / np.sqrt(2)) * (Cl - Cu) -(1j / np.sqrt(2)) * (Sl + Su)

def amplitud(a, b, lam_mm, z, X, Y):
    k = 2*np.pi/lam_mm
    amp = (np.exp(1j*(k*z))/1j)*IF(a,lam_mm,z,X)*IF(b,lam_mm,z,Y)
    amp_max = np.max(amp)
    return amp/amp_max

# ===================================================================
#                    FUNCIÓN DE ACTUALIZACIÓN 
# ===================================================================
def update(val):
    z = z_slider.val

    # --- Cálculo Numérico ---
    dx0 = (lam_mm * z) / (Nx * dx)    # Δ0 = dy0 = dx0
    dy0 = (lam_mm * z) / (Ny * dy)    # Condicion del metodo de Fresnel -> ΔΔ0 = λz/N 
    n0 = np.arange(Nx) - Nx // 2
    m0 = np.arange(Ny) - Ny // 2
    x0 = n0 * dx0
    y0 = m0 * dy0
    X0, Y0 = np.meshgrid(x0, y0)
    

    #------Generamos U[n,m,0]------
    aperture = (np.abs(X0) <= L / 2) & (np.abs(Y0) <= L / 2)
    U0 = aperture.astype(np.complex128)
    
    #------Calculamos U'[n,m0]------
    phase_in = np.exp(1j * (k / (2 * z)) * (X0**2 + Y0**2))
    U0_prima = U0 * phase_in
    
    #------Calculamos U''[n,m,z]------
    U0_prima2 = np.fft.fft2(U0_prima) * (dx0 * dy0)
    
    #------EScalamos U''[n,m,z]------
    phase_out = np.exp(1j * (k / (2 * z)) * (X**2 + Y**2))
    pre_factor = (np.exp(1j * k * z) / (1j * lam_mm * z))
    U_z = pre_factor * phase_out * U0_prima2
    
    #------Reordenamos el campo------
    U_z_shifted = np.fft.fftshift(U_z)
    
    I_numerical = np.abs(U_z_shifted)**2

    if np.max(I_numerical) > 0:
        I_numerical = I_numerical / np.max(I_numerical)

    # --- Cálculo Analítico ---
    I_analytical = np.abs(amplitud(a, b, lam_mm, z, X, Y))**2
    if np.max(I_analytical) > 0:
        I_analytical = I_analytical / np.max(I_analytical)

    # --- Actualización de las Gráficas ---
    im1.set_data(I_analytical)
    im2.set_data(I_numerical)
    fig.suptitle(f"Comparación de Difracción de Fresnel (z = {z:.2f} mm)", fontsize=16)
    fig.canvas.draw_idle()

# ===================================================================
#                  CREACIÓN Y CONEXIÓN DEL SLIDER
# ===================================================================
ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
z_slider = Slider(
    ax=ax_slider,
    label='Distancia z (mm)',
    valmin=1,
    valmax=4000,
    valinit=10,
)

z_slider.on_changed(update)
update(10) # Llamada inicial

plt.show()
