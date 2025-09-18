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

# <--- CORRECCIÓN: Volvemos a la lógica correcta. Fijamos el muestreo del plano de SALIDA (observación)
dx = 0.01  # Paso de muestreo en el plano de salida
dy = dx    

# Coordenadas en el plano de SALIDA (donde observamos)
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

# <--- CORRECCIÓN: El extent inicial lo definimos con los parámetros de salida que conocemos
extent = [-Nx * dx / 2, Nx * dx / 2, -Ny * dy / 2, Ny * dy / 2]

# Gráfica analítica
ax1.set_title("Solución Analítica")
ax1.set_xlabel("x (mm)")
ax1.set_ylabel("y (mm)")
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
# ===================================================================
#                    FUNCIÓN DE ACTUALIZACIÓN 
# ===================================================================
def update(val):
    z = z_slider.val

    # --- Cálculo Numérico ---s
    dx0 = (lam_mm * z) / (Nx * dx)
    dy0 = (lam_mm * z) / (Ny * dy)
    
    # Coordenadas de ENTRADA
    n0 = np.arange(Nx) - Nx // 2
    m0 = np.arange(Ny) - Ny // 2
    x0 = n0 * dx0
    y0 = m0 * dy0
    X0, Y0 = np.meshgrid(x0, y0)
    
    #------Generamos U[n,m,0]------
    aperture = (np.abs(X0) <= L / 2) & (np.abs(Y0) <= L / 2)
    U0 = aperture.astype(np.complex128)
    
    #------Calculamos U'[n0,m0]------
    phase_in = np.exp(1j * (k / (2 * z)) * (X0**2 + Y0**2))
    U0_prima = U0 * phase_in
    
    #------Calculamos U''[n,m,z]------
    # <--- CORRECCIÓN: El escalado debe ser con el área de píxel de ENTRADA.
    U0_prima2 = np.fft.fft2(U0_prima) * (dx0 * dy0)
    
    #------Escalamos U''[n,m,z]------
    phase_out = np.exp(1j * (k / (2 * z)) * (X**2 + Y**2))
    pre_factor = (np.exp(1j * k * z) / (1j * lam_mm * z))
    U_z = pre_factor * phase_out * U0_prima2
    
    #------Reordenamos el campo------
    U_z_shifted = np.fft.fftshift(U_z)
    
    I_numerical = np.abs(U_z_shifted)**2
    if np.max(I_numerical) > 0:
        I_numerical /= np.max(I_numerical)

    # --- Cálculo Analítico ---
    # Se usa la rejilla de salida X, Y, que es la correcta
    I_analytical = np.abs(amplitud(a,lam_mm, z, X, Y))**2
    if np.max(I_analytical) > 0:
        I_analytical /= np.max(I_analytical)

    # --- Actualización de las Gráficas ---
    extent = [-Nx * dx / 2, Nx * dx / 2, -Ny * dy / 2, Ny * dy / 2]
    im1.set_data(I_analytical)
    im1.set_extent(extent)
    im2.set_data(I_numerical)
    im2.set_extent(extent)
    
    fig.suptitle(f"Comparación de Difracción de Fresnel (z = {z:.2f} mm)", fontsize=16)
    fig.canvas.draw_idle()

# ===================================================================
#                  CREACIÓN Y CONEXIÓN DEL SLIDER
# ===================================================================
ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
z_slider = Slider(
    ax=ax_slider,
    label='Distancia z (mm)',
    valmin=100,
    valmax=3000,
    valinit=10, # <--- CORRECCIÓN: Un valor inicial dentro del rango
)

z_slider.on_changed(update)
update(1000) # Llamada inicial con un valor del rango

plt.show()