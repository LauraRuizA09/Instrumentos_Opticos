import numpy as np
import matplotlib.pyplot as plt
from Microscopio import microscopio, plano_pupila

#-------Parámetros Microscopio (todas las distancias en mm)-------
lamda = 533*1e-6  # mm
M = 20
F_TL = 200  # mm
F_MO = F_TL/M  # mm
Le = 0.390  # mm  -> 390 µm
Ln = 0.390  # mm  -> 390 µm
NA = 0.25
RPu = F_MO*np.tan(np.arcsin(NA))

#----------------------Definición del objeto----------------------
Sen = np.flipud(np.loadtxt(
    r"Practicas\Practica_03\MuestrasBio\MuestraBio_E05.csv",
    delimiter=",", dtype=complex
))

# Dimensiones en píxeles
N_eta, N_xi = Sen.shape

# Coordenadas físicas (en µm)
xi = np.linspace(-Le/2, Le/2, N_xi) * 1e3   # mm→µm
eta = np.linspace(-Ln/2, Ln/2, N_eta) * 1e3 # mm→µm

#----------------------Cálculo amplitud y fase---------------------
amplitud = np.abs(Sen)
fase = np.angle(Sen) / np.pi  # fase normalizada a π

plt.figure(figsize=(12, 5))

# ---------------------- Amplitud ----------------------
plt.subplot(1, 2, 1)
plt.imshow(amplitud, cmap='gray',
           extent=[xi.min(), xi.max(), eta.min(), eta.max()],
           origin='lower',     # <--- IMPORTANTE
           vmin=0, vmax=1.1)
plt.title(r"$|S(\xi,\eta)|$")
plt.xlabel(r"$\xi \; [\mu m]$")
plt.ylabel(r"$\eta \; [\mu m]$")
plt.colorbar()

# ---------------------- Fase ----------------------
plt.subplot(1, 2, 2)
im = plt.imshow(fase, cmap='twilight',
                extent=[xi.min(), xi.max(), eta.min(), eta.max()],
                origin='lower',    # <--- IMPORTANTE
                vmin=-np.pi/2, vmax=np.pi/2)

plt.title(r"$\arg(S(\xi,\eta))$")
plt.xlabel(r"$\xi \; [\mu m]$")
plt.ylabel(r"$\eta \; [\mu m]$")

cbar = plt.colorbar(im)
cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
cbar.set_ticklabels([
    r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"
])

plt.tight_layout()
plt.show()
