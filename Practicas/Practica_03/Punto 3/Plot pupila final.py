import numpy as np
import matplotlib.pyplot as plt
from Microscopio import microscopio, plano_pupila

#----------------------Parámetros del microscopio----------------------
lamda = 533e-6  # mm
M = 20
F_TL = 200      # mm
F_MO = F_TL/M   # mm
Le = 0.390      # mm
Ln = 0.390      # mm
NA = 0.25
RPu = F_MO * np.tan(np.arcsin(NA))
D = 2*RPu       # diámetro total de la pupila


#----------------------Objeto----------------------
Sen = np.flipud(np.loadtxt(r"Practicas\Practica_03\MuestrasBio\MuestraBio_E05.csv",
                           delimiter=",", dtype=complex))

#----------------------Pupila y malla----------------------
X, Y, _ = plano_pupila(lamda, Sen, F_MO, Le, Ln)

#----------------------Frecuencia central y pupila final----------------------
fb = 1/100 * NA * F_MO
alpha_opt = 3*np.pi/2
beta = 0.5

P_total = (X**2 + Y**2) <= (RPu)**2
P_central = (X**2 + Y**2) <= fb**2
P = np.ones_like(X, dtype=complex)
P[P_central] = beta * np.exp(1j * alpha_opt)
P[~P_total] = 0.0

#----------------------Simulación del microscopio----------------------
Suv, extent_out = microscopio(M, Sen, P, Le, Ln)
intensity = np.abs(Suv)**2
phase_obj = np.angle(Sen)  # fase del campo inicial

#----------------------Comparación intensidad vs fase inicial----------------------
fig, axes = plt.subplots(1, 2, figsize=(12,5))

# Intensidad de salida
im0 = axes[0].imshow(intensity, cmap='Greys_r', origin='lower', extent=extent_out)
axes[0].set_xlabel("u (mm)")
axes[0].set_ylabel("v (mm)")
axes[0].set_title(r" $|S(u,v)|^2$")
fig.colorbar(im0, ax=axes[0])
extent3=[390/2,-390/2,390/2,-390/2]
# Fase inicial del objeto en unidades de pi
im1 = axes[1].imshow(np.fliplr(np.flipud(phase_obj)), cmap='twilight', origin='lower', vmin=-np.pi/2, vmax=np.pi/2,extent=extent3)
axes[1].set_xlabel(r"$\xi(\mu m)$")
axes[1].set_ylabel(r"$\eta(\mu m)$")
axes[1].set_title(r"$\arg(S(\xi, \eta))$")

# Ajuste de colorbar en múltiplos de pi
cbar = fig.colorbar(im1, ax=axes[1])
pi_ticks = np.arange(-np.pi/2, np.pi/2 + 0.1, np.pi/4)  # -π/2, -π/4, 0, π/4, π/2
cbar.set_ticks(pi_ticks)
cbar.set_ticklabels([r"$-\pi/2$", r"$-\pi/4$", r"$0$", r"$\pi/4$", r"$\pi/2$"])

plt.tight_layout()
plt.show()
