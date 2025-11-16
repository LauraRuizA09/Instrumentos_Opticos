import numpy as np
import matplotlib.pyplot as plt
from Microscopio import microscopio, plano_pupila

#-------Parámetros Microscopio (todas las distancias en mm)-------
lamda = 533*1e-6  # mm
M = 20
F_TL = 200  # mm
F_MO = F_TL / M  # mm
Le = 0.390  # mm
Ln = 0.390  # mm
NA = 0.25
RPu = F_MO * np.tan(np.arcsin(NA))

#----------------------Definición del objeto----------------------
Sen = np.flipud(np.loadtxt(r"Practicas\Practica_03\MuestrasBio\MuestraBio_E05.csv",
                           delimiter=",", dtype=complex))

#----------------------Definición de la pupila--------------------
X, Y, escala1 = plano_pupila(lamda, Sen, F_MO, Le, Ln)
D = RPu*2
P1 = X**2 + Y**2 <= (D/2)**2
P2 = X**2 + Y**2 <= (D/(2*100))**2
P = P1 + (P2 * -0.5)  # α=0.5 en el centro

#----------------------Campo observado----------------------------
Suv, escala2 = microscopio(M, Sen, P, Le, Ln)

#----------------------Plot conjunto------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12,5))

# Pupila
im0 = axes[0].imshow(P, extent=escala1, origin='lower', cmap='Greys_r')
axes[0].set_title('Pupila de campo oscuro')
axes[0].set_xlabel('x [mm]')
axes[0].set_ylabel('y [mm]')
plt.colorbar(im0, ax=axes[0], label='Transmisión')

# Intensidad observada
im1 = axes[1].imshow(np.abs(Suv)**2, extent=escala2, origin='lower', cmap='Greys_r')
axes[1].set_title('Intensidad registrada')
axes[1].set_xlabel('u [mm]')
axes[1].set_ylabel('v [mm]')
plt.colorbar(im1, ax=axes[1], label='Intensidad')

plt.tight_layout()
plt.show()
