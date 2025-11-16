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
X, Y, escala1 = plano_pupila(lamda, Sen, F_MO, Le, Ln)

#----------------------Frecuencia central (campo oscuro)----------------------
fb = 1/100 * NA * F_MO  # mm^-1 en coordenadas espaciales

#----------------------Pupila con carga de fase óptima----------------------
alpha_opt = 3*np.pi/2
P_total = (X**2 + Y**2) <= (RPu)**2
P_central = (X**2 + Y**2) <= fb**2
P = np.ones_like(X, dtype=complex)
P[P_central] = np.exp(1j * alpha_opt)
P[~P_total] = 0.0

#----------------------Simulación del microscopio----------------------
Suv, escala2 = microscopio(M, Sen, P, Le, Ln)
intensity = np.abs(Suv)**2

#----------------------Visualización 2D completa----------------------
plt.imshow(intensity, cmap='Greys_r', origin='lower',
           extent=escala2)
plt.colorbar(label="Intensidad")
plt.xlabel("u (mm)")
plt.ylabel("v (mm)")
plt.title(r"$|S(u,v)|^2$")
plt.show()
