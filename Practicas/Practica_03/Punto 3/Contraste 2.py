import numpy as np
import matplotlib.pyplot as plt
from Microscopio import microscopio, plano_pupila

#-------Parámetros Microscopio (distancias en mm)-------
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
Sen = np.flipud(np.loadtxt(
    r"Practicas\Practica_03\MuestrasBio\MuestraBio_E05.csv",
    delimiter=",", dtype=complex))

#----------------------Pupila y malla----------------------
X, Y, escala = plano_pupila(lamda, Sen, F_MO, Le, Ln)

#----------------------Barrido de cargas de fase (en rad)----------------------
n_points = 1000
alpha_values = np.linspace(0, 6*np.pi, n_points)  # de 0 a 6π
f_b = (1/100) * NA  # frecuencia de corte tomada de campo oscuro
rms_contrasts = []

for alpha in alpha_values:
    # Definir pupila con fase central
    P_total = X**2 + Y**2 <= (D/2)**2
    P_central = X**2 + Y**2 <= ((D/2)*f_b/NA)**2  # f_b en términos de D
    P = np.ones_like(X, dtype=complex)
    P[P_central] = np.exp(1j*alpha)
    P[~P_total] = 0  # fuera de la pupila

    # Simulación del microscopio
    Suv, _ = microscopio(M, Sen, P, Le, Ln)
    intensity = np.abs(Suv)**2

    # Contraste RMS
    I_mean = np.mean(intensity)
    I_rms = np.sqrt(np.mean((intensity - I_mean)**2))
    C_rms = I_rms / I_mean
    rms_contrasts.append(C_rms)

#----------------------Plot del contraste RMS----------------------
plt.figure(figsize=(6,4))
plt.plot(alpha_values, rms_contrasts, 'o-', color='navy')

# Personalizar ticks del eje x en múltiplos de π automáticamente
xticks = np.arange(0, 6.1*np.pi, np.pi/2)
xticklabels = []
for t in xticks:
    multiple = t / np.pi
    if multiple == 0:
        xticklabels.append(r'$0$')
    elif multiple == 1:
        xticklabels.append(r'$\pi$')
    elif multiple == 2:
        xticklabels.append(r'$2\pi$')
    elif multiple == 3:
        xticklabels.append(r'$3\pi$')
    elif multiple == 4:
        xticklabels.append(r'$4\pi$')
    elif multiple == 5:
        xticklabels.append(r'$5\pi$')
    elif multiple == 6:
        xticklabels.append(r'$6\pi$')
    else:  # para fracciones como 1/2, 3/2, etc.
        numerator = int(round(multiple*2))
        xticklabels.append(r'${}\pi/2$'.format(numerator))

plt.xticks(xticks, xticklabels)
plt.xlabel(r'Carga de fase $\alpha$')
plt.ylabel('Contraste RMS')
plt.title(r'Barrido de contraste RMS vs carga de fase')
plt.grid(True, which='both', ls='--')
plt.show()
