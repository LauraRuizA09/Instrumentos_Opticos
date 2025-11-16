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
Sen = np.flipud(np.loadtxt(r"Practicas\Practica_03\MuestrasBio\MuestraBio_E05.csv",
                           delimiter=",", dtype=complex))

#----------------------Pupila y malla----------------------
X, Y, escala = plano_pupila(lamda, Sen, F_MO, Le, Ln)

#----------------------Barrido continuo de fracciones de D----------------------
n_points = 50
fractions_D = np.logspace(-3, 0, n_points)  # de 0.001 a 1 fracción de D
rms_contrasts = []

alpha = 0.5  # atenuación de la componente central

for frac in fractions_D:
    # Definir pupila con región central atenuada
    P_total = X**2 + Y**2 <= (D/2)**2
    P_central = X**2 + Y**2 <= ((D/2)*frac)**2
    P = P_total - alpha*P_central
    
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
plt.plot(fractions_D, rms_contrasts, 'o-', color='navy')
plt.xscale('log')
plt.xlabel(r'Fracción de $f_b / f_{\mathrm{NA}}$')
plt.ylabel('Contraste RMS')
plt.title('Barrido de contraste RMS vs fracción de $f_b/f_{NA}$')
plt.grid(True, which='both', ls='--')
plt.show()