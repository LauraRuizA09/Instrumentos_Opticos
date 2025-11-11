import numpy as np
import matplotlib.pyplot as plt
from Microscopio import microscopio, plano_pupila
#-------Parámetros Microscopio (todas las distancias en mm)-------
lamda=533*1e-6 #mm
M=20
F_TL=200 #mm
F_MO=F_TL/M  #mm
Le=0.390 #mm
Ln=0.390 #mm
#----------------------Definición del objeto----------------------
Sen=np.flipud(np.loadtxt(r"Practicas\Practica_03\MuestrasBio\MuestraBio_E05.csv",delimiter=",",dtype=complex))
#----------------------Definición de la pupila--------------------
X,Y=plano_pupila(lamda,Sen, F_MO, Le, Ln)
D=3
P=(X**2+Y**2)<=(D/2)
#----------------Campo en el plano de observación-----------------
Suv, escala=microscopio(M,Sen,P,Le,Ln)
A=np.array([[1,2,3],[4,5,6]])
plt.imshow(Suv.real,extent=escala,cmap="grey_r")
plt.show()