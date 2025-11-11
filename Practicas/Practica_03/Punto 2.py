import numpy as np
import matplotlib.pyplot as plt
from Microscopio import microscopio, plano_pupila
#-------Parámetros Microscopio (todas las distancias en mm)-------
lamda=533e-6  
F_TL=200 
M=20
F_MO=F_TL/M
NA=0.25
RPu=F_MO*NA
#----------------------Definición del objeto----------------------
Le=.39
Ln=.39
Ne=1080
Nn=1080
de=Le/Ne
dn=Ln/Nn
Eje_Horizontal=np.arange(-Ne/2,Ne/2)
Eje_Vertical=np.arange(-Nn/2,Nn/2)
e,n=Eje_Horizontal*de,Eje_Vertical*dn
E,N=np.meshgrid(e,n)
Sen=np.sin(200*E*np.pi/Le).astype(np.complex128)
#----------------------Definición de la pupila--------------------
X,Y=plano_pupila(lamda,Sen,F_MO, Le, Ln)
P=(X**2+Y**2<=RPu**2)
#-----------------------------------------------------------------
Suv, escala=microscopio(M,Sen,P,Le,Ln)
plt.imshow(Suv.real,extent=escala,cmap="grey_r")
plt.show()