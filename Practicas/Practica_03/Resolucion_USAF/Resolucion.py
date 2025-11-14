import numpy as np
import matplotlib.pyplot as plt
from Microscopio import microscopio, plano_pupila
from PIL import Image
#-------Parámetros Microscopio (todas las distancias en mm)-------
lamda=533e-6  
F_TL=200 
M=20
F_MO=F_TL/M
NA=0.25
RPu=F_MO*NA
#----------------------Definición del objeto----------------------
#Coordenadas plano objeto
Le=.39
Ln=.39
Ne=1080
Nn=1080
de=Le/Ne
dn=Ln/Nn
Eje_Horizontal=np.arange(-Ne/2,Ne/2)
Eje_Vertical=np.arange(-Nn/2,Nn/2)
e,n=Eje_Horizontal*de,Eje_Vertical*dn
E,N=np.meshgrid(e,n) # Rejilla de coordenadas para definir un objeto de manera analítica

#Apartado para cargar un objeto a partir de una imagen blanco y negro
ruta_imagen = r"Practicas/Practica_03/Punto_02/Imagenes Test USAF/T-20-final-rev-1-400x400.jpg" # Reemplaza con tu ruta
img = Image.open(ruta_imagen).convert('L')
img_array = np.array(img) / 255.0
Sen = np.flipud(img_array).astype(complex)

#----------------------Definición del plano pupila--------------------
X,Y=plano_pupila(lamda,Sen,F_MO, Le, Ln)
P=(X**2+Y**2<=RPu**2)
#-----------------------------------------------------------------
Suv, escala=microscopio(M,Sen,P,Le,Ln)
plt.imshow(abs(Suv)**2/np.max(abs(Suv)**2),extent=escala,cmap="grey")
plt.show()