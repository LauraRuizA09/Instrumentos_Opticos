import numpy as np
import matplotlib.pyplot as plt
def plano_pupila(lamda,Sen,F_MO, Le, Ln): #Función para obtener las coordenadas del plano pupila
    Ne=np.shape(Sen)[1]# Se debe tener el mismo número de pixeles por largo y ancho que el campo de entrada
    Nn=np.shape(Sen)[0]# Para que sea compatible con el tamaño del campo en la pupila computado por FFT
    dx=lamda*F_MO/(Le) # Se calcula el cambio en el intervalo espacial 
    dy=lamda*F_MO/(Ln)
    x=np.arange(-Ne/2,Ne/2)*dx #Se crean los ejes
    y=np.arange(-Nn/2,Nn/2)*dy
    X,Y=np.meshgrid(x,y) #Se crea la rejilla de coordenadas
    return (X,Y)
def microscopio(M,Sen,P,Le,Ln):
    #Campo a la entrada de la pupila ( coordenadas (x,y) )
    Sxy=np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Sen)))
    #Campo a la salida de la pupila ( coordenadas (x,y) )
    Sprimaxy=Sxy*P
    #Campo en el plano de observación (Coordenadas (u,v))
    Suv=np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Sprimaxy)))
    escala=[-Le*M/2,Le*M/2,-Ln*M/2,Ln*M/2] #Se calcula la ventana de las coordenadas U,V
    return Suv, escala