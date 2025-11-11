import numpy as np
import matplotlib.pyplot as plt


def plano_pupila(lamda,Sen,F_MO, Le, Ln):
    Ne=np.shape(Sen)[1]
    Nn=np.shape(Sen)[0]
    dx=lamda*F_MO/(Le)
    dy=lamda*F_MO/(Ln)
    x=np.arange(-Ne/2,Ne/2)*dx
    y=np.arange(-Nn/2,Nn/2)*dy
    X,Y=np.meshgrid(x,y)
    return(X,Y)

def microscopio(lamda,M,Sen,P,Le,Ln):
    #Campo en la pupila ( coordenadas (x,y) )
    Sxy=np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Sen)))
    Sprimaxy=Sxy*P
    # Campo en el plano de observación (Coordenadas (u,v))
    Suv=np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Sprimaxy)))
    escala=[-Le*M/2,Le*M/2,-Ln*M/2,Ln*M/2]
    return Suv, escala