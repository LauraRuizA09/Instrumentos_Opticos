#Punto_01

import numpy as np

#Metodo Espectro Angular

# ---------- Parámetros ----------

#-----Muestreo Horizontal-------
Nx = 512        # número de muestras por eje (es mejor utilizar un numero mayor de muestras para mejorar el muestreo)
Lx = 10.0       # tamaño físico de la ventana (mm)
dx = Lx / Nx     # paso espacial Δ
dfx = 1.0 / Lx   # paso en frecuencia Δf

#-----Muestreo Vertical-------
Ny = 512        # número de muestras por eje (es mejor utilizar un numero mayor de muestras para mejorar el muestreo)
Ly = 10.0       # tamaño físico de la ventana (mm)
dy = Ly / Ny     # paso espacial Δ
dfy = 1.0 / Ly   # paso en frecuencia Δf

# ---------- Coordenadas espaciales ----------
n = np.arange(Nx)  # Contadores
m = np.arange(Ny)
x = n * dx                # x = Δx * n
y = m * dx                # y = Δy * m
X, Y = np.meshgrid(x, y)

# ---------- Coordenadas de frecuencia ----------
p = np.arange(Nx) #Contadores
q = np.arange(Ny) 
fx = p * dfx               # fx = Δfx * p
fy = q * dfy               # fy = Δfy * q
FX, FY = np.meshgrid(fx, fy)

# ---------- Campo inicial U(x,y,0) ----------

# Abertura circular de radio r0 (amplitud 1 dentro)
r0 = 1.0  # mm
aperture = (X**2 + Y**2) <= r0**2  #Aqui lo que hacemos es comparar los valores de x y y si estan dentro o fuera del circulo
                                   #si estan dentro su valor es 1 (true) y estan fuera es 0 (false)

U0 = aperture.astype(np.complex128) #Como la funcion anteriro retorna valores booleanos debemso pasarla a valores 
                                    #complejos de matrices para poder hacer FFT



# ---------- A[p,q,0] por FFT ----------

# Aca solo debemos aplicar la trasnsformada de fouriere a la funcio U[n,m,0]
# y multiplicarla por Δx*Δy = Δ², ya que segun la simplificacion y el analisis 
# matematicos y fisico llegamos a esa relacion de aplicar FFT, computacionalmente
# usamos fft2 debido a que estamos trabajndo en dos dimensiones

A0 = np.fft.fft2(U0) * (dx**2)

# ---------- A[p,q,z] ----------

lam_nm = 650.0           # longitud de onda en nanómetros
lam_mm = lam_nm * 1e-6   # conversión a milímetros
k = (2*np.pi) / lam_mm
z = 10                   # en mm
Az = A0 * (np.exp(1j*z*k*np.sqrt(1-(lam_mm*dfx*dfy)*(FX**2 + FY**2))))


# ---------- U[n,m,z] por IFFT ----------

# Aca para obtener la funcion dle campo a la slaida U[n,m,z] debemos pasar 
# del espacio de frecuencias al espacio dimensional por eso utilizamos
# la transformada inversa de fourier de la funcion del espectro angular

Uz = np.fft.fft2(Az) * (dfx*dfy)

# ---------- Re-ordenar el campo U[n,m,z] ----------

# Por la teoria vista de analisis de muestreo vimos que realmente no estamos calculando 
# el orden correcto de nuetsro campo a la salida es decir este esta en posiciones incorrectas
# por ende debemos hacer algo para organizarlo, debemos shiftearlo para que ahora si este centrado