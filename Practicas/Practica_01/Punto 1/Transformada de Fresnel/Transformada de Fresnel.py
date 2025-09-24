import numpy as np
import matplotlib.pyplot as plt
from scipy.special import fresnel

# ===================================================================
#                   Metodo de la Transformada de Fresnel
# ===================================================================s

# ---------- Parámetros Físicos ----------

a = 4                            # Tamaño horizontal de la rendija en mm
b = 1                            # Tamaño vertical de la rendija en mm
# z = 1000                       # Se reemplaza valor fijo por entrada de usuario
lam_nm = 650                     # Longitud de onda en nanómetros
lam_mm = lam_nm * 1e-6           # Conversión a milímetros

# Solicitar al usuario que ingrese el valor de z
z = float(input("Ingrese la distancia de propagación z en mm: "))

#-----Muestreo Horizontal-------

Nx = 1024                        # número de muestras por eje
Lx = 10                          # tamaño físico de la ventana (mm)
dx = Lx / Nx                     # paso espacial Δx

#-----Muestreo Vertical-------

Ny = 1024                        # número de muestras por eje
Ly = 10                          # tamaño físico de la ventana (mm)
dy = Ly / Ny                     # paso espacial Δy
k = 2 * np.pi / lam_mm

# ===================================================================
#    Restricción para el valor de Z (Método Transformada de Fresnel)
# ===================================================================
# De acuerdo con el material del curso (S07b - Límites de la difracción discreta),
# para que el muestreo del kernel de propagación de la Transformada de Fresnel sea adecuado
# y se evite el aliasing, la distancia de propagación z debe cumplir la siguiente condición:
# z >= (N * dx^2) / lambda
# Donde N es el número de muestras y dx es el paso espacial en el plano de entrada.

z_min = (Nx * dx**2) / lam_mm

# Se evalúa la condición. Si no se cumple, se muestra un aviso y el programa no continúa.
if z < z_min:
    print("\n¡ADVERTENCIA!")
    print(f"El valor de z = {z} mm no es permitido para los parámetros de muestreo actuales.")
    print(f"Para el método de la Transformada de Fresnel, z debe ser mayor o igual a {z_min:.2f} mm.")
    print("Este límite asegura un muestreo adecuado de la fase esférica y evita el 'aliasing'")
    print("Por favor, elija un valor de z dentro del rango permitido.")
else:
    # ---------- Coordenadas espaciales iniciales----------
    
    n0 = np.arange(Nx) - Nx//2       # Contadores
    m0 = np.arange(Ny) - Ny//2
    x0 = n0 * dx
    y0 = m0 * dy
    X0, Y0 = np.meshgrid(x0, y0)
    
    # ---------- Coordenadas espaciales finales----------
    
    dx_ = lam_mm * z / (Nx * dx)
    dy_ = lam_mm * z / (Ny * dy)
    n = n0
    m = m0
    x = n * dx_
    y = m * dy_
    X, Y = np.meshgrid(x, y)
    
    # ===================================================================
    #                       Campo inicial U(x,y,0)
    # ===================================================================
    
    # Apertura rectangular centrada en el origen
    aperture = (abs(X0) <= a/2) * (abs(Y0) <= b/2)
    U0 = aperture.astype(np.complex128)
    
    # ===================================================================
    #                       Campo final U(x,y,z)
    # ===================================================================
    
    def fase_esf_parax(U, k, z, X_coord, Y_coord):
        e_arg = (k / (2 * z)) * (X_coord**2 + Y_coord**2)
        return U * np.exp(1j * e_arg)
    
    def escala(U, k, z, X_coord, Y_coord):
        A = np.exp(1j * k * z) / (1j * lam_mm * z)
        return fase_esf_parax(U, k, z, X_coord, Y_coord) * A
    
    # ===================================================================
    #                   Aplicar fase de entrada
    # ===================================================================
    Uprima = fase_esf_parax(U0, k, z, X0, Y0)
    
    # ===================================================================
    #                       Calcular FFT
    # ===================================================================
    Udobleprima = (dx * dy) * np.fft.fft2(Uprima)
    
    # ===================================================================
    #                   Centrar el espectro
    # ===================================================================
    Udobleprimaorga = np.fft.fftshift(Udobleprima)
    
    # ===================================================================
    #               Escalar y aplicar fase de salida
    # ===================================================================
    Usalida = escala(Udobleprimaorga, k, z, X, Y)
    I = np.abs(Usalida)**2
    I_norm = I/np.max(I)
    
    # ===================================================================
    #                       Funcion analítica
    # ===================================================================
    
    def IF(p,lam_mm,z,X):      #Función para calcular la integral de fresnel alrededor del eje genérico "X"
        NF=(p/2)**2/(lam_mm*z)
        l=np.sqrt(2*NF)*(1-(2*X)/p)
        u=np.sqrt(2*NF)*(1+(2*X)/p)
        Su, Cu=fresnel(u)
        Sl, Cl=fresnel(l)
        return (1/np.sqrt(2))*(Cl+Cu)-(1j/np.sqrt(2))*(Sl+Su)
    
    def amplitud(a,lam_mm,z,X,Y):
        k=2*np.pi/lam_mm
        amp=(np.exp(1j*(k*z))/1j)*IF(a,lam_mm,z,X)*IF(b,lam_mm,z,Y)
        amp_max=np.max(amp)
        return amp/amp_max
    
    # ===================================================================
    #                       Graficar resultados
    # ===================================================================
    
    extent = [-Nx * dx_ / 2, Nx * dx_ / 2, -Ny * dy_ / 2, Ny * dy_ / 2]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
    
    #Expresión analítica
    ax1.set_title(f"|$U(x,y,z)|^2$ z = {z} mm (Expresión analítica)")
    ax1.set_xlabel("x (mm)")
    ax1.set_ylabel("y (mm)")
    im1=ax1.imshow(abs(amplitud(a,lam_mm,z,X,Y))**2,cmap="grey", extent=extent, origin='lower')
    
    # Transformada de Fresnel (ax2)
    im2 = ax2.imshow(I_norm, cmap="grey", extent=extent, origin='lower')
    ax2.set_title(f"|$U(x,y,z)|^2$ z = {z} mm (Transformada de Fresnel numérica)")
    ax2.set_xlabel("x (mm)")
    ax2.set_ylabel("y (mm)")
    fig.subplots_adjust(left=0.09,right=1.01)
    fig.colorbar(im2, ax=[ax1, ax2], label="Intensidad Normalizada")
    plt.show()
