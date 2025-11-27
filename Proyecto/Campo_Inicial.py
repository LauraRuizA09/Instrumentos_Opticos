import numpy as np
import matplotlib.pyplot as plt

# ===================================================================
#               Definicion de funciones
# ===================================================================

def mapeo(resolucion, tamano_fisico):

    x = np.linspace(-tamano_fisico/2, tamano_fisico/2, resolucion)
    y = np.linspace(-tamano_fisico/2, tamano_fisico/2, resolucion)
    X, Y = np.meshgrid(x, y)
    dx = tamano_fisico / resolucion
    return X, Y, dx

def generar_onda_sonora(X, Y, frecuencia_espacial, amplitud_rho):

   # Genera el campo de índice de refracción (n) para una onda sonora esférica
   # Basado en Gladstone-Dale: n = 1 + K * rho
    
    # Radio desde el centro
    R = np.sqrt(X**2 + Y**2)
    
    # Modelado de onda esférica (sinusoidal)
    perturbacion = amplitud_rho * np.sin(2 * np.pi * frecuencia_espacial * R)
    
    # Densidad total
    rho_total = RHO_0 - perturbacion
    
    # Índice de refracción
    n_field = 1.0 + (K_GLADSTONE * rho_total)
    
    return n_field

def calcular_gradientes(n_field, dx):

    # Calcula cuánto cambia el indice de refraccion en cada punto, el n es lo que el sistema Schlieren detecta

    # np.gradient usa diferencias centrales
    grad = np.gradient(n_field, dx) 
    dn_dy = grad[0] # Gradiente en eje vertical (filas)
    dn_dx = grad[1] # Gradiente en eje horizontal (columnas)
    
    return dn_dx, dn_dy

def calcular_desviacion_angular(dn_dx, dn_dy, espesor_z):

    # Calcula el ángulo epsilon que se desvía la luz
    # epsilon_x = (1/n) * Integral(dn/dx * dz)
    # Asumimos aproximación paraxial (n ~ 1) y perturbación constante en Z (espesor_z)

    n_promedio = 1.00029 # Aproximado para aire
    
    # Integración: gradiente * longitud de interacción
    epsilon_x = (1 / n_promedio) * dn_dx * espesor_z
    epsilon_y = (1 / n_promedio) * dn_dy * espesor_z
    
    return epsilon_x, epsilon_y

#def plot_simulacion(campo, titulo, cmap='viridis'):

#    plt.figure(figsize=(6, 5))
#    plt.imshow(campo, cmap=cmap)
#    plt.colorbar()
#    plt.title(titulo)
#    plt.axis('off')
#    plt.show()

def visualizar_completo(n_field, dn_dx, dn_dy):

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Mapa de Índice de Refracción (Equivalente a Densidad)
    # Usamos un mapa de color 'viridis' porque la densidad es siempre positiva
    im1 = axes[0].imshow(n_field, cmap='viridis')
    axes[0].set_title("Mapa de Densidad")
    fig.colorbar(im1, ax=axes[0], label="Índice de refracción $n$")
    axes[0].axis('off')
    
    # Para los gradientes, el valor clave es 0 (sin cambio).
    # Usaremos un mapa 'RdBu' (Rojo-Blanco-Azul). Blanco = 0 gradiente.
    # Calculamos el máximo para que el 0 quede centrado en el color blanco.
    max_val_x = np.max(np.abs(dn_dx))
    max_val_y = np.max(np.abs(dn_dy))
    
    # Gradiente Horizontal (dn/dx)
    # Esto simula una Cuchilla colocada VERTICALMENTE
    im2 = axes[1].imshow(dn_dx, cmap='RdBu', vmin=-max_val_x, vmax=max_val_x)
    axes[1].set_title("Gradiente Horizontal $\\partial n / \\partial x$\n(Simulación: Cuchilla Vertical)")
    fig.colorbar(im2, ax=axes[1])
    axes[1].axis('off')
    
    # Gradiente Vertical (dn/dy)
    # Esto simula una Cuchilla colocada HORIZONTALMENTE
    im3 = axes[2].imshow(dn_dy, cmap='RdBu', vmin=-max_val_y, vmax=max_val_y)
    axes[2].set_title("Gradiente Vertical $\\partial n / \\partial y$\n(Simulación: Cuchilla Horizontal)")
    fig.colorbar(im3, ax=axes[2])
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

# ===================================================================
#               Variables fisicas a utilizar
# ===================================================================

# Constantes que pueden sevrir mas adelante

RHO_0 = 1.225        # kg/m^3 (Densidad base)
K_GLADSTONE = 2.26e-4 # m^3/kg (Constante para luz visible)


resolucion = 800           # Resolución HD
L = 0.3                    # 30 cm de zona de prueba
f = 60                     # Frecuencia de la onda visual (anillos)
A_rho = 0.05               # Amplitud exagerada para verla bien
grosor_z = 0.1             # La onda tiene 10 cm de profundidad


# Crear el espacio
X, Y, dx = mapeo(resolucion, L)

# Generar el objeto invisible
n_mapa = generar_onda_sonora(X, Y, f, A_rho)

# Como dobla la luz (Gradientes)
dndx, dndy = calcular_gradientes(n_mapa, dx)

# Angulos de desviación reales (epsilon)
# Esto es lo que entra al sistema óptico
eps_x, eps_y = calcular_desviacion_angular(dndx, dndy, grosor_z)



visualizar_completo(n_mapa, dndx, dndy)