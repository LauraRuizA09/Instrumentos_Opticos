import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, Label, Entry, Button, StringVar, Radiobutton, Frame, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# ===================================================================
#          Simulador de Efecto Talbot para Rejilla Rochi
# ===================================================================


# --- Parámetros Fisicos ---

k_grating = 10              # Frecuencia espacial de la rejilla (ciclos/mm)
T = 1 / k_grating           # Periodo en mm
lam_nm = 633                # Longitud de onda en nanómetros
lam_mm = lam_nm * 1e-6      # Conversión a milímetros

# --- Parámetros de muestreo ---

Nx = 3000                   # Muestras en x
Lx = 2                      # Tamaño físico de la ventana en x (mm)
dx = Lx / Nx                # Paso espacial Δx
Ny = Nx                     # Muestras en y
Ly = 2                      # Tamaño físico de la ventana en y (mm)   
dy = Ly / Ny                # Paso espacial Δy

# ===================================================================
#                  Configuración distancias TALBOT
# ===================================================================

def talbot_distance(L, lam, N):
    #Calcula la distancia para una autoimagen de Talbot normal
    return 2 * (L**2) / lam * N

def phase_reversed(L, lam, N):
    #Calcula la distancia para una autoimagen de Talbot con cambio de fase de pi
    return ((2 * N + 1) * (L**2)) / lam

def subimagenes_talbot(L, lam, N):
    #Calcula la distancia para una autoimagen de Talbot (Fase de pi medios)
    return ((N - 1/2) * (L**2)) / lam

# ===================================================================
#             Funciones para la propagación de Fresnel
# ===================================================================

k = 2 * np.pi / lam_mm
def fase_esf_parax(U, z, X_coord, Y_coord):
    e_arg = (k / (2 * z)) * (X_coord**2 + Y_coord**2)
    return U * np.exp(1j * e_arg)

def escala(U, z, X_coord, Y_coord):
    A = np.exp(1j * k * z) / (1j * lam_mm * z)
    return fase_esf_parax(U, z, X_coord, Y_coord) * A

# ===================================================================
#                              Simulación
# ===================================================================

def run_simulation():
    try:
        N = int(n_entry.get())
        if N <= 0:
            messagebox.showerror("Error", "N debe ser un entero positivo.")
            return
    except ValueError:
        messagebox.showerror("Error", "Entrada no válida. Por favor, ingresa un número entero.")
        return

    # Obtener la función de distancia seleccionada
    selection = function_choice.get()
    if selection == "talbot_distance":
        z = talbot_distance(T, lam_mm, N)
        title_type = f"Autoimagen de Talbot (N={N})"
    elif selection == "phase_reversed":
        z = phase_reversed(T, lam_mm, N)
        title_type = f"Imagen con fase revertida (N={N})"
    elif selection == "subimagenes_talbot":
        z = subimagenes_talbot(T, lam_mm, N)
        title_type = f"Subimagen de Talbot (N={N})"
    else:
        return


    # Se calcula la distancia mínima 'z' requerida para la validez numérica
    z_min = (Nx * dx**2) / lam_mm
    
    #Este paso realmente NO es necesario ya que la minima distancia talbot ya cumple con la condicion
    # pero se deja para referencia en caso de querer cambiar parametros y que no se cumpla

    
    # Se verifica si la z calculada cumple la condición de muestreo de Fresnel
    if z < z_min:
        messagebox.showwarning("Advertencia de Validez", 
                               f"La distancia z = {z:.2f} mm no cumple la condición de Fresnel (z >= {z_min:.2f} mm).\n\n"
                               "Los resultados pueden presentar aliasing y no ser precisos. "
                               "Intenta con un N más grande o ajusta los parámetros de muestreo.")
        # Se decide no detener la simulación para permitir la visualización del error, 
        # pero se podría agregar un 'return' aquí si se deseara.
    # --- FIN DE LA MODIFICACIÓN ---

    print(f"Calculando patrón de difracción para z = {z:.4f} mm (N={N})")


# ===================================================================
#                         Simulación de Propagación
# ===================================================================

    #Parametros de entrada
    n0 = np.arange(Nx) - Nx // 2
    m0 = np.arange(Ny) - Ny // 2
    x0 = n0 * dx
    y0 = m0 * dy
    X0, Y0 = np.meshgrid(x0, y0)
    
    aperture = (np.mod(X0, T) < T / 2).astype(float)
    U0 = aperture.astype(np.complex128)
    
    #Parámetros de salida
    dx_ = lam_mm * z / (Nx * dx)
    dy_ = lam_mm * z / (Ny * dy)
    x = n0 * dx_
    y = m0 * dy_
    X, Y = np.meshgrid(x, y)
    
    #Metodo de trasnfrormada de Fresnel
    Uprima = fase_esf_parax(U0, z, X0, Y0)
    Udobleprima = (dx * dy) * np.fft.fft2(Uprima)
    Udobleprimaorga = np.fft.fftshift(Udobleprima)
    Usalida = escala(Udobleprimaorga, z, X, Y)
    I = np.abs(Usalida)**2
    I_norm = I / np.max(I)

# ===================================================================
#                     Actualizar  Gráficas    
# ===================================================================

    ax1.clear()
    ax1.set_title(f"Rejilla de Entrada - Periodo={T:.1f} mm")
    ax1.set_xlabel("x (mm)")
    ax1.set_ylabel("y (mm)")
    ax1.imshow(abs(U0)**2, cmap="gray", extent=[-Lx/2, Lx/2, -Ly/2, Ly/2], origin='lower')

    ax2.clear()
    ax2.set_title(f"Patrón de difracción a z={z:.4f} mm\n({title_type})")
    ax2.set_xlabel("x (mm)")
    ax2.set_ylabel("y (mm)")
    
    # La siguiente línea calcula la extensión física TOTAL del campo de salida.
    # Es correcto usarla en imshow para que los píxeles se mapeen a las coordenadas correctas.
    extent_salida = [-Nx * dx_ / 2, Nx * dx_ / 2, -Ny * dy_ / 2, Ny * dy_ / 2]
    ax2.imshow(I_norm, cmap="gray", extent=extent_salida, origin='lower')
    
    # --- INICIO DE LA MODIFICACIÓN ---
    # Estas dos líneas FIJAN la ventana de visualización al mismo tamaño de la entrada,
    # resolviendo el problema del "zoom" o alejamiento.
    ax2.set_xlim(-Lx/2, Lx/2)
    ax2.set_ylim(-Ly/2, Ly/2)
    # --- FIN DE LA MODIFICACIÓN ---
    
    fig.canvas.draw()
    fig.canvas.flush_events()


# ===================================================================
#                        Configuración GUI
# ===================================================================

root = Tk()
root.title("Simulador de Efecto Talbot")

# Panel de controles
control_frame = Frame(root, padx=10, pady=10)
control_frame.pack(side="left", fill="both")

Label(control_frame, text="Configuración:", font=("Helvetica", 12, "bold")).pack()
Label(control_frame, text="Selecciona el tipo de imagen:").pack(anchor="w")

function_choice = StringVar(value="talbot_distance")
Radiobutton(control_frame, text="Autoimagen (z_T = 2L²/λ * N)", variable=function_choice, value="talbot_distance").pack(anchor="w")
Radiobutton(control_frame, text="Fase Revertida (z_T = (2N+1)L²/λ)", variable=function_choice, value="phase_reversed").pack(anchor="w")
Radiobutton(control_frame, text="Subimagen (z_T = (N-1/2)L²/λ)", variable=function_choice, value="subimagenes_talbot").pack(anchor="w")

Label(control_frame, text="").pack()
Label(control_frame, text="Ingresa el valor de N:").pack(anchor="w")
n_entry = Entry(control_frame)
n_entry.pack(anchor="w")
n_entry.insert(0, "1")

Button(control_frame, text="Simular", command=run_simulation).pack(pady=10)

# Panel de gráficas
fig = Figure(figsize=(10, 6), dpi=100)
fig.tight_layout(pad=3.0) # Añadido para mejor espaciado de títulos
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(side="right", fill="both", expand=True)

# Iniciar con una simulación por defecto (N=1)
run_simulation()

root.mainloop()