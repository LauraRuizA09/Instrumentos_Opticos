import math

def calcular_resolucion_usaf():
    """
    Calcula y compara la resolución teórica (Abbe) con la resolución
    medida (USAF escalada) de un sistema de microscopio simulado,
    basado en los parámetros del documento "Parcial___Instrumentos_opticos.pdf".
    """
    
    print("¡Hola Laura! Iniciando el cálculo de resolución...")
    print("---------------------------------------------------------")
    print("PASO 1: CÁLCULO DE LA RESOLUCIÓN TEÓRICA (LÍMITE DE ABBE)")
    print("---------------------------------------------------------")

    # --- Parámetros Fijos del Sistema (del PDF) ---
    # Usamos unidades del SI (metros) para todos los cálculos.
    
    # Longitud de onda [cite: 2942]
    lambda_val_m = 533e-9
    
    # Apertura Numérica [cite: 2943]
    na_val = 0.25
    
    # Magnificación del objetivo [cite: 2943, 2945]
    M = 20
    
    # Parámetros del sensor (Alvium 1800 U-811m) [cite: 2954]
    delta_sensor_m = 2.74e-6
    N = 2848.0 # Número de píxeles (resolución)
    
    # Tamaño estándar del test USAF T20-CPG (asumido para replicar los cálculos del PDF)
    # link: https://www.appliedimage.com/product-category/test-targets-and-charts/usaf-targets/usaf-1951-standard-resolution-target-t-20/
    # 10.2 cm = 0.102 m
    L_std_m = 0.102 
    
    # 1.1: Calcular Límite de Abbe
    #  R_Abbe = lambda / (2 * NA)
    r_abbe_m = lambda_val_m / (2 * na_val)
    
    print(f"Parámetros del sistema:")
    print(f"  Longitud de onda (lambda): {lambda_val_m * 1e9:.1f} nm")
    print(f"  Apertura Numérica (NA): {na_val}")
    print(f"Resultado (Teórico):")
    print(f"  Resolución de Abbe (R_Abbe) = {r_abbe_m * 1e6:.3f} µm\n")

    print("---------------------------------------------------------")
    print("PASO 2: CÁLCULO DE LA RESOLUCIÓN MEDIDA (USAF ESCALADO)")
    print("---------------------------------------------------------")

    # 2.1: Obtener entrada del usuario
    try:
        grupo = int(input("Ingresa el número de GRUPO observable: "))
        elemento = int(input("Ingresa el número de ELEMENTO observable: "))
    except ValueError:
        print("Error: Debes ingresar números enteros.")
        return

    # 2.2: Calcular la Resolución Estándar (R_Std) para ese (G, E)
    # Fórmula estándar USAF 1951 (en pares de líneas por mm)
    # R_lp_mm = 2^(Grupo + (Elemento - 1) / 6)
    r_lp_mm = math.pow(2, (grupo + (elemento - 1) / 6.0))
    
    # Convertir a pares de líneas por metro (lp/m)
    r_lp_m = r_lp_mm * 1000.0
    
    # Convertir a tamaño de característica (resolución) en metros
    # R_Std = 1 / (lp/m)
    r_std_m = 1.0 / r_lp_m

    print(f"Cálculo de R_Std (Estándar) para (Grupo {grupo}, Elemento {elemento}):")
    print(f"  Frecuencia espacial = {r_lp_mm:.3f} lp/mm")
    print(f"  Resolución Estándar (R_Std) = {r_std_m * 1e6:.3f} µm")

    # 2.3: Aplicar la lógica de escalado del PDF [cite: 3093-3108, 3113-3121]
    
    # a) Calcular el tamaño de píxel en el plano objeto 
    # Delta_Objeto = Delta_Sensor / M
    delta_objeto_m = delta_sensor_m / M
    print(f"\nCálculo de Escalado (Simulación):")
    print(f"  Tamaño de píxel del sensor (Delta_Sensor) = {delta_sensor_m * 1e6:.3f} µm")
    print(f"  Tamaño de píxel del objeto (Delta_Objeto) = {delta_objeto_m * 1e6:.3f} µm")

    # b) Calcular el número total de píxeles (Y) que tendría el chart estándar 
    # Y = L_Std / Delta_Objeto
    Y_pixels_std = L_std_m / delta_objeto_m
    print(f"  Píxeles en el chart estándar (Y) = {Y_pixels_std:.0f}")

    # c) Usar la regla de tres para encontrar la resolución real (Z) 
    # Y / R_Std = N / Z  =>  Z = (N * R_Std) / Y
    Z_m = (N * r_std_m) / Y_pixels_std
    
    print(f"  Píxeles en la simulación (N) = {N:.0f}")
    print(f"Resultado (Medido):")
    print(f"  Resolución Medida (Z) = {Z_m * 1e6:.3f} µm\n")
    
    print("---------------------------------------------------------")
    print("PASO 3: COMPARACIÓN DE RESULTADOS")
    print("---------------------------------------------------------")
    
    error_abs_m = (abs(Z_m - r_abbe_m)/r_abbe_m) * 100
    
    print(f"  Resolución Teórica (Abbe): {r_abbe_m * 1e6:.3f} µm")
    print(f"  Resolución Medida (USAF):  {Z_m * 1e6:.3f} µm")
    print(f"  Error % = {error_abs_m:.3f} %")
    print("---------------------------------------------------------")

# --- Ejecutar la función ---
calcular_resolucion_usaf()