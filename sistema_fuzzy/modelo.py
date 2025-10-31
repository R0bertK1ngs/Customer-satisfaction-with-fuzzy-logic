# -*- coding: utf-8 -*-
"""
Fuzzy Logic Prototype as a BI Tool
Predictor de satisfacción del cliente usando lógica difusa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Instalar scikit-fuzzy si no está instalado
try:
    import skfuzzy as fuzz
except ImportError:
    print("Instalando scikit-fuzzy...")
    import subprocess
    subprocess.check_call(["pip", "install", "scikit-fuzzy"])
    import skfuzzy as fuzz

def load_data(filename="Netflix_Userbase_Frecuencia.csv"):
    """Cargar el dataset desde la carpeta actual"""
    try:
        df = pd.read_csv(filename)
        print(f"Dataset cargado exitosamente: {df.shape[0]} filas, {df.shape[1]} columnas")
        return df
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{filename}' en la carpeta actual")
        return None

def setup_membership_functions():
    """Definir universos de discurso y funciones de membresía"""
    # Definir universos de discurso
    x_tiempo = np.linspace(0, 36, 200)    # Tiempo de suscripción (meses)
    x_sub = np.linspace(8, 17, 200)       # Tipo de suscripción (8-17)
    x_freq = np.linspace(0, 50, 200)      # Frecuencia de uso (visitas/mes)
    x_sat = np.linspace(0, 100, 200)      # Nivel de satisfacción

    # Definir funciones de membresía
    membership_functions = {}
    
    # Tiempo de suscripción
    membership_functions['tiempo'] = {
        'corto': fuzz.trapmf(x_tiempo, [0, 0, 6, 12]),
        'medio': fuzz.trimf(x_tiempo, [6, 18, 30]),
        'largo': fuzz.trapmf(x_tiempo, [18, 24, 36, 36])
    }
    
    # Tipo de suscripción
    membership_functions['suscripcion'] = {
        'basico': fuzz.trapmf(x_sub, [8, 8, 10, 11]),
        'estandar': fuzz.trapmf(x_sub, [11, 12, 12, 13]),
        'premium': fuzz.trapmf(x_sub, [14, 15, 17, 17])
    }
    
    # Frecuencia de uso
    membership_functions['frecuencia'] = {
        'baja': fuzz.trapmf(x_freq, [0, 0, 5, 10]),
        'media': fuzz.trimf(x_freq, [5, 15, 25]),
        'alta': fuzz.trapmf(x_freq, [20, 30, 50, 50])
    }
    
    # Satisfacción (salida)
    membership_functions['satisfaccion'] = {
        'baja': fuzz.trapmf(x_sat, [0, 0, 20, 40]),
        'media': fuzz.trimf(x_sat, [30, 50, 70]),
        'alta': fuzz.trapmf(x_sat, [60, 75, 100, 100])
    }
    
    universes = {
        'tiempo': x_tiempo,
        'suscripcion': x_sub,
        'frecuencia': x_freq,
        'satisfaccion': x_sat
    }
    
    return membership_functions, universes

def plot_membership_functions(membership_functions, universes):
    """Graficar todas las funciones de membresía"""
    fig, axs = plt.subplots(4, 1, figsize=(10, 16))
    
    # Tiempo de suscripción
    for label, mf in membership_functions['tiempo'].items():
        axs[0].plot(universes['tiempo'], mf, label=label.capitalize())
    axs[0].set_title('Funciones de Membresía - Tiempo de Suscripción')
    axs[0].set_xlabel('Meses')
    axs[0].set_ylabel('Grado de Pertenencia')
    axs[0].legend()
    axs[0].grid(True)
    
    # Tipo de suscripción
    for label, mf in membership_functions['suscripcion'].items():
        axs[1].plot(universes['suscripcion'], mf, label=label.capitalize())
    axs[1].set_title('Funciones de Membresía - Tipo de Suscripción')
    axs[1].set_xlabel('Valor del Plan ($)')
    axs[1].set_ylabel('Grado de Pertenencia')
    axs[1].legend()
    axs[1].grid(True)
    
    # Frecuencia de uso
    for label, mf in membership_functions['frecuencia'].items():
        axs[2].plot(universes['frecuencia'], mf, label=f"{label.capitalize()} Frecuencia")
    axs[2].set_title('Funciones de Membresía - Frecuencia de Uso')
    axs[2].set_xlabel('Visitas por Mes')
    axs[2].set_ylabel('Grado de Pertenencia')
    axs[2].legend()
    axs[2].grid(True)
    
    # Satisfacción
    for label, mf in membership_functions['satisfaccion'].items():
        axs[3].plot(universes['satisfaccion'], mf, label=label.capitalize())
    axs[3].set_title('Funciones de Membresía - Nivel de Satisfacción')
    axs[3].set_xlabel('Nivel (%)')
    axs[3].set_ylabel('Grado de Pertenencia')
    axs[3].legend()
    axs[3].grid(True)
    
    plt.tight_layout()
    plt.show()

# Funciones de membresía manuales para cálculos individuales
def trapmf(x, a, b, c, d):
    """Función de membresía trapezoidal"""
    if x <= a or x >= d:
        return 0.0
    elif a < x < b:
        return (x - a) / (b - a)
    elif b <= x <= c:
        return 1.0
    else:  # c < x < d
        return (d - x) / (d - c)

def trimf(x, a, b, c):
    """Función de membresía triangular"""
    if x <= a or x >= c:
        return 0.0
    elif a < x < b:
        return (x - a) / (b - a)
    else:  # b <= x < c
        return (c - x) / (c - b)

def calculate_membership_degrees(tm, fq, subv):
    """Calcular grados de pertenencia para un registro"""
    degrees = {
        'short': trapmf(tm, 0, 0, 6, 12),
        'medium': trimf(tm, 6, 18, 30),
        'long': trapmf(tm, 18, 24, 36, 36),
        'lowf': trapmf(fq, 0, 0, 5, 10),
        'medf': trimf(fq, 5, 15, 25),
        'highf': trapmf(fq, 20, 30, 50, 50),
        'basic': trimf(subv, 9, 10, 11),
        'standard': trimf(subv, 10, 12, 14),
        'premium': trimf(subv, 13, 15, 17)
    }
    return degrees

def define_fuzzy_rules():
    """Definir las reglas difusas del sistema"""
    rules = [
        # Frecuencia baja
        (['short', 'basic', 'lowf'], ('low', [0, 0, 20, 40])),
        (['short', 'standard', 'lowf'], ('low', [0, 0, 20, 40])),
        (['short', 'premium', 'lowf'], ('low', [0, 0, 20, 40])),
        (['medium', 'basic', 'lowf'], ('low', [0, 0, 20, 40])),
        (['medium', 'standard', 'lowf'], ('low', [0, 0, 20, 40])),
        (['medium', 'premium', 'lowf'], ('low', [0, 0, 20, 40])),
        (['long', 'basic', 'lowf'], ('medium', [30, 50, 70])),
        (['long', 'standard', 'lowf'], ('medium', [30, 50, 70])),
        (['long', 'premium', 'lowf'], ('medium', [30, 50, 70])),

        # Frecuencia media
        (['short', 'basic', 'medf'], ('low', [0, 0, 20, 40])),
        (['short', 'standard', 'medf'], ('low', [0, 0, 20, 40])),
        (['short', 'premium', 'medf'], ('low', [0, 0, 20, 40])),
        (['medium', 'basic', 'medf'], ('medium', [30, 50, 70])),
        (['medium', 'standard', 'medf'], ('medium', [30, 50, 70])),
        (['medium', 'premium', 'medf'], ('medium', [30, 50, 70])),
        (['long', 'basic', 'medf'], ('medium', [30, 50, 70])),
        (['long', 'standard', 'medf'], ('medium', [30, 50, 70])),
        (['long', 'premium', 'medf'], ('medium', [30, 50, 70])),

        # Frecuencia alta
        (['short', 'basic', 'highf'], ('medium', [30, 50, 70])),
        (['short', 'standard', 'highf'], ('medium', [30, 50, 70])),
        (['short', 'premium', 'highf'], ('medium', [30, 50, 70])),
        (['medium', 'basic', 'highf'], ('high', [60, 75, 100, 100])),
        (['medium', 'standard', 'highf'], ('high', [60, 75, 100, 100])),
        (['medium', 'premium', 'highf'], ('high', [60, 75, 100, 100])),
        (['long', 'basic', 'highf'], ('high', [60, 75, 100, 100])),
        (['long', 'standard', 'highf'], ('high', [60, 75, 100, 100])),
        (['long', 'premium', 'highf'], ('high', [60, 75, 100, 100])),
    ]
    return rules

def apply_fuzzy_inference(tm, fq, subv, rules):
    """Aplicar inferencia difusa para un registro"""
    degrees = calculate_membership_degrees(tm, fq, subv)
    x_sat = np.arange(0, 101, 1)
    
    # Agregación
    agg = np.zeros_like(x_sat, dtype=float)
    max_strength = 0.0
    
    for ants, (_, params) in rules:
        strength = min(degrees[ants[0]], degrees[ants[1]], degrees[ants[2]])
        if strength > 0:
            mf_vals = [trapmf(x, *params) if len(params) == 4 else trimf(x, *params) for x in x_sat]
            agg = np.maximum(agg, np.minimum(strength, mf_vals))
            max_strength = max(max_strength, strength)
    
    # Desfuzificación por centroide
    if agg.sum() == 0:
        result = np.nan
    else:
        result = np.sum(x_sat * agg) / np.sum(agg)
    
    return result, max_strength, agg

def analyze_single_record(df, record_index=14):
    """Analizar un registro específico y mostrar gráficos detallados"""
    if record_index >= len(df):
        print(f"Error: El índice {record_index} está fuera del rango del dataset")
        return
    
    row = df.iloc[record_index]
    print(f"\n=== ANÁLISIS DEL REGISTRO #{record_index + 1} ===")
    print("Valores obtenidos del CSV:")
    for col in row.index:
        print(f"  {col}: {row[col]}")
    
    tm = row['Months_diff']
    fq = row['Frequency']
    subv = row['Monthly Revenue']
    
    rules = define_fuzzy_rules()
    result, max_strength, agg = apply_fuzzy_inference(tm, fq, subv, rules)
    
    print(f"\nNivel de satisfacción predicho: {round(result, 2)}%")
    print(f"Fuerza difusa máxima: {round(max_strength, 3)}")
    
    # Gráficos detallados
    plot_detailed_analysis(tm, fq, subv, agg, result)

def plot_detailed_analysis(tm, fq, subv, agg, result):
    """Crear gráficos detallados del análisis"""
    # Gráfico de tiempo de suscripción
    x_tm = np.linspace(0, 40, 400)
    deg_tm = {
        'short': trapmf(tm, 0, 0, 6, 12),
        'medium': trimf(tm, 6, 18, 30),
        'long': trapmf(tm, 18, 24, 36, 36),
    }
    
    plt.figure(figsize=(15, 12))
    
    plt.subplot(2, 2, 1)
    plt.plot(x_tm, [trapmf(x, 0, 0, 6, 12) for x in x_tm], label='Corto')
    plt.plot(x_tm, [trimf(x, 6, 18, 30) for x in x_tm], label='Medio')
    plt.plot(x_tm, [trapmf(x, 18, 24, 36, 36) for x in x_tm], label='Largo')
    plt.axvline(tm, color='red', linestyle='--', label=f'Valor: {tm:.2f}')
    plt.title("Tiempo de Suscripción (Months_diff)")
    plt.xlabel("Meses")
    plt.ylabel("Grado de pertenencia")
    plt.legend()
    plt.grid(True)
    
    # Gráfico de frecuencia
    x_fq = np.linspace(0, 50, 400)
    plt.subplot(2, 2, 2)
    plt.plot(x_fq, [trapmf(x, 0, 0, 5, 10) for x in x_fq], label='Baja')
    plt.plot(x_fq, [trimf(x, 5, 15, 25) for x in x_fq], label='Media')
    plt.plot(x_fq, [trapmf(x, 20, 30, 50, 50) for x in x_fq], label='Alta')
    plt.axvline(fq, color='red', linestyle='--', label=f'Valor: {fq:.2f}')
    plt.title("Frecuencia de Uso")
    plt.xlabel("Veces por mes")
    plt.ylabel("Grado de pertenencia")
    plt.legend()
    plt.grid(True)
    
    # Gráfico de suscripción
    x_mv = np.linspace(8, 18, 400)
    plt.subplot(2, 2, 3)
    plt.plot(x_mv, [trimf(x, 9, 10, 11) for x in x_mv], label='Básico')
    plt.plot(x_mv, [trimf(x, 10, 12, 14) for x in x_mv], label='Estándar')
    plt.plot(x_mv, [trimf(x, 13, 15, 17) for x in x_mv], label='Premium')
    plt.axvline(subv, color='red', linestyle='--', label=f'Valor: {subv:.2f}')
    plt.title("Tipo de Suscripción (Monthly Revenue)")
    plt.xlabel("Precio ($)")
    plt.ylabel("Grado de pertenencia")
    plt.legend()
    plt.grid(True)
    
    # Gráfico de salida agregada
    x_sat = np.arange(0, 101, 1)
    sat_low = fuzz.trapmf(x_sat, [0, 0, 25, 40])
    sat_med = fuzz.trimf(x_sat, [30, 50, 70])
    sat_high = fuzz.trapmf(x_sat, [60, 75, 100, 100])
    
    plt.subplot(2, 2, 4)
    plt.plot(x_sat, sat_low, label='Baja', linestyle='--', alpha=0.7)
    plt.plot(x_sat, sat_med, label='Media', linestyle='--', alpha=0.7)
    plt.plot(x_sat, sat_high, label='Alta', linestyle='--', alpha=0.7)
    plt.fill_between(x_sat, np.zeros_like(x_sat), agg, alpha=0.6, color='blue', label='Salida agregada')
    if not np.isnan(result):
        plt.axvline(result, color='red', linestyle='-', linewidth=2, label=f'Centroide: {round(result, 2)}%')
    plt.title("Salida Difusa: Nivel de Satisfacción")
    plt.xlabel("Nivel de Satisfacción (%)")
    plt.ylabel("Grado de pertenencia")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def predict_all_dataset(df):
    """Realizar predicción para todo el dataset"""
    print("\n=== PREDICCIÓN PARA TODO EL DATASET ===")
    
    rules = define_fuzzy_rules()
    satisfactions = []
    strengths = []
    
    print("Procesando registros...")
    for i, (_, row) in enumerate(df.iterrows()):
        tm = row['Months_diff']
        fq = row['Frequency']
        subv = row['Monthly Revenue']
        
        result, max_strength, _ = apply_fuzzy_inference(tm, fq, subv, rules)
        
        satisfactions.append(round(result, 2) if not np.isnan(result) else 0)
        strengths.append(round(max_strength, 3))
        
        if (i + 1) % 100 == 0:
            print(f"Procesados {i + 1} registros...")
    
    # Crear nuevo DataFrame con resultados
    df_result = df.copy()
    df_result["Predicted_Satisfaction"] = satisfactions
    df_result["Fuzzy_Strength"] = strengths
    
    # Guardar resultados
    output_filename = "salida_con_satisfaccion.csv"
    df_result.to_csv(output_filename, index=False)
    
    print(f"\nProceso completado exitosamente!")
    print(f"Archivo guardado como: '{output_filename}'")
    print(f"Total de registros procesados: {len(df_result)}")
    
    # Mostrar estadísticas básicas
    print(f"\n=== ESTADÍSTICAS DE SATISFACCIÓN PREDICHA ===")
    satisfaction_stats = df_result["Predicted_Satisfaction"].describe()
    print(satisfaction_stats)
    
    return df_result

def main():
    """Función principal"""
    print("=== SISTEMA DE PREDICCIÓN DE SATISFACCIÓN CON LÓGICA DIFUSA ===")
    
    # Cargar datos
    df = load_data()
    if df is None:
        return
    
    print(f"\nPrimeras 5 filas del dataset:")
    print(df.head())
    
    # Configurar funciones de membresía
    membership_functions, universes = setup_membership_functions()
    
    # Mostrar gráficos de funciones de membresía
    print("\nMostrando funciones de membresía...")
    plot_membership_functions(membership_functions, universes)
    
    # Analizar un registro específico (registro 15, índice 14)
    analyze_single_record(df, record_index=14)
    
    # Predecir para todo el dataset
    df_result = predict_all_dataset(df)
    
    print(f"\n=== PROCESO COMPLETADO ===")
    print(f"Últimas 5 filas con predicciones:")
    print(df_result[['Months_diff', 'Frequency', 'Monthly Revenue', 
                     'Predicted_Satisfaction', 'Fuzzy_Strength']].tail())

if __name__ == "__main__":
    main()