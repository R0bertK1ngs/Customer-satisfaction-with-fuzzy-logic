from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from django.http import HttpResponseRedirect, JsonResponse
from django.urls import reverse
from django.conf import settings as django_settings
from .models import MembershipFunction
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import base64
from io import BytesIO
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Configuración de matplotlib
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def trapmf(x, a, b, c, d):
    """Función de membresía trapezoidal"""
    if x <= a or x >= d:
        return 0.0
    elif a < x < b:
        return (x - a) / (b - a)
    elif b <= x <= c:
        return 1.0
    else:
        return (d - x) / (d - c)

def trimf(x, a, b, c):
    """Función de membresía triangular"""
    if x <= a or x >= c:
        return 0.0
    elif a < x < b:
        return (x - a) / (b - a)
    else:
        return (c - x) / (c - b)

def calculate_membership_degrees(tm, fq, subv):
    """Calcula los grados de membresía para todas las variables"""
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

def generate_membership_functions_data():
    """Genera los datos de las funciones de membresía"""
    membership_functions = {
        'tiempo': {
            'names': ['Nuevo', 'Regular', 'Veterano'],
            'functions': ['short', 'medium', 'long'],
            'range': (0, 36),
            'unit': 'meses'
        },
        'frecuencia': {
            'names': ['Baja', 'Media', 'Alta'],
            'functions': ['lowf', 'medf', 'highf'],
            'range': (0, 50),
            'unit': 'visitas/mes'
        },
        'suscripcion': {
            'names': ['Básica', 'Estándar', 'Premium'],
            'functions': ['basic', 'standard', 'premium'],
            'range': (8, 17),
            'unit': 'USD'
        },
        'satisfaccion': {
            'names': ['Insatisfecho', 'Neutral', 'Satisfecho'],
            'functions': ['low', 'medium', 'high'],
            'range': (0, 100),
            'unit': '%'
        }
    }
    return membership_functions

def plot_membership_functions():
    """Genera el gráfico de las funciones de membresía"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Funciones de Membresía del Sistema de Lógica Difusa', fontsize=16, fontweight='bold')
        
        variables_params = {
            'Tiempo de Suscripción (meses)': {
                'range': np.arange(0, 37, 1),
                'functions': {
                    'Nuevo': {'type': 'trap', 'params': [0, 0, 6, 12], 'color': '#e74c3c'},
                    'Regular': {'type': 'tri', 'params': [6, 18, 30], 'color': '#f39c12'},
                    'Veterano': {'type': 'trap', 'params': [18, 24, 36, 36], 'color': '#27ae60'}
                },
                'pos': (0, 0)
            },
            'Frecuencia de Uso (visitas/mes)': {
                'range': np.arange(0, 51, 1),
                'functions': {
                    'Baja': {'type': 'trap', 'params': [0, 0, 5, 10], 'color': '#e74c3c'},
                    'Media': {'type': 'tri', 'params': [5, 15, 25], 'color': '#f39c12'},
                    'Alta': {'type': 'trap', 'params': [20, 30, 50, 50], 'color': '#27ae60'}
                },
                'pos': (0, 1)
            },
            'Tipo de Suscripción (USD)': {
                'range': np.arange(8, 18, 0.1),
                'functions': {
                    'Básica': {'type': 'tri', 'params': [9, 10, 11], 'color': '#e74c3c'},
                    'Estándar': {'type': 'tri', 'params': [10, 12, 14], 'color': '#f39c12'},
                    'Premium': {'type': 'tri', 'params': [13, 15, 17], 'color': '#27ae60'}
                },
                'pos': (1, 0)
            },
            'Nivel de Satisfacción (%)': {
                'range': np.arange(0, 101, 1),
                'functions': {
                    'Insatisfecho': {'type': 'trap', 'params': [0, 0, 20, 40], 'color': '#e74c3c'},
                    'Neutral': {'type': 'tri', 'params': [30, 50, 70], 'color': '#f39c12'},
                    'Satisfecho': {'type': 'trap', 'params': [60, 75, 100, 100], 'color': '#27ae60'}
                },
                'pos': (1, 1)
            }
        }
        
        for var_name, var_data in variables_params.items():
            ax = axes[var_data['pos']]
            x_range = var_data['range']
            
            for func_name, func_data in var_data['functions'].items():
                if func_data['type'] == 'trap':
                    y_vals = [trapmf(x, *func_data['params']) for x in x_range]
                else:
                    y_vals = [trimf(x, *func_data['params']) for x in x_range]
                
                ax.plot(x_range, y_vals, color=func_data['color'], 
                       linewidth=2.5, label=func_name, alpha=0.8)
                ax.fill_between(x_range, y_vals, alpha=0.3, color=func_data['color'])
            
            ax.set_title(var_name, fontweight='bold', fontsize=12)
            ax.set_xlabel('Valor')
            ax.set_ylabel('Grado de Membresía')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.05)
        
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        buffer.seek(0)
        plot_data = buffer.getvalue()
        buffer.close()
        plt.close()
        
        return base64.b64encode(plot_data).decode()
        
    except Exception as e:
        print(f"Error generando gráfico de funciones de membresía: {e}")
        return None

def plot_detailed_analysis(tm, fq, subv, satisfaction, fuzzy_strength):
    """Genera un gráfico detallado del análisis de un registro específico"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Análisis Detallado - Satisfacción Predicha: {satisfaction}%', 
                    fontsize=16, fontweight='bold')
        
        degrees = calculate_membership_degrees(tm, fq, subv)
        
        ax1 = axes[0, 0]
        x_time = np.arange(0, 37, 1)
        y_short = [trapmf(x, 0, 0, 6, 12) for x in x_time]
        y_medium = [trimf(x, 6, 18, 30) for x in x_time]
        y_long = [trapmf(x, 18, 24, 36, 36) for x in x_time]
        
        ax1.plot(x_time, y_short, 'r-', label='Nuevo', linewidth=2)
        ax1.plot(x_time, y_medium, 'orange', label='Regular', linewidth=2)
        ax1.plot(x_time, y_long, 'g-', label='Veterano', linewidth=2)
        ax1.axvline(tm, color='black', linestyle='--', linewidth=2, label=f'Valor actual: {tm}')
        ax1.scatter([tm], [degrees['short']], color='red', s=100, zorder=5)
        ax1.scatter([tm], [degrees['medium']], color='orange', s=100, zorder=5)
        ax1.scatter([tm], [degrees['long']], color='green', s=100, zorder=5)
        ax1.set_title('Tiempo de Suscripción (meses)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        x_freq = np.arange(0, 51, 1)
        y_lowf = [trapmf(x, 0, 0, 5, 10) for x in x_freq]
        y_medf = [trimf(x, 5, 15, 25) for x in x_freq]
        y_highf = [trapmf(x, 20, 30, 50, 50) for x in x_freq]
        
        ax2.plot(x_freq, y_lowf, 'r-', label='Baja', linewidth=2)
        ax2.plot(x_freq, y_medf, 'orange', label='Media', linewidth=2)
        ax2.plot(x_freq, y_highf, 'g-', label='Alta', linewidth=2)
        ax2.axvline(fq, color='black', linestyle='--', linewidth=2, label=f'Valor actual: {fq}')
        ax2.scatter([fq], [degrees['lowf']], color='red', s=100, zorder=5)
        ax2.scatter([fq], [degrees['medf']], color='orange', s=100, zorder=5)
        ax2.scatter([fq], [degrees['highf']], color='green', s=100, zorder=5)
        ax2.set_title('Frecuencia de Uso (visitas/mes)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[1, 0]
        x_sub = np.arange(8, 18, 0.1)
        y_basic = [trimf(x, 9, 10, 11) for x in x_sub]
        y_standard = [trimf(x, 10, 12, 14) for x in x_sub]
        y_premium = [trimf(x, 13, 15, 17) for x in x_sub]
        
        ax3.plot(x_sub, y_basic, 'r-', label='Básica', linewidth=2)
        ax3.plot(x_sub, y_standard, 'orange', label='Estándar', linewidth=2)
        ax3.plot(x_sub, y_premium, 'g-', label='Premium', linewidth=2)
        ax3.axvline(subv, color='black', linestyle='--', linewidth=2, label=f'Valor actual: ${subv}')
        ax3.scatter([subv], [degrees['basic']], color='red', s=100, zorder=5)
        ax3.scatter([subv], [degrees['standard']], color='orange', s=100, zorder=5)
        ax3.scatter([subv], [degrees['premium']], color='green', s=100, zorder=5)
        ax3.set_title('Tipo de Suscripción (USD)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4 = axes[1, 1]
        active_degrees = {k: v for k, v in degrees.items() if v > 0.01}
        if active_degrees:
            labels = list(active_degrees.keys())
            values = list(active_degrees.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
            
            bars = ax4.bar(labels, values, color=colors, alpha=0.7, edgecolor='black')
            ax4.set_title('Grados de Membresía Activos')
            ax4.set_ylabel('Grado de Membresía')
            ax4.set_ylim(0, 1)
            
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        buffer.seek(0)
        plot_data = buffer.getvalue()
        buffer.close()
        plt.close()
        
        return base64.b64encode(plot_data).decode()
        
    except Exception as e:
        print(f"Error generando gráfico de análisis detallado: {e}")
        return None

def define_fuzzy_rules():
    """Define las reglas difusas del sistema"""
    rules = [
        (['short', 'basic', 'lowf'], ('low', [0, 0, 20, 40])),
        (['short', 'standard', 'lowf'], ('low', [0, 0, 20, 40])),
        (['short', 'premium', 'lowf'], ('low', [0, 0, 20, 40])),
        (['medium', 'basic', 'lowf'], ('low', [0, 0, 20, 40])),
        (['medium', 'standard', 'lowf'], ('low', [0, 0, 20, 40])),
        (['medium', 'premium', 'lowf'], ('low', [0, 0, 20, 40])),
        (['long', 'basic', 'lowf'], ('medium', [30, 50, 70])),
        (['long', 'standard', 'lowf'], ('medium', [30, 50, 70])),
        (['long', 'premium', 'lowf'], ('medium', [30, 50, 70])),
        (['short', 'basic', 'medf'], ('low', [0, 0, 20, 40])),
        (['short', 'standard', 'medf'], ('low', [0, 0, 20, 40])),
        (['short', 'premium', 'medf'], ('low', [0, 0, 20, 40])),
        (['medium', 'basic', 'medf'], ('medium', [30, 50, 70])),
        (['medium', 'standard', 'medf'], ('medium', [30, 50, 70])),
        (['medium', 'premium', 'medf'], ('medium', [30, 50, 70])),
        (['long', 'basic', 'medf'], ('medium', [30, 50, 70])),
        (['long', 'standard', 'medf'], ('medium', [30, 50, 70])),
        (['long', 'premium', 'medf'], ('medium', [30, 50, 70])),
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
    """Aplica la inferencia difusa"""
    degrees = calculate_membership_degrees(tm, fq, subv)
    x_sat = np.arange(0, 101, 1)
    agg = np.zeros_like(x_sat, dtype=float)
    max_strength = 0.0
    
    for ants, (_, params) in rules:
        strength = min(degrees[ants[0]], degrees[ants[1]], degrees[ants[2]])
        if strength > 0:
            mf_vals = [trapmf(x, *params) if len(params) == 4 else trimf(x, *params) for x in x_sat]
            agg = np.maximum(agg, np.minimum(strength, mf_vals))
            max_strength = max(max_strength, strength)
    
    if agg.sum() == 0:
        result = np.nan
    else:
        result = np.sum(x_sat * agg) / np.sum(agg)
    
    return result, max_strength, agg

def fuzzy_model_complete(request):
    """Vista principal del modelo difuso completo"""
    try:
        # Cargar el CSV
        csv_path = os.path.join(django_settings.BASE_DIR, 'sistema_fuzzy/Netflix_Userbase_Frecuencia.csv')
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError("El archivo CSV no se encuentra. Por favor, asegúrate de que existe en la ubicación correcta.")
            
        # Cargar datos del CSV
        df = pd.read_csv(csv_path)
        
        # Verificar y renombrar columnas si es necesario
        required_columns = ['Months_diff', 'Frequency', 'Monthly_Revenue']
        column_mapping = {
            'Monthly Revenue': 'Monthly_Revenue',
            # Agrega más mapeos si son necesarios
        }
        
        # Aplicar mapeo de columnas
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df[new_name] = df[old_name]
        
        # Verificar que todas las columnas requeridas existen
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Faltan las siguientes columnas en el CSV: {', '.join(missing_columns)}")

        rules = define_fuzzy_rules()
        satisfactions, strengths = [], []
        
        # Aplicar el modelo difuso a todos los registros
        for _, row in df.iterrows():
            r, s, _ = apply_fuzzy_inference(
                row['Months_diff'], 
                row['Frequency'], 
                row['Monthly_Revenue'], 
                rules
            )
            satisfactions.append(round(r, 2) if not np.isnan(r) else 0)
            strengths.append(round(s, 3))

        df['Predicted_Satisfaction'] = satisfactions
        df['Fuzzy_Strength'] = strengths
        
        stats = df['Predicted_Satisfaction'].describe()
        membership_plot = plot_membership_functions()
        membership_functions_data = generate_membership_functions_data()
        
        max_satisfaction_idx = df['Predicted_Satisfaction'].idxmax()
        analyzed_record_data = df.loc[max_satisfaction_idx]
        
        detailed_plot = plot_detailed_analysis(
            analyzed_record_data['Months_diff'],
            analyzed_record_data['Frequency'],
            analyzed_record_data['Monthly_Revenue'],
            analyzed_record_data['Predicted_Satisfaction'],
            analyzed_record_data['Fuzzy_Strength']
        )
        
        context = {
            'dataset_info': {
                'total_records': len(df),
                'columns': list(df.columns),
                'shape': df.shape
            },
            'membership_plot': membership_plot,
            'membership_functions': membership_functions_data,
            'analyzed_record': {
                'index': max_satisfaction_idx + 1,
                'months_diff': int(analyzed_record_data['Months_diff']),
                'frequency': int(analyzed_record_data['Frequency']),
                'monthly_revenue': round(analyzed_record_data['Monthly_Revenue'], 2),
                'predicted_satisfaction': round(analyzed_record_data['Predicted_Satisfaction'], 1),
                'fuzzy_strength': round(analyzed_record_data['Fuzzy_Strength'], 3)
            },
            'detailed_plot': detailed_plot,
            'statistics': {
                'mean': round(stats['mean'], 2),
                'std': round(stats['std'], 2),
                'min': round(stats['min'], 2),
                'max': round(stats['max'], 2),
                'q25': round(stats['25%'], 2),
                'q50': round(stats['50%'], 2),
                'q75': round(stats['75%'], 2)
            },
            'sample_data': df.head(10).to_dict('records'),
            'total_rules': len(rules),
            'error': None
        }

        return render(request, 'fuzzy_model_complete.html', context)
        
    except Exception as e:
        print(f"Error en fuzzy_model_complete: {e}")
        context = {
            'error': str(e),
            'dataset_info': None,
            'membership_plot': None,
            'membership_functions': None,
            'analyzed_record': None,
            'detailed_plot': None,
            'statistics': None,
            'sample_data': None,
            'total_rules': None
        }
        return render(request, 'fuzzy_model_complete.html', context)

def Home_fuzzy(request):
    """Vista principal del sistema"""
    context = {
        'system_stats': {
            'total_simulations': 0,
            'active_rules': 27,
            'accuracy': 95.2,
            'variables': 3,
        },
        'recent_activities': [],
        'notifications': [],
    }
    return render(request, 'home.html', context)

def membership_functions(request):
    """Vista de funciones de membresía"""
    functions = MembershipFunction.objects.select_related('variable').all()
    return render(request, 'membership_functions.html', {
        'membership_functions': functions
    })

def add_membership_function(request):
    """Vista para agregar función de membresía"""
    if request.method == 'POST':
        messages.success(request, 'Función de membresía agregada exitosamente.')
        return redirect('membership_functions')
    context = {
        'title': 'Agregar Función de Membresía',
        'action': 'add'
    }
    return render(request, 'membership_function_form.html', context)

def edit_membership_function(request, id):
    """Vista para editar función de membresía"""
    membership_function = get_object_or_404(MembershipFunction, id=id)
    if request.method == 'POST':
        messages.success(request, f'Función de membresía "{membership_function.name}" editada exitosamente.')
        return redirect('membership_functions')
    context = {
        'title': 'Editar Función de Membresía',
        'action': 'edit',
        'membership_function': membership_function
    }
    return render(request, 'membership_function_form.html', context)

def delete_membership_function(request, id):
    """Vista para eliminar función de membresía"""
    membership_function = get_object_or_404(MembershipFunction, id=id)
    if request.method == 'POST':
        name = membership_function.name
        membership_function.delete()
        messages.success(request, f'Función de membresía "{name}" eliminada exitosamente.')
        return redirect('membership_functions')
    context = {
        'membership_function': membership_function,
        'title': 'Eliminar Función de Membresía'
    }
    return render(request, 'membership_function_confirm_delete.html', context)

def simulation(request):
    """Vista de simulación"""
    context = {'title': 'Simulación del Sistema Fuzzy'}
    return render(request, 'simulation.html', context)

def fuzzy_rules(request):
    """Vista de reglas difusas"""
    context = {'title': 'Reglas Difusas'}
    return render(request, 'fuzzy_rules.html', context)

def analytics(request):
    """Vista de análisis"""
    context = {'title': 'Análisis del Sistema'}
    return render(request, 'analytics.html', context)

def system_settings(request):
    """Vista de configuración del sistema"""
    context = {'title': 'Configuración del Sistema'}
    return render(request, 'settings.html', context)

def data_management(request):
    """Vista de gestión de datos"""
    context = {'title': 'Gestión de Datos'}
    return render(request, 'data_management.html', context)

def export_data(request):
    """Vista de exportación de datos"""
    context = {'title': 'Exportar Datos'}
    return render(request, 'export_data.html', context)

def about_mamdani(request):
    """Vista acerca del modelo Mamdani"""
    context = {'title': 'Acerca del Modelo Mamdani'}
    return render(request, 'about_mamdani.html', context)

def performance(request):
    """Vista de rendimiento"""
    context = {'title': 'Rendimiento del Sistema'}
    return render(request, 'performance.html', context)

def tutorial(request):
    """Vista de tutorial"""
    context = {'title': 'Tutorial del Sistema'}
    return render(request, 'tutorial.html', context)

def help(request):
    """Vista de ayuda"""
    context = {'title': 'Ayuda'}
    return render(request, 'help.html', context)

def activity_log(request):
    """Vista de registro de actividad"""
    context = {'title': 'Registro de Actividad'}
    return render(request, 'activity_log.html', context)

def api_stats(request):
    """API de estadísticas"""
    stats = {
        'simulations': 15,
        'rules': 27,
        'accuracy': 95.2
    }
    return JsonResponse(stats)