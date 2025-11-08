from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from django.http import HttpResponse, JsonResponse, HttpResponseNotAllowed, HttpResponseRedirect
from django.urls import reverse
from django.conf import settings as django_settings
from .models import MembershipFunction
from .modelo import evaluate_df, evaluate_file, dataframe_to_csv_bytes, trapmf, trimf, calculate_membership_degrees, define_fuzzy_rules, apply_fuzzy_inference
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

# imports (si no están ya)
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import os

# 👇 ya los estás importando arriba; si no, agrégalos:
from .modelo import (
    evaluate_df, evaluate_file, dataframe_to_csv_bytes,
    calculate_membership_degrees, define_fuzzy_rules,
    trapmf, trimf
)

def generate_membership_functions_data():
    """Texto/labels para la lista de funciones en la tarjeta de MF."""
    return {
        'tiempo':  ['Nuevo', 'Regular', 'Veterano'],
        'frecuencia': ['Baja', 'Media', 'Alta'],
        'suscripcion': ['Básica', 'Estándar', 'Premium'],
        'satisfaccion': ['Insatisfecho', 'Neutral', 'Satisfecho']
    }

def plot_membership_functions():
    """Imagen base64 con las MF y sus límites (idéntico a lo definido en Colab)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    x_t = np.arange(0, 37, 1)
    x_f = np.arange(0, 51, 1)
    x_s = np.arange(8, 18, 0.1)
    x_out = np.arange(0, 101, 1)

    # Tiempo
    ax = axes[0,0]
    ax.plot(x_t, [trapmf(x, 0,0,6,12) for x in x_t], label="Nuevo")
    ax.plot(x_t, [trimf(x, 6,18,30) for x in x_t], label="Regular")
    ax.plot(x_t, [trapmf(x,18,24,36,36) for x in x_t], label="Veterano")
    ax.set_title("Tiempo de Suscripción (meses)"); ax.set_ylim(-0.05,1.05); ax.legend()

    # Frecuencia
    ax = axes[0,1]
    ax.plot(x_f, [trapmf(x,0,0,5,10) for x in x_f], label="Baja")
    ax.plot(x_f, [trimf(x,5,15,25) for x in x_f], label="Media")
    ax.plot(x_f, [trapmf(x,20,30,50,50) for x in x_f], label="Alta")
    ax.set_title("Frecuencia de Uso (visitas/mes)"); ax.set_ylim(-0.05,1.05); ax.legend()

    # Suscripción (precio)
    ax = axes[1,0]
    ax.plot(x_s, [trimf(x,9,10,11) for x in x_s], label="Básica")
    ax.plot(x_s, [trimf(x,10,12,14) for x in x_s], label="Estándar")
    ax.plot(x_s, [trimf(x,13,15,17) for x in x_s], label="Premium")
    ax.set_title("Tipo de Suscripción (USD)"); ax.set_ylim(-0.05,1.05); ax.legend()

    # Salida (satisfacción)
    ax = axes[1,1]
    ax.plot(x_out, [trapmf(x,0,0,25,40) for x in x_out], label="Baja")
    ax.plot(x_out, [trimf(x,30,50,70) for x in x_out], label="Media")
    ax.plot(x_out, [trapmf(x,60,75,100,100) for x in x_out], label="Alta")
    ax.set_title("Nivel de Satisfacción (%)"); ax.set_ylim(-0.05,1.05); ax.legend()

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


# def plot_detailed_analysis(tm, fq, subv, satisfaction, fuzzy_strength):
#     """Genera un gráfico detallado del análisis de un registro específico"""
#     try:
#         fig, axes = plt.subplots(2, 2, figsize=(15, 12))
#         fig.suptitle(f'Análisis Detallado - Satisfacción Predicha: {satisfaction}%', 
#                     fontsize=16, fontweight='bold')
        
#         degrees = calculate_membership_degrees(tm, fq, subv)
        
#         ax1 = axes[0, 0]
#         x_time = np.arange(0, 37, 1)
#         y_short = [trapmf(x, 0, 0, 6, 12) for x in x_time]
#         y_medium = [trimf(x, 6, 18, 30) for x in x_time]
#         y_long = [trapmf(x, 18, 24, 36, 36) for x in x_time]
        
#         ax1.plot(x_time, y_short, 'r-', label='Nuevo', linewidth=2)
#         ax1.plot(x_time, y_medium, 'orange', label='Regular', linewidth=2)
#         ax1.plot(x_time, y_long, 'g-', label='Veterano', linewidth=2)
#         ax1.axvline(tm, color='black', linestyle='--', linewidth=2, label=f'Valor actual: {tm}')
#         ax1.scatter([tm], [degrees['short']], color='red', s=100, zorder=5)
#         ax1.scatter([tm], [degrees['medium']], color='orange', s=100, zorder=5)
#         ax1.scatter([tm], [degrees['long']], color='green', s=100, zorder=5)
#         ax1.set_title('Tiempo de Suscripción (meses)')
#         ax1.legend()
#         ax1.grid(True, alpha=0.3)
        
#         ax2 = axes[0, 1]
#         x_freq = np.arange(0, 51, 1)
#         y_lowf = [trapmf(x, 0, 0, 5, 10) for x in x_freq]
#         y_medf = [trimf(x, 5, 15, 25) for x in x_freq]
#         y_highf = [trapmf(x, 20, 30, 50, 50) for x in x_freq]
        
#         ax2.plot(x_freq, y_lowf, 'r-', label='Baja', linewidth=2)
#         ax2.plot(x_freq, y_medf, 'orange', label='Media', linewidth=2)
#         ax2.plot(x_freq, y_highf, 'g-', label='Alta', linewidth=2)
#         ax2.axvline(fq, color='black', linestyle='--', linewidth=2, label=f'Valor actual: {fq}')
#         ax2.scatter([fq], [degrees['lowf']], color='red', s=100, zorder=5)
#         ax2.scatter([fq], [degrees['medf']], color='orange', s=100, zorder=5)
#         ax2.scatter([fq], [degrees['highf']], color='green', s=100, zorder=5)
#         ax2.set_title('Frecuencia de Uso (visitas/mes)')
#         ax2.legend()
#         ax2.grid(True, alpha=0.3)
        
#         ax3 = axes[1, 0]
#         x_sub = np.arange(8, 18, 0.1)
#         y_basic = [trimf(x, 9, 10, 11) for x in x_sub]
#         y_standard = [trimf(x, 10, 12, 14) for x in x_sub]
#         y_premium = [trimf(x, 13, 15, 17) for x in x_sub]
        
#         ax3.plot(x_sub, y_basic, 'r-', label='Básica', linewidth=2)
#         ax3.plot(x_sub, y_standard, 'orange', label='Estándar', linewidth=2)
#         ax3.plot(x_sub, y_premium, 'g-', label='Premium', linewidth=2)
#         ax3.axvline(subv, color='black', linestyle='--', linewidth=2, label=f'Valor actual: ${subv}')
#         ax3.scatter([subv], [degrees['basic']], color='red', s=100, zorder=5)
#         ax3.scatter([subv], [degrees['standard']], color='orange', s=100, zorder=5)
#         ax3.scatter([subv], [degrees['premium']], color='green', s=100, zorder=5)
#         ax3.set_title('Tipo de Suscripción (USD)')
#         ax3.legend()
#         ax3.grid(True, alpha=0.3)
        
#         ax4 = axes[1, 1]
#         active_degrees = {k: v for k, v in degrees.items() if v > 0.01}
#         if active_degrees:
#             labels = list(active_degrees.keys())
#             values = list(active_degrees.values())
#             colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
            
#             bars = ax4.bar(labels, values, color=colors, alpha=0.7, edgecolor='black')
#             ax4.set_title('Grados de Membresía Activos')
#             ax4.set_ylabel('Grado de Membresía')
#             ax4.set_ylim(0, 1)
            
#             for bar, value in zip(bars, values):
#                 height = bar.get_height()
#                 ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
#                         f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
#         ax4.grid(True, alpha=0.3)
        
#         plt.tight_layout()
        
#         buffer = BytesIO()
#         plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight',
#                    facecolor='white', edgecolor='none')
#         buffer.seek(0)
#         plot_data = buffer.getvalue()
#         buffer.close()
#         plt.close()
        
#         return base64.b64encode(plot_data).decode()
        
#     except Exception as e:
#         print(f"Error generando gráfico de análisis detallado: {e}")
#         return None

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
    """
    Vista principal del modelo difuso.
    Permite elegir 'objetivo=baja' (Frec Med) o 'objetivo=alta' (Frec Agv).
    """
    try:
        objetivo = request.GET.get('objetivo', 'baja')  # 'baja' | 'alta'

        # CSV base para demo en la página
        csv_path = os.path.join(django_settings.BASE_DIR, 'sistema_fuzzy', 'Netflix_Userbase_Frecuencia.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError("El archivo base no se encuentra en sistema_fuzzy/Netflix_Userbase_Frecuencia.csv")

        df = pd.read_csv(csv_path)

        # Corre el modelo sobre el dataset base (render demo)
        df_result = evaluate_df(df, objetivo=objetivo)

        # Estadísticas
        stats = df_result['Predicted_Satisfaction'].describe()
        stats_dict = {
            'mean': round(stats['mean'], 2),
            'std': round(stats['std'], 2),
            'min': round(stats['min'], 2),
            'max': round(stats['max'], 2),
            'q25': round(stats['25%'], 2),
            'q50': round(stats['50%'], 2),
            'q75': round(stats['75%'], 2)
        }

        # Registro con máxima satisfacción (para visual detallada)
        best_idx = df_result['Predicted_Satisfaction'].idxmax()
        analyzed_record = df_result.loc[best_idx]

        # ----- gráfico de funciones de membresía (reutiliza tu helper) -----
        membership_plot = plot_membership_functions()
        membership_functions_data = generate_membership_functions_data()

        # ----- gráfico “detallado” (barras de grados de membresía para ese registro) -----
        fq_val = analyzed_record['Frec Med'] if objetivo == 'baja' else analyzed_record['Frec Agv']
        degs = calculate_membership_degrees(
            analyzed_record['Months diff'],
            fq_val,
            analyzed_record['Monthly Revenue']
        )
        fig, ax = plt.subplots(figsize=(10, 4))
        labels = list(degs.keys()); values = list(degs.values())
        bars = ax.bar(labels, values)
        ax.set_ylim(0, 1.05); ax.set_ylabel('Grado de pertenencia')
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_title('Grados de membresía activos (registro analizado)')
        buf = BytesIO(); plt.tight_layout(); plt.savefig(buf, format='png', bbox_inches='tight'); plt.close(fig)
        buf.seek(0)
        detailed_plot = base64.b64encode(buf.getvalue()).decode('utf-8')

        context = {
            'dataset_info': {
                'total_records': len(df_result),
                'columns': list(df_result.columns),
                'shape': df_result.shape,
            },
            'membership_plot': membership_plot,
            'membership_functions_data': membership_functions_data,
            'analyzed_record': {
                'index': int(best_idx) + 1,
                'months_diff': int(analyzed_record['Months diff']),
                'frequency': int(fq_val),
                'monthly_revenue': round(float(analyzed_record['Monthly Revenue']), 2),
                'predicted_satisfaction': round(float(analyzed_record['Predicted_Satisfaction']), 1),
                'fuzzy_strength': round(float(analyzed_record['Fuzzy_Strength']), 3),
            },
            'detailed_plot': detailed_plot,
            'statistics': stats_dict,
            'sample_data': df_result.head(10).to_dict('records'),
            'total_rules': len(define_fuzzy_rules()),
            'objetivo': objetivo,
            'error': None
        }

        return render(request, 'fuzzy_model_complete.html', context)

    except Exception as e:
        context = {
            'error': str(e),
            'dataset_info': None,
            'membership_plot': None,
            'analyzed_record': None,
            'detailed_plot': None,
            'statistics': None,
            'sample_data': None,
            'total_rules': 27,
            'objetivo': request.GET.get('objetivo', 'baja'),
        }
        return render(request, 'fuzzy_model_complete.html', context)

def upload_and_predict(request):
    """
    POST con:
      - file: CSV subido por el usuario
      - objetivo: 'baja' | 'alta'
    Devuelve un CSV enriquecido con Predicted_Satisfaction, Fuzzy_Strength, Satisfaction_Label,
    con formato LATAM (decimales 'xx,yy' entre comillas y separador coma).
    """
    if request.method == 'POST' and request.FILES.get('file'):
        try:
            objetivo = request.POST.get('objetivo', 'baja')
            df_out = evaluate_file(request.FILES['file'], objetivo=objetivo)  # usa Frec Med o Frec Agv

            # CSV EXACTO como tu screenshot: sep=',' y números "xx,yy"
            filename, payload = dataframe_to_csv_bytes(
                df_out,
                latam_strings=True,
                comma_sep_with_quotes=True,
                filename_prefix=f"fuzzy_{objetivo}"
            )

            resp = HttpResponse(payload, content_type='text/csv; charset=utf-8')
            resp['Content-Disposition'] = f'attachment; filename="{filename}"'
            return resp

        except Exception as e:
            return render(request, 'fuzzy_model_complete.html', {
                'error': f'Error al procesar el archivo: {e}',
                'dataset_info': None
            })

    # Si GET o no hay archivo, redirige a la página principal
    return HttpResponseRedirect(reverse('fuzzy_model_complete'))


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