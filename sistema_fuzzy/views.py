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
    """Genera un PNG (base64) con las MF de entrada/salida."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Funciones de Membresía del Sistema de Lógica Difusa', fontsize=16, fontweight='bold')

        # Tiempo (meses)
        x_time = np.arange(0, 37, 1)
        axes[0, 0].plot(x_time, [trapmf(x, 0, 0, 6, 12) for x in x_time], label='Nuevo')
        axes[0, 0].plot(x_time, [trimf(x, 6, 18, 30) for x in x_time], label='Regular')
        axes[0, 0].plot(x_time, [trapmf(x, 18, 24, 36, 36) for x in x_time], label='Veterano')
        axes[0, 0].set_title('Tiempo de Suscripción (meses)'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=.3)

        # Frecuencia (visitas/mes)
        x_fq = np.arange(0, 51, 1)
        axes[0, 1].plot(x_fq, [trapmf(x, 0, 0, 5, 10) for x in x_fq], label='Baja')
        axes[0, 1].plot(x_fq, [trimf(x, 5, 15, 25) for x in x_fq], label='Media')
        axes[0, 1].plot(x_fq, [trapmf(x, 20, 30, 50, 50) for x in x_fq], label='Alta')
        axes[0, 1].set_title('Frecuencia de Uso (visitas/mes)'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=.3)

        # Plan (USD)
        x_sub = np.arange(8, 18, 0.1)
        axes[1, 0].plot(x_sub, [trimf(x, 9, 10, 11) for x in x_sub], label='Básico')
        axes[1, 0].plot(x_sub, [trimf(x, 10, 12, 14) for x in x_sub], label='Estándar')
        axes[1, 0].plot(x_sub, [trimf(x, 13, 15, 17) for x in x_sub], label='Premium')
        axes[1, 0].set_title('Tipo de Suscripción (USD)'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=.3)

        # Satisfacción (%)
        x_sat = np.arange(0, 101, 1)
        axes[1, 1].plot(x_sat, [trapmf(x, 0, 0, 25, 40) for x in x_sat], label='Baja', linestyle='--')
        axes[1, 1].plot(x_sat, [trimf(x, 30, 50, 70) for x in x_sat], label='Media', linestyle='--')
        axes[1, 1].plot(x_sat, [trapmf(x, 60, 75, 100, 100) for x in x_sat], label='Alta', linestyle='--')
        axes[1, 1].set_title('Nivel de Satisfacción (%)'); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=.3)

        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        buf.close()
        plt.close(fig)
        return img_b64
    except Exception:
        plt.close('all')
        return None


def plot_detailed_analysis(tm, fq, subv, satisfaction, fuzzy_strength):
    """
    Genera un PNG (base64) con:
    - MF de tiempo, frecuencia y plan con el valor actual marcado
    - Barras de grados de membresía activos + línea del centroide de salida
    """
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Análisis Detallado - Satisfacción Predicha: {satisfaction:.1f}%', 
                     fontsize=16, fontweight='bold')

        # 1) Tiempo
        x_tm = np.linspace(0, 40, 400)
        axes[0, 0].plot(x_tm, [trapmf(x, 0, 0, 6, 12) for x in x_tm], label='Nuevo')
        axes[0, 0].plot(x_tm, [trimf(x, 6, 18, 30) for x in x_tm], label='Regular')
        axes[0, 0].plot(x_tm, [trapmf(x, 18, 24, 36, 36) for x in x_tm], label='Veterano')
        axes[0, 0].axvline(tm, color='black', linestyle='--', linewidth=2, label=f'Valor: {tm:.0f}')
        axes[0, 0].set_title("Tiempo de Suscripción (meses)")
        axes[0, 0].legend(); axes[0, 0].grid(True, alpha=.3)

        # 2) Frecuencia
        x_fq = np.linspace(0, 50, 400)
        axes[0, 1].plot(x_fq, [trapmf(x, 0, 0, 5, 10) for x in x_fq], label='Baja')
        axes[0, 1].plot(x_fq, [trimf(x, 5, 15, 25) for x in x_fq], label='Media')
        axes[0, 1].plot(x_fq, [trapmf(x, 20, 30, 50, 50) for x in x_fq], label='Alta')
        axes[0, 1].axvline(fq, color='black', linestyle='--', linewidth=2, label=f'Valor: {fq:.0f}')
        axes[0, 1].set_title("Frecuencia de Uso (visitas/mes)")
        axes[0, 1].legend(); axes[0, 1].grid(True, alpha=.3)

        # 3) Plan
        x_sub = np.linspace(8, 18, 400)
        axes[1, 0].plot(x_sub, [trimf(x, 9, 10, 11) for x in x_sub], label='Básico')
        axes[1, 0].plot(x_sub, [trimf(x, 10, 12, 14) for x in x_sub], label='Estándar')
        axes[1, 0].plot(x_sub, [trimf(x, 13, 15, 17) for x in x_sub], label='Premium')
        axes[1, 0].axvline(subv, color='black', linestyle='--', linewidth=2, label=f'Valor: ${subv:.2f}')
        axes[1, 0].set_title("Tipo de Suscripción (USD)")
        axes[1, 0].legend(); axes[1, 0].grid(True, alpha=.3)

        # 4) Barras de grados activos + salida
        degrees = calculate_membership_degrees(tm, fq, subv)
        active = {k: v for k, v in degrees.items() if v > 0.01}
        labels = list(active.keys()); values = list(active.values())

        bars = axes[1, 1].bar(labels, values, color=plt.cm.Set3(np.linspace(0, 1, len(labels))), edgecolor='black')
        axes[1, 1].set_title('Grados de Membresía Activos')
        axes[1, 1].set_ylim(0, 1); axes[1, 1].grid(True, alpha=.3)
        for bar, val in zip(bars, values):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., val + 0.02, f'{val:.2f}',
                            ha='center', va='bottom', fontsize=9)

        # Línea del centroide de salida
        axes[1, 1].axhline(0, color='gray', linewidth=0.5)
        axes[1, 1].twinx().axhline(0, color='white', alpha=0)  # mantener layout
        axes[1, 1].text(0.98, 0.92, f'Centroide: {satisfaction:.1f}%\nFuerza: {fuzzy_strength:.3f}',
                        transform=axes[1, 1].transAxes, ha='right',
                        bbox=dict(boxstyle="round", fc="lavender", ec="#999"))

        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        buf.close()
        plt.close(fig)
        return img_b64
    except Exception:
        plt.close('all')
        return None

def plot_detailed_activations_colab(tm, fq, subv):
    """
    Devuelve un PNG base64 con:
    - 3 gráficos de activación (tiempo, frecuencia, plan) con valor y grado
    - 1 gráfico de salida (agregación + centroide)
    """
    try:
        # ==== 1) Construir agregación y centroide, como en Colab ====
        x_sat = np.arange(0, 101, 1)
        sat_low  = [trapmf(x, 0, 0, 25, 40) for x in x_sat]
        sat_med  = [trimf(x, 30, 50, 70) for x in x_sat]
        sat_high = [trapmf(x, 60, 75, 100, 100) for x in x_sat]

        # grados de entrada
        deg = calculate_membership_degrees(tm, fq, subv)
        # agregar reglas
        rules = define_fuzzy_rules()
        agg = np.zeros_like(x_sat, dtype=float)
        for ants, (_, params) in rules:
            strength = min(deg[ants[0]], deg[ants[1]], deg[ants[2]])
            if strength <= 0:
                continue
            if len(params) == 3:
                mf_vals = np.array([trimf(x, *params) for x in x_sat])
            else:
                mf_vals = np.array([trapmf(x, *params) for x in x_sat])
            agg = np.maximum(agg, np.minimum(strength, mf_vals))

        result = float(np.sum(x_sat * agg) / np.sum(agg)) if agg.sum() > 0 else np.nan

        # ==== 2) Gráficos ====
        fig = plt.figure(figsize=(14, 14))

        # Tiempo
        ax1 = plt.subplot(4, 1, 1)
        x_tm = np.linspace(0, 40, 400)
        deg_tm = {
            'short': trapmf(tm, 0, 0, 6, 12),
            'medium': trimf(tm, 6, 18, 30),
            'long': trapmf(tm, 18, 24, 36, 36),
        }
        ax1.plot(x_tm, [trapmf(x, 0, 0, 6, 12) for x in x_tm], label='short')
        ax1.plot(x_tm, [trimf(x, 6, 18, 30) for x in x_tm], label='medium')
        ax1.plot(x_tm, [trapmf(x, 18, 24, 36, 36) for x in x_tm], label='long')
        ax1.axvline(tm, color='red', linestyle='--', label=f'Valor: {tm:.2f}')
        ax1.axhline(max(deg_tm.values()), color='blue', linestyle='--', label=f'Grado: {max(deg_tm.values()):.2f}')
        ax1.set_title("Tiempo de Suscripción (Months diff)")
        ax1.set_xlabel("Meses"); ax1.set_ylabel("Grado de pertenencia"); ax1.legend(); ax1.grid(True)

        # Frecuencia
        ax2 = plt.subplot(4, 1, 2)
        x_fq = np.linspace(0, 50, 400)
        deg_fq = {
            'lowf':  trapmf(fq, 0, 0, 5, 10),
            'medf':  trimf(fq, 5, 15, 25),
            'highf': trapmf(fq, 20, 30, 50, 50),
        }
        ax2.plot(x_fq, [trapmf(x, 0, 0, 5, 10) for x in x_fq], label='lowf')
        ax2.plot(x_fq, [trimf(x, 5, 15, 25) for x in x_fq], label='medf')
        ax2.plot(x_fq, [trapmf(x, 20, 30, 50, 50) for x in x_fq], label='highf')
        ax2.axvline(fq, color='red', linestyle='--', label=f'Valor: {fq:.2f}')
        ax2.axhline(max(deg_fq.values()), color='blue', linestyle='--', label=f'Grado: {max(deg_fq.values()):.2f}')
        ax2.set_title("Frecuencia de Uso (Frequency)")
        ax2.set_xlabel("Veces por mes"); ax2.set_ylabel("Grado de pertenencia"); ax2.legend(); ax2.grid(True)

        # Plan
        ax3 = plt.subplot(4, 1, 3)
        x_mv = np.linspace(8, 18, 400)
        deg_mv = {
            'basic':    trimf(subv, 9, 10, 11),
            'standard': trimf(subv, 10, 12, 14),
            'premium':  trimf(subv, 13, 15, 17),
        }
        ax3.plot(x_mv, [trimf(x, 9, 10, 11) for x in x_mv], label='basic')
        ax3.plot(x_mv, [trimf(x, 10, 12, 14) for x in x_mv], label='standard')
        ax3.plot(x_mv, [trimf(x, 13, 15, 17) for x in x_mv], label='premium')
        ax3.axvline(subv, color='red', linestyle='--', label=f'Valor: {subv:.2f}')
        ax3.axhline(max(deg_mv.values()), color='blue', linestyle='--', label=f'Per: {max(deg_mv.values()):.2f}')
        ax3.set_title("Tipo de Suscripción (Monthly Revenue)")
        ax3.set_xlabel("Precio ($)"); ax3.set_ylabel("Grado de pertenencia"); ax3.legend(); ax3.grid(True)

        # Salida + centroide
        ax4 = plt.subplot(4, 1, 4)
        ax4.plot(x_sat, sat_low,  label='Baja',  linestyle='--')
        ax4.plot(x_sat, sat_med,  label='Media', linestyle='--')
        ax4.plot(x_sat, sat_high, label='Alta',  linestyle='--')
        ax4.fill_between(x_sat, np.zeros_like(x_sat), agg, alpha=0.5, label='Salida agregada (recorte)')
        if not np.isnan(result):
            ax4.axvline(result, color='red',  linestyle='--', label=f'Centroide: {result:.2f}%')
            ax4.axhline(max(agg), color='blue', linestyle='--', label=f'Máx grado: {max(agg):.2f}')
        ax4.set_title("Salida difusa: Nivel de Satisfacción (Fidelización)")
        ax4.set_xlabel("Nivel de Satisfacción (%)"); ax4.set_ylabel("Grado de pertenencia"); ax4.legend(); ax4.grid(True)

        plt.tight_layout()
        buf = BytesIO(); plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
        buf.seek(0); b64 = base64.b64encode(buf.getvalue()).decode()
        buf.close(); plt.close(fig)
        return b64
    except Exception:
        plt.close('all')
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

# --- VISTA PRINCIPAL ---
# --- VISTA PRINCIPAL ---
def fuzzy_model_complete(request):
    try:
        objetivo = request.POST.get('objetivo') or request.GET.get('objetivo') or 'baja'
        membership_plot = plot_membership_functions()

        using_uploaded = False
        df_result = None
        user_ids = []
        selected_user_id = None
        analyzed = None
        detailed_plot = None
        stats_dict = None
        sample_data = None

        # 1) Si viene CSV en esta petición, procesamos y guardamos resumen en sesión
        if request.method == 'POST' and request.FILES.get('file'):
            using_uploaded = True
            df_result = evaluate_file(request.FILES['file'], objetivo=objetivo)

            # persistimos resumen para habilitar el combo en peticiones posteriores
            _stash_uploaded_summary(request, df_result, objetivo)

            # IDs para el combo (si hay col User ID). Si no, usa índice como string
            if 'User ID' in df_result.columns:
                user_ids = df_result['User ID'].astype(str).tolist()
            else:
                user_ids = df_result.index.astype(str).tolist()

            # estadísticas y muestra SOLO si se subió CSV
            if 'Predicted_Satisfaction' in df_result.columns:
                stats = df_result['Predicted_Satisfaction'].describe()
                stats_dict = {
                    'mean': round(stats.get('mean', 0), 2),
                    'std': round(stats.get('std', 0), 2),
                    'min': round(stats.get('min', 0), 2),
                    'max': round(stats.get('max', 0), 2),
                    'q25': round(stats.get('25%', 0), 2),
                    'q50': round(stats.get('50%', 0), 2),
                    'q75': round(stats.get('75%', 0), 2),
                }

            # columna auxiliar para la tabla (según objetivo)
            dfv = df_result.copy()
            if objetivo == 'baja' and 'Frec Med' in dfv.columns:
                dfv['Frequency_Selected'] = dfv['Frec Med']
            elif objetivo == 'alta' and 'Frec Agv' in dfv.columns:
                dfv['Frequency_Selected'] = dfv['Frec Agv']
            elif 'Frequency' in dfv.columns:
                dfv['Frequency_Selected'] = dfv['Frequency']
            else:
                dfv['Frequency_Selected'] = 0

            # normalizamos nombres para la tabla de muestra
            rename_map = {'Months diff': 'Months_diff', 'Monthly Revenue': 'Monthly_Revenue'}
            for a, b in rename_map.items():
                if a in dfv.columns and b not in dfv.columns:
                    dfv[b] = dfv[a]

            sample_data = dfv.head(10).to_dict('records')

            # ¿piden análisis detallado?
            selected_user_id = request.POST.get('user_id')
            if selected_user_id:
                # obtener fila por User ID o índice
                if 'User ID' in df_result.columns:
                    match = df_result[df_result['User ID'].astype(str) == selected_user_id]
                    row = match.iloc[0] if not match.empty else df_result.iloc[0]
                else:
                    # si no hay User ID, interpretamos user_id como índice (si es dígito)
                    if selected_user_id.isdigit() and int(selected_user_id) < len(df_result):
                        row = df_result.iloc[int(selected_user_id)]
                    else:
                        row = df_result.iloc[0]

                # valores para gráficas
                def _pick(series, *names, default=0):
                    for n in names:
                        if n in series.index: 
                            return series[n]
                    return default

                tm = float(_pick(row, 'Months diff', 'Months_diff', default=0))
                if objetivo == 'baja':
                    fq = float(_pick(row, 'Frec Med', 'Frequency', default=0))
                else:
                    fq = float(_pick(row, 'Frec Agv', 'Frequency', default=0))
                subv = float(_pick(row, 'Monthly Revenue', 'Monthly_Revenue', default=10))

                detailed_plot = plot_detailed_activations_colab(tm, fq, subv)
                analyzed = {
                    'index': (int(row.name) + 1) if isinstance(row.name, (int, np.integer)) else row.name,
                    'months_diff': int(tm),
                    'frequency': int(fq),
                    'monthly_revenue': round(subv, 2),
                    'predicted_satisfaction': round(float(row.get('Predicted_Satisfaction', 0)), 1),
                    'fuzzy_strength': round(float(row.get('Fuzzy_Strength', 0)), 3),
                }

        # 2) Si NO viene archivo en esta petición, usamos el estado guardado en sesión (fallback)
        else:
            sess = _get_uploaded_summary_from_session(request)
            if sess['has_csv']:
                using_uploaded = True
                user_ids = sess['user_ids']
                # si no vino 'objetivo' explícito en la URL/POST, usa el de sesión
                if not request.POST.get('objetivo') and not request.GET.get('objetivo'):
                    objetivo = sess['objetivo']

        # 3) Render
        context = {
            'membership_plot': membership_plot,
            'objetivo': objetivo,

            # control de interfaz
            'using_uploaded': using_uploaded,
            'user_ids': user_ids,                 # para el combo
            'selected_user_id': selected_user_id, # último elegido (si hay)
            'analyzed_record': analyzed,          # None hasta que elijan ID
            'detailed_plot': detailed_plot,       # None hasta que elijan ID

            # bloques que solo se muestran tras subir CSV
            'statistics': stats_dict,
            'sample_data': sample_data,

            'total_rules': 27,
            'error': None
        }
        return render(request, 'fuzzy_model_complete.html', context)

    except Exception as e:
        return render(request, 'fuzzy_model_complete.html', {
            'error': str(e),
            'membership_plot': None,
            'objetivo': request.POST.get('objetivo') or request.GET.get('objetivo') or 'baja',
            'using_uploaded': False,
            'user_ids': [],
            'selected_user_id': None,
            'analyzed_record': None,
            'detailed_plot': None,
            'statistics': None,
            'sample_data': None,
            'total_rules': 27
        })


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

# --- helpers de sesión ---
def _stash_uploaded_summary(request, df, objetivo):
    """Guarda en sesión que sí hay un CSV subido y la lista de User IDs."""
    if 'User ID' in df.columns:
        user_ids = df['User ID'].astype(str).tolist()
    else:
        # fallback si el CSV no trae 'User ID'
        user_ids = df.index.astype(str).tolist()
    request.session['has_uploaded_csv'] = True
    request.session['uploaded_user_ids'] = user_ids
    request.session['uploaded_objetivo'] = objetivo

def _get_uploaded_summary_from_session(request):
    """Lee de sesión el estado del último CSV subido."""
    return {
        'has_csv': request.session.get('has_uploaded_csv', False),
        'user_ids': request.session.get('uploaded_user_ids', []),
        'objetivo': request.session.get('uploaded_objetivo', 'baja'),
    }



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