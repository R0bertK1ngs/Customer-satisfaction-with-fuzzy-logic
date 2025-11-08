# sistema_fuzzy/modelo.py
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import csv

# =========================
#  Membresías (manuales)
# =========================
def trapmf(x, a, b, c, d):
    if x <= a or x >= d:
        return 0.0
    elif a < x < b:
        return (x - a) / (b - a)
    elif b <= x <= c:
        return 1.0
    else:  # c < x < d
        return (d - x) / (d - c)

def trimf(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    elif a < x < b:
        return (x - a) / (b - a)
    else:  # b <= x < c
        return (c - x) / (c - b)

# =========================
#  Reglas (27) Mamdani
# =========================
def define_fuzzy_rules():
    rules = [
        # lowf
        (['short', 'basic', 'lowf'],   ('low',    [0, 0, 20, 40])),
        (['short', 'standard', 'lowf'],('low',    [0, 0, 20, 40])),
        (['short', 'premium', 'lowf'], ('low',    [0, 0, 20, 40])),

        (['medium','basic', 'lowf'],   ('low',    [0, 0, 20, 40])),
        (['medium','standard','lowf'], ('low',    [0, 0, 20, 40])),
        (['medium','premium','lowf'],  ('low',    [0, 0, 20, 40])),

        (['long',  'basic', 'lowf'],   ('medium', [30, 50, 70])),
        (['long',  'standard','lowf'], ('medium', [30, 50, 70])),
        (['long',  'premium','lowf'],  ('medium', [30, 50, 70])),

        # medf
        (['short', 'basic', 'medf'],   ('low',    [0, 0, 20, 40])),
        (['short', 'standard','medf'], ('low',    [0, 0, 20, 40])),
        (['short', 'premium','medf'],  ('low',    [0, 0, 20, 40])),

        (['medium','basic', 'medf'],   ('medium', [30, 50, 70])),
        (['medium','standard','medf'], ('medium', [30, 50, 70])),
        (['medium','premium','medf'],  ('medium', [30, 50, 70])),

        (['long',  'basic', 'medf'],   ('medium', [30, 50, 70])),
        (['long',  'standard','medf'], ('medium', [30, 50, 70])),
        (['long',  'premium','medf'],  ('medium', [30, 50, 70])),

        # highf
        (['short', 'basic', 'highf'],   ('medium', [30, 50, 70])),
        (['short', 'standard','highf'], ('medium', [30, 50, 70])),
        (['short', 'premium','highf'],  ('medium', [30, 50, 70])),

        (['medium','basic', 'highf'],   ('high',   [60, 75, 100, 100])),
        (['medium','standard','highf'], ('high',   [60, 75, 100, 100])),
        (['medium','premium','highf'],  ('high',   [60, 75, 100, 100])),

        (['long',  'basic', 'highf'],   ('high',   [60, 75, 100, 100])),
        (['long',  'standard','highf'], ('high',   [60, 75, 100, 100])),
        (['long',  'premium','highf'],  ('high',   [60, 75, 100, 100])),
    ]
    return rules

# =========================
#  Grados de membresía
# =========================
def calculate_membership_degrees(tm, fq, subv):
    """
    tm: Months diff (meses de suscripción)
    fq: Frecuencia mensual (según objetivo: Frec Med o Frec Agv)
    subv: Monthly Revenue (10/12/15)
    """
    return {
        # Tiempo de suscripción
        'short':   trapmf(tm, 0, 0, 6, 12),
        'medium':  trimf(tm, 6, 18, 30),
        'long':    trapmf(tm, 18, 24, 36, 36),
        # Frecuencia (visitas/mes)
        'lowf':    trapmf(fq, 0, 0, 5, 10),
        'medf':    trimf(fq, 5, 15, 25),
        'highf':   trapmf(fq, 20, 30, 50, 50),
        # Plan via precio
        'basic':   trimf(subv, 9, 10, 11),
        'standard':trimf(subv, 10, 12, 14),
        'premium': trimf(subv, 13, 15, 17),
    }

# =========================
#  Inferencia Mamdani
# =========================
def apply_fuzzy_inference(tm, fq, subv, rules):
    """
    Retorna:
      - result: score defuzz (centroide) 0-100
      - max_strength: máxima fuerza (min de antecedentes) usada en agregación
    """
    degrees = calculate_membership_degrees(tm, fq, subv)
    x_sat = np.arange(0, 101, 1)  # universo salida
    # MF salida (Colab final: baja = [0,0,25,40], media = [30,50,70], alta = [60,75,100,100])
    sat_low  = lambda x: trapmf(x, 0, 0, 25, 40)
    sat_med  = lambda x: trimf(x, 30, 50, 70)
    sat_high = lambda x: trapmf(x, 60, 75, 100, 100)

    agg = np.zeros_like(x_sat, dtype=float)
    max_strength = 0.0

    for ants, (ctype, params) in rules:
        # fuerza de la regla (min de antecedentes)
        strength = min(degrees[ants[0]], degrees[ants[1]], degrees[ants[2]])
        if strength <= 0:
            continue

        # consecuente segun params (trimf si 3; trapmf si 4)
        if len(params) == 3:
            mf_vals = np.array([trimf(x, *params) for x in x_sat])
        else:
            mf_vals = np.array([trapmf(x, *params) for x in x_sat])

        # recorte por implicación min y agregación max
        agg = np.maximum(agg, np.minimum(strength, mf_vals))
        max_strength = max(max_strength, strength)

    # defuzz centroide
    if agg.sum() == 0:
        result = np.nan
    else:
        result = float(np.sum(x_sat * agg) / np.sum(agg))
    return result, max_strength

# =========================
#  Etiquetado lingüístico
# =========================
def get_satisfaction_label(value):
    low_deg  = trapmf(value, 0, 0, 25, 40)
    med_deg  = trimf(value, 30, 50, 70)
    high_deg = trapmf(value, 60, 75, 100, 100)
    degrees = {
        "Satisfaccion Baja":  low_deg,
        "Satisfaccion Media": med_deg,
        "Satisfaccion Alta":  high_deg,
    }
    return max(degrees, key=degrees.get)

# =========================
#  Helpers extracción datos
# =========================
def _get_from_record(record, *names, default=None):
    """Obtiene el primer nombre de columna existente (case-sensitive) del registro/serie/dict."""
    if hasattr(record, "get"):
        get = record.get
    elif isinstance(record, pd.Series):
        get = record.__getitem__
    else:
        # dict-like fallback
        get = lambda k: record[k]

    for n in names:
        if isinstance(record, pd.Series):
            if n in record.index:
                return record[n]
        else:
            if n in record:
                return get(n)
    return default

def _select_frequency(record, objetivo):
    """
    objetivo: 'baja' -> usa 'Frec Med'
              'alta' -> usa 'Frec Agv'
    fallback: 'Frequency' si existiera (compat)
    """
    if objetivo == "baja":
        val = _get_from_record(record, "Frec Med", "Frec_Med", default=None)
    else:  # "alta"
        val = _get_from_record(record, "Frec Agv", "Frec_Agv", default=None)

    if val is None:
        val = _get_from_record(record, "Frequency", default=0.0)
    return float(val)

# =========================
#  API pública
# =========================
def evaluate_single(record, objetivo="baja"):
    """
    record: dict o pd.Series con al menos:
      - 'Months diff' (o 'Months_diff')
      - 'Monthly Revenue' (o 'Monthly_Revenue')
      - 'Frec Med' y/o 'Frec Agv' (o 'Frequency' como fallback)
    objetivo: 'baja' | 'alta'  (elige mediana vs promedio)
    """
    tm   = float(_get_from_record(record, "Months diff", "Months_diff", default=0.0))
    fq   = _select_frequency(record, objetivo)
    subv = float(_get_from_record(record, "Monthly Revenue", "Monthly_Revenue", default=10.0))

    rules = define_fuzzy_rules()
    score, strength = apply_fuzzy_inference(tm, fq, subv, rules)

    if np.isnan(score):
        label = "Sin información"
        score_out = 0.0
        strength_out = 0.0
    else:
        label = get_satisfaction_label(score)
        score_out = round(score, 2)
        strength_out = round(float(strength), 3)

    return {
        "Predicted_Satisfaction": score_out,
        "Fuzzy_Strength": strength_out,
        "Satisfaction_Label": label,
        # opcional: eco de entradas usadas
        "Inputs": {"Months diff": tm, "Freq used": fq, "Monthly Revenue": subv, "Objetivo": objetivo},
    }

def evaluate_df(df: pd.DataFrame, objetivo="baja") -> pd.DataFrame:
    """
    Devuelve una copia del DataFrame con columnas agregadas:
      - Predicted_Satisfaction
      - Fuzzy_Strength
      - Satisfaction_Label
    Usa 'Frec Med' para objetivo='baja' y 'Frec Agv' para objetivo='alta'.
    """
    rules = define_fuzzy_rules()
    x_sat = np.arange(0, 101, 1)  # (solo para garantizar consistencia si deseas usarlo luego)

    results_score = []
    results_strength = []
    results_label = []

    for _, row in df.iterrows():
        tm   = float(_get_from_record(row, "Months diff", "Months_diff", default=0.0))
        fq   = _select_frequency(row, objetivo)
        subv = float(_get_from_record(row, "Monthly Revenue", "Monthly_Revenue", default=10.0))

        score, strength = apply_fuzzy_inference(tm, fq, subv, rules)
        if np.isnan(score):
            score_val = 0.0
            strength_val = 0.0
            label = "Sin información"
        else:
            score_val = round(float(score), 2)
            strength_val = round(float(strength), 3)
            label = get_satisfaction_label(score_val)

        results_score.append(score_val)
        results_strength.append(strength_val)
        results_label.append(label)

    out = df.copy()
    out["Predicted_Satisfaction"] = results_score
    out["Fuzzy_Strength"] = results_strength
    out["Satisfaction_Label"] = results_label
    return out

# ======== Entrada CSV (file-like) y salida CSV ========

def _read_uploaded_csv(file_obj):
    """
    Lee un CSV subido desde el front (InMemoryUploadedFile, TemporaryUploadedFile, file-like).
    Intenta UTF-8/',' y fallback a ';'/decimal=','.
    Devuelve un DataFrame.
    """
    import pandas as pd
    file_obj.seek(0)
    try:
        # Caso estándar: UTF-8, separador coma
        return pd.read_csv(file_obj)
    except Exception:
        file_obj.seek(0)
        try:
            # Fallback común en ES/LA: separador ';', decimales ','
            return pd.read_csv(file_obj, sep=';', decimal=',', encoding='utf-8')
        except Exception:
            file_obj.seek(0)
            # Último intento: intentar sin encoding explícito
            return pd.read_csv(file_obj, sep=';', decimal=',')

def evaluate_file(file_obj, objetivo="baja"):
    """
    Procesa un archivo CSV subido desde el front y devuelve un DataFrame
    enriquecido con:
      - Predicted_Satisfaction (0–100)
      - Fuzzy_Strength
      - Satisfaction_Label (Baja/Media/Alta, vía MF)
    objetivo:
      - 'baja' -> usa 'Frec Med'
      - 'alta' -> usa 'Frec Agv'
    """
    df_in = _read_uploaded_csv(file_obj)
    df_out = evaluate_df(df_in, objetivo=objetivo)
    return df_out

def dataframe_to_csv_bytes(
    df: pd.DataFrame,
    latam_strings=True,          # <- para "32,12"
    comma_sep_with_quotes=True,  # <- separador coma + valores entre comillas si llevan coma decimal
    filename_prefix='salida_con_satisfaccion'
):
    """
    Genera CSV:
      - Si latam_strings=True: convierte Predicted_Satisfaction y Fuzzy_Strength a 'xx,yy' (str)
      - Si comma_sep_with_quotes=True: usa sep=',' y cita valores con coma decimal ("32,12")
        (esto reproduce el formato de tu captura de Excel).
    Retorna (filename, bytes)
    """
    import io, datetime

    out_df = df.copy()
    if latam_strings:
        # convierte solo estas dos columnas a 'xx,yy'
        for col in ["Predicted_Satisfaction", "Fuzzy_Strength"]:
            if col in out_df.columns:
                out_df[col] = out_df[col].map(lambda s: str(s).replace('.', ',') if pd.notnull(s) else s)

    sep = ','
    quoting = csv.QUOTE_MINIMAL if comma_sep_with_quotes else csv.QUOTE_NONE

    buf = io.StringIO()
    out_df.to_csv(
        buf,
        index=False,
        sep=sep,
        quoting=quoting,   # <- esto hará "32,12" cuando lleve coma
        quotechar='"'
    )
    content = buf.getvalue().encode('utf-8')

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{filename_prefix}_{ts}.csv"
    return filename, content


# Fin del archivo sistema_fuzzy/modelo.py