import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


st.set_page_config(
    page_title="Predictor de Diabetes",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).parent.parent / "models" / "random_forest_regressor_n_estimators-300_RS-17.sav"

st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1a6faf;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f6ff;
        border-radius: 12px;
        padding: 1.2rem 1rem;
        text-align: center;
        border: 1px solid #c8ddf5;
    }
    .result-positive {
        background: #fff0f0;
        border: 2px solid #e53935;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .result-negative {
        background: #f0fff4;
        border: 2px solid #43a047;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .result-title {
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }
    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1a6faf;
        border-bottom: 2px solid #c8ddf5;
        padding-bottom: 0.3rem;
        margin-bottom: 1rem;
    }
    .stSlider > div > div > div > div { background: #1a6faf; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, set):
        obj = next(iter(obj))
    return obj

try:
    model = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False


FEATURE_INFO = {
    "Pregnancies":              {"label": "Embarazos",                       "min": 0,   "max": 17,   "default": 3,    "step": 1,   "desc": "Número de embarazos"},
    "Glucose":                  {"label": "Glucosa (mg/dL)",                 "min": 0,   "max": 200,  "default": 120,  "step": 1,   "desc": "Concentración de glucosa"},
    "BloodPressure":            {"label": "Presión Arterial Diastólica (mmHg)", "min": 0, "max": 122, "default": 72,  "step": 1,   "desc": "Presión arterial diastólica"},
    "Insulin":                  {"label": "Insulina (μU/mL)",                 "min": 0,   "max": 846,  "default": 79,   "step": 1,   "desc": "Insulina sérica a las 2 horas"},
    "BMI":                      {"label": "IMC (kg/m²)",                      "min": 0.0, "max": 67.1, "default": 31.0, "step": 0.1, "desc": "Índice de Masa Corporal"},
    "DiabetesPedigreeFunction": {"label": "Función Pedigrí de Diabetes",      "min": 0.0, "max": 2.5,  "default": 0.47, "step": 0.01,"desc": "Probabilidad de diabetes según historial familiar"},
    "Age":                      {"label": "Edad (años)",                      "min": 21,  "max": 81,   "default": 33,   "step": 1,   "desc": "Edad en años"},
}

FEATURE_MEANS = {
    "Pregnancies": 3.8, "Glucose": 120.9, "BloodPressure": 69.1,
    "Insulin": 79.8, "BMI": 31.9,
    "DiabetesPedigreeFunction": 0.47, "Age": 33.2,
}
FEATURE_STDS = {
    "Pregnancies": 3.4, "Glucose": 31.97, "BloodPressure": 19.36,
    "Insulin": 115.24, "BMI": 7.88,
    "DiabetesPedigreeFunction": 0.33, "Age": 11.76,
}
FEATURE_HEALTHY_RANGES = {
    "Pregnancies": (0, 5), "Glucose": (70, 100), "BloodPressure": (60, 80),
    "Insulin": (16, 166), "BMI": (18.5, 24.9),
    "DiabetesPedigreeFunction": (0.0, 0.5), "Age": (21, 40),
}


st.markdown('<div class="main-header"> Predictor de Diabetes</div>', unsafe_allow_html=True)

if not model_loaded:
    st.error(f"⚠️ No se encontró el modelo en: `{MODEL_PATH}`\n\nVerifica que el archivo `.sav` exista en esa ruta.")
    st.stop()


st.sidebar.markdown("## Datos del Paciente")
st.sidebar.markdown("Ajusta los valores para obtener la predicción.")

user_input = {}
for feature, info in FEATURE_INFO.items():
    st.sidebar.markdown(f"**{info['label']}**  \n<small>{info['desc']}</small>", unsafe_allow_html=True)
    if info["step"] == 1:
        val = st.sidebar.slider(
            label=info["label"],
            min_value=int(info["min"]),
            max_value=int(info["max"]),
            value=int(info["default"]),
            step=1,
            label_visibility="collapsed",
        )
    else:
        val = st.sidebar.slider(
            label=info["label"],
            min_value=float(info["min"]),
            max_value=float(info["max"]),
            value=float(info["default"]),
            step=float(info["step"]),
            label_visibility="collapsed",
        )
    user_input[feature] = val

predict_btn = st.sidebar.button("🔍 Predecir", use_container_width=True, type="primary")


tab1, tab2, tab3 = st.tabs(["Resultado y Análisis", "Dashboard Visual", "Información del Modelo"])


with tab1:
    input_df = pd.DataFrame([user_input])


    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    prob_positivo = proba[1]
    prob_negativo = proba[0]

    col_res, col_prob = st.columns([1, 1])

    with col_res:
        st.markdown('<div class="section-title">Resultado de la Predicción</div>', unsafe_allow_html=True)
        if prediction == 1:
            st.markdown(f"""
            <div class="result-positive">
                <div class="result-title" style="color:#e53935;">Posible Diabetes Detectada</div>
                <div>El modelo estima un riesgo <strong>ALTO</strong> de diabetes.</div>
                <div style="margin-top:0.5rem;font-size:0.9rem;color:#888;">Se recomienda consultar con un médico.</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-negative">
                <div class="result-title" style="color:#43a047;">Sin Señales de Diabetes</div>
                <div>El modelo estima un riesgo <strong>BAJO</strong> de diabetes.</div>
                <div style="margin-top:0.5rem;font-size:0.9rem;color:#888;">Continúa con hábitos saludables.</div>
            </div>""", unsafe_allow_html=True)

    with col_prob:
        st.markdown('<div class="section-title">Probabilidades del Modelo</div>', unsafe_allow_html=True)
        fig_prob, ax_prob = plt.subplots(figsize=(4, 2.8))
        colors = ["#43a047", "#e53935"]
        bars = ax_prob.barh(["Sin Diabetes", "Con Diabetes"], [prob_negativo, prob_positivo], color=colors)
        for bar, val in zip(bars, [prob_negativo, prob_positivo]):
            ax_prob.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                         f"{val:.1%}", va="center", fontsize=11, fontweight="bold")
        ax_prob.set_xlim(0, 1.15)
        ax_prob.set_xlabel("Probabilidad")
        ax_prob.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax_prob.set_title("Distribución de Probabilidad")
        fig_prob.tight_layout()
        st.pyplot(fig_prob)
        plt.close()

    st.markdown("---")

    st.markdown('<div class="section-title">Comparación con Rangos de Referencia Saludables</div>', unsafe_allow_html=True)

    cols = st.columns(4)
    for i, (feat, val) in enumerate(user_input.items()):
        lo, hi = FEATURE_HEALTHY_RANGES[feat]
        info = FEATURE_INFO[feat]
        in_range = lo <= val <= hi
        icon = "✅" if in_range else "⚠️"
        color = "#43a047" if in_range else "#e53935"
        with cols[i % 4]:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:1.4rem">{icon}</div>
                <div style="font-weight:600;font-size:0.85rem;color:#333;">{info['label']}</div>
                <div style="font-size:1.3rem;font-weight:700;color:{color};">{val}</div>
                <div style="font-size:0.75rem;color:#777;">Rango: {lo} – {hi}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    with st.expander("📋 Ver valores ingresados"):
        display_df = pd.DataFrame({
            "Variable": [FEATURE_INFO[f]["label"] for f in user_input],
            "Valor": list(user_input.values()),
            "Media del dataset": [FEATURE_MEANS[f] for f in user_input],
        })
        st.dataframe(display_df, use_container_width=True, hide_index=True)

with tab2:
    st.markdown('<div class="section-title">📈 Análisis Visual del Paciente vs. Población</div>', unsafe_allow_html=True)


    st.markdown("**Valor del paciente vs. Media del dataset**")
    feat_labels = [FEATURE_INFO[f]["label"] for f in FEATURE_INFO]
    means = [FEATURE_MEANS[f] for f in FEATURE_INFO]
    patient_vals = [user_input[f] for f in FEATURE_INFO]

    fig_b, ax_b = plt.subplots(figsize=(5, 5))
    x = np.arange(len(feat_labels))
    width = 0.35
    bars1 = ax_b.bar(x - width/2, means, width, label="Media dataset", color="#90caf9", edgecolor="white")
    bars2 = ax_b.bar(x + width/2, patient_vals, width, label="Paciente", color="#1a6faf", edgecolor="white")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(feat_labels, rotation=45, ha="right", fontsize=7.5)
    ax_b.legend(fontsize=8)
    ax_b.set_title("Comparación por variable", fontsize=10)
    ax_b.grid(axis="y", alpha=0.3)
    fig_b.tight_layout()
    st.pyplot(fig_b)
    plt.close()

  
    st.markdown("---")
    st.markdown('<div class="section-title"> Importancia de Variables en el Modelo</div>', unsafe_allow_html=True)

    importances = model.feature_importances_
    feat_names = list(FEATURE_INFO.keys())
    feat_labels_fi = [FEATURE_INFO[f]["label"] for f in feat_names]
    sorted_idx = np.argsort(importances)

    fig_fi, ax_fi = plt.subplots(figsize=(8, 4))
    colors_fi = ["#1a6faf" if f in ["Glucose", "BMI", "Age", "DiabetesPedigreeFunction"] else "#90caf9"
                 for f in [feat_names[i] for i in sorted_idx]]
    bars_fi = ax_fi.barh(
        [feat_labels_fi[i] for i in sorted_idx],
        importances[sorted_idx],
        color=colors_fi,
        edgecolor="white",
    )
    for bar, val in zip(bars_fi, importances[sorted_idx]):
        ax_fi.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                   f"{val:.3f}", va="center", fontsize=8.5)
    ax_fi.set_xlabel("Importancia")
    ax_fi.set_title("Feature Importances — Random Forest (n=300)")
    ax_fi.grid(axis="x", alpha=0.3)
    fig_fi.tight_layout()
    st.pyplot(fig_fi)
    plt.close()

# ══════════════════════════════════════════════
# TAB 3 — INFO DEL MODELO
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">ℹ️ Detalles del Modelo</div>', unsafe_allow_html=True)

    col_i1, col_i2 = st.columns(2)
    with col_i1:
        st.markdown("""
| Parámetro | Valor |
|---|---|
| **Algoritmo** | Random Forest Classifier |
| **n_estimators** | 300 |
| **random_state** | 17 |
| **Dataset** | Pima Indians Diabetes |
| **Variable objetivo** | Outcome (0/1) |
""")
    with col_i2:
        st.markdown("""
| Variable | Descripción |
|---|---|
| Pregnancies | Número de embarazos |
| Glucose | Glucosa en plasma (mg/dL) |
| BloodPressure | Presión arterial diastólica |
| Insulin | Insulina sérica 2h |
| BMI | Índice de Masa Corporal |
| DiabetesPedigreeFunction | Función pedigrí |
| Age | Edad (años) |
""")

    st.markdown("""
---
> ⚕️ **Aviso importante:** Esta herramienta es solo para fines educativos y demostrativos.
> No reemplaza el diagnóstico médico profesional. Ante cualquier duda, consulta a un médico.
""")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
