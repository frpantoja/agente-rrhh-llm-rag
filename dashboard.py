"""
Dashboard de observabilidad para el Agente de RRHH.

Visualiza las trazas registradas en logs/trazas.jsonl: latencia por
consulta, distribucion de uso de herramientas, tasa de error, y
deteccion de anomalias mediante z-score sobre la ventana de consultas
recientes.

Ejecucion:
    streamlit run dashboard.py
"""

import statistics
from collections import Counter

import pandas as pd
import streamlit as st

from src.observabilidad.trazas import leer_trazas

st.set_page_config(page_title="Observabilidad - Agente RRHH", layout="wide")

st.title("Dashboard de Observabilidad — Agente de RRHH")
st.caption("Comercial Andina SpA · Métricas basadas en trazas reales del agente")


def cargar_datos():
    eventos = leer_trazas()
    consultas = [
        e for e in eventos
        if e.get("span") == "consulta_completa" and e.get("event") in ("end", "error")
    ]
    return eventos, consultas


eventos, consultas = cargar_datos()

if not consultas:
    st.warning(
        "No hay trazas registradas todavia. Ejecuta `python app.py` y realiza "
        "algunas consultas antes de ver el dashboard."
    )
    st.stop()

df = pd.DataFrame(consultas)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

# --- Métricas principales ---
total = len(df)
errores = len(df[df["event"] == "error"])
tasa_error = errores / total if total else 0
latencia_promedio = df["duration_ms"].mean()
latencia_p95 = df["duration_ms"].quantile(0.95)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Consultas totales", total)
col2.metric("Tasa de error", f"{tasa_error:.1%}")
col3.metric("Latencia promedio", f"{latencia_promedio:.0f} ms")
col4.metric("Latencia p95", f"{latencia_p95:.0f} ms")

estado = "Saludable" if tasa_error <= 0.10 else ("Degradado" if tasa_error <= 0.30 else "Critico")
color_estado = {"Saludable": "green", "Degradado": "orange", "Critico": "red"}[estado]
st.markdown(f"**Estado del agente:** :{color_estado}[{estado}]")

st.divider()

# --- Latencia por consulta ---
st.subheader("Latencia por consulta")
st.line_chart(df.set_index("timestamp")["duration_ms"])

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Uso de herramientas")
    herramientas_todas = []
    for h in df["herramientas"].dropna():
        if isinstance(h, list):
            herramientas_todas.extend(h)
    if herramientas_todas:
        conteo = Counter(herramientas_todas)
        st.bar_chart(pd.Series(conteo))
    else:
        st.info("Sin uso de herramientas registrado todavía.")

with col_b:
    st.subheader("Anomalías de latencia (z-score)")
    if "es_anomalia" in df.columns:
        anomalias = df[df["es_anomalia"] == True]
        st.metric("Anomalías detectadas", len(anomalias))
        if not anomalias.empty:
            st.dataframe(
                anomalias[["timestamp", "trace_id", "duration_ms", "z_score"]],
                hide_index=True,
            )
    else:
        st.info("Aún no hay suficientes datos para detectar anomalías.")

st.divider()

# --- Tabla de trazas detallada ---
st.subheader("Registro de consultas")
columnas_mostrar = ["timestamp", "trace_id", "duration_ms", "status", "herramientas"]
columnas_disponibles = [c for c in columnas_mostrar if c in df.columns]
st.dataframe(df[columnas_disponibles].sort_values("timestamp", ascending=False), hide_index=True)

st.divider()

# --- Trazas lentas ---
st.subheader("Consultas más lentas")
umbral_lento = st.slider("Umbral de latencia (ms)", 100, 5000, 1500, step=100)
lentas = df[df["duration_ms"] > umbral_lento]
st.write(f"{len(lentas)} consultas superaron {umbral_lento}ms")
if not lentas.empty:
    st.dataframe(
        lentas[["timestamp", "trace_id", "duration_ms"]].sort_values(
            "duration_ms", ascending=False
        ),
        hide_index=True,
    )
