"""Tests para el modulo de metricas de observabilidad."""

from src.observabilidad.metricas import AgentMetrics, ConsistenciaAnalyzer, AnomalyDetector


# --- AgentMetrics ---

def test_metrics_sin_interacciones():
    m = AgentMetrics()
    assert m.tasa_error == 0.0
    assert m.tasa_exito == 1.0
    assert m.latencia_promedio_ms == 0.0
    assert m.estado_salud() == "saludable"


def test_metrics_registrar_interaccion_exitosa():
    m = AgentMetrics()
    m.registrar_interaccion(duracion_ms=200, exito=True, herramientas=["consultar_documentos"])
    assert m.interacciones == 1
    assert m.tasa_exito == 1.0
    assert m.herramientas_usadas["consultar_documentos"] == 1


def test_metrics_registrar_interaccion_fallida():
    m = AgentMetrics()
    m.registrar_interaccion(duracion_ms=200, exito=False)
    assert m.errores == 1
    assert m.tasa_error == 1.0


def test_metrics_estado_degradado():
    m = AgentMetrics()
    for _ in range(8):
        m.registrar_interaccion(duracion_ms=100, exito=True)
    for _ in range(2):
        m.registrar_interaccion(duracion_ms=100, exito=False)
    assert m.estado_salud() == "degradado"


def test_metrics_estado_critico():
    m = AgentMetrics()
    for _ in range(5):
        m.registrar_interaccion(duracion_ms=100, exito=False)
    for _ in range(5):
        m.registrar_interaccion(duracion_ms=100, exito=True)
    assert m.estado_salud() == "critico"


def test_metrics_latencia_p95():
    m = AgentMetrics()
    for d in [100, 100, 100, 100, 100, 100, 100, 100, 100, 1000]:
        m.registrar_interaccion(duracion_ms=d, exito=True)
    assert m.latencia_p95_ms == 1000


def test_metrics_resumen_tiene_campos_clave():
    m = AgentMetrics()
    m.registrar_interaccion(duracion_ms=150, exito=True, herramientas=["consultar_documentos"])
    resumen = m.resumen()
    assert "interacciones" in resumen
    assert "tasa_exito" in resumen
    assert "estado_salud" in resumen
    assert "herramientas_usadas" in resumen


# --- ConsistenciaAnalyzer ---

def test_consistencia_sin_muestras_suficientes():
    c = ConsistenciaAnalyzer()
    c.registrar("consulta_simple", 100)
    resultado = c.consistencia_por_categoria()
    assert resultado["consulta_simple"]["cv"] is None


def test_consistencia_calcula_coeficiente_variacion():
    c = ConsistenciaAnalyzer()
    c.registrar("consulta_simple", 100)
    c.registrar("consulta_simple", 110)
    c.registrar("consulta_simple", 90)
    resultado = c.consistencia_por_categoria()
    assert resultado["consulta_simple"]["muestras"] == 3
    assert resultado["consulta_simple"]["coeficiente_variacion"] is not None


# --- AnomalyDetector ---

def test_anomaly_detector_sin_historia_no_detecta():
    detector = AnomalyDetector()
    resultado = detector.procesar(500)
    assert resultado["es_anomalia"] is False


def test_anomaly_detector_detecta_pico():
    detector = AnomalyDetector(window_size=20, z_threshold=2.0)
    base = [95, 100, 105, 98, 102, 97, 103, 99, 101, 100,
            96, 104, 98, 102, 100]
    for valor in base:
        detector.procesar(valor)
    resultado = detector.procesar(5000)
    assert resultado["es_anomalia"] is True
    assert resultado["z_score"] > 2.0
