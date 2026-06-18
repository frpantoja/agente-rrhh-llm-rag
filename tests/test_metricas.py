# tests/test_metricas.py
"""Tests para el modulo de metricas de observabilidad."""

from src.observabilidad.metricas import (
    AgentMetrics,
    ConsistenciaAnalyzer,
    AnomalyDetector,
    CasoPrueba,
    PrecisionEvaluator,
)


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
    assert "tokens_totales" in resumen
    assert "precision_recuperacion_promedio" in resumen


# --- Uso de recursos (tokens) ---

def test_metrics_sin_tokens_por_defecto():
    m = AgentMetrics()
    m.registrar_interaccion(duracion_ms=100, exito=True)
    assert m.tokens_totales == 0
    assert m.tokens_promedio_por_consulta == 0.0


def test_metrics_acumula_tokens_por_consulta():
    m = AgentMetrics()
    m.registrar_interaccion(duracion_ms=100, exito=True, tokens_entrada=500, tokens_salida=120)
    m.registrar_interaccion(duracion_ms=100, exito=True, tokens_entrada=300, tokens_salida=80)
    assert m.tokens_entrada_totales == 800
    assert m.tokens_salida_totales == 200
    assert m.tokens_totales == 1000
    assert m.tokens_promedio_por_consulta == 500.0


def test_metrics_estimar_costo_usd_es_mayor_a_cero_con_tokens():
    m = AgentMetrics()
    m.registrar_interaccion(duracion_ms=100, exito=True, tokens_entrada=10_000, tokens_salida=2_000)
    assert m.estimar_costo_usd() > 0


# --- Precision de recuperacion ---

def test_metrics_precision_recuperacion_none_sin_datos():
    m = AgentMetrics()
    m.registrar_interaccion(duracion_ms=100, exito=True)
    assert m.precision_recuperacion_promedio is None


def test_metrics_precision_recuperacion_promedia_scores():
    m = AgentMetrics()
    m.registrar_interaccion(duracion_ms=100, exito=True, scores_relevancia=[0.8, 0.6])
    m.registrar_interaccion(duracion_ms=100, exito=True, scores_relevancia=[0.4])
    assert m.precision_recuperacion_promedio == 0.6


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


# --- PrecisionEvaluator ---

def _casos_de_ejemplo():
    return [
        CasoPrueba(
            pregunta="¿Cuántos días de vacaciones me corresponden?",
            herramientas_aceptables=["consultar_documentos"],
            palabras_clave=["vacaciones", "días"],
        ),
        CasoPrueba(
            pregunta="¿Cuál es la capital de Francia?",
            herramientas_aceptables=[],
            palabras_clave=["recursos humanos"],
        ),
    ]


def test_precision_evaluator_pregunta_no_registrada_retorna_none():
    evaluador = PrecisionEvaluator(casos=_casos_de_ejemplo())
    resultado = evaluador.evaluar(
        pregunta="Una pregunta que no esta en los casos de prueba",
        herramientas_usadas=["consultar_documentos"],
        respuesta="Cualquier respuesta",
    )
    assert resultado is None


def test_precision_evaluator_herramienta_correcta():
    evaluador = PrecisionEvaluator(casos=_casos_de_ejemplo())
    resultado = evaluador.evaluar(
        pregunta="¿Cuántos días de vacaciones me corresponden?",
        herramientas_usadas=["consultar_documentos"],
        respuesta="Tienes derecho a 15 días de vacaciones al año.",
    )
    assert resultado["herramienta_correcta"] is True
    assert resultado["precision_contenido"] == 1.0


def test_precision_evaluator_herramienta_incorrecta():
    evaluador = PrecisionEvaluator(casos=_casos_de_ejemplo())
    resultado = evaluador.evaluar(
        pregunta="¿Cuántos días de vacaciones me corresponden?",
        herramientas_usadas=["generar_resumen"],
        respuesta="Aquí tienes un resumen.",
    )
    assert resultado["herramienta_correcta"] is False


def test_precision_evaluator_fuera_de_alcance_sin_herramientas_es_correcto():
    evaluador = PrecisionEvaluator(casos=_casos_de_ejemplo())
    resultado = evaluador.evaluar(
        pregunta="¿Cuál es la capital de Francia?",
        herramientas_usadas=[],
        respuesta="Solo atiendo temas de Recursos Humanos.",
    )
    assert resultado["herramienta_correcta"] is True


def test_precision_evaluator_resumen_agrega_resultados():
    evaluador = PrecisionEvaluator(casos=_casos_de_ejemplo())
    evaluador.evaluar(
        pregunta="¿Cuántos días de vacaciones me corresponden?",
        herramientas_usadas=["consultar_documentos"],
        respuesta="Tienes 15 días de vacaciones.",
    )
    evaluador.evaluar(
        pregunta="¿Cuál es la capital de Francia?",
        herramientas_usadas=["consultar_documentos"],
        respuesta="Lo siento, no puedo responder eso.",
    )
    resumen = evaluador.resumen()
    assert resumen["casos_evaluados"] == 2
    assert resumen["precision_herramienta"] == 0.5


def test_precision_evaluator_resumen_vacio_sin_evaluaciones():
    evaluador = PrecisionEvaluator(casos=_casos_de_ejemplo())
    resumen = evaluador.resumen()
    assert resumen["casos_evaluados"] == 0
    assert resumen["precision_herramienta"] is None