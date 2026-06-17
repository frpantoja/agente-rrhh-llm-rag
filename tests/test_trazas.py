"""Tests para el modulo de trazabilidad."""

import json
import tempfile
from pathlib import Path

from src.observabilidad.trazas import (
    hash_consulta,
    nuevo_trace_id,
    trace_event,
    span,
    leer_trazas,
)


def test_hash_consulta_es_consistente():
    h1 = hash_consulta("¿Cuántos días de vacaciones tengo?")
    h2 = hash_consulta("¿Cuántos días de vacaciones tengo?")
    assert h1 == h2


def test_hash_consulta_distinto_para_textos_distintos():
    h1 = hash_consulta("consulta uno")
    h2 = hash_consulta("consulta dos")
    assert h1 != h2


def test_hash_consulta_no_revela_texto_original():
    texto = "Pedro Gonzalez, RUT 12.345.678-9"
    h = hash_consulta(texto)
    assert texto not in h
    assert len(h) == 16


def test_nuevo_trace_id_es_unico():
    t1 = nuevo_trace_id()
    t2 = nuevo_trace_id()
    assert t1 != t2


def test_span_registra_evento_exitoso():
    trace_id = nuevo_trace_id()
    with span(trace_id, "operacion_test"):
        pass  # operacion exitosa


def test_span_relanza_excepcion():
    trace_id = nuevo_trace_id()
    raised = False
    try:
        with span(trace_id, "operacion_con_error"):
            raise ValueError("error de prueba")
    except ValueError:
        raised = True
    assert raised


def test_leer_trazas_archivo_inexistente():
    eventos = leer_trazas("/ruta/que/no/existe.jsonl")
    assert eventos == []


def test_leer_trazas_parsea_jsonl():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"trace_id": "abc", "span": "test", "event": "start"}) + "\n")
        f.write(json.dumps({"trace_id": "abc", "span": "test", "event": "end"}) + "\n")
        ruta = f.name

    eventos = leer_trazas(ruta)
    assert len(eventos) == 2
    assert eventos[0]["trace_id"] == "abc"

    Path(ruta).unlink()
