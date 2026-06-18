"""
Modulo de trazabilidad para el agente de RRHH.

Implementa un trace_id unico por consulta que conecta todos los pasos
de su procesamiento (retrieval, llamadas a herramientas, generacion).
Cada paso se registra como un span con su duracion, permitiendo
reconstruir el flujo completo de cualquier peticion.

Aplica minimizacion de datos: las consultas de los trabajadores se
registran como hash, no como texto plano, en linea con la Ley 21.719
sobre proteccion de datos personales.
"""

import hashlib
import json
import logging
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("trazas")
logger.setLevel(logging.INFO)
logger.propagate = False  # evita que basicConfig() del logger raiz duplique la salida

if not logger.handlers:
    file_handler = logging.FileHandler(LOG_DIR / "trazas.jsonl", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)


def hash_consulta(texto: str) -> str:
    """
    Genera un hash SHA-256 truncado de la consulta del trabajador.

    Se usa en lugar del texto completo para minimizar el almacenamiento
    de datos personales en los logs, conforme a principios de
    minimizacion de datos (Ley 21.719, GDPR Art. 5).
    """
    return hashlib.sha256(texto.encode()).hexdigest()[:16]


def nuevo_trace_id() -> str:
    """Genera un identificador unico de traza para una nueva consulta."""
    return str(uuid.uuid4())


def trace_event(trace_id: str, span: str, event: str,
                 duration_ms: float = None, metadata: dict = None) -> None:
    """
    Registra un evento de traza en formato JSON estructurado.

    Args:
        trace_id: Identificador unico de la consulta completa.
        span: Nombre de la operacion (ej: 'consultar_documentos', 'llm_call').
        event: Tipo de evento ('start', 'end', 'error').
        duration_ms: Duracion en milisegundos (solo para eventos 'end').
        metadata: Datos adicionales del evento (sin informacion sensible).
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trace_id": trace_id,
        "span": span,
        "event": event,
    }
    if duration_ms is not None:
        entry["duration_ms"] = round(duration_ms, 2)
    if metadata:
        entry.update(metadata)

    logger.info(json.dumps(entry, ensure_ascii=False))


@contextmanager
def span(trace_id: str, span_name: str, metadata: dict = None):
    """
    Context manager que mide la duracion de un bloque y registra
    su inicio, fin o error automaticamente.

    Uso:
        with span(trace_id, "consultar_documentos"):
            resultado = consultar_documentos(pregunta)
    """
    inicio = time.time()
    trace_event(trace_id, span_name, "start", metadata=metadata)
    try:
        yield
        duracion = (time.time() - inicio) * 1000
        trace_event(trace_id, span_name, "end", duration_ms=duracion,
                    metadata={"status": "ok"})
    except Exception as e:
        duracion = (time.time() - inicio) * 1000
        trace_event(trace_id, span_name, "error", duration_ms=duracion,
                    metadata={"status": "failed", "error_type": type(e).__name__})
        raise


def leer_trazas(archivo: str = None) -> list:
    """
    Lee y parsea todas las entradas del archivo de trazas JSON.

    Returns:
        Lista de diccionarios, uno por cada evento registrado.
    """
    ruta = Path(archivo) if archivo else LOG_DIR / "trazas.jsonl"
    if not ruta.exists():
        return []

    eventos = []
    with open(ruta, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if linea:
                try:
                    eventos.append(json.loads(linea))
                except json.JSONDecodeError:
                    continue
    return eventos
