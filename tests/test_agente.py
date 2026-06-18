# tests/test_agente.py
"""Tests para el agente funcional."""

from types import SimpleNamespace
from unittest.mock import patch
from src.agente import RespuestaAgente, extraer_tokens_de_mensajes


def test_respuesta_agente_estructura():
    respuesta = RespuestaAgente(
        respuesta="Test",
        herramientas_usadas=["consultar_documentos"],
        consulta_original="¿Pregunta?",
        memoria_activa="2 mensajes",
    )
    assert respuesta.respuesta == "Test"
    assert len(respuesta.herramientas_usadas) == 1
    assert respuesta.consulta_original == "¿Pregunta?"


def test_respuesta_agente_defaults():
    respuesta = RespuestaAgente(respuesta="Test")
    assert respuesta.herramientas_usadas == []
    assert respuesta.consulta_original == ""
    assert respuesta.memoria_activa == ""
    assert respuesta.tokens_entrada == 0
    assert respuesta.tokens_salida == 0


@patch("src.agente.GITHUB_TOKEN", "")
def test_agente_sin_token_falla():
    """Verifica que se lanza error sin GITHUB_TOKEN."""
    from src.agente import AgenteRRHH

    raised = False
    try:
        AgenteRRHH()
    except ValueError:
        raised = True
    assert raised, "Debería lanzar ValueError sin GITHUB_TOKEN"


# --- extraer_tokens_de_mensajes ---

def _mensaje_con_uso(input_tokens, output_tokens):
    return SimpleNamespace(
        usage_metadata={"input_tokens": input_tokens, "output_tokens": output_tokens}
    )


def _mensaje_sin_uso():
    return SimpleNamespace()


def test_extraer_tokens_sin_mensajes():
    tokens_entrada, tokens_salida = extraer_tokens_de_mensajes([])
    assert tokens_entrada == 0
    assert tokens_salida == 0


def test_extraer_tokens_ignora_mensajes_sin_usage_metadata():
    mensajes = [_mensaje_sin_uso(), _mensaje_sin_uso()]
    tokens_entrada, tokens_salida = extraer_tokens_de_mensajes(mensajes)
    assert tokens_entrada == 0
    assert tokens_salida == 0


def test_extraer_tokens_suma_todos_los_pasos_del_agente():
    # Simula un ReAct de 2 pasos: razonamiento + respuesta final
    mensajes = [
        _mensaje_sin_uso(),  # HumanMessage / SystemMessage
        _mensaje_con_uso(500, 50),
        _mensaje_sin_uso(),  # ToolMessage
        _mensaje_con_uso(620, 90),
    ]
    tokens_entrada, tokens_salida = extraer_tokens_de_mensajes(mensajes)
    assert tokens_entrada == 1120
    assert tokens_salida == 140