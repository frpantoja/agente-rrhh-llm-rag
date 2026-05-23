"""Tests para el agente funcional."""

from unittest.mock import patch
from src.agente import RespuestaAgente


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
