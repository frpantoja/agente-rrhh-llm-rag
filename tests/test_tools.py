"""Tests para las herramientas del agente."""

from src.tools.consulta_tool import consultar_documentos
from src.tools.escritura_tool import generar_resumen
from src.tools.razonamiento_tool import analizar_situacion_laboral


def test_consulta_tool_tiene_nombre():
    assert consultar_documentos.name == "consultar_documentos"


def test_consulta_tool_tiene_descripcion():
    assert "documentos" in consultar_documentos.description.lower()
    assert "rrhh" in consultar_documentos.description.lower() or \
           "vacaciones" in consultar_documentos.description.lower()


def test_escritura_tool_tiene_nombre():
    assert generar_resumen.name == "generar_resumen"


def test_escritura_tool_tiene_descripcion():
    assert "resumen" in generar_resumen.description.lower()


def test_razonamiento_tool_tiene_nombre():
    assert analizar_situacion_laboral.name == "analizar_situacion_laboral"


def test_razonamiento_tool_tiene_descripcion():
    assert "situación" in analizar_situacion_laboral.description.lower() or \
           "situacion" in analizar_situacion_laboral.description.lower()
