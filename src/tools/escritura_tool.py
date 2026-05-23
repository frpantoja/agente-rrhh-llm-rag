"""
Herramienta de escritura y generación de documentos (Tool de escritura).

Permite al agente generar resúmenes, correos formales y documentos
estructurados a partir de la información de RRHH.

Forma parte del conjunto de herramientas disponibles
"""

from langchain_core.tools import tool


@tool
def generar_resumen(tema: str) -> str:
    """
    Genera un resumen estructurado sobre un tema específico de RRHH.

    Usa esta herramienta cuando el trabajador pida:
    - Un resumen de un procedimiento o política
    - Un documento formal o comunicación interna
    - Una explicación paso a paso de un trámite
    - Un correo para solicitar vacaciones, permisos, etc.

    Args:
        tema: El tema o procedimiento sobre el cual generar el resumen.

    Returns:
        Instrucciones para el agente indicando que debe generar el resumen.
    """
    return (
        f"El trabajador necesita un documento/resumen sobre: {tema}. "
        f"Primero consulta los documentos internos con la herramienta "
        f"'consultar_documentos' para obtener la información base, "
        f"y luego genera un resumen claro y estructurado con esa información. "
        f"Si el trabajador pide un correo o solicitud formal, usa un formato "
        f"apropiado con saludo, cuerpo y despedida."
    )
