"""
Herramienta de razonamiento normativo (Tool de razonamiento).

Permite al agente analizar situaciones laborales complejas que requieren
interpretar múltiples fuentes normativas y aplicar lógica de negocio.

Forma parte del conjunto de herramientas disponibles
"""

from langchain_core.tools import tool


@tool
def analizar_situacion_laboral(situacion: str) -> str:
    """
    Analiza una situación laboral compleja que requiere razonamiento sobre normativas.

    Usa esta herramienta cuando el trabajador plantee:
    - Un caso específico que requiere interpretar varias normas
    - Una comparación entre lo que dice el reglamento interno y la ley
    - Una situación con múltiples condiciones (ej: licencia + vacaciones)
    - Preguntas del tipo "¿qué pasa si...?" o "¿tengo derecho a...?"
    - Dudas sobre qué normativa aplica a su caso particular

    Args:
        situacion: Descripción de la situación laboral a analizar.

    Returns:
        Instrucciones para el agente sobre cómo analizar la situación.
    """
    return (
        f"El trabajador plantea la siguiente situación: {situacion}. "
        f"Para analizar correctamente este caso, sigue estos pasos:\n"
        f"1. Usa 'consultar_documentos' para buscar normativa interna relevante.\n"
        f"2. Usa 'consultar_documentos' para buscar normativa legal (externa) aplicable.\n"
        f"3. Compara ambas fuentes e identifica qué normas aplican al caso.\n"
        f"4. Presenta un análisis estructurado con:\n"
        f"   - Normativa aplicable (interna y externa)\n"
        f"   - Derechos del trabajador según la situación\n"
        f"   - Procedimiento recomendado\n"
        f"   - Recomendación de consultar con RRHH si el caso es complejo"
    )
