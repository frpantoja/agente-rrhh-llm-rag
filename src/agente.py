"""
Agente funcional de RRHH con herramientas, memoria y planificación.

Este módulo implementa el agente central del sistema, que integra:
- Herramientas de consulta, escritura y razonamiento (IE1, IE2)
- Memoria de corto plazo conversacional (IE3)
- Recuperación de contexto semántico via RAG (IE4)
- Planificación y toma de decisiones adaptativa (IE5, IE6)

El agente usa LangGraph con el patrón ReAct (Reasoning + Acting),
que permite al agente razonar paso a paso sobre qué herramienta usar
antes de responder.

Flujo de decisión del agente:
1. Recibe la consulta del trabajador.
2. Analiza el tipo de consulta (clasificación).
3. Decide qué herramienta(s) usar según el contexto.
4. Ejecuta las herramientas necesarias.
5. Genera una respuesta integrada con la información obtenida.
6. Almacena la interacción en memoria de corto plazo.
"""

import logging
from dataclasses import dataclass, field
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from config.settings import (
    GITHUB_TOKEN,
    OPENAI_BASE_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
)
from src.tools.consulta_tool import consultar_documentos
from src.tools.escritura_tool import generar_resumen
from src.tools.razonamiento_tool import analizar_situacion_laboral
from src.memoria import MemoriaConversacional, crear_memoria_corto_plazo

logger = logging.getLogger(__name__)


# Prompt del sistema para el agente con instrucciones de planificación
AGENT_SYSTEM_PROMPT = """Eres un agente inteligente de Recursos Humanos de Comercial Andina SpA.

Tu rol es asistir a los trabajadores con consultas sobre vacaciones, permisos,
licencias médicas, beneficios, horarios y normativas internas.

### HERRAMIENTAS DISPONIBLES:
Tienes acceso a las siguientes herramientas y debes elegir la más adecuada:

1. **consultar_documentos**: Para buscar información en documentos internos y externos.
   Úsala cuando necesites datos específicos sobre políticas, procedimientos o normativas.

2. **generar_resumen**: Para crear resúmenes, correos formales o documentos estructurados.
   Úsala cuando el trabajador pida un documento, resumen o comunicación formal.

3. **analizar_situacion_laboral**: Para analizar casos complejos con múltiples normativas.
   Úsala cuando el trabajador plantee situaciones del tipo "¿qué pasa si...?"

### ESTRATEGIA DE PLANIFICACIÓN:
Antes de responder, planifica tu enfoque:

1. **Identifica el tipo de consulta**:
   - Consulta simple → usa consultar_documentos directamente.
   - Solicitud de documento → usa generar_resumen (que te pedirá consultar primero).
   - Caso complejo → usa analizar_situacion_laboral (que te guiará en el análisis).
   - Saludo o consulta no-RRHH → responde directamente sin herramientas.

2. **Considera el historial**: Revisa la conversación previa para entender el contexto.
   Si el trabajador dice "¿y sobre eso?" o "explica más", usa el historial.

3. **Combina herramientas si es necesario**: Para casos complejos, puedes usar
   varias herramientas en secuencia.

### REGLAS:
- Responde ÚNICAMENTE con información de los documentos. NO inventes datos.
- Si no encuentras información, indícalo y sugiere contactar a RRHH.
- Si la pregunta no es de RRHH, indica amablemente que solo atiendes temas laborales.
- Usa lenguaje formal, claro y preciso.
- Cita las fuentes cuando sea posible.
"""


@dataclass
class RespuestaAgente:
    """Estructura de respuesta del agente."""
    respuesta: str
    herramientas_usadas: List[str] = field(default_factory=list)
    consulta_original: str = ""
    memoria_activa: str = ""


class AgenteRRHH:
    """
    Agente funcional de RRHH con capacidades de consulta, escritura y razonamiento.

    Implementa el patrón ReAct (Reasoning + Acting) mediante LangGraph:
    - El agente razona sobre qué herramienta usar.
    - Ejecuta la herramienta seleccionada.
    - Observa el resultado y decide si necesita más información.
    - Genera la respuesta final integrando toda la información.

    Attributes:
        _llm: Modelo de lenguaje para el agente.
        _tools: Lista de herramientas disponibles.
        _memoria: Memoria de corto plazo (conversacional).
        _agent: Agente ReAct de LangGraph.
    """

    def __init__(self):
        if not GITHUB_TOKEN:
            raise ValueError(
                "No se encontró GITHUB_TOKEN en el archivo .env. "
                "Consulta el README para instrucciones de configuración."
            )

        # Inicializar LLM
        self._llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            api_key=GITHUB_TOKEN,
            base_url=OPENAI_BASE_URL,
        )

        # Registrar herramientas
        self._tools = [
            consultar_documentos,
            generar_resumen,
            analizar_situacion_laboral,
        ]

        # Crear memoria de corto plazo
        self._memoria = crear_memoria_corto_plazo()

        # Construir agente ReAct con LangGraph
        self._agent = create_react_agent(
            model=self._llm,
            tools=self._tools,
        )

        logger.info(
            "AgenteRRHH inicializado (modelo=%s, herramientas=%d, memoria=activa)",
            LLM_MODEL,
            len(self._tools),
        )

    def consultar(self, pregunta: str) -> RespuestaAgente:
        """
        Procesa una consulta del trabajador mediante el agente.

        El agente sigue este flujo:
        1. Recibe la consulta y revisa el historial de conversación.
        2. Planifica qué herramienta(s) necesita.
        3. Ejecuta las herramientas seleccionadas.
        4. Genera una respuesta integrada.
        5. Almacena la interacción en memoria.

        Args:
            pregunta: Consulta del trabajador en lenguaje natural.

        Returns:
            Objeto RespuestaAgente con la respuesta y metadatos.
        """
        logger.info("Nueva consulta al agente: '%s'", pregunta)

        try:
            # Construir mensajes con historial + consulta actual
            mensajes = [SystemMessage(content=AGENT_SYSTEM_PROMPT)]
            mensajes.extend(self._memoria.obtener_mensajes())
            mensajes.append(HumanMessage(content=pregunta))

            # Ejecutar agente
            resultado = self._agent.invoke({"messages": mensajes})

            # Extraer respuesta final
            respuesta_texto = resultado["messages"][-1].content

            # Extraer herramientas usadas
            herramientas = []
            for msg in resultado["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        herramientas.append(tc["name"])

            # Guardar en memoria de corto plazo
            self._memoria.agregar_interaccion(pregunta, respuesta_texto)

            respuesta = RespuestaAgente(
                respuesta=respuesta_texto,
                herramientas_usadas=herramientas,
                consulta_original=pregunta,
                memoria_activa=self._memoria.obtener_resumen(),
            )

            logger.info(
                "Respuesta generada (herramientas: %s)",
                ", ".join(herramientas) if herramientas else "ninguna",
            )

            return respuesta

        except Exception as e:
            logger.error("Error en el agente: %s", e, exc_info=True)
            return RespuestaAgente(
                respuesta=(
                    "Ocurrió un error al procesar tu consulta. "
                    "Por favor, intenta reformularla o contacta al área de RRHH."
                ),
                consulta_original=pregunta,
            )

    def obtener_estado_memoria(self) -> str:
        """Retorna el estado actual de la memoria del agente."""
        return self._memoria.obtener_resumen()

    def limpiar_memoria(self) -> None:
        """Limpia la memoria de corto plazo del agente."""
        self._memoria.limpiar()
        logger.info("Memoria de corto plazo limpiada")
