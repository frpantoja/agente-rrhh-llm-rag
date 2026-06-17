"""
Agente funcional de RRHH con herramientas, memoria y planificación.

Este módulo implementa el agente central del sistema, que integra:
- Herramientas de consulta, escritura y razonamiento
- Memoria conversacional con tres estrategias (buffer, window, summary)
- Recuperación de contexto semántico via RAG
- Planificación y toma de decisiones adaptativa

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
import time
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
    MEMORY_TYPE,
    MEMORY_WINDOW_SIZE,
)
from src.tools.consulta_tool import consultar_documentos
from src.tools.escritura_tool import generar_resumen
from src.tools.razonamiento_tool import analizar_situacion_laboral
from src.memoria import crear_memoria
from src.observabilidad.trazas import nuevo_trace_id, trace_event, hash_consulta
from src.observabilidad.metricas import AgentMetrics, ConsistenciaAnalyzer, AnomalyDetector

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

### ESTRATEGIA DE PLANIFICACIÓN (orden de evaluación con prioridades):
Sigue este orden estricto de evaluación antes de actuar:

PASO 1 - Filtro de alcance: ¿La consulta es sobre RRHH (vacaciones, permisos,
licencias, beneficios, horarios, normativas)? Si NO, responde directamente sin
usar herramientas y termina aquí.

PASO 2 - Clasificación por prioridad (evalúa en este orden, la primera que
calce gana):
   a) ¿Requiere un documento, resumen o correo formal? → usa generar_resumen
      (tiene prioridad sobre una consulta simple, ya que generar_resumen
      internamente te pedirá consultar documentos primero).
   b) ¿Es un caso con múltiples condiciones o requiere cruzar normativa interna
      y externa (preguntas tipo "¿qué pasa si...?")? → usa
      analizar_situacion_laboral (puede requerir llamar consultar_documentos
      más de una vez, una por cada fuente normativa).
   c) Si no calza con (a) ni (b) → consulta simple, usa consultar_documentos
      directamente.

PASO 3 - Resolución de contexto: Revisa si la consulta depende del historial.
Si el trabajador dice "¿y sobre eso?", "explica más" o "¿y cómo las solicito?"
(una pregunta corta que depende del tema previo), NO pases esa frase corta
directamente a consultar_documentos. Reformula la consulta incorporando el
tema explícito del historial antes de llamar a la herramienta.

Ejemplo: si se habló de vacaciones y el trabajador pregunta "¿cómo las
solicito?", llama a consultar_documentos con la consulta reformulada
"cómo solicitar vacaciones", no con el texto literal "¿cómo las solicito?".

PASO 4 - Secuenciación: Si una herramienta requiere el resultado de otra
(ej: generar_resumen necesita primero el resultado de consultar_documentos),
ejecútalas en ese orden y combina los resultados en la respuesta final.

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
    trace_id: str = ""
    duracion_ms: float = 0.0


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

        # Crear memoria segun configuracion
        if MEMORY_TYPE == "summary":
            self._memoria = crear_memoria("summary", llm=self._llm)
        elif MEMORY_TYPE == "buffer":
            self._memoria = crear_memoria("buffer")
        else:
            self._memoria = crear_memoria("window", window_size=MEMORY_WINDOW_SIZE)

        # Construir agente ReAct con LangGraph
        self._agent = create_react_agent(
            model=self._llm,
            tools=self._tools,
        )

        # Componentes de observabilidad
        self._metrics = AgentMetrics()
        self._consistencia = ConsistenciaAnalyzer()
        self._anomaly_detector = AnomalyDetector()

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

        Cada consulta genera un trace_id único que conecta los eventos
        registrados (retrieval, llamadas a herramientas, generación),
        y se acumulan métricas de latencia, éxito y uso de herramientas
        para fines de observabilidad.

        Args:
            pregunta: Consulta del trabajador en lenguaje natural.

        Returns:
            Objeto RespuestaAgente con la respuesta, metadatos y trace_id.
        """
        trace_id = nuevo_trace_id()
        inicio = time.time()

        trace_event(
            trace_id, "consulta_completa", "start",
            metadata={"consulta_hash": hash_consulta(pregunta)},
        )
        logger.info("Nueva consulta al agente [trace_id=%s]", trace_id)

        try:
            # Construir mensajes con historial + consulta actual
            mensajes = [SystemMessage(content=AGENT_SYSTEM_PROMPT)]
            mensajes.extend(self._memoria.obtener_mensajes())
            mensajes.append(HumanMessage(content=pregunta))

            # Ejecutar agente (span de invocación completa del LLM/tools)
            t_agente = time.time()
            resultado = self._agent.invoke({"messages": mensajes})
            duracion_agente_ms = (time.time() - t_agente) * 1000
            trace_event(
                trace_id, "agent_invoke", "end",
                duration_ms=duracion_agente_ms,
                metadata={"status": "ok"},
            )

            # Extraer respuesta final
            respuesta_texto = resultado["messages"][-1].content

            # Extraer herramientas usadas
            herramientas = []
            for msg in resultado["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        herramientas.append(tc["name"])
                        trace_event(
                            trace_id, f"tool:{tc['name']}", "end",
                            metadata={"status": "ok"},
                        )

            # Guardar en memoria de corto plazo
            self._memoria.agregar_interaccion(pregunta, respuesta_texto)

            duracion_total_ms = (time.time() - inicio) * 1000

            # Registrar métricas de la interacción
            self._metrics.registrar_interaccion(
                duracion_ms=duracion_total_ms,
                exito=True,
                herramientas=herramientas,
            )
            categoria = herramientas[0] if herramientas else "sin_herramienta"
            self._consistencia.registrar(categoria, len(respuesta_texto))
            anomalia = self._anomaly_detector.procesar(duracion_total_ms, trace_id)

            trace_event(
                trace_id, "consulta_completa", "end",
                duration_ms=duracion_total_ms,
                metadata={
                    "status": "ok",
                    "herramientas": herramientas,
                    "es_anomalia": anomalia["es_anomalia"],
                    "z_score": anomalia["z_score"],
                },
            )

            respuesta = RespuestaAgente(
                respuesta=respuesta_texto,
                herramientas_usadas=herramientas,
                consulta_original=pregunta,
                memoria_activa=self._memoria.obtener_resumen(),
                trace_id=trace_id,
                duracion_ms=round(duracion_total_ms, 2),
            )

            logger.info(
                "Respuesta generada [trace_id=%s] (herramientas: %s, %.0fms)",
                trace_id,
                ", ".join(herramientas) if herramientas else "ninguna",
                duracion_total_ms,
            )

            return respuesta

        except Exception as e:
            duracion_total_ms = (time.time() - inicio) * 1000
            self._metrics.registrar_interaccion(
                duracion_ms=duracion_total_ms, exito=False,
            )
            trace_event(
                trace_id, "consulta_completa", "error",
                duration_ms=duracion_total_ms,
                metadata={"status": "failed", "error_type": type(e).__name__},
            )
            logger.error(
                "Error en el agente [trace_id=%s]: %s", trace_id, e, exc_info=True
            )
            return RespuestaAgente(
                respuesta=(
                    "Ocurrió un error al procesar tu consulta. "
                    "Por favor, intenta reformularla o contacta al área de RRHH."
                ),
                consulta_original=pregunta,
                trace_id=trace_id,
                duracion_ms=round(duracion_total_ms, 2),
            )

    def obtener_metricas(self) -> dict:
        """Retorna el resumen de métricas acumuladas de la sesión."""
        return self._metrics.resumen()

    def obtener_consistencia(self) -> dict:
        """Retorna el análisis de consistencia por tipo de consulta."""
        return self._consistencia.consistencia_por_categoria()

    def obtener_estado_memoria(self) -> str:
        """Retorna el estado actual de la memoria del agente."""
        return self._memoria.obtener_resumen()

    def limpiar_memoria(self) -> None:
        """Limpia la memoria de corto plazo del agente."""
        self._memoria.limpiar()
        logger.info("Memoria de corto plazo limpiada")
