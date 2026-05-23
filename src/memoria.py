"""
Módulo de memoria del agente (corto y largo plazo).

Implementa dos tipos de memoria para el agente:

1. Memoria de corto plazo (historial de mensajes):
   - Mantiene el historial de la conversación actual.
   - Ventana deslizante de las últimas K interacciones.
   - Permite al agente hacer seguimiento de preguntas anteriores.

2. Memoria de largo plazo (RAG / FAISS):
   - Base vectorial con los documentos de RRHH.
   - Recuperación semántica de contexto relevante.
   - Persistente entre sesiones (guardada en disco).

La separación en dos tipos de memoria permite:
- Continuidad en flujos prolongados mediante historial conversacional.
- Recuperación de contexto semántico desde la base documental.
"""

import logging
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger(__name__)

# Cantidad de interacciones a mantener en memoria de corto plazo
MEMORY_WINDOW_SIZE = 5


class MemoriaConversacional:
    """
    Memoria de corto plazo con ventana deslizante.

    Almacena las últimas K interacciones (pregunta + respuesta) para
    mantener continuidad conversacional. Esto permite:
    - Resolución de referencias: entiende "¿y sobre eso?" o "explica más".
    - Contexto acumulado: puede combinar info de varias preguntas.

    La ventana evita que el contexto crezca sin límite.
    """

    def __init__(self, window_size: int = MEMORY_WINDOW_SIZE):
        self._history = InMemoryChatMessageHistory()
        self._window_size = window_size
        logger.info("Memoria conversacional creada (ventana=%d)", window_size)

    def agregar_interaccion(self, pregunta: str, respuesta: str) -> None:
        """Agrega una interacción al historial."""
        self._history.add_message(HumanMessage(content=pregunta))
        self._history.add_message(AIMessage(content=respuesta))
        self._recortar_ventana()

    def _recortar_ventana(self) -> None:
        """Mantiene solo las últimas K interacciones (2K mensajes)."""
        max_mensajes = self._window_size * 2
        mensajes = self._history.messages
        if len(mensajes) > max_mensajes:
            self._history.messages = mensajes[-max_mensajes:]

    def obtener_mensajes(self) -> list:
        """Retorna los mensajes del historial."""
        return self._history.messages

    def obtener_resumen(self) -> str:
        """Retorna un resumen del estado de la memoria."""
        mensajes = self._history.messages
        if not mensajes:
            return "La memoria está vacía. No hay conversaciones previas."

        return (
            f"Memoria activa: {len(mensajes)} mensajes almacenados.\n"
            f"Ventana: últimas {self._window_size} interacciones."
        )

    def limpiar(self) -> None:
        """Limpia toda la memoria."""
        self._history.clear()
        logger.info("Memoria conversacional limpiada")


def crear_memoria_corto_plazo() -> MemoriaConversacional:
    """Crea una instancia de memoria de corto plazo."""
    return MemoriaConversacional(window_size=MEMORY_WINDOW_SIZE)


def obtener_resumen_memoria(memoria: MemoriaConversacional) -> str:
    """Obtiene un resumen legible del estado actual de la memoria."""
    return memoria.obtener_resumen()
