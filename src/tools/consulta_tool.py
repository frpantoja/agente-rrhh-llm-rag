"""
Herramienta de consulta documental (Tool de consulta).

Permite al agente buscar información en los documentos internos y externos
de RRHH mediante búsqueda semántica en la base vectorial FAISS.

Forma parte del conjunto de herramientas disponibles
"""

import logging
from typing import Optional

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool

from config.settings import (
    GITHUB_TOKEN,
    OPENAI_BASE_URL,
    EMBEDDING_MODEL,
    FAISS_INDEX_DIR,
    RETRIEVAL_K,
    RETRIEVAL_FINAL_K,
    SIMILARITY_THRESHOLD,
)

logger = logging.getLogger(__name__)

# Instancia global del vectorstore (se carga una sola vez)
_vectorstore: Optional[FAISS] = None


def _get_vectorstore() -> FAISS:
    """Carga el vectorstore con lazy loading (singleton)."""
    global _vectorstore
    if _vectorstore is None:
        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=GITHUB_TOKEN,
            base_url=OPENAI_BASE_URL,
        )
        _vectorstore = FAISS.load_local(
            FAISS_INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("Base vectorial cargada desde '%s'", FAISS_INDEX_DIR)
    return _vectorstore


@tool
def consultar_documentos(consulta: str) -> str:
    """
    Busca información relevante en los documentos internos y externos de RRHH.

    Usa esta herramienta cuando el trabajador pregunte sobre:
    - Vacaciones, feriados legales o días de descanso
    - Permisos administrativos
    - Licencias médicas
    - Beneficios de la empresa
    - Horarios y jornada laboral
    - Reglamento interno o normativas
    - Código del Trabajo o normativa laboral

    Args:
        consulta: La pregunta o tema a buscar en los documentos.

    Returns:
        Fragmentos relevantes de los documentos con sus fuentes.
    """
    vectorstore = _get_vectorstore()

    docs_con_score = vectorstore.similarity_search_with_relevance_scores(
        consulta, k=RETRIEVAL_K
    )

    # Filtrar por umbral de similitud
    docs_relevantes = [
        (doc, score) for doc, score in docs_con_score
        if score >= SIMILARITY_THRESHOLD
    ][:RETRIEVAL_FINAL_K]

    if not docs_relevantes:
        return (
            "No se encontraron documentos relevantes para esta consulta. "
            "La información solicitada no está disponible en la base documental."
        )

    resultado = []
    for i, (doc, score) in enumerate(docs_relevantes, 1):
        titulo = doc.metadata.get("titulo", "Sin título")
        tipo = doc.metadata.get("tipo", "general")
        resultado.append(
            f"[Fragmento {i}] (Fuente: {titulo} | Tipo: {tipo} | "
            f"Relevancia: {score:.2f})\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(resultado)
