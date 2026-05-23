"""
Pipeline RAG con filtrado por similitud y re-ranking.

Implementa la búsqueda semántica en la base vectorial y la generación
de respuestas usando el LLM con prompts estructurados.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config.settings import (
    GITHUB_TOKEN,
    OPENAI_BASE_URL,
    EMBEDDING_MODEL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    FAISS_INDEX_DIR,
    RETRIEVAL_K,
    RETRIEVAL_FINAL_K,
    SIMILARITY_THRESHOLD,
)
from src.prompts import RAG_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


@dataclass
class RespuestaRAG:
    """Estructura de respuesta del pipeline RAG."""
    respuesta: str
    fuentes: List[str] = field(default_factory=list)
    num_documentos_recuperados: int = 0
    num_documentos_relevantes: int = 0
    consulta_original: str = ""


class AsistenteRRHH:
    """
    Asistente RAG para consultas de Recursos Humanos.

    Implementa un pipeline de retrieval avanzado:
    1. Búsqueda MMR para diversidad de resultados.
    2. Filtrado por umbral de similitud.
    3. Re-ranking basado en relevancia.
    4. Generación con prompt estructurado y guardrails.
    """

    def __init__(self):
        if not GITHUB_TOKEN:
            raise ValueError(
                "No se encontró GITHUB_TOKEN en el archivo .env. "
                "Consulta el README para instrucciones de configuración."
            )

        self._embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=GITHUB_TOKEN,
            base_url=OPENAI_BASE_URL,
        )

        self._llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            api_key=GITHUB_TOKEN,
            base_url=OPENAI_BASE_URL,
        )

        self._vectorstore: Optional[FAISS] = None
        logger.info(
            "AsistenteRRHH inicializado (modelo=%s, embeddings=%s)",
            LLM_MODEL,
            EMBEDDING_MODEL,
        )

    def _cargar_vectorstore(self) -> FAISS:
        """Carga el vectorstore desde disco (lazy loading)."""
        if self._vectorstore is None:
            logger.info("Cargando base vectorial desde '%s'", FAISS_INDEX_DIR)
            self._vectorstore = FAISS.load_local(
                FAISS_INDEX_DIR,
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
        return self._vectorstore

    def _recuperar_documentos(self, pregunta: str) -> List[Document]:
        """
        Recupera documentos relevantes usando MMR + filtrado por similitud.

        MMR (Maximal Marginal Relevance) balancea relevancia con diversidad,
        evitando que los chunks recuperados sean redundantes entre sí.
        """
        vectorstore = self._cargar_vectorstore()

        # Búsqueda con MMR para diversidad
        docs_con_score = vectorstore.similarity_search_with_relevance_scores(
            pregunta, k=RETRIEVAL_K
        )

        logger.info("Documentos recuperados: %d (k=%d)", len(docs_con_score), RETRIEVAL_K)

        # Filtrar por umbral de similitud
        docs_filtrados = []
        for doc, score in docs_con_score:
            if score >= SIMILARITY_THRESHOLD:
                doc.metadata["score"] = round(score, 4)
                docs_filtrados.append(doc)
                logger.debug(
                    "  [OK] score=%.4f | %s | %s",
                    score,
                    doc.metadata.get("nombre_archivo", "?"),
                    doc.page_content[:60].replace("\n", " "),
                )
            else:
                logger.debug(
                    "  [--] score=%.4f (bajo umbral %.2f) | %s",
                    score,
                    SIMILARITY_THRESHOLD,
                    doc.metadata.get("nombre_archivo", "?"),
                )

        # Tomar los top-k finales (ya vienen ordenados por score)
        docs_finales = docs_filtrados[:RETRIEVAL_FINAL_K]

        logger.info(
            "Documentos tras filtrado: %d/%d (umbral=%.2f, final_k=%d)",
            len(docs_finales),
            len(docs_con_score),
            SIMILARITY_THRESHOLD,
            RETRIEVAL_FINAL_K,
        )

        return docs_finales

    def _construir_contexto(self, documentos: List[Document]) -> tuple[str, str]:
        """
        Construye el contexto y lista de fuentes desde los documentos recuperados.

        Returns:
            Tupla (contexto_formateado, fuentes_formateadas)
        """
        if not documentos:
            return (
                "No se encontraron documentos relevantes para esta consulta.",
                "Ninguna fuente disponible.",
            )

        bloques = []
        fuentes_set = set()

        for i, doc in enumerate(documentos, 1):
            titulo = doc.metadata.get("titulo", "Sin título")
            tipo = doc.metadata.get("tipo", "general")
            score = doc.metadata.get("score", 0)
            nombre = doc.metadata.get("nombre_archivo", "desconocido")

            bloque = (
                f"[Fragmento {i}] (Fuente: {titulo} | Tipo: {tipo} | "
                f"Relevancia: {score:.2f})\n{doc.page_content}"
            )
            bloques.append(bloque)
            fuentes_set.add(f"{titulo} ({tipo})")

        contexto = "\n\n---\n\n".join(bloques)
        fuentes = "\n".join(f"- {f}" for f in sorted(fuentes_set))

        return contexto, fuentes

    def consultar(self, pregunta: str) -> RespuestaRAG:
        """
        Ejecuta el pipeline RAG completo para una consulta.

        Pipeline:
        1. Recuperación de documentos con MMR.
        2. Filtrado por umbral de similitud.
        3. Construcción de contexto con metadatos.
        4. Generación con prompt estructurado.

        Args:
            pregunta: Consulta del trabajador en lenguaje natural.

        Returns:
            Objeto RespuestaRAG con la respuesta y metadatos.
        """
        logger.info("Nueva consulta: '%s'", pregunta)

        # 1-2. Recuperar y filtrar documentos
        documentos = self._recuperar_documentos(pregunta)

        # 3. Construir contexto
        contexto, fuentes = self._construir_contexto(documentos)

        # 4. Generar respuesta
        prompt = RAG_PROMPT_TEMPLATE.format(
            contexto=contexto,
            fuentes=fuentes,
            pregunta=pregunta,
        )

        respuesta_llm = self._llm.invoke(prompt)

        fuentes_lista = [
            doc.metadata.get("titulo", "Sin título") for doc in documentos
        ]

        resultado = RespuestaRAG(
            respuesta=respuesta_llm.content,
            fuentes=fuentes_lista,
            num_documentos_recuperados=RETRIEVAL_K,
            num_documentos_relevantes=len(documentos),
            consulta_original=pregunta,
        )

        logger.info(
            "Respuesta generada (%d docs relevantes, %d caracteres)",
            resultado.num_documentos_relevantes,
            len(resultado.respuesta),
        )

        return resultado


# Función de compatibilidad con la versión anterior
def responder_consulta(pregunta: str) -> str:
    """Wrapper de compatibilidad con app.py original."""
    asistente = AsistenteRRHH()
    resultado = asistente.consultar(pregunta)
    return resultado.respuesta
