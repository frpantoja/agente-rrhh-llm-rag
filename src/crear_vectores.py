"""
Módulo de creación de base vectorial.

Carga los documentos, los divide en fragmentos con separadores
optimizados para español, genera embeddings y los almacena
en un índice FAISS para búsqueda semántica.
"""

import logging

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from config.settings import (
    GITHUB_TOKEN,
    OPENAI_BASE_URL,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    FAISS_INDEX_DIR,
)
from src.cargar_documentos import cargar_documentos

logger = logging.getLogger(__name__)


def crear_splitter() -> RecursiveCharacterTextSplitter:
    """
    Crea un text splitter optimizado para documentos en español de RRHH.

    Usa separadores jerárquicos que respetan la estructura de los documentos:
    secciones numeradas, párrafos y oraciones.
    """
    separadores = [
        "\n\n",       # Separar por secciones/párrafos dobles
        "\n",          # Separar por líneas
        ". ",          # Separar por oraciones
        ", ",          # Separar por cláusulas
        " ",           # Separar por palabras (último recurso)
    ]

    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=separadores,
        length_function=len,
        is_separator_regex=False,
    )


def crear_embeddings() -> OpenAIEmbeddings:
    """Crea instancia de embeddings con la configuración centralizada."""
    if not GITHUB_TOKEN:
        raise ValueError(
            "No se encontró GITHUB_TOKEN en el archivo .env. "
            "Consulta el README para instrucciones de configuración."
        )

    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=GITHUB_TOKEN,
        base_url=OPENAI_BASE_URL,
    )


def crear_base_vectorial() -> None:
    """
    Pipeline completo de indexación:
    1. Carga documentos con metadatos.
    2. Divide en chunks preservando metadatos.
    3. Genera embeddings y crea índice FAISS.
    4. Guarda el índice en disco.
    """
    logger.info("=== Iniciando creación de base vectorial ===")

    # 1. Cargar documentos
    documentos = cargar_documentos()
    if not documentos:
        logger.error("No se encontraron documentos. Abortando.")
        return

    # 2. Dividir en chunks
    splitter = crear_splitter()
    chunks = splitter.split_documents(documentos)

    logger.info(
        "Chunking completado: %d documentos → %d chunks (size=%d, overlap=%d)",
        len(documentos),
        len(chunks),
        CHUNK_SIZE,
        CHUNK_OVERLAP,
    )

    # Log de distribución de chunks por documento
    from collections import Counter
    distribucion = Counter(c.metadata.get("nombre_archivo", "?") for c in chunks)
    for nombre, cantidad in distribucion.most_common():
        logger.info("  %s: %d chunks", nombre, cantidad)

    # 3. Crear embeddings e indexar
    embeddings = crear_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 4. Guardar
    vectorstore.save_local(FAISS_INDEX_DIR)
    logger.info("Base vectorial guardada en '%s'", FAISS_INDEX_DIR)
    logger.info("=== Indexación completada exitosamente ===")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    crear_base_vectorial()
