"""
Módulo de carga de documentos con extracción de metadatos.

Maneja la lectura de archivos .txt desde las carpetas de datos,
extrayendo metadatos como tipo de documento, título y fuente.
Soporta detección automática de encoding.
"""

import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from config.settings import DATA_DIR

logger = logging.getLogger(__name__)


def detectar_encoding(ruta: Path) -> str:
    """Detecta si el archivo tiene BOM UTF-8 o UTF-16 y retorna el encoding adecuado."""
    with open(ruta, "rb") as f:
        raw = f.read(4)
    if raw[:3] == b"\xef\xbb\xbf":
        return "utf-8-sig"
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return "utf-16"
    return "utf-8"


def clasificar_documento(ruta: Path) -> str:
    """Clasifica un documento como 'interno' o 'externo' según su ubicación."""
    partes = ruta.parts
    if "internos" in partes:
        return "interno"
    elif "externos" in partes:
        return "externo"
    return "general"


def extraer_titulo(contenido: str) -> str:
    """Extrae el título del documento desde la primera línea no vacía."""
    for linea in contenido.split("\n"):
        linea_limpia = linea.strip()
        if linea_limpia:
            return linea_limpia
    return "Sin título"


def cargar_documentos(ruta_base: str = DATA_DIR) -> List[Document]:
    """
    Carga todos los archivos .txt desde la ruta base con metadatos enriquecidos.

    Cada documento incluye metadatos:
    - source: ruta del archivo
    - tipo: 'interno' o 'externo'
    - titulo: primera línea del documento
    - nombre_archivo: nombre del archivo sin extensión

    Args:
        ruta_base: Directorio raíz donde buscar documentos .txt

    Returns:
        Lista de objetos Document de LangChain con contenido y metadatos.
    """
    ruta = Path(ruta_base)
    documentos: List[Document] = []

    if not ruta.exists():
        logger.error("La carpeta no existe: %s", ruta_base)
        return documentos

    archivos = list(ruta.rglob("*.txt"))
    logger.info("Archivos encontrados: %d en %s", len(archivos), ruta_base)

    for archivo in sorted(archivos):
        try:
            encoding = detectar_encoding(archivo)
            contenido = archivo.read_text(encoding=encoding).strip()

            if not contenido:
                logger.warning("Archivo vacío, omitido: %s", archivo.name)
                continue

            metadata = {
                "source": str(archivo),
                "tipo": clasificar_documento(archivo),
                "titulo": extraer_titulo(contenido),
                "nombre_archivo": archivo.stem,
            }

            doc = Document(page_content=contenido, metadata=metadata)
            documentos.append(doc)
            logger.info(
                "Cargado: %s (tipo=%s, %d caracteres)",
                archivo.name,
                metadata["tipo"],
                len(contenido),
            )

        except Exception as e:
            logger.error("Error al leer %s: %s", archivo.name, e)

    logger.info(
        "Total documentos cargados: %d (internos: %d, externos: %d)",
        len(documentos),
        sum(1 for d in documentos if d.metadata["tipo"] == "interno"),
        sum(1 for d in documentos if d.metadata["tipo"] == "externo"),
    )

    return documentos


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    docs = cargar_documentos()
    for doc in docs:
        print(f"  [{doc.metadata['tipo']}] {doc.metadata['titulo']}")
