"""
Configuración centralizada del proyecto.
Todas las constantes y parámetros configurables están aquí.
"""

import os
from dotenv import load_dotenv

load_dotenv()


# --- Credenciales ---
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://models.inference.ai.azure.com")

# --- Modelos ---
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0

# --- RAG: Chunking ---
CHUNK_SIZE = 300
CHUNK_OVERLAP = 80

# --- RAG: Retrieval ---
RETRIEVAL_K = 5                 # Documentos candidatos a recuperar
RETRIEVAL_FINAL_K = 3           # Documentos finales tras re-ranking
SIMILARITY_THRESHOLD = 0.3      # Umbral mínimo de similitud (0 a 1)

# --- Rutas ---
DATA_DIR = "data"
FAISS_INDEX_DIR = "faiss_index"

# --- Logging ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
