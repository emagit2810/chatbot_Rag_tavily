"""
rag_indexer2.py → Versión 100% segura para Render Free / Starter (512 MB RAM)
Carga perezosa REAL del modelo de embeddings → no se toca hasta que realmente haces una query
"""

import logging
import os
from pathlib import Path
from typing import Optional, List

# --- LangChain ---
from agent_factory import get_rate_limited_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- LlamaIndex (imports tardíos para no cargar nada pesado al importar) ---
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
    Document,
)
from llama_index.core.schema import NodeWithScore

# NO importamos HuggingFaceEmbedding aquí arriba → lo hacemos dentro de la función

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rutas
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data")).resolve()
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", PROJECT_ROOT / "storage")).resolve()

# Configuración
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "4"))

# Variables globales que se rellenan SOLO cuando se necesiten
_index: Optional[VectorStoreIndex] = None
_retriever = None
_langchain_chain = None
_embed_model_initialized = False  # bandera para no inicializar dos veces


def _lazy_init_embed_model() -> None:
    """Se ejecuta una sola vez y solo cuando realmente vamos a usar el índice"""
    global _embed_model_initialized
    if _embed_model_initialized:
        return

    logger.info("[RAG] Cargando modelo de embeddings por primera vez: %s", EMBED_MODEL_NAME)
    # Import tardío → evita cargar transformers/torch al hacer solo "import rag_indexer2"
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    _embed_model_initialized = True


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def _build_index() -> VectorStoreIndex:
    logger.info("[RAG] Construyendo índice nuevo desde %s", DATA_DIR)
    documents = SimpleDirectoryReader(str(DATA_DIR), recursive=True).load_data()

    if not documents:
        logger.warning("[RAG] Carpeta data/ vacía → índice vacío")
        documents = [Document(text="Índice inicializado sin documentos.")]

    _lazy_init_embed_model()  # ← aquí sí se carga el modelo

    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=str(STORAGE_DIR))
    logger.info("[RAG] Índice creado y guardado en %s", STORAGE_DIR)
    return index


def _load_or_build_index() -> VectorStoreIndex:
    global _index
    if _index is not None:
        return _index

    _ensure_dirs()
    docstore_path = STORAGE_DIR / "docstore.json"

    # Siempre configuramos el embedder antes de cargar o construir
    _lazy_init_embed_model()

    if docstore_path.exists():
        logger.info("[RAG] Cargando índice existente desde %s", STORAGE_DIR)
        storage_context = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR))
        _index = load_index_from_storage(storage_context)
    else:
        _index = _build_index()

    return _index


def get_langchain_chain():
    global _langchain_chain
    if _langchain_chain:
        return _langchain_chain

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Falta GOOGLE_API_KEY")

    llm = get_rate_limited_llm(temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")))

    template = """Eres un asistente útil. Usa SOLO el siguiente contexto para responder.
Si no sabes la respuesta con el contexto, di que no lo sabes.

Contexto:
{context}

Pregunta: {question}

Respuesta:"""

    prompt = ChatPromptTemplate.from_template(template)
    _langchain_chain = prompt | llm | StrOutputParser()
    return _langchain_chain


def ensure_rag_ready() -> VectorStoreIndex:
    """Función pública que garantiza que el índice esté listo"""
    return _load_or_build_index()


def retrieve_context(question: str, top_k: int = DEFAULT_TOP_K) -> str:
    ensure_rag_ready()

    global _retriever
    if _retriever is None:
        _retriever = _index.as_retriever(similarity_top_k=top_k)

    nodes: List[NodeWithScore] = _retriever.retrieve(question)
    if not nodes:
        return "No se encontró contexto relevante."

    texts = [node.text for node in nodes]
    for i, txt in enumerate(texts):
        logger.info(f"[Contexto {i+1}] {txt[:120]}...")

    return "\n\n---\n\n".join(texts)


def get_context(question: str, top_k: int = DEFAULT_TOP_K) -> str:
    return retrieve_context(question, top_k)


def query_rag(question: str, top_k: int = DEFAULT_TOP_K) -> str:
    try:
        context = get_context(question, top_k)
        chain = get_langchain_chain()
        response = chain.invoke({"context": context, "question": question})
        return response
    except Exception as e:
        logger.error("Error en query_rag: %s", e)
        return "Lo siento, ocurrió un error interno."


# Para pruebas locales
if __name__ == "__main__":
    pregunta = "¿Qué documentos tengo cargados?"
    print("Pregunta:", pregunta)
    print("Respuesta:", query_rag(pregunta))