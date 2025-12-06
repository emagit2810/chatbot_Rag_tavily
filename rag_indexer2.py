"""
RAG Robusto:
- LlamaIndex: Indexación y Retrieval (Búsqueda).
- LangChain: Orquestación y Generación con Gemini.
- Evita el conflicto 'langchain.base_language' al no usar el wrapper LangChainLLM.
"""

import logging
import os
from pathlib import Path
from typing import Optional, List

# --- Imports de LangChain ---
from agent_factory import get_rate_limited_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Imports de LlamaIndex ---
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Configuración de Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de Rutas
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data")).resolve()
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", PROJECT_ROOT / "storage")).resolve()

# Configuración de Modelos
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))

_index: Optional[VectorStoreIndex] = None
_retriever = None
_langchain_chain = None
_embed_model: Optional[HuggingFaceEmbedding] = None

def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

def _configure_components() -> None:
    """
    Configura el Embedder de LlamaIndex y el LLM de LangChain por separado.
    """
    # 1. Configurar Embeddings en LlamaIndex (Solo para buscar).
    #    OJO: No leemos Settings.embed_model para evitar que LlamaIndex
    #    intente autoconfigurar OpenAI por defecto.
    logger.info("[RAG] Forzando embedder local: %s", EMBED_MODEL_NAME)
    Settings.embed_model = _get_embed_model()
    # IMPORTANTE: Dejamos Settings.llm en None o por defecto, 
    # porque NO usaremos LlamaIndex para generar la respuesta, solo para buscar.
    Settings.llm = None


def _get_embed_model() -> HuggingFaceEmbedding:
    """
    Devuelve y cachea una instancia de HuggingFaceEmbedding compatible con LlamaIndex.
    """
    global _embed_model
    if _embed_model is None:
        _embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    return _embed_model

def _build_index() -> VectorStoreIndex:
    logger.info("[RAG] Cargando documentos desde %s", DATA_DIR)
    documents = SimpleDirectoryReader(str(DATA_DIR), recursive=True).load_data()

    if not documents:
        logger.warning("[RAG] data/ vacío; el índice se creará vacío.")
        # Crear un documento dummy para evitar error si está vacío
        from llama_index.core import Document
        documents = [Document(text="Índice inicializado sin documentos.")]

    _configure_components()
    
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=str(STORAGE_DIR))
    logger.info("[RAG] Índice creado y persistido en %s", STORAGE_DIR)
    return index

def _load_or_build_index() -> VectorStoreIndex:
    global _index
    if _index is not None:
        return _index

    _ensure_dirs()
    docstore_path = STORAGE_DIR / "docstore.json"

    _configure_components()

    if docstore_path.exists():
        logger.info("[RAG] Cargando índice desde %s", STORAGE_DIR)
        storage_context = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR))
        _index = load_index_from_storage(storage_context)
    else:
        _index = _build_index()

    return _index

def get_langchain_chain():
    """Configura la cadena de procesamiento de LangChain (LCEL)"""
    global _langchain_chain
    if _langchain_chain:
        return _langchain_chain
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Falta GOOGLE_API_KEY")

    temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    llm = get_rate_limited_llm(temperature=temperature)

    # Definir el Prompt
    template = """Eres un asistente útil. Usa SOLO el siguiente contexto para responder la pregunta. 
    Si no sabes la respuesta basándote en el contexto, di que no lo sabes.
    
    Contexto:
    {context}
    
    Pregunta: {question}
    
    Respuesta:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Crear la cadena: Prompt -> LLM -> String Output
    _langchain_chain = prompt | llm | StrOutputParser()
    return _langchain_chain

def retrieve_context(question: str, top_k: int = DEFAULT_TOP_K) -> str:
    """Usa LlamaIndex SOLO para recuperar texto."""
    global _retriever
    ensure_rag_ready()
    
    if _retriever is None and _index is not None:
        _retriever = _index.as_retriever(similarity_top_k=top_k)

    nodes: List[NodeWithScore] = _retriever.retrieve(question)
    
    if not nodes:
        return "No se encontró contexto relevante."

    # Extraer texto y loggear para depuración
    texts = [n.text for n in nodes]
    for i, txt in enumerate(texts):
        logger.info(f"[Contexto {i+1}] {txt[:100]}...")
    
    return "\n\n---\n\n".join(texts)

def query_rag(question: str, top_k: int = DEFAULT_TOP_K) -> str:
    """
    Función principal:
    1. Obtiene contexto con LlamaIndex.
    2. Genera respuesta con LangChain + Gemini.
    """
    try:
        # 1. Recuperación (LlamaIndex)
        context_text = get_context(question, top_k)
        
        # 2. Generación (LangChain)
        logger.info("[RAG] Generando respuesta para '%s' usando %s chunks de contexto.", question[:60].replace("\n", " "), top_k)
        chain = get_langchain_chain()
        response = chain.invoke({
            "context": context_text,
            "question": question
        })

        logger.info("[RAG] Generación completada para la pregunta.")
        return response
        
    except Exception as e:
        logger.warning("RAG: se produjo una excepción durante el flujo: %s", e)
        return "Lo siento, ocurrió un error al procesar tu solicitud."


def ensure_rag_ready() -> VectorStoreIndex:
    """Garanticé que el índice esté cargado/construido con el embedder correcto."""
    return _load_or_build_index()


def get_context(question: str, top_k: int = DEFAULT_TOP_K) -> str:
    """Devuelve el contexto recuperado para que otros módulos puedan usarlo fácilmente."""
    return retrieve_context(question, top_k)


# Bloque para probar directamente si ejecutas el script
if __name__ == "__main__":
    # Asegúrate de setear tu API KEY aquí o en variables de entorno
    # os.environ["GOOGLE_API_KEY"] = "TU_API_KEY"
    
    pregunta = "Que hay en mis documentos?" 
    print(f"Pregunta: {pregunta}")
    respuesta = query_rag(pregunta)
    print(f"Respuesta:\n{respuesta}")
