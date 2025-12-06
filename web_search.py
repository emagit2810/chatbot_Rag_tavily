'''
Módulo simple para búsqueda web con Tavily + Gemini.
'''

import logging
import os
import re
from typing import List

from tavily import TavilyClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from agent_factory import get_rate_limited_llm

logger = logging.getLogger(__name__)

_MAX_TAVILY_RESULTS = 3
_MAX_CHUNKS = 3
_SPLIT_PATTERN = re.compile(r'(?<=[.!?])\s+')
_tavily_client = None


def get_tavily_client() -> TavilyClient:
    '''Obtiene o crea el cliente Tavily.'''
    global _tavily_client
    if _tavily_client is None:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY no encontrada en variables de entorno.")
        _tavily_client = TavilyClient(api_key=api_key)
        logger.info("[WEB] Cliente Tavily inicializado.")
    return _tavily_client


def _truncate_sentence(sentence: str, max_chars: int = 220) -> str:
    sentence = sentence.strip()
    if len(sentence) <= max_chars:
        return sentence
    return sentence[:max_chars].rstrip() + "…"


def _build_chunk(index: int, title: str, url: str, content: str) -> str:
    lines: List[str] = []
    title_text = title.strip() or "Sin título"
    source_text = url.strip() or "Fuente desconocida"
    lines.append(f"[{index}] {title_text}")
    lines.append(f"Fuente: {source_text}")

    sentences = [
        _truncate_sentence(s)
        for s in _SPLIT_PATTERN.split(content or "")
        if s.strip()
    ]

    for sentence in sentences[:3]:
        lines.append(sentence)

    if len(lines) < 4:
        lines.append("Información limitada disponible.")

    return "\n".join(lines[:7])


def search_web(query: str, max_results: int = _MAX_TAVILY_RESULTS) -> str:
    '''Busca en la web utilizando Tavily y devuelve hasta 3 chunks resumidos.'''
    try:
        trimmed_query = query.strip()
        if not trimmed_query:
            return "Consulta vacía."

        client = get_tavily_client()
        logger.info(f"[WEB] Buscando en Tavily: {trimmed_query[:80]}")

        response = client.search(
            query=trimmed_query,
            max_results=min(max_results, _MAX_TAVILY_RESULTS),
            search_depth="basic",
        )

        results = response.get("results", [])
        if not results:
            logger.info("[WEB] No se encontraron resultados.")
            return "No se encontraron resultados en la web."

        chunks: List[str] = []
        for idx, result in enumerate(results, 1):
            if len(chunks) >= _MAX_CHUNKS:
                break
            title = result.get("title", "")
            url = result.get("url", "")
            content = result.get("content", "")
            chunk = _build_chunk(idx, title, url, content)
            chunks.append(chunk)

        logger.info(f"[WEB] Procesados {len(chunks)} chunks para el contexto.")
        return "\n\n---\n\n".join(chunks)

    except Exception as exc:
        logger.warning(f"[WEB] Falló la búsqueda en Tavily: {exc}")
        return f"Error al buscar en la web: {str(exc)}"


def answer_web_search(question: str, max_results: int = _MAX_TAVILY_RESULTS) -> dict:
    '''Responde usando una búsqueda web y Gemini.'''
    context = search_web(question, max_results=max_results)
    try:
        template = '''Eres un asistente útil. Usa SOLO la siguiente información de la web para responder la pregunta.

Sé breve y directo. Si la información no es suficiente, di que no encontraste información relevante.

Información de la web:
{context}

Pregunta: {question}

Respuesta concisa:'''

        prompt = ChatPromptTemplate.from_template(template)
        llm = get_rate_limited_llm(temperature=0.3)
        chain = prompt | llm | StrOutputParser()

        logger.info("[WEB] Generando respuesta con Gemini (web search).")
        response = chain.invoke(
            {"context": context, "question": question}
        )

        return {"answer": response, "context": context}

    except Exception as exc:
        logger.warning(f"[WEB] Error en el flujo web search: {exc}")
        fallback_context = context or "No hay contexto disponible."
        return {
            "answer": "No pude completar la búsqueda web en este momento.",
            "context": fallback_context,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pregunta = "¿Qué pasa hoy en el mundo?"
    resultado = answer_web_search(pregunta, max_results=2)
    print(f"Respuesta: {resultado['answer']}")
    print("Contexto:")
    print(resultado["context"])
