# -*- coding: utf-8 -*-
import logging
import os
from langchain_core.tools import tool
from tavily import TavilyClient

import rag_indexer2

logger = logging.getLogger("deep_agent.tools")

# Lazy singleton client para evitar reinicializar la conexión constantemente
_tavily_client = None
_tavily_last_ts = 0.0
_tavily_min_delay = 8.0  # ~7.5 rpm máx


def get_tavily_client():
    global _tavily_client
    if not _tavily_client:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("ERROR: TAVILY_API_KEY no encontrada en variables de entorno.")
        try:
            _tavily_client = TavilyClient(api_key=api_key)
            logger.info("Cliente Tavily inicializado correctamente.")
        except Exception:
            logger.exception("Fallo al inicializar TavilyClient.")
            raise
    return _tavily_client


@tool
def internet_search(query: str, max_results: int = 5) -> str:
    """Busca información actual en la web usando Tavily."""
    global _tavily_last_ts

    logger.info("Iniciando búsqueda web para: %s", query)
    print(f"[WEB] Buscando en internet: '{query}'")

    client = get_tavily_client()
    try:
        # Rate limit simple para no exceder ~7 rpm
        import time

        now = time.time()
        elapsed = now - _tavily_last_ts
        if elapsed < _tavily_min_delay:
            wait_time = _tavily_min_delay - elapsed
            wait_msg = f"Esperando {wait_time:.2f}s por límite de tasa..."
            logger.info("Rate limit Tavily: %s", wait_msg)
            print(f"[WEB] {wait_msg}")
            time.sleep(wait_time)

        _tavily_last_ts = time.time()
        logger.info("Llamando a Tavily API: query='%s', max_results=%s", query, max_results)
        print(f"[WEB] Realizando búsqueda (máx. {max_results} resultados)...")

        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",
        )

        results = response.get("results", [])
        logger.info("Tavily devolvió %s resultados", len(results))

        if not results:
            no_results_msg = "No se encontraron resultados relevantes para esta búsqueda."
            logger.warning(no_results_msg)
            print("[WEB] ¡Advertencia! No se encontraron resultados.")
            return no_results_msg

        logger.debug("Resultados obtenidos: %s", results)
        print(f"[WEB] Encontrados {len(results)} resultados")

        context_parts = []
        for i, res in enumerate(results, 1):
            content = res.get("content", "Sin contenido").strip()
            url = res.get("url", "Sin URL")
            title = res.get("title", "Sin título")
            entry = (
                f"--- Resultado {i} ---\n"
                f"Título: {title}\n"
                f"Fuente: {url}\n"
                f"Contenido: {content}\n"
            )
            context_parts.append(entry)

        return "\n".join(context_parts)

    except Exception as e:
        logger.exception("Error al conectar con Tavily para la query '%s'", query)
        return f"Error al conectar con el motor de búsqueda: {str(e)}"


@tool
def rag_search(query: str) -> str:
    """Busca en documentos locales usando el índice RAG interno."""
    try:
        logger.info("Iniciando búsqueda RAG para consulta: %s", query)
        print(f"[RAG] Buscando en documentos locales: '{query}'")

        rag_indexer2.ensure_rag_ready()

        logger.debug("Ejecutando consulta en el índice RAG")
        response = rag_indexer2.query_rag(query)
        logger.info("Búsqueda RAG completada exitosamente")
        print("[RAG] Búsqueda completada")
        preview = str(response)
        print(f"[RAG] Preview 160 chars: {preview[:160].replace(chr(10), ' ')}...")
        print(f"[RAG] Respuesta total (chars): {len(preview)}")

        return response
    except Exception as e:
        logger.exception("Error en rag_search para la query '%s'", query)
        return f"Error al consultar el índice RAG: {str(e)}"
