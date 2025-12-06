#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backend minimalista con FastAPI:
- Solo dos modos: llm_only y rag_search.
- RAG solo devuelve contexto (llama_index), el LLM se ejecuta en LangChain.
- Sin dependencias de DeepAgents ni agent_factory.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from agent_factory import quick_llm_answer
from pydantic import BaseModel, Field, ValidationError
import uvicorn

import rag_indexer2 as rag_indexer
import web_search

# --------------------------------------------------------------------------- #
# Configuración y constantes
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("deep_agent")

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

INDEX_PATH = os.path.join(_ROOT, "index.html")
IMAGES_DIR = os.path.join(_ROOT, "images")

MODE_REQUIREMENTS: Dict[str, List[str]] = {
    "llm_only": ["GOOGLE_API_KEY"],
    "rag_search": [],  # solo embeddings locales
    "web_search": ["GOOGLE_API_KEY", "TAVILY_API_KEY"],
}

load_dotenv()
os.environ.setdefault("LLM_TEMPERATURE", "0.3")


# --------------------------------------------------------------------------- #
# Helpers de entorno
# --------------------------------------------------------------------------- #
def _missing_keys(mode: str) -> List[str]:
    required = MODE_REQUIREMENTS.get(mode, [])
    return [k for k in required if not os.getenv(k)]


def answer_llm_only(question: str) -> str:
    return quick_llm_answer(question)


def answer_rag(question: str) -> Dict[str, Any]:
    context = rag_indexer.get_context(question)

    if not context.strip():
        return {
            "answer": (
                "No encontré información relevante en la base de conocimientos.\n"
                "Añade documentos a la carpeta 'data/' y reconstruye el índice."
            ),
            "context": "",
            "source": "rag",
        }

    prompt = (
        "Usa exclusivamente el siguiente contexto para responder a la pregunta.\n"
        "Si la respuesta no está en el contexto, dilo claramente.\n\n"
        "Contexto:\n"
        f"{context}\n\n"
        "Pregunta del usuario:\n"
        f"{question}\n\n"
        "Responde de forma clara y concisa en español."
    )

    answer = answer_llm_only(prompt)
    return {"answer": answer, "context": context, "source": "rag"}


# --------------------------------------------------------------------------- #
# Modelos Pydantic
# --------------------------------------------------------------------------- #
class ChatRequest(BaseModel):
    query: str
    mode: Literal["llm_only", "rag_search", "web_search"] = "llm_only"
    thread_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    mode: Literal["llm_only", "rag_search", "web_search"]
    source: Literal["llm", "rag", "web"]
    context: Optional[str] = None
    timestamp: str
    missing_env: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Núcleo de negocio
# --------------------------------------------------------------------------- #
def run_investigation(query: str, mode: str = "llm_only") -> Dict[str, Any]:
    q = (query or "").strip()
    if not q:
        raise ValueError("La consulta está vacía.")

    normalized_mode = (mode or "llm_only").strip().lower()
    if normalized_mode not in MODE_REQUIREMENTS:
        allowed = ", ".join(MODE_REQUIREMENTS.keys())
        raise ValueError(f"Modo no soportado: {mode}. Usa uno de: {allowed}")

    missing_env = _missing_keys(normalized_mode)
    if missing_env:
        raise EnvironmentError(
            f"Faltan variables para el modo '{normalized_mode}': {', '.join(missing_env)}"
        )

    timestamp = datetime.now().isoformat()

    if normalized_mode == "llm_only":
        answer = answer_llm_only(q)
        return {
            "answer": answer,
            "mode": normalized_mode,
            "source": "llm",
            "context": None,
            "timestamp": timestamp,
            "missing_env": [],
        }

    try:
        if normalized_mode == "rag_search":
            rag_indexer.ensure_rag_ready()
            rag_result = answer_rag(q)
            return {
                "answer": rag_result["answer"],
                "mode": normalized_mode,
                "source": "rag",
                "context": rag_result.get("context"),
                "timestamp": timestamp,
                "missing_env": [],
            }

        web_result = web_search.answer_web_search(q)
        return {
            "answer": web_result.get("answer", "No se obtuvo respuesta."),
            "mode": normalized_mode,
            "source": "web",
            "context": web_result.get("context"),
            "timestamp": timestamp,
            "missing_env": [],
        }

    except RuntimeError as exc:
        raise EnvironmentError(str(exc))


# --------------------------------------------------------------------------- #
# FastAPI
# --------------------------------------------------------------------------- #
app = FastAPI(title="Deep Research Agent API", version="2.0.0")

LOCAL_ORIGINS = [
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
    "http://localhost",
    "http://localhost:8000",
]
PRODUCTION_ORIGIN = os.getenv("PRODUCTION_ORIGIN", "").strip()

allowed_origins = LOCAL_ORIGINS.copy()
if PRODUCTION_ORIGIN:
    allowed_origins.append(PRODUCTION_ORIGIN)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.isdir(IMAGES_DIR):
    app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")


@app.on_event("startup")
async def startup_event():
    try:
        logger.info("[startup] Inicializando índice RAG...")
        rag_indexer.ensure_rag_ready()
        logger.info("[startup] RAG listo.")
    except Exception as exc:
        logger.warning("[startup] No se pudo inicializar RAG: %s", exc)


async def _parse_chat_payload(request: Request) -> ChatRequest:
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="El cuerpo de la solicitud está vacío.")

    text = body.decode("utf-8", errors="ignore").strip()
    if not text:
        raise HTTPException(status_code=400, detail="La consulta está vacía.")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return ChatRequest(query=text)

    if isinstance(parsed, str):
        return ChatRequest(query=parsed)
    if isinstance(parsed, dict):
        try:
            return ChatRequest(**parsed)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    raise HTTPException(
        status_code=422,
        detail="El cuerpo debe ser texto plano o JSON con la clave 'query'.",
    )


@app.get("/health")
def health():
    return {"status": "ok", "time": int(time.time())}


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    if not os.path.exists(INDEX_PATH):
        raise HTTPException(status_code=404, detail="index.html no encontrado")
    return FileResponse(INDEX_PATH, media_type="text/html")


@app.get("/api/info")
def api_info():
    return {
        "status": "ok",
        "message": "Deep Research Agent API",
        "endpoints": ["/health", "/modes", "/chat"],
        "docs": ["/docs", "/redoc"],
    }


@app.get("/modes")
def modes():
    return {
        "modes": ["llm_only", "rag_search", "web_search"],
        "descriptions": {
            "llm_only": "Respuesta directa del modelo sin contexto.",
            "rag_search": "Consulta tu índice local (RAG).",
            "web_search": "Busca en la web y responde con contexto limitado.",
        },
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: Request):
    payload = await _parse_chat_payload(request)
    try:
        result = run_investigation(payload.query, payload.mode)
        return ChatResponse(**result)
    except EnvironmentError as e:
        raise HTTPException(status_code=422, detail=f"ENV_ERROR: {e}")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"VALUE_ERROR: {e}")
    except Exception as e:
        logger.exception("/chat Exception inesperada: %s", e)
        raise HTTPException(status_code=500, detail=f"SERVER_ERROR: {e}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
