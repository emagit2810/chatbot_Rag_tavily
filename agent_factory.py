# -*- coding: utf-8 -*-
"""
Gemini Rate Limiting Wrapper.
Compatible con:
- google-generativeai==0.8.5
- langchain-google-genai==1.0.10
"""

from __future__ import annotations

import datetime
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

# Configuración de entorno
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

# --- IMPORTS ---
from deepagents import create_deep_agent
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.outputs import ChatResult

# IMPORTANTE: Esta es la librería correcta para las versiones que instalaste.
# No la cambies, ya que mantiene la capacidad de usar "bind_tools" (necesario para agentes).
from langchain_google_genai import ChatGoogleGenerativeAI

# Intenta importar tu prompt de sistema (las herramientas se cargan bajo demanda).
try:
    from instructions import SYSTEM_PROMPT
except ImportError:
    SYSTEM_PROMPT = "Eres un asistente útil."

logger = logging.getLogger("deep_agent.factory")


def _load_tools():
    """Carga internet_search y rag_search desde tools.py con fallback liviano."""
    try:
        from tools import internet_search, rag_search
    except ImportError:
        def internet_search(query: str) -> str:
            """Simulación de búsqueda web si Tavily no está disponible."""
            return "Simulación de búsqueda."

        def rag_search(query: str) -> str:
            """Simulación de RAG si no hay datos locales."""
            return "Simulación de RAG."

    return internet_search, rag_search

# --------------------------------------------------------------------------- #
# Respuesta rápida solo LLM (sin herramientas ni LangGraph)
# --------------------------------------------------------------------------- #
def quick_llm_answer(text: str) -> str:
    """
    Respuesta rápida solo LLM, sin herramientas ni LangGraph.
    Pensado para el modo `llm_only` desde FastAPI.
    """
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("Falta GOOGLE_API_KEY")

    temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    llm = get_rate_limited_llm(temperature=temperature)
    res = llm.invoke([HumanMessage(content=text)])
    return getattr(res, "content", str(res))


def get_rate_limited_llm(
    temperature: Optional[float] = None,
    timeout: Optional[int] = None,
    max_retries: Optional[int] = None,
) -> RateLimitedGeminiLLM:
    """
    Construye la instancia RateLimitedGeminiLLM usando la prioridad configurada.
    Los límites de RPM/RPD se respetan automáticamente (15 RPM, 1,500 RPD).
    """
    params: Dict[str, Any] = {}
    if temperature is not None:
        params["temperature"] = temperature
    if timeout is not None:
        params["timeout"] = timeout
    if max_retries is not None:
        params["max_retries"] = max_retries
    return RateLimitedGeminiLLM(**params)

# --------------------------------------------------------------------------- #
# Configuración de modelos (Gemini 2.5 family)
# --------------------------------------------------------------------------- #
# Usamos Gemini 2.5 Flash-Lite como modelo primario y Gemini 2.5 Flash como respaldo.
PRIMARY_MODEL = "gemini-2.5-flash-lite"
SECOND_MODEL = "gemini-2.5-flash"
FALLBACK_MODEL = SECOND_MODEL
MODELS_PRIORITY = [PRIMARY_MODEL, SECOND_MODEL]

MODEL_LIMITS: Dict[str, Dict[str, int]] = {
    PRIMARY_MODEL: {"RPD": 1500, "RPM": 15, "MIN_DELAY_SECONDS": 4},
    SECOND_MODEL: {"RPD": 1500, "RPM": 15, "MIN_DELAY_SECONDS": 4},
}

THRESHOLDS: Dict[str, int] = {
    PRIMARY_MODEL: 80,
    SECOND_MODEL: 80,
}

MAX_RETRIES = 3
BACKOFF_BASE_MULTIPLIER = 2
SWITCH_THRESHOLD_PERCENT = 80

# --------------------------------------------------------------------------- #
# Excepciones
# --------------------------------------------------------------------------- #
class QuotaExhaustedException(Exception): pass
class RateLimitException(Exception): pass
class ModelInitializationError(Exception): pass

# --------------------------------------------------------------------------- #
# Request Tracker (Singleton)
# --------------------------------------------------------------------------- #
class RequestTracker:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False): return
        self._model_stats = {}
        self._date_reset = datetime.date.today()
        self._stats_lock = threading.Lock()
        for model_id in MODEL_LIMITS.keys():
            self._model_stats[model_id] = {"requests_today": 0, "last_request_time": 0.0}
        self._initialized = True

    def _check_daily_reset(self):
        today = datetime.date.today()
        if today != self._date_reset:
            with self._stats_lock:
                if today != self._date_reset:
                    for mid in self._model_stats:
                        self._model_stats[mid]["requests_today"] = 0
                    self._date_reset = today

    def get_status_summary(self):
        self._check_daily_reset()
        summary = {}
        with self._stats_lock:
            for mid, stats in self._model_stats.items():
                limit = MODEL_LIMITS.get(mid, {}).get("RPD", 1000)
                used = stats["requests_today"]
                summary[mid] = {
                    "requests_today": used,
                    "limit_rpd": limit,
                    "usage_percent": round((used / limit) * 100, 1) if limit > 0 else 0
                }
        return summary

    def can_make_request(self, model_id: str) -> bool:
        self._check_daily_reset()
        with self._stats_lock:
            limit = MODEL_LIMITS.get(model_id, {}).get("RPD", 1000)
            return self._model_stats[model_id]["requests_today"] < limit

    def record_request(self, model_id: str):
        self._check_daily_reset()
        with self._stats_lock:
            if model_id in self._model_stats:
                self._model_stats[model_id]["requests_today"] += 1
                self._model_stats[model_id]["last_request_time"] = time.time()

    def apply_rate_limit(self, model_id: str):
        min_delay = MODEL_LIMITS.get(model_id, {}).get("MIN_DELAY_SECONDS", 5)
        with self._stats_lock:
            last_time = self._model_stats.get(model_id, {}).get("last_request_time", 0)
        
        elapsed = time.time() - last_time
        if elapsed < min_delay:
            time.sleep(min_delay - elapsed)

request_tracker = RequestTracker()

# --------------------------------------------------------------------------- #
# Quota helpers
# --------------------------------------------------------------------------- #
def get_quota_status() -> Dict[str, Any]:
    """Expose the current quota usage summary for the rate-limited Gemini models."""
    return request_tracker.get_status_summary()

# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
class ModelRouter:
    def __init__(self, models_priority=None):
        self.models = models_priority or MODELS_PRIORITY
        self._current_model = self.models[0]

    def get_optimal_model(self) -> str:
        tracker = request_tracker
        # 1. Buscar modelo bajo el umbral
        for mid in self.models:
            status = tracker.get_status_summary().get(mid, {})
            if tracker.can_make_request(mid) and status.get("usage_percent", 100) < THRESHOLDS.get(mid, 80):
                self._current_model = mid
                return mid
        # 2. Buscar cualquiera con cuota
        for mid in self.models:
            if tracker.can_make_request(mid):
                self._current_model = mid
                return mid
        raise QuotaExhaustedException("Todos los modelos han agotado su cuota diaria.")

# --------------------------------------------------------------------------- #
# Rate Limited Wrapper
# --------------------------------------------------------------------------- #
class RateLimitedGeminiLLM(BaseChatModel):
    """
    Wrapper compatible con LangChain Core y DeepAgents.
    """
    primary_model: str = PRIMARY_MODEL
    fallback_model: str = FALLBACK_MODEL
    temperature: float = 0
    max_retries: int = MAX_RETRIES
    timeout: Optional[int] = 60
    
    # Campos privados no serializables por Pydantic
    _router: ModelRouter = None
    _llm_instances: Dict[str, ChatGoogleGenerativeAI] = {}
    _init_lock: threading.Lock = None

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Inicialización de objetos privados fuera de Pydantic
        object.__setattr__(self, "_router", ModelRouter(MODELS_PRIORITY))
        object.__setattr__(self, "_llm_instances", {})
        object.__setattr__(self, "_init_lock", threading.Lock())
        self._initialize_models()

    def _initialize_models(self):
        """Inicializa las instancias reales de ChatGoogleGenerativeAI"""
        for mid in MODELS_PRIORITY:
            try:
                # AQUÍ ES DONDE SE USA LA DEPENDENCIA QUE INSTALASTE (v1.0.10)
                llm = ChatGoogleGenerativeAI(
                    model=mid,
                    temperature=self.temperature,
                    max_retries=1, 
                    timeout=self.timeout,
                    google_api_key=os.getenv("GOOGLE_API_KEY")
                )
                self._llm_instances[mid] = llm
            except Exception as e:
                logger.error(f"Error init modelo {mid}: {e}")

    @property
    def _llm_type(self) -> str:
        return "rate-limited-gemini-router"

    def _generate(self, messages: List[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        attempts = 0
        while attempts < self.max_retries:
            try:
                attempts += 1
                mid = self._router.get_optimal_model()
                logger.info("GeminiLLM: intento=%s modelo=%s", attempts, mid)

                llm = self._llm_instances.get(mid)
                if not llm:
                    logger.error("GeminiLLM: modelo %s no inicializado", mid)
                    raise ModelInitializationError(f"Modelo {mid} no cargado.")

                request_tracker.apply_rate_limit(mid)
                logger.debug("GeminiLLM: rate_limit aplicado para %s", mid)

                # Delegamos la generación al modelo real de LangChain
                result = llm._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

                request_tracker.record_request(mid)
                logger.info("GeminiLLM: respuesta OK modelo=%s", mid)
                return result

            except Exception as e:
                error_str = str(e).lower()
                logger.warning("GeminiLLM: error en intento %s: %s", attempts, e)
                if "429" in error_str or "quota" in error_str:
                    time.sleep(2 * attempts)
                    continue
                logger.error("GeminiLLM: error no recuperable: %s", e, exc_info=True)
                raise e

        logger.error("GeminiLLM: Max retries exceeded")
        raise RateLimitException("Max retries exceeded")

    # Necesario para que DeepAgents pueda asignar herramientas al agente
    def bind_tools(self, tools, tool_choice=None, **kwargs):
        bound_instances = {}
        for mid, llm in self._llm_instances.items():
            # ChatGoogleGenerativeAI tiene soporte nativo para bind_tools
            bound_instances[mid] = llm.bind_tools(tools, tool_choice=tool_choice, **kwargs)
        
        # Clonamos el wrapper
        new = self.__class__(
            primary_model=self.primary_model,
            fallback_model=self.fallback_model,
            temperature=self.temperature,
            max_retries=self.max_retries,
            timeout=self.timeout
        )
        object.__setattr__(new, "_llm_instances", bound_instances)
        return new

# --------------------------------------------------------------------------- #
# Factory Deep Agent
# --------------------------------------------------------------------------- #
def build_deep_agent(checkpointer=None, tool_mode: str = "internet_search"):
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("Falta GOOGLE_API_KEY")

    logger.info("Construyendo agente con Gemini 2.5 Flash-Lite (prioritario).")

    internet_search, rag_search = _load_tools()

    # Creamos nuestro LLM personalizado
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    llm = get_rate_limited_llm(temperature=temperature)

    # Selección de herramientas
    tools_map = {
        "internet_search": [internet_search],
        "rag_search": [rag_search],
        "both": [internet_search, rag_search],
        "llm_only": [],  # sin herramientas realmente
    }
    selected_tools = tools_map.get(tool_mode, [internet_search])

    # Crear agente usando DeepAgents
    # DeepAgents espera un objeto compatible con LangChain, que es lo que hemos creado.
    agent = create_deep_agent(
        model=llm,
        tools=selected_tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
    return agent

# --------------------------------------------------------------------------- #
# Main Execution
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    if os.getenv("GOOGLE_API_KEY"):
        print("✅ API Key encontrada.")
        try:
            agent = build_deep_agent(tool_mode="internet_search")
            print("✅ Agente construido exitosamente con dependencias compatibles.")
            
            # Prueba rápida (opcional, descomentar para probar)
            # print("Probando invocación...")
            # res = agent.invoke({"messages": [("user", "Hola, ¿estás funcionando?")]}, config={"configurable": {"thread_id": "1"}})
            # print(res)
            
        except Exception as e:
            print(f"❌ Error construyendo agente: {e}")
    else:
        print("⚠️ Configura tu GOOGLE_API_KEY")
