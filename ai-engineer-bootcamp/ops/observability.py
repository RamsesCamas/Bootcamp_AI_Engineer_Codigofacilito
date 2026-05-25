"""
Clase 16 — Observabilidad con Arize Phoenix + OpenTelemetry.

Sigue el patrón oficial de la documentación de Phoenix:
    https://docs.arize.com/phoenix/

    - PHOENIX_COLLECTOR_ENDPOINT apunta al endpoint BASE (sin /v1/traces).
    - auto_instrument=True para que Phoenix parchee LangChain/LangGraph/OpenAI
      al importarlos.
    - register() debe llamarse ANTES de importar langchain_openai / langgraph.

Si TRACING_ENABLED=false, es un no-op silencioso (para HF Spaces, Streamlit
Cloud free, etc., donde no hay Phoenix corriendo).

Uso:
    from ops.observability import init_tracing
    init_tracing()                            # al arrancar la app
    from langchain_openai import ChatOpenAI   # importar DESPUÉS
"""

from __future__ import annotations

import os
import logging

logger = logging.getLogger(__name__)

_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_DIM = "\033[2m"
_RESET = "\033[0m"

# Endpoint BASE de Phoenix (sin /v1/traces — se lo añade el cliente).
DEFAULT_COLLECTOR_ENDPOINT = "http://localhost:6006"

_already_initialized = False


def _tracing_enabled() -> bool:
    val = os.getenv("TRACING_ENABLED", "true").strip().lower()
    return val in ("1", "true", "yes", "on")


def _resolve_endpoint() -> str:
    """
    Orden de precedencia:
      1. PHOENIX_COLLECTOR_ENDPOINT (nombre canónico de la doc)
      2. PHOENIX_ENDPOINT (alias de compatibilidad con versiones previas)
      3. default localhost:6006
    """
    return (
        os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
        or os.getenv("PHOENIX_ENDPOINT")
        or DEFAULT_COLLECTOR_ENDPOINT
    ).rstrip("/")


def init_tracing(service_name: str = "docops-agent") -> bool:
    """
    Registra el tracer de Phoenix con autoinstrumentación.

    IMPORTANTE: llama a esta función ANTES de importar langchain_openai,
    langgraph, etc. Si se importan antes, el patch no se aplica.

    Returns:
        True si tracing quedó activo, False si está deshabilitado o falló.
    """
    global _already_initialized

    if _already_initialized:
        return True

    if not _tracing_enabled():
        print(f"{_DIM}[tracing] TRACING_ENABLED=false → tracing deshabilitado{_RESET}")
        return False

    endpoint = _resolve_endpoint()

    # Normaliza: si alguien pasó el endpoint con /v1/traces, lo recortamos,
    # porque phoenix.otel.register espera el base URL.
    if endpoint.endswith("/v1/traces"):
        endpoint = endpoint[: -len("/v1/traces")]

    # Expórtalo en el environment — algunos integradores lo leen directamente.
    os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = endpoint

    try:
        from phoenix.otel import register
    except ImportError as e:
        print(
            f"{_YELLOW}[tracing] ⚠️  Dependencia faltante (módulo: {e.name}).\n"
            f"          Las trazas NO llegarán a Phoenix.\n"
            f"          Instala: pip install arize-phoenix-otel "
            f"openinference-instrumentation-langchain{_RESET}"
        )
        return False

    try:
        register(
            project_name=service_name,
            endpoint=f"{endpoint}/v1/traces",  # register SÍ quiere el path OTLP
            auto_instrument=True,               # parchea langchain, langgraph, openai…
        )
    except Exception as e:
        print(
            f"{_YELLOW}[tracing] No pude inicializar Phoenix en {endpoint} "
            f"({type(e).__name__}: {e}). La app sigue sin tracing.{_RESET}"
        )
        return False

    _already_initialized = True
    print(
        f"{_GREEN}[tracing] Tracing activo → {endpoint} "
        f"(project={service_name}, auto_instrument=True){_RESET}"
    )
    return True


def trace_url() -> str:
    """Devuelve la URL de la UI de Phoenix (el base, sin /v1/traces)."""
    return _resolve_endpoint()
