"""Métricas de calidad para sistemas RAG.

Implementa las 4 métricas estándar usadas por RAGAS/DeepEval, pero construidas
desde cero usando Groq como LLM-as-judge y sentence-transformers como embedder.
El objetivo pedagógico es que los estudiantes vean exactamente cómo se calcula
cada número, sin cajas negras.

Métricas:
    - faithfulness: ¿la respuesta está respaldada por el contexto?
    - context_precision: ¿los chunks recuperados son relevantes?
    - context_recall: ¿el contexto cubre el ground truth?
    - answer_relevancy: ¿la respuesta apunta a la pregunta?

Todas retornan un float en [0.0, 1.0].
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from openai import OpenAI
from sentence_transformers import SentenceTransformer, util


DEFAULT_JUDGE_MODEL = "llama-3.3-70b-versatile"
_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

_EMBEDDER: SentenceTransformer | None = None
_JUDGE_CLIENT: OpenAI | None = None


def _get_embedder() -> SentenceTransformer:
    """Singleton del embedder. Se carga una sola vez por proceso."""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer(_EMBED_MODEL_NAME)
    return _EMBEDDER


def _get_judge_client() -> OpenAI:
    """Singleton del cliente Groq para el judge."""
    global _JUDGE_CLIENT
    if _JUDGE_CLIENT is None:
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY no está definido. Expórtalo o añádelo a tu .env antes de "
                "correr las métricas."
            )
        base_url = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        _JUDGE_CLIENT = OpenAI(api_key=api_key, base_url=base_url)
    return _JUDGE_CLIENT


def _groq_chat_with_retry(
    messages: list[dict[str, str]],
    model: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 512,
    response_format: dict | None = None,
    max_attempts: int = 3,
) -> str:
    """Llama a Groq con backoff exponencial (1s, 2s, 4s)."""
    client = _get_judge_client()
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if response_format is not None:
                kwargs["response_format"] = response_format
            response = client.chat.completions.create(**kwargs)
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            last_exc = exc
            if attempt == max_attempts - 1:
                break
            time.sleep(2**attempt)
    raise RuntimeError(f"Groq falló tras {max_attempts} intentos: {last_exc}")


def _parse_json_defensive(raw: str) -> dict:
    """Intenta parsear JSON; si falla, extrae el primer objeto entre llaves."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError(f"Respuesta sin JSON válido: {raw[:200]}")
        return json.loads(match.group(0))


# ---------- Helpers públicos ----------

def extract_claims(answer: str, judge_model: str = DEFAULT_JUDGE_MODEL) -> list[str]:
    """Descompone una respuesta en claims atómicos verificables."""
    prompt = (
        "Descompón esta respuesta en claims atómicos verificables. Cada claim "
        "debe ser una afirmación independiente que pueda contrastarse contra "
        "una fuente. Devuelve JSON con la forma exacta:\n"
        '{"claims": ["claim 1", "claim 2", ...]}\n\n'
        f"Respuesta:\n{answer}"
    )
    raw = _groq_chat_with_retry(
        messages=[{"role": "user", "content": prompt}],
        model=judge_model,
        response_format={"type": "json_object"},
        max_tokens=512,
    )
    data = _parse_json_defensive(raw)
    claims = data.get("claims", [])
    return [c for c in claims if isinstance(c, str) and c.strip()]


def verify_claim(
    claim: str, context: str, judge_model: str = DEFAULT_JUDGE_MODEL
) -> bool:
    """¿El contexto respalda el claim? Responde SI/NO."""
    prompt = (
        "Tu tarea es verificar si el CONTEXTO respalda el CLAIM. "
        "Responde únicamente con SI o NO, sin explicación.\n\n"
        f"CONTEXTO:\n{context}\n\nCLAIM:\n{claim}\n\nRespuesta:"
    )
    raw = _groq_chat_with_retry(
        messages=[{"role": "user", "content": prompt}],
        model=judge_model,
        max_tokens=8,
    )
    return "SI" in raw.upper()


# ---------- Métricas RAG ----------

def faithfulness(
    answer: str, context: str, judge_model: str = DEFAULT_JUDGE_MODEL
) -> float:
    """Proporción de claims de la respuesta respaldados por el contexto."""
    claims = extract_claims(answer, judge_model=judge_model)
    if not claims:
        return 0.0
    supported = sum(1 for c in claims if verify_claim(c, context, judge_model=judge_model))
    return supported / len(claims)


def context_precision(
    retrieved: list[str], question: str, judge_model: str = DEFAULT_JUDGE_MODEL
) -> float:
    """Fracción de chunks recuperados que son relevantes a la pregunta."""
    if not retrieved:
        return 0.0
    relevant = 0
    for chunk in retrieved:
        prompt = (
            "¿El siguiente CHUNK es relevante para responder la PREGUNTA? "
            "Responde únicamente SI o NO.\n\n"
            f"PREGUNTA:\n{question}\n\nCHUNK:\n{chunk}\n\nRespuesta:"
        )
        raw = _groq_chat_with_retry(
            messages=[{"role": "user", "content": prompt}],
            model=judge_model,
            max_tokens=8,
        )
        if "SI" in raw.upper():
            relevant += 1
    return relevant / len(retrieved)


def context_recall(
    retrieved: list[str], ground_truth: str, judge_model: str = DEFAULT_JUDGE_MODEL
) -> float:
    """Fracción de claims del ground_truth cubiertos por los chunks recuperados."""
    claims = extract_claims(ground_truth, judge_model=judge_model)
    if not claims:
        return 0.0
    context_str = "\n\n".join(retrieved)
    covered = sum(1 for c in claims if verify_claim(c, context_str, judge_model=judge_model))
    return covered / len(claims)


def answer_relevancy(
    answer: str,
    question: str,
    n: int = 5,
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> float:
    """Similitud promedio entre la pregunta original y N preguntas reconstruidas
    desde la respuesta. Mide si la respuesta va al punto."""
    prompt = (
        f"A partir de la RESPUESTA, genera exactamente {n} preguntas distintas "
        "que serían respondidas por ella. Devuelve JSON con la forma exacta:\n"
        '{"questions": ["pregunta 1", ...]}\n\n'
        f"RESPUESTA:\n{answer}"
    )
    raw = _groq_chat_with_retry(
        messages=[{"role": "user", "content": prompt}],
        model=judge_model,
        response_format={"type": "json_object"},
        max_tokens=512,
    )
    data = _parse_json_defensive(raw)
    generated = [q for q in data.get("questions", []) if isinstance(q, str) and q.strip()]
    if not generated:
        return 0.0

    embedder = _get_embedder()
    emb_question = embedder.encode(question, convert_to_tensor=True)
    emb_generated = embedder.encode(generated, convert_to_tensor=True)
    sims = util.cos_sim(emb_question, emb_generated)[0]
    return float(sims.mean().item())


def compute_all_rag_metrics(
    question: str,
    answer: str,
    retrieved: list[str],
    ground_truth: str,
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> dict[str, float]:
    """Calcula las 4 métricas RAG y las devuelve en un dict."""
    context_str = "\n\n".join(retrieved)
    return {
        "faithfulness": faithfulness(answer, context_str, judge_model=judge_model),
        "context_precision": context_precision(retrieved, question, judge_model=judge_model),
        "context_recall": context_recall(retrieved, ground_truth, judge_model=judge_model),
        "answer_relevancy": answer_relevancy(answer, question, judge_model=judge_model),
    }


if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table

    from data.loader import load_eval_set

    console = Console()
    console.print("[bold cyan]evals/metrics.py — smoke test[/bold cyan]")
    console.print(f"Judge: [yellow]{DEFAULT_JUDGE_MODEL}[/yellow]")
    console.print(f"Embedder: [yellow]{_EMBED_MODEL_NAME}[/yellow]")

    _, _, test = load_eval_set()
    ex = test[0]
    console.print(f"\nEjemplo: [dim]{ex.id}[/dim] ({ex.category})")
    console.print(f"  Q: {ex.question}")
    console.print(f"  A (ground_truth): {ex.answer}")

    # Simulamos una respuesta: aquí reutilizamos el ground_truth para que el
    # smoke test muestre métricas altas. En la clase, la answer vendría del
    # sistema bajo prueba.
    simulated_answer = ex.answer

    started = time.perf_counter()
    metrics = compute_all_rag_metrics(
        question=ex.question,
        answer=simulated_answer,
        retrieved=ex.contexts,
        ground_truth=ex.answer,
    )
    elapsed = time.perf_counter() - started

    table = Table(title="RAG metrics", header_style="bold cyan")
    table.add_column("Métrica", style="bold")
    table.add_column("Valor", justify="right")
    for name, value in metrics.items():
        color = "green" if value >= 0.7 else ("yellow" if value >= 0.4 else "red")
        table.add_row(name, f"[{color}]{value:.3f}[/{color}]")
    console.print(table)
    console.print(f"[dim]elapsed: {elapsed:.1f}s[/dim]")
