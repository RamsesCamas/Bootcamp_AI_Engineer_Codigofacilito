"""Compara los 3 patrones de Module de DSPy sobre la misma Signature.

Este demo existe para que los estudiantes vean, en una sola ejecución, la
diferencia entre:
  - dspy.Predict:         input -> output, sin razonamiento explícito.
  - dspy.ChainOfThought:  input -> rationale -> output (CoT implícito).
  - dspy.ReAct:           input -> (pensar + actuar con tools)* -> output.

Todos comparten la signature AnswerWithContext. El tool de ReAct es un mock
que devuelve texto fijo — el foco pedagógico es el patrón, no el retrieval.
"""

from __future__ import annotations

import os
import time

import dspy
import litellm

from dspy_lab.signatures import AnswerWithContext


def mock_search(query: str) -> str:
    """Mock de un retriever. Devuelve siempre el mismo contexto fijo."""
    return (
        "El servicio auth-service corre como daemon systemd en el puerto 8081. "
        "Para reiniciarlo: sudo systemctl restart auth-service. "
        "Verificar estado post-restart con systemctl status auth-service."
    )


def _truncate(text: str, max_chars: int = 80) -> str:
    text = (text or "").replace("\n", " ")
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def demo_modules() -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print("[bold cyan]dspy_lab/modules.py — Predict vs CoT vs ReAct[/bold cyan]")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit(
            "[error] GROQ_API_KEY no está definido. Expórtalo o añádelo a .env."
        )

    litellm.num_retries = 5
    litellm.request_timeout = 120

    lm = dspy.LM(
        "groq/openai/gpt-oss-120b",
        api_key=api_key,
        temperature=0.0,
        max_tokens=2048,
    )
    dspy.configure(lm=lm)
    console.print(f"LM: [yellow]groq/openai/gpt-oss-120b[/yellow]  temperature=0.0")

    context = (
        "El servicio auth-service corre como daemon systemd y escucha en el "
        "puerto 8081 por defecto. Para reiniciarlo: sudo systemctl restart "
        "auth-service. Verificación: systemctl status auth-service debe "
        "mostrar 'active (running)'."
    )
    question = "¿Cómo reinicio el servicio de autenticación y cómo verifico que quedó activo?"

    rows: list[tuple[str, float, int, str, str]] = []

    # 1. Predict
    predict = dspy.Predict(AnswerWithContext)
    started = time.perf_counter()
    out = predict(context=context, question=question)
    latency_ms = (time.perf_counter() - started) * 1000
    tokens = _last_total_tokens(lm)
    rows.append(("Predict", latency_ms, tokens, "", _truncate(out.answer)))

    # 2. ChainOfThought
    cot = dspy.ChainOfThought(AnswerWithContext)
    started = time.perf_counter()
    out_cot = cot(context=context, question=question)
    latency_ms = (time.perf_counter() - started) * 1000
    tokens = _last_total_tokens(lm)
    rationale = getattr(out_cot, "reasoning", "") or getattr(out_cot, "rationale", "")
    rows.append(("ChainOfThought", latency_ms, tokens, _truncate(rationale, 50), _truncate(out_cot.answer)))

    # 3. ReAct con mock_search. ReAct usa una signature question->answer, así
    # que le pasamos una signature compatible (solo question).
    class QuestionToAnswer(dspy.Signature):
        """Responde la pregunta buscando información con la herramienta."""

        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    react = dspy.ReAct(QuestionToAnswer, tools=[mock_search], max_iters=3)
    started = time.perf_counter()
    try:
        out_react = react(question=question)
        react_answer = _truncate(out_react.answer)
        react_err = ""
    except Exception as exc:  # pragma: no cover — depende del runtime DSPy
        react_answer = "(error)"
        react_err = str(exc)[:60]
    latency_ms = (time.perf_counter() - started) * 1000
    tokens = _last_total_tokens(lm)
    extra = react_err or "trace: mock_search()"
    rows.append(("ReAct (+mock_search)", latency_ms, tokens, extra, react_answer))

    table = Table(title="DSPy modules comparison", header_style="bold cyan")
    table.add_column("Módulo", style="bold")
    table.add_column("Latencia (ms)", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Extra", style="dim")
    table.add_column("Output")
    for name, ms, toks, extra, ans in rows:
        table.add_row(name, f"{ms:.0f}", str(toks), extra, ans)
    console.print(table)


def _last_total_tokens(lm: dspy.LM) -> int:
    """Intenta extraer tokens totales de la última llamada en el history de la LM."""
    history = getattr(lm, "history", None) or []
    if not history:
        return 0
    last = history[-1]
    usage = last.get("usage") or {}
    total = usage.get("total_tokens") or 0
    if total:
        return int(total)
    return int((usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0))


if __name__ == "__main__":
    demo_modules()
