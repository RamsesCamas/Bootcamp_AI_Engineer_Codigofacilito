"""Compilación EN VIVO con BootstrapFewShot.

Diseñado para ejecutarse en ~60-90s durante la clase. El objetivo es que los
estudiantes VEAN que compilar un módulo DSPy produce un programa con few-shots
curadas automáticamente, y que la mejora es medible.

Flujo:
  1. Configura la LM Groq.
  2. Carga 10 ejemplos del trainset y 5 del valset.
  3. Evalúa el baseline (ChainOfThought) sobre val.
  4. Compila con BootstrapFewShot (usa docops_metric como guía).
  5. Evalúa el optimized sobre val.
  6. Imprime tabla comparativa baseline vs optimized.
"""

from __future__ import annotations

import os
import time

import dspy
import litellm
from dspy.teleprompt import BootstrapFewShot
from rich.console import Console
from rich.table import Table

from data.loader import load_eval_set
from dspy_lab.metric import docops_metric
from dspy_lab.signatures import AnswerWithContext


TRAIN_SIZE = 10
VAL_SIZE = 5


def _truncate(text: str, max_chars: int = 70) -> str:
    text = (text or "").replace("\n", " ")
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def _evaluate(program: dspy.Module, examples: list[dspy.Example]) -> tuple[float, list[str]]:
    scores: list[float] = []
    answers: list[str] = []
    for ex in examples:
        try:
            pred = program(context=ex.context, question=ex.question)
            score = docops_metric(ex, pred)
        except Exception as exc:  # pragma: no cover
            pred = type("P", (), {"answer": f"(error: {exc})"})()
            score = 0.0
        scores.append(score)
        answers.append(pred.answer)
    return (sum(scores) / len(scores) if scores else 0.0, answers)


def main() -> None:
    console = Console()
    console.print("[bold cyan]dspy_lab/optimize_demo.py — BootstrapFewShot en vivo[/bold cyan]")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit(
            "[error] GROQ_API_KEY no está definido. Expórtalo o añádelo a .env."
        )

    # Groq free tier: 8k TPM para openai/gpt-oss-120b.
    # Retries con backoff automático via litellm para no reventar por rate limit.
    litellm.num_retries = 5
    litellm.request_timeout = 120

    lm = dspy.LM(
        "groq/openai/gpt-oss-120b",
        api_key=api_key,
        temperature=0.0,
        max_tokens=2048,
    )
    dspy.configure(lm=lm)
    console.print(
        f"LM: [yellow]groq/openai/gpt-oss-120b[/yellow]  "
        f"optimizer: [yellow]BootstrapFewShot[/yellow]  "
        f"metric: [yellow]docops_metric[/yellow]"
    )

    trainset, valset, _ = load_eval_set()
    trainset = trainset[:TRAIN_SIZE]
    valset = valset[:VAL_SIZE]
    console.print(f"trainset={len(trainset)}  valset={len(valset)}")

    qa_baseline = dspy.ChainOfThought(AnswerWithContext)

    # 1) Evaluar baseline.
    console.print("\n[bold]1/3[/bold] Evaluando baseline (ChainOfThought) sobre val...")
    t0 = time.perf_counter()
    baseline_score, baseline_answers = _evaluate(qa_baseline, valset)
    console.print(f"    baseline score: [yellow]{baseline_score:.3f}[/yellow]  "
                  f"([dim]{time.perf_counter() - t0:.1f}s[/dim])")

    # 2) Compilar.
    console.print("\n[bold]2/3[/bold] Compilando con BootstrapFewShot...")
    optimizer = BootstrapFewShot(
        metric=docops_metric,
        max_bootstrapped_demos=5,
        max_labeled_demos=5,
        max_rounds=2,
    )
    t0 = time.perf_counter()
    qa_optimized = optimizer.compile(qa_baseline, trainset=trainset)
    console.print(f"    compile time: [dim]{time.perf_counter() - t0:.1f}s[/dim]")

    # 3) Evaluar optimized.
    console.print("\n[bold]3/3[/bold] Evaluando optimized sobre val...")
    t0 = time.perf_counter()
    optimized_score, optimized_answers = _evaluate(qa_optimized, valset)
    console.print(f"    optimized score: [yellow]{optimized_score:.3f}[/yellow]  "
                  f"([dim]{time.perf_counter() - t0:.1f}s[/dim])")

    # Tabla comparativa.
    delta = optimized_score - baseline_score
    delta_color = "green" if delta > 0 else ("yellow" if delta == 0 else "red")
    console.print(
        f"\n[bold]Delta docops_metric:[/bold] "
        f"[{delta_color}]{delta:+.3f}[/{delta_color}]"
    )

    table = Table(title="Baseline vs Optimized por ejemplo", header_style="bold cyan")
    table.add_column("pregunta", style="bold")
    table.add_column("baseline", style="yellow")
    table.add_column("optimized", style="green")
    for ex, b_ans, o_ans in zip(valset, baseline_answers, optimized_answers):
        table.add_row(_truncate(ex.question, 40), _truncate(b_ans), _truncate(o_ans))
    console.print(table)

    # Resumen numérico.
    summary = Table(title="Resumen", header_style="bold cyan")
    summary.add_column("programa")
    summary.add_column("docops_metric (avg val)")
    summary.add_row("baseline (CoT)", f"{baseline_score:.3f}")
    summary.add_row("optimized (BootstrapFewShot)", f"{optimized_score:.3f}")
    console.print(summary)


if __name__ == "__main__":
    main()
