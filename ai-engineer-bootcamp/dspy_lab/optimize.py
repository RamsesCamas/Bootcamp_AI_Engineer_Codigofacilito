"""Pipeline de compilación COMPLETA con MIPROv2.

Correr OFFLINE antes de clase. Toma ~10-15 minutos y puede quemar varios
miles de tokens de Groq. Cachea el resultado en JSON — si ya existe, aborta
(pasar --force para recompilar).

Output:
  - dspy_lab/qa_optimized_mipro.json  (programa compilado)
  - evals/reports/mipro_report.md     (comparativa baseline vs optimized)
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import dspy
import litellm
from dspy.teleprompt import MIPROv2
from rich.console import Console

from data.loader import load_eval_set
from dspy_lab.metric import docops_metric
from dspy_lab.signatures import AnswerWithContext


OUT_PROGRAM = Path("dspy_lab/qa_optimized_mipro.json")
OUT_REPORT = Path("evals/reports/mipro_report.md")


def _avg_score(program: dspy.Module, examples: list[dspy.Example]) -> float:
    scores: list[float] = []
    for ex in examples:
        try:
            pred = program(context=ex.context, question=ex.question)
            scores.append(docops_metric(ex, pred))
        except Exception:
            scores.append(0.0)
    return sum(scores) / len(scores) if scores else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="Recompilar aunque el cache exista."
    )
    args = parser.parse_args()

    console = Console()
    console.print("[bold cyan]dspy_lab/optimize.py — MIPROv2 full compile[/bold cyan]")

    if OUT_PROGRAM.exists() and not args.force:
        console.print(f"[yellow]cache hit[/yellow] → {OUT_PROGRAM}")
        console.print("Usa --force para recompilar.")
        return

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit(
            "[error] GROQ_API_KEY no está definido. Expórtalo o añádelo a .env."
        )

    # Groq free tier: 8k TPM para openai/gpt-oss-120b.
    # Configuramos retries agresivos en litellm para que espere automáticamente.
    litellm.num_retries = 5
    litellm.request_timeout = 120

    lm = dspy.LM(
        "groq/openai/gpt-oss-120b",
        api_key=api_key,
        temperature=0.0,
        max_tokens=2048,
    )
    dspy.configure(lm=lm)

    trainset, valset, testset = load_eval_set(splits=(0.6, 0.2, 0.2))
    console.print(
        f"train={len(trainset)}  val={len(valset)}  test={len(testset)}"
    )

    qa_baseline = dspy.ChainOfThought(AnswerWithContext)

    console.print("Evaluando baseline...")
    t0 = time.perf_counter()
    baseline_score = _avg_score(qa_baseline, valset)
    baseline_time = time.perf_counter() - t0
    console.print(f"  baseline: {baseline_score:.3f}  ({baseline_time:.1f}s)")

    console.print("Compilando con MIPROv2 (auto=light)...")
    optimizer = MIPROv2(metric=docops_metric, auto="light")
    t0 = time.perf_counter()
    qa_optimized = optimizer.compile(
        qa_baseline, trainset=trainset, valset=valset, requires_permission_to_run=False
    )
    compile_time = time.perf_counter() - t0
    console.print(f"  compile: {compile_time:.1f}s")

    t0 = time.perf_counter()
    optimized_score = _avg_score(qa_optimized, valset)
    eval_time = time.perf_counter() - t0
    console.print(f"  optimized: {optimized_score:.3f}  ({eval_time:.1f}s)")

    OUT_PROGRAM.parent.mkdir(parents=True, exist_ok=True)
    qa_optimized.save(str(OUT_PROGRAM))
    console.print(f"Programa guardado en {OUT_PROGRAM}")

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    delta = optimized_score - baseline_score
    OUT_REPORT.write_text(
        f"""# MIPROv2 compile report

## Config

- LM: `groq/openai/gpt-oss-120b`
- Metric: `docops_metric` (0.6 * faithfulness + 0.4 * answer_relevancy)
- Optimizer: `MIPROv2(auto="light")`
- Splits: train={len(trainset)} / val={len(valset)} / test={len(testset)}

## Results

| Program | docops_metric (avg val) |
|---------|------------------------:|
| Baseline (ChainOfThought) | {baseline_score:.3f} |
| Optimized (MIPROv2) | {optimized_score:.3f} |
| **Delta** | **{delta:+.3f}** |

## Times

- Baseline eval: {baseline_time:.1f}s
- Compile: {compile_time:.1f}s
- Optimized eval: {eval_time:.1f}s
""",
        encoding="utf-8",
    )
    console.print(f"Reporte guardado en {OUT_REPORT}")


if __name__ == "__main__":
    main()
