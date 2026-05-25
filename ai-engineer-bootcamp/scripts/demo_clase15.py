"""Orquestador de la Clase 15.

Uso:
  python scripts/demo_clase15.py               # corre los 6 pasos con pausas
  python scripts/demo_clase15.py --step N      # corre solo el paso N (1..6)
  python scripts/demo_clase15.py --dry-run     # valida imports + dataset, sin LLM

Pasos:
  1. Setup (config, GROQ_API_KEY, warm-up embedder)
  2. RAG metrics (4 métricas sobre 1 ejemplo)
  3. Agent metrics (tool_call + trajectory sobre traza fake)
  4. DSPy modules (Predict vs CoT vs ReAct)
  5. DSPy compile en vivo (BootstrapFewShot)
  6. Comparación baseline vs optimized sobre 5 ejemplos
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.panel import Panel


console = Console()


def _require_groq_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        console.print(
            "[red]GROQ_API_KEY no está definido.[/red] Expórtalo o añádelo a .env."
        )
        sys.exit(1)
    return key


def _banner(n: int, title: str) -> None:
    console.print(Panel(f"[bold cyan]Paso {n}: {title}[/bold cyan]", expand=False))


def step1_setup(dry_run: bool = False) -> None:
    _banner(1, "Setup")
    console.print(f"Python: {sys.version.split()[0]}")
    console.print(f"CWD: {os.getcwd()}")
    if dry_run:
        _require_groq_key_soft()
    else:
        _require_groq_key()
    from data.loader import load_eval_set

    train, val, test = load_eval_set()
    console.print(f"Dataset DocOps: train={len(train)}  val={len(val)}  test={len(test)}")
    if not dry_run:
        from evals.metrics import _get_embedder
        t0 = time.perf_counter()
        _get_embedder()
        console.print(f"Embedder listo en {time.perf_counter() - t0:.1f}s")


def _require_groq_key_soft() -> None:
    """En dry-run solo avisa, no falla."""
    if not os.environ.get("GROQ_API_KEY"):
        console.print("[yellow]⚠  GROQ_API_KEY no definido (dry-run, se continúa)[/yellow]")


def step2_rag_metrics() -> None:
    _banner(2, "RAG metrics")
    _require_groq_key()
    from data.loader import load_eval_set
    from evals.metrics import compute_all_rag_metrics
    from rich.table import Table

    _, _, test = load_eval_set()
    ex = test[0]
    console.print(f"Ejemplo: [dim]{ex.id}[/dim] ({ex.category})")
    console.print(f"  Q: {ex.question}")

    metrics = compute_all_rag_metrics(
        question=ex.question,
        answer=ex.answer,
        retrieved=ex.contexts,
        ground_truth=ex.answer,
    )
    table = Table(title="RAG metrics", header_style="bold cyan")
    table.add_column("Métrica", style="bold")
    table.add_column("Valor", justify="right")
    for name, value in metrics.items():
        color = "green" if value >= 0.7 else ("yellow" if value >= 0.4 else "red")
        table.add_row(name, f"[{color}]{value:.3f}[/{color}]")
    console.print(table)


def step3_agent_metrics() -> None:
    _banner(3, "Agent metrics")
    # Re-ejecutamos la misma demo del módulo agent_metrics.
    import subprocess
    subprocess.run(
        [sys.executable, "-m", "evals.agent_metrics"], check=True
    )


def step4_modules() -> None:
    _banner(4, "DSPy modules (Predict / CoT / ReAct)")
    _require_groq_key()
    from dspy_lab.modules import demo_modules
    demo_modules()


def step5_compile() -> None:
    _banner(5, "DSPy compile en vivo (BootstrapFewShot)")
    _require_groq_key()
    from dspy_lab.optimize_demo import main as optimize_main
    optimize_main()


def step6_final() -> None:
    _banner(6, "Resumen")
    console.print(
        "Corrió el pipeline completo: RAG metrics + Agent metrics + DSPy modules + "
        "compile en vivo.\n\n"
        "Siguiente: ejecutar [bold]python -m dspy_lab.optimize[/bold] offline para "
        "probar MIPROv2 sobre el dataset completo."
    )


STEPS = {
    1: step1_setup,
    2: step2_rag_metrics,
    3: step3_agent_metrics,
    4: step4_modules,
    5: step5_compile,
    6: step6_final,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int, choices=list(STEPS.keys()))
    parser.add_argument("--dry-run", action="store_true",
                        help="Valida imports + dataset, sin llamar a Groq.")
    args = parser.parse_args()

    if args.dry_run:
        console.print("[bold cyan]demo_clase15 — DRY RUN[/bold cyan]")
        step1_setup(dry_run=True)
        # Imports de validación
        from evals import metrics, agent_metrics  # noqa: F401
        from dspy_lab import signatures, modules, metric, optimize_demo  # noqa: F401
        console.print("[green]✓ imports OK[/green]")
        return

    if args.step is not None:
        STEPS[args.step]()
        return

    for i in sorted(STEPS):
        STEPS[i]()
        if i < max(STEPS):
            try:
                input("Press enter to continue...")
            except EOFError:
                break


if __name__ == "__main__":
    main()
