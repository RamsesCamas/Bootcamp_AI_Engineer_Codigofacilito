"""Métricas determinísticas para evaluar agentes (tool-calling + trayectoria).

A diferencia de las métricas RAG, estas NO llaman al LLM: comparan estructuras
de datos (llamadas a tools, trazas de ejecución) contra una referencia. Son
rápidas, deterministas y perfectas para CI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ExpectedCall:
    """Descripción de una llamada esperada a un tool."""

    tool: str
    args: dict[str, Any]


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Longest Common Subsequence clásico. O(len(a)*len(b))."""
    if not a or not b:
        return 0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, ai in enumerate(a, start=1):
        for j, bj in enumerate(b, start=1):
            if ai == bj:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[len(a)][len(b)]


def _value_fuzzy_equal(a: Any, b: Any) -> bool:
    """Igualdad con tolerancia: exact match o substring insensible a mayúsculas."""
    if a == b:
        return True
    if isinstance(a, str) and isinstance(b, str):
        al, bl = a.lower().strip(), b.lower().strip()
        return al == bl or al in bl or bl in al
    return False


def tool_call_accuracy(
    actual: list[dict[str, Any]], expected: list[ExpectedCall]
) -> dict[str, float]:
    """Compara llamadas reales del agente contra las esperadas.

    Args:
        actual: lista de dicts con keys "tool", "args" y opcionalmente "result".
        expected: lista de ExpectedCall.

    Returns:
        Dict con:
          - tool_match: Jaccard sobre los sets de nombres de tools usados.
          - arg_match: fracción de keys alineadas por posición con valores fuzzy-iguales.
          - order_match: LCS(nombres_actual, nombres_esperados) / len(esperados).
    """
    if not expected:
        return {"tool_match": 0.0, "arg_match": 0.0, "order_match": 0.0}

    actual_tools = [c.get("tool", "") for c in actual]
    expected_tools = [c.tool for c in expected]

    set_actual = set(actual_tools)
    set_expected = set(expected_tools)
    if set_actual or set_expected:
        tool_match = len(set_actual & set_expected) / len(set_actual | set_expected)
    else:
        tool_match = 0.0

    pair_count = min(len(actual), len(expected))
    total_keys = 0
    matched_keys = 0
    for i in range(pair_count):
        a_args = actual[i].get("args", {}) or {}
        e_args = expected[i].args or {}
        keys = set(a_args) | set(e_args)
        for k in keys:
            total_keys += 1
            if k in a_args and k in e_args and _value_fuzzy_equal(a_args[k], e_args[k]):
                matched_keys += 1
    arg_match = (matched_keys / total_keys) if total_keys else 1.0

    order_match = _lcs_length(actual_tools, expected_tools) / len(expected_tools)

    return {
        "tool_match": round(tool_match, 4),
        "arg_match": round(arg_match, 4),
        "order_match": round(order_match, 4),
    }


def evaluate_trajectory(
    trace: list[dict[str, Any]], optimal_steps: int
) -> dict[str, float | int]:
    """Evalúa la trayectoria de ejecución del agente.

    Args:
        trace: lista de pasos con shape {"type": "tool_call"|"think"|"final",
               "tool": str?, "args": dict?, "result": Any?, "error": str?}.
        optimal_steps: cantidad ideal de pasos para resolver la tarea.

    Returns:
        Dict con:
          - step_efficiency: optimal_steps / len(trace), clamp [0, 1].
          - redundancy: cantidad de tool_calls con (tool, args) repetido.
          - recovery_rate: de los tool_calls con error, fracción seguida por
            un tool_call sin error (en cualquier posición posterior adyacente).
    """
    if not trace:
        return {"step_efficiency": 0.0, "redundancy": 0, "recovery_rate": 0.0}

    step_efficiency = min(1.0, optimal_steps / len(trace))

    tool_calls = [
        (i, step) for i, step in enumerate(trace) if step.get("type") == "tool_call"
    ]

    seen: set[tuple[str, str]] = set()
    redundancy = 0
    for _, step in tool_calls:
        key = (step.get("tool", ""), repr(sorted((step.get("args") or {}).items())))
        if key in seen:
            redundancy += 1
        else:
            seen.add(key)

    failures = [idx for idx, (_, step) in enumerate(tool_calls) if step.get("error")]
    if not failures:
        recovery_rate = 1.0
    else:
        recovered = 0
        for tc_idx in failures:
            if tc_idx + 1 < len(tool_calls) and not tool_calls[tc_idx + 1][1].get("error"):
                recovered += 1
        recovery_rate = recovered / len(failures)

    return {
        "step_efficiency": round(step_efficiency, 4),
        "redundancy": redundancy,
        "recovery_rate": round(recovery_rate, 4),
    }


if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    console.print("[bold cyan]evals/agent_metrics.py — smoke test[/bold cyan]")

    # Caso 1: tool_call_accuracy
    actual_calls = [
        {"tool": "search_docs", "args": {"query": "reinicio auth-service"}, "result": "..."},
        {"tool": "search_docs", "args": {"query": "reinicio auth-service"}, "result": "..."},  # redundante
        {"tool": "run_command", "args": {"cmd": "systemctl restart auth-service"}, "result": "ok"},
    ]
    expected_calls = [
        ExpectedCall(tool="search_docs", args={"query": "reiniciar auth-service"}),
        ExpectedCall(tool="run_command", args={"cmd": "systemctl restart auth-service"}),
    ]
    tca = tool_call_accuracy(actual_calls, expected_calls)

    table = Table(title="tool_call_accuracy", header_style="bold cyan")
    table.add_column("Métrica", style="bold")
    table.add_column("Valor", justify="right")
    for name, value in tca.items():
        color = "green" if value >= 0.7 else ("yellow" if value >= 0.4 else "red")
        table.add_row(name, f"[{color}]{value:.3f}[/{color}]")
    console.print(table)

    # Caso 2: evaluate_trajectory con fallo + recovery + redundancia
    trace = [
        {"type": "think", "content": "Necesito buscar el runbook"},
        {"type": "tool_call", "tool": "search_docs", "args": {"query": "auth restart"}},
        {"type": "tool_call", "tool": "search_docs", "args": {"query": "auth restart"}},  # redundante
        {"type": "tool_call", "tool": "run_command", "args": {"cmd": "systemctl restart x"}, "error": "no such service"},
        {"type": "tool_call", "tool": "run_command", "args": {"cmd": "systemctl restart auth-service"}, "result": "ok"},
        {"type": "final", "content": "Servicio reiniciado."},
    ]
    traj = evaluate_trajectory(trace, optimal_steps=2)

    panel_body = (
        f"step_efficiency: [yellow]{traj['step_efficiency']:.3f}[/yellow]\n"
        f"redundancy:      [yellow]{traj['redundancy']}[/yellow]\n"
        f"recovery_rate:   [yellow]{traj['recovery_rate']:.3f}[/yellow]\n"
        f"(trace length: {len(trace)} steps, optimal: 2)"
    )
    console.print(Panel(panel_body, title="evaluate_trajectory", border_style="cyan"))
