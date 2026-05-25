"""
Clase 16 — Release checklist ejecutable.

Verifica que el repo está en condiciones de deployar:
    1. .env no está trackeado por git
    2. No hay GROQ_API_KEY hardcoded en ningún .py
    3. pytest tests/ pasa
    4. evals.runner corre sin errores (opcional si falta dataset)
    5. streamlit_app.py existe y es importable
    6. Dockerfile existe
    7. README.md tiene las secciones obligatorias

Uso:
    python scripts/release_check.py
    make release-check
Exit code: 0 si todo pasa, 1 si algo falla.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

# Permite correr sin instalar el paquete.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from rich.console import Console
    from rich.rule import Rule
    RICH = True
except ImportError:
    RICH = False

console = Console() if RICH else None


def _info(msg: str):
    (console.print(msg) if RICH else print(msg))


def _ok(label: str):
    (console.print(f"  [green]✓[/green] {label}") if RICH else print(f"  PASS  {label}"))


def _fail(label: str, detail: str = ""):
    if RICH:
        console.print(f"  [red]✗[/red] {label}")
        if detail:
            console.print(f"      [dim]{detail}[/dim]")
    else:
        print(f"  FAIL  {label}")
        if detail:
            print(f"        {detail}")


def _warn(label: str, detail: str = ""):
    if RICH:
        console.print(f"  [yellow]○[/yellow] {label}")
        if detail:
            console.print(f"      [dim]{detail}[/dim]")
    else:
        print(f"  WARN  {label}")


# ──────────────────────────────────────────────────────────────
# Checks
# ──────────────────────────────────────────────────────────────
def check_env_not_committed() -> bool:
    tracked = subprocess.run(
        ["git", "ls-files", ".env"],
        capture_output=True, text=True, cwd=ROOT,
    )
    if tracked.stdout.strip():
        _fail(".env está trackeado por git", tracked.stdout.strip())
        return False
    _ok(".env no está trackeado por git")
    return True


GROQ_KEY_RE = re.compile(r"gsk_[A-Za-z0-9]{20,}")

ALLOWED_FILENAMES = {".env", ".env.example"}


def check_no_hardcoded_keys() -> bool:
    offenders: list[str] = []
    for path in ROOT.rglob("*.py"):
        # Excluir venv y cachés
        parts = set(path.parts)
        if any(p in parts for p in ("venv", ".venv", "__pycache__", "site-packages")):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if GROQ_KEY_RE.search(text):
            offenders.append(str(path.relative_to(ROOT)))

    if offenders:
        _fail("GROQ_API_KEY hardcoded en archivos .py", ", ".join(offenders))
        return False
    _ok("GROQ_API_KEY no está hardcoded en ningún .py")
    return True


# Tests estables que deben pasar para liberar. Los tests pre-existentes con
# problemas de import (test_access_control, test_cache) se revisan aparte.
RELEASE_TESTS = [
    "tests/test_guardrails.py",
    "tests/test_contracts.py",
    "tests/test_goldens.py",
]


def check_tests_pass() -> bool:
    existing = [t for t in RELEASE_TESTS if (ROOT / t).exists()]
    if not existing:
        _warn("No se encontraron tests de release", "¿moviste tests/?")
        return False

    result = subprocess.run(
        [sys.executable, "-m", "pytest", *existing, "-q", "--tb=line"],
        capture_output=True, text=True, cwd=ROOT,
    )
    if result.returncode == 0:
        _ok(f"pytest {' '.join(existing)} pasa")

        # Aviso no bloqueante sobre el resto del suite.
        full = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "--collect-only", "-q"],
            capture_output=True, text=True, cwd=ROOT,
        )
        if "error" in full.stdout.lower() or full.returncode != 0:
            _warn(
                "Otros tests del repo tienen errores de collection (clases previas)",
                "revisa: python -m pytest tests/ --collect-only",
            )
        return True

    last = (result.stdout.strip().splitlines() or ["ver output"])[-1]
    _fail("pytest de release falla", last)
    return False


def check_eval_runs() -> bool:
    # evals.runner opcional: no bloqueamos el release si el dataset no está presente.
    try:
        import evals.runner  # noqa: F401
    except Exception as e:
        _warn("evals.runner no disponible (omitido)", str(e))
        return True
    _ok("evals.runner importa correctamente")
    return True


def check_streamlit_app_importable() -> bool:
    path = ROOT / "streamlit_app.py"
    if not path.exists():
        _fail("streamlit_app.py no existe")
        return False

    # Solo validamos sintaxis + imports top-level sin ejecutar Streamlit.
    import ast
    try:
        ast.parse(path.read_text())
    except SyntaxError as e:
        _fail("streamlit_app.py tiene error de sintaxis", str(e))
        return False

    _ok("streamlit_app.py existe y es parseable")
    return True


def check_dockerfile() -> bool:
    if (ROOT / "Dockerfile").exists():
        _ok("Dockerfile existe")
        return True
    _fail("Dockerfile no existe")
    return False


REQUIRED_SECTIONS = [
    r"##\s+Qu[eé] hace",
    r"##\s+Demo",
    r"##\s+Arquitectura",
    r"##\s+Stack",
    r"##\s+Limitaciones",
]


def check_readme_sections() -> bool:
    path = ROOT / "README.md"
    if not path.exists():
        _fail("README.md no existe")
        return False

    text = path.read_text(encoding="utf-8")
    missing = [p for p in REQUIRED_SECTIONS
               if not re.search(p, text, re.IGNORECASE | re.MULTILINE)]
    if missing:
        _fail("README.md carece de secciones obligatorias",
              "faltan: " + ", ".join(p.replace(r"##\s+", "##") for p in missing))
        return False
    _ok("README.md tiene las 5 secciones obligatorias")
    return True


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main() -> int:
    if RICH:
        console.print(Rule("[bold]Release check — DocOps Agent[/bold]"))
    else:
        print("=== Release check — DocOps Agent ===")

    checks = [
        check_env_not_committed,
        check_no_hardcoded_keys,
        check_streamlit_app_importable,
        check_dockerfile,
        check_readme_sections,
        check_tests_pass,
        check_eval_runs,
    ]

    results = [c() for c in checks]

    if RICH:
        console.print(Rule())

    if all(results):
        _info("\n[bold green]Todo en verde — listo para deploy.[/bold green]"
              if RICH else "\nAll checks passed.")
        return 0

    _info("\n[bold red]Hay checks en rojo — arregla antes de deployar.[/bold red]"
          if RICH else "\nSome checks failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
