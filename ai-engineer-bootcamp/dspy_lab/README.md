# dspy_lab — compilación de prompts con DSPy

Este paquete contiene los artefactos de la Clase 15: signatures, módulos,
métrica objetivo y dos pipelines de compilación.

## Cuándo usar cada compile

| Script | Optimizer | Ejemplos | Tiempo | Para qué |
|--------|-----------|---------:|-------:|----------|
| `optimize_demo.py` | `BootstrapFewShot` | 10 train / 5 val | ~60-90s | Correr **en vivo** en clase. El objetivo es ver compilación funcionar y mejora medible. |
| `optimize.py` | `MIPROv2(auto="light")` | full train / val | ~10-15 min | Correr **offline** antes de clase. Guarda el programa y un reporte para mostrarlo. |

## Flujo sugerido

```bash
# En vivo (clase)
python -m dspy_lab.modules            # compara Predict vs CoT vs ReAct
python -m dspy_lab.optimize_demo      # compila con BootstrapFewShot y muestra mejora

# Offline (pre-clase)
python -m dspy_lab.optimize           # compila con MIPROv2, cachea el resultado
python -m dspy_lab.optimize --force   # recompila ignorando el cache
```

## Archivos

- `signatures.py` — AnswerWithContext / RewriteQuery / ClassifyIntent.
- `modules.py` — `demo_modules()` compara Predict / CoT / ReAct.
- `metric.py` — `docops_metric` y `docops_metric_with_cost` (reusa `evals/metrics.py`).
- `optimize.py` — MIPROv2 full (offline).
- `optimize_demo.py` — BootstrapFewShot rápido (en vivo).
- `qa_optimized_mipro.json` — programa compilado (generado por `optimize.py`).

## Requisitos

- `GROQ_API_KEY` en `.env`.
- `dspy>=2.5,<3.0` (instalado vía `requirements.txt`).
