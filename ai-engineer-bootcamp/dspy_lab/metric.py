"""Métrica objetivo para compilar módulos DSPy.

DSPy necesita una función Python `metric(example, prediction, trace=None) -> float`
para guiar optimizadores como BootstrapFewShot y MIPROv2. Aquí reusamos las
métricas de evals/metrics.py para combinar faithfulness + answer_relevancy.
"""

from __future__ import annotations

import logging
from typing import Any

from evals.metrics import answer_relevancy, faithfulness


logger = logging.getLogger(__name__)


def docops_metric(example: Any, prediction: Any, trace: Any = None) -> float:
    """Métrica combinada para QA del DocOps Agent.

    60% faithfulness (¿la answer se sostiene en el contexto?)
    + 40% answer_relevancy (¿la answer va al grano?).

    Requiere que example tenga .context y .question, y que prediction tenga
    .answer — que es exactamente la forma de AnswerWithContext.

    Si el judge falla (ej. rate limit), devolvemos 0.0 con warning en vez de
    propagar la excepción, porque optimizadores como BootstrapFewShot se caen
    cuando la métrica explota.
    """
    try:
        f = faithfulness(prediction.answer, example.context)
        r = answer_relevancy(prediction.answer, example.question)
    except Exception as exc:
        logger.warning("docops_metric falló: %s", exc)
        return 0.0
    return 0.6 * f + 0.4 * r


def docops_metric_with_cost(example: Any, prediction: Any, trace: Any = None) -> float:
    """Versión con penalización por longitud del trace (aprox. costo)."""
    quality = docops_metric(example, prediction, trace)
    cost_penalty = 0.001 * len(trace) if trace else 0
    return quality - cost_penalty
