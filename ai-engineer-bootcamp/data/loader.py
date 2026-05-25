"""Loader del dataset DocOps para la Clase 15.

Devuelve splits train/val/test como listas de `dspy.Example`, listos para
entrenar/evaluar módulos DSPy. Los inputs declarados son `context` y
`question`; el `answer` y el `contexts` original quedan disponibles como
atributos del Example pero no como inputs.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import dspy


DEFAULT_PATH = Path(__file__).parent / "eval_set_docops.jsonl"


def load_eval_set(
    path: str | Path = DEFAULT_PATH,
    splits: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 42,
) -> tuple[list[dspy.Example], list[dspy.Example], list[dspy.Example]]:
    """Carga el JSONL y devuelve (trainset, valset, testset) como dspy.Example.

    Cada Example tiene los campos:
      - question (input)
      - context (input, string unificado de los chunks)
      - answer (label)
      - contexts (list[str], chunks originales por si una métrica los quiere)
      - category (factual | how-to | troubleshooting)
      - id
    """
    if abs(sum(splits) - 1.0) > 1e-6:
        raise ValueError(f"splits debe sumar 1.0, recibido {splits}")

    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    rng = random.Random(seed)
    rng.shuffle(rows)

    n = len(rows)
    n_train = int(n * splits[0])
    n_val = int(n * splits[1])

    train_rows = rows[:n_train]
    val_rows = rows[n_train : n_train + n_val]
    test_rows = rows[n_train + n_val :]

    def to_example(row: dict) -> dspy.Example:
        context_str = "\n\n".join(row["contexts"])
        ex = dspy.Example(
            question=row["question"],
            context=context_str,
            answer=row["ground_truth"],
            contexts=row["contexts"],
            category=row["category"],
            id=row["id"],
        ).with_inputs("context", "question")
        return ex

    return (
        [to_example(r) for r in train_rows],
        [to_example(r) for r in val_rows],
        [to_example(r) for r in test_rows],
    )


if __name__ == "__main__":
    train, val, test = load_eval_set()
    print(f"train={len(train)} val={len(val)} test={len(test)}")
    print("Ejemplo train[0]:")
    print(f"  id: {train[0].id}")
    print(f"  category: {train[0].category}")
    print(f"  question: {train[0].question}")
    print(f"  answer: {train[0].answer[:80]}...")
    print(f"  inputs: {train[0].inputs().keys()}")
