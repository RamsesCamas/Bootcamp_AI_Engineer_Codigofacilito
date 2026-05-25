"""Signatures de DSPy para el DocOps Agent.

Una Signature es la interfaz declarativa de un módulo: qué entra, qué sale,
y qué se espera que haga. DSPy usa docstring + InputField/OutputField para
compilar prompts automáticamente.
"""

from __future__ import annotations

import dspy


class AnswerWithContext(dspy.Signature):
    """Responde la pregunta usando solo el contexto provisto."""

    context: str = dspy.InputField(desc="Documentos recuperados relevantes a la pregunta")
    question: str = dspy.InputField(desc="Pregunta del usuario en lenguaje natural")
    answer: str = dspy.OutputField(desc="Respuesta concisa basada únicamente en el contexto")


class RewriteQuery(dspy.Signature):
    """Reformula la query del usuario para mejorar el retrieval."""

    user_query: str = dspy.InputField(desc="Query original del usuario")
    rewritten: str = dspy.OutputField(desc="Query optimizada para BM25 + embeddings")


class ClassifyIntent(dspy.Signature):
    """Clasifica la intención del usuario en una de 3 categorías."""

    user_query: str = dspy.InputField(desc="Query del usuario")
    intent: str = dspy.OutputField(
        desc="Una de: 'factual', 'how-to', 'troubleshooting'"
    )
