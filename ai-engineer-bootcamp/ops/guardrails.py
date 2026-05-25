"""
Clase 16 — Guardrails de seguridad para el DocOps Agent.

Tres capas complementarias:
    1. InputGuardrail  — bloquea prompt injection y mensajes abusivos.
    2. OutputGuardrail — redacta PII (emails, teléfonos MX, RFC, CURP, tarjetas).
    3. ToolGuardrail   — clasifica tools por riesgo y gestiona rate limits.

Todo en regex propio (sin Presidio): para una clase en vivo los estudiantes
necesitan ver los patrones directamente, no una caja negra.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


# ─────────────────────────────────────────────────────────────────
# Resultado común
# ─────────────────────────────────────────────────────────────────
@dataclass
class GuardrailResult:
    blocked: bool
    reason: str | None = None
    scrubbed_text: str | None = None


# ─────────────────────────────────────────────────────────────────
# 1. Input guardrail
# ─────────────────────────────────────────────────────────────────
class InputGuardrail:
    """Valida mensajes de entrada del usuario antes de invocar al agente."""

    MAX_LENGTH = 4000

    INJECTION_PATTERNS = [
        r"ignore\s+(the\s+)?previous\s+instructions?",
        r"ignore\s+(all\s+)?prior\s+instructions?",
        r"disregard\s+(all|the)\s+(above|prior|previous)",
        r"you\s+are\s+now\s+",
        r"system\s+prompt",
        r"reveal\s+your\s+instructions?",
        r"forget\s+everything",
        r"override\s+your\s+rules",
    ]

    def __init__(self, max_length: int = MAX_LENGTH):
        self.max_length = max_length
        self._compiled = [
            re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS
        ]

    def check(self, user_message: str) -> GuardrailResult:
        if not isinstance(user_message, str) or not user_message.strip():
            return GuardrailResult(
                blocked=True, reason="Mensaje vacío o inválido."
            )

        if len(user_message) > self.max_length:
            return GuardrailResult(
                blocked=True,
                reason=(
                    f"Mensaje demasiado largo "
                    f"({len(user_message)} > {self.max_length} caracteres)."
                ),
            )

        for pattern in self._compiled:
            if pattern.search(user_message):
                return GuardrailResult(
                    blocked=True,
                    reason=(
                        "Posible prompt injection detectado "
                        f"(patrón: '{pattern.pattern}')."
                    ),
                )

        return GuardrailResult(blocked=False)


# ─────────────────────────────────────────────────────────────────
# 2. Output guardrail (PII scrubbing)
# ─────────────────────────────────────────────────────────────────
class OutputGuardrail:
    """Redacta PII de las respuestas antes de mostrarlas al usuario."""

    # Orden IMPORTA — tarjetas primero, luego teléfonos (ambos son dígitos).
    PATTERNS: list[tuple[str, str, re.Pattern]] = [
        (
            "EMAIL",
            "[EMAIL]",
            re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
        ),
        (
            "CARD",
            "[CARD]",
            # 13-19 dígitos con separadores opcionales (espacios o guiones).
            re.compile(r"\b(?:\d[ -]?){12,18}\d\b"),
        ),
        (
            "CURP",
            "[CURP]",
            re.compile(
                r"\b[A-Z][AEIOUX][A-Z]{2}\d{2}"
                r"(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])"
                r"[HM][A-Z]{2}[B-DF-HJ-NP-TV-Z]{3}[0-9A-Z]\d\b",
                re.IGNORECASE,
            ),
        ),
        (
            "RFC",
            "[RFC]",
            re.compile(
                r"\b[A-ZÑ&]{3,4}\d{6}(?:[A-Z0-9]{3})?\b",
                re.IGNORECASE,
            ),
        ),
        (
            "PHONE",
            "[PHONE]",
            # Teléfonos MX: opcional +52, 10 dígitos con separadores opcionales.
            re.compile(
                r"(?:\+?52[\s-]?)?(?:\d[\s-]?){9}\d"
            ),
        ),
    ]

    def scrub(self, response: str) -> GuardrailResult:
        if not isinstance(response, str):
            return GuardrailResult(
                blocked=False, scrubbed_text=str(response)
            )

        scrubbed = response
        detected: list[str] = []

        for name, replacement, pattern in self.PATTERNS:
            new_text, count = pattern.subn(replacement, scrubbed)
            if count > 0:
                detected.append(f"{name}×{count}")
            scrubbed = new_text

        reason = ", ".join(detected) if detected else None
        return GuardrailResult(
            blocked=False,
            reason=reason,
            scrubbed_text=scrubbed,
        )


# ─────────────────────────────────────────────────────────────────
# 3. Tool guardrail (clasificación por riesgo)
# ─────────────────────────────────────────────────────────────────
RiskLevel = Literal["read", "write_reversible", "write_destructive"]

TOOL_RISK_LEVELS: dict[str, RiskLevel] = {
    # Lectura: sin side effects.
    "search_docs": "read",
    "retrieve": "read",
    "vector_search": "read",
    "list_documents": "read",

    # Escritura reversible: puedes deshacer con otra llamada.
    "create_ticket": "write_reversible",
    "send_email_draft": "write_reversible",
    "tag_document": "write_reversible",
    "update_status": "write_reversible",

    # Destructivo: requiere aprobación humana.
    "delete_document": "write_destructive",
    "execute_sql": "write_destructive",
    "shell_exec": "write_destructive",
    "deploy_production": "write_destructive",
}

RATE_LIMITS: dict[RiskLevel, int] = {
    "read": 60,
    "write_reversible": 20,
    "write_destructive": 2,
}


class ToolGuardrail:
    """Decide qué tools necesitan HITL y cuántas calls/min permite cada una."""

    UNKNOWN_RISK: RiskLevel = "write_destructive"

    def __init__(
        self,
        risk_levels: dict[str, RiskLevel] | None = None,
        rate_limits: dict[RiskLevel, int] | None = None,
    ):
        self.risk_levels = risk_levels or TOOL_RISK_LEVELS
        self.rate_limits = rate_limits or RATE_LIMITS

    def risk_of(self, tool_name: str) -> RiskLevel:
        return self.risk_levels.get(tool_name, self.UNKNOWN_RISK)

    def require_approval(self, tool_name: str) -> bool:
        return self.risk_of(tool_name) == "write_destructive"

    def rate_limit_for(self, tool_name: str) -> int:
        return self.rate_limits[self.risk_of(tool_name)]
