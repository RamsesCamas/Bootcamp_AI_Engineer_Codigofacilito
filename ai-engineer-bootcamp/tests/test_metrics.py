"""Tests para evals/metrics.py y evals/agent_metrics.py.

Los tests que tocarían a Groq mockean `_groq_chat_with_retry` — no queremos
que el CI dependa de red ni de un API key.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from evals import metrics
from evals.agent_metrics import (
    ExpectedCall,
    evaluate_trajectory,
    tool_call_accuracy,
)


# ------------------- metrics.extract_claims -------------------

def test_extract_claims_json_limpio():
    fake_resp = '{"claims": ["El servicio corre en 8081", "Se reinicia con systemctl"]}'
    with patch.object(metrics, "_groq_chat_with_retry", return_value=fake_resp):
        claims = metrics.extract_claims("texto irrelevante")
    assert claims == ["El servicio corre en 8081", "Se reinicia con systemctl"]


def test_extract_claims_json_con_prefacio():
    fake_resp = 'Aquí tienes la salida:\n{"claims": ["claim único"]}\nFin.'
    with patch.object(metrics, "_groq_chat_with_retry", return_value=fake_resp):
        claims = metrics.extract_claims("texto")
    assert claims == ["claim único"]


def test_extract_claims_ignora_no_string():
    fake_resp = '{"claims": ["ok", 42, "", "otro"]}'
    with patch.object(metrics, "_groq_chat_with_retry", return_value=fake_resp):
        claims = metrics.extract_claims("texto")
    assert claims == ["ok", "otro"]


# ------------------- metrics.verify_claim -------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("SI", True),
        ("si, claramente", True),
        ("NO", False),
        ("no aplica", False),
        ("SI respalda el claim", True),
    ],
)
def test_verify_claim_parses_si_no(raw, expected):
    with patch.object(metrics, "_groq_chat_with_retry", return_value=raw):
        assert metrics.verify_claim("un claim", "un contexto") is expected


# ------------------- metrics.faithfulness -------------------

def test_faithfulness_todos_soportados():
    with patch.object(metrics, "extract_claims", return_value=["c1", "c2"]):
        with patch.object(metrics, "verify_claim", return_value=True):
            assert metrics.faithfulness("a", "ctx") == 1.0


def test_faithfulness_ninguno_soportado():
    with patch.object(metrics, "extract_claims", return_value=["c1", "c2"]):
        with patch.object(metrics, "verify_claim", return_value=False):
            assert metrics.faithfulness("a", "ctx") == 0.0


def test_faithfulness_mixto():
    with patch.object(metrics, "extract_claims", return_value=["c1", "c2", "c3", "c4"]):
        with patch.object(metrics, "verify_claim", side_effect=[True, False, True, True]):
            assert metrics.faithfulness("a", "ctx") == 0.75


def test_faithfulness_sin_claims():
    with patch.object(metrics, "extract_claims", return_value=[]):
        assert metrics.faithfulness("a", "ctx") == 0.0


# ------------------- agent_metrics.tool_call_accuracy -------------------

def test_tool_call_accuracy_match_perfecto():
    actual = [
        {"tool": "search", "args": {"q": "foo"}},
        {"tool": "run", "args": {"cmd": "ls"}},
    ]
    expected = [
        ExpectedCall(tool="search", args={"q": "foo"}),
        ExpectedCall(tool="run", args={"cmd": "ls"}),
    ]
    result = tool_call_accuracy(actual, expected)
    assert result["tool_match"] == 1.0
    assert result["arg_match"] == 1.0
    assert result["order_match"] == 1.0


def test_tool_call_accuracy_tools_desordenados():
    actual = [
        {"tool": "run", "args": {"cmd": "ls"}},
        {"tool": "search", "args": {"q": "foo"}},
    ]
    expected = [
        ExpectedCall(tool="search", args={"q": "foo"}),
        ExpectedCall(tool="run", args={"cmd": "ls"}),
    ]
    result = tool_call_accuracy(actual, expected)
    assert result["tool_match"] == 1.0
    # LCS {run, search} vs {search, run} = 1, /2 = 0.5
    assert result["order_match"] == 0.5


def test_tool_call_accuracy_expected_vacio():
    result = tool_call_accuracy([{"tool": "x", "args": {}}], [])
    assert result == {"tool_match": 0.0, "arg_match": 0.0, "order_match": 0.0}


def test_tool_call_accuracy_arg_fuzzy():
    actual = [{"tool": "search", "args": {"q": "restart AUTH-SERVICE"}}]
    expected = [ExpectedCall(tool="search", args={"q": "restart auth-service"})]
    result = tool_call_accuracy(actual, expected)
    # Fuzzy equal (case-insensitive + substring)
    assert result["arg_match"] == 1.0


# ------------------- agent_metrics.evaluate_trajectory -------------------

def test_trajectory_ideal():
    trace = [
        {"type": "tool_call", "tool": "search", "args": {"q": "x"}},
        {"type": "tool_call", "tool": "run", "args": {"cmd": "ls"}},
        {"type": "final"},
    ]
    result = evaluate_trajectory(trace, optimal_steps=3)
    assert result["step_efficiency"] == 1.0
    assert result["redundancy"] == 0
    assert result["recovery_rate"] == 1.0  # sin errores → 1.0 por convención


def test_trajectory_con_redundancia():
    trace = [
        {"type": "tool_call", "tool": "search", "args": {"q": "x"}},
        {"type": "tool_call", "tool": "search", "args": {"q": "x"}},
        {"type": "tool_call", "tool": "search", "args": {"q": "x"}},
    ]
    result = evaluate_trajectory(trace, optimal_steps=1)
    assert result["redundancy"] == 2


def test_trajectory_fallo_y_recovery():
    trace = [
        {"type": "tool_call", "tool": "run", "args": {"cmd": "bad"}, "error": "fail"},
        {"type": "tool_call", "tool": "run", "args": {"cmd": "good"}},
    ]
    result = evaluate_trajectory(trace, optimal_steps=1)
    assert result["recovery_rate"] == 1.0


def test_trajectory_fallo_sin_recovery():
    trace = [
        {"type": "tool_call", "tool": "run", "args": {"cmd": "bad"}, "error": "fail"},
    ]
    result = evaluate_trajectory(trace, optimal_steps=1)
    assert result["recovery_rate"] == 0.0


def test_trajectory_vacio():
    result = evaluate_trajectory([], optimal_steps=3)
    assert result == {"step_efficiency": 0.0, "redundancy": 0, "recovery_rate": 0.0}
