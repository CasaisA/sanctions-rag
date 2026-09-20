"""The planner and critic are the parts of the agent worth pinning down."""
from __future__ import annotations

from sanctions_rag.agent import critique, plan
from sanctions_rag.rag import Evidence


def _ev(caption: str) -> Evidence:
    return Evidence("id-" + caption[:4], caption, "Organization", ["ru"], ["sanction"], ["test"], 1.0, caption)


def test_plan_drops_leading_question_words():
    subs = plan("Which Russian banks are sanctioned and who owns them?")
    assert not any(s.lower().startswith("which") for s in subs)
    assert any("Russian" in s for s in subs)


def test_plan_keeps_quoted_spans_verbatim():
    assert "Gazprombank Joint Stock Company" in plan('Who owns "Gazprombank Joint Stock Company"?')


def test_plan_is_bounded():
    assert len(plan("A B C D E F G H I J K L")) <= 4


def test_critique_fails_without_evidence():
    ok, reason, follow = critique("", [], ["Gazprombank"])
    assert not ok and "no evidence" in reason and follow == []


def test_critique_asks_for_missing_subquery():
    ok, reason, follow = critique("x", [_ev("Gazprombank")], ["Gazprombank", "Rosneft"])
    assert not ok and "Rosneft" in reason and follow


def test_critique_passes_when_all_covered():
    ev = [_ev("Gazprombank"), _ev("Rosneft holding"), _ev("Transneft group")]
    ok, reason, follow = critique("x", ev, ["Gazprombank", "Rosneft"])
    assert ok and follow == []
