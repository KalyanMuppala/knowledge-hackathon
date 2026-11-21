"""End-to-end on the harness' self-test fixture (no private data needed)."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SELFTEST = ROOT / "harness" / "selftest"


@pytest.fixture(scope="module")
def ks():
    from run_eval import load_snapshots
    import solution
    system = solution.build()
    system.llm.client = None                       # rule-based path, deterministic
    for snap in load_snapshots(SELFTEST / "snapshots"):
        system.ingest(snap)
    return system


def test_current_and_as_of(ks):
    assert "3.50%" in ks.answer("What is the 6-month certificate APY?").text
    old = ks.answer("What was the 6-month certificate APY in January 2026?", as_of="2026-01-31").text
    assert "2.95%" in old and "3.50%" not in old


def test_refuses_unlisted_product(ks):
    ans = ks.answer("What is the 60-month jumbo certificate rate?")
    assert not ans.confident and not any(ch.isdigit() for ch in ans.text)


def test_repeat_is_identical(ks):
    q = "Current APY on a six month share certificate?"
    assert ks.answer(q).text == ks.answer(q).text


def test_conflict_found_and_resolved(ks):
    conflicts = ks.conflicts()
    hit = [c for c in conflicts if "6-month" in c.topic]
    assert hit and hit[0].resolution.startswith("3.50%")
    assert {"2.95%", "3.50%"} <= {v.split()[0] for v in hit[0].values}


def test_llm_path_and_refusal_guard(ks):
    class FakeLLM:
        available = True
        def __init__(self, reply): self.reply = reply
        def complete_json(self, system, user): return self.reply

    ans = ks.answerer
    real = ans.llm
    try:
        ans.llm = FakeLLM({"answer": "The replacement fee is $2.50.", "confident": True,
                           "sources": [1], "conflict_note": ""})
        from ks.query import analyze
        res = ans.answer(analyze("How much is a lost debit card replacement?", None), ks.store.latest)
        assert res["text"] == "The replacement fee is $2.50." and res["sources"]
        ans.llm = FakeLLM({"answer": "Not listed, but call 555-0100.", "confident": False,
                           "sources": [], "conflict_note": ""})
        res = ans.answer(analyze("What is the overdraft fee?", None), ks.store.latest)
        assert not res["confident"] and not any(ch.isdigit() for ch in res["text"])
    finally:
        ans.llm = real
