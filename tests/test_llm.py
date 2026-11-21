import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "submission"))

from ks.llm import LLM, parse_json  # noqa: E402


class BadRequest(Exception):
    status_code = 400


class FakeMessages:
    def __init__(self, fail_times=0):
        self.calls = []
        self.fail_times = fail_times

    def create(self, **kw):
        self.calls.append(kw)
        if len(self.calls) <= self.fail_times:
            raise BadRequest("unsupported parameter")
        body = {"answer": "5.39% APR", "confident": True, "sources": [1], "conflict_note": ""}
        return SimpleNamespace(stop_reason="end_turn",
                               content=[SimpleNamespace(type="text", text=json.dumps(body))])


def make_llm(fake, model="claude-sonnet-5-5"):
    llm = LLM.__new__(LLM)
    llm.model, llm.timeout, llm.error, llm._level, llm._warned = model, 5, None, 0, True
    llm.client = SimpleNamespace(messages=fake)
    return llm


def test_request_shape_for_sonnet():
    fake = FakeMessages()
    out = make_llm(fake).complete_json("sys", "user")
    assert out["answer"] == "5.39% APR"
    req = fake.calls[0]
    assert req["thinking"] == {"type": "between_tools"}
    assert req["output_config"]["effort"] == "low"
    assert req["output_config"]["format"]["type"] == "json_schema"


def test_drops_optional_params_on_400():
    fake = FakeMessages(fail_times=2)
    llm = make_llm(fake)
    assert llm.complete_json("sys", "user")["confident"] is True
    assert "thinking" not in fake.calls[1] and "format" in fake.calls[1]["output_config"]
    assert "output_config" not in fake.calls[2]
    # remembers the level that worked
    llm.complete_json("sys", "again")
    assert "output_config" not in fake.calls[3]


def test_parse_json_tolerates_wrapping():
    assert parse_json('Here you go: {"answer": "x", "confident": false}')["answer"] == "x"
    assert parse_json("not json") is None
