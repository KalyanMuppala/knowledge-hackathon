"""Small wrapper around the Anthropic Messages API.

The model only phrases answers from evidence we already picked and annotated, so a
fast mid-size model is enough. If the API rejects a parameter (e.g. a model that
doesn't support turning thinking off), we retry with fewer options instead of
failing the question; any other error returns None and the caller falls back to
the rule-based answer.

Env vars: ANTHROPIC_API_KEY (required for this path), KS_MODEL (default
claude-sonnet-5-5), KS_LLM_TIMEOUT (seconds, default 15). Worst case is two
attempts, which stays well under the harness' 60 s per-answer limit.
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Optional

DEFAULT_MODEL = "claude-sonnet-5-5"

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confident": {"type": "boolean"},
        "sources": {"type": "array", "items": {"type": "integer"}},
        "conflict_note": {"type": "string"},
    },
    "required": ["answer", "confident", "sources", "conflict_note"],
    "additionalProperties": False,
}


def _speed_params(model: str) -> dict:
    """Keep latency down: low effort, and no extended thinking where the model allows it."""
    m = model.lower()
    if "haiku-4-5" in m:
        return {}
    params: dict = {"effort": "low"}
    if "sonnet-5-5" in m:
        params["thinking"] = {"type": "between_tools"}     # 5.5 rejects "disabled"
    elif "haiku-5" in m or m.endswith("sonnet-5"):
        params["thinking"] = {"type": "disabled"}
    return params                   # Opus 5.x / Fable: thinking is always on, effort low


class LLM:
    def __init__(self) -> None:
        self.model = os.getenv("KS_MODEL", DEFAULT_MODEL)
        self.timeout = float(os.getenv("KS_LLM_TIMEOUT", "15"))
        self.client = None
        self.error: Optional[str] = None
        self._level = 0             # 0 = all options, 1 = no thinking/effort, 2 = plain JSON
        self._warned = False
        if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
            self.error = "ANTHROPIC_API_KEY not set"
            return
        try:
            import anthropic
            self.client = anthropic.Anthropic(timeout=self.timeout, max_retries=1)
        except Exception as e:      # package missing, bad config...
            self._note(e)

    @property
    def available(self) -> bool:
        return self.client is not None

    def _note(self, e: Exception) -> None:
        self.error = f"{type(e).__name__}: {str(e)[:300]}"
        if not self._warned:
            print(f"[ks] LLM call failed, using rule-based answers instead: {self.error}", file=sys.stderr)
            self._warned = True

    def _request(self, level: int, system: str, user: str, max_tokens: int) -> dict:
        req: dict = {"model": self.model, "max_tokens": max_tokens, "system": system,
                     "messages": [{"role": "user", "content": user}]}
        if level <= 1:
            req["output_config"] = {"format": {"type": "json_schema", "schema": ANSWER_SCHEMA}}
        if level == 0:
            sp = _speed_params(self.model)
            if "effort" in sp:
                req.setdefault("output_config", {})["effort"] = sp["effort"]
            if "thinking" in sp:
                req["thinking"] = sp["thinking"]
        if level == 2:
            req["system"] = system + "\n\nReply with the JSON object only."
        if any(x in self.model for x in ("opus-5", "fable")):
            req["max_tokens"] = max(max_tokens, 4000)       # room for always-on thinking
        return req

    def complete_json(self, system: str, user: str, max_tokens: int = 900) -> Optional[dict]:
        if not self.client:
            return None
        while self._level <= 2:
            try:
                resp = self.client.messages.create(**self._request(self._level, system, user, max_tokens))
                break
            except Exception as e:
                if getattr(e, "status_code", None) == 400 and self._level < 2:
                    self._level += 1            # drop optional parameters and try again
                    continue
                self._note(e)
                return None
        else:
            return None
        if getattr(resp, "stop_reason", None) == "refusal":
            return None
        text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")
        return parse_json(text)


def parse_json(text: str) -> Optional[dict]:
    text = text.strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None
