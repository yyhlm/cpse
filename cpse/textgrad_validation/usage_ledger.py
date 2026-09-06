"""Thread-safe, append-only accounting for every physical API attempt."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_WRITE_LOCK = threading.Lock()


class ApiUsageLedger:
    """Persist one JSON object per physical request attempt.

    The ledger deliberately records unavailable usage as unavailable rather
    than as zero.  This matters for failed or interrupted requests, whose
    billing status can only be resolved from provider-side billing records.
    """

    def __init__(self, path: Path):
        self.path = path

    def record(self, event: dict[str, Any]) -> None:
        row = {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            **event,
        }
        encoded = json.dumps(row, ensure_ascii=False, sort_keys=True)
        with _WRITE_LOCK:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(encoded + "\n")


def normalized_usage(metadata: Any) -> dict[str, int]:
    """Normalize Responses, Chat Completions, and Gemini usage shapes."""
    if not isinstance(metadata, dict):
        return {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
        }
    input_tokens = int(metadata.get("input_tokens") or metadata.get("prompt_tokens") or metadata.get("promptTokenCount") or 0)
    output_tokens = int(metadata.get("output_tokens") or metadata.get("completion_tokens") or metadata.get("candidatesTokenCount") or 0)
    total_tokens = int(metadata.get("total_tokens") or metadata.get("totalTokenCount") or input_tokens + output_tokens)
    input_details = metadata.get("input_tokens_details") or metadata.get("prompt_tokens_details") or {}
    output_details = metadata.get("output_tokens_details") or metadata.get("completion_tokens_details") or {}
    cached = int(input_details.get("cached_tokens") or metadata.get("cachedContentTokenCount") or 0) if isinstance(input_details, dict) else 0
    reasoning = int(output_details.get("reasoning_tokens") or metadata.get("thoughtsTokenCount") or 0) if isinstance(output_details, dict) else 0
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning,
        "total_tokens": total_tokens,
    }
