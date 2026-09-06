from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

from .artifacts import sha256_json, write_json

_CACHE_VERSION = 1
_ROLES = {"extraction", "extraction_two_stage", "judge", "gold_audit"}
_WRITE_LOCK = threading.Lock()


class SharedCache:
    """Content-addressed storage for successful, fully validated API results."""

    def __init__(self, root: Path, model_name: str = "unknown-model"):
        self._root = root
        self._model_name = _safe_component(model_name)

    def fingerprint(self, role: str, fingerprint_inputs: dict[str, Any]) -> str:
        self._validate_role(role)
        return sha256_json({"version": _CACHE_VERSION, "role": role, "fingerprint_inputs": fingerprint_inputs})

    def get(self, role: str, fingerprint_inputs: dict[str, Any]) -> dict[str, Any] | None:
        fingerprint = self.fingerprint(role, fingerprint_inputs)
        path = self._path(role, fingerprint)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(value, dict):
            return None
        if value.get("version") != _CACHE_VERSION or value.get("role") != role or value.get("fingerprint") != fingerprint:
            return None
        if value.get("fingerprint_inputs") != fingerprint_inputs or not isinstance(value.get("result"), dict):
            return None
        if value.get("result_sha256") != sha256_json(value["result"]):
            return None
        return value

    def put(
        self,
        role: str,
        fingerprint_inputs: dict[str, Any],
        result: dict[str, Any],
        response_metadata: dict[str, Any],
        *,
        source_run_id: str,
    ) -> dict[str, Any]:
        if not isinstance(result, dict) or not result:
            raise ValueError("Shared cache only stores non-empty successful result objects.")
        fingerprint = self.fingerprint(role, fingerprint_inputs)
        entry = {
            "version": _CACHE_VERSION,
            "role": role,
            "fingerprint": fingerprint,
            "fingerprint_inputs": fingerprint_inputs,
            "result": result,
            "response_metadata": response_metadata,
            "result_sha256": sha256_json(result),
            "source_run_id": source_run_id,
        }
        with _WRITE_LOCK:
            existing = self.get(role, fingerprint_inputs)
            if existing is not None:
                return existing
            write_json(self._path(role, fingerprint), entry)
        return entry

    def materialize_hit(self, entry: dict[str, Any]) -> dict[str, Any]:
        return {
            "cache_hit": True,
            "cache_fingerprint": entry["fingerprint"],
            "cache_source_run_id": entry["source_run_id"],
            "result": entry["result"],
            "response_metadata": entry.get("response_metadata", {}),
        }

    def _path(self, role: str, fingerprint: str) -> Path:
        return self._root / self._model_name / role / fingerprint / "artifact.json"
    @staticmethod
    def _validate_role(role: str) -> None:
        if role not in _ROLES:
            raise ValueError(f"Unknown shared cache role: {role}")


def _safe_component(value: str) -> str:
    """Keep model names readable while preventing path traversal."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "unknown-model"
