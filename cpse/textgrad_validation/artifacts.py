from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

_SECRET_KEYS = {"api_key", "authorization", "bearer", "file_data"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=_json_default)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def model_config_fingerprint(config: Any) -> str:
    data = asdict(config) if is_dataclass(config) else dict(config)
    data.pop("api_key_env", None)
    # api_protocol is a semantic fingerprint field — changing it changes manifest
    # and cache keys, preventing cross-protocol confusion.
    return sha256_json(data)


def sanitize_for_storage(value: Any) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for key, child in value.items():
            if str(key).lower() in _SECRET_KEYS:
                sanitized[str(key)] = "[REDACTED]"
            else:
                sanitized[str(key)] = sanitize_for_storage(child)
        return sanitized
    if isinstance(value, list | tuple):
        return [sanitize_for_storage(child) for child in value]
    return value


def write_json(path: Path, value: Any) -> None:
    _atomic_write(path, json.dumps(sanitize_for_storage(value), ensure_ascii=False, indent=2, default=_json_default) + "\n")


def write_text(path: Path, text: str) -> None:
    _atomic_write(path, text)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = canonical_json(sanitize_for_storage(value)) + "\n"
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(line)


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", delete=False, dir=path.parent, prefix=f".{path.name}."
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
        temporary_name = handle.name
    os.replace(temporary_name, path)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", delete=False, dir=path.parent, prefix=f".{path.name}."
    ) as handle:
        handle.write(text)
        temporary_name = handle.name
    os.replace(temporary_name, path)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")
