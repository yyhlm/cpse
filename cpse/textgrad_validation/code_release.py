"""Build a source-only, review-safe code release for the test experiment."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import yaml


_PRIVATE_CONFIG_KEYS = {"api_key_env", "base_url", "proxy"}
_EXCLUDED_TOP_LEVEL = {"data", "results", "artifact_release", "ppt", "ppt_projects", "presentations", "casestudy", "__pycache__"}
_EXCLUDED_NAMES = {".env", ".env.example", "output.log"}
_REQUIREMENTS = "# Core experiment runtime\ntextgrad==0.1.8\nopenai\njsonschema\npyyaml\n\n# Optional optimization baselines / labeled demonstrations\ndspy\npypdf\n\n# Offline tests and statistics\npytest\nscipy\n"


def build_code_release(*, source_root: Path, output_dir: Path) -> Path:
    """Copy the self-contained experiment source while excluding data and secrets."""
    source_root = source_root.resolve()
    output_dir = output_dir.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Test source directory does not exist: {source_root}")
    if output_dir == source_root or source_root in output_dir.parents:
        raise ValueError("Code-release output must not be inside the source directory.")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    target_test = output_dir / "test"
    target_test.mkdir(parents=True)
    for source in sorted(source_root.iterdir()):
        if source.name in _EXCLUDED_TOP_LEVEL or source.name in _EXCLUDED_NAMES:
            continue
        destination = target_test / source.name
        if source.is_dir():
            shutil.copytree(source, destination, ignore=_ignore_private_runtime_files)
        elif source.is_file():
            shutil.copy2(source, destination)
    for config_path in target_test.rglob("*.yaml"):
        _redact_yaml_file(config_path)
    (output_dir / "requirements.txt").write_text(_REQUIREMENTS, encoding="utf-8")
    _write_readme(output_dir)
    _write_checksums(output_dir)
    return output_dir


def _ignore_private_runtime_files(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name in _EXCLUDED_NAMES or name in {"__pycache__", ".pytest_cache"} or name.endswith(".pdf") or name.endswith(".pyc")}


def _redact_yaml_file(path: Path) -> None:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return
    if value is not None:
        path.write_text(yaml.safe_dump(_redact(value), allow_unicode=True, sort_keys=False), encoding="utf-8")


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _redact(child) for key, child in value.items() if str(key) not in _PRIVATE_CONFIG_KEYS}
    if isinstance(value, list):
        return [_redact(child) for child in value]
    return value


def _write_readme(output_dir: Path) -> None:
    text = """# Source release: TextGrad scientific PDF extraction experiment

This package contains the full `cpse/` experiment source: implementation, prompts, sanitized configuration template, schema, and user documentation. It deliberately excludes PDFs, Gold annotations, run results, caches, environment files, credentials, service URLs, and proxy settings.

## Reproduction boundary

Install `requirements.txt`, place an authorized PDF/Gold dataset under `cpse/data/`, configure your own endpoint and credentials locally, then follow `cpse/README.md`. Published result verification should use the paired public artifact bundle, which contains frozen predictions and evaluation artifacts.

The schema and prompts are included because they define the evaluated protocol. Raw source documents are excluded because redistribution rights remain with the respective publishers.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def _write_checksums(output_dir: Path) -> None:
    files = [path for path in sorted(output_dir.rglob("*")) if path.is_file() and path.name not in {"MANIFEST.json", "SHA256SUMS.txt"}]
    rows = []
    lines = []
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        relative = path.relative_to(output_dir).as_posix()
        rows.append({"path": relative, "sha256": digest, "size_bytes": path.stat().st_size})
        lines.append(f"{digest}  {relative}")
    (output_dir / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {"format_version": "source-release-v1", "file_count": len(rows), "files": rows, "exclusions": ["datasets and Gold annotations", "original PDFs", "run results and caches", "API credentials", "service base URLs", "proxy settings"]}
    (output_dir / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
