"""Build a source-only, review-safe code release for the test experiment."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import yaml


_PRIVATE_CONFIG_KEYS = {"api_key_env", "base_url", "proxy"}
_PUBLIC_TOP_LEVEL = {"__init__.py", "README.md", "schema.json", "tests", "textgrad_validation"}
_EXCLUDED_NAMES = {".env", ".env.example", "output.log"}
_REQUIREMENTS = """# Core experiment runtime
textgrad==0.1.8
openai
httpx
jsonschema
pyyaml
pymupdf

# External optimization baselines / labeled demonstrations
gepa==0.1.4
dspy==3.3.1
pypdf

# Offline tests and statistics
pytest
scipy
"""


def code_release_output_dir(output_root: Path) -> Path:
    """Return the public source-release directory beside the results directory."""
    return output_root.parent / "code_release"


def build_code_release(*, source_root: Path, output_dir: Path) -> Path:
    """Copy the self-contained experiment source while excluding data and secrets."""
    source_root = source_root.resolve()
    output_dir = output_dir.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Test source directory does not exist: {source_root}")
    if output_dir == source_root or (source_root in output_dir.parents and output_dir.parent != source_root):
        raise ValueError("Code-release output may only use the excluded code_release directory directly under the source root.")
    if output_dir.exists():
        for child in output_dir.iterdir():
            if child.name == ".git":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    target_package = output_dir / "cpse"
    target_package.mkdir(parents=True)
    (target_package / "data").mkdir()
    (target_package / "data" / ".gitkeep").write_text("", encoding="utf-8")
    for source in sorted(source_root.iterdir()):
        if source.name not in _PUBLIC_TOP_LEVEL or source.name in _EXCLUDED_NAMES:
            continue
        destination = target_package / source.name
        if source.is_dir():
            shutil.copytree(source, destination, ignore=_ignore_private_runtime_files)
        elif source.is_file():
            shutil.copy2(source, destination)
    for config_path in target_package.rglob("*.yaml"):
        _redact_yaml_file(config_path)
    (output_dir / "requirements.txt").write_text(_REQUIREMENTS, encoding="utf-8")
    (output_dir / ".env.example").write_text("MODEL_API_KEY=\n", encoding="utf-8")
    (output_dir / ".gitignore").write_text(
        ".env\n.venv/\n__pycache__/\n.pytest_cache/\n*.py[cod]\ncpse/data/*\n!cpse/data/.gitkeep\ncpse/results/\ncpse/artifact_release/\n",
        encoding="utf-8",
    )
    _write_readme(output_dir)
    _write_checksums(output_dir)
    return output_dir


def _ignore_private_runtime_files(_directory: str, names: list[str]) -> set[str]:
    return {
        name
        for name in names
        if name in _EXCLUDED_NAMES
        or name in {
            "__pycache__",
            ".pytest_cache",
            "test_casestudy_metrics.py",
            "test_gemini_stability.py",
            "gemini_stability.py",
            "probe_reasoning.py",
        }
        or name.endswith(".pdf")
        or name.endswith(".pyc")
    }


def _redact_yaml_file(path: Path) -> None:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return
    if value is not None:
        path.write_text(yaml.safe_dump(_redact(value), allow_unicode=True, sort_keys=False), encoding="utf-8")


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        redacted = {key: _redact(child) for key, child in value.items() if str(key) not in _PRIVATE_CONFIG_KEYS}
        if "base_url" in value:
            redacted["base_url"] = "https://api.example.com/v1"
        if "api_key_env" in value:
            redacted["api_key_env"] = "MODEL_API_KEY"
        if "proxy" in value:
            redacted["proxy"] = None
        return redacted
    if isinstance(value, list):
        return [_redact(child) for child in value]
    return value


def _write_readme(output_dir: Path) -> None:
    text = """# Contract-Preserving Scientific PDF Extraction

This repository contains the implementation, prompts, schema, sanitized experiment configurations, and offline tests for low-resource scientific PDF extraction with TextGrad. It deliberately excludes copyrighted PDFs, Gold annotations, run results, caches, credentials, private service URLs, and proxy settings.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Copy `cpse/textgrad_validation/config.example.yaml`, replace the model and endpoint placeholders, and set `MODEL_API_KEY` using `.env.example` as a template. Place an authorized PDF/Gold dataset under `cpse/data/`, then run:

```bash
python -m cpse.textgrad_validation --config <config.yaml> --run-id <run-id>
python -m pytest -q cpse/tests
```

The primary two-stage configuration is `cpse/textgrad_validation/config_two_stage2.yaml`. Detailed modes, artifact layouts, recovery commands, and ablation protocols are documented in `cpse/README.md` and `cpse/textgrad_validation/README.md`.

## Reproduction boundary

Published result verification should use the paired public artifact bundle containing frozen predictions and evaluation artifacts. This source release alone cannot reproduce PDF-dependent scores without authorized source documents and Gold annotations.

The schema and prompts are included because they define the evaluated protocol. Raw source documents are excluded because redistribution rights remain with the respective publishers.

## License

No open-source license is selected automatically. Add the intended license before making the GitHub repository public.
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
