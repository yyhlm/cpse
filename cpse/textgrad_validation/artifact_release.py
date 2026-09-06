"""Build a review-safe, checksum-addressed public artifact bundle."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .usage_summary import summarize_run_usage


_PRIVATE_CONFIG_KEYS = {"api_key_env", "base_url", "proxy"}
_PUBLIC_RUN_DIRS = {
    "analysis", "deterministic_metrics", "judge_only", "leakage_audit",
    "prompts", "schemas", "statistics", "training",
}


def build_artifact_release(*, run_dir: Path, data_dir: Path, output_dir: Path) -> Path:
    """Create a self-contained public bundle without PDFs or service details."""
    run_dir = run_dir.resolve()
    output_dir = output_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    if output_dir == run_dir or run_dir in output_dir.parents:
        raise ValueError("Artifact output must not be inside the source run directory.")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    _copy_predictions(run_dir, output_dir)
    _copy_gold(data_dir, output_dir / "gold")
    _copy_protocol(run_dir, output_dir / "protocol")
    _copy_usage(run_dir, output_dir / "usage")
    _copy_run_artifacts(run_dir, output_dir / "run_artifacts")
    _write_readme(output_dir, run_dir.name)
    _write_checksums(output_dir)
    return output_dir


def _copy_predictions(run_dir: Path, output_dir: Path) -> None:
    for split in ("blind_test", "training"):
        split_root = run_dir / split
        if not split_root.is_dir():
            continue
        for source in split_root.rglob("prediction.json"):
            relative = source.relative_to(split_root)
            parts = list(relative.parts)
            if "documents" in parts:
                parts.remove("documents")
            destination = output_dir / "predictions" / split / Path(*parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            for companion in ("validation.json", "judge.result.json", "extraction.metadata.json"):
                companion_source = source.parent / companion
                if companion_source.is_file():
                    shutil.copy2(companion_source, destination.parent / companion)


def _copy_gold(data_dir: Path, destination: Path) -> None:
    if not data_dir.is_dir():
        return
    destination.mkdir(parents=True, exist_ok=True)
    for source in sorted(data_dir.glob("*.json")):
        shutil.copy2(source, destination / source.name)


def _copy_protocol(run_dir: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    meta = run_dir / "meta"
    for name in ("manifest.json", "checkpoint.json", "frozen_prompts.json"):
        source = meta / name
        if source.is_file():
            shutil.copy2(source, destination / name)
    snapshot = meta / "config.snapshot.json"
    if snapshot.is_file():
        value = json.loads(snapshot.read_text(encoding="utf-8"))
        (destination / "config.snapshot.public.json").write_text(
            json.dumps(_redact(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _copy_usage(run_dir: Path, destination: Path) -> None:
    ledger = run_dir / "meta" / "api_calls.jsonl"
    if not ledger.is_file():
        return
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ledger, destination / ledger.name)
    (destination / "usage_summary.json").write_text(
        json.dumps(summarize_run_usage(run_dir), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _copy_run_artifacts(run_dir: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for source in sorted(run_dir.iterdir()):
        if source.is_dir() and source.name in _PUBLIC_RUN_DIRS:
            shutil.copytree(source, destination / source.name)
        elif source.is_file() and source.suffix.lower() in {".json", ".csv", ".md", ".txt"} and source.name != "output.log":
            shutil.copy2(source, destination / source.name)
    for forbidden in list(destination.rglob("*.pdf")) + list(destination.rglob(".env")):
        forbidden.unlink()


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _redact(child) for key, child in value.items() if str(key) not in _PRIVATE_CONFIG_KEYS}
    if isinstance(value, list):
        return [_redact(child) for child in value]
    return value


def _write_readme(output_dir: Path, run_id: str) -> None:
    (output_dir / "README.md").write_text(
        f"""# Public artifact: {run_id}

This bundle contains frozen prompts/schemas, split and configuration metadata,
Gold JSON annotations, predictions, evaluation outputs, deterministic metrics,
statistical analyses, and checksums for the named experiment run.

Original PDFs are intentionally excluded because their redistribution rights
belong to the respective publishers. Reproduce PDF-dependent steps by obtaining
the cited papers independently and verifying them against the hashes in the run
manifest. Service URLs, proxies, and credential environment-variable names are
also removed from the public configuration snapshot.

`MANIFEST.json` and `SHA256SUMS.txt` make the release content-addressable. Model
names and API protocols are retained for scientific reproducibility.
""",
        encoding="utf-8",
    )


def _write_checksums(output_dir: Path) -> None:
    files = [path for path in sorted(output_dir.rglob("*")) if path.is_file() and path.name not in {"MANIFEST.json", "SHA256SUMS.txt"}]
    rows = []
    checksum_lines = []
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        relative = path.relative_to(output_dir).as_posix()
        rows.append({"path": relative, "sha256": digest, "size_bytes": path.stat().st_size})
        checksum_lines.append(f"{digest}  {relative}")
    (output_dir / "SHA256SUMS.txt").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    manifest = {
        "format_version": "public-artifact-v1",
        "file_count": len(rows),
        "files": rows,
        "exclusions": ["original PDFs", "API credentials", "service base URLs", "proxy settings", "runtime output.log"],
    }
    (output_dir / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
