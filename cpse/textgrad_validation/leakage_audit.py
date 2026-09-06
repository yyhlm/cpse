"""Offline, exact-match audit for training facts in frozen run artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .artifacts import write_json, write_text


def _json_path(path: list[str | int]) -> str:
    result = "$"
    for part in path:
        result += f"[{part}]" if isinstance(part, int) else f".{part}"
    return result


def _normalise(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _is_high_risk_string(candidate: str) -> bool:
    """Keep identifiers and unusually specific strings, not general field vocabulary."""
    compact = candidate.replace(" ", "")
    if re.search(r"(?:doi\s*:\s*)?10\.\d{4,9}/\S+", candidate, flags=re.IGNORECASE):
        return True
    if len(candidate) >= 30:
        return True
    if len(compact) >= 5 and any(char.isalpha() for char in compact) and any(char.isdigit() for char in compact):
        return True
    return bool(re.fullmatch(r"[A-Z][A-Z0-9_-]{4,}", compact))


def _training_facts(value: Any, path: list[str | int] | None = None) -> list[tuple[str, str]]:
    path = [] if path is None else path
    if isinstance(value, dict):
        return [fact for key, item in value.items() for fact in _training_facts(item, [*path, key])]
    if isinstance(value, list):
        return [fact for index, item in enumerate(value) for fact in _training_facts(item, [*path, index])]
    if isinstance(value, str):
        candidate = " ".join(value.split())
        if _is_high_risk_string(candidate):
            return [(candidate, _json_path(path))]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        candidate = str(value)
        if len(candidate) >= 5:
            return [(candidate, _json_path(path))]
    return []


def _descriptions(value: Any, path: list[str | int] | None = None) -> dict[str, str]:
    path = [] if path is None else path
    found: dict[str, str] = {}
    if isinstance(value, dict):
        description = value.get("description")
        if isinstance(description, str):
            found[_json_path([*path, "description"])] = description
        for key, item in value.items():
            found.update(_descriptions(item, [*path, key]))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.update(_descriptions(item, [*path, index]))
    return found


def _without_descriptions(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _without_descriptions(item) for key, item in value.items() if key != "description"}
    if isinstance(value, list):
        return [_without_descriptions(item) for item in value]
    return value


def _final_artifacts(run_dir: Path) -> list[Path]:
    optimized = run_dir / "blind_test" / "optimized"
    paths = sorted(optimized.glob("prompt*.txt"))
    schema = optimized / "schema.json"
    if schema.exists():
        paths.append(schema)
    return paths


def audit_run_leakage(run_dir: Path) -> dict[str, Any]:
    """Return an exact-match, review-oriented audit of frozen final artifacts."""
    manifest_path = run_dir / "meta" / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("Leakage audit requires meta/manifest.json.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    training_records = [record for record in manifest.get("documents", []) if record.get("split") == "train"]
    if not training_records:
        raise RuntimeError("Leakage audit found no training documents in the manifest.")

    fact_paths: dict[str, list[str]] = {}
    canonical_values: dict[str, str] = {}
    for record in training_records:
        gold_path = Path(record["gold"]["path"])
        document_id = str(record.get("id") or gold_path.stem)
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        for value, path in _training_facts(gold):
            key = _normalise(value)
            fact_paths.setdefault(key, []).append(f"{document_id}:{path}")
            canonical_values.setdefault(key, value)

    artifacts = _final_artifacts(run_dir)
    hits: list[dict[str, Any]] = []
    for artifact in artifacts:
        text = artifact.read_text(encoding="utf-8")
        normalised_text = _normalise(text)
        for key, paths in fact_paths.items():
            if key in normalised_text:
                hits.append({
                    "artifact": str(artifact.relative_to(run_dir)).replace("\\", "/"),
                    "value": canonical_values[key],
                    "training_gold_paths": sorted(set(paths)),
                })

    initial_schema_path = Path(manifest["schema"]["path"])
    final_schema_path = run_dir / "blind_test" / "optimized" / "schema.json"
    structural_contract_equal: bool | None = None
    description_changes: list[str] = []
    if initial_schema_path.exists() and final_schema_path.exists():
        initial_schema = json.loads(initial_schema_path.read_text(encoding="utf-8"))
        final_schema = json.loads(final_schema_path.read_text(encoding="utf-8"))
        structural_contract_equal = _without_descriptions(initial_schema) == _without_descriptions(final_schema)
        initial_descriptions = _descriptions(initial_schema)
        final_descriptions = _descriptions(final_schema)
        description_changes = sorted({
            *{path for path, value in final_descriptions.items() if initial_descriptions.get(path) != value},
            *{path for path in initial_descriptions if path not in final_descriptions},
        })

    return {
        "run_id": run_dir.name,
        "training_document_ids": [str(record.get("id") or Path(record["gold"]["path"]).stem) for record in training_records],
        "training_document_count": len(training_records),
        "artifacts_scanned": [str(path.relative_to(run_dir)).replace("\\", "/") for path in artifacts],
        "high_risk_training_fact_count": len(fact_paths),
        "exact_training_fact_hits": hits,
        "structural_contract_equal": structural_contract_equal,
        "description_changes": description_changes,
        "limitations": [
            "This is an exact-string audit over identifiers, long strings, and precise numeric values from training Gold, not a semantic proof of no memorization.",
            "A hit is a review item, not proof of harmful leakage; generic scientific terms may legitimately recur.",
            "No hit does not establish that the final artifacts contain no paraphrased training-specific facts.",
        ],
    }


def write_run_leakage_audit(run_dir: Path) -> Path:
    """Write a reviewable JSON and Markdown audit under the run directory."""
    report = audit_run_leakage(run_dir)
    output_dir = run_dir / "leakage_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", report)
    lines = [
        "# Training-fact leakage audit",
        "",
        f"- Run: `{report['run_id']}`",
        f"- Training documents: {', '.join(report['training_document_ids'])}",
        f"- High-risk training facts inspected: {report['high_risk_training_fact_count']} (identifiers, long strings, and precise numeric values)",
        f"- Structural contract equal excluding descriptions: `{report['structural_contract_equal']}`",
        f"- Changed description paths: {len(report['description_changes'])}",
        "",
        "## Exact training-fact matches requiring review",
        "",
    ]
    if report["exact_training_fact_hits"]:
        lines.extend(["| Artifact | Matched value | Training Gold path(s) |", "|---|---|---|"])
        for hit in report["exact_training_fact_hits"]:
            lines.append(f"| `{hit['artifact']}` | `{hit['value']}` | {', '.join(f'`{path}`' for path in hit['training_gold_paths'])} |")
    else:
        lines.append("No exact matches were found among the high-risk training Gold leaf values.")
    lines.extend(["", "## Interpretation limits", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    write_text(output_dir / "report.md", "\n".join(lines) + "\n")
    return output_dir
