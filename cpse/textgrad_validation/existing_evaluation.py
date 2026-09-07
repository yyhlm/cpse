from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .artifacts import sha256_file, write_csv, write_json
from .models import ExperimentConfig, PredictionArtifact
from .schema_contract import dsl_to_json_schema
from .schema_description import reinforce_required_emission
from .validator import validate_prediction


_SAFE_LABEL = re.compile(r"^[A-Za-z0-9._-]+$")


def load_existing_prediction(document_dir: Path, schema: dict[str, Any]) -> PredictionArtifact | None:
    """Reconstruct an immutable extraction artifact without calling a model."""
    for name in ("extraction.response.json", "prediction.json"):
        path = document_dir / name
        if path.is_file():
            return validate_prediction(path.read_text(encoding="utf-8"), schema)
    return None


def load_arm_schema(config: ExperimentConfig, run_dir: Path, arm: str) -> dict[str, Any]:
    dsl_path = run_dir / "blind_test" / arm / "schema.json"
    if not dsl_path.is_file():
        frozen_name = "baseline-schema.json" if arm == "baseline" else "final-best-schema.json"
        frozen_path = run_dir / "schemas" / frozen_name
        dsl_path = frozen_path if frozen_path.is_file() else config.schema_path
    dsl = reinforce_required_emission(json.loads(dsl_path.read_text(encoding="utf-8")))
    return dsl_to_json_schema(dsl)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def run_judge_only(
    config,
    run_id,
    judge,
    *,
    label,
    force=False,
    arms: tuple[str, ...] | None = None,
    document_ids: tuple[str, ...] | None = None,
):
    """Re-score frozen blind predictions into an isolated, resumable namespace."""
    if not isinstance(label, str) or not _SAFE_LABEL.fullmatch(label):
        raise ValueError("Judge-only label must contain only letters, digits, '.', '_' or '-'.")
    run_dir = config.output_root / run_id
    blind_root = run_dir / "blind_test"
    if not blind_root.is_dir():
        raise FileNotFoundError(f"Existing blind-test directory not found: {blind_root}")
    output_dir = run_dir / "judge_only" / label
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "output.log"

    def log(message: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    selected_arms = arms or ("baseline", "optimized")
    invalid_arms = sorted(set(selected_arms) - {"baseline", "optimized"})
    if invalid_arms:
        raise ValueError(f"Unsupported Judge-only arms: {invalid_arms}")
    selected_ids = set(document_ids) if document_ids else None
    arm_documents: dict[str, list[Path]] = {"baseline": [], "optimized": []}
    for arm in selected_arms:
        documents_root = blind_root / arm / "documents"
        documents = sorted(path for path in documents_root.iterdir() if path.is_dir()) if documents_root.is_dir() else []
        arm_documents[arm] = [path for path in documents if selected_ids is None or path.name in selected_ids]
    log(
        f"[judge_only] label={label} model={config.judge.model} force={force} "
        f"arms={','.join(selected_arms)} baseline={len(arm_documents['baseline'])} "
        f"optimized={len(arm_documents['optimized'])}"
    )
    by_document: dict[str, dict[str, float]] = {}
    for arm in selected_arms:
        documents_root = blind_root / arm / "documents"
        if not documents_root.is_dir():
            log(f"[judge_only/{arm}] skipped: documents directory not found")
            continue
        schema = load_arm_schema(config, run_dir, arm)
        documents = arm_documents[arm]
        for index, document_dir in enumerate(documents, start=1):
            document_id = document_dir.name
            gold_path = config.data_dir / f"{document_id}.json"
            pdf_path = config.data_dir / f"{document_id}.pdf"
            if not gold_path.is_file():
                log(f"[judge_only/{arm}] {index}/{len(documents)} {document_id}: skipped (gold missing)")
                continue
            result_path = output_dir / "documents" / document_id / f"{arm}.judge.json"
            failure_path = output_dir / "documents" / document_id / f"{arm}.judge.failure.json"
            if result_path.is_file() and not force:
                stored = json.loads(result_path.read_text(encoding="utf-8"))
                score = float(stored["score"])
                reused = True
            else:
                # --judge-force must not leave an old score looking complete if the
                # replacement Judge response is malformed or otherwise fails to parse.
                if force:
                    result_path.unlink(missing_ok=True)
                prediction = load_existing_prediction(document_dir, schema)
                if prediction is None:
                    log(f"[judge_only/{arm}] {index}/{len(documents)} {document_id}: skipped (prediction missing)")
                    continue
                result = None
                error_text = None
                for attempt in range(config.judge.max_retries + 1):
                    try:
                        result = judge.judge(
                            schema=schema,
                            prediction=prediction,
                            gold=json.loads(gold_path.read_text(encoding="utf-8")),
                            pdf_path=pdf_path,
                        )
                    except Exception as exc:
                        error_text = f"{type(exc).__name__}: {exc}"
                        if attempt < config.judge.max_retries:
                            log(f"[judge_only/{arm}] {index}/{len(documents)} {document_id}: judge retry {attempt + 1}/{config.judge.max_retries} after {type(exc).__name__}")
                        continue
                    break
                if result is None:
                    write_json(failure_path, {
                        "error": error_text,
                        "attempts": config.judge.max_retries + 1,
                        "document_id": document_id,
                        "arm": arm,
                        "judge_label": label,
                    })
                    log(f"[judge_only/{arm}] {index}/{len(documents)} {document_id}: judge FAILED ({error_text or 'unknown error'})")
                    continue
                failure_path.unlink(missing_ok=True)
                score = result.score
                write_json(result_path, {
                    **asdict(result),
                    "document_id": document_id,
                    "arm": arm,
                    "judge_label": label,
                    "source_prediction": str(document_dir.relative_to(run_dir)),
                })
                reused = False
            by_document.setdefault(document_id, {})[arm] = score
            log(
                f"[judge_only/{arm}] {index}/{len(documents)} {document_id}: score={score:.1f}"
                + (" (reuse)" if reused else "")
            )
    rows: list[dict[str, Any]] = []
    for document_id, scores in sorted(by_document.items()):
        baseline = scores.get("baseline")
        optimized = scores.get("optimized")
        delta = optimized - baseline if baseline is not None and optimized is not None else None
        rows.append({"document_id": document_id, "baseline_score": baseline, "optimized_score": optimized, "delta": delta})
    paired = [row for row in rows if row["delta"] is not None]
    deltas = [float(row["delta"]) for row in paired]
    baseline_scores = [float(row["baseline_score"]) for row in rows if row["baseline_score"] is not None]
    optimized_scores = [float(row["optimized_score"]) for row in rows if row["optimized_score"] is not None]
    summary = {
        "source_run_id": run_id,
        "judge_label": label,
        "judge_model": config.judge.model,
        "judge_prompt_sha256": sha256_file(config.judge_system_prompt_path),
        "include_pdf": config.include_pdf,
        "include_error_locations": config.include_error_locations,
        "selected_arms": list(selected_arms),
        "selected_document_ids": list(document_ids) if document_ids else None,
        "baseline_document_count": len(baseline_scores),
        "optimized_document_count": len(optimized_scores),
        "paired_count": len(paired),
        "baseline_mean": _mean(baseline_scores),
        "optimized_mean": _mean(optimized_scores),
        "mean_paired_delta": _mean(deltas),
        "wins": sum(value > 0 for value in deltas),
        "ties": sum(value == 0 for value in deltas),
        "losses": sum(value < 0 for value in deltas),
    }
    write_csv(output_dir / "documents.csv", rows, ["document_id", "baseline_score", "optimized_score", "delta"])
    write_json(output_dir / "summary.json", summary)
    def display(value: float | None) -> str:
        return f"{value:.1f}" if value is not None else "none"
    log(
        f"[judge_only] paired={summary['paired_count']} "
        f"baseline_mean={display(summary['baseline_mean'])} "
        f"optimized_mean={display(summary['optimized_mean'])} "
        f"mean_paired_delta={display(summary['mean_paired_delta'])} "
        f"wins/ties/losses={summary['wins']}/{summary['ties']}/{summary['losses']}"
    )
    return output_dir

