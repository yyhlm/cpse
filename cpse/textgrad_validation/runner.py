from __future__ import annotations

import json
import os
import shutil
import threading
import time
from datetime import datetime
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .analysis import analyze_blind_document, summarize_blind_analysis
from .artifacts import (
    append_jsonl,
    model_config_fingerprint,
    sha256_file,
    sha256_json,
    write_json,
    write_text,
)
from .config import config_fingerprint, snapshot_config
from .dataset import discover_pairs, select_split
from .gold_audit import GoldAuditor
from .judge import GoldJudge, parse_judge_result
from .shared_cache import SharedCache
from .validator import prediction_or_failure_payload
from .models import AlternatingCandidate, CandidateSummary, DatasetSplit, ExperimentConfig, GoldAuditFinding, GoldAuditResult, JudgeResult, PredictionArtifact, RunManifest
from .optimizer import AlternatingSchemaDescriptionOptimizer, EvidenceRoutingTwoStageOptimizer, TwoStageOptimizer
from .reporting import write_alternating_report, write_alternating_training_summary, write_blind_test_summary, write_report, write_training_summary
from .responses_client import api_operation
from .schema_contract import dsl_to_json_schema
from .two_stage import extract_two_stage, TwoStageBatchAlignmentError, TwoStageError, TwoStageValidationError
from .concurrency import DaemonThreadPoolExecutor, RunCancelled
from .schema_description import (
    SchemaDescriptionPatchError,
    _resolve_description,
    ambiguous_candidates,
    parse_description_patch_text,
    reinforce_required_emission,
    structural_fingerprint,
    validate_patch_document,
)

# Minimum training-mean improvement over round-000 (baseline) required to justify
# running the blind test. A tie or sub-threshold gain yields a predetermined or
# near-predetermined optimized arm, so the 17-doc x 2-arm blind run is skipped to
# avoid extraction/judge API cost on an outcome within training noise.
MIN_BLIND_TEST_IMPROVEMENT = 3.0

# Kept as a named protocol family so the coverage-plan extension never changes
# dispatch, resume, or reporting behavior of the established two-stage mode.
TWO_STAGE_OPTIMIZATION_MODES = frozenset({
    "two_stage_alternating_schema_description",
    "two_stage_coverage_plan_schema_description",
    "two_stage_evidence_routing_schema_description",
})


def _is_two_stage_mode(mode: str) -> bool:
    return mode in TWO_STAGE_OPTIMIZATION_MODES


def _paired_retry_summary(records: list[dict[str, Any]]) -> dict[str, float | int | None]:
    """Return full-run paired metrics for retry progress logging."""
    valid = [
        item for item in records
        if item.get("baseline_status") == "valid"
        and item.get("optimized_status") == "valid"
        and isinstance(item.get("baseline_score"), (int, float))
        and isinstance(item.get("optimized_score"), (int, float))
    ]
    if not valid:
        return {"valid_paired": 0, "document_count": len(records), "baseline_mean": None, "optimized_mean": None, "mean_paired_delta": None}
    baseline_scores = [float(item["baseline_score"]) for item in valid]
    optimized_scores = [float(item["optimized_score"]) for item in valid]
    return {
        "valid_paired": len(valid),
        "document_count": len(records),
        "baseline_mean": sum(baseline_scores) / len(baseline_scores),
        "optimized_mean": sum(optimized_scores) / len(optimized_scores),
        "mean_paired_delta": sum(optimized - baseline for baseline, optimized in zip(baseline_scores, optimized_scores)) / len(valid),
    }


def _format_retry_metric(value: float | int | None) -> str:
    return f"{value:.1f}" if isinstance(value, (int, float)) else "?"


_PATH_SELECTION_SYSTEM_PROMPT = (
    "You choose which schema field a description-patch author intended to modify. "
    "Each entry lists candidate full DSL paths, each with its CURRENT description. "
    "Pick the single candidate whose field/description best matches the intent text. "
    "Return exactly one JSON object: "
    '{"selections":[{"index":<int>,"path":"<chosen candidate path>"}]}. '
    "Only use paths from the provided candidates; never invent or modify a path."
)


def _locked_manifest_view(manifest: dict[str, Any]) -> str:
    """SHA-256 of the manifest's run-defining core.

    The relaxed field set is ``max_iterations`` (allowed to change for round
    extension/re-selection) and ``config_fingerprint`` (which bundles prompt-file
    hashes, include_pdf, cache, train_ids, request format, etc.). ``config_fingerprint``
    is intentionally NOT compared on resume: the schema-description prompts are
    consumed only during training, and the training result is frozen into the run
    directory (``prompts/``, ``schemas/``, checkpoint) before the blind test runs,
    so a later edit to those source prompt files does not affect the frozen blind
    evaluation. The remaining fields (documents/data hashes, schema sha, model
    fingerprints, initial prompt, optimization mode, split algorithm) stay locked:
    changing them still aborts the run and requires a new run id.
    """
    view = dict(manifest)
    view.pop("max_iterations", None)
    view.pop("config_fingerprint", None)
    return sha256_json(view)


def _retry_budgets(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        role: snapshot.get(role, {}).get("max_retries")
        for role in ("extractor", "judge", "gold_audit")
        if isinstance(snapshot.get(role), dict)
    }


def _retry_proxies(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        role: snapshot.get(role, {}).get("proxy")
        for role in ("extractor", "judge", "gold_audit")
        if isinstance(snapshot.get(role), dict)
    }


def _select_best_candidate(candidates: list[CandidateSummary], *, default: CandidateSummary | None = None) -> CandidateSummary | None:
    """Pick the frozen best candidate; ties prefer the *later* iteration.

    Scans forward and accepts the last candidate whose ``mean_score`` is at
    least as high as every prior best. This matches the optimizer's online
    ``mean_score >= best_score`` acceptance rule (see ``_summarize_candidate``)
    so the frozen best equals the best the optimizer would keep live at the
    same iteration count, instead of reverting to the earliest tied candidate.
    """
    best: CandidateSummary | None = None
    best_score: float | None = None
    for candidate in candidates:
        if candidate.mean_score is None:
            continue
        if best is None or candidate.mean_score >= best_score:
            best = candidate
            best_score = candidate.mean_score
    return best if best is not None else default


def _select_best_alternating_candidate(candidates: list[AlternatingCandidate]) -> AlternatingCandidate | None:
    """Return the highest-scoring valid joint candidate; later ties win."""
    best: AlternatingCandidate | None = None
    for candidate in candidates:
        if candidate.phase != "joint" or not candidate.accepted or candidate.mean_score is None:
            continue
        if best is None or candidate.mean_score >= best.mean_score:
            best = candidate
    return best


def _round_of(candidate_id: str) -> int:
    """Return the numeric round index from an alternating candidate id."""
    prefix, _, _rest = candidate_id.partition("/")
    return int(prefix.replace("round-", "") or 0)


def _last_effective_alternating_index(candidates: list[Any]) -> int | None:
    """Index of the last accepted joint candidate with a mean score, or None.

    Accepts either dicts or ``AlternatingCandidate`` dataclass instances.
    """
    for index in range(len(candidates) - 1, -1, -1):
        candidate = candidates[index]
        if isinstance(candidate, dict):
            phase, accepted, mean = candidate.get("phase"), candidate.get("accepted"), candidate.get("mean_score")
        else:
            phase, accepted, mean = candidate.phase, candidate.accepted, candidate.mean_score
        if phase == "joint" and accepted and mean is not None:
            return index
    return None


def _deserialize_alternating_candidates(raw: Any) -> list[AlternatingCandidate]:
    if not isinstance(raw, list):
        raise RuntimeError("Alternating checkpoint candidates block is invalid.")
    candidates: list[AlternatingCandidate] = []
    for value in raw:
        try:
            candidates.append(
                AlternatingCandidate(
                    candidate_id=str(value["candidate_id"]),
                    phase=str(value["phase"]),
                    parent_candidate_id=value.get("parent_candidate_id"),
                    schema_prompt_hash=str(value["schema_prompt_hash"]),
                    extraction_prompt_hash=str(value["extraction_prompt_hash"]),
                    schema_sha256=str(value["schema_sha256"]),
                    structural_sha256=str(value["structural_sha256"]),
                    evidence_prompt_hash=str(value.get("evidence_prompt_hash", "")),
                    evidence_routing_prompt_hash=str(value.get("evidence_routing_prompt_hash", "")),
                    resolve_prompt_hash=str(value.get("resolve_prompt_hash", "")),
                    patch_sha256=value.get("patch_sha256"),
                    changed_description_paths=tuple(value.get("changed_description_paths", [])),
                    validation_status=str(value.get("validation_status", "valid")),
                    document_scores=dict(value.get("document_scores", {})),
                    mean_score=value.get("mean_score"),
                    accepted=bool(value.get("accepted", False)),
                    decision_reason=str(value.get("decision_reason", "")),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Alternating checkpoint candidate summary is invalid.") from exc
    return candidates


class ExperimentRunner:
    """Artifact-first orchestration for training, frozen blind tests, and audit."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        extractor: Any,
        judge: GoldJudge,
        auditor: GoldAuditor,
        optimizer_factory: Any,
        schema_patch_client: Any = None,
        schema_patch_system_prompt: str | None = None,
        textgrad_engine: Any = None,
    ):
        self.config = config
        self.extractor = extractor
        self.judge = judge
        self.auditor = auditor
        self.optimizer_factory = optimizer_factory
        self.schema_patch_client = schema_patch_client
        self.schema_patch_system_prompt = schema_patch_system_prompt
        self.textgrad_engine = textgrad_engine
        self._log_file: Path | None = None
        self._two_stage_api_semaphore: threading.BoundedSemaphore | None = None
        self._cancel_event = threading.Event()

    def _log(self, msg: str, end: str = "\n") -> None:
        """Print to console and append to the run log file, with timestamp."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full = f"[{ts}] {msg}"
        print(full, end=end, flush=True)
        if self._log_file is not None:
            try:
                with open(self._log_file, "a", encoding="utf-8") as handle:
                    handle.write(full + end if end == "\n" else full)
            except OSError:
                pass

    def run(
        self,
        run_id: str,
        max_iterations: int | None = None,
        smoke_blind_doc: str | None = None,
        smoke_training: bool = False,
        resume: str = "allow",
        invocation_command: str | None = None,
    ) -> Path:
        run_dir = self.config.output_root / run_id
        meta_dir = run_dir / "meta"
        if resume == "never" and run_dir.exists():
            raise RuntimeError(f"Run directory already exists and --resume=never was requested: {run_dir}")
        if resume == "require" and not (meta_dir / "manifest.json").exists():
            raise RuntimeError(f"No resumable run manifest exists for --resume=require: {run_dir}")
        self._log_file = run_dir / "output.log"
        self._cancel_event.clear()
        if _is_two_stage_mode(self.config.optimization_mode):
            self._two_stage_api_semaphore = threading.BoundedSemaphore(max(1, self.config.max_parallel_calls))
        self._log(f"[run] run_id={run_id} train_docs={self.config.train_count} max_iterations={max_iterations or self.config.max_iterations} resume={resume}")
        self._log(f"[run] output log: {self._log_file}")
        if not self.config.cache_enabled:
            self._log("[cache] disabled: forcing fresh extraction, judge, and audit calls")
        run_dir, split, schema = self.prepare_run(run_id)
        if invocation_command:
            self._log(f"[run] command: {invocation_command}")
            append_jsonl(
                run_dir / "meta" / "events.jsonl",
                {"event": "cli_invocation", "command": invocation_command},
            )
        checkpoint = self._load_checkpoint(run_dir)
        manifest_sha256 = sha256_json(json.loads((run_dir / "meta" / "manifest.json").read_text(encoding="utf-8")))
        if checkpoint and checkpoint.get("manifest_sha256") != manifest_sha256:
            raise RuntimeError("Existing run checkpoint does not match the current manifest.")
        if checkpoint is None:
            checkpoint = {"version": 1, "manifest_sha256": manifest_sha256, "stages": {}}
            self._write_checkpoint(run_dir, checkpoint)
        textgrad_log_dir = run_dir / "meta" / "textgrad_logs"
        textgrad_log_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TEXTGRAD_LOG_DIR"] = str(textgrad_log_dir)
        baseline_prompt = self.config.initial_prompt_path.read_text(encoding="utf-8").strip()
        train_records = split.train if not smoke_training else split.train[:1]

        requested_max_iterations = max_iterations or self.config.max_iterations
        if self.config.optimization_mode in {"description_only", "alternating_schema_description"}:
            try:
                return self._run_alternating(
                    run_dir, split, requested_max_iterations, smoke_blind_doc, smoke_training, checkpoint
                )
            except KeyboardInterrupt:
                self._cancel_event.set()
                raise
        if self.config.optimization_mode == "two_stage_evidence_routing_schema_description":
            try:
                return self._run_evidence_routing_two_stage(
                    run_dir, split, requested_max_iterations, smoke_blind_doc, smoke_training, checkpoint
                )
            except KeyboardInterrupt:
                self._cancel_event.set()
                raise
        if _is_two_stage_mode(self.config.optimization_mode):
            try:
                return self._run_two_stage(
                    run_dir, split, requested_max_iterations, smoke_blind_doc, smoke_training, checkpoint
                )
            except KeyboardInterrupt:
                self._cancel_event.set()
                raise
        training = checkpoint.get("training")
        training_complete = bool(checkpoint["stages"].get("training_complete"))
        n_used = checkpoint.get("max_iterations_used")
        optimized_dirty = False

        if training_complete and isinstance(training, dict):
            baseline_prompt, best_prompt, candidates = self._load_training_checkpoint(run_dir, checkpoint)
            if isinstance(n_used, int) and requested_max_iterations > n_used:
                self._log(f"[resume] max_iterations {n_used}->{requested_max_iterations}: resuming training from candidate-{n_used + 1:03d}")
                best_prompt, candidates = self._resume_training(
                    run_dir, candidates, baseline_prompt, best_prompt, n_used + 1, requested_max_iterations, train_records, schema
                )
                checkpoint["training"]["candidates"] = [asdict(c) for c in candidates]
                checkpoint["training"]["best_sha256"] = sha256_json(best_prompt)
                checkpoint["max_iterations_used"] = requested_max_iterations
                checkpoint["max_iterations_requested"] = requested_max_iterations
                optimized_dirty = True
                self._write_checkpoint(run_dir, checkpoint)
                self.freeze_prompts(run_dir, baseline_prompt, best_prompt)
            elif isinstance(n_used, int) and requested_max_iterations < n_used:
                self._log(f"[resume] max_iterations {n_used}->{requested_max_iterations}: recomputing frozen best over first {requested_max_iterations + 1} candidates")
                best_prompt, candidates = self._recompute_frozen_best(run_dir, candidates, requested_max_iterations, baseline_prompt, n_used)
                checkpoint["training"]["candidates"] = [asdict(c) for c in candidates]
                checkpoint["training"]["best_sha256"] = sha256_json(best_prompt)
                checkpoint["max_iterations_used"] = requested_max_iterations
                checkpoint["max_iterations_requested"] = requested_max_iterations
                checkpoint["stages"]["blind_optimized_complete"] = False
                checkpoint["stages"]["primary_complete"] = False
                checkpoint["stages"]["report_complete"] = False
                optimized_dirty = True
                self._write_checkpoint(run_dir, checkpoint)
                self.freeze_prompts(run_dir, baseline_prompt, best_prompt)
            else:
                self._log("[resume] reused completed training and frozen prompts")
                if not isinstance(n_used, int):
                    checkpoint["max_iterations_used"] = requested_max_iterations
                    checkpoint["max_iterations_requested"] = requested_max_iterations
                    self._write_checkpoint(run_dir, checkpoint)
        elif training_complete and not isinstance(training, dict):
            raise RuntimeError("Checkpoint marks training complete but the training block is missing.")
        else:
            def evaluate_training(candidate_id: str, prompt: str) -> dict[str, JudgeResult | None]:
                return self.evaluate_prompt(run_dir, candidate_id, prompt, train_records, schema)

            self._log(f"[train] baseline_candidate=candidate-000 baseline_prompt={baseline_prompt!r}")
            optimizer = self.optimizer_factory(evaluate_training)
            best_prompt, candidates = optimizer.optimize(
                baseline_prompt,
                requested_max_iterations,
                evaluate_candidate=evaluate_training,
            )
            self.freeze_prompts(run_dir, baseline_prompt, best_prompt)
            checkpoint["training"] = {
                "candidates": [asdict(candidate) for candidate in candidates],
                "baseline_sha256": sha256_json(baseline_prompt),
                "best_sha256": sha256_json(best_prompt),
            }
            checkpoint["stages"]["training_complete"] = True
            checkpoint["max_iterations_used"] = requested_max_iterations
            checkpoint["max_iterations_requested"] = requested_max_iterations
            self._write_checkpoint(run_dir, checkpoint)
        best_candidate = _select_best_candidate(candidates, default=None)
        for c in candidates:
            scores = ", ".join(
                f"{doc_id}={c.document_scores[doc_id]:.1f}" if c.document_scores.get(doc_id) is not None else f"{doc_id}=失败"
                for doc_id in sorted(c.document_scores)
            )
            mean = f"{c.mean_score:.2f}" if c.mean_score is not None else "无"
            self._log(f"[train] {c.candidate_id}: mean={mean} accepted={'✓' if c.accepted else '✗'} {scores}")
        self._log(f"[frozen] best={best_candidate.candidate_id if best_candidate else 'none'} best_mean={best_candidate.mean_score if best_candidate else 'none'}")
        blind_records = list(split.blind_test)
        if smoke_blind_doc:
            blind_records = [record for record in blind_records if record.pair.document_id == smoke_blind_doc]
            if len(blind_records) != 1:
                raise ValueError("--blind-doc must identify one blind-test document in the frozen split.")
        self._log(f"[blind_test] documents={len(blind_records)} (smoke={smoke_blind_doc is not None})")
        self._log(f"[blind_test] baseline extraction+judge for {len(blind_records)} documents...")
        baseline_results = self.evaluate_prompt(
            run_dir, "baseline", baseline_prompt, tuple(blind_records), schema, stage="blind_test"
        )
        checkpoint["stages"]["blind_baseline_complete"] = True
        self._write_checkpoint(run_dir, checkpoint)
        self._log(f"[blind_test] optimized extraction+judge for {len(blind_records)} documents...")
        if optimized_dirty or checkpoint["stages"].get("blind_optimized_complete") is False:
            # The frozen best changed (resumed or recomputed training) or the
            # optimized arm was never completed. Clear stale optimized artifacts
            # so they are regenerated from the current best. The shared cache
            # repopulates valid hits by prompt fingerprint, so unchanged
            # best<->document pairs stay free.
            self._clear_optimized_blind_documents(run_dir, blind_records)
        optimized_results = self.evaluate_prompt(
            run_dir, "optimized", best_prompt, tuple(blind_records), schema, stage="blind_test"
        )
        checkpoint["stages"]["blind_optimized_complete"] = True
        self._write_checkpoint(run_dir, checkpoint)
        paired: list[dict[str, Any]] = []
        for index, record in enumerate(blind_records, start=1):
            document_id = record.pair.document_id
            baseline_result = baseline_results[document_id]
            baseline_status = "valid" if baseline_result else "failed"
            optimized_result = optimized_results[document_id]
            optimized_status = "valid" if optimized_result else "failed"
            score_b = f"{baseline_result.score:.1f}" if baseline_result else "?"
            score_o = f"{optimized_result.score:.1f}" if optimized_result else "?"
            self._log(f"[blind_test] {index}/{len(blind_records)} {document_id}: 基线={score_b} 优化={score_o}")
            paired.append(
                {
                    "document_id": document_id,
                    "baseline_score": baseline_result.score if baseline_result else None,
                    "optimized_score": optimized_result.score if optimized_result else None,
                    "baseline_status": baseline_status,
                    "optimized_status": optimized_status,
                    "baseline_validation_errors": self._validation_error_count(run_dir, "baseline", document_id),
                    "optimized_validation_errors": self._validation_error_count(run_dir, "optimized", document_id),
                }
            )
            self._log(f"[blind_test]  {index}/{len(blind_records)} {document_id}: baseline={baseline_status} optimized={optimized_status}")
        audit_records = tuple(train_records) + tuple(blind_records)
        audit_results = (
            self.run_gold_audit(run_dir, audit_records, schema) if self.config.gold_audit_enabled else []
        )
        valid_paired = sum(1 for p in paired if p["baseline_status"] == "valid" and p["optimized_status"] == "valid")
        checkpoint["paired"] = paired
        checkpoint["stages"]["primary_complete"] = True
        self._write_checkpoint(run_dir, checkpoint)
        self._log(f"[gold_audit] enabled={self.config.gold_audit_enabled} records={len(audit_records)}")
        self.finish_report(run_dir, candidates, paired, audit_results)
        checkpoint["stages"]["report_complete"] = True
        self._write_checkpoint(run_dir, checkpoint)
        self._log(f"[done] run_dir={run_dir} valid_paired={valid_paired}/{len(paired)}")
        return run_dir

    def run_blind_baseline_only(self, run_id: str, *, invocation_command: str | None = None) -> Path:
        """Run one frozen direct-extraction baseline over the complete blind split."""
        run_dir = self.config.output_root / run_id
        if (run_dir / "blind_test" / "baseline" / "documents").exists():
            raise RuntimeError(f"A blind baseline already exists for run_id={run_id}; use a new run id.")
        self._log_file = run_dir / "output.log"
        self._cancel_event.clear()
        run_dir, split, schema = self.prepare_run(run_id)
        self._log(f"[blind_baseline_only] run_id={run_id} documents={len(split.blind_test)}")
        self._log(f"[blind_baseline_only] output log: {self._log_file}")
        if not self.config.cache_enabled:
            self._log("[cache] disabled: forcing fresh extraction and judge calls")
        if invocation_command:
            self._log(f"[run] command: {invocation_command}")
            append_jsonl(run_dir / "meta" / "events.jsonl", {"event": "cli_invocation", "command": invocation_command})

        baseline_prompt = self.config.initial_prompt_path.read_text(encoding="utf-8").strip()
        base_schema_dsl = reinforce_required_emission(json.loads(self.config.schema_path.read_text(encoding="utf-8")))
        self._log(f"[blind_baseline_only] direct extraction+judge for {len(split.blind_test)} documents...")
        results = self.evaluate_prompt(
            run_dir, "baseline", baseline_prompt, tuple(split.blind_test), schema,
            stage="blind_test", schema_dsl=base_schema_dsl,
        )
        documents = [
            {
                "document_id": record.pair.document_id,
                "score": results[record.pair.document_id].score if results[record.pair.document_id] is not None else None,
                "status": "valid" if results[record.pair.document_id] is not None else "failed",
                "validation_errors": self._validation_error_count(run_dir, "baseline", record.pair.document_id),
            }
            for record in split.blind_test
        ]
        scores = [float(item["score"]) for item in documents if isinstance(item["score"], (int, float))]
        summary = {
            "mode": self.config.optimization_mode if self.config.optimization_mode in {"schema_free_direct", "few_shot_direct"} else "direct_blind_baseline_only",
            "document_count": len(documents),
            "valid_count": len(scores),
            "failed_count": len(documents) - len(scores),
            "mean_score": sum(scores) / len(scores) if scores else None,
            "schema_invalid_count": sum(item["validation_errors"] > 0 for item in documents),
        }
        write_json(run_dir / "blind_baseline_documents.json", documents)
        write_json(run_dir / "blind_baseline_summary.json", summary)
        append_jsonl(run_dir / "meta" / "events.jsonl", {"event": "blind_baseline_only_complete", **summary})
        self._log(f"[blind_baseline_only] mean={_format_retry_metric(summary['mean_score'])} ({summary['valid_count']}/{summary['document_count']} docs)")
        self._log(f"[done] run_dir={run_dir} blind_baseline_only")
        return run_dir

    def report_only(self, run_id: str) -> Path:
        # Report regeneration reads only the run's own persisted artifacts, so it
        # must not be gated on the current config fingerprint matching the run
        # manifest (e.g. an explicit `api_protocol` field later added to
        # config.yaml would otherwise block a pure read-only report rebuild).
        run_dir = self.config.output_root / run_id
        self._log_file = run_dir / "output.log"
        checkpoint = self._load_checkpoint(run_dir)
        if checkpoint is None or not checkpoint["stages"].get("primary_complete"):
            raise RuntimeError("--report-only requires a primary-complete checkpoint.")
        paired = checkpoint.get("paired")
        if not isinstance(paired, list):
            raise RuntimeError("Primary checkpoint has no paired blind-test records.")
        audit_results = [
            result
            for path in (run_dir / "gold_audit").glob("*.json")
            if (result := self._load_cached_audit_result(path)) is not None
        ]
        if self.config.optimization_mode in {"description_only", "alternating_schema_description"}:
            alt = checkpoint.get("alternating")
            if not isinstance(alt, dict) or not checkpoint["stages"].get("alternating_training_complete"):
                raise RuntimeError("--report-only requires a completed alternating training checkpoint.")
            _schema_prompt, _extraction_prompt, selected_dsl, candidates = self._load_alternating_checkpoint(run_dir, alt)
            base_schema_dsl = json.loads((run_dir / "schemas" / "baseline-schema.json").read_text(encoding="utf-8"))
            self.finish_alternating_report(run_dir, candidates, paired, audit_results, base_schema_dsl, selected_dsl)
        elif self.config.optimization_mode == "two_stage_evidence_routing_schema_description":
            alt = checkpoint.get("alternating")
            if not isinstance(alt, dict) or not checkpoint["stages"].get("alternating_training_complete"):
                raise RuntimeError("--report-only requires a completed four-variable two-stage training checkpoint.")
            _schema_prompt, _evidence_prompt, _routing_prompt, _resolve_prompt, selected_dsl, candidates = self._load_evidence_routing_two_stage_checkpoint(run_dir, alt)
            base_schema_dsl = json.loads((run_dir / "schemas" / "baseline-schema.json").read_text(encoding="utf-8"))
            self.finish_alternating_report(run_dir, candidates, paired, audit_results, base_schema_dsl, selected_dsl)
        elif _is_two_stage_mode(self.config.optimization_mode):
            alt = checkpoint.get("alternating")
            if not isinstance(alt, dict) or not checkpoint["stages"].get("alternating_training_complete"):
                raise RuntimeError("--report-only requires a completed two-stage training checkpoint.")
            _schema_prompt, _extraction_prompt, selected_dsl, candidates = self._load_alternating_checkpoint(run_dir, alt)
            base_schema_dsl = json.loads((run_dir / "schemas" / "baseline-schema.json").read_text(encoding="utf-8"))
            self.finish_alternating_report(run_dir, candidates, paired, audit_results, base_schema_dsl, selected_dsl)
        else:
            _baseline, _best, candidates = self._load_training_checkpoint(run_dir, checkpoint)
            self.finish_report(run_dir, candidates, paired, audit_results)
        return run_dir

    def retry_gold_audit(self, run_id: str) -> Path:
        if not self.config.gold_audit_enabled:
            raise RuntimeError("--retry-gold-audit requires gold_audit.enabled: true.")
        run_dir, split, schema = self.prepare_run(run_id)
        checkpoint = self._load_checkpoint(run_dir)
        if checkpoint is None or not checkpoint["stages"].get("primary_complete"):
            raise RuntimeError("--retry-gold-audit requires a primary-complete checkpoint.")
        _baseline, _best, candidates = self._load_training_checkpoint(run_dir, checkpoint)
        audit_results = self.run_gold_audit(run_dir, (*split.train, *split.blind_test), schema)
        paired = checkpoint.get("paired")
        if not isinstance(paired, list):
            raise RuntimeError("Primary checkpoint has no paired blind-test records.")
        self.finish_report(run_dir, candidates, paired, audit_results)
        return run_dir

    def retry_failed_primary(self, run_id: str) -> Path:
        """Retry only failed blind-test arms while retaining frozen prompts/results.

        This recovery mode permits a changed retry budget only. It does not alter
        the frozen manifest or config snapshot; the policy transition is instead
        recorded in the run event log.
        """
        run_dir = self.config.output_root / run_id
        self._log_file = run_dir / "output.log"
        self._log(f"[retry_failed_primary] run_id={run_id}")
        run_dir, split, schema, prior_snapshot = self._prepare_retry_failed_primary(run_id)
        checkpoint = self._load_checkpoint(run_dir)
        if checkpoint is None or not checkpoint["stages"].get("primary_complete"):
            raise RuntimeError("--retry-failed-primary requires a primary-complete checkpoint.")
        mode = self.config.optimization_mode
        base_schema_dsl: dict[str, Any] | None = None
        selected_schema_dsl: dict[str, Any] | None = None
        if mode in {"description_only", "alternating_schema_description"}:
            alt = checkpoint.get("alternating")
            if not isinstance(alt, dict) or not checkpoint["stages"].get("alternating_training_complete"):
                raise RuntimeError("--retry-failed-primary requires a completed alternating training checkpoint.")
            _schema_prompt, best_prompt, selected_schema_dsl, candidates = self._load_alternating_checkpoint(run_dir, alt)
            try:
                baseline_prompt = (run_dir / "prompts" / "baseline.txt").read_text(encoding="utf-8").strip()
                base_schema_dsl = json.loads((run_dir / "schemas" / "baseline-schema.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError("Alternating retry artifacts are missing.") from exc
        elif mode == "two_stage_evidence_routing_schema_description":
            alt = checkpoint.get("alternating")
            if not isinstance(alt, dict) or not checkpoint["stages"].get("alternating_training_complete"):
                raise RuntimeError("--retry-failed-primary requires a completed four-variable two-stage checkpoint.")
            _schema_prompt, best_evidence, best_routing, best_resolve, selected_schema_dsl, candidates = self._load_evidence_routing_two_stage_checkpoint(run_dir, alt)
            try:
                baseline_evidence = (run_dir / "prompts" / "baseline.evidence.txt").read_text(encoding="utf-8").strip()
                baseline_routing = (run_dir / "prompts" / "baseline.evidence-routing.txt").read_text(encoding="utf-8").strip()
                baseline_resolve = (run_dir / "prompts" / "baseline.resolve.txt").read_text(encoding="utf-8").strip()
                base_schema_dsl = json.loads((run_dir / "schemas" / "baseline-schema.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError("Four-variable two-stage retry artifacts are missing.") from exc
        elif _is_two_stage_mode(mode):
            alt = checkpoint.get("alternating")
            if not isinstance(alt, dict) or not checkpoint["stages"].get("alternating_training_complete"):
                raise RuntimeError("--retry-failed-primary requires a completed two-stage training checkpoint.")
            _schema_prompt, best_evidence, best_resolve, selected_schema_dsl, candidates = self._load_two_stage_checkpoint(run_dir, alt)
            try:
                baseline_evidence = (run_dir / "prompts" / "baseline.evidence.txt").read_text(encoding="utf-8").strip()
                baseline_resolve = (run_dir / "prompts" / "baseline.resolve.txt").read_text(encoding="utf-8").strip()
                base_schema_dsl = json.loads((run_dir / "schemas" / "baseline-schema.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError("Two-stage retry artifacts are missing.") from exc
        else:
            baseline_prompt, best_prompt, candidates = self._load_training_checkpoint(run_dir, checkpoint)
        paired = checkpoint.get("paired")
        if not isinstance(paired, list):
            raise RuntimeError("Primary checkpoint has no paired blind-test records.")
        before_summary = _paired_retry_summary(paired)
        self._log(
            "[retry_failed_primary] before: "
            f"valid_paired={before_summary['valid_paired']}/{before_summary['document_count']} "
            f"baseline_mean={_format_retry_metric(before_summary['baseline_mean'])} "
            f"optimized_mean={_format_retry_metric(before_summary['optimized_mean'])} "
            f"mean_paired_delta={_format_retry_metric(before_summary['mean_paired_delta'])}"
        )

        by_id = {record.pair.document_id: record for record in split.blind_test}
        failed_by_arm: dict[str, list[str]] = {"baseline": [], "optimized": []}
        recovery_reasons: dict[str, dict[str, str]] = {"baseline": {}, "optimized": {}}
        for arm in ("baseline", "optimized"):
            status_key = f"{arm}_status"
            for item in paired:
                document_id = str(item["document_id"])
                if item.get(status_key) != "valid":
                    failed_by_arm[arm].append(document_id)
                    recovery_reasons[arm][document_id] = "checkpoint_status_not_valid"
                elif not self._blind_document_artifacts_complete(run_dir, arm, document_id):
                    failed_by_arm[arm].append(document_id)
                    recovery_reasons[arm][document_id] = "required_artifact_missing"
        for arm, document_ids in failed_by_arm.items():
            unknown = sorted(set(document_ids) - set(by_id))
            if unknown:
                raise RuntimeError(f"Primary checkpoint references documents outside the frozen split: {unknown}")
            self._clear_blind_documents(run_dir, arm, document_ids)

        current_snapshot = snapshot_config(self.config)
        append_jsonl(
            run_dir / "meta" / "events.jsonl",
            {
                "event": "retry_failed_primary_started",
                "retry_policy_before": _retry_budgets(prior_snapshot),
                "retry_policy_after": _retry_budgets(current_snapshot),
                "proxy_before": _retry_proxies(prior_snapshot),
                "proxy_after": _retry_proxies(current_snapshot),
                "failed_documents": failed_by_arm,
                "recovery_reasons": recovery_reasons,
            },
        )
        failed_baseline_records = tuple(by_id[item] for item in failed_by_arm["baseline"])
        failed_optimized_records = tuple(by_id[item] for item in failed_by_arm["optimized"])
        if mode in {"description_only", "alternating_schema_description"}:
            assert base_schema_dsl is not None and selected_schema_dsl is not None
            baseline_results = self.evaluate_prompt(
                run_dir, "baseline", baseline_prompt, failed_baseline_records, None, stage="blind_test", schema_dsl=base_schema_dsl
            )
            optimized_results = self.evaluate_prompt(
                run_dir, "optimized", best_prompt, failed_optimized_records, None, stage="blind_test", schema_dsl=selected_schema_dsl
            )
            validation_error_count = lambda arm, document_id: self._validation_error_count(run_dir, arm, document_id)
        elif mode == "two_stage_evidence_routing_schema_description":
            assert base_schema_dsl is not None and selected_schema_dsl is not None
            baseline_results = self._evaluate_two_stage(
                run_dir, "blind_test", "baseline", baseline_evidence, baseline_resolve, failed_baseline_records, base_schema_dsl, evidence_routing_prompt=baseline_routing
            )
            optimized_results = self._evaluate_two_stage(
                run_dir, "blind_test", "optimized", best_evidence, best_resolve, failed_optimized_records, selected_schema_dsl, evidence_routing_prompt=best_routing
            )
            validation_error_count = lambda arm, document_id: self._two_stage_validation_error_count(run_dir, f"blind_test/{arm}", document_id)
        elif _is_two_stage_mode(mode):
            assert base_schema_dsl is not None and selected_schema_dsl is not None
            baseline_results = self._evaluate_two_stage(
                run_dir, "blind_test", "baseline", baseline_evidence, baseline_resolve, failed_baseline_records, base_schema_dsl
            )
            optimized_results = self._evaluate_two_stage(
                run_dir, "blind_test", "optimized", best_evidence, best_resolve, failed_optimized_records, selected_schema_dsl
            )
            validation_error_count = lambda arm, document_id: self._two_stage_validation_error_count(run_dir, f"blind_test/{arm}", document_id)
        else:
            baseline_results = self.evaluate_prompt(
                run_dir, "baseline", baseline_prompt, failed_baseline_records, schema, stage="blind_test"
            )
            optimized_results = self.evaluate_prompt(
                run_dir, "optimized", best_prompt, failed_optimized_records, schema, stage="blind_test"
            )
            validation_error_count = lambda arm, document_id: self._validation_error_count(run_dir, arm, document_id)
        refreshed: list[dict[str, Any]] = []
        for item in paired:
            document_id = str(item["document_id"])
            updated = dict(item)
            if document_id in baseline_results:
                result = baseline_results[document_id]
                updated["baseline_score"] = result.score if result is not None else None
                updated["baseline_status"] = "valid" if result is not None else "failed"
            if document_id in optimized_results:
                result = optimized_results[document_id]
                updated["optimized_score"] = result.score if result is not None else None
                updated["optimized_status"] = "valid" if result is not None else "failed"
            updated["baseline_validation_errors"] = validation_error_count("baseline", document_id)
            updated["optimized_validation_errors"] = validation_error_count("optimized", document_id)
            refreshed.append(updated)
        checkpoint["paired"] = refreshed
        prior_by_id = {str(item["document_id"]): item for item in paired}
        refreshed_by_id = {str(item["document_id"]): item for item in refreshed}
        for document_id in failed_by_arm["baseline"]:
            prior = prior_by_id[document_id]
            current = refreshed_by_id[document_id]
            self._log(
                f"[retry_failed_primary] {document_id}: "
                f"baseline={_format_retry_metric(prior.get('baseline_score'))} -> {_format_retry_metric(current.get('baseline_score'))} "
                f"optimized={_format_retry_metric(current.get('optimized_score'))}"
            )
        for document_id in failed_by_arm["optimized"]:
            prior = prior_by_id[document_id]
            current = refreshed_by_id[document_id]
            self._log(
                f"[retry_failed_primary] {document_id}: "
                f"baseline={_format_retry_metric(current.get('baseline_score'))} "
                f"optimized={_format_retry_metric(prior.get('optimized_score'))} -> {_format_retry_metric(current.get('optimized_score'))}"
            )
        after_summary = _paired_retry_summary(refreshed)
        self._log(
            "[retry_failed_primary] after: "
            f"valid_paired={after_summary['valid_paired']}/{after_summary['document_count']} "
            f"baseline_mean={_format_retry_metric(after_summary['baseline_mean'])} "
            f"optimized_mean={_format_retry_metric(after_summary['optimized_mean'])} "
            f"mean_paired_delta={_format_retry_metric(after_summary['mean_paired_delta'])}"
        )
        checkpoint["stages"]["blind_baseline_complete"] = True
        checkpoint["stages"]["blind_optimized_complete"] = True
        checkpoint["stages"]["primary_complete"] = True
        self._write_checkpoint(run_dir, checkpoint)
        audit_results = [
            result
            for path in (run_dir / "gold_audit").glob("*.json")
            if (result := self._load_cached_audit_result(path)) is not None
        ]
        if mode in {"description_only", "alternating_schema_description", "two_stage_evidence_routing_schema_description"} or _is_two_stage_mode(mode):
            assert base_schema_dsl is not None and selected_schema_dsl is not None
            self.finish_alternating_report(run_dir, candidates, refreshed, audit_results, base_schema_dsl, selected_schema_dsl)
        else:
            self.finish_report(run_dir, candidates, refreshed, audit_results)
        checkpoint["stages"]["report_complete"] = True
        self._write_checkpoint(run_dir, checkpoint)
        append_jsonl(
            run_dir / "meta" / "events.jsonl",
            {
                "event": "retry_failed_primary_complete",
                "valid_paired": sum(1 for item in refreshed if item.get("baseline_status") == "valid" and item.get("optimized_status") == "valid"),
                "total_paired": len(refreshed),
            },
        )
        return run_dir

    def _load_checkpoint(self, run_dir: Path) -> dict[str, Any] | None:
        path = run_dir / "meta" / "checkpoint.json"
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Existing run checkpoint is unreadable: {path}") from exc
        if not isinstance(value, dict) or value.get("version") != 1 or not isinstance(value.get("stages"), dict):
            raise RuntimeError(f"Existing run checkpoint has an unsupported format: {path}")
        return value

    def _write_checkpoint(self, run_dir: Path, checkpoint: dict[str, Any]) -> None:
        write_json(run_dir / "meta" / "checkpoint.json", checkpoint)

    def _load_training_checkpoint(
        self, run_dir: Path, checkpoint: dict[str, Any]
    ) -> tuple[str, str, list[CandidateSummary]]:
        training = checkpoint.get("training")
        if not isinstance(training, dict) or not isinstance(training.get("candidates"), list):
            raise RuntimeError("Training checkpoint is incomplete.")
        baseline_path = run_dir / "prompts" / "baseline.txt"
        best_path = run_dir / "prompts" / "final-best.txt"
        try:
            baseline = baseline_path.read_text(encoding="utf-8").strip()
            best = best_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise RuntimeError("Training checkpoint prompt artifacts are missing.") from exc
        if sha256_json(baseline) != training.get("baseline_sha256") or sha256_json(best) != training.get("best_sha256"):
            raise RuntimeError("Training checkpoint prompt artifacts do not match their recorded hashes.")
        try:
            candidates = [
                CandidateSummary(
                    candidate_id=str(value["candidate_id"]),
                    prompt_hash=str(value["prompt_hash"]),
                    parent_candidate_id=value.get("parent_candidate_id"),
                    document_scores=dict(value["document_scores"]),
                    mean_score=value.get("mean_score"),
                    accepted=bool(value["accepted"]),
                    decision_reason=str(value["decision_reason"]),
                )
                for value in training["candidates"]
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Training checkpoint candidate summaries are invalid.") from exc
        return baseline, best, candidates

    def prepare_run(self, run_id: str) -> tuple[Path, DatasetSplit, dict[str, Any]]:
        run_dir = self.config.output_root / run_id
        meta_dir = run_dir / "meta"
        pairs = discover_pairs(self.config.data_dir)
        split = select_split(pairs, self.config.train_count, self.config.split_algorithm_version, train_ids=self.config.train_ids)
        schema_text = self.config.schema_path.read_text(encoding="utf-8")
        manifest = self._manifest(split, schema_text)
        manifest_path = meta_dir / "manifest.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            # ``max_iterations`` is the only relaxable field (excluded from
            # ``config_fingerprint`` and from this comparison), so a changed
            # ``max_iterations`` does NOT abort. Every other manifest field is
            # compared via a locked view that strips ``max_iterations``.
            if _locked_manifest_view(existing) != _locked_manifest_view(manifest):
                raise RuntimeError("Existing run manifest does not match current inputs or configuration.")
        else:
            write_json(manifest_path, manifest)
            write_json(meta_dir / "config.snapshot.json", snapshot_config(self.config))
        return run_dir, split, dsl_to_json_schema(reinforce_required_emission(json.loads(schema_text)))

    def _prepare_retry_failed_primary(self, run_id: str) -> tuple[Path, DatasetSplit, dict[str, Any], dict[str, Any]]:
        """Validate immutable inputs while allowing only retry-budget drift."""
        run_dir = self.config.output_root / run_id
        meta_dir = run_dir / "meta"
        manifest_path = meta_dir / "manifest.json"
        snapshot_path = meta_dir / "config.snapshot.json"
        if not manifest_path.is_file() or not snapshot_path.is_file():
            raise RuntimeError("--retry-failed-primary requires an existing manifest and config snapshot.")
        pairs = discover_pairs(self.config.data_dir)
        split = select_split(pairs, self.config.train_count, self.config.split_algorithm_version, train_ids=self.config.train_ids)
        schema_text = self.config.schema_path.read_text(encoding="utf-8")
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        current_manifest = self._manifest(split, schema_text)
        existing_view = dict(existing_manifest)
        current_view = dict(current_manifest)
        for view in (existing_view, current_view):
            view.pop("max_iterations", None)
            view.pop("config_fingerprint", None)
            view.pop("model_fingerprints", None)
        if sha256_json(existing_view) != sha256_json(current_view):
            raise RuntimeError("Existing run manifest does not match immutable inputs for --retry-failed-primary.")
        try:
            prior_snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Existing config snapshot is unreadable: {snapshot_path}") from exc
        if not isinstance(prior_snapshot, dict):
            raise RuntimeError(f"Existing config snapshot has an unsupported format: {snapshot_path}")
        return run_dir, split, dsl_to_json_schema(reinforce_required_emission(json.loads(schema_text))), prior_snapshot

    def run_gold_audit(
        self, run_dir: Path, records: tuple[Any, ...] | list[Any], schema: dict[str, Any]
    ) -> list[GoldAuditResult]:
        results: list[GoldAuditResult] = []
        for record in records:
            artifact_path = run_dir / "gold_audit" / f"{record.pair.document_id}.json"
            cached = self._load_cached_audit_result(artifact_path) if self.config.cache_enabled else None
            if cached is not None:
                results.append(cached)
                continue
            cache = SharedCache(self.config.output_root / "_cache", self.config.gold_audit.model)
            audit_inputs = self._audit_cache_inputs(record, schema)
            shared = cache.get("gold_audit", audit_inputs) if self.config.cache_enabled else None
            if shared is not None:
                try:
                    result = GoldAuditResult(
                        findings=tuple(GoldAuditFinding(**finding) for finding in shared["result"]["findings"])
                    )
                except (KeyError, TypeError, ValueError):
                    shared = None
                else:
                    write_json(artifact_path, {"status": "complete", **cache.materialize_hit(shared)})
                    results.append(result)
                    continue
            try:
                result = self.auditor.audit(pdf_path=record.pair.pdf_path, schema=schema, gold=record.pair.gold)
            except Exception as exc:
                write_json(artifact_path, {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
                append_jsonl(
                    run_dir / "meta" / "events.jsonl",
                    {"event": "gold_audit_failed", "document_id": record.pair.document_id, "artifact": str(artifact_path.relative_to(run_dir))},
                )
                continue
            if self.config.cache_enabled:
                cache_entry = cache.put("gold_audit", audit_inputs, asdict(result), {}, source_run_id=run_dir.name)
                cache_provenance = {"cache_hit": False, "cache_fingerprint": cache_entry["fingerprint"]}
            else:
                cache_provenance = {"cache_hit": False}
            write_json(artifact_path, {"status": "complete", "result": asdict(result), **cache_provenance})
            append_jsonl(
                run_dir / "meta" / "events.jsonl",
                {"event": "gold_audit_complete", "document_id": record.pair.document_id, "artifact": str(artifact_path.relative_to(run_dir))},
            )
            results.append(result)
        return results

    def _load_cached_audit_result(self, artifact_path: Path) -> GoldAuditResult | None:
        if not artifact_path.is_file():
            return None
        try:
            raw = json.loads(artifact_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and raw.get("status") == "complete":
                return GoldAuditResult(
                    findings=tuple(
                        GoldAuditFinding(**finding) for finding in raw["result"]["findings"]
                    )
                )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
        return None

    def evaluate_prompt(
        self,
        run_dir: Path,
        candidate_id: str,
        prompt: str,
        records: tuple[Any, ...],
        schema: dict[str, Any],
        stage: str = "training",
        *,
        schema_dsl: dict[str, Any] | None = None,
    ) -> dict[str, JudgeResult | None]:
        """Evaluate ``prompt`` over ``records`` under a schema.

        ``schema`` is the converted JSON Schema. When ``schema_dsl`` is provided
        (alternating mode), it overrides ``schema``: the candidate DSL is
        converted for validation/judging, its text (descriptions included) is
        what the extractor sends to the model, and both cache keys hash the DSL
        so a description-only change invalidates extraction/judge caches.
        """
        if schema_dsl is None and (self.config.optimization_mode == "single" or _is_two_stage_mode(self.config.optimization_mode)):
            # Keep the single-prompt ablation's baseline identical to alternating
            # mode's round-000: both must send and hash the reinforced base DSL.
            # The only remaining experimental variable is whether descriptions
            # are subsequently optimized.
            schema_dsl = reinforce_required_emission(json.loads(self.config.schema_path.read_text(encoding="utf-8")))
        if schema_dsl is not None:
            schema = dsl_to_json_schema(schema_dsl)
            schema_text = json.dumps(schema_dsl, ensure_ascii=False, sort_keys=True)
        else:
            schema_text = None
        write_text(run_dir / stage / candidate_id / "prompt.txt", prompt)
        def evaluate_document(record: Any) -> tuple[str, JudgeResult | None]:
            document_id = record.pair.document_id
            document_dir = run_dir / stage / candidate_id / "documents" / document_id
            cached_result = self._load_cached_judge_result(document_dir) if self.config.cache_enabled else None
            if cached_result is not None:
                return document_id, cached_result
            extraction_inputs = self._extraction_cache_inputs(record, schema, prompt, schema_dsl=schema_dsl)
            cache = SharedCache(self.config.output_root / "_cache", self.config.extractor.model)
            cached_extraction = cache.get("extraction", extraction_inputs) if self.config.cache_enabled else None
            if cached_extraction is not None:
                prediction, metadata = self._prediction_from_cache(cached_extraction)
                self._write_prediction_artifacts(document_dir, prediction, metadata, cache.materialize_hit(cached_extraction))
            else:
                try:
                    if schema_dsl is not None:
                        prediction, metadata = self.extractor.extract(record.pair.pdf_path, prompt, schema_text=schema_text, schema=schema)
                    else:
                        prediction, metadata = self.extractor.extract(record.pair.pdf_path, prompt)
                except Exception as exc:
                    write_json(document_dir / "extraction.failure.json", {"error": f"{type(exc).__name__}: {exc}"})
                    self._log(f"  [{stage}/{candidate_id}] {document_id}: extraction FAILED ({type(exc).__name__})")
                    return document_id, None
                if self.config.cache_enabled:
                    cache_entry = cache.put(
                        "extraction",
                        extraction_inputs,
                        self._prediction_cache_payload(prediction),
                        metadata,
                        source_run_id=run_dir.name,
                    )
                    cache_provenance = {"cache_hit": False, "cache_fingerprint": cache_entry["fingerprint"]}
                else:
                    cache_provenance = {"cache_hit": False}
                self._write_prediction_artifacts(document_dir, prediction, metadata, cache_provenance)
            if not prediction.is_valid:
                first = prediction.validation_errors[0] if prediction.validation_errors else {}
                self._log(
                    f"  [warn] [{stage}/{candidate_id}] {document_id}: prediction FAILED schema validation "
                    f"({len(prediction.validation_errors)} errors, first={first.get('path', '?')}) — score is not trustworthy"
                )
            judge_inputs = self._judge_cache_inputs(record, schema, prediction, schema_dsl=schema_dsl)
            judge_cache = SharedCache(self.config.output_root / "_cache", self.config.judge.model)
            cached_judge = judge_cache.get("judge", judge_inputs) if self.config.cache_enabled else None
            if cached_judge is not None:
                try:
                    result = parse_judge_result(
                        json.dumps(cached_judge["result"]), include_error_locations=self.config.include_error_locations, include_pdf=self.config.include_pdf
                    )
                except (KeyError, TypeError, ValueError):
                    cached_judge = None
                else:
                    write_json(document_dir / "judge.result.json", {**asdict(result), **judge_cache.materialize_hit(cached_judge)})
                    self._log(f"  [{stage}/{candidate_id}] {document_id}: judge score={result.score:.1f} (cache)")
                    return document_id, result
            try:
                result = self.judge.judge(
                    schema=schema,
                    prediction=prediction,
                    gold=record.pair.gold,
                    pdf_path=record.pair.pdf_path,
                )
            except Exception as exc:
                first_error = f"{type(exc).__name__}: {exc}"
                result = None
                for retry in range(1, self.config.judge.max_retries + 1):
                    try:
                        result = self.judge.judge(
                            schema=schema,
                            prediction=prediction,
                            gold=record.pair.gold,
                            pdf_path=record.pair.pdf_path,
                        )
                    except Exception:
                        continue
                    else:
                        self._log(f"  [{stage}/{candidate_id}] {document_id}: judge recovered on retry {retry}, score={result.score:.1f}")
                        break
                if result is None:
                    write_json(document_dir / "judge.failure.json", {"error": first_error})
                    self._log(f"  [{stage}/{candidate_id}] {document_id}: judge FAILED ({first_error.split(':')[0]})")
            if result is not None:
                if self.config.cache_enabled:
                    cache_entry = judge_cache.put("judge", judge_inputs, asdict(result), {}, source_run_id=run_dir.name)
                    cache_provenance = {"cache_hit": False, "cache_fingerprint": cache_entry["fingerprint"]}
                else:
                    cache_provenance = {"cache_hit": False}
                write_json(document_dir / "judge.result.json", {**asdict(result), **cache_provenance})
                self._log(f"  [{stage}/{candidate_id}] {document_id}: judge score={result.score:.1f}")
            return document_id, result

        workers = min(self.config.max_parallel_calls, len(records))
        if workers <= 1:
            evaluated = (evaluate_document(record) for record in records)
        else:
            # Poll future.done() with a short sleep so the main thread returns to
            # Python bytecode regularly, making Ctrl+C (SIGINT) responsive on
            # Windows. A blocking as_completed + future.result() sleeps the main
            # thread in a C-level condition variable where KeyboardInterrupt
            # cannot be delivered until a worker completes and wakes the waiter.
            executor = DaemonThreadPoolExecutor(max_workers=workers, thread_name_prefix="textgrad-eval")
            futures = {executor.submit(evaluate_document, record): record for record in records}
            evaluated: list[tuple[str, Any]] = []
            pending = set(futures)
            try:
                while pending:
                    done = {f for f in pending if f.done()}
                    if done:
                        for f in done:
                            evaluated.append(f.result())
                        pending -= done
                    else:
                        time.sleep(0.2)
            except KeyboardInterrupt:
                self._cancel_event.set()
                for future in futures:
                    future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            finally:
                executor.shutdown(wait=False)
        result_map = dict(evaluated)
        valid_scores = [r.score for r in result_map.values() if r is not None]
        if valid_scores:
            self._log(f"  [{stage}/{candidate_id}] mean={sum(valid_scores)/len(valid_scores):.1f} ({len(valid_scores)}/{len(records)} docs)")
        return result_map

    def _audit_cache_inputs(self, record: Any, schema: dict[str, Any]) -> dict[str, Any]:
        return {
            "pdf_sha256": record.pair.pdf_sha256,
            "gold_sha256": record.pair.gold_sha256,
            "converted_schema_sha256": sha256_json(schema),
            "audit_system_prompt_sha256": sha256_file(self.config.gold_audit_system_prompt_path),
            "model_config_sha256": model_config_fingerprint(self.config.gold_audit),
            "request_format_version": self.config.request_format_version,
            "retry_budget": self.config.gold_audit.max_retries,
        }

    def _extraction_cache_inputs(self, record: Any, schema: dict[str, Any], prompt: str, *, schema_dsl: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "pdf_sha256": record.pair.pdf_sha256,
            "raw_schema_sha256": sha256_json(schema_dsl) if schema_dsl is not None else sha256_file(self.config.schema_path),
            "converted_schema_sha256": sha256_json(schema),
            "extraction_system_prompt_sha256": sha256_file(self.config.extraction_system_prompt_path),
            "extraction_prompt_sha256": sha256_json(prompt),
            "model_config_sha256": model_config_fingerprint(self.config.extractor),
            "request_format_version": self.config.request_format_version,
            "retry_budget": self.config.extractor.max_retries,
        }

    def _judge_cache_inputs(self, record: Any, schema: dict[str, Any], prediction: PredictionArtifact, *, schema_dsl: dict[str, Any] | None = None) -> dict[str, Any]:
        inputs = {
            "prediction_sha256": sha256_json(prediction_or_failure_payload(prediction)),
            "gold_sha256": record.pair.gold_sha256,
            "converted_schema_sha256": sha256_json(schema),
            # The DSL hash (descriptions included) invalidates the judge cache
            # when the schema the extractor used changes, even if the prediction
            # text happens to be identical.
            "raw_schema_sha256": sha256_json(schema_dsl) if schema_dsl is not None else sha256_file(self.config.schema_path),
            "judge_system_prompt_sha256": sha256_file(self.config.judge_system_prompt_path),
            "model_config_sha256": model_config_fingerprint(self.config.judge),
            "include_error_locations": self.config.include_error_locations,
            "include_pdf": self.config.include_pdf,
            "request_format_version": self.config.request_format_version,
            "retry_budget": self.config.judge.max_retries,
        }
        if self.config.include_pdf:
            inputs["pdf_sha256"] = record.pair.pdf_sha256
        return inputs

    @staticmethod
    def _prediction_cache_payload(prediction: PredictionArtifact) -> dict[str, Any]:
        return {
            "raw_response": prediction.raw_response,
            "parsed_prediction": prediction.parsed_prediction,
            "validation_errors": list(prediction.validation_errors),
        }

    @staticmethod
    def _prediction_from_cache(entry: dict[str, Any]) -> tuple[PredictionArtifact, dict[str, Any]]:
        payload = entry["result"]
        prediction = PredictionArtifact(
            raw_response=str(payload["raw_response"]),
            parsed_prediction=payload.get("parsed_prediction"),
            validation_errors=tuple(payload.get("validation_errors", [])),
        )
        return prediction, dict(entry.get("response_metadata", {}))

    @staticmethod
    def _write_prediction_artifacts(
        document_dir: Path,
        prediction: PredictionArtifact,
        metadata: dict[str, Any],
        provenance: dict[str, Any],
    ) -> None:
        raw_response = prediction.raw_response
        try:
            raw_response = json.dumps(json.loads(raw_response), ensure_ascii=False, indent=2) + "\n"
        except (json.JSONDecodeError, TypeError, ValueError):
            pass  # not valid JSON — keep the raw text verbatim for audit
        write_text(document_dir / "extraction.response.json", raw_response)
        write_json(document_dir / "validation.json", {"errors": list(prediction.validation_errors)})
        if prediction.is_valid:
            write_json(document_dir / "prediction.json", prediction.parsed_prediction)
        write_json(document_dir / "extraction.metadata.json", {**metadata, "cache": provenance})

    def _load_cached_judge_result(self, document_dir: Path) -> JudgeResult | None:
        """Reuse a completed document evaluation when resuming an interrupted run."""
        result_path = document_dir / "judge.result.json"
        if not result_path.is_file():
            return None
        try:
            raw = result_path.read_text(encoding="utf-8")
            return parse_judge_result(raw, include_error_locations=self.config.include_error_locations, include_pdf=self.config.include_pdf)
        except (OSError, ValueError, json.JSONDecodeError):
            return None

    def _resume_training(
        self,
        run_dir: Path,
        candidates: list[CandidateSummary],
        baseline_prompt: str,
        best_prompt: str,
        n_used: int,
        max_iterations: int,
        train_records: tuple[Any, ...],
        schema: dict[str, Any],
    ) -> tuple[str, list[CandidateSummary]]:
        """Continue TextGrad optimization from iteration ``n_used`` to ``max_iterations``.

        The optimizer's variable is re-seeded from ``best_prompt``; its internal
        backward state is not persisted, so this is a deterministic-but-not-
        bit-exact continuation (see ``TextGradPromptOptimizer.optimize_resume``).
        """
        best_candidate = _select_best_candidate(candidates, default=candidates[0])
        best_results: dict[str, JudgeResult | None] = {}
        for record in train_records:
            document_id = record.pair.document_id
            document_dir = run_dir / "training" / best_candidate.candidate_id / "documents" / document_id
            cached = self._load_cached_judge_result(document_dir)
            if cached is not None:
                best_results[document_id] = cached
                continue
            results = self.evaluate_prompt(run_dir, best_candidate.candidate_id, best_prompt, (record,), schema)
            best_results[document_id] = results.get(document_id)

        def evaluate_training(candidate_id: str, prompt: str) -> dict[str, JudgeResult | None]:
            return self.evaluate_prompt(run_dir, candidate_id, prompt, train_records, schema)

        optimizer = self.optimizer_factory(evaluate_training)
        new_best_prompt, _best_candidate, new_candidates = optimizer.optimize_resume(
            initial_prompt=baseline_prompt,
            best_prompt=best_prompt,
            best_results=best_results,
            existing_candidates=candidates,
            start_iteration=n_used,
            max_iterations=max_iterations,
            evaluate_candidate=evaluate_training,
        )
        return new_best_prompt, [*candidates, *new_candidates]

    def _recompute_frozen_best(
        self,
        run_dir: Path,
        candidates: list[CandidateSummary],
        max_iterations: int,
        baseline_prompt: str,
        n_used: int,
    ) -> tuple[str, list[CandidateSummary]]:
        """Recompute the frozen best over the first ``max_iterations + 1`` candidates.

        Used when ``max_iterations`` is decreased. Candidate history is preserved
        (no truncation); only the frozen best prompt is updated. The old
        ``final-best.txt`` is archived to ``final-best.archived-<n_used>.txt``.
        """
        keep = candidates[: max_iterations + 1]
        for candidate in keep:
            prompt_path = run_dir / "training" / candidate.candidate_id / "prompt.txt"
            if not prompt_path.is_file():
                raise RuntimeError(
                    f"Cannot recompute frozen best: missing prompt artifact for {candidate.candidate_id} at {prompt_path}"
                )
        best = _select_best_candidate(keep, default=keep[0])
        old_best_path = run_dir / "prompts" / "final-best.txt"
        if old_best_path.is_file():
            archive_path = run_dir / "prompts" / f"final-best.archived-{n_used}.txt"
            shutil.copy2(old_best_path, archive_path)
            append_jsonl(
                run_dir / "meta" / "events.jsonl",
                {"event": "max_iterations_decreased", "old_n": n_used, "new_m": max_iterations, "archived_best": str(archive_path.relative_to(run_dir))},
            )
        else:
            append_jsonl(
                run_dir / "meta" / "events.jsonl",
                {"event": "max_iterations_decreased", "old_n": n_used, "new_m": max_iterations},
            )
        best_prompt = (run_dir / "training" / best.candidate_id / "prompt.txt").read_text(encoding="utf-8").strip()
        self._log(f"[resume] recomputed best={best.candidate_id} best_mean={best.mean_score}")
        return best_prompt, candidates

    def _clear_optimized_blind_documents(self, run_dir: Path, blind_records: list[Any]) -> None:
        self._clear_blind_documents(run_dir, "optimized", [record.pair.document_id for record in blind_records])

    def _validation_error_count(self, run_dir: Path, arm: str, document_id: str) -> int | None:
        """Number of schema-validation errors on a blind-arm prediction, or None if unknown."""
        path = run_dir / "blind_test" / arm / "documents" / document_id / "validation.json"
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return len(value.get("errors", []))
        except (OSError, ValueError, json.JSONDecodeError):
            return None

    @staticmethod
    def _blind_document_artifacts_complete(run_dir: Path, arm: str, document_id: str) -> bool:
        """Whether a checkpoint-valid blind document still has its report inputs."""
        document_dir = run_dir / "blind_test" / arm / "documents" / document_id
        validation_path = document_dir / "validation.json"
        if not validation_path.is_file() or not (document_dir / "judge.result.json").is_file():
            return False
        try:
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        if validation.get("errors"):
            return (document_dir / "extraction.response.json").is_file()
        return (document_dir / "prediction.json").is_file()

    def trim_checkpoint_to_round(self, run_id: str, round_index: int) -> None:
        """Reuse accepted rounds ``0..round_index``, dropping later rounds and stale blind results.

        Lets a finished/partial run continue from round ``round_index + 1`` with the
        current code instead of re-running accepted rounds. Only valid when the target
        round is an accepted joint candidate (its frozen prompts/schema are re-seeded).
        The checkpoint is backed up first; manifest matching is still enforced on resume.
        """
        run_dir = self.config.output_root / run_id
        cp_path = run_dir / "meta" / "checkpoint.json"
        if not cp_path.is_file():
            raise RuntimeError(f"No checkpoint to trim: {cp_path}")
        backup = run_dir / "meta" / f"checkpoint.bak-{datetime.now():%Y%m%d%H%M}.json"
        shutil.copy2(cp_path, backup)
        cp = json.loads(cp_path.read_text(encoding="utf-8"))
        alt = cp.get("alternating")
        candidates = alt.get("candidates") if isinstance(alt, dict) else None
        if not isinstance(candidates, list) or round_index < 0 or round_index >= len(candidates):
            raise RuntimeError(f"round-{round_index:03d} not present in checkpoint (have {len(candidates) if isinstance(candidates, list) else 0} candidates).")
        target = candidates[round_index]
        if target.get("phase") != "joint" or not target.get("accepted"):
            raise RuntimeError(f"round-{round_index:03d} must be an accepted joint candidate to reuse.")

        alt["candidates"] = candidates[: round_index + 1]
        alt["mode"] = self.config.optimization_mode
        cp["max_iterations_used"] = round_index
        cp["max_iterations_requested"] = round_index
        cp["stages"] = {"alternating_training_complete": True}

        self._reseed_and_drop(run_dir, candidates, round_index)
        cp_path.write_text(json.dumps(cp, ensure_ascii=False, indent=2), encoding="utf-8")
        self._log_reused_rounds(candidates[: round_index + 1])
        self._log(f"[resume-from-round] trimmed to round-{round_index:03d}; backup={backup.name}")

    def _log_reused_rounds(self, candidates: list[dict[str, Any]]) -> None:
        """Make a manual --resume-from-round decision auditable in the console log."""
        for candidate in candidates:
            scores = candidate.get("document_scores") or {}
            rendered = ", ".join(
                f"{document_id}={score:.1f}" if isinstance(score, (int, float)) else f"{document_id}=失败"
                for document_id, score in sorted(scores.items())
            )
            mean = candidate.get("mean_score")
            mean_text = f"{mean:.2f}" if isinstance(mean, (int, float)) else "无"
            self._log(f"[resume-from-round] reuse {candidate.get('candidate_id', '?')}: mean={mean_text} {rendered}")

    def _reseed_and_drop(self, run_dir: Path, candidates: list[dict[str, Any]], keep_idx: int) -> None:
        """Re-seed frozen prompts/schema from round ``keep_idx`` and drop later artifacts.

        Preserves the baseline blind-test arm (schema/prompt independent of training);
        only the optimized arm is dropped because the selected state may change.
        """
        target = candidates[keep_idx]
        joint = run_dir / "training" / target["candidate_id"]
        prompts = run_dir / "prompts"
        schemas = run_dir / "schemas"
        prompts.mkdir(exist_ok=True)
        schemas.mkdir(exist_ok=True)
        if _is_two_stage_mode(self.config.optimization_mode):
            evidence_path = joint / "prompt.evidence.txt"
            resolve_path = joint / "prompt.resolve.txt"
            if not evidence_path.is_file() or not resolve_path.is_file():
                raise RuntimeError(f"Cannot re-seed two-stage round {target['candidate_id']}: evidence/resolve prompt artifacts are missing.")
            evidence = evidence_path.read_text(encoding="utf-8")
            resolve = resolve_path.read_text(encoding="utf-8")
            (prompts / "final-best.txt").write_text(evidence, encoding="utf-8")
            (prompts / "final-best.evidence.txt").write_text(evidence, encoding="utf-8")
            if self.config.optimization_mode == "two_stage_evidence_routing_schema_description":
                routing_path = joint / "prompt.evidence-routing.txt"
                if not routing_path.is_file():
                    raise RuntimeError(f"Cannot re-seed four-variable round {target["candidate_id"]}: evidence-routing prompt artifact is missing.")
                (prompts / "final-best.evidence-routing.txt").write_text(routing_path.read_text(encoding="utf-8"), encoding="utf-8")
            (prompts / "final-best.resolve.txt").write_text(resolve, encoding="utf-8")
        else:
            (prompts / "final-best.txt").write_text((joint / "prompt.txt").read_text(encoding="utf-8"), encoding="utf-8")
        (prompts / "final-best-schema-description.txt").write_text(
            (joint / "schema-prompt.txt").read_text(encoding="utf-8"), encoding="utf-8")
        (schemas / "final-best-schema.json").write_text((joint / "schema.json").read_text(encoding="utf-8"), encoding="utf-8")
        for candidate in candidates[keep_idx + 1:]:
            dropped = run_dir / "training" / candidate["candidate_id"]
            if dropped.is_dir():
                shutil.rmtree(dropped)
        optimized = run_dir / "blind_test" / "optimized"
        if optimized.is_dir():
            shutil.rmtree(optimized)

    def _resolve_patch_ambiguities(self, raw: str, schema_dsl: dict[str, Any], *, debug_path: Path | None = None) -> str:
        """Resolve broken paths against the current schema and retain valid patches.

        If the raw patch validates as-is, return it unchanged. Otherwise, for every
        patch whose path cannot resolve but whose trailing field name matches real
        description nodes, ask the LLM to pick the intended candidate path, then
        re-validate each patch independently. A still-invalid patch is removed;
        independent valid changes remain eligible for evaluation. The LLM is only
        allowed to select paths supplied from the current schema.

        When ``debug_path`` is given and the fast path (raw validates as-is) is not
        taken, a ``patch.debug.json`` is written there recording the original raw,
        the first validation error, the ambiguous-path candidates, the LLM path
        selections, and which patches were dropped. This lets a ``proposal_failed``
        round be diagnosed after the fact — otherwise only the final (post-fix) raw
        survives on disk and the cause of the failure is lost.
        """
        try:
            document = parse_description_patch_text(raw)
            validated = validate_patch_document(schema_dsl, document)
            if validated == document["patches"]:
                return raw
            return json.dumps({"patches": validated}, ensure_ascii=False)
        except SchemaDescriptionPatchError as exc:
            first_error = str(exc)
        try:
            document = parse_description_patch_text(raw)
        except SchemaDescriptionPatchError as parse_exc:
            if debug_path is not None:
                write_json(debug_path, {
                    "stage": "parse_failed",
                    "original_raw": raw,
                    "first_validation_error": first_error,
                    "parse_error": str(parse_exc),
                    "final_raw": raw,
                })
            return raw
        ambiguous = []
        for index, patch in enumerate(document["patches"]):
            candidates = ambiguous_candidates(schema_dsl, patch["path"])
            if candidates:
                candidate_info = []
                for candidate_path in candidates:
                    try:
                        desc = _resolve_description(schema_dsl, candidate_path)
                    except SchemaDescriptionPatchError:
                        desc = ""
                    candidate_info.append({"path": candidate_path, "description": desc})
                ambiguous.append({
                    "patch_index": index,
                    "original_path": patch["path"],
                    "intent": patch["description"],
                    "candidates": candidate_info,
                })
        selected: dict[int, str] | None = None
        if ambiguous:
            selected = self._select_patch_paths(ambiguous)
            if selected:
                for patch_index, chosen in selected.items():
                    if 0 <= patch_index < len(document["patches"]):
                        document["patches"][patch_index]["path"] = chosen

        valid_patches = []
        dropped_patches = []
        for index, patch in enumerate(document["patches"]):
            try:
                validate_patch_document(schema_dsl, {"patches": [patch]})
            except SchemaDescriptionPatchError as drop_exc:
                dropped_patches.append({
                    "patch_index": index,
                    "path": patch["path"],
                    "intent": patch["description"],
                    "drop_reason": str(drop_exc),
                })
                continue
            valid_patches.append(patch)
        final_raw = json.dumps({"patches": valid_patches}, ensure_ascii=False) if len(valid_patches) != len(document["patches"]) else json.dumps(document, ensure_ascii=False)
        if debug_path is not None:
            write_json(debug_path, {
                "stage": "ambiguity_resolved",
                "original_raw": raw,
                "first_validation_error": first_error,
                "ambiguous_patches": ambiguous,
                "llm_selections": {str(k): v for k, v in (selected or {}).items()},
                "dropped_patches": dropped_patches,
                "valid_patch_count": len(valid_patches),
                "original_patch_count": len(document["patches"]),
                "final_raw": final_raw,
            })
        return final_raw

    def _select_patch_paths(self, ambiguous: list[dict[str, Any]]) -> dict[int, str] | None:
        """Ask the patch LLM to choose one real candidate path per ambiguous patch."""
        if self.schema_patch_client is None:
            return None
        try:
            with api_operation("schema_patch_path_selection"):
                selection = self.schema_patch_client.complete_json(
                    system_prompt=_PATH_SELECTION_SYSTEM_PROMPT,
                    payload={"patches": ambiguous},
                )
            value = json.loads(selection)
        except (json.JSONDecodeError, ValueError, TypeError):
            return None
        chosen: dict[int, str] = {}
        for item in value.get("selections", []):
            try:
                patch_index = int(item["index"])
                path = str(item["path"])
            except (KeyError, TypeError, ValueError):
                continue
            entry = next((a for a in ambiguous if a["patch_index"] == patch_index), None)
            if entry is not None:
                valid_paths = [candidate["path"] for candidate in entry["candidates"]]
                if path in valid_paths:
                    chosen[patch_index] = path
        return chosen if chosen else None

    @staticmethod
    def _clear_blind_documents(run_dir: Path, arm: str, document_ids: list[str]) -> None:
        documents_root = run_dir / "blind_test" / arm / "documents"
        for document_id in document_ids:
            document_dir = documents_root / document_id
            if document_dir.is_dir():
                shutil.rmtree(document_dir)

    # ---- alternating schema-description mode ----

    def _run_alternating(
        self,
        run_dir: Path,
        split: DatasetSplit,
        requested_rounds: int,
        smoke_blind_doc: str | None,
        smoke_training: bool,
        checkpoint: dict[str, Any],
    ) -> Path:
        schema_text = self.config.schema_path.read_text(encoding="utf-8")
        base_schema_dsl = reinforce_required_emission(json.loads(schema_text))
        baseline_prompt = self.config.initial_prompt_path.read_text(encoding="utf-8").strip()
        train_records = split.train if not smoke_training else split.train[:1]

        alt = checkpoint.get("alternating")
        training_complete = bool(checkpoint["stages"].get("alternating_training_complete"))
        n_used = checkpoint.get("max_iterations_used")
        optimized_dirty = False

        def checkpoint_round(
            schema_prompt: str,
            extraction_prompt: str,
            selected_schema: dict[str, Any],
            candidates: list[AlternatingCandidate],
        ) -> None:
            self._checkpoint_alternating_round(
                run_dir, checkpoint, requested_rounds,
                mode=self.config.optimization_mode,
                schema_prompt=schema_prompt,
                extraction_prompt=extraction_prompt,
                selected_dsl=selected_schema,
                candidates=candidates,
            )

        if training_complete and isinstance(alt, dict):
            if alt.get("training_algorithm_version") != "joint-round-v2":
                raise RuntimeError("Existing alternating run uses the retired two-phase algorithm; start a new run ID for joint-round-v2.")
            schema_prompt, extraction_prompt, selected_dsl, candidates = self._load_alternating_checkpoint(run_dir, alt)
            # Auto-reuse: if trailing rounds are not accepted (proposal_failed / no
            # score), roll back to the last effective round and continue from there
            # with current code, instead of re-running accepted rounds or freezing
            # on a failed tail.
            declared = n_used if isinstance(n_used, int) else len(candidates) - 1
            effective = _last_effective_alternating_index(candidates)
            if effective is not None and effective < declared:
                self._log(f"[resume] auto-trim: rounds {effective + 1}..{declared} not effective; reusing through round-{effective:03d}")
                self._reseed_and_drop(run_dir, candidates, effective)
                candidates = candidates[: effective + 1]
                alt["candidates"] = candidates
                checkpoint["max_iterations_used"] = effective
                checkpoint["max_iterations_requested"] = effective
                checkpoint["stages"]["alternating_training_complete"] = True
                for stage in ("blind_optimized_complete", "primary_complete", "report_complete"):
                    checkpoint["stages"].pop(stage, None)
                n_used = effective
                optimized_dirty = True
            if isinstance(n_used, int) and requested_rounds > n_used:
                self._log(f"[resume] alternating rounds {n_used}->{requested_rounds}: continuing from round-{n_used + 1:03d}")
                best_results = self._load_best_alternating_results(run_dir, candidates, train_records, selected_dsl)
                schema_prompt, extraction_prompt, selected_dsl, candidates = self._resume_alternating_training(
                    run_dir, schema_text, train_records, schema_prompt, extraction_prompt,
                    selected_dsl, best_results, candidates, n_used + 1, requested_rounds,
                    on_round_complete=checkpoint_round,
                )
                checkpoint["alternating"] = self._alternating_checkpoint_payload(schema_prompt, extraction_prompt, selected_dsl, candidates)
                checkpoint["max_iterations_used"] = requested_rounds
                checkpoint["max_iterations_requested"] = requested_rounds
                optimized_dirty = True
                self._write_checkpoint(run_dir, checkpoint)
                self._freeze_alternating(run_dir, base_schema_dsl, schema_prompt, extraction_prompt, selected_dsl)
            elif isinstance(n_used, int) and requested_rounds < n_used:
                self._log(f"[resume] alternating rounds {n_used}->{requested_rounds}: recomputing selected state over rounds <= {requested_rounds}")
                schema_prompt, extraction_prompt, selected_dsl, candidates = self._recompute_alternating_selected(
                    run_dir, base_schema_dsl, candidates, requested_rounds
                )
                checkpoint["alternating"] = self._alternating_checkpoint_payload(schema_prompt, extraction_prompt, selected_dsl, candidates)
                checkpoint["max_iterations_used"] = requested_rounds
                checkpoint["max_iterations_requested"] = requested_rounds
                checkpoint["stages"]["blind_optimized_complete"] = False
                checkpoint["stages"]["primary_complete"] = False
                checkpoint["stages"]["report_complete"] = False
                optimized_dirty = True
                self._write_checkpoint(run_dir, checkpoint)
                self._freeze_alternating(run_dir, base_schema_dsl, schema_prompt, extraction_prompt, selected_dsl)
            else:
                self._log("[resume] reused completed alternating training and frozen artifacts")
                if not isinstance(n_used, int):
                    checkpoint["max_iterations_used"] = requested_rounds
                    checkpoint["max_iterations_requested"] = requested_rounds
                    self._write_checkpoint(run_dir, checkpoint)
        elif isinstance(alt, dict) and isinstance(alt.get("candidates"), list) and alt["candidates"]:
            schema_prompt, extraction_prompt, selected_dsl, candidates = self._load_alternating_checkpoint(run_dir, alt)
            last_round = _round_of(candidates[-1].candidate_id)
            if last_round < requested_rounds:
                self._log(f"[resume] recovered partial alternating training through round-{last_round:03d}; continuing at round-{last_round + 1:03d}")
                best_results = self._load_best_alternating_results(run_dir, candidates, train_records, selected_dsl)
                schema_prompt, extraction_prompt, selected_dsl, candidates = self._resume_alternating_training(
                    run_dir, schema_text, train_records, schema_prompt, extraction_prompt,
                    selected_dsl, best_results, candidates, last_round + 1, requested_rounds,
                    on_round_complete=checkpoint_round,
                )
            else:
                self._log(f"[resume] recovered completed alternating rounds through round-{last_round:03d}")
            checkpoint["alternating"] = self._alternating_checkpoint_payload(schema_prompt, extraction_prompt, selected_dsl, candidates)
            checkpoint["stages"]["alternating_training_complete"] = True
            checkpoint["max_iterations_used"] = requested_rounds
            checkpoint["max_iterations_requested"] = requested_rounds
            self._write_checkpoint(run_dir, checkpoint)
        else:
            schema_prompt, extraction_prompt, selected_dsl, candidates = self._run_alternating_training(
                run_dir, schema_text, train_records, requested_rounds,
                on_round_complete=checkpoint_round,
            )
            checkpoint["alternating"] = self._alternating_checkpoint_payload(schema_prompt, extraction_prompt, selected_dsl, candidates)
            checkpoint["stages"]["alternating_training_complete"] = True
            checkpoint["max_iterations_used"] = requested_rounds
            checkpoint["max_iterations_requested"] = requested_rounds
            self._write_checkpoint(run_dir, checkpoint)
            self._freeze_alternating(run_dir, base_schema_dsl, schema_prompt, extraction_prompt, selected_dsl)

        # The blind-test gate and frozen artifacts must refer to the same
        # highest-scoring joint candidate, not merely the last valid proposal.
        schema_prompt, extraction_prompt, selected_dsl, candidates = self._load_selected_alternating_state(
            run_dir, base_schema_dsl, candidates
        )
        checkpoint["alternating"] = self._alternating_checkpoint_payload(
            schema_prompt, extraction_prompt, selected_dsl, candidates
        )
        checkpoint["alternating"]["mode"] = self.config.optimization_mode
        self._write_checkpoint(run_dir, checkpoint)
        self._freeze_alternating(run_dir, base_schema_dsl, schema_prompt, extraction_prompt, selected_dsl)

        self._log_alternating_candidates(candidates)
        # Early exit: only run the blind test when optimization produced a real,
        # non-trivial training gain over round-000 (baseline). A tie or sub-
        # threshold gain yields a predetermined or near-predetermined optimized
        # arm, so running both blind arms wastes extraction/judge cost on an
        # outcome within training noise.
        round_zero_mean = next(
            (c.mean_score for c in candidates if c.candidate_id == "round-000/joint" and c.mean_score is not None),
            None,
        )
        best_mean = max((c.mean_score for c in candidates if c.mean_score is not None), default=None)
        if (
            round_zero_mean is not None
            and best_mean is not None
            and (best_mean - round_zero_mean) < MIN_BLIND_TEST_IMPROVEMENT
        ):
            best_candidate = next((c.candidate_id for c in candidates if c.mean_score == best_mean), "round-000/joint")
            improvement = best_mean - round_zero_mean
            self._log(
                f"[skip] optimization gain below blind-test threshold "
                f"(best {best_candidate}={best_mean:.1f} - baseline {round_zero_mean:.1f} = {improvement:+.1f} < "
                f"{MIN_BLIND_TEST_IMPROVEMENT:.1f}); skipping blind test"
            )
            write_json(
                run_dir / "blind_test_skipped.json",
                {
                    "reason": "optimization gain below blind-test threshold",
                    "best_candidate": best_candidate,
                    "best_mean": best_mean,
                    "baseline_mean": round_zero_mean,
                    "improvement": improvement,
                    "threshold": MIN_BLIND_TEST_IMPROVEMENT,
                    "blind_test": "skipped",
                },
            )
            checkpoint["stages"]["blind_skipped_optimization_failed"] = True
            self._write_checkpoint(run_dir, checkpoint)
            self._log(f"[done] run_dir={run_dir} blind test skipped (optimization gain below threshold)")
            return run_dir
        selected_schema = dsl_to_json_schema(selected_dsl)

        blind_records = list(split.blind_test)
        if smoke_blind_doc:
            blind_records = [record for record in blind_records if record.pair.document_id == smoke_blind_doc]
            if len(blind_records) != 1:
                raise ValueError("--blind-doc must identify one blind-test document in the frozen split.")
        self._log(f"[blind_test] documents={len(blind_records)} (smoke={smoke_blind_doc is not None})")
        self._log(f"[blind_test] baseline extraction+judge with base schema for {len(blind_records)} documents...")
        baseline_results = self.evaluate_prompt(run_dir, "baseline", baseline_prompt, tuple(blind_records), None, stage="blind_test", schema_dsl=base_schema_dsl)
        checkpoint["stages"]["blind_baseline_complete"] = True
        self._write_checkpoint(run_dir, checkpoint)
        self._log(f"[blind_test] optimized extraction+judge with selected schema for {len(blind_records)} documents...")
        if optimized_dirty or checkpoint["stages"].get("blind_optimized_complete") is False:
            self._clear_optimized_blind_documents(run_dir, blind_records)
        optimized_results = self.evaluate_prompt(
            run_dir, "optimized", extraction_prompt, tuple(blind_records), None, stage="blind_test", schema_dsl=selected_dsl
        )
        checkpoint["stages"]["blind_optimized_complete"] = True
        self._write_checkpoint(run_dir, checkpoint)

        paired: list[dict[str, Any]] = []
        for index, record in enumerate(blind_records, start=1):
            document_id = record.pair.document_id
            baseline_result = baseline_results[document_id]
            baseline_status = "valid" if baseline_result else "failed"
            optimized_result = optimized_results[document_id]
            optimized_status = "valid" if optimized_result else "failed"
            score_b = f"{baseline_result.score:.1f}" if baseline_result else "?"
            score_o = f"{optimized_result.score:.1f}" if optimized_result else "?"
            self._log(f"[blind_test] {index}/{len(blind_records)} {document_id}: 基线={score_b} 优化={score_o}")
            paired.append(
                {
                    "document_id": document_id,
                    "baseline_score": baseline_result.score if baseline_result else None,
                    "optimized_score": optimized_result.score if optimized_result else None,
                    "baseline_status": baseline_status,
                    "optimized_status": optimized_status,
                    "baseline_validation_errors": self._validation_error_count(run_dir, "baseline", document_id),
                    "optimized_validation_errors": self._validation_error_count(run_dir, "optimized", document_id),
                }
            )
        audit_records = tuple(train_records) + tuple(blind_records)
        audit_results = (
            self.run_gold_audit(run_dir, audit_records, selected_schema) if self.config.gold_audit_enabled else []
        )
        valid_paired = sum(1 for p in paired if p["baseline_status"] == "valid" and p["optimized_status"] == "valid")
        checkpoint["paired"] = paired
        checkpoint["stages"]["primary_complete"] = True
        self._write_checkpoint(run_dir, checkpoint)
        self.finish_alternating_report(run_dir, candidates, paired, audit_results, base_schema_dsl, selected_dsl)
        checkpoint["stages"]["report_complete"] = True
        self._write_checkpoint(run_dir, checkpoint)
        self._log(f"[done] run_dir={run_dir} valid_paired={valid_paired}/{len(paired)}")
        return run_dir

    def _run_alternating_training(
        self,
        run_dir: Path,
        schema_text: str,
        train_records: tuple[Any, ...],
        rounds: int,
        on_round_complete: Any = None,
    ) -> tuple[str, str, dict[str, Any], list[AlternatingCandidate]]:
        base_schema_dsl = reinforce_required_emission(json.loads(schema_text))
        schema_prompt = self.config.schema_description_initial_prompt_path.read_text(encoding="utf-8").strip()
        extraction_prompt = self.config.initial_prompt_path.read_text(encoding="utf-8").strip()
        optimizer = self._make_alternating_optimizer(run_dir, train_records, base_schema_dsl, on_round_complete=on_round_complete)
        return optimizer.optimize(schema_prompt, extraction_prompt, rounds)

    def _resume_alternating_training(
        self,
        run_dir: Path,
        schema_text: str,
        train_records: tuple[Any, ...],
        schema_prompt: str,
        extraction_prompt: str,
        selected_dsl: dict[str, Any],
        best_results: dict[str, JudgeResult | None],
        existing_candidates: list[AlternatingCandidate],
        start_round: int,
        max_rounds: int,
        on_round_complete: Any = None,
    ) -> tuple[str, str, dict[str, Any], list[AlternatingCandidate]]:
        optimizer = self._make_alternating_optimizer(
            run_dir, train_records, reinforce_required_emission(json.loads(schema_text)), on_round_complete=on_round_complete
        )
        return optimizer.optimize_resume(
            schema_prompt, extraction_prompt, selected_dsl, best_results,
            existing_candidates, start_round, max_rounds,
        )

    def _make_alternating_optimizer(
        self,
        run_dir: Path,
        train_records: tuple[Any, ...],
        base_schema_dsl: dict[str, Any],
        on_round_complete: Any = None,
    ) -> AlternatingSchemaDescriptionOptimizer:
        if self.textgrad_engine is None:
            raise RuntimeError("Alternating mode requires a TextGrad engine.")

        def propose_schema_patch(phase_id: str, schema_prompt_text: str, schema_dsl: dict[str, Any], results: dict[str, JudgeResult | None]) -> str:
            if self.schema_patch_client is None or not self.schema_patch_system_prompt:
                raise RuntimeError("Alternating mode requires a schema-patch client and system prompt.")
            feedback = "\n".join(
                f"[{doc_id}] {result.optimization_feedback}"
                for doc_id, result in sorted(results.items())
                if result is not None and result.optimization_feedback
            )
            payload = {
                "schema_dsl": schema_dsl,
                "schema_patch_prompt": schema_prompt_text,
                "training_feedback": feedback,
            }
            with api_operation("schema_patch_proposal"):
                raw = self.schema_patch_client.complete_json(
                    system_prompt=self.schema_patch_system_prompt,
                    payload=payload,
                    reasoning_effort=None,  # format-following patch task needs thinking; bypass judge's "none"
                )
            # Closed-set fallback: if the patch path cannot resolve (broken prefix,
            # wrong intermediate layer), let the LLM choose from the real candidate
            # description paths instead of failing the whole round.
            phase_dir = run_dir / "training" / phase_id
            phase_dir.mkdir(parents=True, exist_ok=True)
            raw = self._resolve_patch_ambiguities(raw, schema_dsl, debug_path=phase_dir / "patch.debug.json")
            write_text(phase_dir / "schema-prompt.txt", schema_prompt_text)
            write_text(phase_dir / "patch.raw.txt", raw)
            write_json(phase_dir / "schema.proposed.json", schema_dsl)
            return raw

        def evaluate_state(phase_id: str, prompt: str, schema_dsl: dict[str, Any]) -> dict[str, JudgeResult | None]:
            phase_dir = run_dir / "training" / phase_id
            phase_dir.mkdir(parents=True, exist_ok=True)
            if phase_id == "round-000/joint":
                write_text(phase_dir / "schema-prompt.txt", self.config.schema_description_initial_prompt_path.read_text(encoding="utf-8").strip())
            write_json(phase_dir / "schema.json", schema_dsl)
            return self.evaluate_prompt(run_dir, phase_id, prompt, train_records, None, stage="training", schema_dsl=schema_dsl)

        return AlternatingSchemaDescriptionOptimizer(
            engine=self.textgrad_engine,
            propose_patch=propose_schema_patch,
            evaluate=evaluate_state,
            base_schema_dsl=base_schema_dsl,
            optimize_extraction_prompt=self.config.optimization_mode != "description_only",
            on_round_complete=on_round_complete,
        )

    # ---- two-stage evidence/resolve + schema-description mode ----

    def _run_two_stage(
        self,
        run_dir: Path,
        split: DatasetSplit,
        requested_rounds: int,
        smoke_blind_doc: str | None,
        smoke_training: bool,
        checkpoint: dict[str, Any],
    ) -> Path:
        """Two-stage index→resolve extraction + description-only schema patches.

        Supports resume: a completed two-stage training checkpoint is loaded and
        training continues from the next round instead of restarting. Mirrors
        ``_run_alternating``'s reuse/continue/recompute branches; the last
        accepted state is re-seeded into the optimizer so rounds continue from
        where the interrupted run stopped.
        """
        schema_text = self.config.schema_path.read_text(encoding="utf-8")
        base_schema_dsl = reinforce_required_emission(json.loads(schema_text))
        baseline_evidence = self.config.evidence_initial_prompt_path.read_text(encoding="utf-8").strip()
        baseline_resolve = self.config.resolve_initial_prompt_path.read_text(encoding="utf-8").strip()
        train_records = split.train if not smoke_training else split.train[:1]

        alt = checkpoint.get("alternating")
        persisted_evidence_sha = alt.get("evidence_prompt_sha256") if isinstance(alt, dict) else None
        persisted_resolve_sha = alt.get("resolve_prompt_sha256") if isinstance(alt, dict) else None
        training_complete = bool(checkpoint["stages"].get("alternating_training_complete"))
        n_used = checkpoint.get("max_iterations_used")
        optimized_dirty = False

        def checkpoint_round(
            schema_prompt: str,
            evidence_prompt: str,
            resolve_prompt: str,
            selected_schema: dict[str, Any],
            candidates: list[AlternatingCandidate],
        ) -> None:
            self._checkpoint_alternating_round(
                run_dir, checkpoint, requested_rounds,
                mode=self.config.optimization_mode,
                schema_prompt=schema_prompt,
                extraction_prompt=evidence_prompt,
                selected_dsl=selected_schema,
                candidates=candidates,
                evidence_prompt=evidence_prompt,
                resolve_prompt=resolve_prompt,
            )

        if training_complete and isinstance(alt, dict):
            best_schema_prompt, best_evidence_prompt, best_resolve_prompt, selected_dsl, candidates = self._load_two_stage_checkpoint(run_dir, alt)
            if isinstance(n_used, int) and requested_rounds > n_used:
                self._log(f"[resume] two-stage rounds {n_used}->{requested_rounds}: continuing from round-{n_used + 1:03d}")
                best_results = self._load_best_two_stage_results(run_dir, candidates, train_records, best_evidence_prompt, best_resolve_prompt, selected_dsl)
                optimizer = self._make_two_stage_optimizer(run_dir, train_records, base_schema_dsl, on_round_complete=checkpoint_round)
                best_schema_prompt, best_evidence_prompt, best_resolve_prompt, selected_dsl, candidates = optimizer.optimize_resume(
                    best_schema_prompt, best_evidence_prompt, best_resolve_prompt,
                    selected_dsl, best_results, candidates, n_used + 1, requested_rounds,
                )
                checkpoint["alternating"] = self._alternating_checkpoint_payload(best_schema_prompt, best_evidence_prompt, selected_dsl, candidates)
                checkpoint["max_iterations_used"] = requested_rounds
                checkpoint["max_iterations_requested"] = requested_rounds
                optimized_dirty = True
                self._write_checkpoint(run_dir, checkpoint)
            elif isinstance(n_used, int) and requested_rounds < n_used:
                self._log(f"[resume] two-stage rounds {n_used}->{requested_rounds}: recomputing selected state over rounds <= {requested_rounds}")
                best_schema_prompt, best_evidence_prompt, best_resolve_prompt, selected_dsl, candidates = self._recompute_two_stage_selected(
                    run_dir, base_schema_dsl, candidates, requested_rounds
                )
                checkpoint["alternating"] = self._alternating_checkpoint_payload(best_schema_prompt, best_evidence_prompt, selected_dsl, candidates)
                checkpoint["max_iterations_used"] = requested_rounds
                checkpoint["max_iterations_requested"] = requested_rounds
                for stage in ("blind_optimized_complete", "primary_complete", "report_complete"):
                    checkpoint["stages"].pop(stage, None)
                optimized_dirty = True
                self._write_checkpoint(run_dir, checkpoint)
            else:
                self._log("[resume] reused completed two-stage training and frozen artifacts")
                if not isinstance(n_used, int):
                    checkpoint["max_iterations_used"] = requested_rounds
                    checkpoint["max_iterations_requested"] = requested_rounds
                    self._write_checkpoint(run_dir, checkpoint)
        elif isinstance(alt, dict) and isinstance(alt.get("candidates"), list) and alt["candidates"]:
            candidates = _deserialize_alternating_candidates(alt["candidates"])
            last_round = _round_of(candidates[-1].candidate_id)
            best_schema_prompt, best_evidence_prompt, best_resolve_prompt, selected_dsl, _loaded = self._load_two_stage_checkpoint(run_dir, alt)
            if last_round >= requested_rounds:
                self._log(f"[resume] recovered completed two-stage rounds through round-{last_round:03d}")
            else:
                self._log(f"[resume] recovered partial two-stage training through round-{last_round:03d}; continuing at round-{last_round + 1:03d}")
                best_results = self._load_best_two_stage_results(
                    run_dir, candidates, train_records, best_evidence_prompt, best_resolve_prompt, selected_dsl
                )
                optimizer = self._make_two_stage_optimizer(run_dir, train_records, base_schema_dsl, on_round_complete=checkpoint_round)
                best_schema_prompt, best_evidence_prompt, best_resolve_prompt, selected_dsl, candidates = optimizer.optimize_resume(
                    best_schema_prompt, best_evidence_prompt, best_resolve_prompt,
                    selected_dsl, best_results, candidates, last_round + 1, requested_rounds,
                )
            checkpoint["alternating"] = self._alternating_checkpoint_payload(best_schema_prompt, best_evidence_prompt, selected_dsl, candidates)
            checkpoint["stages"]["alternating_training_complete"] = True
            checkpoint["max_iterations_used"] = requested_rounds
            checkpoint["max_iterations_requested"] = requested_rounds
            self._write_checkpoint(run_dir, checkpoint)
        else:
            optimizer = self._make_two_stage_optimizer(run_dir, train_records, base_schema_dsl, on_round_complete=checkpoint_round)
            schema_prompt = self.config.schema_description_initial_prompt_path.read_text(encoding="utf-8").strip()
            best_schema_prompt, best_evidence_prompt, best_resolve_prompt, selected_dsl, candidates = optimizer.optimize(
                schema_prompt, baseline_evidence, baseline_resolve, requested_rounds
            )
            checkpoint["alternating"] = self._alternating_checkpoint_payload(best_schema_prompt, best_evidence_prompt, selected_dsl, candidates)
            checkpoint["stages"]["alternating_training_complete"] = True
            checkpoint["max_iterations_used"] = requested_rounds
            checkpoint["max_iterations_requested"] = requested_rounds
            self._write_checkpoint(run_dir, checkpoint)

        # TextGrad keeps the last valid proposal live so it can make the next
        # update from that state.  That continuation state is not necessarily
        # the training-score maximum.  Freeze and blind-test the independently
        # selected maximum, exactly as the single-stage alternating runner does.
        best_schema_prompt, best_evidence_prompt, best_resolve_prompt, selected_dsl, candidates = self._recompute_two_stage_selected(
            run_dir, base_schema_dsl, candidates, requested_rounds
        )
        selected_evidence_sha = sha256_json(best_evidence_prompt)
        selected_resolve_sha = sha256_json(best_resolve_prompt)
        if (
            (persisted_evidence_sha is not None and persisted_evidence_sha != selected_evidence_sha)
            or (persisted_resolve_sha is not None and persisted_resolve_sha != selected_resolve_sha)
        ):
            # A pre-fix run may already have optimized blind artifacts made
            # with its last valid (but not best) round.  Do not mix them with
            # the newly selected training-best state.
            optimized_dirty = True
            checkpoint["stages"]["blind_optimized_complete"] = False
            checkpoint["stages"]["primary_complete"] = False
            checkpoint["stages"]["report_complete"] = False

        checkpoint["alternating"] = self._alternating_checkpoint_payload(
            best_schema_prompt, best_evidence_prompt, selected_dsl, candidates
        )
        checkpoint["alternating"]["mode"] = self.config.optimization_mode
        checkpoint["alternating"]["evidence_prompt_sha256"] = selected_evidence_sha
        checkpoint["alternating"]["resolve_prompt_sha256"] = selected_resolve_sha
        self._write_checkpoint(run_dir, checkpoint)
        self._freeze_alternating(run_dir, base_schema_dsl, best_schema_prompt, best_evidence_prompt, selected_dsl)
        # Also freeze the two-stage evidence/resolve prompts as side artifacts.
        write_text(run_dir / "prompts" / "final-best.evidence.txt", best_evidence_prompt)
        write_text(run_dir / "prompts" / "final-best.resolve.txt", best_resolve_prompt)
        write_text(run_dir / "prompts" / "baseline.evidence.txt", baseline_evidence)
        write_text(run_dir / "prompts" / "baseline.resolve.txt", baseline_resolve)

        self._log_alternating_candidates(candidates)

        # Early exit on sub-threshold gain (mirror alternating's gate).
        round_zero_mean = next(
            (c.mean_score for c in candidates if c.candidate_id == "round-000/joint" and c.mean_score is not None),
            None,
        )
        best_mean = max((c.mean_score for c in candidates if c.mean_score is not None), default=None)
        if (
            round_zero_mean is not None
            and best_mean is not None
            and (best_mean - round_zero_mean) < MIN_BLIND_TEST_IMPROVEMENT
        ):
            best_candidate = next((c.candidate_id for c in candidates if c.mean_score == best_mean), "round-000/joint")
            improvement = best_mean - round_zero_mean
            self._log(
                f"[skip] optimization gain below blind-test threshold "
                f"(best {best_candidate}={best_mean:.1f} - baseline {round_zero_mean:.1f} = {improvement:+.1f} < "
                f"{MIN_BLIND_TEST_IMPROVEMENT:.1f}); skipping blind test"
            )
            write_json(
                run_dir / "blind_test_skipped.json",
                {
                    "reason": "optimization gain below blind-test threshold",
                    "best_candidate": best_candidate,
                    "best_mean": best_mean,
                    "baseline_mean": round_zero_mean,
                    "improvement": improvement,
                    "threshold": MIN_BLIND_TEST_IMPROVEMENT,
                    "blind_test": "skipped",
                },
            )
            checkpoint["stages"]["blind_skipped_optimization_failed"] = True
            self._write_checkpoint(run_dir, checkpoint)
            self._log(f"[done] run_dir={run_dir} blind test skipped (optimization gain below threshold)")
            return run_dir
        selected_schema = dsl_to_json_schema(selected_dsl)

        blind_records = list(split.blind_test)
        if smoke_blind_doc:
            blind_records = [record for record in blind_records if record.pair.document_id == smoke_blind_doc]
            if len(blind_records) != 1:
                raise ValueError("--blind-doc must identify one blind-test document in the frozen split.")
        self._log(f"[blind_test] documents={len(blind_records)} (smoke={smoke_blind_doc is not None})")
        self._log(f"[blind_test] baseline two-stage extraction+judge for {len(blind_records)} documents...")
        baseline_results = self._evaluate_two_stage(
            run_dir, "blind_test", "baseline", baseline_evidence, baseline_resolve, tuple(blind_records), base_schema_dsl
        )
        checkpoint["stages"]["blind_baseline_complete"] = True
        self._write_checkpoint(run_dir, checkpoint)
        self._log(f"[blind_test] optimized two-stage extraction+judge for {len(blind_records)} documents...")
        if optimized_dirty or checkpoint["stages"].get("blind_optimized_complete") is False:
            self._clear_optimized_blind_documents(run_dir, blind_records)
        optimized_results = self._evaluate_two_stage(
            run_dir, "blind_test", "optimized", best_evidence_prompt, best_resolve_prompt, tuple(blind_records), selected_dsl
        )
        checkpoint["stages"]["blind_optimized_complete"] = True
        self._write_checkpoint(run_dir, checkpoint)

        paired: list[dict[str, Any]] = []
        for index, record in enumerate(blind_records, start=1):
            document_id = record.pair.document_id
            baseline_result = baseline_results.get(document_id)
            baseline_status = "valid" if baseline_result else "failed"
            optimized_result = optimized_results.get(document_id)
            optimized_status = "valid" if optimized_result else "failed"
            score_b = f"{baseline_result.score:.1f}" if baseline_result else "?"
            score_o = f"{optimized_result.score:.1f}" if optimized_result else "?"
            self._log(f"[blind_test] {index}/{len(blind_records)} {document_id}: 基线={score_b} 优化={score_o}")
            paired.append(
                {
                    "document_id": document_id,
                    "baseline_score": baseline_result.score if baseline_result else None,
                    "optimized_score": optimized_result.score if optimized_result else None,
                    "baseline_status": baseline_status,
                    "optimized_status": optimized_status,
                    "baseline_validation_errors": self._two_stage_validation_error_count(run_dir, "blind_test/baseline", document_id),
                    "optimized_validation_errors": self._two_stage_validation_error_count(run_dir, "blind_test/optimized", document_id),
                }
            )
        audit_records = tuple(train_records) + tuple(blind_records)
        audit_results = (
            self.run_gold_audit(run_dir, audit_records, selected_schema) if self.config.gold_audit_enabled else []
        )
        valid_paired = sum(1 for p in paired if p["baseline_status"] == "valid" and p["optimized_status"] == "valid")
        checkpoint["paired"] = paired
        checkpoint["stages"]["primary_complete"] = True
        self._write_checkpoint(run_dir, checkpoint)
        self.finish_alternating_report(run_dir, candidates, paired, audit_results, base_schema_dsl, selected_dsl)
        checkpoint["stages"]["report_complete"] = True
        self._write_checkpoint(run_dir, checkpoint)
        self._log(f"[done] run_dir={run_dir} valid_paired={valid_paired}/{len(paired)}")
        return run_dir

    def _evaluate_two_stage(
        self,
        run_dir: Path,
        stage_dir: str,
        candidate_id: str,
        evidence_prompt: str,
        resolve_prompt: str,
        records: tuple[Any, ...],
        schema_dsl: dict[str, Any],
        evidence_routing_prompt: str | None = None,
    ) -> dict[str, JudgeResult | None]:
        """Run two-stage extraction + judge for each record, persisting per-doc artifacts.

        ``stage_dir`` is the relative directory under ``run_dir`` (e.g.
        ``training``, ``blind_test/baseline``). ``candidate_id`` names the
        sub-directory inside ``stage_dir`` (e.g. ``round-000/joint`` or
        ``baseline``). Cache keys carry a ``stage`` marker so two-stage
        extractions never collide with single-stage alternating cache entries.
        """
        from .shared_cache import SharedCache

        schema = dsl_to_json_schema(schema_dsl)
        evidence_system_prompt = self.config.evidence_system_prompt_path.read_text(encoding="utf-8")
        resolve_system_prompt = self.config.resolve_system_prompt_path.read_text(encoding="utf-8")
        coverage_plan_system_prompt = (
            self.config.coverage_plan_system_prompt_path.read_text(encoding="utf-8")
            if self.config.optimization_mode == "two_stage_coverage_plan_schema_description"
            else (
                self.config.evidence_routing_system_prompt_path.read_text(encoding="utf-8")
                if self.config.optimization_mode == "two_stage_evidence_routing_schema_description"
                else None
            )
        )
        phase_dir = run_dir / stage_dir / candidate_id
        phase_dir.mkdir(parents=True, exist_ok=True)
        write_text(phase_dir / "prompt.evidence.txt", evidence_prompt)
        if evidence_routing_prompt is not None:
            write_text(phase_dir / "prompt.evidence-routing.txt", evidence_routing_prompt)
        write_text(phase_dir / "prompt.resolve.txt", resolve_prompt)
        if coverage_plan_system_prompt is not None:
            write_text(phase_dir / "prompt.coverage-plan-system.txt", coverage_plan_system_prompt)
        write_json(phase_dir / "schema.json", schema_dsl)

        client = self.extractor if hasattr(self.extractor, "complete_pdf_json") else None
        extractor_client = client or self._extractor_client_for_two_stage()

        def _evaluate_doc(record: Any) -> tuple[str, JudgeResult | None]:
            """Evaluate one two-stage document (extraction + judge)."""
            document_id = record.pair.document_id
            document_dir = phase_dir / "documents" / document_id
            document_dir.mkdir(parents=True, exist_ok=True)
            cached_result = self._load_cached_judge_result(document_dir) if self.config.cache_enabled else None
            if cached_result is not None:
                return document_id, cached_result
            extraction_inputs = self._two_stage_cache_inputs(record, schema, evidence_prompt, resolve_prompt, schema_dsl, evidence_routing_prompt)
            cache = SharedCache(self.config.output_root / "_cache", self.config.extractor.model)
            cached_extraction = cache.get("extraction_two_stage", extraction_inputs) if self.config.cache_enabled else None
            prediction: PredictionArtifact | None = None
            metadata: dict[str, Any] = {}
            if cached_extraction is not None:
                prediction, metadata = self._prediction_from_cache(cached_extraction)
                self._write_two_stage_prediction_artifacts(document_dir, prediction, metadata, cache.materialize_hit(cached_extraction))
            else:
                try:
                    prediction, metadata = extract_two_stage(
                        client=extractor_client,
                        pdf_path=record.pair.pdf_path,
                        root_schema_dsl=schema_dsl,
                        root_json_schema=schema,
                        evidence_system_prompt=evidence_system_prompt,
                        evidence_prompt=evidence_prompt,
                        resolve_system_prompt=resolve_system_prompt,
                        resolve_prompt=resolve_prompt,
                        coverage_plan_system_prompt=coverage_plan_system_prompt,
                        evidence_routing_prompt=evidence_routing_prompt,
                        max_parallel_batches=self.config.max_parallel_calls,
                        api_call_semaphore=self._two_stage_api_semaphore,
                        cancel_event=self._cancel_event,
                    )
                except RunCancelled:
                    raise
                except TwoStageValidationError as exc:
                    write_json(
                        document_dir / "extraction.failure.json",
                        {
                            "error": f"TwoStageValidationError: {exc}",
                            "stage": exc.stage,
                            "batch": exc.batch,
                            "raw_response": exc.raw_response,
                            "validation_errors": list(exc.validation_errors),
                        },
                    )
                    self._log(f"  [{stage_dir}/{candidate_id}] {document_id}: two-stage FAILED ({exc})")
                    return document_id, None
                except TwoStageBatchAlignmentError as exc:
                    write_json(
                        document_dir / "extraction.failure.json",
                        {
                            "error": f"TwoStageBatchAlignmentError: {exc}",
                            "stage": exc.stage,
                            "batch": exc.batch,
                            "requested_identities": exc.requested_identities,
                            "covered_identities": exc.covered_identities,
                            "polymer_identities": exc.polymer_identities,
                            "raw_response": exc.raw_response,
                        },
                    )
                    self._log(f"  [{stage_dir}/{candidate_id}] {document_id}: two-stage FAILED ({exc})")
                    return document_id, None
                except TwoStageError as exc:
                    failure: dict[str, Any] = {
                        "error": f"TwoStageError: {exc}",
                        "stage": exc.stage,
                    }
                    if exc.raw_response:
                        failure["raw_response"] = exc.raw_response
                    if exc.diagnostics:
                        failure["diagnostics"] = exc.diagnostics
                    write_json(document_dir / "extraction.failure.json", failure)
                    self._log(f"  [{stage_dir}/{candidate_id}] {document_id}: two-stage FAILED ({exc})")
                    return document_id, None
                except Exception as exc:
                    write_json(document_dir / "extraction.failure.json", {"error": f"{type(exc).__name__}: {exc}"})
                    self._log(f"  [{stage_dir}/{candidate_id}] {document_id}: two-stage FAILED ({type(exc).__name__})")
                    return document_id, None
                if self.config.optimization_mode == "two_stage_evidence_routing_schema_description" and "coverage_plan" in metadata:
                    # Keep the legacy field for shared parser compatibility, while exposing the
                    # user-facing evidence-routing name in new experiment artifacts.
                    metadata["evidence_routing"] = metadata["coverage_plan"]
                if self._cancel_event.is_set():
                    raise RunCancelled("run cancelled after two-stage extraction")
                if self.config.cache_enabled:
                    cache_entry = cache.put(
                        "extraction_two_stage",
                        extraction_inputs,
                        self._prediction_cache_payload(prediction),
                        metadata,
                        source_run_id=run_dir.name,
                    )
                    cache_provenance = {"cache_hit": False, "cache_fingerprint": cache_entry["fingerprint"]}
                else:
                    cache_provenance = {"cache_hit": False}
                self._write_two_stage_prediction_artifacts(document_dir, prediction, metadata, cache_provenance)
            assert prediction is not None
            if self._cancel_event.is_set():
                raise RunCancelled("run cancelled before judging")
            if not prediction.is_valid:
                first = prediction.validation_errors[0] if prediction.validation_errors else {}
                self._log(
                    f"  [warn] [{stage_dir}/{candidate_id}] {document_id}: prediction FAILED schema validation "
                    f"({len(prediction.validation_errors)} errors, first={first.get('path', '?')}) — score is not trustworthy"
                )
            judge_inputs = self._judge_cache_inputs(record, schema, prediction, schema_dsl=schema_dsl)
            judge_cache = SharedCache(self.config.output_root / "_cache", self.config.judge.model)
            cached_judge = judge_cache.get("judge", judge_inputs) if self.config.cache_enabled else None
            if cached_judge is not None:
                try:
                    result = parse_judge_result(
                        json.dumps(cached_judge["result"]), include_error_locations=self.config.include_error_locations, include_pdf=self.config.include_pdf
                    )
                except (KeyError, TypeError, ValueError):
                    cached_judge = None
                else:
                    write_json(document_dir / "judge.result.json", {**asdict(result), **judge_cache.materialize_hit(cached_judge)})
                    self._log(f"  [{stage_dir}/{candidate_id}] {document_id}: judge score={result.score:.1f} (cache)")
                    return document_id, result
            try:
                if self._two_stage_api_semaphore is None:
                    result = self.judge.judge(
                        schema=schema,
                        prediction=prediction,
                        gold=record.pair.gold,
                        pdf_path=record.pair.pdf_path,
                    )
                else:
                    with self._two_stage_api_semaphore:
                        result = self.judge.judge(
                            schema=schema,
                            prediction=prediction,
                            gold=record.pair.gold,
                            pdf_path=record.pair.pdf_path,
                        )
            except Exception as exc:
                first_error = f"{type(exc).__name__}: {exc}"
                result = None
                for retry in range(1, self.config.judge.max_retries + 1):
                    try:
                        if self._two_stage_api_semaphore is None:
                            result = self.judge.judge(
                                schema=schema,
                                prediction=prediction,
                                gold=record.pair.gold,
                                pdf_path=record.pair.pdf_path,
                            )
                        else:
                            with self._two_stage_api_semaphore:
                                result = self.judge.judge(
                                    schema=schema,
                                    prediction=prediction,
                                    gold=record.pair.gold,
                                    pdf_path=record.pair.pdf_path,
                                )
                    except Exception:
                        continue
                    else:
                        self._log(f"  [{stage_dir}/{candidate_id}] {document_id}: judge recovered on retry {retry}, score={result.score:.1f}")
                        break
                if result is None:
                    write_json(document_dir / "judge.failure.json", {"error": first_error})
                    self._log(f"  [{stage_dir}/{candidate_id}] {document_id}: judge FAILED ({first_error.split(':')[0]})")
            if self._cancel_event.is_set():
                raise RunCancelled("run cancelled after judging")
            if result is not None:
                if self.config.cache_enabled:
                    cache_entry = judge_cache.put("judge", judge_inputs, asdict(result), {}, source_run_id=run_dir.name)
                    cache_provenance = {"cache_hit": False, "cache_fingerprint": cache_entry["fingerprint"]}
                else:
                    cache_provenance = {"cache_hit": False}
                write_json(document_dir / "judge.result.json", {**asdict(result), **cache_provenance})
                self._log(f"  [{stage_dir}/{candidate_id}] {document_id}: judge score={result.score:.1f}")
            return document_id, result

        workers = min(self.config.max_parallel_calls, len(records))
        if workers <= 1:
            evaluated = (_evaluate_doc(record) for record in records)
        else:
            executor = DaemonThreadPoolExecutor(max_workers=workers, thread_name_prefix="textgrad-eval-two")
            futures = {executor.submit(_evaluate_doc, record): record for record in records}
            evaluated: list[tuple[str, JudgeResult | None]] = []
            pending = set(futures)
            try:
                while pending:
                    done = {f for f in pending if f.done()}
                    if done:
                        for f in done:
                            evaluated.append(f.result())
                        pending -= done
                    else:
                        time.sleep(0.2)
            except KeyboardInterrupt:
                self._cancel_event.set()
                for future in futures:
                    future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            finally:
                executor.shutdown(wait=False)
        result_map = dict(evaluated)
        valid_scores = [r.score for r in result_map.values() if r is not None]
        if valid_scores:
            self._log(f"  [{stage_dir}/{candidate_id}] mean={sum(valid_scores)/len(valid_scores):.1f} ({len(valid_scores)}/{len(records)} docs)")
        return result_map

    def _extractor_client_for_two_stage(self) -> Any:
        """Return the raw Responses API client used by the extractor.

        ``extract_two_stage`` calls ``client.complete_pdf_json`` directly, so it
        needs the underlying ``ResponsesPdfClient`` rather than the
        ``PdfExtractor`` wrapper. The CLI wires the same client object into
        ``self.extractor`` for two-stage mode (see cli.py).
        """
        if hasattr(self.extractor, "complete_pdf_json"):
            return self.extractor
        inner = getattr(self.extractor, "_client", None)
        if inner is not None and hasattr(inner, "complete_pdf_json"):
            return inner
        raise RuntimeError("Two-stage mode requires the raw ResponsesPdfClient (with complete_pdf_json) as the extractor.")

    def _two_stage_cache_inputs(
        self, record: Any, schema: dict[str, Any], evidence_prompt: str, resolve_prompt: str, schema_dsl: dict[str, Any], evidence_routing_prompt: str | None = None
    ) -> dict[str, Any]:
        return {
            "stage": "two_stage",
            "optimization_mode": self.config.optimization_mode,
            "pdf_sha256": record.pair.pdf_sha256,
            "raw_schema_sha256": sha256_json(schema_dsl),
            "converted_schema_sha256": sha256_json(schema),
            "evidence_system_prompt_sha256": sha256_file(self.config.evidence_system_prompt_path),
            "evidence_prompt_sha256": sha256_json(evidence_prompt),
            "resolve_system_prompt_sha256": sha256_file(self.config.resolve_system_prompt_path),
            "resolve_prompt_sha256": sha256_json(resolve_prompt),
            "coverage_plan_system_prompt_sha256": (
                sha256_file(self.config.coverage_plan_system_prompt_path)
                if self.config.optimization_mode == "two_stage_coverage_plan_schema_description"
                else (
                    sha256_file(self.config.evidence_routing_system_prompt_path)
                    if self.config.optimization_mode == "two_stage_evidence_routing_schema_description"
                    else None
                )
            ),
            "evidence_routing_prompt_sha256": (
                sha256_json(evidence_routing_prompt) if evidence_routing_prompt is not None else None
            ),
            "model_config_sha256": model_config_fingerprint(self.config.extractor),
            "request_format_version": self.config.request_format_version,
            "retry_budget": self.config.extractor.max_retries,
        }

    @staticmethod
    def _write_two_stage_prediction_artifacts(
        document_dir: Path,
        prediction: PredictionArtifact,
        metadata: dict[str, Any],
        provenance: dict[str, Any],
    ) -> None:
        raw_response = prediction.raw_response
        try:
            raw_response = json.dumps(json.loads(raw_response), ensure_ascii=False, indent=2) + "\n"
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        write_text(document_dir / "extraction.response.json", raw_response)
        write_json(document_dir / "validation.json", {"errors": list(prediction.validation_errors)})
        if prediction.is_valid:
            write_json(document_dir / "prediction.json", prediction.parsed_prediction)
        write_json(document_dir / "extraction.metadata.json", {**metadata, "cache": provenance})

    def _two_stage_validation_error_count(self, run_dir: Path, stage_dir: str, document_id: str) -> int | None:
        path = run_dir / stage_dir / "documents" / document_id / "validation.json"
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return len(value.get("errors", []))
        except (OSError, ValueError, json.JSONDecodeError):
            return None

    def _make_two_stage_optimizer(
        self,
        run_dir: Path,
        train_records: tuple[Any, ...],
        base_schema_dsl: dict[str, Any],
        on_round_complete: Any = None,
    ) -> TwoStageOptimizer:
        if self.textgrad_engine is None:
            raise RuntimeError("Two-stage mode requires a TextGrad engine.")
        if self.schema_patch_client is None or not self.schema_patch_system_prompt:
            raise RuntimeError("Two-stage mode requires a schema-patch client and system prompt.")

        def propose_schema_patch(phase_id: str, schema_prompt_text: str, schema_dsl: dict[str, Any], results: dict[str, JudgeResult | None]) -> str:
            feedback = "\n".join(
                f"[{doc_id}] {result.optimization_feedback}"
                for doc_id, result in sorted(results.items())
                if result is not None and result.optimization_feedback
            )
            payload = {
                "schema_dsl": schema_dsl,
                "schema_patch_prompt": schema_prompt_text,
                "training_feedback": feedback,
            }
            with api_operation("schema_patch_proposal"):
                raw = self.schema_patch_client.complete_json(
                    system_prompt=self.schema_patch_system_prompt,
                    payload=payload,
                    reasoning_effort=None,
                )
            phase_dir = run_dir / "training" / phase_id
            phase_dir.mkdir(parents=True, exist_ok=True)
            raw = self._resolve_patch_ambiguities(raw, schema_dsl, debug_path=phase_dir / "patch.debug.json")
            write_text(phase_dir / "schema-prompt.txt", schema_prompt_text)
            write_text(phase_dir / "patch.raw.txt", raw)
            write_json(phase_dir / "schema.proposed.json", schema_dsl)
            return raw

        def evaluate_state(phase_id: str, evidence_prompt: str, resolve_prompt: str, schema_dsl: dict[str, Any]) -> dict[str, JudgeResult | None]:
            phase_dir = run_dir / "training" / phase_id
            phase_dir.mkdir(parents=True, exist_ok=True)
            if phase_id == "round-000/joint":
                write_text(phase_dir / "schema-prompt.txt", self.config.schema_description_initial_prompt_path.read_text(encoding="utf-8").strip())
            write_json(phase_dir / "schema.json", schema_dsl)
            return self._evaluate_two_stage(run_dir, "training", phase_id, evidence_prompt, resolve_prompt, train_records, schema_dsl)

        return TwoStageOptimizer(
            engine=self.textgrad_engine,
            propose_patch=propose_schema_patch,
            evaluate=evaluate_state,
            base_schema_dsl=base_schema_dsl,
            on_round_complete=on_round_complete,
        )

    def _make_evidence_routing_two_stage_optimizer(
        self,
        run_dir: Path,
        train_records: tuple[Any, ...],
        base_schema_dsl: dict[str, Any],
        on_round_complete: Any = None,
    ) -> EvidenceRoutingTwoStageOptimizer:
        """Build the genuinely four-variable two-stage TextGrad optimizer."""
        base = self._make_two_stage_optimizer(run_dir, train_records, base_schema_dsl)

        def evaluate_state(
            phase_id: str,
            evidence_prompt: str,
            routing_prompt: str,
            resolve_prompt: str,
            schema_dsl: dict[str, Any],
        ) -> dict[str, JudgeResult | None]:
            phase_dir = run_dir / "training" / phase_id
            phase_dir.mkdir(parents=True, exist_ok=True)
            if phase_id == "round-000/joint":
                write_text(
                    phase_dir / "schema-prompt.txt",
                    self.config.schema_description_initial_prompt_path.read_text(encoding="utf-8").strip(),
                )
            write_json(phase_dir / "schema.json", schema_dsl)
            return self._evaluate_two_stage(
                run_dir, "training", phase_id, evidence_prompt, resolve_prompt,
                train_records, schema_dsl, evidence_routing_prompt=routing_prompt,
            )

        return EvidenceRoutingTwoStageOptimizer(
            engine=self.textgrad_engine,
            propose_patch=base._propose_patch,
            evaluate=evaluate_state,
            base_schema_dsl=base_schema_dsl,
            on_round_complete=on_round_complete,
        )

    def _load_evidence_routing_two_stage_checkpoint(
        self, run_dir: Path, alt: dict[str, Any]
    ) -> tuple[str, str, str, str, dict[str, Any], list[AlternatingCandidate]]:
        candidates = _deserialize_alternating_candidates(alt.get("candidates"))
        selected = _select_best_alternating_candidate(candidates)
        if selected is None:
            raise RuntimeError("Cannot resume four-variable two-stage training without a scored joint candidate.")
        candidate_dir = run_dir / "training" / selected.candidate_id
        try:
            return (
                (candidate_dir / "schema-prompt.txt").read_text(encoding="utf-8").strip(),
                (candidate_dir / "prompt.evidence.txt").read_text(encoding="utf-8").strip(),
                (candidate_dir / "prompt.evidence-routing.txt").read_text(encoding="utf-8").strip(),
                (candidate_dir / "prompt.resolve.txt").read_text(encoding="utf-8").strip(),
                json.loads((candidate_dir / "schema.json").read_text(encoding="utf-8")),
                candidates,
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Cannot resume four-variable two-stage training: missing artifacts for {selected.candidate_id}"
            ) from exc

    def _recompute_evidence_routing_two_stage_selected(
        self,
        run_dir: Path,
        base_schema_dsl: dict[str, Any],
        candidates: list[AlternatingCandidate],
        requested_rounds: int,
    ) -> tuple[str, str, str, str, dict[str, Any], list[AlternatingCandidate]]:
        in_scope = [candidate for candidate in candidates if _round_of(candidate.candidate_id) <= requested_rounds]
        selected = _select_best_alternating_candidate(in_scope)
        if selected is None:
            return (
                self.config.schema_description_initial_prompt_path.read_text(encoding="utf-8").strip(),
                self.config.evidence_initial_prompt_path.read_text(encoding="utf-8").strip(),
                self.config.evidence_routing_initial_prompt_path.read_text(encoding="utf-8").strip(),
                self.config.resolve_initial_prompt_path.read_text(encoding="utf-8").strip(),
                base_schema_dsl,
                candidates,
            )
        return self._load_evidence_routing_two_stage_checkpoint(
            run_dir, {"candidates": [asdict(candidate) for candidate in in_scope]}
        )

    def _run_evidence_routing_two_stage(
        self,
        run_dir: Path,
        split: DatasetSplit,
        requested_rounds: int,
        smoke_blind_doc: str | None,
        smoke_training: bool,
        checkpoint: dict[str, Any],
    ) -> Path:
        """Four-variable schema/evidence/routing/resolve training and paired blind test."""
        base_schema_dsl = reinforce_required_emission(json.loads(self.config.schema_path.read_text(encoding="utf-8")))
        baseline_schema = self.config.schema_description_initial_prompt_path.read_text(encoding="utf-8").strip()
        baseline_evidence = self.config.evidence_initial_prompt_path.read_text(encoding="utf-8").strip()
        baseline_routing = self.config.evidence_routing_initial_prompt_path.read_text(encoding="utf-8").strip()
        baseline_resolve = self.config.resolve_initial_prompt_path.read_text(encoding="utf-8").strip()
        train_records = split.train if not smoke_training else split.train[:1]
        alt = checkpoint.get("alternating")
        candidates: list[AlternatingCandidate]

        def checkpoint_round(schema_prompt, evidence_prompt, routing_prompt, resolve_prompt, selected_dsl, current_candidates):
            self._checkpoint_alternating_round(
                run_dir, checkpoint, requested_rounds, mode=self.config.optimization_mode,
                schema_prompt=schema_prompt, extraction_prompt=evidence_prompt, selected_dsl=selected_dsl,
                candidates=current_candidates, evidence_prompt=evidence_prompt,
                evidence_routing_prompt=routing_prompt, resolve_prompt=resolve_prompt,
            )

        optimizer = self._make_evidence_routing_two_stage_optimizer(
            run_dir, train_records, base_schema_dsl, on_round_complete=checkpoint_round
        )
        n_used = checkpoint.get("max_iterations_used")
        if isinstance(alt, dict) and isinstance(alt.get("candidates"), list) and alt["candidates"]:
            schema_prompt, evidence_prompt, routing_prompt, resolve_prompt, selected_dsl, candidates = (
                self._load_evidence_routing_two_stage_checkpoint(run_dir, alt)
            )
            last_round = _round_of(candidates[-1].candidate_id)
            start_round = max(last_round + 1, 1)
            if requested_rounds >= start_round:
                self._log(f"[resume] recovered four-variable training through round-{last_round:03d}; continuing at round-{start_round:03d}")
                best_results = self._load_best_two_stage_results(
                    run_dir, candidates, train_records, evidence_prompt, resolve_prompt, selected_dsl,
                    evidence_routing_prompt=routing_prompt,
                )
                schema_prompt, evidence_prompt, routing_prompt, resolve_prompt, selected_dsl, candidates = optimizer.optimize_resume(
                    schema_prompt, evidence_prompt, routing_prompt, resolve_prompt, selected_dsl,
                    best_results, candidates, start_round, requested_rounds,
                )
            elif isinstance(n_used, int) and requested_rounds < n_used:
                self._log(f"[resume] four-variable rounds {n_used}->{requested_rounds}: selecting the best completed round in scope")
        else:
            self._log("[train] four-variable baseline: schema-description + evidence + evidence-routing + resolve")
            schema_prompt, evidence_prompt, routing_prompt, resolve_prompt, selected_dsl, candidates = optimizer.optimize(
                baseline_schema, baseline_evidence, baseline_routing, baseline_resolve, requested_rounds
            )

        schema_prompt, evidence_prompt, routing_prompt, resolve_prompt, selected_dsl, candidates = (
            self._recompute_evidence_routing_two_stage_selected(run_dir, base_schema_dsl, candidates, requested_rounds)
        )
        checkpoint["alternating"] = self._alternating_checkpoint_payload(
            schema_prompt, evidence_prompt, selected_dsl, candidates,
            evidence_prompt=evidence_prompt, evidence_routing_prompt=routing_prompt, resolve_prompt=resolve_prompt,
        )
        checkpoint["alternating"]["mode"] = self.config.optimization_mode
        checkpoint["stages"]["alternating_training_complete"] = True
        checkpoint["max_iterations_used"] = requested_rounds
        checkpoint["max_iterations_requested"] = requested_rounds
        self._write_checkpoint(run_dir, checkpoint)
        self._freeze_alternating(run_dir, base_schema_dsl, schema_prompt, evidence_prompt, selected_dsl)
        prompts_dir = run_dir / "prompts"
        write_text(prompts_dir / "baseline.evidence.txt", baseline_evidence)
        write_text(prompts_dir / "baseline.evidence-routing.txt", baseline_routing)
        write_text(prompts_dir / "baseline.resolve.txt", baseline_resolve)
        write_text(prompts_dir / "final-best.evidence.txt", evidence_prompt)
        write_text(prompts_dir / "final-best.evidence-routing.txt", routing_prompt)
        write_text(prompts_dir / "final-best.resolve.txt", resolve_prompt)
        frozen = json.loads((run_dir / "meta" / "frozen_prompts.json").read_text(encoding="utf-8"))
        frozen.update({
            "baseline_evidence_routing_sha256": sha256_json(baseline_routing),
            "final_best_evidence_routing_sha256": sha256_json(routing_prompt),
        })
        write_json(run_dir / "meta" / "frozen_prompts.json", frozen)
        self._log_alternating_candidates(candidates)

        round_zero = next((c.mean_score for c in candidates if c.candidate_id == "round-000/joint"), None)
        best_mean = max((c.mean_score for c in candidates if c.mean_score is not None), default=None)
        if round_zero is not None and best_mean is not None and best_mean - round_zero < MIN_BLIND_TEST_IMPROVEMENT:
            self._log(f"[skip] four-variable optimization gain {best_mean - round_zero:+.1f} is below blind-test threshold")
            return run_dir

        blind_records = list(split.blind_test)
        if smoke_blind_doc:
            blind_records = [record for record in blind_records if record.pair.document_id == smoke_blind_doc]
            if len(blind_records) != 1:
                raise ValueError("--blind-doc must identify one blind-test document in the frozen split.")
        self._log(f"[blind_test] documents={len(blind_records)} (smoke={smoke_blind_doc is not None})")
        baseline_results = self._evaluate_two_stage(
            run_dir, "blind_test", "baseline", baseline_evidence, baseline_resolve,
            tuple(blind_records), base_schema_dsl, evidence_routing_prompt=baseline_routing,
        )
        checkpoint["stages"]["blind_baseline_complete"] = True
        self._write_checkpoint(run_dir, checkpoint)
        optimized_results = self._evaluate_two_stage(
            run_dir, "blind_test", "optimized", evidence_prompt, resolve_prompt,
            tuple(blind_records), selected_dsl, evidence_routing_prompt=routing_prompt,
        )
        checkpoint["stages"]["blind_optimized_complete"] = True
        paired: list[dict[str, Any]] = []
        for index, record in enumerate(blind_records, start=1):
            doc_id = record.pair.document_id
            baseline = baseline_results.get(doc_id)
            optimized = optimized_results.get(doc_id)
            self._log(f"[blind_test] {index}/{len(blind_records)} {doc_id}: 基线={baseline.score if baseline else '?'} 优化={optimized.score if optimized else '?'}")
            paired.append({
                "document_id": doc_id,
                "baseline_score": baseline.score if baseline else None,
                "optimized_score": optimized.score if optimized else None,
                "baseline_status": "valid" if baseline else "failed",
                "optimized_status": "valid" if optimized else "failed",
                "baseline_validation_errors": self._two_stage_validation_error_count(run_dir, "blind_test/baseline", doc_id),
                "optimized_validation_errors": self._two_stage_validation_error_count(run_dir, "blind_test/optimized", doc_id),
            })
        audit_records = tuple(train_records) + tuple(blind_records)
        audit_results = self.run_gold_audit(run_dir, audit_records, dsl_to_json_schema(selected_dsl)) if self.config.gold_audit_enabled else []
        checkpoint["paired"] = paired
        checkpoint["stages"]["primary_complete"] = True
        self._write_checkpoint(run_dir, checkpoint)
        self.finish_alternating_report(run_dir, candidates, paired, audit_results, base_schema_dsl, selected_dsl)
        checkpoint["stages"]["report_complete"] = True
        self._write_checkpoint(run_dir, checkpoint)
        self._log(f"[done] run_dir={run_dir} valid_paired={sum(1 for item in paired if item["baseline_status"] == "valid" and item["optimized_status"] == "valid")}/{len(paired)}")
        return run_dir
    def _load_best_alternating_results(
        self,
        run_dir: Path,
        candidates: list[AlternatingCandidate],
        train_records: tuple[Any, ...],
        selected_dsl: dict[str, Any],
    ) -> dict[str, JudgeResult | None]:
        best_candidate = _select_best_alternating_candidate(candidates)
        if best_candidate is None:
            raise RuntimeError("Cannot resume alternating training without a scored joint candidate.")
        prompt_path = run_dir / "training" / best_candidate.candidate_id / "prompt.txt"
        if not prompt_path.is_file():
            raise RuntimeError(f"Cannot resume alternating training: missing prompt artifact for {best_candidate.candidate_id}")
        best_prompt = prompt_path.read_text(encoding="utf-8").strip()
        results: dict[str, JudgeResult | None] = {}
        for record in train_records:
            document_id = record.pair.document_id
            cached = self._load_cached_judge_result(run_dir / "training" / best_candidate.candidate_id / "documents" / document_id)
            if cached is not None:
                results[document_id] = cached
                continue
            results[document_id] = self.evaluate_prompt(
                run_dir, best_candidate.candidate_id, best_prompt, (record,), None, schema_dsl=selected_dsl
            ).get(document_id)
        return results

    def _load_two_stage_checkpoint(
        self, run_dir: Path, alt: dict[str, Any]
    ) -> tuple[str, str, str, dict[str, Any], list[AlternatingCandidate]]:
        """Load the training-best two-stage state for continuation.

        Returns ``(schema_prompt, evidence_prompt, resolve_prompt, selected_dsl,
        candidates)``. The three prompts and schema are taken verbatim from the
        highest-scoring valid joint candidate's persisted phase artifacts. This
        aligns continuation, frozen artifacts, and blind-test selection; the
        schema-description prompt for that round is also persisted
        (``schema-prompt.txt``).
        """
        candidates = _deserialize_alternating_candidates(alt.get("candidates"))
        best = _select_best_alternating_candidate(candidates)
        if best is None:
            raise RuntimeError("Cannot resume two-stage training without a scored joint candidate.")
        candidate_dir = run_dir / "training" / best.candidate_id
        try:
            evidence_prompt = (candidate_dir / "prompt.evidence.txt").read_text(encoding="utf-8").strip()
            resolve_prompt = (candidate_dir / "prompt.resolve.txt").read_text(encoding="utf-8").strip()
            schema_prompt = (candidate_dir / "schema-prompt.txt").read_text(encoding="utf-8").strip()
            selected_dsl = json.loads((candidate_dir / "schema.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Cannot resume two-stage training: missing artifacts for {best.candidate_id}"
            ) from exc
        return schema_prompt, evidence_prompt, resolve_prompt, selected_dsl, candidates

    def _load_best_two_stage_results(
        self, run_dir: Path, candidates: list[AlternatingCandidate], train_records: tuple[Any, ...],
        evidence_prompt: str, resolve_prompt: str, selected_dsl: dict[str, Any],
        evidence_routing_prompt: str | None = None,
    ) -> dict[str, JudgeResult | None]:
        """Load per-doc judge results for the last accepted two-stage candidate.

        A document whose persisted judge result cannot be re-parsed (e.g. the
        judge wrote float breakdown values in a non-PDF run re-read under the
        integer rule) is re-evaluated rather than failing the resume.
        """
        best_candidate = _select_best_alternating_candidate(candidates)
        if best_candidate is None:
            raise RuntimeError("Cannot resume two-stage training without a scored joint candidate.")
        need_recompute = []
        results: dict[str, JudgeResult | None] = {}
        for record in train_records:
            document_id = record.pair.document_id
            cached = self._load_cached_judge_result(run_dir / "training" / best_candidate.candidate_id / "documents" / document_id)
            if cached is not None:
                results[document_id] = cached
            else:
                need_recompute.append(record)
        if need_recompute:
            self._log(
                f"[resume] recomputing {len(need_recompute)} two-stage best-round result(s) for {best_candidate.candidate_id}"
            )
            recomputed = self._evaluate_two_stage(
                run_dir, "training", best_candidate.candidate_id, evidence_prompt, resolve_prompt,
                tuple(need_recompute), selected_dsl, evidence_routing_prompt=evidence_routing_prompt,
            )
            results.update(recomputed)
        return results

    def _recompute_two_stage_selected(
        self,
        run_dir: Path,
        base_schema_dsl: dict[str, Any],
        candidates: list[AlternatingCandidate],
        requested_rounds: int,
    ) -> tuple[str, str, str, dict[str, Any], list[AlternatingCandidate]]:
        """Recompute the two-stage selected state over rounds <= requested_rounds.

        Candidate history is preserved; only the selected schema and prompts are
        re-derived from the best candidate's persisted phase artifacts within
        scope. Mirrors ``_recompute_alternating_selected``.
        """
        in_scope = [c for c in candidates if _round_of(c.candidate_id) <= requested_rounds]
        selected = _select_best_alternating_candidate(in_scope)
        if selected is None:
            return (
                self.config.schema_description_initial_prompt_path.read_text(encoding="utf-8").strip(),
                self.config.evidence_initial_prompt_path.read_text(encoding="utf-8").strip(),
                self.config.resolve_initial_prompt_path.read_text(encoding="utf-8").strip(),
                base_schema_dsl,
                candidates,
            )
        candidate_dir = run_dir / "training" / selected.candidate_id
        try:
            return (
                (candidate_dir / "schema-prompt.txt").read_text(encoding="utf-8").strip(),
                (candidate_dir / "prompt.evidence.txt").read_text(encoding="utf-8").strip(),
                (candidate_dir / "prompt.resolve.txt").read_text(encoding="utf-8").strip(),
                json.loads((candidate_dir / "schema.json").read_text(encoding="utf-8")),
                candidates,
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Cannot recompute two-stage selected state: missing artifacts for {selected.candidate_id}"
            ) from exc

    def _load_alternating_checkpoint(
        self, run_dir: Path, alt: dict[str, Any]
    ) -> tuple[str, str, dict[str, Any], list[AlternatingCandidate]]:
        candidates = _deserialize_alternating_candidates(alt.get("candidates"))
        schema_prompt_path = run_dir / "prompts" / "final-best-schema-description.txt"
        extraction_prompt_path = run_dir / "prompts" / "final-best.txt"
        schema_path = run_dir / "schemas" / "final-best-schema.json"
        try:
            schema_prompt = schema_prompt_path.read_text(encoding="utf-8").strip()
            extraction_prompt = extraction_prompt_path.read_text(encoding="utf-8").strip()
            selected_dsl = json.loads(schema_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            selected = _select_best_alternating_candidate(candidates)
            if selected is None:
                raise RuntimeError("Alternating training checkpoint artifacts are missing.")
            candidate_dir = run_dir / "training" / selected.candidate_id
            try:
                schema_prompt = (candidate_dir / "schema-prompt.txt").read_text(encoding="utf-8").strip()
                extraction_prompt = (candidate_dir / "prompt.txt").read_text(encoding="utf-8").strip()
                selected_dsl = json.loads((candidate_dir / "schema.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError("Alternating training checkpoint artifacts are missing.") from exc
        return schema_prompt, extraction_prompt, selected_dsl, candidates

    def _recompute_alternating_selected(
        self,
        run_dir: Path,
        base_schema_dsl: dict[str, Any],
        candidates: list[AlternatingCandidate],
        requested_rounds: int,
    ) -> tuple[str, str, dict[str, Any], list[AlternatingCandidate]]:
        """Recompute the frozen selected state over rounds <= ``requested_rounds``.

        Candidate history is preserved; only the selected schema, schema prompt,
        and extraction prompt are re-derived from persisted phase artifacts. This
        mirrors the single-mode ``_recompute_frozen_best`` semantics.
        """
        in_scope = [c for c in candidates if _round_of(c.candidate_id) <= requested_rounds]
        selected = _select_best_alternating_candidate(in_scope)
        if selected is None:
            return (
                self.config.schema_description_initial_prompt_path.read_text(encoding="utf-8").strip(),
                self.config.initial_prompt_path.read_text(encoding="utf-8").strip(),
                base_schema_dsl,
                candidates,
            )
        candidate_dir = run_dir / "training" / selected.candidate_id
        try:
            return (
                (candidate_dir / "schema-prompt.txt").read_text(encoding="utf-8").strip(),
                (candidate_dir / "prompt.txt").read_text(encoding="utf-8").strip(),
                json.loads((candidate_dir / "schema.json").read_text(encoding="utf-8")),
                candidates,
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot recompute joint alternating state: missing artifacts for {selected.candidate_id}") from exc

    def _load_selected_alternating_state(
        self,
        run_dir: Path,
        base_schema_dsl: dict[str, Any],
        candidates: list[AlternatingCandidate],
    ) -> tuple[str, str, dict[str, Any], list[AlternatingCandidate]]:
        """Load frozen artifacts from the candidate selected by the training gate."""
        selected = _select_best_alternating_candidate(candidates)
        if selected is None:
            return (
                self.config.schema_description_initial_prompt_path.read_text(encoding="utf-8").strip(),
                self.config.initial_prompt_path.read_text(encoding="utf-8").strip(),
                base_schema_dsl,
                candidates,
            )
        candidate_dir = run_dir / "training" / selected.candidate_id
        try:
            return (
                (candidate_dir / "schema-prompt.txt").read_text(encoding="utf-8").strip(),
                (candidate_dir / "prompt.txt").read_text(encoding="utf-8").strip(),
                json.loads((candidate_dir / "schema.json").read_text(encoding="utf-8")),
                candidates,
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Cannot freeze selected joint state: missing artifacts for {selected.candidate_id}"
            ) from exc

    @staticmethod
    def _alternating_checkpoint_payload(
        schema_prompt: str,
        extraction_prompt: str,
        selected_dsl: dict[str, Any],
        candidates: list[AlternatingCandidate],
        *,
        evidence_prompt: str | None = None,
        evidence_routing_prompt: str | None = None,
        resolve_prompt: str | None = None,
    ) -> dict[str, Any]:
        return {
            "mode": "alternating_schema_description",
            "training_algorithm_version": "joint-round-v2",
            "schema_prompt_sha256": sha256_json(schema_prompt),
            "extraction_prompt_sha256": sha256_json(extraction_prompt),
            "evidence_prompt_sha256": sha256_json(evidence_prompt) if evidence_prompt is not None else None,
            "evidence_routing_prompt_sha256": sha256_json(evidence_routing_prompt) if evidence_routing_prompt is not None else None,
            "resolve_prompt_sha256": sha256_json(resolve_prompt) if resolve_prompt is not None else None,
            "selected_schema_sha256": sha256_json(selected_dsl),
            "selected_structural_sha256": structural_fingerprint(selected_dsl),
            "candidates": [asdict(c) for c in candidates],
        }

    def _checkpoint_alternating_round(
        self,
        run_dir: Path,
        checkpoint: dict[str, Any],
        requested_rounds: int,
        *,
        mode: str,
        schema_prompt: str,
        extraction_prompt: str,
        selected_dsl: dict[str, Any],
        candidates: list[AlternatingCandidate],
        evidence_prompt: str | None = None,
        evidence_routing_prompt: str | None = None,
        resolve_prompt: str | None = None,
    ) -> None:
        """Persist the last completed joint round for either alternating mode."""
        payload = self._alternating_checkpoint_payload(
            schema_prompt, extraction_prompt, selected_dsl, candidates,
            evidence_prompt=evidence_prompt, evidence_routing_prompt=evidence_routing_prompt, resolve_prompt=resolve_prompt,
        )
        payload["mode"] = mode
        if evidence_prompt is not None:
            payload["evidence_prompt_sha256"] = sha256_json(evidence_prompt)
        if evidence_routing_prompt is not None:
            payload["evidence_routing_prompt_sha256"] = sha256_json(evidence_routing_prompt)
        if resolve_prompt is not None:
            payload["resolve_prompt_sha256"] = sha256_json(resolve_prompt)
        checkpoint["alternating"] = payload
        checkpoint["max_iterations_used"] = _round_of(candidates[-1].candidate_id)
        checkpoint["max_iterations_requested"] = requested_rounds
        checkpoint["stages"]["alternating_training_complete"] = False
        self._write_checkpoint(run_dir, checkpoint)
        latest = candidates[-1]
        if latest.validation_status != "valid":
            self._log(
                f"[train] {latest.candidate_id}: status={latest.validation_status} "
                f"accepted=✗ reason={latest.decision_reason}"
            )

    def _freeze_alternating(
        self,
        run_dir: Path,
        base_schema_dsl: dict[str, Any],
        schema_prompt: str,
        extraction_prompt: str,
        selected_dsl: dict[str, Any],
    ) -> None:
        prompts_dir = run_dir / "prompts"
        schemas_dir = run_dir / "schemas"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        schemas_dir.mkdir(parents=True, exist_ok=True)
        write_text(prompts_dir / "baseline.txt", self.config.initial_prompt_path.read_text(encoding="utf-8").strip())
        write_text(prompts_dir / "final-best.txt", extraction_prompt)
        write_text(prompts_dir / "baseline-schema-description.txt", self.config.schema_description_initial_prompt_path.read_text(encoding="utf-8").strip())
        write_text(prompts_dir / "final-best-schema-description.txt", schema_prompt)
        write_json(schemas_dir / "baseline-schema.json", base_schema_dsl)
        write_json(schemas_dir / "final-best-schema.json", selected_dsl)
        write_json(
            run_dir / "meta" / "frozen_prompts.json",
            {
                "baseline_sha256": sha256_json(self.config.initial_prompt_path.read_text(encoding="utf-8").strip()),
                "best_sha256": sha256_json(extraction_prompt),
                "baseline_schema_description_sha256": sha256_json(self.config.schema_description_initial_prompt_path.read_text(encoding="utf-8").strip()),
                "final_best_schema_description_sha256": sha256_json(schema_prompt),
                "baseline_schema_sha256": sha256_json(base_schema_dsl),
                "final_best_schema_sha256": sha256_json(selected_dsl),
                "final_best_structural_sha256": structural_fingerprint(selected_dsl),
            },
        )

    def _log_alternating_candidates(self, candidates: list[AlternatingCandidate]) -> None:
        for candidate in candidates:
            scores = ", ".join(
                f"{doc_id}={candidate.document_scores[doc_id]:.1f}" if candidate.document_scores.get(doc_id) is not None else f"{doc_id}=失败"
                for doc_id in sorted(candidate.document_scores)
            )
            mean = f"{candidate.mean_score:.2f}" if candidate.mean_score is not None else "无"
            self._log(f"[train] {candidate.candidate_id}: mean={mean} accepted={'✓' if candidate.accepted else '✗'} {scores}")

    def finish_alternating_report(
        self,
        run_dir: Path,
        candidates: list[AlternatingCandidate],
        blind_records: list[dict[str, Any]],
        audit_results: list[GoldAuditResult],
        base_schema_dsl: dict[str, Any],
        selected_dsl: dict[str, Any],
    ) -> None:
        write_alternating_training_summary(run_dir / "training_summary.csv", candidates)
        blind_summary = write_blind_test_summary(run_dir / "blind_test_documents.csv", blind_records, run_dir=run_dir)
        write_json(run_dir / "blind_test_summary.json", blind_summary)
        analysis_summary = self._write_blind_analysis(run_dir, blind_records)
        audit_status = self._audit_status(run_dir)
        write_alternating_report(
            run_dir / "report.md",
            training=candidates,
            blind_summary=blind_summary,
            audit_results=audit_results,
            audit_status=audit_status,
            analysis_summary=analysis_summary,
            base_schema_sha256=sha256_json(base_schema_dsl),
            selected_schema_sha256=sha256_json(selected_dsl),
            selected_structural_sha256=structural_fingerprint(selected_dsl),
            selected_candidate_id=(
                selected.candidate_id
                if (selected := _select_best_alternating_candidate(candidates)) is not None
                else None
            ),
            optimization_mode=self.config.optimization_mode,
        )
        append_jsonl(
            run_dir / "meta" / "events.jsonl",
            {
                "event": "alternating_report",
                "candidate_count": len(candidates),
                "base_schema_sha256": sha256_json(base_schema_dsl),
                "selected_schema_sha256": sha256_json(selected_dsl),
                "selected_structural_sha256": structural_fingerprint(selected_dsl),
                "changed_description_count": sum(1 for c in candidates if c.changed_description_paths),
            },
        )

    def freeze_prompts(self, run_dir: Path, baseline: str, best: str) -> None:
        write_text(run_dir / "prompts" / "baseline.txt", baseline)
        write_text(run_dir / "prompts" / "final-best.txt", best)
        write_json(run_dir / "meta" / "frozen_prompts.json", {"baseline_sha256": sha256_json(baseline), "best_sha256": sha256_json(best)})

    def finish_report(self, run_dir: Path, candidates: list[Any], blind_records: list[dict[str, Any]], audit_results: list[GoldAuditResult]) -> None:
        write_training_summary(run_dir / "training_summary.csv", candidates)
        blind_summary = write_blind_test_summary(run_dir / "blind_test_documents.csv", blind_records, run_dir=run_dir)
        write_json(run_dir / "blind_test_summary.json", blind_summary)
        analysis_summary = self._write_blind_analysis(run_dir, blind_records)
        audit_status = self._audit_status(run_dir)
        write_report(
            run_dir / "report.md",
            training=candidates,
            blind_summary=blind_summary,
            audit_results=audit_results,
            audit_status=audit_status,
            analysis_summary=analysis_summary,
        )

    def _write_blind_analysis(self, run_dir: Path, blind_records: list[dict[str, Any]]) -> dict[str, Any]:
        pairs = {pair.document_id: pair for pair in discover_pairs(self.config.data_dir)}
        analyses: list[dict[str, Any]] = []
        for record in blind_records:
            document_id = str(record["document_id"])
            pair = pairs.get(document_id)
            baseline = self._analysis_arm(run_dir, "baseline", document_id, record.get("baseline_score"), record.get("baseline_status"))
            optimized = self._analysis_arm(run_dir, "optimized", document_id, record.get("optimized_score"), record.get("optimized_status"))
            audit = self._load_cached_audit_result(run_dir / "gold_audit" / f"{document_id}.json")
            audit_counts = {status: 0 for status in ("supported", "unsupported", "ambiguous", "possible_omission")}
            if audit is not None:
                for finding in audit.findings:
                    audit_counts[finding.status] += 1
            gold = pair.gold if pair is not None else None
            analysis = analyze_blind_document(
                document_id=document_id,
                baseline=baseline,
                optimized=optimized,
                gold=gold,
                audit_counts=audit_counts,
            )
            write_json(run_dir / "analysis" / "documents" / f"{document_id}.json", analysis)
            analyses.append(analysis)
        summary = summarize_blind_analysis(analyses)
        write_json(run_dir / "analysis" / "blind_test_summary.json", summary)
        return summary

    def _analysis_arm(
        self, run_dir: Path, arm: str, document_id: str, score: Any, status: Any
    ) -> dict[str, Any]:
        document_dir = run_dir / "blind_test" / arm / "documents" / document_id
        prediction_path = document_dir / "prediction.json"
        validation_path = document_dir / "validation.json"
        metadata_path = document_dir / "extraction.metadata.json"
        result_path = document_dir / "judge.result.json"
        failure_path = document_dir / "extraction.failure.json"
        if not failure_path.is_file():
            failure_path = document_dir / "judge.failure.json"
        prediction: Any = None
        feedback: str | None = None
        validation_errors: list[Any] = []
        metadata: dict[str, Any] = {}
        failure: str | None = None
        try:
            if prediction_path.is_file():
                prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
            if validation_path.is_file():
                validation_errors = list(json.loads(validation_path.read_text(encoding="utf-8")).get("errors", []))
            if metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if result_path.is_file():
                feedback = json.loads(result_path.read_text(encoding="utf-8")).get("optimization_feedback")
            if failure_path.is_file():
                failure = json.loads(failure_path.read_text(encoding="utf-8")).get("error")
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            failure = f"Artifact read failure: {type(exc).__name__}: {exc}"
        return {
            "score": score,
            "status": status,
            "prediction": prediction,
            "feedback": feedback,
            "validation_errors": validation_errors,
            "repair_attempts": metadata.get("repair_attempts", 0),
            "failure": failure,
            "artifact_paths": {
                "prediction": str(prediction_path.relative_to(run_dir)),
                "judge": str(result_path.relative_to(run_dir)),
                "failure": str(failure_path.relative_to(run_dir)),
            },
        }

    def _audit_status(self, run_dir: Path) -> dict[str, int]:
        artifacts = list((run_dir / "gold_audit").glob("*.json"))
        successful = sum(self._load_cached_audit_result(path) is not None for path in artifacts)
        failed = sum(1 for path in artifacts if self._load_cached_audit_result(path) is None)
        return {"scheduled": len(artifacts), "successful": successful, "failed": failed, "pending": 0}

    def _manifest(self, split: DatasetSplit, schema_text: str) -> dict[str, Any]:
        manifest = {
            "manifest_version": "1",
            "split_algorithm_version": split.algorithm_version,
            "schema": {"path": str(self.config.schema_path), "sha256": sha256_file(self.config.schema_path)},
            "initial_prompt_sha256": sha256_file(self.config.initial_prompt_path),
            "model_fingerprints": {
                "extractor": model_config_fingerprint(self.config.extractor),
                "judge": model_config_fingerprint(self.config.judge),
                "gold_audit": model_config_fingerprint(self.config.gold_audit),
            },
            # ``max_iterations`` is recorded for inspection only; it is excluded
            # from ``config_fingerprint`` so it can be relaxed on resume. The
            # checkpoint tracks the actually-run count (N) vs the requested
            # count (M) separately. The manifest is never rewritten on resume.
            "max_iterations": self.config.max_iterations,
            "config_fingerprint": config_fingerprint(self.config),
            "documents": [
                {
                    "id": record.pair.document_id,
                    "pdf": {"path": str(record.pair.pdf_path), "sha256": record.pair.pdf_sha256},
                    "gold": {"path": str(record.pair.gold_path), "sha256": record.pair.gold_sha256},
                    "complexity": asdict(record.complexity),
                    "rank": record.rank,
                    "split": record.split,
                }
                for record in (*split.train, *split.blind_test)
            ],
        }
        # Only non-single modes add the mode key so legacy single-mode manifests
        # keep matching on resume (mirrors config_fingerprint's conditional keys).
        if self.config.optimization_mode != "single":
            manifest["optimization_mode"] = self.config.optimization_mode
        return manifest















