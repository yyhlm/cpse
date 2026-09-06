"""Two-stage extraction orchestration: index + resolve, per-polymer subtree only.

Stage 1 emits an intermediate structure (``metadata_evidence`` + ``identity_manifest``)
— never the final schema. Stage 2 resolves batches of identities (≤5 each) into
complete polymer records. Batches are
deterministically sliced from ``identity_manifest`` in source order, never overlapping.

Stage-1 manifest duplicates are deterministically collapsed before batching. Later
coverage mismatches are preserved as warnings for the judge rather than rejecting
an otherwise usable candidate. Shared process steps may be referenced or expanded
across identities but never fabricated.
"""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, wait
from dataclasses import dataclass, field
from typing import Any

from .validator import validate_prediction
from .models import PredictionArtifact
from .concurrency import DaemonThreadPoolExecutor, RunCancelled

# Fields that identify a polymer without its content (properties/process/characterization).
IDENTITY_FIELDS = (
    "聚合物分类名称",
    "聚合物分类编码",
    "名称",
    "身份标识",
    "样本形态",
    "结构特征_L1",
    "结构特征_L2",
    "位置索引",
)
CONTENT_FIELDS = ("性质", "工艺流程", "表征")
BATCH_SIZE = 5


class TwoStageError(ValueError):
    """Raised when two-stage extraction or merging fails deterministically."""

    def __init__(
        self,
        message: str,
        *,
        stage: int | None = None,
        raw_response: str = "",
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        self.stage = stage
        self.raw_response = raw_response
        self.diagnostics = diagnostics or {}
        super().__init__(message)


class TwoStageValidationError(TwoStageError):
    """A stage schema failure retaining the raw response for diagnosis."""

    def __init__(
        self,
        *,
        stage: int,
        raw_response: str,
        validation_errors: tuple[dict[str, str], ...],
        batch: int | None = None,
    ) -> None:
        self.batch = batch
        self.validation_errors = validation_errors
        location = f"stage {stage}" if batch is None else f"stage {stage} batch {batch}"
        super().__init__(
            f"{location} failed validation: {len(validation_errors)} errors",
            stage=stage,
            raw_response=raw_response,
            diagnostics={"validation_errors": list(validation_errors), "batch": batch},
        )


class TwoStageBatchAlignmentError(TwoStageError):
    """A schema-valid stage-2 batch that does not match its requested identities."""

    def __init__(
        self,
        *,
        batch: int,
        requested_identities: list[str],
        covered_identities: list[str],
        polymer_identities: list[str],
        raw_response: str,
        reason: str,
    ) -> None:
        self.batch = batch
        self.requested_identities = requested_identities
        self.covered_identities = covered_identities
        self.polymer_identities = polymer_identities
        super().__init__(
            f"stage 2 batch {batch}: {reason}",
            stage=2,
            raw_response=raw_response,
            diagnostics={
                "batch": batch,
                "requested_identities": requested_identities,
                "covered_identities": covered_identities,
                "polymer_identities": polymer_identities,
            },
        )


@dataclass
class TwoStageResult:
    merged_prediction: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    stage1_raw: str = ""
    stage2_batches: list[dict[str, Any]] = field(default_factory=list)


def _identity_key(identity: dict[str, Any]) -> str:
    """Stable key for an identity entry; prefer 身份标识, fall back to 名称."""
    return str(identity.get("身份标识") or identity.get("名称") or "")


def _strip_content(identity: dict[str, Any]) -> dict[str, Any]:
    """Keep only identity fields (drop any content the model may have leaked)."""
    return {k: identity.get(k) for k in IDENTITY_FIELDS if k in identity}


def _parse_json_object(raw_response: str) -> dict[str, Any] | None:
    """Return a parsed object without repairing it, or ``None`` if unusable."""
    try:
        value = json.loads(raw_response)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def stage1_intermediate_schema(root_dsl: dict[str, Any]) -> dict[str, Any]:
    """Build the JSON Schema for stage 1's intermediate output.

    Stage 1 returns ``{metadata_evidence, identity_manifest}`` where
    ``identity_manifest`` is an array of identity-only polymer stubs. It must NOT
    contain 性质/工艺流程/表征 — those are stage 2's job.

    Identity-field property nodes are converted from the repository DSL form
    (``"required": True`` as a per-field flag) to standard JSON Schema so
    ``validate_prediction`` accepts them.
    """
    from .schema_contract import _node_schema

    literature = root_dsl.get("文献信息", {}) or {}
    polymer_item = (root_dsl.get("聚合物", {}) or {}).get("items", {}) or {}
    raw_props = polymer_item.get("properties", {}) or {}
    identity_props = {
        k: _node_schema(raw_props[k], k)
        for k in IDENTITY_FIELDS
        if k in raw_props
    }
    identity_required = [k for k in IDENTITY_FIELDS if raw_props.get(k, {}).get("required") is True and not raw_props.get(k, {}).get("gold_optional")]
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "metadata_evidence": _node_schema(literature, "文献信息"),
            "identity_manifest": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": identity_props,
                    "required": identity_required,
                },
            },
        },
        "required": ["metadata_evidence", "identity_manifest"],
    }


def stage2_batch_schema(root_dsl: dict[str, Any]) -> dict[str, Any]:
    """Build the JSON Schema for a stage 2 batch.

    A batch is ``{covered_identities: [...], 聚合物: [...]}``. The polymer items use the full
    root polymer subtree — stage 2 is where schema constraints are strongest.

    The polymer subtree is converted from the repository DSL form
    (``"required": True`` as a per-field flag) to standard JSON Schema so
    ``validate_prediction`` accepts them.
    """
    from .schema_contract import _node_schema

    polymer_def = root_dsl.get("聚合物", {}) or {}
    polymer_schema = _node_schema(polymer_def, "聚合物")
    batch: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "covered_identities": {
                "type": "array",
                "items": {"type": "string"},
            },
            "聚合物": polymer_schema,
        },
        "required": ["covered_identities", "聚合物"],
    }
    return batch


def coverage_plan_schema() -> dict[str, Any]:
    """Schema for the optional, intermediate-only Stage-1.5 coverage plan."""
    item = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "manifest_id": {"type": "string"},
            "identity_key": {"type": "string"},
            "field_groups": {"type": "array", "items": {"type": "string", "enum": list(CONTENT_FIELDS)}},
            "evidence_anchors": {"type": "array", "items": {"type": "string"}},
            "shared_with_manifest_ids": {"type": "array", "items": {"type": "string"}},
            "reported_test_without_value": {"type": "boolean"},
        },
        "required": ["manifest_id", "identity_key", "field_groups"],
    }
    return {
        "type": "object", "additionalProperties": False,
        "properties": {"coverage_plan": {"type": "array", "items": item}},
        "required": ["coverage_plan"],
    }


def _normalize_coverage_plan(parsed: dict[str, Any] | None, manifest: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Keep only plan rows that are safely routable to the current manifest."""
    if not isinstance(parsed, dict) or not isinstance(parsed.get("coverage_plan"), list):
        return [], ["coverage_plan is not a usable object with a coverage_plan array; continuing without plan"]
    expected = {f"m{index + 1:04d}": _identity_key(identity) for index, identity in enumerate(manifest)}
    usable: list[dict[str, Any]] = []
    warnings: list[str] = []
    for index, entry in enumerate(parsed["coverage_plan"]):
        if not isinstance(entry, dict):
            warnings.append(f"coverage_plan entry {index} is not an object; ignored")
            continue
        manifest_id = entry.get("manifest_id")
        identity_key = entry.get("identity_key")
        if not isinstance(manifest_id, str) or not isinstance(identity_key, str) or expected.get(manifest_id) != identity_key:
            warnings.append(f"coverage_plan entry {index} does not match the current manifest routing; ignored")
            continue
        usable.append(entry)
    return usable, warnings


def _slice_manifest(manifest: list[dict[str, Any]], batch_size: int = BATCH_SIZE) -> list[list[int]]:
    """Deterministically slice manifest indices into non-overlapping batches."""
    return [list(range(i, min(i + batch_size, len(manifest)))) for i in range(0, len(manifest), batch_size)]


def _is_blank_identity_value(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _deduplicate_identity_manifest(manifest: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Collapse repeated identity stubs in stable source order.

    The stage-1 manifest is a routing index, not a scored final record. Exact
    repeated stubs therefore do not justify discarding an otherwise evaluable
    document. Keep the first occurrence and fill only its blank identity fields
    from compatible later stubs. If a duplicate key has conflicting non-blank
    identity fields, retain both records: the local batch index distinguishes
    them without inventing a scientific identifier.
    """
    unique: list[dict[str, Any]] = []
    first_index: dict[str, int] = {}
    positions: dict[str, int] = {}
    warnings: list[str] = []

    for index, identity in enumerate(manifest):
        key = _identity_key(identity)
        existing_index = positions.get(key)
        if existing_index is None:
            unique.append(dict(identity))
            positions[key] = len(unique) - 1
            first_index[key] = index
            continue

        conflicts: list[str] = []
        for field in IDENTITY_FIELDS:
            original = unique[existing_index].get(field)
            duplicate = identity.get(field)
            if (
                not _is_blank_identity_value(original)
                and not _is_blank_identity_value(duplicate)
                and original != duplicate
            ):
                conflicts.append(field)

        if conflicts:
            unique.append(dict(identity))
            warnings.append(
                f"identity_manifest collision retained for {key!r} at index {index}: "
                f"conflicting identity fields: {', '.join(conflicts)}"
            )
            continue

        primary = unique[existing_index]
        filled: list[str] = []
        for field in IDENTITY_FIELDS:
            original = primary.get(field)
            duplicate = identity.get(field)
            if _is_blank_identity_value(original) and not _is_blank_identity_value(duplicate):
                primary[field] = duplicate
                filled.append(field)

        warnings.append(
            f"identity_manifest deduplicated {key!r}: kept first entry at index {first_index[key]}; "
            f"dropped duplicate at index {index}"
        )
        if filled:
            warnings.append(
                f"identity_manifest {key!r}: filled first-entry identity fields from duplicate: {', '.join(filled)}"
            )
        if conflicts:
            warnings.append(
                f"identity_manifest {key!r}: conflicting identity fields retained from first entry: {', '.join(conflicts)}"
            )
    return unique, warnings


def validate_stage2_batch_alignment(
    batch: dict[str, Any],
    requested_identities: list[str],
    *,
    batch_idx: int,
    raw_response: str = "",
) -> list[str]:
    """Report stage-2 coverage mismatches without discarding a usable prediction.

    Identity coverage is an extraction-quality outcome: missing, duplicated, or
    extra records must reach the judge and receive the corresponding penalty.
    Only unparseable/minimally unusable stage output is rejected elsewhere.
    """
    covered = batch.get("covered_identities") or []
    polymers = batch.get("聚合物") or []
    polymer_identities = [_identity_key(poly) for poly in polymers if isinstance(poly, dict)]

    warnings: list[str] = []
    duplicate_requested = sorted(key for key, count in Counter(requested_identities).items() if count > 1)
    if duplicate_requested:
        warnings.append(
            f"batch {batch_idx}: requested identity strings repeat {duplicate_requested}; "
            "distinct local manifest positions are retained"
        )

    def mismatch(label: str, actual: list[str]) -> str | None:
        expected = Counter(requested_identities)
        observed = Counter(actual)
        if observed == expected:
            return None
        missing = sorted((expected - observed).elements())
        extra = sorted((observed - expected).elements())
        duplicated = sorted(key for key, count in observed.items() if count > expected.get(key, 0))
        return f"batch {batch_idx}: {label} differ from requested identities (missing={missing}; extra={extra}; duplicated={duplicated})"

    warnings.extend(
        warning for warning in (mismatch("covered_identities", list(covered)), mismatch("polymer identities", polymer_identities)) if warning
    )
    return warnings


def merge_two_stage(
    stage1: dict[str, Any],
    stage2_batches: list[dict[str, Any]],
    *,
    batch_size: int = BATCH_SIZE,
) -> tuple[dict[str, Any], list[str]]:
    """Merge stage 1 + stage 2 batches into the final root-schema prediction.

    Preserves partial, duplicate, and extra extraction records for judging;
    those conditions are reported as warnings rather than treated as transport
    failures. Raises only when a minimal mergeable structure is absent.
    """
    warnings: list[str] = []
    manifest, manifest_warnings = _deduplicate_identity_manifest(stage1.get("identity_manifest") or [])
    warnings.extend(manifest_warnings)
    manifest_keys = [_identity_key(m) for m in manifest]
    expected = set(manifest_keys)

    merged_polymers: list[dict[str, Any]] = []
    literature = stage1.get("metadata_evidence")
    if not isinstance(literature, dict):
        raise TwoStageError("stage 1 metadata_evidence must be an object")
    seen: dict[str, int] = {}  # identity_key -> batch index that first declared it
    declared_total = 0

    for batch_idx, batch in enumerate(stage2_batches):
        covered = batch.get("covered_identities") or []
        if not isinstance(covered, list):
            raise TwoStageError(f"batch {batch_idx}: covered_identities must be an array")
        declared_total += len(covered)
        # Cross-batch bleed detection (deterministic): identity declared again.
        for key in covered:
            if key in seen:
                warnings.append(
                    f"batch {batch_idx}: identity {key!r} was already covered by batch {seen[key]}; later batch overwrites (model bled across batches)"
                )
            seen[key] = batch_idx
        for poly in batch.get("聚合物", []) or []:
            key = _identity_key(poly)
            if key not in expected:
                warnings.append(f"batch {batch_idx}: polymer {key!r} not in manifest (fabricated?)")
            merged_polymers.append(poly)

    # Exactly-once enforcement: every manifest identity must appear exactly once.
    poly_keys = [_identity_key(p) for p in merged_polymers]
    poly_count: dict[str, int] = {}
    for k in poly_keys:
        poly_count[k] = poly_count.get(k, 0) + 1
    missing = expected - set(poly_count)
    extra = set(poly_count) - expected
    duplicated = {k for k, c in poly_count.items() if c > 1}
    if missing:
        warnings.append(f"missing polymer identities: {sorted(missing)[:5]} (of {len(missing)})")
    if duplicated:
        warnings.append(f"duplicate polymer identities: {sorted(duplicated)[:5]} (of {len(duplicated)})")
    if extra:
        warnings.append(f"extra polymer identities not in manifest: {sorted(extra)[:5]} (of {len(extra)})")

    merged: dict[str, Any] = {}
    merged["文献信息"] = literature
    merged["聚合物"] = merged_polymers
    return merged, warnings


def extract_two_stage(
    *,
    client: Any,
    pdf_path: Any,
    root_schema_dsl: dict[str, Any],
    root_json_schema: dict[str, Any],
    evidence_system_prompt: str,
    evidence_prompt: str,
    resolve_system_prompt: str,
    resolve_prompt: str,
    coverage_plan_system_prompt: str | None = None,
    evidence_routing_prompt: str | None = None,
    batch_size: int = BATCH_SIZE,
    max_parallel_batches: int = 1,
    api_call_semaphore: Any | None = None,
    cancel_event: Any | None = None,
) -> tuple[PredictionArtifact, dict[str, Any]]:
    """Run the two-stage index→resolve extraction and merge.

    Stage 1: one call → metadata_evidence + identity_manifest.
    Stage 2: ⌈N/batch_size⌉ calls. Each batch
    declares covered_identities and returns complete polymer records for them.
    Coverage mismatches are retained as warnings so the judge can score the
    partial/extra output instead of treating it as a transport failure.
    """
    from .schema_contract import dsl_to_json_schema

    if max_parallel_batches < 1:
        raise ValueError("max_parallel_batches must be at least 1")

    metadata: dict[str, Any] = {
        "stages": [],
        "stage_validation_warnings": [],
        "stage_alignment_warnings": [],
        "stage1_manifest_warnings": [],
    }

    def _check_cancelled() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise RunCancelled("two-stage extraction cancelled")

    def _complete_pdf_json(**kwargs: Any) -> tuple[str, dict[str, Any]]:
        _check_cancelled()
        def call_client() -> tuple[str, dict[str, Any]]:
            try:
                return client.complete_pdf_json(**kwargs)
            except TypeError as exc:
                # Older compatible transports do not expose the optional
                # capability-scope argument. Keep the protocol backward
                # compatible; production Responses/Gemini clients receive it.
                if "json_schema_capability_scope" not in str(exc):
                    raise
                fallback_kwargs = dict(kwargs)
                fallback_kwargs.pop("json_schema_capability_scope", None)
                return client.complete_pdf_json(**fallback_kwargs)
        if api_call_semaphore is None:
            return call_client()
        while not api_call_semaphore.acquire(timeout=0.2):
            _check_cancelled()
        try:
            _check_cancelled()
            return call_client()
        finally:
            api_call_semaphore.release()

    stage1_schema = stage1_intermediate_schema(root_schema_dsl)
    stage1_raw, stage1_meta = _complete_pdf_json(
        pdf_path=pdf_path,
        system_prompt=evidence_system_prompt,
        payload={"schema": json.dumps(stage1_schema, ensure_ascii=False), "extraction_prompt": evidence_prompt},
        json_schema=stage1_schema,
        json_schema_name="two_stage_stage1",
        json_schema_capability_scope="two_stage_stage1",
    )
    stage1_artifact = validate_prediction(stage1_raw, stage1_schema)
    if not stage1_artifact.is_valid:
        stage1_parsed = _parse_json_object(stage1_raw)
        if stage1_parsed is None:
            raise TwoStageValidationError(
                stage=1,
                raw_response=stage1_raw,
                validation_errors=stage1_artifact.validation_errors,
            )
        metadata["stage_validation_warnings"].append(
            {"stage": 1, "validation_errors": list(stage1_artifact.validation_errors)}
        )
    else:
        stage1_parsed = stage1_artifact.parsed_prediction or {}
    manifest = stage1_parsed.get("identity_manifest") or []
    if (
        not isinstance(stage1_parsed.get("metadata_evidence"), dict)
        or not isinstance(manifest, list)
        or not manifest
        or any(not isinstance(identity, dict) or not _identity_key(identity) for identity in manifest)
    ):
        raise TwoStageError(
            "stage 1 lacks the minimum object metadata or usable identity_manifest required to continue",
            stage=1,
            raw_response=stage1_raw,
            diagnostics={"parsed_stage1": stage1_parsed},
        )
    raw_manifest_count = len(manifest)
    manifest, manifest_warnings = _deduplicate_identity_manifest(manifest)
    stage1_parsed = {**stage1_parsed, "identity_manifest": manifest}
    metadata["stage1_manifest_warnings"].extend(manifest_warnings)
    metadata["stages"].append(
        {
            "stage": 1,
            "usage": stage1_meta.get("usage"),
            "manifest_count": len(manifest),
            "raw_manifest_count": raw_manifest_count,
        }
    )

    batches = _slice_manifest(manifest, batch_size)
    batch_schema = stage2_batch_schema(root_schema_dsl)
    coverage_plan_entries: list[dict[str, Any]] = []
    metadata["coverage_plan"] = {"status": "disabled", "warnings": []}
    if coverage_plan_system_prompt is not None:
        plan_schema = coverage_plan_schema()
        routing_system_prompt = coverage_plan_system_prompt
        if evidence_routing_prompt is not None:
            routing_system_prompt = (
                f"{coverage_plan_system_prompt.rstrip()}\n\n"
                "Evidence-routing instructions (learned; apply them without changing the protocol):\n"
                f"{evidence_routing_prompt.strip()}"
            )
        planned_manifest = [
            {"manifest_id": f"m{index + 1:04d}", **_strip_content(identity)}
            for index, identity in enumerate(manifest)
        ]
        try:
            plan_raw, plan_meta = _complete_pdf_json(
                pdf_path=pdf_path,
                system_prompt=routing_system_prompt,
                payload={"coverage_plan_schema": json.dumps(plan_schema, ensure_ascii=False), "identity_manifest": planned_manifest},
                json_schema=plan_schema,
                json_schema_name="two_stage_coverage_plan",
                json_schema_capability_scope="two_stage_coverage_plan",
            )
        except (KeyboardInterrupt, RunCancelled):
            raise
        except Exception as exc:
            metadata["coverage_plan"] = {
                "status": "fallback_without_plan",
                "entry_count": 0,
                "warnings": [f"coverage_plan request failed ({type(exc).__name__}: {exc}); continuing without plan"],
            }
            metadata["stages"].append({"stage": "1.5", "entry_count": 0, "status": "request_failed"})
        else:
            plan_artifact = validate_prediction(plan_raw, plan_schema)
            plan_parsed = plan_artifact.parsed_prediction if plan_artifact.is_valid else _parse_json_object(plan_raw)
            coverage_plan_entries, plan_warnings = _normalize_coverage_plan(plan_parsed, manifest)
            if not plan_artifact.is_valid:
                plan_warnings.insert(0, f"coverage_plan schema validation warning: {len(plan_artifact.validation_errors)} errors")
            metadata["coverage_plan"] = {
                "status": "used" if coverage_plan_entries else "fallback_without_plan",
                "entry_count": len(coverage_plan_entries), "warnings": plan_warnings,
                "raw_response": plan_raw, "validation_errors": list(plan_artifact.validation_errors), "usage": plan_meta.get("usage"),
            }
            metadata["stages"].append({"stage": "1.5", "usage": plan_meta.get("usage"), "entry_count": len(coverage_plan_entries)})
    def _resolve_batch(batch_idx: int, indices: list[int]) -> tuple[int, list[str], str, dict[str, Any]]:
        covered_keys = [_identity_key(manifest[i]) for i in indices]
        covered_manifest_ids = [f"m{i + 1:04d}" for i in indices]
        batch_payload = {
            "schema": json.dumps(batch_schema, ensure_ascii=False),
            "extraction_prompt": resolve_prompt,
            "covered_identities": covered_keys,
            "covered_manifest_ids": covered_manifest_ids,
            "identity_manifest": [
                {"manifest_id": manifest_id, **_strip_content(manifest[index])}
                for manifest_id, index in zip(covered_manifest_ids, indices)
            ],
        }
        if coverage_plan_system_prompt is not None:
            requested_ids = set(covered_manifest_ids)
            batch_payload["coverage_plan"] = [entry for entry in coverage_plan_entries if entry.get("manifest_id") in requested_ids]
        raw, batch_meta = _complete_pdf_json(
            pdf_path=pdf_path,
            system_prompt=resolve_system_prompt,
            payload=batch_payload,
            json_schema=batch_schema,
            json_schema_name=f"two_stage_stage2_batch_{batch_idx}",
            json_schema_capability_scope="two_stage_stage2",
        )
        return batch_idx, covered_keys, raw, batch_meta

    batch_outputs: dict[int, tuple[list[str], str, dict[str, Any]]] = {}
    batch_workers = min(max_parallel_batches, len(batches))
    if batch_workers <= 1:
        for batch_idx, indices in enumerate(batches):
            _check_cancelled()
            result_idx, covered_keys, raw, batch_meta = _resolve_batch(batch_idx, indices)
            batch_outputs[result_idx] = (covered_keys, raw, batch_meta)
    else:
        executor = DaemonThreadPoolExecutor(max_workers=batch_workers, thread_name_prefix="textgrad-stage2")
        futures = []
        try:
            futures = [executor.submit(_resolve_batch, batch_idx, indices) for batch_idx, indices in enumerate(batches)]
            pending = set(futures)
            while pending:
                _check_cancelled()
                done, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
                for future in done:
                    result_idx, covered_keys, raw, batch_meta = future.result()
                    batch_outputs[result_idx] = (covered_keys, raw, batch_meta)
        except (KeyboardInterrupt, RunCancelled):
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            executor.shutdown(wait=False)

    stage2_results: list[dict[str, Any]] = []
    for batch_idx in range(len(batches)):
        covered_keys, raw, batch_meta = batch_outputs[batch_idx]
        batch_artifact = validate_prediction(raw, batch_schema)
        if not batch_artifact.is_valid:
            batch_parsed = _parse_json_object(raw)
            if (
                batch_parsed is None
                or not isinstance(batch_parsed.get("covered_identities"), list)
                or not isinstance(batch_parsed.get("聚合物"), list)
            ):
                raise TwoStageValidationError(
                    stage=2,
                    batch=batch_idx,
                    raw_response=raw,
                    validation_errors=batch_artifact.validation_errors,
                )
            metadata["stage_validation_warnings"].append(
                {
                    "stage": 2,
                    "batch": batch_idx,
                    "validation_errors": list(batch_artifact.validation_errors),
                }
            )
        else:
            batch_parsed = batch_artifact.parsed_prediction or {}
        alignment_warnings = validate_stage2_batch_alignment(
            batch_parsed,
            covered_keys,
            batch_idx=batch_idx,
            raw_response=raw,
        )
        metadata["stage_alignment_warnings"].extend(alignment_warnings)
        stage2_results.append(batch_parsed)
        metadata["stages"].append({"stage": 2, "batch": batch_idx, "usage": batch_meta.get("usage"), "covered": len(covered_keys)})

    merged, warnings = merge_two_stage(stage1_parsed, stage2_results, batch_size=batch_size)
    metadata["warnings"] = warnings
    metadata["batch_count"] = len(batches)
    # Preserve parseable schema inconsistencies exactly as returned. The final
    # validator records them for reports and the judge still receives the JSON;
    # no required-field fill or other repair is performed in two-stage mode.
    final_artifact = validate_prediction(json.dumps(merged, ensure_ascii=False), root_json_schema)
    metadata["merged_raw"] = json.dumps(merged, ensure_ascii=False)
    return final_artifact, metadata



