"""Offline accounting for extraction usage preserved in experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_REFERENCE_PRICING_USD_PER_MTOK = {
    # Snapshot used only for reproducible reference estimates. Provider-side
    # invoices remain authoritative for actual billed cost.
    "gpt-5.6-sol": {"input": 4.0, "cached_input": 0.4, "output": 20.0},
    "gpt-5.6-terra": {"input": 2.0, "cached_input": 0.2, "output": 12.0},
}


def _empty_tokens() -> dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "reasoning_tokens": 0}


def _add_usage(total: dict[str, int], usage: dict[str, Any]) -> None:
    total["input_tokens"] += int(usage.get("input_tokens") or 0)
    total["output_tokens"] += int(usage.get("output_tokens") or 0)
    total["total_tokens"] += int(usage.get("total_tokens") or 0)
    details = usage.get("output_tokens_details")
    if isinstance(details, dict):
        total["reasoning_tokens"] += int(details.get("reasoning_tokens") or 0)


def _metadata_usages(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    usage = metadata.get("usage")
    if isinstance(usage, dict):
        return [usage]
    stages = metadata.get("stages")
    if not isinstance(stages, list):
        return []
    return [item["usage"] for item in stages if isinstance(item, dict) and isinstance(item.get("usage"), dict)]


def summarize_run_usage(run_dir: Path) -> dict[str, Any]:
    """Summarize extraction token metadata without treating cache reuse as fresh cost."""
    logical = _empty_tokens()
    fresh = _empty_tokens()
    logical_calls = 0
    fresh_calls = 0
    cached_documents = 0
    no_usage = 0

    for path in sorted(run_dir.rglob("extraction.metadata.json")):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        usages = _metadata_usages(metadata)
        if not usages:
            no_usage += 1
            continue
        cache = metadata.get("cache")
        cache_hit = isinstance(cache, dict) and bool(cache.get("cache_hit"))
        if cache_hit:
            cached_documents += 1
        for usage in usages:
            _add_usage(logical, usage)
            logical_calls += 1
            if not cache_hit:
                _add_usage(fresh, usage)
                fresh_calls += 1

    result = {
        "logical_workload": {"call_count": logical_calls, **logical},
        "observed_fresh_extraction": {"call_count": fresh_calls, **fresh},
        "cached_extraction_documents": cached_documents,
        "metadata_files_without_usage": no_usage,
        "scope_note": (
            "Extraction usage is reconstructed from extraction.metadata.json. "
            "Judge, optimizer, failed/retried calls, and latency are excluded when "
            "their usage was not persisted."
        ),
    }
    ledger_path = run_dir / "meta" / "api_calls.jsonl"
    if ledger_path.is_file():
        rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        result.update(_summarize_complete_ledger(rows))
        result["scope_note"] = (
            "Complete API ledger counts every persisted physical attempt. "
            "Attempts without returned usage remain unknown rather than being treated as zero. "
            "Cost is a reproducible public-list-price estimate, not an invoice."
        )
    return result


def _summarize_complete_ledger(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        logical_ids = {str(row.get("logical_call_id")) for row in items if row.get("logical_call_id")}
        successful = [row for row in items if row.get("status") == "success"]
        unknown = [row for row in items if not row.get("usage_available")]
        input_tokens = sum(int(row.get("input_tokens") or 0) for row in successful)
        cached_tokens = sum(int(row.get("cached_input_tokens") or 0) for row in successful)
        output_tokens = sum(int(row.get("output_tokens") or 0) for row in successful)
        reasoning_tokens = sum(int(row.get("reasoning_tokens") or 0) for row in successful)
        total_tokens = sum(int(row.get("total_tokens") or 0) for row in successful)
        cost = sum(_reference_cost(row) for row in successful)
        return {
            "logical_call_count": len(logical_ids),
            "physical_attempt_count": len(items),
            "successful_attempt_count": len(successful),
            "failed_attempt_count": len(items) - len(successful),
            "retry_attempt_count": max(0, len(items) - len(logical_ids)),
            "unknown_usage_attempt_count": len(unknown),
            "input_tokens": input_tokens,
            "cached_input_tokens": cached_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "total_tokens": total_tokens,
            "attempt_latency_seconds": round(sum(float(row.get("duration_seconds") or 0.0) for row in items), 6),
            "retry_backoff_seconds": round(sum(float(row.get("retry_backoff_seconds") or 0.0) for row in items), 6),
            "estimated_reference_cost_usd": round(cost, 9),
        }

    by_operation: dict[str, Any] = {}
    for operation in sorted({str(row.get("operation") or "unspecified") for row in rows}):
        by_operation[operation] = summarize([row for row in rows if str(row.get("operation") or "unspecified") == operation])
    return {
        "complete_api_ledger": summarize(rows),
        "by_operation": by_operation,
        "pricing_reference": {
            "basis": "public_list_price_snapshot_2026-09-01",
            "usd_per_million_tokens": _REFERENCE_PRICING_USD_PER_MTOK,
        },
    }


def _reference_cost(row: dict[str, Any]) -> float:
    pricing = _REFERENCE_PRICING_USD_PER_MTOK.get(str(row.get("model_requested") or ""))
    if pricing is None:
        return 0.0
    input_tokens = int(row.get("input_tokens") or 0)
    cached = min(input_tokens, int(row.get("cached_input_tokens") or 0))
    uncached = input_tokens - cached
    output = int(row.get("output_tokens") or 0)
    return (
        uncached * pricing["input"]
        + cached * pricing["cached_input"]
        + output * pricing["output"]
    ) / 1_000_000
