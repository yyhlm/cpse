from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .artifacts import write_csv, write_text
from .models import AlternatingCandidate, CandidateSummary, GoldAuditResult


def write_training_summary(path: Path, candidates: Iterable[CandidateSummary]) -> None:
    candidates = list(candidates)
    document_ids = sorted({document_id for item in candidates for document_id in item.document_scores})
    rows = []
    for item in candidates:
        row: dict[str, Any] = {
            "candidate_id": item.candidate_id,
            "prompt_hash": item.prompt_hash,
            "parent_candidate_id": item.parent_candidate_id or "",
            "mean_score": "" if item.mean_score is None else item.mean_score,
            "accepted": item.accepted,
            "decision_reason": item.decision_reason,
        }
        row.update({f"score_{doc_id}": "" if item.document_scores.get(doc_id) is None else item.document_scores[doc_id] for doc_id in document_ids})
        rows.append(row)
    write_csv(
        path,
        rows,
        ["candidate_id", "prompt_hash", "parent_candidate_id", "mean_score", "accepted", "decision_reason"]
        + [f"score_{doc_id}" for doc_id in document_ids],
    )


def write_alternating_training_summary(path: Path, candidates: Iterable[AlternatingCandidate]) -> None:
    """Write an additive training CSV for alternating schema-description runs.

    Columns preserve the legacy orientation (candidate id, hashes, parent, mean,
    accepted, decision) and add phase/schema/patch metadata so report-only and
    downstream analysis can reconstruct each phase without re-deriving it.
    """
    candidates = list(candidates)
    document_ids = sorted({document_id for item in candidates for document_id in item.document_scores})
    rows = []
    for item in candidates:
        row: dict[str, Any] = {
            "candidate_id": item.candidate_id,
            "phase": item.phase,
            "parent_candidate_id": item.parent_candidate_id or "",
            "schema_prompt_hash": item.schema_prompt_hash,
            "extraction_prompt_hash": item.extraction_prompt_hash,
            "schema_sha256": item.schema_sha256,
            "structural_sha256": item.structural_sha256,
            "patch_sha256": item.patch_sha256 or "",
            "changed_description_paths": ";".join(item.changed_description_paths),
            "validation_status": item.validation_status,
            "mean_score": "" if item.mean_score is None else item.mean_score,
            "accepted": item.accepted,
            "decision_reason": item.decision_reason,
        }
        row.update({f"score_{doc_id}": "" if item.document_scores.get(doc_id) is None else item.document_scores[doc_id] for doc_id in document_ids})
        rows.append(row)
    write_csv(
        path,
        rows,
        [
            "candidate_id",
            "phase",
            "parent_candidate_id",
            "schema_prompt_hash",
            "extraction_prompt_hash",
            "schema_sha256",
            "structural_sha256",
            "patch_sha256",
            "changed_description_paths",
            "validation_status",
            "mean_score",
            "accepted",
            "decision_reason",
        ]
        + [f"score_{doc_id}" for doc_id in document_ids],
    )


def write_blind_test_summary(
    path: Path,
    records: list[dict[str, Any]],
    *,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    rows = []
    valid_deltas: list[float] = []
    trusted_deltas: list[float] = []
    untrusted = 0
    for record in records:
        baseline, optimized = record.get("baseline_score"), record.get("optimized_score")
        delta = optimized - baseline if isinstance(baseline, (int, float)) and isinstance(optimized, (int, float)) else None
        rows.append({**record, "delta": "" if delta is None else delta})
        if delta is not None:
            valid_deltas.append(delta)
        b_err = record.get("baseline_validation_errors")
        o_err = record.get("optimized_validation_errors")
        has_error = (b_err is not None and b_err > 0) or (o_err is not None and o_err > 0)
        if has_error:
            untrusted += 1
        elif delta is not None:
            # Trusted: both arms passed schema validation AND the pair is comparable.
            trusted_deltas.append(delta)
    write_csv(
        path,
        rows,
        ["document_id", "baseline_score", "optimized_score", "delta", "baseline_status", "optimized_status",
         "baseline_validation_errors", "optimized_validation_errors"],
    )
    summary = {
        "document_count": len(records),
        "paired_valid_count": len(valid_deltas),
        "mean_paired_delta": sum(valid_deltas) / len(valid_deltas) if valid_deltas else None,
        "wins": sum(delta > 0 for delta in valid_deltas),
        "ties": sum(delta == 0 for delta in valid_deltas),
        "losses": sum(delta < 0 for delta in valid_deltas),
        "unpaired_or_failed_count": len(records) - len(valid_deltas),
        "untrusted_paired_count": untrusted,
        "trusted_paired_count": len(trusted_deltas),
        "trusted_mean_delta": sum(trusted_deltas) / len(trusted_deltas) if trusted_deltas else None,
        "trusted_median_delta": _median(trusted_deltas),
        "trusted_wins": sum(delta > 0 for delta in trusted_deltas),
        "trusted_ties": sum(delta == 0 for delta in trusted_deltas),
        "trusted_losses": sum(delta < 0 for delta in trusted_deltas),
    }
    if run_dir is not None:
        dim = _dimension_breakdown(run_dir, records)
        summary.update(dim)
    return summary


def _dimension_breakdown(run_dir: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-dimension score_breakdown and absolute scores across both arms.

    Reads each blind-test document's judge.result.json under
    ``run_dir/blind_test/{baseline,optimized}/documents/<id>/`` and averages the
    four judge dimensions (document_sample / process / properties /
    characterization) per arm, plus the absolute baseline/optimized scores. Only
    documents with a judge.result.json present on BOTH arms contribute.
    """
    dims = ("document_sample", "process", "properties", "characterization")
    base_sb: dict[str, list[float]] = {d: [] for d in dims}
    opt_sb: dict[str, list[float]] = {d: [] for d in dims}
    base_abs: list[float] = []
    opt_abs: list[float] = []
    for record in records:
        doc_id = record.get("document_id")
        if not doc_id:
            continue
        b_sb, b_score = _load_judge_breakdown(run_dir, "baseline", doc_id)
        o_sb, o_score = _load_judge_breakdown(run_dir, "optimized", doc_id)
        if b_sb is None or o_sb is None:
            continue
        for d in dims:
            base_sb[d].append(float(b_sb.get(d, 0) or 0))
            opt_sb[d].append(float(o_sb.get(d, 0) or 0))
        if b_score is not None:
            base_abs.append(float(b_score))
        if o_score is not None:
            opt_abs.append(float(o_score))
    dimension_deltas: dict[str, dict[str, float | None]] = {}
    for d in dims:
        if base_sb[d] and opt_sb[d]:
            bm = sum(base_sb[d]) / len(base_sb[d])
            om = sum(opt_sb[d]) / len(opt_sb[d])
            dimension_deltas[d] = {"baseline_mean": bm, "optimized_mean": om, "delta": om - bm}
        else:
            dimension_deltas[d] = {"baseline_mean": None, "optimized_mean": None, "delta": None}
    return {
        "baseline_abs_mean": (sum(base_abs) / len(base_abs)) if base_abs else None,
        "optimized_abs_mean": (sum(opt_abs) / len(opt_abs)) if opt_abs else None,
        "dimension_breakdown_doc_count": len(base_sb["document_sample"]) if base_sb["document_sample"] else 0,
        "dimension_deltas": dimension_deltas,
    }


def _load_judge_breakdown(run_dir: Path, arm: str, doc_id: str) -> tuple[dict[str, Any] | None, float | None]:
    """Load score_breakdown and score from a blind-test judge result, or (None, None)."""
    p = run_dir / "blind_test" / arm / "documents" / doc_id / "judge.result.json"
    try:
        import json as _json
        value = _json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None, None
    breakdown = value.get("score_breakdown") if isinstance(value.get("score_breakdown"), dict) else None
    score = value.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        score = None
    return breakdown, score


def write_report(
    path: Path,
    *,
    training: list[CandidateSummary],
    blind_summary: dict[str, Any],
    audit_results: Iterable[GoldAuditResult],
    audit_status: dict[str, int] | None = None,
    analysis_summary: dict[str, Any] | None = None,
) -> None:
    audit_results = list(audit_results)
    best = None
    for item in training:
        if item.mean_score is None:
            continue
        if best is None or item.mean_score >= best.mean_score:
            best = item
    lines = [
        "# TextGrad 金标验证实验报告",
        "",
        *_blind_conclusion_lines(blind_summary),
        "",
        "## 训练过程",
        "",
        "| 候选编号 | 均分 | 是否接受 | 原因 | 父候选 |",
        "|---------|------|---------|------|-------|",
    ]
    for item in training:
        mean = f"{item.mean_score:.2f}" if item.mean_score is not None else "失败"
        parent = item.parent_candidate_id or "-"
        reason = item.decision_reason
        lines.append(f"| {item.candidate_id} | {mean} | {'✓' if item.accepted else '✗'} | {reason} | {parent} |")
    if analysis_summary:
        lines.extend(_analysis_report_lines(analysis_summary))
    accepted_str = "、".join(item.candidate_id for item in training if item.accepted)
    lines.extend(
        [
            "",
            f"- 最终选定的最佳候选: {best.candidate_id if best else '无'}",
            f"- 接受的候选: {accepted_str or '无'}",
            f"- 最佳训练均分: {_format_number(best.mean_score if best else None)}",
            "",
            "训练分数仅作诊断,实验结论以盲测配对结果为准。",
            "",
            *_audit_lines(audit_results, audit_status),
        ]
    )
    write_text(path, "\n".join(lines))


def write_alternating_report(
    path: Path,
    *,
    training: list[AlternatingCandidate],
    blind_summary: dict[str, Any],
    audit_results: Iterable[GoldAuditResult],
    audit_status: dict[str, int] | None = None,
    analysis_summary: dict[str, Any] | None = None,
    base_schema_sha256: str | None = None,
    selected_schema_sha256: str | None = None,
    selected_structural_sha256: str | None = None,
    selected_candidate_id: str | None = None,
    optimization_mode: str = "alternating_schema_description",
) -> None:
    audit_results = list(audit_results)
    best_joint = next((c for c in training if c.candidate_id == selected_candidate_id), None)
    if best_joint is None:
        # Compatibility for reports reconstructed from legacy checkpoints which
        # did not persist an explicit selected candidate identifier.
        best_joint = next(
            (c for c in reversed(training) if c.phase == "joint" and c.accepted),
            next((c for c in training if c.phase == "joint"), None),
        )
    mode_descriptions = {
        "description_only": (
            "# TextGrad schema-description-only 消融实验报告",
            "每轮只更新 schema-description 补丁提示词；抽取提示词逐字冻结，用于隔离 schema 语义描述的贡献。",
        ),
        "opro_prompt_only": (
            "OPRO 外部基线：schema 与 description 冻结，优化模型根据历史候选及其训练分数"
            "逐轮提出 extraction prompt；不使用 TextGrad 或裁判自然语言反馈。"
        ),
        "gepa_prompt_only": (
            "官方 GEPA prompt-only 外部基线：schema 与 description 冻结，通过自定义 GEPAAdapter "
            "向反思模型提供逐篇分数、分项分数和裁判反馈。"
        ),
        "mipro_v2_instruction_only": (
            "官方 DSPy MIPROv2 instruction-only 外部基线：bootstrapped/labeled demonstrations 均为0，"
            "只搜索 extraction instruction，schema 与 description 冻结。"
        ),
        "mipro_v2": (
            "官方 DSPy MIPROv2 instruction+labeled-demo 基线：从三篇训练PDF的确定性文本和Gold中"
            "选择最多一个demonstration；bootstrapped demos为0。"
        ),
        "alternating_schema_description": (
            "# TextGrad 联合 schema-description 优化实验报告",
            "每轮从同一已接受状态的反馈中独立更新 schema-description 与抽取提示词，再共同重抽取并评分；抽取仍为单阶段 PDF→JSON。",
        ),
        "two_stage_alternating_schema_description": (
            "# TextGrad 两阶段联合优化实验报告",
            "每轮联合更新 schema-description、evidence/index 与 resolve 三个提示变量，再执行 identity-first 两阶段抽取并评分。",
        ),
    }
    title, mode_description = mode_descriptions.get(
        optimization_mode,
        ("# TextGrad schema-description 优化实验报告", "按冻结运行产物生成报告。"),
    )
    lines = [
        title,
        "",
        f"本报告来自 `optimization_mode: {optimization_mode}`。{mode_description}",
        "",
        *_blind_conclusion_lines(blind_summary),
        "",
        "## 交替训练过程",
        "",
        "| 轮次 | 候选类型 | 均分 | 是否接受 | 校验状态 | 变更路径 | 原因 |",
        "|------|---------|------|---------|---------|---------|------|",
    ]
    for item in training:
        mean = f"{item.mean_score:.2f}" if item.mean_score is not None else "无"
        changed = ";".join(item.changed_description_paths) or "-"
        lines.append(
            f"| {item.candidate_id} | {item.phase} | {mean} | {'✓' if item.accepted else '✗'} | "
            f"{item.validation_status} | {changed} | {item.decision_reason} |"
        )
    if analysis_summary:
        lines.extend(_analysis_report_lines(analysis_summary))
    lines.extend(
        [
            "",
            f"- 最终选定的联合阶段: {best_joint.candidate_id if best_joint else '无'}",
            f"- 最终最佳训练均分: {_format_number(best_joint.mean_score if best_joint else None)}",
            f"- 基线 schema sha256: {base_schema_sha256 or '-'}",
            f"- 选定 schema sha256: {selected_schema_sha256 or '-'}",
            f"- 选定 schema 结构指纹(忽略 description): {selected_structural_sha256 or '-'}",
            "",
            "训练分数仅作诊断,实验结论以盲测配对结果为准。盲测 baseline 臂使用基线 schema，"
            "optimized 臂使用最终选定 schema 与该模式冻结的最优提示词状态。",
            "",
            *_audit_lines(audit_results, audit_status),
        ]
    )
    write_text(path, "\n".join(lines))


def _blind_conclusion_lines(blind_summary: dict[str, Any]) -> list[str]:
    lines = [
        "## 盲测主结论",
        "",
        f"- 盲测文档数: {blind_summary['document_count']}",
        f"- 有效配对比较数: {blind_summary['paired_valid_count']}",
        f"- 优化提示词评分增量均值(全部配对): {_format_number(blind_summary['mean_paired_delta'])}",
        f"- 胜 / 平 / 负: {blind_summary['wins']} / {blind_summary['ties']} / {blind_summary['losses']}",
        f"- 无法配对或失败的文档数: {blind_summary['unpaired_or_failed_count']}",
        f"- 校验失败(分数不可信)的配对: {blind_summary.get('untrusted_paired_count', '未知')}",
    ]
    trusted_count = blind_summary.get("trusted_paired_count")
    if trusted_count:
        lines.extend(
            [
                "",
                f"- **可信配对(两臂均通过 schema 校验) {trusted_count} 对**: "
                f"增量均值 {_format_number(blind_summary.get('trusted_mean_delta'))}, "
                f"增量中位数 {_format_number(blind_summary.get('trusted_median_delta'))}, "
                f"胜 / 平 / 负 {blind_summary.get('trusted_wins')} / {blind_summary.get('trusted_ties')} / {blind_summary.get('trusted_losses')}",
                "  - 全部配对均值包含校验失败配对(其分数基于未通过 schema 校验的原始输出,不可全信);"
                " 可信配对只统计两臂都通过校验的文档,结论应优先参考该项。",
            ]
        )
    # Absolute optimized score + per-dimension breakdown (only when present).
    if blind_summary.get("optimized_abs_mean") is not None:
        lines.extend(
            [
                "",
                f"- **绝对分数**: baseline 均值 {_format_number(blind_summary.get('baseline_abs_mean'))}"
                f" → optimized 均值 {_format_number(blind_summary.get('optimized_abs_mean'))}"
                f" (基于 {blind_summary.get('dimension_breakdown_doc_count')} 篇两臂均有评分)",
            ]
        )
    dim_deltas = blind_summary.get("dimension_deltas")
    if dim_deltas:
        lines.extend(
            [
                "",
                "| 维度 | baseline 均值 | optimized 均值 | 维度增量 |",
                "|------|--------------|----------------|----------|",
            ]
        )
        _DIM_CN = {
            "document_sample": "样品/身份 (document_sample)",
            "process": "工艺 (process)",
            "properties": "性质 (properties)",
            "characterization": "表征 (characterization)",
        }
        for dim, v in dim_deltas.items():
            if v.get("delta") is None:
                continue
            cn = _DIM_CN.get(dim, dim)
            lines.append(
                f"| {cn} | {_format_number(v.get('baseline_mean'))} | "
                f"{_format_number(v.get('optimized_mean'))} | {_format_number(v.get('delta'))} |"
            )
        lines.append("")
    return lines


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _audit_lines(audit_results: list[GoldAuditResult], audit_status: dict[str, int] | None) -> list[str]:
    audit_status = audit_status or {}
    audit_counts: Counter[str] = Counter(
        finding.status for result in audit_results for finding in result.findings
    )
    lines = [
        "## 金标注审计(不影响分数)",
        "",
        f"- 已计划审计文档数: {audit_status.get('scheduled', len(audit_results))}",
        f"- 成功审计文档数: {audit_status.get('successful', len(audit_results))}",
        f"- 失败审计文档数: {audit_status.get('failed', 0)}",
        f"- 待审计文档数: {audit_status.get('pending', 0)}",
        "",
    ]
    for status in ("supported", "unsupported", "ambiguous", "possible_omission"):
        label = {"supported": "有原文支持", "unsupported": "无原文支持", "ambiguous": "不确定", "possible_omission": "可能遗漏"}[status]
        lines.append(f"- {label}: {audit_counts[status]}")
    lines.extend(
        [
            "",
            "金标注审计结果不修改 Gold JSON、不重跑评分、不影响 TextGrad 提示词选择。",
            "",
        ]
    )
    return lines



def _analysis_report_lines(summary: dict[str, Any]) -> list[str]:
    lines = [
        "",
        "## 可追溯原因分析",
        "",
        f"- 配对评分增量中位数: {_format_number(summary.get('median_delta'))}",
        f"- 前三项正向增量贡献占比: {_format_number(summary.get('top_positive_contribution_share'))}",
        f"- 证据分类计数: {summary.get('classification_counts', {})}",
        "- 局部文档证据: `analysis/documents/<document-id>.json`；汇总证据: `analysis/blind_test_summary.json`。",
        "- 说明：JSON 差异与裁判反馈是同现证据，不证明某个字段变化必然导致评分变化；Gold 审计也不用于判定哪一臂更好。",
        "",
        "| 最大正向变化 | 增量 | 最大负向变化 | 增量 |",
        "|--------------|------|--------------|------|",
    ]
    positive = summary.get("largest_positive_deltas", [])
    negative = summary.get("largest_negative_deltas", [])
    for index in range(max(len(positive), len(negative))):
        plus = positive[index] if index < len(positive) else {}
        minus = negative[index] if index < len(negative) else {}
        lines.append(
            f"| {plus.get('document_id', '-')} | {_format_number(plus.get('delta'))} | "
            f"{minus.get('document_id', '-')} | {_format_number(minus.get('delta'))} |"
        )
    return lines


def _format_number(value: float | None) -> str:
    return "无有效数据" if value is None else f"{value:.3f}"
