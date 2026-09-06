from __future__ import annotations

import csv
import itertools
import math
import random
from pathlib import Path
from statistics import median
from typing import Any

from .artifacts import write_csv, write_json, write_text


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _percentile(sorted_values: list[float], probability: float) -> float | None:
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _bootstrap_mean_ci(
    values: list[float], *, samples: int, seed: int
) -> list[float | None]:
    if not values:
        return [None, None]
    rng = random.Random(seed)
    means = sorted(
        sum(rng.choice(values) for _ in values) / len(values)
        for _ in range(samples)
    )
    return [_percentile(means, 0.025), _percentile(means, 0.975)]


def _sign_flip_permutation_p(values: list[float]) -> float | None:
    nonzero = [value for value in values if value != 0]
    if not nonzero:
        return 1.0 if values else None
    observed = abs(sum(nonzero))
    if len(nonzero) <= 20:
        extreme = 0
        total = 1 << len(nonzero)
        for signs in itertools.product((-1.0, 1.0), repeat=len(nonzero)):
            statistic = abs(sum(sign * value for sign, value in zip(signs, nonzero)))
            if statistic >= observed - 1e-12:
                extreme += 1
        return extreme / total
    rng = random.Random(20260825)
    samples = 200000
    extreme = 0
    for _ in range(samples):
        statistic = abs(sum(value if rng.random() < 0.5 else -value for value in nonzero))
        if statistic >= observed - 1e-12:
            extreme += 1
    return (extreme + 1) / (samples + 1)


def paired_statistics(
    deltas: list[float], *, bootstrap_samples: int = 10000, seed: int = 20260825
) -> dict[str, Any]:
    values = [float(value) for value in deltas]
    return {
        "count": len(values),
        "mean": _mean(values),
        "median": median(values) if values else None,
        "wins": sum(value > 0 for value in values),
        "ties": sum(value == 0 for value in values),
        "losses": sum(value < 0 for value in values),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "bootstrap_95_ci": _bootstrap_mean_ci(values, samples=bootstrap_samples, seed=seed),
        "exact_sign_flip_permutation_p_two_sided": _sign_flip_permutation_p(values),
    }


def _average_ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for index in range(start, end):
            ranks[ordered[index][0]] = average_rank
        start = end
    return ranks


def rank_correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right):
        raise ValueError("Rank correlation inputs must have the same length.")
    if len(left) < 2:
        return None
    left_ranks = _average_ranks([float(value) for value in left])
    right_ranks = _average_ranks([float(value) for value in right])
    left_mean = sum(left_ranks) / len(left_ranks)
    right_mean = sum(right_ranks) / len(right_ranks)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left_ranks, right_ranks)
    )
    left_scale = sum((value - left_mean) ** 2 for value in left_ranks)
    right_scale = sum((value - right_mean) ** 2 for value in right_ranks)
    denominator = math.sqrt(left_scale * right_scale)
    return numerator / denominator if denominator else None


def _direction(value: float) -> int:
    return 1 if value > 0 else (-1 if value < 0 else 0)


def direction_agreement(left: list[float], right: list[float]) -> dict[str, Any]:
    if len(left) != len(right):
        raise ValueError("Direction agreement inputs must have the same length.")
    same = sum(_direction(a) == _direction(b) for a, b in zip(left, right))
    return {
        "count": len(left),
        "same_direction": same,
        "same_direction_rate": same / len(left) if left else None,
    }


def _read_delta_csv(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    result: dict[str, float] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            document_id = row.get("document_id")
            delta = row.get("delta")
            if document_id and delta not in (None, ""):
                result[document_id] = float(delta)
    return result


def _read_deterministic_deltas(path: Path) -> dict[str, dict[str, float]]:
    if not path.is_file():
        return {}
    by_document: dict[str, dict[str, dict[str, float]]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            document_id = row.get("document_id")
            arm = row.get("arm")
            if not document_id or arm not in {"baseline", "optimized"}:
                continue
            by_document.setdefault(document_id, {})[arm] = {
                "strict_leaf_recall": float(row["strict_leaf_recall"]),
                "strict_leaf_f1": float(row["strict_leaf_f1"]),
                "property_tuple_recall": float(row["property_tuple_recall"]),
                "property_tuple_f1": float(row["property_tuple_f1"]),
            }
    result: dict[str, dict[str, float]] = {}
    for document_id, arms in by_document.items():
        if "baseline" not in arms or "optimized" not in arms:
            continue
        result[document_id] = {
            key + "_delta": arms["optimized"][key] - arms["baseline"][key]
            for key in ("strict_leaf_recall", "strict_leaf_f1", "property_tuple_recall", "property_tuple_f1")
        }
    return result


def _association(
    score_deltas: dict[str, float], deterministic: dict[str, dict[str, float]]
) -> dict[str, Any]:
    document_ids = sorted(set(score_deltas) & set(deterministic))
    scores = [score_deltas[document_id] for document_id in document_ids]
    return {
        "count": len(document_ids),
        "strict_leaf_recall_delta_spearman": rank_correlation(
            scores,
            [deterministic[document_id]["strict_leaf_recall_delta"] for document_id in document_ids],
        ),
        "strict_leaf_f1_delta_spearman": rank_correlation(
            scores,
            [deterministic[document_id]["strict_leaf_f1_delta"] for document_id in document_ids],
        ),
        "property_tuple_recall_delta_spearman": rank_correlation(
            scores,
            [deterministic[document_id]["property_tuple_recall_delta"] for document_id in document_ids],
        ),
        "property_tuple_f1_delta_spearman": rank_correlation(
            scores,
            [deterministic[document_id]["property_tuple_f1_delta"] for document_id in document_ids],
        ),
    }


def _statistics_report(summary: dict[str, Any]) -> str:
    def display(value: float | None) -> str:
        return f"{value:.3f}" if value is not None else "n/a"

    primary = summary["primary"]
    ci = primary["bootstrap_95_ci"]
    lines = [
        "# 配对盲测统计报告",
        "",
        "全部统计均从冻结结果离线计算，不调用模型，不修改主评分。",
        "",
        "## 主裁判",
        "",
        f"- 配对数：{primary['count']}",
        f"- delta 均值 / 中位数：{primary['mean']:.3f} / {primary['median']:.3f}",
        f"- 胜 / 平 / 负：{primary['wins']} / {primary['ties']} / {primary['losses']}",
        f"- paired bootstrap 95% CI：[{ci[0]:.3f}, {ci[1]:.3f}]",
        f"- 双侧精确符号翻转 permutation p：{primary['exact_sign_flip_permutation_p_two_sided']:.6f}",
        "",
        "## Judge 稳健性",
        "",
        "| Judge | mean delta | median delta | 胜/平/负 | 与主裁判 Spearman | 方向一致率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, judge in summary["judges"].items():
        paired = judge["paired"]
        comparison = judge["vs_primary"]
        rho = comparison["delta_spearman"]
        rate = comparison["direction"]["same_direction_rate"]
        lines.append(
            f"| {label} | {paired['mean']:.3f} | {paired['median']:.3f} | "
            f"{paired['wins']}/{paired['ties']}/{paired['losses']} | "
            f"{display(rho)} | {display(rate)} |"
        )
    lines.extend([
        "",
        "## 与严格叶子重叠指标的关系",
        "",
        "这里的严格指标不处理语义等价、单位换算或科学同义表达，只作为探索性辅助证据。",
        "",
    ])
    return "\n".join(lines) + "\n"


def evaluate_run_statistics(
    config, run_id: str, *, bootstrap_samples: int = 10000, seed: int = 20260825
) -> Path:
    run_dir = config.output_root / run_id
    primary_deltas = _read_delta_csv(run_dir / "blind_test_documents.csv")
    if not primary_deltas:
        raise FileNotFoundError(f"Primary paired blind-test scores not found: {run_dir / 'blind_test_documents.csv'}")
    deterministic = _read_deterministic_deltas(run_dir / "deterministic_metrics" / "documents.csv")
    summary: dict[str, Any] = {
        "source_run_id": run_id,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "primary": paired_statistics(
            list(primary_deltas.values()), bootstrap_samples=bootstrap_samples, seed=seed
        ),
        "primary_vs_deterministic": _association(primary_deltas, deterministic),
        "judges": {},
    }
    judge_deltas: dict[str, dict[str, float]] = {}
    judge_root = run_dir / "judge_only"
    if judge_root.is_dir():
        for label_dir in sorted(path for path in judge_root.iterdir() if path.is_dir()):
            deltas = _read_delta_csv(label_dir / "documents.csv")
            if not deltas:
                continue
            judge_deltas[label_dir.name] = deltas
            shared = sorted(set(primary_deltas) & set(deltas))
            primary_values = [primary_deltas[document_id] for document_id in shared]
            judge_values = [deltas[document_id] for document_id in shared]
            summary["judges"][label_dir.name] = {
                "paired": paired_statistics(
                    list(deltas.values()), bootstrap_samples=bootstrap_samples, seed=seed
                ),
                "vs_primary": {
                    "count": len(shared),
                    "delta_spearman": rank_correlation(primary_values, judge_values),
                    "direction": direction_agreement(primary_values, judge_values),
                },
                "vs_deterministic": _association(deltas, deterministic),
            }
    document_ids = sorted(set(primary_deltas) | set(deterministic) | set().union(*(set(v) for v in judge_deltas.values())))
    rows: list[dict[str, Any]] = []
    for document_id in document_ids:
        row: dict[str, Any] = {
            "document_id": document_id,
            "primary_delta": primary_deltas.get(document_id),
            **deterministic.get(document_id, {}),
        }
        for label, deltas in judge_deltas.items():
            row[f"{label}_delta"] = deltas.get(document_id)
        rows.append(row)
    output_dir = run_dir / "statistics"
    columns = list(rows[0]) if rows else ["document_id", "primary_delta"]
    write_csv(output_dir / "per_document.csv", rows, columns)
    write_json(output_dir / "summary.json", summary)
    write_text(output_dir / "report.md", _statistics_report(summary))
    return output_dir
