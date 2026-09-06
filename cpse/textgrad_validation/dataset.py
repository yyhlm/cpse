from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .artifacts import sha256_file
from .models import ComplexityMetrics, DatasetSplit, DocumentPair, RunManifest, SplitDocument

_SPLIT_ALGORITHM_VERSION = "structural-v1"


def discover_pairs(data_dir: Path, strict: bool = True) -> list[DocumentPair]:
    pdfs = _index_by_stem(data_dir, ".pdf")
    golds = _index_by_stem(data_dir, ".json")
    duplicate_stems = [stem for stem, paths in {**pdfs, **golds}.items() if len(paths) != 1]
    if duplicate_stems:
        raise ValueError(f"Duplicate input stems: {', '.join(sorted(duplicate_stems))}")

    all_stems = sorted(set(pdfs) | set(golds))
    missing = [stem for stem in all_stems if stem not in pdfs or stem not in golds]
    if missing and strict:
        raise ValueError(f"Missing PDF/JSON counterpart for: {', '.join(missing)}")

    pairs: list[DocumentPair] = []
    for stem in all_stems:
        if stem not in pdfs or stem not in golds:
            continue
        pdf_path = pdfs[stem][0]
        gold_path = golds[stem][0]
        if pdf_path.stat().st_size == 0:
            raise ValueError(f"PDF is empty: {pdf_path}")
        try:
            gold = json.loads(gold_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Gold JSON is invalid: {gold_path}: {exc}") from exc
        pairs.append(
            DocumentPair(
                document_id=stem,
                pdf_path=pdf_path,
                gold_path=gold_path,
                gold=gold,
                pdf_sha256=sha256_file(pdf_path),
                gold_sha256=sha256_file(gold_path),
            )
        )
    return pairs


def compute_complexity(value: Any) -> ComplexityMetrics:
    stats = Counter()
    populated_paths: set[str] = set()

    def visit(current: Any, path: str, depth: int) -> None:
        stats["node_count"] += 1
        stats["max_depth"] = max(stats["max_depth"], depth)
        if isinstance(current, dict):
            stats["object_key_count"] += len(current)
            for key, child in current.items():
                visit(child, f"{path}.{key}" if path else str(key), depth + 1)
        elif isinstance(current, list):
            stats["array_item_count"] += len(current)
            for index, child in enumerate(current):
                visit(child, f"{path}[{index}]", depth + 1)
        elif _is_populated_scalar(current):
            stats["populated_scalar_count"] += 1
            populated_paths.add(path)

    visit(value, "$", 0)
    # Structural-v1 favors breadth and populated, deeply structured annotations.
    score = (
        stats["node_count"]
        + stats["object_key_count"]
        + stats["array_item_count"] * 2
        + stats["populated_scalar_count"] * 3
        + len(populated_paths) * 4
        + stats["max_depth"] * 5
    )
    return ComplexityMetrics(
        node_count=stats["node_count"],
        object_key_count=stats["object_key_count"],
        array_item_count=stats["array_item_count"],
        populated_scalar_count=stats["populated_scalar_count"],
        max_depth=stats["max_depth"],
        populated_path_count=len(populated_paths),
        score=score,
    )


def select_split(
    pairs: list[DocumentPair], train_count: int = 3, algorithm_version: str = _SPLIT_ALGORITHM_VERSION,
    train_ids: tuple[str, ...] | None = None,
) -> DatasetSplit:
    if algorithm_version != _SPLIT_ALGORITHM_VERSION:
        raise ValueError(f"Unsupported split algorithm version: {algorithm_version}")
    if len(pairs) != 20:
        raise ValueError(f"This fixed experiment requires exactly 20 complete PDF/JSON pairs, found {len(pairs)}.")
    if train_count not in {1, 2, 3}:
        raise ValueError("This fixed experiment supports 1, 2, or 3 active TextGrad training documents.")
    if train_count < 3 and train_ids is None:
        raise ValueError("1/2-shot ablations require an explicit 3-document train_ids pool to keep the blind test fixed.")
    if train_ids is not None and len(train_ids) != 3:
        raise ValueError(f"train_ids must have exactly 3 document IDs, got {len(train_ids)}: {train_ids}")
    if train_ids is not None:
        unknown = [tid for tid in train_ids if tid not in {p.document_id for p in pairs}]
        if unknown:
            raise ValueError(f"train_ids not found in document pairs: {unknown}")
        # Use the explicit IDs as the training set; compute complexity for all.
        id_to_pair = {p.document_id: p for p in pairs}
        train_pairs = [id_to_pair[tid] for tid in train_ids[:train_count]]
        # ``train_ids`` is the fixed three-document training pool. Documents in
        # that pool but inactive for a 1/2-shot ablation stay excluded from the
        # blind test, so all three ablations share the original 17 documents.
        blind_pairs = [p for p in pairs if p.document_id not in train_ids]
        train_records = tuple(
            SplitDocument(pair=p, complexity=compute_complexity(p.gold), split="train", rank=rank)
            for rank, p in enumerate(train_pairs, start=1)
        )
        blind_records = tuple(
            SplitDocument(pair=p, complexity=compute_complexity(p.gold), split="blind_test", rank=rank)
            for rank, p in enumerate(blind_pairs, start=1)
        )
    else:
        ranked = sorted(
            ((pair, compute_complexity(pair.gold)) for pair in pairs),
            key=lambda item: (-item[1].score, item[0].document_id),
        )
        records = tuple(
            SplitDocument(pair=pair, complexity=complexity, split="train" if rank <= train_count else "blind_test", rank=rank)
            for rank, (pair, complexity) in enumerate(ranked, start=1)
        )
        train_records = tuple(record for record in records if record.split == "train")
        blind_records = tuple(record for record in records if record.split == "blind_test")
    return DatasetSplit(
        algorithm_version=algorithm_version,
        train=train_records,
        blind_test=blind_records,
    )


def manifest_to_dict(manifest: RunManifest) -> dict[str, Any]:
    return asdict(manifest)


def _index_by_stem(data_dir: Path, suffix: str) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in sorted(data_dir.glob(f"*{suffix}")):
        index.setdefault(path.stem, []).append(path)
    return index


def _is_populated_scalar(value: Any) -> bool:
    return value is not None and value != "" and not isinstance(value, (dict, list))
