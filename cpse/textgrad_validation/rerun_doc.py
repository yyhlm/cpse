"""Re-run one blind-test optimized document with the frozen best prompt to check
whether a low score was an occasional model stumble or a stable failure.

Reuses the same PdfExtractor + GoldJudge construction as cli.py so the re-run is
directly comparable. Writes results under <run>/blind_test/optimized/documents/
<doc>/rerun/.

Usage:
    python -m test.textgrad_validation.rerun_doc ^
        --run-id v2-goldnorm-focus-sol-single-01 ^
        --config test/textgrad_validation/config_single.yaml ^
        --doc 0a9e02596a30ab0978db4ea35996d3e3 --count 3
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .artifacts import write_json
from .config import load_config
from .extractor import PdfExtractor
from .judge import GoldJudge
from .responses_client import ResponsesPdfClient
from .schema_contract import dsl_to_json_schema
from .schema_description import reinforce_required_emission


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--doc", required=True)
    parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    run_dir = config.output_root / args.run_id
    prompt = (run_dir / "prompts" / "final-best.txt").read_text(encoding="utf-8").strip()
    system_prompt = config.extraction_system_prompt_path.read_text(encoding="utf-8")
    schema_dsl = reinforce_required_emission(json.loads(config.schema_path.read_text(encoding="utf-8")))
    schema = dsl_to_json_schema(schema_dsl)
    schema_text = json.dumps(schema_dsl, ensure_ascii=False, sort_keys=True)

    extractor_client = ResponsesPdfClient(config.extractor)
    judge_client = ResponsesPdfClient(config.judge)
    extractor = PdfExtractor(
        extractor_client,
        schema=schema,
        schema_text=schema_text,
        system_prompt=system_prompt,
        max_repair_attempts=1,
    )
    judge = GoldJudge(
        judge_client,
        system_prompt=config.judge_system_prompt_path.read_text(encoding="utf-8"),
        include_error_locations=config.include_error_locations,
        include_pdf=config.include_pdf,
    )

    pdf = config.data_dir / f"{args.doc}.pdf"
    gold = json.loads((config.data_dir / f"{args.doc}.json").read_text(encoding="utf-8"))

    out_dir = run_dir / "blind_test" / "optimized" / "documents" / args.doc / "rerun"
    out_dir.mkdir(parents=True, exist_ok=True)

    scores = []
    for i in range(args.count):
        prediction, meta = extractor.extract(pdf, prompt, schema_text=schema_text, schema=schema)
        if not prediction.is_valid:
            print(f"[run {i}] INVALID ({len(prediction.validation_errors)} errors); skipped")
            write_json(out_dir / f"run{i}_invalid.json", {"errors": list(prediction.validation_errors)})
            continue
        try:
            result = judge.judge(schema=schema, prediction=prediction, gold=gold, pdf_path=pdf)
        except Exception as exc:
            print(f"[run {i}] judge failed: {type(exc).__name__}: {exc}")
            continue
        scores.append(result.score)
        print(
            f"[run {i}] score={result.score} breakdown={result.score_breakdown} "
            f"outtok={meta.get('usage', {}).get('output_tokens')}"
        )
        write_json(out_dir / f"run{i}.json", {
            **asdict(result),
            "usage": meta.get("usage"),
            "repair_attempts": meta.get("repair_attempts"),
        })

    if scores:
        print(f"\nrerun scores: {scores}  mean={sum(scores) / len(scores):.1f}  (original optimized score was before rerun)")
    else:
        print("no valid reruns produced")


if __name__ == "__main__":
    sys.exit(main())
