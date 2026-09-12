from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from .config import load_config
from .artifact_release import build_artifact_release
from .code_release import build_code_release, code_release_output_dir
from .dataset import discover_pairs, select_split
from .deterministic_metrics import evaluate_existing_run
from .existing_evaluation import run_judge_only
from .extractor import PdfExtractor
from .gold_audit import GoldAuditor
from .judge import GoldJudge
from .leakage_audit import write_run_leakage_audit
from .optimizer import GepaManifestOptimizer, GepaPromptOptimizer, MiproV2InstructionOptimizer, MiproV2Optimizer, OproPromptOptimizer, TextGradPromptOptimizer
from .responses_client import ResponsesPdfClient, ResponsesTextGradEngine
from .runner import ExperimentRunner
from .schema_contract import dsl_to_json_schema
from .statistics import evaluate_run_statistics
from .usage_summary import summarize_run_usage
from .usage_ledger import ApiUsageLedger


def _invocation_command() -> str:
    """Return a copyable module invocation using the original CLI arguments."""
    arguments = subprocess.list2cmdline(sys.argv[1:])
    return "python -m cpse.textgrad_validation" + (f" {arguments}" if arguments else "")


def _order_train_pool_for_subset(
    pool: tuple[str, ...], selected: tuple[str, ...]
) -> tuple[tuple[str, ...], int]:
    if len(selected) not in {1, 2, 3}:
        raise ValueError("--train-doc must select one, two, or three documents.")
    if len(set(selected)) != len(selected):
        raise ValueError("--train-doc values must be unique.")
    unknown = [document_id for document_id in selected if document_id not in pool]
    if unknown:
        raise ValueError(f"--train-doc must select documents from the fixed training pool: {unknown}")
    remaining = tuple(document_id for document_id in pool if document_id not in selected)
    return selected + remaining, len(selected)


def _skip_blind_baseline_error(optimization_mode: str) -> str | None:
    unsupported = {
        "description_only",
        "alternating_schema_description",
        "two_stage_coverage_plan_schema_description",
        "two_stage_evidence_routing_schema_description",
    }
    if optimization_mode in unsupported:
        return "--skip-blind-baseline is not supported for this optimization mode."
    return None


def _enable_usage(client, *, role: str, ledger: ApiUsageLedger):
    """Enable accounting when the concrete client supports it.

    Keeping this post-construction preserves compatibility with tests and
    external adapters that still implement the original client factory.
    """
    configure = getattr(client, "configure_usage", None)
    if callable(configure):
        configure(role=role, usage_recorder=ledger)
    return client


def _labeled_demonstrations(config) -> list[dict[str, str]]:
    """Build deterministic PDF-text -> Gold examples from the frozen train pool."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Labeled PDF-text demonstrations require pypdf") from exc
    pairs = {pair.document_id: pair for pair in discover_pairs(config.data_dir)}
    demonstrations = []
    for document_id in config.train_ids or ():
        pair = pairs[document_id]
        reader = PdfReader(str(pair.pdf_path))
        document_text = "\n\n".join((page.extract_text() or "") for page in reader.pages).strip()
        if not document_text:
            raise RuntimeError(f"Could not extract demonstration text from {pair.pdf_path}")
        demonstrations.append({
            "document_id": document_id,
            "document_text": document_text,
            "gold_json": json.dumps(pair.gold, ensure_ascii=False, sort_keys=True),
        })
    return demonstrations


def _mipro_labeled_demonstrations(config) -> list[dict[str, str]]:
    return _labeled_demonstrations(config)


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone TextGrad gold-standard validation experiment.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-id", type=str)
    parser.add_argument("--resume", choices=("never", "allow", "require"), default="allow")
    parser.add_argument("--resume-from-round", type=int, default=None,
                        help="alternating/two-stage: reuse accepted rounds 0..N, continue from round N+1 with current code")
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--train-count", type=int, choices=(1, 2, 3),
                        help="Activate the first N documents from the fixed 3-document train_ids pool; the blind test remains fixed.")
    parser.add_argument("--train-doc", action="append", default=None, metavar="DOCUMENT_ID",
                        help="Select a document from the fixed 3-document training pool; repeat for a multi-document subset.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-training", action="store_true", help="Also shrink training to one document and one iteration (implies --smoke).")
    parser.add_argument("--blind-doc", type=str)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--retry-gold-audit", action="store_true")
    parser.add_argument("--retry-failed-primary", action="store_true", help="Retry only failed blind-test primary evaluations, reusing frozen prompts and valid results.")
    parser.add_argument("--judge-only", metavar="LABEL", help="Re-score frozen baseline/optimized predictions with the configured judge and write isolated results under judge_only/LABEL.")
    parser.add_argument("--judge-force", action="store_true", help="With --judge-only, overwrite that label's completed re-scoring artifacts.")
    parser.add_argument("--judge-arm", choices=("baseline", "optimized", "both"), default="both",
                        help="With --judge-only, score only the selected blind-test arm (default: both).")
    parser.add_argument("--judge-doc", action="append", default=None, metavar="DOCUMENT_ID",
                        help="With --judge-only, score only this document ID; repeat to select multiple documents.")
    parser.add_argument("--metrics-only", action="store_true", help="Compute deterministic auxiliary metrics from frozen predictions without any API calls.")
    parser.add_argument("--statistics-only", action="store_true", help="Compute paired confidence intervals, permutation tests, judge agreement, and metric correlations without API calls.")
    parser.add_argument("--usage-only", action="store_true", help="Summarize persisted extraction token usage without API calls; separates logical workload from fresh non-cache calls.")
    parser.add_argument("--artifact-release", action="store_true", help="Build a redacted, checksum-addressed public artifact bundle without API calls or PDFs.")
    parser.add_argument("--code-release", action="store_true", help="Build a source-only, redacted code release without API calls, datasets, or run results.")
    parser.add_argument("--leakage-audit", action="store_true", help="Audit frozen final prompts/schema for exact high-risk facts from training Gold without API calls.")
    parser.add_argument("--blind-baseline-only", action="store_true", help="Run only the fixed direct baseline on blind-test documents; do not train or evaluate an optimized arm.")
    parser.add_argument("--retry-below-score", type=float, metavar="SCORE",
                        help="With --blind-baseline-only or --skip-blind-baseline, retry scores below SCORE; use a passing third attempt, otherwise select the median-scoring result.")
    parser.add_argument("--reselect-score-retries", type=float, metavar="SCORE",
                        help="Offline: rebuild final baseline/optimized artifacts from saved adaptive attempts using SCORE; makes no API calls.")
    parser.add_argument("--skip-blind-baseline", action="store_true", help="After optimization, skip blind-test baseline extraction and judging; run only the optimized arm.")
    parser.add_argument("--force-blind-test", action="store_true", help="Run held-out evaluation even when the selected state's training gain is below the usual blind-test threshold.")
    parser.add_argument("--execution-ablation", action="store_true", help="Re-use a completed two-stage run's frozen state and compare one-shot versus bounded manifest resolution without retraining.")
    parser.add_argument("--stage2-replay", metavar="LABEL", help="Re-run bounded Stage 2 using manifests saved by execution ablation, without rerunning Stage 1.")
    parser.add_argument("--stage2-doc", action="append", default=None, metavar="DOCUMENT_ID",
                        help="With --stage2-replay, process only this document ID; repeat for multiple documents.")
    parser.add_argument("--preflight", action="store_true", help="Validate data and print the deterministic split without API calls.")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.train_doc is not None:
        if args.train_count is not None:
            parser.error("--train-doc cannot be combined with --train-count.")
        if config.train_ids is None:
            parser.error("--train-doc requires config train_ids to define the fixed 3-document training pool.")
        try:
            train_ids, train_count = _order_train_pool_for_subset(config.train_ids, tuple(args.train_doc))
        except ValueError as exc:
            parser.error(str(exc))
        config = replace(config, train_ids=train_ids, train_count=train_count)
    if args.train_count is not None:
        if config.train_ids is None:
            parser.error("--train-count requires config train_ids to define the fixed 3-document training pool.")
        config = replace(config, train_count=args.train_count)
    if args.max_iterations is not None and args.max_iterations <= 0:
        parser.error("--max-iterations must be positive.")
    if args.smoke and not args.blind_doc:
        parser.error("--smoke requires --blind-doc to preserve a fixed blind-test sample.")
    if args.smoke_training and not args.blind_doc:
        parser.error("--smoke-training requires --blind-doc.")
    if args.preflight:
        pairs = discover_pairs(config.data_dir)
        split = select_split(pairs, config.train_count, config.split_algorithm_version, train_ids=config.train_ids)
        dsl_to_json_schema(json.loads(config.schema_path.read_text(encoding="utf-8")))
        print(json.dumps({
            "complete_pairs": len(pairs),
            "train": [record.pair.document_id for record in split.train],
            "blind_test": [record.pair.document_id for record in split.blind_test],
        }, ensure_ascii=False, indent=2))
        return 0
    special_mode_count = sum((args.report_only, args.retry_gold_audit, args.retry_failed_primary, bool(args.judge_only), args.metrics_only, args.statistics_only, args.usage_only, args.artifact_release, args.code_release, args.leakage_audit, args.blind_baseline_only, args.execution_ablation, bool(args.stage2_replay), args.reselect_score_retries is not None))
    if special_mode_count > 1:
        parser.error("Report, retry, judge-only, metrics-only, statistics-only, usage-only, leakage-audit, and blind-baseline-only modes are mutually exclusive.")
    if args.judge_force and not args.judge_only:
        parser.error("--judge-force requires --judge-only LABEL.")
    if (args.judge_arm != "both" or args.judge_doc) and not args.judge_only:
        parser.error("--judge-arm and --judge-doc require --judge-only LABEL.")
    if args.stage2_doc and not args.stage2_replay:
        parser.error("--stage2-doc requires --stage2-replay LABEL.")
    if args.retry_below_score is not None:
        if not (args.blind_baseline_only or args.skip_blind_baseline):
            parser.error("--retry-below-score requires --blind-baseline-only or --skip-blind-baseline.")
        if not 0 <= args.retry_below_score <= 100:
            parser.error("--retry-below-score must be between 0 and 100.")
    if args.reselect_score_retries is not None and not 0 <= args.reselect_score_retries <= 100:
        parser.error("--reselect-score-retries must be between 0 and 100.")
    if special_mode_count and not args.run_id and not args.code_release:
        parser.error("--run-id is required for report, retry, judge-only, metrics-only, statistics-only, usage-only, leakage-audit, and blind-baseline-only modes.")
    if special_mode_count and (args.smoke or args.smoke_training or args.blind_doc):
        parser.error("Report, retry, judge-only, metrics-only, statistics-only, usage-only, leakage-audit, and blind-baseline-only modes cannot be used with smoke options.")
    if not args.run_id and not args.code_release:
        parser.error("--run-id is required for an API experiment run.")
    usage_ledger = ApiUsageLedger(config.output_root / args.run_id / "meta" / "api_calls.jsonl") if args.run_id else None
    if args.resume_from_round is not None:
        if args.resume_from_round < 0:
            parser.error("--resume-from-round must be >= 0.")
        if special_mode_count:
            parser.error("--resume-from-round cannot be combined with report/retry modes.")
        if config.optimization_mode not in {"description_only", "alternating_schema_description", "two_stage_alternating_schema_description", "two_stage_coverage_plan_schema_description", "two_stage_evidence_routing_schema_description"}:
            parser.error("--resume-from-round is only for description_only or alternating schema-description modes.")

    if args.resume_from_round is not None:
        trimmer = ExperimentRunner(config, extractor=None, judge=None, auditor=None, optimizer_factory=None)
        trimmer.trim_checkpoint_to_round(args.run_id, args.resume_from_round)

    if args.report_only:
        runner = ExperimentRunner(config, extractor=None, judge=None, auditor=None, optimizer_factory=None)
        print(runner.report_only(args.run_id))
        return 0
    if args.retry_gold_audit:
        def _audit_api_log(msg: str) -> None:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [api] {msg}", flush=True)

        audit_client = _enable_usage(ResponsesPdfClient(config.gold_audit, on_retry=_audit_api_log), role="gold_audit", ledger=usage_ledger)
        auditor = GoldAuditor(audit_client, config.gold_audit_system_prompt_path.read_text(encoding="utf-8"))
        runner = ExperimentRunner(config, extractor=None, judge=None, auditor=auditor, optimizer_factory=None)
        print(runner.retry_gold_audit(args.run_id))
        return 0
    if args.metrics_only:
        print(evaluate_existing_run(config, args.run_id))
        return 0
    if args.statistics_only:
        print(evaluate_run_statistics(config, args.run_id))
        return 0
    if args.usage_only:
        print(json.dumps(summarize_run_usage(config.output_root / args.run_id), ensure_ascii=False, indent=2))
        return 0
    if args.artifact_release:
        print(build_artifact_release(
            run_dir=config.output_root / args.run_id,
            data_dir=config.data_dir,
            output_dir=config.output_root.parent / "artifact_release" / args.run_id,
        ))
        return 0
    if args.code_release:
        print(build_code_release(
            source_root=config.output_root.parent,
            output_dir=code_release_output_dir(config.output_root),
        ))
        return 0
    if args.leakage_audit:
        print(write_run_leakage_audit(config.output_root / args.run_id))
        return 0
    if args.reselect_score_retries is not None:
        runner = ExperimentRunner(config, extractor=None, judge=None, auditor=None, optimizer_factory=None)
        print(runner.reselect_score_retries(args.run_id, threshold=args.reselect_score_retries))
        return 0
    if args.judge_only:
        def _judge_api_log(msg: str) -> None:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [api] {msg}", flush=True)

        judge_client = _enable_usage(ResponsesPdfClient(config.judge, on_retry=_judge_api_log), role="judge", ledger=usage_ledger)
        judge = GoldJudge(
            judge_client,
            config.judge_system_prompt_path.read_text(encoding="utf-8"),
            include_error_locations=config.include_error_locations,
            include_pdf=config.include_pdf,
            enforce_coverage_audit=config.judge_protocol == "schema_free_content_audit",
        )
        selected_arms = ("baseline", "optimized") if args.judge_arm == "both" else (args.judge_arm,)
        print(run_judge_only(
            config,
            args.run_id,
            judge,
            label=args.judge_only,
            force=args.judge_force,
            arms=selected_arms,
            document_ids=tuple(args.judge_doc) if args.judge_doc else None,
        ))
        return 0
    if args.execution_ablation:
        if config.optimization_mode not in {"two_stage_alternating_schema_description", "two_stage_coverage_plan_schema_description", "two_stage_evidence_routing_schema_description"}:
            parser.error("--execution-ablation requires a two-stage optimization mode.")

        def _execution_api_log(msg: str) -> None:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [api] {msg}", flush=True)

        extractor_client = _enable_usage(ResponsesPdfClient(config.extractor, on_retry=_execution_api_log), role="extraction", ledger=usage_ledger)
        judge_client = _enable_usage(ResponsesPdfClient(config.judge, on_retry=_execution_api_log), role="judge", ledger=usage_ledger)
        judge = GoldJudge(
            judge_client,
            config.judge_system_prompt_path.read_text(encoding="utf-8"),
            include_error_locations=config.include_error_locations,
            include_pdf=config.include_pdf,
            enforce_coverage_audit=config.judge_protocol == "schema_free_content_audit",
        )
        runner = ExperimentRunner(config, extractor=extractor_client, judge=judge, auditor=None, optimizer_factory=None)
        print(runner.run_execution_ablation(args.run_id))
        return 0
    if args.stage2_replay:
        if config.optimization_mode not in {"two_stage_alternating_schema_description", "two_stage_coverage_plan_schema_description", "two_stage_evidence_routing_schema_description"}:
            parser.error("--stage2-replay requires a two-stage optimization mode.")

        def _stage2_replay_api_log(msg: str) -> None:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [api] {msg}", flush=True)

        extractor_client = _enable_usage(ResponsesPdfClient(config.extractor, on_retry=_stage2_replay_api_log), role="extraction", ledger=usage_ledger)
        judge_client = _enable_usage(ResponsesPdfClient(config.judge, on_retry=_stage2_replay_api_log), role="judge", ledger=usage_ledger)
        judge = GoldJudge(
            judge_client,
            config.judge_system_prompt_path.read_text(encoding="utf-8"),
            include_error_locations=config.include_error_locations,
            include_pdf=config.include_pdf,
            enforce_coverage_audit=config.judge_protocol == "schema_free_content_audit",
        )
        runner = ExperimentRunner(config, extractor=extractor_client, judge=judge, auditor=None, optimizer_factory=None)
        print(runner.run_stage2_replay(
            args.run_id,
            args.stage2_replay,
            document_ids=tuple(args.stage2_doc) if args.stage2_doc else None,
        ))
        return 0
    if args.blind_baseline_only:
        schema_text = config.schema_path.read_text(encoding="utf-8")
        schema = dsl_to_json_schema(json.loads(schema_text))
        def _api_log(msg: str) -> None:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [api] {msg}", flush=True)

        extractor_client = _enable_usage(ResponsesPdfClient(config.extractor, on_retry=_api_log), role="extraction", ledger=usage_ledger)
        judge_client = _enable_usage(ResponsesPdfClient(config.judge, on_retry=_api_log), role="judge", ledger=usage_ledger)
        extractor = PdfExtractor(
            extractor_client,
            schema=schema,
            schema_text=schema_text,
            system_prompt=config.extraction_system_prompt_path.read_text(encoding="utf-8"),
            schema_free=config.optimization_mode == "schema_free_direct",
            demonstrations=tuple(_labeled_demonstrations(config)) if config.optimization_mode == "few_shot_direct" else (),
        )
        judge = GoldJudge(
            judge_client,
            config.judge_system_prompt_path.read_text(encoding="utf-8"),
            include_error_locations=config.include_error_locations,
            include_pdf=config.include_pdf,
            enforce_coverage_audit=config.judge_protocol == "schema_free_content_audit",
        )
        runner = ExperimentRunner(config, extractor=extractor, judge=judge, auditor=None, optimizer_factory=None)
        print(runner.run_blind_baseline_only(
            args.run_id,
            invocation_command=_invocation_command(),
            retry_below_score=args.retry_below_score,
        ))
        return 0

    if config.optimization_mode in {"schema_free_direct", "few_shot_direct"}:
        parser.error(f"optimization_mode={config.optimization_mode} is a baseline-only mode; add --blind-baseline-only.")
    skip_baseline_error = _skip_blind_baseline_error(config.optimization_mode) if args.skip_blind_baseline else None
    if skip_baseline_error:
        parser.error(skip_baseline_error)

    if args.retry_failed_primary:
        schema_text = config.schema_path.read_text(encoding="utf-8")
        schema = dsl_to_json_schema(json.loads(schema_text))
        def _api_log(msg: str) -> None:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [api] {msg}", flush=True)

        extractor_client = _enable_usage(ResponsesPdfClient(config.extractor, on_retry=_api_log), role="extraction", ledger=usage_ledger)
        judge_client = _enable_usage(ResponsesPdfClient(config.judge, on_retry=_api_log), role="judge", ledger=usage_ledger)
        extractor = PdfExtractor(
            extractor_client,
            schema=schema,
            schema_text=schema_text,
            system_prompt=config.extraction_system_prompt_path.read_text(encoding="utf-8"),
        )
        judge = GoldJudge(
            judge_client,
            config.judge_system_prompt_path.read_text(encoding="utf-8"),
            include_error_locations=config.include_error_locations,
            include_pdf=config.include_pdf,
            enforce_coverage_audit=config.judge_protocol == "schema_free_content_audit",
        )
        runner = ExperimentRunner(config, extractor=extractor, judge=judge, auditor=None, optimizer_factory=None)
        print(runner.retry_failed_primary(args.run_id))
        return 0

    schema_text = config.schema_path.read_text(encoding="utf-8")
    schema = dsl_to_json_schema(json.loads(schema_text))

    def _api_log(msg: str) -> None:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [api] {msg}", flush=True)

    extractor_client = _enable_usage(ResponsesPdfClient(config.extractor, on_retry=_api_log), role="extraction", ledger=usage_ledger)
    judge_client = _enable_usage(ResponsesPdfClient(config.judge, on_retry=_api_log), role="judge", ledger=usage_ledger)
    audit_client = _enable_usage(ResponsesPdfClient(config.gold_audit, on_retry=_api_log), role="gold_audit", ledger=usage_ledger)
    textgrad_engine = ResponsesTextGradEngine(judge_client)
    if config.optimization_mode in {"description_only", "alternating_schema_description"}:
        # The schema-patch model call reuses the judge role/model. The runner
        # builds the alternating optimizer itself from the TextGrad engine; the
        # legacy single-prompt optimizer_factory is unused in this mode.
        extractor = PdfExtractor(
            extractor_client,
            schema=schema,
            schema_text=schema_text,
            system_prompt=config.extraction_system_prompt_path.read_text(encoding="utf-8"),
        )
        schema_patch_client = judge_client
        schema_patch_system_prompt = config.schema_description_system_prompt_path.read_text(encoding="utf-8")
        optimizer_factory = None
    elif config.optimization_mode in {"gepa_manifest", "two_stage_alternating_schema_description", "two_stage_coverage_plan_schema_description", "two_stage_evidence_routing_schema_description"}:
        # Two-stage mode drives the raw ResponsesPdfClient directly
        # (extract_two_stage calls complete_pdf_json per stage); the PdfExtractor
        # wrapper is not used. The schema-patch call reuses the judge client.
        extractor = extractor_client
        schema_patch_client = judge_client if config.optimization_mode != "gepa_manifest" else None
        schema_patch_system_prompt = (
            config.schema_description_system_prompt_path.read_text(encoding="utf-8")
            if config.optimization_mode != "gepa_manifest" else None
        )
        if config.optimization_mode == "gepa_manifest":
            def optimizer_factory(evaluate_training):
                return GepaManifestOptimizer(client=judge_client, evaluate_prompts=evaluate_training)
        else:
            optimizer_factory = None
    else:
        extractor = PdfExtractor(
            extractor_client,
            schema=schema,
            schema_text=schema_text,
            system_prompt=config.extraction_system_prompt_path.read_text(encoding="utf-8"),
        )

        if config.optimization_mode == "opro_prompt_only":
            def optimizer_factory(evaluate_training):
                return OproPromptOptimizer(client=judge_client, evaluate_prompt=evaluate_training)
        elif config.optimization_mode == "gepa_prompt_only":
            def optimizer_factory(evaluate_training):
                return GepaPromptOptimizer(client=judge_client, evaluate_prompt=evaluate_training)
        elif config.optimization_mode == "mipro_v2_instruction_only":
            def optimizer_factory(evaluate_training):
                return MiproV2InstructionOptimizer(client=judge_client, evaluate_prompt=evaluate_training)
        elif config.optimization_mode == "mipro_v2":
            demonstrations = _mipro_labeled_demonstrations(config)

            def optimizer_factory(evaluate_training):
                return MiproV2Optimizer(
                    client=judge_client,
                    evaluate_prompt=evaluate_training,
                    demonstrations=demonstrations,
                )
        else:
            def optimizer_factory(_evaluate_training):
                return TextGradPromptOptimizer(
                    engine=textgrad_engine,
                    evaluate_prompt=lambda _prompt: {},
                )

        schema_patch_client = None
        schema_patch_system_prompt = None

    judge = GoldJudge(
        judge_client,
        config.judge_system_prompt_path.read_text(encoding="utf-8"),
        include_error_locations=config.include_error_locations,
        include_pdf=config.include_pdf,
        enforce_coverage_audit=config.judge_protocol == "schema_free_content_audit",
    )
    auditor = GoldAuditor(audit_client, config.gold_audit_system_prompt_path.read_text(encoding="utf-8"))

    runner = ExperimentRunner(
        config,
        extractor=extractor,
        judge=judge,
        auditor=auditor,
        optimizer_factory=optimizer_factory,
        schema_patch_client=schema_patch_client,
        schema_patch_system_prompt=schema_patch_system_prompt,
        textgrad_engine=(
            textgrad_engine
            if config.optimization_mode in {"description_only", "alternating_schema_description", "two_stage_alternating_schema_description", "two_stage_coverage_plan_schema_description", "two_stage_evidence_routing_schema_description"}
            else None
        ),
    )
    try:
        run_dir = runner.run(
            args.run_id,
            max_iterations=1 if (args.smoke or args.smoke_training) else args.max_iterations,
            smoke_blind_doc=args.blind_doc if (args.smoke or args.smoke_training) else None,
            smoke_training=args.smoke or args.smoke_training,
            resume=args.resume,
            invocation_command=_invocation_command(),
            skip_blind_baseline=args.skip_blind_baseline,
            force_blind_test=args.force_blind_test,
            retry_below_score=args.retry_below_score,
        )
    except KeyboardInterrupt:
        return 130
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

