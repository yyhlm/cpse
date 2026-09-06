from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from .artifacts import model_config_fingerprint, sha256_file, sha256_json
from .models import ExperimentConfig, ModelConfig


_DOTENV_LOADED: set[Path] = set()


def load_dotenv(dotenv_path: Path) -> None:
    """Load one .env file without overriding environment variables already set."""
    dotenv_path = dotenv_path.resolve()
    if dotenv_path in _DOTENV_LOADED:
        return
    _DOTENV_LOADED.add(dotenv_path)
    if not dotenv_path.is_file():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_config(path: Path, project_root: Path | None = None) -> ExperimentConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Experiment config must be a YAML object.")
    base = project_root or locate_project_root(path)
    # Experiment-local credentials take precedence over the optional repo-root
    # .env; neither source replaces an already-exported environment variable.
    load_dotenv(path.parent / ".env")
    shared_env_path = raw.get("shared_env_path")
    if shared_env_path is not None:
        if not isinstance(shared_env_path, str) or not shared_env_path.strip():
            raise ValueError("Config shared_env_path must be a non-empty string when provided.")
        load_dotenv(_resolve(base, shared_env_path))
    load_dotenv(base / ".env")
    paths = raw.get("paths")
    roles = raw.get("roles")
    if not isinstance(paths, dict) or not isinstance(roles, dict):
        raise ValueError("Config requires paths and roles objects.")

    prompt_dir = _resolve(base, paths.get("prompt_dir", "cpse/textgrad_validation/prompts"))
    evaluation = raw.get("evaluation", {})
    if not isinstance(evaluation, dict):
        raise ValueError("Config evaluation must be an object.")
    include_error_locations = bool(evaluation.get("include_error_locations", False))
    include_pdf = bool(evaluation.get("include_pdf", False))
    optimization_mode = str(raw.get("optimization_mode", "single"))
    if optimization_mode not in (
        "single",
        "opro_prompt_only",
        "gepa_prompt_only",
        "mipro_v2_instruction_only",
        "mipro_v2",
        "schema_free_direct",
        "few_shot_direct",
        "description_only",
        "alternating_schema_description",
        "two_stage_alternating_schema_description",
        "two_stage_coverage_plan_schema_description",
        "two_stage_evidence_routing_schema_description",
    ):
        raise ValueError(f"Unsupported optimization_mode: {optimization_mode!r}.")
    raw_train_ids = raw.get("train_ids")
    train_ids = tuple(str(x) for x in raw_train_ids) if raw_train_ids else None
    if train_ids is not None and len(train_ids) != 3:
        raise ValueError(f"train_ids must have exactly 3 document IDs, got {len(train_ids)}: {train_ids}")
    if train_ids is not None and len(set(train_ids)) != 3:
        raise ValueError("train_ids must contain three unique document IDs.")
    if optimization_mode == "few_shot_direct" and train_ids is None:
        raise ValueError("few_shot_direct requires an explicit three-document train_ids pool.")
    if optimization_mode == "schema_free_direct" and include_error_locations:
        raise ValueError("schema_free_direct does not support schema-path error locations.")
    if optimization_mode == "schema_free_direct" and include_pdf:
        judge_prompt_name = "judge_schema_free_with_pdf_system.txt"
    elif optimization_mode == "schema_free_direct":
        judge_prompt_name = "judge_schema_free_system.txt"
    elif include_pdf and include_error_locations:
        judge_prompt_name = "judge_with_pdf_locations_system.txt"
    elif include_pdf:
        judge_prompt_name = "judge_with_pdf_system.txt"
    elif include_error_locations:
        judge_prompt_name = "judge_with_locations_system.txt"
    else:
        judge_prompt_name = "judge_system.txt"
    schema_patch = raw.get("schema_patch", {})
    if not isinstance(schema_patch, dict):
        raise ValueError("Config schema_patch must be an object.")
    cache = raw.get("cache", {})
    if not isinstance(cache, dict):
        raise ValueError("Config cache must be an object.")
    schema_prompt_dir = _resolve(base, paths.get("prompt_dir", "cpse/textgrad_validation/prompts"))
    config = ExperimentConfig(
        data_dir=_resolve(base, _required(paths, "data_dir")),
        schema_path=_resolve(base, _required(paths, "schema_path")),
        output_root=_resolve(base, _required(paths, "output_root")),
        initial_prompt_path=prompt_dir / ("schema_free_initial.txt" if optimization_mode == "schema_free_direct" else ("few_shot_initial.txt" if optimization_mode == "few_shot_direct" else "extraction_initial.txt")),
        extraction_system_prompt_path=prompt_dir / ("schema_free_system.txt" if optimization_mode == "schema_free_direct" else ("few_shot_system.txt" if optimization_mode == "few_shot_direct" else "extraction_system.txt")),
        judge_system_prompt_path=prompt_dir / judge_prompt_name,
        gold_audit_system_prompt_path=prompt_dir / "gold_audit_system.txt",
        extractor=_parse_role(roles, "extractor"),
        judge=_parse_role(roles, "judge"),
        gold_audit=_parse_role(roles, "gold_audit"),
        include_error_locations=include_error_locations,
        include_pdf=include_pdf,
        gold_audit_enabled=bool(raw.get("gold_audit", {}).get("enabled", False)),
        max_iterations=_positive_int(raw.get("max_iterations", 3), "max_iterations"),
        max_parallel_calls=_positive_int(raw.get("max_parallel_calls", 3), "max_parallel_calls"),
        train_count=_positive_int(raw.get("train_count", 3), "train_count"),
        train_ids=train_ids,
        split_algorithm_version=str(raw.get("split_algorithm_version", "structural-v1")),
        request_format_version=str(raw.get("request_format_version", "responses-pdf-v1")),
        optimization_mode=optimization_mode,
        schema_description_system_prompt_path=(
            schema_prompt_dir / "schema_description_system.txt"
            if optimization_mode in {"description_only", "alternating_schema_description", "two_stage_alternating_schema_description", "two_stage_coverage_plan_schema_description", "two_stage_evidence_routing_schema_description"}
            else None
        ),
        schema_description_initial_prompt_path=(
            schema_prompt_dir / "schema_description_initial.txt"
            if optimization_mode in {"description_only", "alternating_schema_description", "two_stage_alternating_schema_description", "two_stage_coverage_plan_schema_description", "two_stage_evidence_routing_schema_description"}
            else None
        ),
        evidence_initial_prompt_path=(schema_prompt_dir / "evidence_initial.txt" if optimization_mode in {"two_stage_alternating_schema_description", "two_stage_coverage_plan_schema_description", "two_stage_evidence_routing_schema_description"} else None),
        evidence_system_prompt_path=(schema_prompt_dir / "evidence_system.txt" if optimization_mode in {"two_stage_alternating_schema_description", "two_stage_coverage_plan_schema_description", "two_stage_evidence_routing_schema_description"} else None),
        resolve_initial_prompt_path=(schema_prompt_dir / "resolve_initial.txt" if optimization_mode in {"two_stage_alternating_schema_description", "two_stage_coverage_plan_schema_description", "two_stage_evidence_routing_schema_description"} else None),
        resolve_system_prompt_path=(schema_prompt_dir / "resolve_system.txt" if optimization_mode in {"two_stage_alternating_schema_description", "two_stage_coverage_plan_schema_description", "two_stage_evidence_routing_schema_description"} else None),
        coverage_plan_system_prompt_path=(schema_prompt_dir / "coverage_plan_system.txt" if optimization_mode == "two_stage_coverage_plan_schema_description" else None),
        evidence_routing_initial_prompt_path=(schema_prompt_dir / "evidence_routing_initial.txt" if optimization_mode == "two_stage_evidence_routing_schema_description" else None),
        evidence_routing_system_prompt_path=(schema_prompt_dir / "evidence_routing_system.txt" if optimization_mode == "two_stage_evidence_routing_schema_description" else None),
        schema_patch_policy_version=str(schema_patch.get("policy_version", "1")),
        cache_enabled=bool(cache.get("enabled", True)),
    )
    if config.train_count not in {1, 2, 3}:
        raise ValueError("train_count must be 1, 2, or 3.")
    if config.train_count < 3 and config.train_ids is None:
        raise ValueError("1/2-shot ablations require an explicit 3-document train_ids pool.")
    if config.optimization_mode == "few_shot_direct" and config.train_count != 3:
        raise ValueError("few_shot_direct is the fixed 3-shot baseline and requires train_count: 3.")
    _validate_files(config)
    return config


def require_api_key(config: ModelConfig) -> str:
    value = os.environ.get(config.api_key_env, "").strip()
    if not value:
        raise RuntimeError(f"Required API key environment variable is missing: {config.api_key_env}")
    return value


def locate_project_root(config_path: Path) -> Path:
    """Locate the release root. config_path is e.g. cpse/textgrad_validation/config.yaml."""
    return config_path.parent.parent.parent.resolve()


def config_fingerprint(config: ExperimentConfig) -> str:
    # ``max_iterations`` is intentionally excluded so it can be relaxed on an
    # existing run (increase runs new iterations; decrease recomputes the frozen
    # best). The checkpoint tracks N (used) vs M (requested) separately. Every
    # other field remains locked: changing any of them still aborts in
    # ``prepare_run`` and requires a new run id.
    fingerprint_data = {
        "extractor": model_config_fingerprint(config.extractor),
        "judge": model_config_fingerprint(config.judge),
        "gold_audit": model_config_fingerprint(config.gold_audit),
        "max_parallel_calls": config.max_parallel_calls,
        "cache_enabled": config.cache_enabled,
        "train_count": config.train_count,
        "split_algorithm_version": config.split_algorithm_version,
        "request_format_version": config.request_format_version,
        "evaluation_include_pdf": config.include_pdf,
        "prompt_hashes": {
            "initial": sha256_file(config.initial_prompt_path),
            "extraction_system": sha256_file(config.extraction_system_prompt_path),
            "judge_system": sha256_file(config.judge_system_prompt_path),
            "gold_audit_system": sha256_file(config.gold_audit_system_prompt_path),
        },
    }
    # The alternating-mode keys are added only when that mode is active so legacy
    # single-mode manifests (which never contained them) keep matching on resume.
    if config.optimization_mode != "single":
        fingerprint_data["optimization_mode"] = config.optimization_mode
    if config.optimization_mode in {"description_only", "alternating_schema_description", "two_stage_alternating_schema_description", "two_stage_coverage_plan_schema_description", "two_stage_evidence_routing_schema_description"}:
        fingerprint_data["alternating_training_algorithm_version"] = "joint-round-v2"
        fingerprint_data["schema_patch_policy_version"] = config.schema_patch_policy_version
        fingerprint_data["prompt_hashes"]["schema_description_system"] = sha256_file(config.schema_description_system_prompt_path)
        fingerprint_data["prompt_hashes"]["schema_description_initial"] = sha256_file(config.schema_description_initial_prompt_path)
    if config.optimization_mode in {"two_stage_alternating_schema_description", "two_stage_coverage_plan_schema_description", "two_stage_evidence_routing_schema_description"}:
        fingerprint_data["two_stage_protocol_version"] = ("evidence-routing-resolve-section-v1" if config.optimization_mode == "two_stage_evidence_routing_schema_description" else ("evidence-coverage-resolve-section-v1" if config.optimization_mode == "two_stage_coverage_plan_schema_description" else "evidence-resolve-section-v1"))
        fingerprint_data["prompt_hashes"].update({
            "evidence_initial": sha256_file(config.evidence_initial_prompt_path),
            "evidence_system": sha256_file(config.evidence_system_prompt_path),
            "resolve_initial": sha256_file(config.resolve_initial_prompt_path),
            "resolve_system": sha256_file(config.resolve_system_prompt_path),
            **({"coverage_plan_system": sha256_file(config.coverage_plan_system_prompt_path)} if config.optimization_mode == "two_stage_coverage_plan_schema_description" else {}),
            **({"evidence_routing_initial": sha256_file(config.evidence_routing_initial_prompt_path), "evidence_routing_system": sha256_file(config.evidence_routing_system_prompt_path)} if config.optimization_mode == "two_stage_evidence_routing_schema_description" else {}),
        })
    return sha256_json(fingerprint_data)


def snapshot_config(config: ExperimentConfig) -> dict[str, Any]:
    data = asdict(config)
    for role in ("extractor", "judge", "gold_audit"):
        data[role].pop("api_key_env", None)
    return data


def _parse_role(roles: dict[str, Any], name: str) -> ModelConfig:
    raw = roles.get(name)
    if not isinstance(raw, dict):
        raise ValueError(f"Config roles.{name} must be an object.")
    return ModelConfig(
        model=str(_required(raw, "model")),
        base_url=str(raw["base_url"]) if raw.get("base_url") else None,
        api_key_env=str(_required(raw, "api_key_env")),
        temperature=float(raw.get("temperature", 0)),
        max_output_tokens=_optional_positive_int(raw.get("max_output_tokens", 8192), f"roles.{name}.max_output_tokens"),
        timeout_seconds=float(raw.get("timeout_seconds", 180)),
        max_retries=_nonnegative_int(raw.get("max_retries", 2), f"roles.{name}.max_retries"),
        use_instructions=bool(raw.get("use_instructions", False)),
        proxy=str(raw["proxy"]) if raw.get("proxy") else None,
        api_protocol=str(raw.get("api_protocol", "responses")),
        reasoning_effort=str(raw["reasoning_effort"]) if raw.get("reasoning_effort") else None,
    )


def _validate_files(config: ExperimentConfig) -> None:
    paths = [
        config.data_dir,
        config.schema_path,
        config.initial_prompt_path,
        config.extraction_system_prompt_path,
        config.judge_system_prompt_path,
        config.gold_audit_system_prompt_path,
    ]
    if config.optimization_mode in {"description_only", "alternating_schema_description", "two_stage_alternating_schema_description", "two_stage_coverage_plan_schema_description", "two_stage_evidence_routing_schema_description"}:
        paths.append(config.schema_description_system_prompt_path)
        paths.append(config.schema_description_initial_prompt_path)
    if config.optimization_mode in {"two_stage_alternating_schema_description", "two_stage_coverage_plan_schema_description", "two_stage_evidence_routing_schema_description"}:
        paths.extend((config.evidence_initial_prompt_path, config.evidence_system_prompt_path, config.resolve_initial_prompt_path, config.resolve_system_prompt_path))
    if config.optimization_mode == "two_stage_coverage_plan_schema_description":
        paths.append(config.coverage_plan_system_prompt_path)
    if config.optimization_mode == "two_stage_evidence_routing_schema_description":
        paths.extend((config.evidence_routing_initial_prompt_path, config.evidence_routing_system_prompt_path))
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Required experiment input does not exist: {path}")


def _resolve(base: Path, raw: Any) -> Path:
    path = Path(str(raw))
    return path if path.is_absolute() else (base / path).resolve()


def _required(mapping: dict[str, Any], key: str) -> Any:
    value = mapping.get(key)
    if value in (None, ""):
        raise ValueError(f"Missing required config value: {key}")
    return value


def _positive_int(value: Any, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive.")
    return parsed


def _optional_positive_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive or null.")
    return parsed


def _nonnegative_int(value: Any, name: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative.")
    return parsed







