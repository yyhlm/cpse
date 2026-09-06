from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

JsonValue = dict[str, Any] | list[Any] | str | int | float | bool | None
SplitName = Literal["train", "blind_test"]


@dataclass(frozen=True)
class DocumentPair:
    document_id: str
    pdf_path: Path
    gold_path: Path
    gold: JsonValue
    pdf_sha256: str
    gold_sha256: str


@dataclass(frozen=True)
class ComplexityMetrics:
    node_count: int
    object_key_count: int
    array_item_count: int
    populated_scalar_count: int
    max_depth: int
    populated_path_count: int
    score: int


@dataclass(frozen=True)
class SplitDocument:
    pair: DocumentPair
    complexity: ComplexityMetrics
    split: SplitName
    rank: int


@dataclass(frozen=True)
class DatasetSplit:
    algorithm_version: str
    train: tuple[SplitDocument, ...]
    blind_test: tuple[SplitDocument, ...]


@dataclass(frozen=True)
class ModelConfig:
    model: str
    base_url: str | None
    api_key_env: str
    temperature: float
    max_output_tokens: int | None
    timeout_seconds: float
    max_retries: int
    use_instructions: bool = False
    proxy: str | None = None
    # API 协议："responses"（默认，OpenAI Responses API，PDF 文件输入）
    # | "chat_completions_pdf_images"（OpenAI Chat Completions，PDF 逐页渲染成图片输入）
    # | "gemini_generate_content"（Google Gemini 原生 generateContent，PDF 字节 inline_data 直传）。
    api_protocol: str = "responses"
    # 推理开关：设为 "none" 关闭模型的 thinking（SenseNova 网关实测 `reasoning_effort: "none"`
    # 生效，completion_tokens 从 ~247 降到 2）。用于 chat_completions_pdf_images 通道；
    # responses 通道暂不映射。属于模型指纹的一部分，变更需新 run-id。
    reasoning_effort: str | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    data_dir: Path
    schema_path: Path
    output_root: Path
    initial_prompt_path: Path
    extraction_system_prompt_path: Path
    judge_system_prompt_path: Path
    gold_audit_system_prompt_path: Path
    extractor: ModelConfig
    judge: ModelConfig
    gold_audit: ModelConfig
    include_error_locations: bool = False
    include_pdf: bool = False
    gold_audit_enabled: bool = False
    max_iterations: int = 3
    max_parallel_calls: int = 3
    train_count: int = 3
    train_ids: tuple[str, ...] | None = None
    split_algorithm_version: str = "structural-v1"
    request_format_version: str = "responses-pdf-v1"
    # Alternating schema-description mode. ``single`` preserves the legacy
    # prompt-only optimizer; ``alternating_schema_description`` runs joint
    # schema-description patch and extraction-prompt optimization rounds
    optimization_mode: str = "single"
    schema_description_system_prompt_path: Path | None = None
    schema_description_initial_prompt_path: Path | None = None
    evidence_initial_prompt_path: Path | None = None
    evidence_system_prompt_path: Path | None = None
    resolve_initial_prompt_path: Path | None = None
    resolve_system_prompt_path: Path | None = None
    # Fixed protocol prompt for the optional Stage-1.5 coverage plan; it is not a TextGrad variable.
    coverage_plan_system_prompt_path: Path | None = None
    # Four-variable two-stage mode: a fixed protocol plus a separately learned
    # evidence-routing instruction for the Stage-1.5 routing map.
    evidence_routing_initial_prompt_path: Path | None = None
    evidence_routing_system_prompt_path: Path | None = None
    schema_patch_policy_version: str = "1"
    cache_enabled: bool = True
    # Direct comparison baselines. These modes never invoke an optimizer.
    # schema_free_direct omits the extraction contract; few_shot_direct injects
    # the fixed three-document training pool as labeled demonstrations.


@dataclass(frozen=True)
class PredictionArtifact:
    raw_response: str
    parsed_prediction: JsonValue | None
    validation_errors: tuple[dict[str, str], ...]

    @property
    def is_valid(self) -> bool:
        return self.parsed_prediction is not None and not self.validation_errors


@dataclass(frozen=True)
class PathError:
    path: str
    category: Literal[
        "omission",
        "hallucination",
        "incorrect_value",
        "incorrect_unit",
        "incorrect_condition",
        "schema_error",
    ]
    evidence: str


@dataclass(frozen=True)
class JudgeResult:
    score: float
    optimization_feedback: str
    path_errors: tuple[PathError, ...] = field(default_factory=tuple)
    score_breakdown: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateSummary:
    candidate_id: str
    prompt_hash: str
    parent_candidate_id: str | None
    document_scores: dict[str, float | None]
    mean_score: float | None
    accepted: bool
    decision_reason: str


@dataclass(frozen=True)
class AlternatingCandidate:
    """One jointly evaluated schema-and-extraction training round.

    The baseline is ``round-000/joint``. Every later candidate holds both
    TextGrad-updated prompts, a validated description-only schema patch, and
    the scores from their single combined evaluation. The pair and schema are
    accepted or rolled back atomically.
    """

    candidate_id: str  # e.g. "round-001/joint"
    phase: Literal["joint"]
    parent_candidate_id: str | None
    schema_prompt_hash: str
    extraction_prompt_hash: str
    schema_sha256: str  # full DSL hash including descriptions
    structural_sha256: str  # description-stripped structural fingerprint
    evidence_prompt_hash: str = ""
    evidence_routing_prompt_hash: str = ""
    resolve_prompt_hash: str = ""
    patch_sha256: str | None = None
    changed_description_paths: tuple[str, ...] = ()
    validation_status: str = "valid"  # "valid" | "no_valid_patch" | "proposal_failed" | "rejected"
    document_scores: dict[str, float | None] = field(default_factory=dict)
    mean_score: float | None = None
    accepted: bool = False
    decision_reason: str = ""


@dataclass(frozen=True)
class CallFingerprint:
    role: str
    document_id: str
    document_hash: str
    schema_hash: str
    prompt_hash: str
    model_config_hash: str
    request_format_version: str
    max_retries: int

    def as_key(self) -> str:
        return "|".join(
            (
                self.role,
                self.document_id,
                self.document_hash,
                self.schema_hash,
                self.prompt_hash,
                self.model_config_hash,
                self.request_format_version,
                str(self.max_retries),
            )
        )


@dataclass(frozen=True)
class RunManifest:
    manifest_version: str
    split_algorithm_version: str
    schema_path: str
    schema_sha256: str
    initial_prompt_sha256: str
    model_fingerprints: dict[str, str]
    documents: tuple[SplitDocument, ...]
    config_fingerprint: str


@dataclass(frozen=True)
class GoldAuditFinding:
    path: str
    status: Literal["supported", "unsupported", "ambiguous", "possible_omission"]
    page: str | None
    evidence: str


@dataclass(frozen=True)
class GoldAuditResult:
    findings: tuple[GoldAuditFinding, ...] = field(default_factory=tuple)




