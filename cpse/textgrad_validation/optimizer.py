from __future__ import annotations

import json
from dataclasses import replace
from collections.abc import Callable, Sequence
from typing import Any

from .artifacts import sha256_json
from .judge import _GOLD_MATCH_BREAKDOWN_LIMITS, _PDF_BREAKDOWN_LIMITS
from .models import AlternatingCandidate, CandidateSummary, JudgeResult
from .responses_client import api_operation
from .schema_description import (
    SchemaDescriptionPatchError,
    apply_description_patch_document,
    parse_description_patch_text,
    retain_valid_description_patches,
    structural_fingerprint,
    validate_patch_document,
)

EvaluatePrompt = Callable[[str], dict[str, JudgeResult | None]]


_OPRO_SYSTEM_PROMPT = """You are the optimizer in OPRO (Optimization by PROmpting).
Propose exactly one improved scientific-PDF-to-JSON extraction prompt from the complete optimization history.
Use candidate objective scores as the optimization signal. Never copy or
memorize paper-specific names, identifiers, numerical values, JSONPaths, or phrases. The schema is immutable:
do not add, remove, rename, retype, or relocate fields, and do not propose schema-description edits. Preserve
JSON-only output and make the result directly usable as the extraction prompt. Return exactly one JSON object:
{"prompt":"..."}
"""


_GEPA_PROPOSAL_SYSTEM_PROMPT = """You are the proposal component of GEPA.
Revise the supplied scientific-PDF-to-JSON extraction instruction using the reflective evaluation records.
Return one directly usable, general extraction instruction. Learn transferable extraction rules; do not
memorize paper-specific names, identifiers, values, JSONPaths, or phrases. The schema is immutable: do not
add, remove, rename, retype, or relocate fields. Preserve JSON-only output requirements.
Return exactly one JSON object with the requested component names as string values.
"""

_GEPA_MANIFEST_PROPOSAL_SYSTEM_PROMPT = """You are the proposal component of GEPA.
Jointly revise two prompts in a manifest-conditioned scientific-PDF extraction pipeline:
manifest_prompt discovers document metadata and a complete ordered material-identity manifest;
resolve_prompt converts bounded, non-overlapping manifest slices into full nested records.
Use the reflective evaluation records to learn reusable rules, preserve each prompt's role, and do
not memorize paper-specific names, identifiers, values, JSONPaths, or phrases. The schema and all
field descriptions are immutable. Return exactly one JSON object with the requested component
names as string values.
"""


def _feedback_for_optimizer(feedback: str) -> str:
    """Return the judge feedback for the optimizer: the concrete CURRENT_ERROR
    failure instance AND the transferable GENERAL_RULE, kept together.

    The backward engine is an LLM: it generalizes better from a concrete
    failure pattern plus the abstract rule than from the rule alone. Document
    specifics (sample names, IDs, values) are deliberately retained here as the
    pattern anchor — the backward engine's eval system prompt and the step
    constraints instruct it to generalize, and the blind test catches any
    overfit to training-only specifics.
    """
    return feedback.strip()


class TextGradPromptOptimizer:
    """Optimizes exactly one extraction prompt; every valid evaluation is accepted."""

    def __init__(self, *, engine: Any, evaluate_prompt: EvaluatePrompt, constraints: Sequence[str] = ()):
        self._engine = engine
        self._evaluate_prompt = evaluate_prompt
        self._constraints = list(constraints) or [
            "Return only a directly usable extraction prompt.",
            "Preserve the schema and JSON-only output requirements.",
            "Keep the prompt general.",
            "Never instruct output that violates the schema structure: do not create fields, nesting, arrays, "
            "records, or entities the schema does not allow, do not flatten or merge records the schema "
            "distinguishes, and keep every value in the exact schema-declared field.",
            "Emit empty values only for keys genuinely unreported; never emit a sample or property record "
            "whose content is entirely empty to pad the output.",
        ]

    def optimize(
        self,
        initial_prompt: str,
        max_iterations: int,
        *,
        evaluate_candidate: Callable[[str, str], dict[str, JudgeResult | None]] | None = None,
    ) -> tuple[str, list[CandidateSummary]]:
        evaluate = evaluate_candidate or (lambda _candidate_id, prompt: self._evaluate_prompt(prompt))
        import textgrad as tg

        variable = tg.Variable(
            initial_prompt,
            requires_grad=True,
            role_description="The sole mutable PDF-to-JSON extraction prompt.",
        )
        tg.set_backward_engine(self._engine, override=True)
        return self._optimize_with_variable(tg, variable, max_iterations, evaluate)

    def _optimize_with_variable(
        self, tg: Any, variable: Any, max_iterations: int, evaluate: Callable[[str, str], dict[str, JudgeResult | None]]
    ) -> tuple[str, list[CandidateSummary]]:
        summaries: list[CandidateSummary] = []
        current_results = evaluate("candidate-000", str(variable.value))
        baseline = self._summarize_candidate("candidate-000", str(variable.value), None, results=current_results)
        summaries.append(baseline)
        best_prompt = str(variable.value)
        best_results = current_results

        for iteration in range(1, max_iterations + 1):
            losses = _build_per_document_losses(tg, variable, best_results, self._engine)
            if not losses:
                summaries.append(
                    CandidateSummary(
                        candidate_id=f"candidate-{iteration:03d}",
                        prompt_hash="",
                        parent_candidate_id=summaries[-1].candidate_id,
                        document_scores={},
                        mean_score=None,
                        accepted=False,
                        decision_reason="No complete valid training feedback is available.",
                    )
                )
                continue
            loss = tg.sum(losses)
            optimizer = tg.TextualGradientDescent(parameters=[variable], engine=self._engine, constraints=self._constraints)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            proposal = str(variable.value).strip()
            current_results = evaluate(f"candidate-{iteration:03d}", proposal)
            candidate = self._summarize_candidate(
                f"candidate-{iteration:03d}",
                proposal,
                summaries[-1].candidate_id,
                results=current_results,
            )
            summaries.append(candidate)
            if candidate.accepted:
                best_prompt = proposal
                best_results = current_results
            else:
                variable.set_value(best_prompt)
        return best_prompt, summaries

    def optimize_resume(
        self,
        initial_prompt: str,
        best_prompt: str,
        best_results: dict[str, JudgeResult | None],
        existing_candidates: list[CandidateSummary],
        start_iteration: int,
        max_iterations: int,
        *,
        evaluate_candidate: Callable[[str, str], dict[str, JudgeResult | None]],
    ) -> tuple[str, CandidateSummary, list[CandidateSummary]]:
        """Resume iterations from ``start_iteration`` to ``max_iterations``.

        Returns ``(best_prompt, best_candidate, new_candidates)``. The TextGrad
        variable is re-seeded from ``best_prompt``; internal backward state
        (gradients, optimizer momentum) from the original run is not persisted,
        so this is a fresh but deterministic continuation — not a bit-exact
        trajectory replay. Callers must not assume resumed candidate scores
        equal a single full run's scores at the same iteration index.
        """
        import textgrad as tg

        variable = tg.Variable(
            best_prompt,
            requires_grad=True,
            role_description="The sole mutable PDF-to-JSON extraction prompt.",
        )
        tg.set_backward_engine(self._engine, override=True)

        best_candidate = next(
            (c for c in reversed(existing_candidates) if c.accepted and c.mean_score is not None),
            existing_candidates[0],
        )
        current_best_results = dict(best_results)
        new_candidates: list[CandidateSummary] = []

        for iteration in range(start_iteration, max_iterations + 1):
            candidate_id = f"candidate-{iteration:03d}"
            losses = _build_per_document_losses(tg, variable, current_best_results, self._engine)
            if not losses:
                placeholder = CandidateSummary(
                    candidate_id=candidate_id,
                    prompt_hash="",
                    parent_candidate_id=best_candidate.candidate_id,
                    document_scores={},
                    mean_score=None,
                    accepted=False,
                    decision_reason="No complete valid training feedback is available.",
                )
                new_candidates.append(placeholder)
                continue
            loss = tg.sum(losses)
            optimizer = tg.TextualGradientDescent(parameters=[variable], engine=self._engine, constraints=self._constraints)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            proposal = str(variable.value).strip()
            results = evaluate_candidate(candidate_id, proposal)
            candidate = self._summarize_candidate(
                candidate_id,
                proposal,
                best_candidate.candidate_id,
                results=results,
            )
            new_candidates.append(candidate)
            if candidate.accepted:
                best_prompt = proposal
                best_candidate = candidate
                current_best_results = results
            else:
                variable.set_value(best_prompt)
        return best_prompt, best_candidate, new_candidates

    def _summarize_candidate(
        self,
        candidate_id: str,
        prompt: str,
        parent_id: str | None,
        results: dict[str, JudgeResult | None],
    ) -> CandidateSummary:
        from .artifacts import sha256_json

        document_scores = {doc_id: (result.score if result is not None else None) for doc_id, result in results.items()}
        valid_scores = [score for score in document_scores.values() if score is not None]
        expected = len(results)
        if len(valid_scores) != expected:
            return CandidateSummary(
                candidate_id=candidate_id,
                prompt_hash=sha256_json(prompt),
                parent_candidate_id=parent_id,
                document_scores=document_scores,
                mean_score=None,
                accepted=False,
                decision_reason=f"All {expected} training documents require valid judge scores.",
            )
        mean_score = sum(valid_scores) / len(valid_scores)
        return CandidateSummary(
            candidate_id=candidate_id,
            prompt_hash=sha256_json(prompt),
            parent_candidate_id=parent_id,
            document_scores=document_scores,
            mean_score=mean_score,
            accepted=True,
            decision_reason="Baseline candidate." if parent_id is None else "Valid training scores.",
        )


class OproPromptOptimizer:
    """History-conditioned black-box prompt search following the OPRO pattern."""

    def __init__(self, *, client: Any, evaluate_prompt: EvaluatePrompt):
        self._client = client
        self._evaluate_prompt = evaluate_prompt

    def optimize(
        self,
        initial_prompt: str,
        max_iterations: int,
        *,
        evaluate_candidate: Callable[[str, str], dict[str, JudgeResult | None]] | None = None,
    ) -> tuple[str, list[CandidateSummary]]:
        evaluate = evaluate_candidate or (lambda _candidate_id, prompt: self._evaluate_prompt(prompt))
        baseline_results = evaluate("candidate-000", initial_prompt)
        baseline = self._summarize(
            "candidate-000", initial_prompt, None, baseline_results, best_mean=None, baseline=True
        )
        summaries = [baseline]
        history = [self._history_entry("candidate-000", initial_prompt, baseline_results)]
        best_prompt = initial_prompt
        best_mean = baseline.mean_score

        for iteration in range(1, max_iterations + 1):
            proposal = self._propose(history)
            candidate_id = f"candidate-{iteration:03d}"
            results = evaluate(candidate_id, proposal)
            candidate = self._summarize(
                candidate_id, proposal, summaries[-1].candidate_id, results, best_mean=best_mean, baseline=False
            )
            summaries.append(candidate)
            history.append(self._history_entry(candidate_id, proposal, results))
            if candidate.mean_score is not None and (best_mean is None or candidate.mean_score >= best_mean):
                best_prompt = proposal
                best_mean = candidate.mean_score
        return best_prompt, summaries

    def optimize_resume(
        self,
        initial_prompt: str,
        best_prompt: str,
        best_results: dict[str, JudgeResult | None],
        existing_candidates: list[CandidateSummary],
        start_iteration: int,
        max_iterations: int,
        *,
        evaluate_candidate: Callable[[str, str], dict[str, JudgeResult | None]],
    ) -> tuple[str, CandidateSummary, list[CandidateSummary]]:
        best_candidate = max(
            (candidate for candidate in existing_candidates if candidate.mean_score is not None),
            key=lambda candidate: (candidate.mean_score, candidate.candidate_id),
        )
        # Checkpoints retain hashes and scores but not every prompt body. Seed
        # resumed OPRO with the initial and current-best prompts plus the full
        # historical score trajectory; fresh uninterrupted runs retain all text.
        history = [{
            "candidate_id": candidate.candidate_id,
            "prompt": (
                initial_prompt if candidate.candidate_id == "candidate-000"
                else best_prompt if candidate.candidate_id == best_candidate.candidate_id
                else "[prompt body retained in run artifact]"
            ),
            "mean_score": candidate.mean_score,
            "document_scores": candidate.document_scores,
        } for candidate in existing_candidates]
        best_mean = best_candidate.mean_score
        new_candidates: list[CandidateSummary] = []
        for iteration in range(start_iteration, max_iterations + 1):
            proposal = self._propose(history)
            candidate_id = f"candidate-{iteration:03d}"
            results = evaluate_candidate(candidate_id, proposal)
            candidate = self._summarize(
                candidate_id, proposal, best_candidate.candidate_id, results, best_mean=best_mean, baseline=False
            )
            new_candidates.append(candidate)
            history.append(self._history_entry(candidate_id, proposal, results))
            if candidate.mean_score is not None and (best_mean is None or candidate.mean_score >= best_mean):
                best_prompt = proposal
                best_candidate = candidate
                best_mean = candidate.mean_score
        return best_prompt, best_candidate, new_candidates

    def _propose(self, history: list[dict[str, Any]]) -> str:
        history_snapshot = json.loads(json.dumps(history, ensure_ascii=False))
        with api_operation("optimizer_proposal"):
            raw = self._client.complete_json(system_prompt=_OPRO_SYSTEM_PROMPT, payload={"history": history_snapshot})
        value = json.loads(raw)
        prompt = str(value.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("OPRO response must contain a non-empty prompt.")
        return prompt

    @staticmethod
    def _history_entry(
        candidate_id: str, prompt: str, results: dict[str, JudgeResult | None]
    ) -> dict[str, Any]:
        scores = {doc_id: (result.score if result is not None else None) for doc_id, result in results.items()}
        return {
            "candidate_id": candidate_id,
            "prompt": prompt,
            "mean_score": _mean_score(results),
            "document_scores": scores,
        }

    @staticmethod
    def _summarize(
        candidate_id: str,
        prompt: str,
        parent_id: str | None,
        results: dict[str, JudgeResult | None],
        *,
        best_mean: float | None,
        baseline: bool,
    ) -> CandidateSummary:
        scores = {doc_id: (result.score if result is not None else None) for doc_id, result in results.items()}
        mean_score = _mean_score(results)
        accepted = mean_score is not None and (baseline or best_mean is None or mean_score >= best_mean)
        if mean_score is None:
            reason = "All training documents require valid judge scores."
        elif baseline:
            reason = "Baseline candidate."
        elif accepted:
            reason = "OPRO proposal matched or improved the best training mean."
        else:
            reason = "OPRO proposal did not improve the best training mean."
        return CandidateSummary(
            candidate_id=candidate_id,
            prompt_hash=sha256_json(prompt),
            parent_candidate_id=parent_id,
            document_scores=scores,
            mean_score=mean_score,
            accepted=accepted,
            decision_reason=reason,
        )


class GepaPromptOptimizer:
    """Official GEPA adapter for the prompt-only PDF extraction search space."""

    def __init__(
        self,
        *,
        client: Any,
        evaluate_prompt: EvaluatePrompt,
        optimize_fn: Callable[..., Any] | None = None,
    ):
        self._client = client
        self._evaluate_prompt = evaluate_prompt
        self._optimize_fn = optimize_fn

    def optimize(
        self,
        initial_prompt: str,
        max_iterations: int,
        *,
        evaluate_candidate: Callable[[str, str], dict[str, JudgeResult | None]] | None = None,
    ) -> tuple[str, list[CandidateSummary]]:
        evaluate = evaluate_candidate or (lambda _candidate_id, prompt: self._evaluate_prompt(prompt))
        prompt_results: dict[str, dict[str, JudgeResult | None]] = {}
        prompt_candidate_ids: dict[str, str] = {initial_prompt: "candidate-000"}

        def evaluate_one(prompt: str) -> dict[str, JudgeResult | None]:
            if prompt not in prompt_results:
                candidate_id = prompt_candidate_ids.setdefault(prompt, f"candidate-{len(prompt_candidate_ids):03d}")
                prompt_results[prompt] = evaluate(candidate_id, prompt)
            return prompt_results[prompt]

        baseline_results = evaluate_one(initial_prompt)
        document_ids = sorted(baseline_results)

        try:
            from gepa import EvaluationBatch, optimize as official_optimize
        except ImportError as exc:
            raise RuntimeError("gepa_prompt_only requires optional dependency gepa==0.1.4") from exc

        outer = self

        class Adapter:
            def evaluate(self, batch, candidate, capture_traces=False):
                prompt = str(candidate["extraction_prompt"])
                results = evaluate_one(prompt)
                outputs = []
                scores = []
                traces = []
                for document_id in batch:
                    result = results.get(document_id)
                    outputs.append(result.optimization_feedback if result is not None else "evaluation failed")
                    scores.append((result.score / 100.0) if result is not None else 0.0)
                    if capture_traces:
                        traces.append({
                            "document_id": document_id,
                            "score": result.score if result is not None else 0.0,
                            "score_breakdown": (result.score_breakdown or {}) if result is not None else {},
                            "feedback": result.optimization_feedback if result is not None else "evaluation failed",
                        })
                return EvaluationBatch(
                    outputs=outputs,
                    scores=scores,
                    trajectories=traces if capture_traces else None,
                    num_metric_calls=len(batch),
                )

            def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
                records = [{
                    "Inputs": {"document_id": trace["document_id"]},
                    "Generated Outputs": {"score": trace["score"], "score_breakdown": trace["score_breakdown"]},
                    "Feedback": trace["feedback"],
                } for trace in (eval_batch.trajectories or [])]
                return {component: records for component in components_to_update}

            def propose_new_texts(self, candidate, reflective_dataset, components_to_update):
                """GEPA 0.1.4's supported adapter-owned proposal hook.

                The project's Responses client is not GEPA's ReflectionLM interface, so proposal
                generation lives here rather than relying on GEPA's LiteLLM-based default reflector.
                """
                with api_operation("optimizer_proposal"):
                    raw = outer._client.complete_json(
                    system_prompt=_GEPA_PROPOSAL_SYSTEM_PROMPT,
                    payload={
                        "candidate": {name: str(candidate[name]) for name in components_to_update},
                        "reflective_dataset": {
                            name: list(reflective_dataset.get(name, ())) for name in components_to_update
                        },
                        "components_to_update": list(components_to_update),
                    },
                    )
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise ValueError("GEPA proposal must be a JSON object.")
                proposals: dict[str, str] = {}
                for component in components_to_update:
                    proposal = str(value.get(component, "")).strip()
                    if not proposal:
                        raise ValueError(f"GEPA proposal must contain a non-empty {component!r} value.")
                    proposals[component] = proposal
                return proposals

        def reflection_lm(prompt):
            with api_operation("optimizer_reflection"):
                return outer._client.complete_text(system_prompt=None, prompt=prompt, json_mode=False)

        optimize_fn = self._optimize_fn or official_optimize
        result = optimize_fn(
            seed_candidate={"extraction_prompt": initial_prompt},
            trainset=document_ids,
            valset=document_ids,
            adapter=Adapter(),
            reflection_lm=reflection_lm,
            candidate_selection_strategy="pareto",
            frontier_type="instance",
            reflection_minibatch_size=len(document_ids),
            max_metric_calls=(max_iterations + 1) * len(document_ids),
            acceptance_criterion="improvement_or_equal",
            display_progress_bar=False,
            cache_evaluation=False,
            seed=0,
        )

        summaries: list[CandidateSummary] = []
        best_mean: float | None = None
        for index, candidate in enumerate(result.candidates):
            prompt = str(candidate["extraction_prompt"]).strip()
            results = evaluate_one(prompt)
            mean_score = _mean_score(results)
            parent_indexes = result.parents[index] if index < len(result.parents) else []
            parent_index = next((item for item in parent_indexes if item is not None), None)
            parent_id = None if parent_index is None else prompt_candidate_ids[str(result.candidates[parent_index]["extraction_prompt"])]
            accepted = mean_score is not None and (best_mean is None or mean_score >= best_mean)
            summaries.append(OproPromptOptimizer._summarize(
                prompt_candidate_ids[prompt], prompt, parent_id, results,
                best_mean=best_mean, baseline=index == 0,
            ))
            if accepted:
                best_mean = mean_score
        best = result.best_candidate
        best_prompt = str(best["extraction_prompt"] if isinstance(best, dict) else best).strip()
        return best_prompt, summaries

    def optimize_resume(self, *args, **kwargs):
        raise RuntimeError(
            "Official GEPA owns its candidate/Pareto state; extend max_iterations with a new run-id instead of "
            "the legacy single-mode optimize_resume path."
        )


class GepaManifestOptimizer:
    """Official GEPA search over manifest-construction and record-resolution prompts."""

    def __init__(self, *, client: Any, evaluate_prompts: Callable[[str, str], dict[str, JudgeResult | None]], optimize_fn: Callable[..., Any] | None = None):
        self._client = client
        self._evaluate_prompts = evaluate_prompts
        self._optimize_fn = optimize_fn

    def optimize(self, initial_manifest_prompt: str, initial_resolve_prompt: str, max_iterations: int, *, evaluate_candidate: Callable[[str, str, str], dict[str, JudgeResult | None]] | None = None) -> tuple[tuple[str, str], list[CandidateSummary]]:
        seed = {"manifest_prompt": initial_manifest_prompt, "resolve_prompt": initial_resolve_prompt}
        results_by_key: dict[str, dict[str, JudgeResult | None]] = {}
        ids_by_key: dict[str, str] = {}

        def key(candidate: dict[str, Any]) -> str:
            return json.dumps({name: str(candidate[name]) for name in ("manifest_prompt", "resolve_prompt")}, ensure_ascii=False, sort_keys=True)

        def evaluate_one(candidate: dict[str, Any]) -> dict[str, JudgeResult | None]:
            candidate_key = key(candidate)
            candidate_id = ids_by_key.setdefault(candidate_key, f"candidate-{len(ids_by_key):03d}")
            if candidate_key not in results_by_key:
                manifest_prompt = str(candidate["manifest_prompt"])
                resolve_prompt = str(candidate["resolve_prompt"])
                results_by_key[candidate_key] = (
                    evaluate_candidate(candidate_id, manifest_prompt, resolve_prompt)
                    if evaluate_candidate is not None
                    else self._evaluate_prompts(manifest_prompt, resolve_prompt)
                )
            return results_by_key[candidate_key]

        baseline_results = evaluate_one(seed)
        document_ids = sorted(baseline_results)
        try:
            from gepa import EvaluationBatch, optimize as official_optimize
        except ImportError as exc:
            raise RuntimeError("gepa_manifest requires optional dependency gepa==0.1.4") from exc
        outer = self

        class Adapter:
            def evaluate(self, batch, candidate, capture_traces=False):
                results = evaluate_one(candidate)
                outputs, scores, traces = [], [], []
                for document_id in batch:
                    result = results.get(document_id)
                    feedback = result.optimization_feedback if result is not None else "evaluation failed"
                    outputs.append(feedback)
                    scores.append((result.score / 100.0) if result is not None else 0.0)
                    if capture_traces:
                        traces.append({"document_id": document_id, "score": result.score if result else 0.0, "score_breakdown": (result.score_breakdown or {}) if result else {}, "feedback": feedback})
                return EvaluationBatch(outputs=outputs, scores=scores, trajectories=traces if capture_traces else None, num_metric_calls=len(batch))

            def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
                records = [{"Inputs": {"document_id": trace["document_id"]}, "Generated Outputs": {"score": trace["score"], "score_breakdown": trace["score_breakdown"]}, "Feedback": trace["feedback"]} for trace in (eval_batch.trajectories or [])]
                return {component: records for component in components_to_update}

            def propose_new_texts(self, candidate, reflective_dataset, components_to_update):
                with api_operation("optimizer_proposal"):
                    raw = outer._client.complete_json(system_prompt=_GEPA_MANIFEST_PROPOSAL_SYSTEM_PROMPT, payload={"candidate": {name: str(candidate[name]) for name in components_to_update}, "reflective_dataset": {name: list(reflective_dataset.get(name, ())) for name in components_to_update}, "components_to_update": list(components_to_update)})
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise ValueError("GEPA proposal must be a JSON object.")
                proposals = {component: str(value.get(component, "")).strip() for component in components_to_update}
                missing = next((name for name, value in proposals.items() if not value), None)
                if missing is not None:
                    raise ValueError(f"GEPA proposal must contain a non-empty {missing!r} value.")
                return proposals

        def reflection_lm(prompt):
            with api_operation("optimizer_reflection"):
                return outer._client.complete_text(system_prompt=None, prompt=prompt, json_mode=False)

        result = (self._optimize_fn or official_optimize)(seed_candidate=seed, trainset=document_ids, valset=document_ids, adapter=Adapter(), reflection_lm=reflection_lm, candidate_selection_strategy="pareto", frontier_type="instance", reflection_minibatch_size=len(document_ids), max_metric_calls=(max_iterations + 1) * len(document_ids), acceptance_criterion="improvement_or_equal", display_progress_bar=False, cache_evaluation=False, seed=0)
        summaries: list[CandidateSummary] = []
        best_mean: float | None = None
        for index, candidate in enumerate(result.candidates):
            candidate_key = key(candidate)
            candidate_results = evaluate_one(candidate)
            mean_score = _mean_score(candidate_results)
            parent_indexes = result.parents[index] if index < len(result.parents) else []
            parent_index = next((item for item in parent_indexes if item is not None), None)
            parent_id = None if parent_index is None else ids_by_key[key(result.candidates[parent_index])]
            accepted = mean_score is not None and (best_mean is None or mean_score >= best_mean)
            summaries.append(CandidateSummary(candidate_id=ids_by_key[candidate_key], prompt_hash=sha256_json(json.loads(candidate_key)), parent_candidate_id=parent_id, document_scores={doc_id: (item.score if item else None) for doc_id, item in candidate_results.items()}, mean_score=mean_score, accepted=accepted, decision_reason="baseline" if index == 0 else ("non-decreasing mean" if accepted else "lower mean")))
            if accepted:
                best_mean = mean_score
        best = result.best_candidate
        return (str(best["manifest_prompt"]).strip(), str(best["resolve_prompt"]).strip()), summaries


class MiproV2InstructionOptimizer:
    """Official DSPy MIPROv2 with both labeled and bootstrapped demos disabled."""

    def __init__(
        self,
        *,
        client: Any,
        evaluate_prompt: EvaluatePrompt,
        compile_fn: Callable[..., str] | None = None,
    ):
        self._client = client
        self._evaluate_prompt = evaluate_prompt
        self._compile_fn = compile_fn

    def optimize(
        self,
        initial_prompt: str,
        max_iterations: int,
        *,
        evaluate_candidate: Callable[[str, str], dict[str, JudgeResult | None]] | None = None,
    ) -> tuple[str, list[CandidateSummary]]:
        evaluate = evaluate_candidate or (lambda _candidate_id, prompt: self._evaluate_prompt(prompt))
        prompt_results: dict[str, dict[str, JudgeResult | None]] = {}
        prompt_ids: dict[str, str] = {}

        def evaluate_instruction(prompt: str) -> dict[str, JudgeResult | None]:
            prompt = prompt.strip()
            if prompt not in prompt_results:
                candidate_id = prompt_ids.setdefault(prompt, f"candidate-{len(prompt_ids):03d}")
                prompt_results[prompt] = evaluate(candidate_id, prompt)
            return prompt_results[prompt]

        baseline_results = evaluate_instruction(initial_prompt)
        document_ids = sorted(baseline_results)
        compile_fn = self._compile_fn or self._compile_official
        best_prompt = compile_fn(
            initial_prompt=initial_prompt,
            document_ids=document_ids,
            evaluate_instruction=evaluate_instruction,
            client=self._client,
            num_candidates=max_iterations + 1,
            num_trials=max_iterations + 1,
            max_bootstrapped_demos=0,
            max_labeled_demos=0,
        ).strip()
        if not best_prompt:
            raise ValueError("MIPROv2 instruction-only returned an empty instruction.")
        evaluate_instruction(best_prompt)

        summaries: list[CandidateSummary] = []
        best_mean: float | None = None
        for index, (prompt, results) in enumerate(prompt_results.items()):
            mean_score = _mean_score(results)
            accepted = mean_score is not None and (best_mean is None or mean_score >= best_mean)
            summaries.append(CandidateSummary(
                candidate_id=prompt_ids[prompt],
                prompt_hash=sha256_json(prompt),
                parent_candidate_id=None if index == 0 else summaries[-1].candidate_id,
                document_scores={
                    doc_id: (result.score if result is not None else None) for doc_id, result in results.items()
                },
                mean_score=mean_score,
                accepted=accepted,
                decision_reason=(
                    "Baseline candidate." if index == 0 else
                    "MIPROv2 instruction candidate matched or improved the best training mean." if accepted else
                    "MIPROv2 instruction candidate did not improve the best training mean."
                ),
            ))
            if accepted:
                best_mean = mean_score
        return best_prompt, summaries

    @staticmethod
    def _compile_official(
        *,
        initial_prompt: str,
        document_ids: list[str],
        evaluate_instruction: Callable[[str], dict[str, JudgeResult | None]],
        client: Any,
        num_candidates: int,
        num_trials: int,
        max_bootstrapped_demos: int,
        max_labeled_demos: int,
    ) -> str:
        try:
            import dspy
        except ImportError as exc:
            raise RuntimeError("mipro_v2_instruction_only requires optional dependency dspy==3.3.1") from exc

        class ResponsesDspyLM(dspy.BaseLM):
            forward_contract = "typed_lm"

            def __init__(self):
                super().__init__(model="responses/project-judge", model_type="responses", cache=False)

            def forward(self, request):
                serialized = json.dumps(
                    [message.model_dump(mode="json") for message in request.messages], ensure_ascii=False
                )
                with api_operation("optimizer_compile"):
                    text = client.complete_text(system_prompt=None, prompt=serialized, json_mode=False)
                return dspy.LMResponse.from_text(text, model=self.model)

        signature = dspy.Signature("document_id -> score").with_instructions(initial_prompt)

        class CallbackProgram(dspy.Module):
            def __init__(self):
                super().__init__()
                self.extractor = dspy.Predict(signature)

            def forward(self, document_id):
                instruction = self.extractor.signature.instructions.strip()
                result = evaluate_instruction(instruction).get(str(document_id))
                return dspy.Prediction(score=(result.score / 100.0) if result is not None else 0.0)

        examples = [dspy.Example(document_id=document_id).with_inputs("document_id") for document_id in document_ids]

        def metric(_example, prediction, trace=None):
            return float(prediction.score)

        lm = ResponsesDspyLM()
        optimizer = dspy.MIPROv2(
            metric=metric,
            prompt_model=lm,
            task_model=lm,
            auto=None,
            num_candidates=num_candidates,
            num_threads=1,
            seed=0,
            verbose=False,
        )
        best_program = optimizer.compile(
            CallbackProgram(),
            trainset=examples,
            valset=examples,
            num_trials=num_trials,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
            minibatch=False,
            program_aware_proposer=True,
            data_aware_proposer=False,
            tip_aware_proposer=True,
            fewshot_aware_proposer=False,
        )
        return best_program.extractor.signature.instructions

    def optimize_resume(self, *args, **kwargs):
        raise RuntimeError(
            "MIPROv2 owns its Bayesian-search state; extend the budget with a new run-id instead of the "
            "legacy single-mode optimize_resume path."
        )


class MiproV2Optimizer(MiproV2InstructionOptimizer):
    """DSPy MIPROv2 instruction + labeled PDF-text/Gold demonstration baseline."""

    def __init__(
        self,
        *,
        client: Any,
        evaluate_prompt: EvaluatePrompt,
        demonstrations: list[dict[str, str]],
        compile_fn: Callable[..., str] | None = None,
    ):
        super().__init__(client=client, evaluate_prompt=evaluate_prompt)
        self._demonstrations = demonstrations
        self._full_compile_fn = compile_fn

    def optimize(
        self,
        initial_prompt: str,
        max_iterations: int,
        *,
        evaluate_candidate: Callable[[str, str], dict[str, JudgeResult | None]] | None = None,
    ) -> tuple[str, list[CandidateSummary]]:
        evaluate = evaluate_candidate or (lambda _candidate_id, prompt: self._evaluate_prompt(prompt))
        prompt_results: dict[str, dict[str, JudgeResult | None]] = {}
        prompt_ids: dict[str, str] = {}

        def evaluate_instruction(prompt: str) -> dict[str, JudgeResult | None]:
            prompt = prompt.strip()
            if prompt not in prompt_results:
                candidate_id = prompt_ids.setdefault(prompt, f"candidate-{len(prompt_ids):03d}")
                prompt_results[prompt] = evaluate(candidate_id, prompt)
            return prompt_results[prompt]

        baseline_results = evaluate_instruction(initial_prompt)
        document_ids = sorted(baseline_results)
        compile_fn = self._full_compile_fn or self._compile_official_full
        best_prompt = compile_fn(
            initial_prompt=initial_prompt,
            document_ids=document_ids,
            demonstrations=self._demonstrations,
            evaluate_instruction=evaluate_instruction,
            client=self._client,
            num_candidates=max_iterations + 1,
            num_trials=max_iterations + 1,
            max_bootstrapped_demos=0,
            max_labeled_demos=1,
        ).strip()
        if not best_prompt:
            raise ValueError("MIPROv2 returned an empty instruction/demo prompt.")
        evaluate_instruction(best_prompt)

        summaries: list[CandidateSummary] = []
        best_mean: float | None = None
        for index, (prompt, results) in enumerate(prompt_results.items()):
            mean_score = _mean_score(results)
            accepted = mean_score is not None and (best_mean is None or mean_score >= best_mean)
            summaries.append(CandidateSummary(
                candidate_id=prompt_ids[prompt],
                prompt_hash=sha256_json(prompt),
                parent_candidate_id=None if index == 0 else summaries[-1].candidate_id,
                document_scores={
                    doc_id: (result.score if result is not None else None) for doc_id, result in results.items()
                },
                mean_score=mean_score,
                accepted=accepted,
                decision_reason=(
                    "Baseline candidate." if index == 0 else
                    "MIPROv2 instruction/demo candidate matched or improved the best training mean." if accepted else
                    "MIPROv2 instruction/demo candidate did not improve the best training mean."
                ),
            ))
            if accepted:
                best_mean = mean_score
        return best_prompt, summaries

    @staticmethod
    def _compile_official_full(
        *,
        initial_prompt: str,
        document_ids: list[str],
        demonstrations: list[dict[str, str]],
        evaluate_instruction: Callable[[str], dict[str, JudgeResult | None]],
        client: Any,
        num_candidates: int,
        num_trials: int,
        max_bootstrapped_demos: int,
        max_labeled_demos: int,
    ) -> str:
        try:
            import dspy
        except ImportError as exc:
            raise RuntimeError("mipro_v2 requires optional dependency dspy==3.3.1") from exc

        class ResponsesDspyLM(dspy.BaseLM):
            forward_contract = "typed_lm"

            def __init__(self):
                super().__init__(model="responses/project-judge", model_type="responses", cache=False)

            def forward(self, request):
                serialized = json.dumps(
                    [message.model_dump(mode="json") for message in request.messages], ensure_ascii=False
                )
                with api_operation("optimizer_compile"):
                    text = client.complete_text(system_prompt=None, prompt=serialized, json_mode=False)
                return dspy.LMResponse.from_text(text, model=self.model)

        signature = dspy.Signature("document_id, document_text -> gold_json").with_instructions(initial_prompt)

        class CallbackProgram(dspy.Module):
            def __init__(self):
                super().__init__()
                self.extractor = dspy.Predict(signature)

            def rendered_prompt(self):
                instruction = self.extractor.signature.instructions.strip()
                if not self.extractor.demos:
                    return instruction
                blocks = []
                for index, demo in enumerate(self.extractor.demos, start=1):
                    blocks.append(
                        f"DEMONSTRATION {index}\nINPUT_PDF_TEXT:\n{demo.document_text}\n"
                        f"TARGET_JSON:\n{demo.gold_json}"
                    )
                return instruction + "\n\nLABELED PDF-TO-JSON DEMONSTRATIONS:\n\n" + "\n\n".join(blocks)

            def forward(self, document_id, document_text):
                prompt = self.rendered_prompt()
                result = evaluate_instruction(prompt).get(str(document_id))
                return dspy.Prediction(
                    gold_json="",
                    score=(result.score / 100.0) if result is not None else 0.0,
                )

        by_id = {item["document_id"]: item for item in demonstrations}
        examples = [
            dspy.Example(
                document_id=document_id,
                document_text=by_id[document_id]["document_text"],
                gold_json=by_id[document_id]["gold_json"],
            ).with_inputs("document_id", "document_text")
            for document_id in document_ids
        ]

        def metric(_example, prediction, trace=None):
            return float(prediction.score)

        lm = ResponsesDspyLM()
        optimizer = dspy.MIPROv2(
            metric=metric,
            prompt_model=lm,
            task_model=lm,
            auto=None,
            num_candidates=num_candidates,
            num_threads=1,
            seed=0,
            verbose=False,
        )
        _configure_labeled_only_mipro_candidates(
            optimizer,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
        )
        best_program = optimizer.compile(
            CallbackProgram(),
            trainset=examples,
            valset=examples,
            num_trials=num_trials,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
            minibatch=False,
            program_aware_proposer=True,
            data_aware_proposer=True,
            tip_aware_proposer=True,
            fewshot_aware_proposer=True,
        )
        return best_program.rendered_prompt()


def _configure_labeled_only_mipro_candidates(
    optimizer: Any, *, max_bootstrapped_demos: int, max_labeled_demos: int
) -> None:
    """Avoid DSPy 3.3.1 sampling bootstrapped demo counts from the empty range 1..0."""
    if max_bootstrapped_demos == 0 and max_labeled_demos > 0:
        optimizer.num_fewshot_candidates = min(3, optimizer.num_fewshot_candidates)


# Union of the judge's dimension maxima (PDF four-dim and non-PDF two-dim keys do
# not overlap), used to render each score against its true maximum.
_DIMENSION_LIMITS = {**_PDF_BREAKDOWN_LIMITS, **_GOLD_MATCH_BREAKDOWN_LIMITS}


def _format_score_breakdown(result: JudgeResult | None) -> str:
    """Render a document's per-dimension judge scores as ``name: score/max`` lines.

    The judge fills ``score_breakdown`` (PDF: document_sample/process/properties/
    characterization; non-PDF: coverage/accuracy) and enforces
    ``score == sum(score_breakdown)``. Feeding these into the loss lets the
    backward engine see exactly which dimension collapsed (e.g. process 0/30)
    instead of having a 30-point process crash drown in a single total score.
    """
    breakdown = (result.score_breakdown if result is not None else None) or {}
    if not breakdown:
        return ""
    lines = [f"{name}: {score}/{_DIMENSION_LIMITS[name]}" for name, score in sorted(breakdown.items())]
    return "\n".join(lines)


def _build_per_document_losses(
    tg: Any,
    variable: Any,
    results: dict[str, JudgeResult | None],
    engine: Any,
    eval_system_prompt: str = (
        "Improve the extraction prompt using the supplied evaluator feedback, "
        "which lists per-dimension scores (SCORE_BREAKDOWN) and pairs a concrete "
        "failure instance (CURRENT_ERROR) with a transferable extraction rule "
        "(GENERAL_RULE). Prioritize the dimensions with the lowest scores in "
        "SCORE_BREAKDOWN when choosing which failures to fix — fix the weak "
        "dimensions broadly, not only the single lowest one. Learn the general "
        "pattern from the concrete instance; do not copy or memorize "
        "document-specific facts, identifiers, values, or paths. Do not change "
        "its JSON/schema role."
    ),
) -> list[Any]:
    """Build one ``TextLoss`` per document so each contributes its own gradient.

    Each loss carries the document's concrete failure instance AND its
    transferable rule; the backward engine is an LLM that generalizes better
    from the concrete pattern plus the abstract rule than from the rule alone.
    Per-dimension scores from ``score_breakdown`` are injected as
    ``SCORE_BREAKDOWN`` so a dimension collapse is directly visible to the
    optimizer instead of being masked by the total score.
    """
    losses: list[Any] = []
    for doc_id in sorted(results):
        result = results[doc_id]
        if result is None:
            continue
        fb = _feedback_for_optimizer(result.optimization_feedback)
        if not fb:
            continue
        breakdown_block = _format_score_breakdown(result)
        extra = f"\n\nSCORE_BREAKDOWN:\n{breakdown_block}" if breakdown_block else ""
        doc_loss = tg.TextLoss(
            eval_system_prompt=eval_system_prompt,
            engine=engine,
        )(
            variable
            + tg.Variable(
                f"\n\nEVALUATOR FEEDBACK:\n{fb}{extra}",
                requires_grad=False,
                role_description=(
                    "Evaluator feedback: per-dimension scores plus a concrete failure "
                    "instance and a transferable rule."
                ),
            )
        )
        losses.append(doc_loss)
    return losses


_SCHEMA_PROMPT_EVAL_SYSTEM_PROMPT = (
    "Improve the schema-description patch proposal prompt using the supplied evaluator feedback, "
    "which lists per-dimension scores (SCORE_BREAKDOWN) and pairs a concrete failure instance "
    "(CURRENT_ERROR) with a transferable rule (GENERAL_RULE). "
    "Prioritize the dimensions with the lowest scores in SCORE_BREAKDOWN when choosing which "
    "failures to fix — fix the weak dimensions broadly, not only the single lowest one. "
    "Learn the general pattern from the concrete instance; do not copy or memorize document-specific facts, "
    "identifiers, values, or paths, and do not gain authority "
    "to change schema structure, field names, types, requiredness, items, or properties. "
    "The fixed schema-patch system prompt alone defines the output protocol — exactly "
    "{\"patches\":[{\"path\":\"$.<full-DSL-path>.description\",\"description\":\"...\"}]} with full paths that "
    "include every \".properties\" level — and your improved prompt must not override it or teach any other format."
)


def _mean_score(results: dict[str, JudgeResult | None]) -> float | None:
    scores = [result.score for result in results.values() if result is not None]
    if not scores or len(scores) != len(results):
        return None
    return sum(scores) / len(scores)


class AlternatingSchemaDescriptionOptimizer:
    """Optimize two prompts, then score one combined schema/extraction state per round."""

    def __init__(
        self,
        *,
        engine: Any,
        propose_patch: Callable[[str, str, dict[str, Any], dict[str, JudgeResult | None]], str],
        evaluate: Callable[[str, str, dict[str, Any]], dict[str, JudgeResult | None]],
        base_schema_dsl: dict[str, Any],
        schema_constraints: Sequence[str] = (),
        extraction_constraints: Sequence[str] = (),
        optimize_extraction_prompt: bool = True,
        on_round_complete: Callable[[str, str, dict[str, Any], list[AlternatingCandidate]], None] | None = None,
    ):
        self._engine = engine
        self._propose_patch = propose_patch
        self._evaluate = evaluate
        self._base_schema_dsl = base_schema_dsl
        self._on_round_complete = on_round_complete
        self._optimize_extraction_prompt = optimize_extraction_prompt
        self._schema_constraints = list(schema_constraints) or [
            "Return only a directly usable schema-description patch proposal prompt.",
            "Do not add document-specific facts, identifiers, values, or paths.",
            "Never change the output protocol defined by the fixed schema-patch system prompt: the patch model "
            "must keep returning exactly {\"patches\":[{\"path\":\"$.<full-DSL-path>.description\",\"description\":\"...\"}]} "
            "with full paths that include every \".properties\" level. Do not teach it any other key (scope, array) "
            "or any abbreviated path that omits \".properties\".",
        ]
        self._extraction_constraints = list(extraction_constraints) or [
            "Return only a directly usable extraction prompt.",
            "Preserve the schema and JSON-only output requirements.",
            "Keep the prompt general.",
            "Never instruct output that violates the schema structure: do not create fields, nesting, arrays, "
            "records, or entities the schema does not allow, do not flatten or merge records the schema "
            "distinguishes, and keep every value in the exact schema-declared field.",
            "Require emitting every schema-required key; empty object/string when absent, never omit.",
            "Emit empty values only for keys genuinely unreported; never emit a sample or property record "
            "whose content is entirely empty to pad the output.",
            "Strictness must not reduce coverage: keep extracting every schema-representable sample, step, "
            "and property, never trading completeness for caution.",
        ]

    def optimize(
        self, schema_prompt: str, extraction_prompt: str, max_rounds: int
    ) -> tuple[str, str, dict[str, Any], list[AlternatingCandidate]]:
        import textgrad as tg

        schema_var, extract_var = self._variables(tg, schema_prompt, extraction_prompt)
        baseline_results = self._evaluate("round-000/joint", extraction_prompt, self._base_schema_dsl)
        baseline_mean = _mean_score(baseline_results)
        candidates = [
            self._summarize(
                "round-000/joint", None, schema_prompt, extraction_prompt,
                self._base_schema_dsl, _scores(baseline_results), baseline_mean,
                True, "Baseline candidate."
            )
        ]
        self._notify_round_complete(schema_prompt, extraction_prompt, self._base_schema_dsl, candidates)
        return self._run_rounds(
            tg, schema_var, extract_var, self._base_schema_dsl, extraction_prompt,
            baseline_results, candidates, 1, max_rounds,
        )

    def optimize_resume(
        self,
        schema_prompt: str,
        extraction_prompt: str,
        selected_schema_dsl: dict[str, Any],
        best_results: dict[str, JudgeResult | None],
        existing_candidates: list[AlternatingCandidate],
        start_round: int,
        max_rounds: int,
    ) -> tuple[str, str, dict[str, Any], list[AlternatingCandidate]]:
        import textgrad as tg

        schema_var, extract_var = self._variables(tg, schema_prompt, extraction_prompt)
        return self._run_rounds(
            tg, schema_var, extract_var, selected_schema_dsl, extraction_prompt,
            best_results, list(existing_candidates), start_round, max_rounds,
        )

    def _variables(self, tg: Any, schema_prompt: str, extraction_prompt: str) -> tuple[Any, Any]:
        tg.set_backward_engine(self._engine, override=True)
        return (
            tg.Variable(schema_prompt, requires_grad=True, role_description="Prompt that proposes description-only schema patches."),
            tg.Variable(extraction_prompt, requires_grad=True, role_description="The sole mutable PDF-to-JSON extraction prompt."),
        )

    def _run_rounds(
        self, tg: Any, schema_var: Any, extract_var: Any, selected_schema: dict[str, Any],
        best_extraction_prompt: str, best_results: dict[str, JudgeResult | None],
        candidates: list[AlternatingCandidate], start_round: int, max_rounds: int,
    ) -> tuple[str, str, dict[str, Any], list[AlternatingCandidate]]:
        best_schema_prompt = str(schema_var.value).strip()
        for round_index in range(start_round, max_rounds + 1):
            phase_id = f"round-{round_index:03d}/joint"
            parent = candidates[-1].candidate_id
            previous_schema_prompt = best_schema_prompt
            previous_extraction_prompt = best_extraction_prompt
            schema_losses = _build_per_document_losses(
                tg, schema_var, best_results, self._engine, eval_system_prompt=_SCHEMA_PROMPT_EVAL_SYSTEM_PROMPT
            )
            extraction_losses = (
                _build_per_document_losses(tg, extract_var, best_results, self._engine)
                if self._optimize_extraction_prompt
                else []
            )
            # Update the two prompts with SEPARATE optimizers, each constrained to
            # its own role. A single optimizer over both variables with merged
            # constraints leaks the schema-patch protocol into the extraction
            # prompt (TextGrad rewrote it to emit {"patches":[...]} instead of
            # extracting PDFs). Both variables still update every round.
            step_error: str | None = None
            if schema_losses:
                schema_optimizer = tg.TextualGradientDescent(
                    parameters=[schema_var], engine=self._engine, constraints=self._schema_constraints
                )
                schema_optimizer.zero_grad()
                tg.sum(schema_losses).backward()
                try:
                    schema_optimizer.step()
                except Exception as exc:
                    step_error = f"schema optimizer step failed: {exc}"
                    schema_var.set_value(previous_schema_prompt)
            if extraction_losses and step_error is None:
                extraction_optimizer = tg.TextualGradientDescent(
                    parameters=[extract_var], engine=self._engine, constraints=self._extraction_constraints
                )
                extraction_optimizer.zero_grad()
                tg.sum(extraction_losses).backward()
                try:
                    extraction_optimizer.step()
                except Exception as exc:
                    step_error = f"extraction optimizer step failed: {exc}"
                    extract_var.set_value(previous_extraction_prompt)
            candidate_schema_prompt = str(schema_var.value).strip()
            candidate_extraction_prompt = str(extract_var.value).strip()
            candidate_schema = selected_schema
            patch_sha: str | None = None
            changed_paths: tuple[str, ...] = ()
            validation_status = "proposal_failed"
            validation_error: str | None = step_error
            if step_error is None:
                try:
                    raw_patch = self._propose_patch(phase_id, candidate_schema_prompt, selected_schema, best_results)
                    patch_document = parse_description_patch_text(raw_patch)
                    patch_document, dropped = retain_valid_description_patches(selected_schema, patch_document)
                    if not patch_document["patches"] and not self._optimize_extraction_prompt:
                        validation_status = "no_valid_patch"
                        validation_error = "All proposed description patches were dropped: " + "; ".join(dropped)
                    else:
                        candidate_schema = apply_description_patch_document(selected_schema, patch_document)
                        patch_sha = sha256_json(patch_document)
                        changed_paths = tuple(str(patch["path"]) for patch in patch_document["patches"])
                        validation_status = "valid"
                except (SchemaDescriptionPatchError, ValueError, TypeError, RuntimeError) as exc:
                    # RuntimeError covers API/truncation exhaustion from the patch
                    # client; a failed proposal must degrade to proposal_failed,
                    # never kill the whole run.
                    validation_error = str(exc)
            results = self._evaluate(phase_id, candidate_extraction_prompt, candidate_schema) if validation_status == "valid" else {}
            mean = _mean_score(results)
            accepted = validation_status == "valid" and mean is not None
            reason = (
                "Valid description-only patch with complete training scores."
                if accepted else
                f"Joint candidate rejected ({validation_status}): {validation_error or 'incomplete training scores'}.")
            candidates.append(self._summarize(
                phase_id, parent, candidate_schema_prompt, candidate_extraction_prompt, candidate_schema,
                _scores(results), mean, accepted, reason, patch_sha, changed_paths, validation_status,
            ))
            if accepted:
                best_schema_prompt = candidate_schema_prompt
                best_extraction_prompt = candidate_extraction_prompt
                selected_schema = candidate_schema
                best_results = results
            else:
                schema_var.set_value(previous_schema_prompt)
                extract_var.set_value(previous_extraction_prompt)
            self._notify_round_complete(best_schema_prompt, best_extraction_prompt, selected_schema, candidates)
        return best_schema_prompt, best_extraction_prompt, selected_schema, candidates

    def _notify_round_complete(
        self,
        schema_prompt: str,
        extraction_prompt: str,
        selected_schema: dict[str, Any],
        candidates: list[AlternatingCandidate],
    ) -> None:
        if self._on_round_complete is not None:
            self._on_round_complete(schema_prompt, extraction_prompt, selected_schema, list(candidates))

    def _summarize(
        self, candidate_id: str, parent: str | None, schema_prompt: str, extraction_prompt: str,
        schema_dsl: dict[str, Any], document_scores: dict[str, float | None], mean_score: float | None,
        accepted: bool, decision_reason: str, patch_sha: str | None = None,
        changed_paths: tuple[str, ...] = (), validation_status: str = "valid",
    ) -> AlternatingCandidate:
        return AlternatingCandidate(
            candidate_id=candidate_id, phase="joint", parent_candidate_id=parent,
            schema_prompt_hash=sha256_json(schema_prompt), extraction_prompt_hash=sha256_json(extraction_prompt),
            schema_sha256=sha256_json(schema_dsl), structural_sha256=structural_fingerprint(schema_dsl),
            patch_sha256=patch_sha, changed_description_paths=changed_paths, validation_status=validation_status,
            document_scores=document_scores, mean_score=mean_score, accepted=accepted, decision_reason=decision_reason,
        )


def _scores(results: dict[str, JudgeResult | None]) -> dict[str, float | None]:
    return {doc_id: (result.score if result is not None else None) for doc_id, result in results.items()}


class TwoStageOptimizer:
    """Optimize three prompts jointly: evidence, resolve, and schema-description.

    One combined evaluation per round (two-stage extraction + judge). Each of the
    three variables is updated by its own ``TextualGradientDescent`` optimizer
    with its own constraints, but all three share the same per-document loss from
    ``_build_per_document_losses`` against the same best-results state.
    """

    def __init__(
        self,
        *,
        engine: Any,
        propose_patch: Callable[[str, str, dict[str, Any], dict[str, JudgeResult | None]], str],
        evaluate: Callable[[str, str, str, dict[str, Any]], dict[str, JudgeResult | None]],
        base_schema_dsl: dict[str, Any],
        schema_constraints: Sequence[str] = (),
        evidence_constraints: Sequence[str] = (),
        resolve_constraints: Sequence[str] = (),
        on_round_complete: Callable[[str, str, str, dict[str, Any], list[AlternatingCandidate]], None] | None = None,
    ):
        self._engine = engine
        self._propose_patch = propose_patch
        self._evaluate = evaluate
        self._base_schema_dsl = base_schema_dsl
        self._on_round_complete = on_round_complete
        self._schema_constraints = list(schema_constraints) or [
            "Return only a directly usable schema-description patch proposal prompt.",
            "Do not add document-specific facts, identifiers, values, or paths.",
            "Never change the output protocol defined by the fixed schema-patch system prompt: the patch model "
            "must keep returning exactly {\"patches\":[{\"path\":\"$.<full-DSL-path>.description\",\"description\":\"...\"}]} "
            "with full paths that include every \".properties\" level. Do not teach it any other key (scope, array) "
            "or any abbreviated path that omits \".properties\".",
        ]
        self._evidence_constraints = list(evidence_constraints) or [
            "Return only a directly usable evidence collection prompt.",
            "Preserve the schema and JSON-only output requirements.",
            "Keep the prompt general.",
            "Never instruct output that violates the schema structure: do not create fields, nesting, arrays, "
            "records, or entities the schema does not allow, do not flatten or merge records the schema "
            "distinguishes, and keep every value in the exact schema-declared field.",
            "Require emitting every schema-required key; empty object/string when absent, never omit.",
            "Emit empty values only for keys genuinely unreported; never emit a sample or property record "
            "whose content is entirely empty to pad the output.",
            "Strictness must not reduce coverage: keep extracting every schema-representable sample, step, "
            "and property, never trading completeness for caution.",
        ]
        self._resolve_constraints = list(resolve_constraints) or [
            "Return only a directly usable resolve prompt.",
            "Preserve the schema and JSON-only output requirements.",
            "Keep the prompt general.",
            "Never instruct output that violates the schema structure: do not create fields, nesting, arrays, "
            "records, or entities the schema does not allow, do not flatten or merge records the schema "
            "distinguishes, and keep every value in the exact schema-declared field.",
            "Require emitting every schema-required key; empty object/string when absent, never omit.",
            "Emit empty values only for keys genuinely unreported; never emit a sample or property record "
            "whose content is entirely empty to pad the output.",
            "Strictness must not reduce coverage: keep extracting every schema-representable sample, step, "
            "and property, never trading completeness for caution.",
        ]

    def optimize(
        self, schema_prompt: str, evidence_prompt: str, resolve_prompt: str, max_rounds: int
    ) -> tuple[str, str, str, dict[str, Any], list[AlternatingCandidate]]:
        import textgrad as tg

        schema_var, evidence_var, resolve_var = self._variables(tg, schema_prompt, evidence_prompt, resolve_prompt)
        baseline_results = self._evaluate("round-000/joint", evidence_prompt, resolve_prompt, self._base_schema_dsl)
        baseline_mean = _mean_score(baseline_results)
        candidates = [
            self._summarize(
                "round-000/joint", None, schema_prompt, evidence_prompt, resolve_prompt,
                self._base_schema_dsl, _scores(baseline_results), baseline_mean,
                True, "Baseline candidate."
            )
        ]
        self._notify_round_complete(schema_prompt, evidence_prompt, resolve_prompt, self._base_schema_dsl, candidates)
        return self._run_rounds(
            tg, schema_var, evidence_var, resolve_var, self._base_schema_dsl,
            evidence_prompt, resolve_prompt, baseline_results, candidates, 1, max_rounds,
        )

    def optimize_resume(
        self,
        schema_prompt: str,
        evidence_prompt: str,
        resolve_prompt: str,
        selected_schema: dict[str, Any],
        best_results: dict[str, JudgeResult | None],
        existing_candidates: list[AlternatingCandidate],
        start_round: int,
        max_rounds: int,
    ) -> tuple[str, str, str, dict[str, Any], list[AlternatingCandidate]]:
        """Resume two-stage rounds from ``start_round`` to ``max_rounds``.

        Re-seeds the three TextGrad variables from the last accepted state
        (``schema_prompt``/``evidence_prompt``/``resolve_prompt``) and reruns the
        same loop as ``optimize`` over the remaining rounds, appending to
        ``existing_candidates``. Internal TextGrad backward state (gradients,
        optimizer momentum) is not persisted, so this is a fresh but
        deterministic continuation — not a bit-exact trajectory replay. Callers
        must not assume resumed candidate scores equal a single full run's scores
        at the same round index.
        """
        import textgrad as tg

        schema_var, evidence_var, resolve_var = self._variables(tg, schema_prompt, evidence_prompt, resolve_prompt)
        return self._run_rounds(
            tg, schema_var, evidence_var, resolve_var, selected_schema,
            evidence_prompt, resolve_prompt, best_results, existing_candidates,
            start_round, max_rounds,
        )

    def _variables(self, tg: Any, schema_prompt: str, evidence_prompt: str, resolve_prompt: str) -> tuple[Any, Any, Any]:
        tg.set_backward_engine(self._engine, override=True)
        return (
            tg.Variable(schema_prompt, requires_grad=True, role_description="Prompt that proposes description-only schema patches."),
            tg.Variable(evidence_prompt, requires_grad=True, role_description="Prompt that collects evidence and builds an identity manifest."),
            tg.Variable(resolve_prompt, requires_grad=True, role_description="Prompt that resolves identity batches into complete polymer records."),
        )

    def _run_rounds(
        self, tg: Any, schema_var: Any, evidence_var: Any, resolve_var: Any,
        selected_schema: dict[str, Any], best_evidence_prompt: str, best_resolve_prompt: str,
        best_results: dict[str, JudgeResult | None], candidates: list[AlternatingCandidate],
        start_round: int, max_rounds: int,
    ) -> tuple[str, str, str, dict[str, Any], list[AlternatingCandidate]]:
        best_schema_prompt = str(schema_var.value).strip()
        for round_index in range(start_round, max_rounds + 1):
            phase_id = f"round-{round_index:03d}/joint"
            parent = candidates[-1].candidate_id
            previous_schema_prompt = best_schema_prompt
            previous_evidence_prompt = best_evidence_prompt
            previous_resolve_prompt = best_resolve_prompt

            # Schema-description variable: backward with schema eval prompt.
            schema_losses = _build_per_document_losses(
                tg, schema_var, best_results, self._engine, eval_system_prompt=_SCHEMA_PROMPT_EVAL_SYSTEM_PROMPT
            )
            # Evidence variable: backward with extraction eval prompt.
            evidence_losses = _build_per_document_losses(tg, evidence_var, best_results, self._engine)
            # Resolve variable: backward with extraction eval prompt.
            resolve_losses = _build_per_document_losses(tg, resolve_var, best_results, self._engine)

            step_error: str | None = None

            if schema_losses:
                schema_optimizer = tg.TextualGradientDescent(
                    parameters=[schema_var], engine=self._engine, constraints=self._schema_constraints
                )
                schema_optimizer.zero_grad()
                tg.sum(schema_losses).backward()
                try:
                    schema_optimizer.step()
                except Exception as exc:
                    step_error = f"schema optimizer step failed: {exc}"
                    schema_var.set_value(previous_schema_prompt)

            if evidence_losses and step_error is None:
                evidence_optimizer = tg.TextualGradientDescent(
                    parameters=[evidence_var], engine=self._engine, constraints=self._evidence_constraints
                )
                evidence_optimizer.zero_grad()
                tg.sum(evidence_losses).backward()
                try:
                    evidence_optimizer.step()
                except Exception as exc:
                    step_error = f"evidence optimizer step failed: {exc}"
                    evidence_var.set_value(previous_evidence_prompt)

            if resolve_losses and step_error is None:
                resolve_optimizer = tg.TextualGradientDescent(
                    parameters=[resolve_var], engine=self._engine, constraints=self._resolve_constraints
                )
                resolve_optimizer.zero_grad()
                tg.sum(resolve_losses).backward()
                try:
                    resolve_optimizer.step()
                except Exception as exc:
                    step_error = f"resolve optimizer step failed: {exc}"
                    resolve_var.set_value(previous_resolve_prompt)

            candidate_schema_prompt = str(schema_var.value).strip()
            candidate_evidence_prompt = str(evidence_var.value).strip()
            candidate_resolve_prompt = str(resolve_var.value).strip()
            candidate_schema = selected_schema
            patch_sha: str | None = None
            changed_paths: tuple[str, ...] = ()
            validation_status = "proposal_failed"
            validation_error: str | None = step_error

            if step_error is None:
                try:
                    raw_patch = self._propose_patch(phase_id, candidate_schema_prompt, selected_schema, best_results)
                    patch_document = parse_description_patch_text(raw_patch)
                    patch_document, dropped = retain_valid_description_patches(selected_schema, patch_document)
                    candidate_schema = apply_description_patch_document(selected_schema, patch_document)
                    patch_sha = sha256_json(patch_document)
                    changed_paths = tuple(str(patch["path"]) for patch in patch_document["patches"])
                    validation_status = "valid"
                except (SchemaDescriptionPatchError, ValueError, TypeError, RuntimeError) as exc:
                    validation_error = str(exc)

            results = self._evaluate(phase_id, candidate_evidence_prompt, candidate_resolve_prompt, candidate_schema) if validation_status == "valid" else {}
            mean = _mean_score(results)
            accepted = validation_status == "valid" and mean is not None
            reason = (
                "Valid description-only patch with complete training scores."
                if accepted else
                f"Joint candidate rejected ({validation_status}): {validation_error or 'incomplete training scores'}.")
            candidates.append(self._summarize(
                phase_id, parent, candidate_schema_prompt, candidate_evidence_prompt, candidate_resolve_prompt,
                candidate_schema, _scores(results), mean, accepted, reason, patch_sha, changed_paths, validation_status,
            ))
            if accepted:
                best_schema_prompt = candidate_schema_prompt
                best_evidence_prompt = candidate_evidence_prompt
                best_resolve_prompt = candidate_resolve_prompt
                selected_schema = candidate_schema
                best_results = results
            else:
                schema_var.set_value(previous_schema_prompt)
                evidence_var.set_value(previous_evidence_prompt)
                resolve_var.set_value(previous_resolve_prompt)
            self._notify_round_complete(
                best_schema_prompt, best_evidence_prompt, best_resolve_prompt, selected_schema, candidates
            )
        return best_schema_prompt, best_evidence_prompt, best_resolve_prompt, selected_schema, candidates

    def _notify_round_complete(
        self,
        schema_prompt: str,
        evidence_prompt: str,
        resolve_prompt: str,
        selected_schema: dict[str, Any],
        candidates: list[AlternatingCandidate],
    ) -> None:
        if self._on_round_complete is not None:
            self._on_round_complete(schema_prompt, evidence_prompt, resolve_prompt, selected_schema, list(candidates))

    def _summarize(
        self, candidate_id: str, parent: str | None, schema_prompt: str, evidence_prompt: str,
        resolve_prompt: str, schema_dsl: dict[str, Any], document_scores: dict[str, float | None],
        mean_score: float | None, accepted: bool, decision_reason: str,
        patch_sha: str | None = None, changed_paths: tuple[str, ...] = (),
        validation_status: str = "valid",
    ) -> AlternatingCandidate:
        return AlternatingCandidate(
            candidate_id=candidate_id, phase="joint", parent_candidate_id=parent,
            schema_prompt_hash=sha256_json(schema_prompt),
            extraction_prompt_hash=sha256_json(evidence_prompt),
            evidence_prompt_hash=sha256_json(evidence_prompt),
            resolve_prompt_hash=sha256_json(resolve_prompt),
            schema_sha256=sha256_json(schema_dsl), structural_sha256=structural_fingerprint(schema_dsl),
            patch_sha256=patch_sha, changed_description_paths=changed_paths, validation_status=validation_status,
            document_scores=document_scores, mean_score=mean_score, accepted=accepted, decision_reason=decision_reason,
        )


class EvidenceRoutingTwoStageOptimizer(TwoStageOptimizer):
    """Four-variable two-stage optimizer: index, routing, resolve, schema text."""

    def __init__(self, *, evaluate, routing_constraints=(), on_round_complete=None, **kwargs):
        super().__init__(evaluate=lambda *_args: {}, on_round_complete=None, **kwargs)
        self._evaluate_routing = evaluate
        self._routing_callback = on_round_complete
        self._routing_constraints = list(routing_constraints) or [
            "Return only a directly usable evidence-routing prompt.",
            "Keep the prompt general and PDF-grounded; never use Gold or paper-specific facts.",
            "Route only supplied manifest identities to schema field groups, evidence anchors, and explicit shared relations.",
            "Do not create final JSON fields, entities, values, or unsupported relations.",
        ]

    def optimize(self, schema_prompt, evidence_prompt, routing_prompt, resolve_prompt, max_rounds):
        import textgrad as tg
        schema_var, evidence_var, resolve_var = self._variables(tg, schema_prompt, evidence_prompt, resolve_prompt)
        routing_var = tg.Variable(routing_prompt, requires_grad=True, role_description="Prompt that routes entity evidence to field groups before resolve.")
        baseline = self._evaluate_routing("round-000/joint", evidence_prompt, routing_prompt, resolve_prompt, self._base_schema_dsl)
        candidates = [replace(
            self._summarize("round-000/joint", None, schema_prompt, evidence_prompt, resolve_prompt, self._base_schema_dsl, _scores(baseline), _mean_score(baseline), True, "Baseline candidate."),
            evidence_routing_prompt_hash=sha256_json(routing_prompt),
        )]
        self._notify_routing(schema_prompt, evidence_prompt, routing_prompt, resolve_prompt, self._base_schema_dsl, candidates)
        return self._run_routing_rounds(tg, schema_var, evidence_var, routing_var, resolve_var, self._base_schema_dsl, baseline, candidates, 1, max_rounds)

    def _run_routing_rounds(self, tg, schema_var, evidence_var, routing_var, resolve_var, selected_schema, best_results, candidates, start_round, max_rounds):
        for round_index in range(start_round, max_rounds + 1):
            phase_id = f"round-{round_index:03d}/joint"
            previous = [str(var.value).strip() for var in (schema_var, evidence_var, routing_var, resolve_var)]
            losses = [
                _build_per_document_losses(tg, schema_var, best_results, self._engine, eval_system_prompt=_SCHEMA_PROMPT_EVAL_SYSTEM_PROMPT),
                _build_per_document_losses(tg, evidence_var, best_results, self._engine),
                _build_per_document_losses(tg, routing_var, best_results, self._engine),
                _build_per_document_losses(tg, resolve_var, best_results, self._engine),
            ]
            error = None
            for var, loss, constraints, label, prior in zip((schema_var,evidence_var,routing_var,resolve_var), losses, (self._schema_constraints,self._evidence_constraints,self._routing_constraints,self._resolve_constraints), ("schema","evidence","routing","resolve"), previous):
                if not loss or error is not None:
                    continue
                opt = tg.TextualGradientDescent(parameters=[var], engine=self._engine, constraints=constraints)
                opt.zero_grad(); tg.sum(loss).backward()
                try: opt.step()
                except Exception as exc:
                    error = f"{label} optimizer step failed: {exc}"; var.set_value(prior)
            schema_text, evidence_text, routing_text, resolve_text = [str(var.value).strip() for var in (schema_var,evidence_var,routing_var,resolve_var)]
            candidate_schema, patch_sha, changed, status = selected_schema, None, (), "proposal_failed"
            if error is None:
                try:
                    patch_document = parse_description_patch_text(self._propose_patch(phase_id, schema_text, selected_schema, best_results))
                    patch_document, _dropped = retain_valid_description_patches(selected_schema, patch_document)
                    candidate_schema = apply_description_patch_document(selected_schema, patch_document)
                    patch_sha, changed, status = sha256_json(patch_document), tuple(str(p["path"]) for p in patch_document["patches"]), "valid"
                except (SchemaDescriptionPatchError, ValueError, TypeError, RuntimeError) as exc: error = str(exc)
            results = self._evaluate_routing(phase_id, evidence_text, routing_text, resolve_text, candidate_schema) if status == "valid" else {}
            mean = _mean_score(results); accepted = status == "valid" and mean is not None
            candidates.append(replace(
                self._summarize(phase_id, candidates[-1].candidate_id, schema_text, evidence_text, resolve_text, candidate_schema, _scores(results), mean, accepted, "Valid four-prompt candidate." if accepted else f"Joint candidate rejected ({status}): {error or 'incomplete training scores'}.", patch_sha, changed, status),
                evidence_routing_prompt_hash=sha256_json(routing_text),
            ))
            if accepted: selected_schema, best_results = candidate_schema, results
            else:
                for var, prior in zip((schema_var,evidence_var,routing_var,resolve_var), previous): var.set_value(prior)
            current = [str(var.value).strip() for var in (schema_var,evidence_var,routing_var,resolve_var)]
            self._notify_routing(*current, selected_schema, candidates)
        values = [str(var.value).strip() for var in (schema_var,evidence_var,routing_var,resolve_var)]
        return (*values, selected_schema, candidates)

    def _notify_routing(self, schema_prompt, evidence_prompt, routing_prompt, resolve_prompt, schema_dsl, candidates):
        if self._routing_callback is not None:
            self._routing_callback(schema_prompt, evidence_prompt, routing_prompt, resolve_prompt, schema_dsl, list(candidates))


    def optimize_resume(
        self,
        schema_prompt: str,
        evidence_prompt: str,
        routing_prompt: str,
        resolve_prompt: str,
        selected_schema: dict[str, Any],
        best_results: dict[str, JudgeResult | None],
        existing_candidates: list[AlternatingCandidate],
        start_round: int,
        max_rounds: int,
    ) -> tuple[str, str, str, str, dict[str, Any], list[AlternatingCandidate]]:
        """Resume all four learnable variables from the selected training state."""
        import textgrad as tg

        schema_var, evidence_var, resolve_var = self._variables(tg, schema_prompt, evidence_prompt, resolve_prompt)
        routing_var = tg.Variable(
            routing_prompt,
            requires_grad=True,
            role_description="Prompt that routes entity evidence to field groups before resolve.",
        )
        return self._run_routing_rounds(
            tg, schema_var, evidence_var, routing_var, resolve_var,
            selected_schema, best_results, existing_candidates, start_round, max_rounds,
        )


