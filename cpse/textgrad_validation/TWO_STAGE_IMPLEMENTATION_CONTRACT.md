# two_stage_alternating_schema_description 实现契约

## 目标
在已完成的 `two_stage.py`（两阶段编排核心逻辑）基础上，把 `two_stage_alternating_schema_description` mode 完整接进 runner + optimizer，使其可 smoke 运行。3 个 TextGrad 变量：evidence prompt + resolve prompt + schema-description prompt。

## 已完成（不要重做）
- `test/textgrad_validation/two_stage.py`：`extract_two_stage`、`merge_two_stage`、`stage1_intermediate_schema`、`stage2_batch_schema`、`_slice_manifest`、`TwoStageError`。10 个单测全过。
- `test/textgrad_validation/prompts/`：`evidence_system.txt`（固定）、`evidence_initial.txt`（可优化初始）、`resolve_system.txt`（固定）、`resolve_initial.txt`（可优化初始）已写好。
- `test/textgrad_validation/config.py`：mode 已注册（line 62 校验、line 112-115 四个 prompt 路径）。ExperimentConfig 应已有 `evidence_initial_prompt_path/evidence_system_prompt_path/resolve_initial_prompt_path/resolve_system_prompt_path` 字段——确认存在，缺则补。

## 需要实现

### 1. optimizer.py：新增 TwoStageOptimizer 类（3 变量）
照 `AlternatingSchemaDescriptionOptimizer` 的模式，但 3 个变量：schema_var + evidence_var + resolve_var。

- `__init__(engine, propose_patch, evaluate, base_schema_dsl, schema_constraints, evidence_constraints, resolve_constraints)`：evidence/resolve 用 extraction constraints（参考 AlternatingSchemaDescriptionOptimizer 的 `_extraction_constraints` 默认值，含空占位符禁令那条）。
- `optimize(schema_prompt, evidence_prompt, resolve_prompt, max_rounds) -> (schema_prompt, evidence_prompt, resolve_prompt, selected_schema_dsl, candidates)`
- `_run_rounds` 照 AlternatingSchemaDescriptionOptimizer 的写，3 个变量各自 `TextualGradientDescent`、各自 backward（**共用同一份 best_results 的 `_build_per_document_losses`**）、各自 step。evidence_var 和 resolve_var 用 `_build_per_document_losses`（extraction eval prompt），schema_var 用 `_SCHEMA_PROMPT_EVAL_SYSTEM_PROMPT`。
- 候选仍是"联合候选"：3 变量一起 step 后，用候选 schema + 候选 evidence + 候选 resolve 调 `evaluate(phase_id, evidence_prompt, resolve_prompt, schema_dsl)` 评分一次。接受/回滚按 mean_score（无门槛接受，仅非法/缺分回滚，和 alternating 一致）。
- 用 `AlternatingCandidate` dataclass（已存在），candidate_id 用 `round-NNN/joint`。
- `_summarize` / `_scores` / `_mean_score` / `_build_per_document_losses` / `_format_score_breakdown` 复用 optimizer.py 现有函数。

### 2. runner.py：two_stage 分支
在以下分支点加 `elif self.config.optimization_mode == "two_stage_alternating_schema_description":`：

- **line 246 `_run` 主入口**：two_stage 走一个新 `_run_two_stage`（可大量复用 `_run_alternating` 的 resume/recompute/freeze 逻辑，但调 TwoStageOptimizer + 两阶段 evaluate）。**最小实现**：先写一个不完整复刻 resume 的简化版（只支持新 run，resume 报"two_stage resume 暂不支持，用新 run-id"），后续再补 resume——这样能先 smoke。**（2026-08-18 已补全 resume：`_run_two_stage` 现支持与 alternating 相同的续跑/重算语义，新增 `_load_two_stage_checkpoint`/`_load_best_two_stage_results`/`_recompute_two_stage_selected`，并新增 `TwoStageOptimizer.optimize_resume`；`_evaluate_two_stage` 改为按文档并行。）**
- **line 400 报告入口**：two_stage 走 `finish_alternating_report`（复用，产物结构一致）。
- **line 688 baseline schema**：`if schema_dsl is None and mode in ("single", "two_stage_alternating_schema_description")` → 用 reinforce base DSL（和 single 一样对齐 baseline）。
- **manifest line 1743**：two_stage 也要写 optimization_mode key。

### 3. runner.py：两阶段 evaluate
新增 `_evaluate_two_stage(run_dir, phase_id, evidence_prompt, resolve_prompt, records, schema_dsl)`：
- 对每个 record 调 `extract_two_stage(client=self.extractor._client 或 ResponsesPdfClient, pdf_path, root_schema_dsl=schema_dsl, root_json_schema=dsl_to_json_schema(schema_dsl), evidence_system_prompt, evidence_prompt, resolve_system_prompt, resolve_prompt)`。
- 写产物：`training/<phase>/documents/<doc>/stage1/`、`stage2-batch-K/`、合并的 `extraction.response.json` + `prediction.json` + `validation.json` + `judge.result.json`。
- 两阶段任一阶段失败（TwoStageError）→ 该文档 result=None（和 alternating 的 extraction failure 一致处理）。
- **校验失败拒绝候选**：和 alternating 一致（validate 后 is_valid False 也给 judge 打分，记 warn）——这条是 alternating 现状；用户要的"任一训练文档校验失败候选直接拒绝"是更严的约束，**先按 alternating 现状（不拒绝，记 warn）实现**，拒绝逻辑作为后续增强（避免一开始就太严导致跑不起来）。
- judge 用现有 self.judge.judge（对合并后的 prediction 评分）。

### 4. runner.py：_make_two_stage_optimizer
照 `_make_alternating_optimizer` 写 `_make_two_stage_optimizer`：
- propose_patch 同 alternating（schema-desc patch 不变）。
- evaluate_state 调 `_evaluate_two_stage`（而不是 evaluate_prompt）。
- 传 evidence_constraints/resolve_constraints（用 extraction constraints 默认）。

### 5. cache key
两阶段 extraction cache key 加 `"stage": "two_stage"` 标记（在 `_extraction_cache_inputs` 或专门给 two_stage 的 cache inputs），避免和 alternating 的单次 cache 混。judge cache key 不变（对合并后 prediction 评分）。

### 6. 测试
- optimizer TwoStageOptimizer 的单测：3 变量各自 step、联合候选、回滚。用 FakeVariable/FakeLoss/FakeOptimizer 模式（参考 test_textgrad_validation_optimizer.py 现有 fake）。
- runner two_stage evaluate 的单测（用 fake client，构造 stage1 返回 manifest + stage2 返回 batches，验证合并 + 产物写入）。

## 关键约束（用户明确要求）
1. identity_manifest 确定性切分（`_slice_manifest` 已实现），按原始出现顺序，不重叠。
2. 每个 resolve 输出必须声明 covered_identities，只覆盖本批；合并检查恰好一次（`merge_two_stage` 已实现，漏/重/凭空 raise TwoStageError）。
3. 共享工艺可被多身份引用/展开，不允许凭空补充（merge 阶段凭空聚合物 raise）。
4. 候选 schema DSL 同时提供给阶段1、阶段2（`extract_two_stage` 已传 root_schema_dsl 给两阶段）。
5. 合并后完整 schema validation + judge（`extract_two_stage` 已做 fill_missing_required + validate_prediction + 返回 PredictionArtifact）。
6. 3 变量各自 backward，共用同一份"合并后整体分数 + score_breakdown + judge feedback"（`_build_per_document_losses` 已含 SCORE_BREAKDOWN 注入）。

## 不要碰
- alternating / single 的现有逻辑（只在分支点加 elif，不改原路径）。
- two_stage.py 已完成的核心逻辑（除非单测发现 bug 才修）。
- 4 个 prompt 文件内容（已定稿）。

## 验证
- `python -m pytest -q test/tests` 全绿（含新 two_stage optimizer/runner 测试）。
- `python -m test.textgrad_validation --config test/textgrad_validation/config_two_stage.yaml --run-id smoke-two-stage-01 --smoke` 能跑（需先建 config_two_stage.yaml，照 config_final.yaml 改 optimization_mode + 引用 4 个 prompt；smoke 用 1 篇训练 + 1 轮）。

## 实现顺序
1. config_two_stage.yaml（照 config_final.yaml，optimization_mode 改 two_stage，确认 4 prompt 路径被读到）。
2. optimizer.py TwoStageOptimizer 类 + 单测。
3. runner.py 分支（_run_two_stage 简化版 + _evaluate_two_stage + _make_two_stage_optimizer）+ 单测。
4. smoke 跑通。
5. README 加 two_stage mode 文档。

每步做完跑 `pytest -q test/tests` 确认全绿再下一步。
