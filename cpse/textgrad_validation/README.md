# `textgrad_validation` 实验包说明

本包实现 `test/` 下的可复现 TextGrad 验证闭环。它把 PDF 抽取、Gold 对比、提示词/description 优化、盲测和报告放在同一个可审计运行目录中。

顶层实验说明和快速命令见 [../README.md](../README.md)。本文件说明实现行为、配置和产物含义。

## 1. 运行闭环

```text
PDF + Gold JSON + schema DSL
        │
        ├─ 数据发现与固定划分（3 篇训练，其余盲测）
        ├─ baseline 训练评估
        ├─ TextGrad 训练：候选生成 → 训练裁判 → 接受/回滚
        ├─ 冻结最佳候选
        ├─ baseline / optimized 配对盲测
        ├─ 可选 Gold PDF 审计
        └─ CSV、JSON、Markdown 与确定性差异分析
```

优化在训练集上完成；实验结论只取盲测中的配对差值。训练均分提高不等于泛化提高。

## 2. 两种抽取模式

### 2.1 单阶段：`alternating_schema_description`

```text
PDF + schema + extraction prompt → 完整 JSON → validate → judge
```

每轮把 schema-description 补丁提示词与抽取提示词作为一个联合候选：两者从同一轮已接受状态的反馈独立更新，随后只对候选组合评估一次。候选必须满足运行策略的有效性条件，才会成为下一轮的当前状态。

单阶段的抽取器会保存原始输出。针对明确的解析或缺失 required 字段问题，可执行受限修复；修复不会覆盖原始响应。其他 schema 违规仍会以原始/修复后的校验状态写入产物。

### 2.2 两阶段：`two_stage_alternating_schema_description`

```text
PDF
 ├─ stage 1：metadata_evidence + identity_manifest
 └─ stage 2：每批 identity → covered_identities + 聚合物 records
             ↓
          合并文献信息、聚合物和工艺/性质
             ↓
          最终 validate → judge
```

它面向单阶段容易折叠实体的高样品数论文。阶段 1 建立可追踪的 identity 清单；阶段 2 在较小的实体范围中完成 JSON 解析；合并器恢复完整文档。

两阶段联合优化 3 个变量：schema-description、evidence/index prompt 和 resolve prompt。三者与候选 schema 一起构成一个原子候选。

### 2.3 可选 Stage 1.5：`two_stage_coverage_plan_schema_description`

这是与既有 `two_stage_alternating_schema_description` 并列的**新优化模式**，不会修改旧模式、旧缓存或既有结果。它在 stage 1 的 identity manifest 和 stage 2 resolve 之间增加一次 PDF-only 的固定 coverage-plan 调用：

```text
identity_manifest → coverage_plan（中间产物）→ 按 batch 过滤后传给 stage 2
```

coverage plan 的每项只包含 `manifest_id`、`identity_key`、相关字段组（`性质` / `工艺流程` / `表征`）、可选的证据锚点、显式共享关系和“测试已报告但没有数值”的标记。它**不是最终 JSON schema 的字段**，不进入 prediction、Gold Judge 或下游；其目标是让 stage 2 在保持完整 PDF 可见的前提下，显式知道每个实体应覆盖哪些信息组与共享关系。

第一版的 `coverage_plan_system.txt` 是固定协议，仍只联合优化原来的 3 个变量：schema-description、evidence/index prompt、resolve prompt。因此可把它与普通 two-stage 做干净的“中间接口”消融，而不是把收益混同为多了第四个可学习 prompt。规划器输出无效、无法解析、或与 manifest 路由不一致时，只记录 `coverage_plan` warning 并以空 plan 继续 resolve；不会修复、不会臆造，也不会让整篇文献失败。

使用新配置：

```bash
python -m test.textgrad_validation --config test/textgrad_validation/config_two_stage_coverage_plan.yaml --run-id v2-goldnorm-focus-sol-two-stage-coverage-plan-01
```
### 2.3 两阶段的固定协议

- `BATCH_SIZE = 5` 是 `two_stage.py` 的代码常量，不是配置参数。
- stage 1 的 `metadata_evidence` 负责文献信息；stage 2 batch 0 不再重复输出文献信息。
- 每个 stage 2 batch 只可返回 `covered_identities` 与 `聚合物`；请求中另带本地 `manifest_id`，仅用于区分同名或同裸 identity 的不同条目，绝不写入最终 JSON。
- stage 1 完成后，独立 stage 2 batch 可并行。`max_parallel_calls` 是整个两阶段评估的全局 API 在途预算：stage 1、所有文档的 stage 2 batch 与 judge 共用它，避免外层文档并行和内层 batch 并行相乘而压垮网关。
- `covered_identities`、请求的 identity 与聚合物记录 identity 的一致性会被检查并写入 warning。它们是抽取质量信号，不再因漏覆盖、重复或跨批混入而丢弃整篇可解析 prediction。相同 identity key 的 stage-1 重复 stub 会稳定去重；`位置索引`只是证据溯源信息，即使不同也不构成科学身份冲突。若名称、样本形态或结构特征等身份语义字段确实冲突，则保留为不同 manifest 条目，由本地 `manifest_id` 路由。

## 3. 校验、warning 与硬失败

模型输出的原始文本始终保存。严格 JSON Schema 校验结果写入 `validation.json`；错误不会被悄悄删除。

| 情况 | 单阶段 | 两阶段 |
| --- | --- | --- |
| 原始文本无法解析为 JSON | 失败；保存原始文本和错误 | 对应 stage 硬失败；保存失败诊断 |
| 可解析但不符合 schema | 记录校验错误；必要时走受限修复路径 | 记录 `stage_validation_warnings`，继续合并和裁判，不自动修复 |
| stage 1 缺对象 metadata 或有效 identity 清单 | 不适用 | 硬失败 |
| stage 2 缺对象、`covered_identities` 或 `聚合物`列表 | 不适用 | 硬失败 |
| identity 覆盖不一致，或 stage-1 可恢复重复 stub | 不适用 | 写入 `stage1_manifest_warnings` / `stage_alignment_warnings` 后继续；最终由 judge 评价漏抽、重复或错误归属 |
| 最终 JSON 仍不符合总 schema | 记录最终校验结果 | 同左；若可解析，仍送裁判，但评分标为不完全可信 |

这一策略刻意区分两类问题：可解析的实体覆盖/表示问题应保留并交给 Gold 裁判；不可解析或缺少最小阶段结构才停止。两阶段不会为了“通过校验”改写 stage 输出，也不会用 Gold 补造身份标识。

## 4. Gold 裁判与 PDF

Gold 是目标内容和表示的主要依据。`evaluation.include_pdf` 控制裁判是否额外接收 PDF：

- `false`：只比较 schema、prediction 与 Gold，成本最低。
- `true`：PDF 只用于核验冲突、Gold 外内容、疑似无依据内容及疑似 Gold 问题；PDF 中 Gold 未收录的内容不会变成额外必答项。

含 PDF 的主 Judge 使用语义评分协议：不因 schema、字段名、JSON 层级、分类编码、ID、空值形式或信息位置本身扣分；但信息必须已抽取，并绑定到正确样品、条目与关键条件。Gold 按样品/条件/溶剂区分的条目若被合并或压缩而丢失粒度或条件关联，仍按部分漏抽扣分。等价单位和等价表达不扣分。

Judge 的输出至少包含 `score` 和 `optimization_feedback`；可选 `path_errors` 用于人工定位。TextGrad 接收来自训练样本的具体错误模式和泛化规则，但优化约束要求候选提示词不得记忆论文特定名称、编号、数值或句子。

可解析但 schema-invalid 的 prediction 仍会给 judge，以避免“格式失败等于没有信息”的误判；报告中应将它与两臂均通过校验的可信配对分开解读。

`gold_audit` 是完全独立的后置流程：它读取 PDF、Gold 和 schema，检查 Gold 的原文支持度。审计不会修改 Gold、评分、提示词或候选选择。

## 5. 服务端结构化输出与协议

`responses_client.py` 支持三个协议。

| 协议 | 配置值 | PDF 传递方式 | 两阶段 schema 护栏 |
| --- | --- | --- | --- |
| Responses | `responses` | Base64 `input_file` | 通过 `text.format: json_schema` 传入每个阶段的动态 schema；使用 `strict: false` 兼容 DSL 的可选字段。|
| Gemini | `gemini_generate_content` | `inlineData` | 通过 `generationConfig.responseSchema` 传入转换后的 schema。stage 1 通常可用；复杂 stage 2 schema 首次被网关拒绝时，记录 `gemini-schema-fallback` 并仅重发一次无 schema 的 JSON-mode 请求；该客户端实例会记忆 stage 2 不支持，后续 batch 直接 JSON mode。|
| Chat Completions 图片 | `chat_completions_pdf_images` | PDF 页面渲染为 PNG | 无原生 PDF file 协议；适合兼容性回退，需注意大 PDF 可能触发 413。|

本地 JSON Schema 校验始终是最终判据；原生 schema 只是第一层输出护栏。Gemini 的 stage 2 fallback 是当前已知兼容性限制：首次 400 后，能力记忆只关闭本次运行余下 stage 2 的 `responseSchema`，不会影响 stage 1 或下一次新建的运行客户端；这不是网络重试失效。

## 6. 配置要点

配置文件位于本目录。常用组合如下：

| 文件 | 模式/模型 | 说明 |
| --- | --- | --- |
| `config_prompt_only.yaml` | prompt-only / Sol | 与正式配置对齐，仅优化 extraction prompt，schema description 冻结 |
| `config_opro.yaml` | OPRO prompt-only / Sol | 外部黑盒优化基线；根据历史 prompt 与客观分数逐轮提出 extraction prompt，不使用 TextGrad |
| `config_gepa.yaml` | GEPA prompt-only / Sol | 官方 `gepa` 适配器；使用逐篇执行轨迹和富反馈反思优化 extraction prompt |
| `config_mipro_v2_instruction_only.yaml` | MIPROv2 instruction-only / Sol | 官方 DSPy MIPROv2；所有 demonstrations 设为0，只搜索 instruction |
| `config_mipro_v2.yaml` | MIPROv2 instruction+labeled-demo / Sol | 从训练PDF全文→Gold中搜索最多1个labeled demonstration；不使用bootstrap demo |
| `config_schema_free_direct.yaml` | 无 Schema 直抽 / Sol | 抽取器不接收 schema；Judge 与主方法完全一致，接收 schema、Gold 与 PDF 并按主 PDF 语义协议评分 |
| `config_few_shot_direct.yaml` | 固定 3-shot 直抽 / Sol | 固定三篇训练 PDF 全文→Gold 作为 ICL 示例；不运行优化器 |
| `config_judge_terra.yaml` | Judge-only / Terra | 仅重评已有冻结 prediction，不用于抽取训练 |
| `config_judge_gemini.yaml` | Judge-only / Gemini | 仅重评已有冻结 prediction，不用于抽取训练 |
| `config_description_only.yaml` | description-only / Sol | 与 `config_final.yaml` 对齐，仅优化 schema description，抽取 prompt 冻结 |
| `config_final.yaml` | 单阶段 / Sol | 含 PDF 的 Sol 单阶段示例 |
| `config.yaml` | 单阶段 / Terra | 当前 Terra 单阶段示例 |
| `config_two_stage_terra.yaml` | 两阶段 / Terra | Responses 原生 schema 护栏版本 |
| `config_two_stage_gemini.yaml` | 两阶段 / Gemini | Gemini `responseSchema` + 受控 fallback 版本 |

关键字段：

```yaml
optimization_mode: alternating_schema_description
# 或 opro_prompt_only / gepa_prompt_only / mipro_v2_instruction_only /
# description_only / two_stage_alternating_schema_description / two_stage_coverage_plan_schema_description

evaluation:
  include_pdf: true
  include_error_locations: false

cache:
  enabled: true

max_iterations: 5
max_parallel_calls: 6
train_count: 3
train_ids: ["..."]
request_format_version: responses-pdf-v2-native-schema
```

`request_format_version` 是请求格式/缓存隔离标识。改变 API 协议、schema 传递形式、提示词、Gold、总 schema、训练集或模型配置后，应使用新 run id；不能将不同条件的结果混在同一 manifest 中。

并发仅作用于彼此独立的文档调用。增大 `max_parallel_calls` 会缩短运行时间，也会提高网关拥塞、连接失败和限流风险。

## 7. 重试与恢复

客户端会对 429、5xx、超时和连接错误重试，间隔按配置策略增长。400、401、403、404、413、422 等确定性请求错误默认不重试。

例外是 Gemini 的 `responseSchema` 400：系统识别为 schema 兼容性问题后，仅用 JSON mode 再请求一次。这不是无限重试。

```bash
# 正常运行 / 默认允许恢复
python -m test.textgrad_validation --config <config.yaml> --run-id <run-id>

# 要求已有且匹配的运行目录
python -m test.textgrad_validation --config <config.yaml> --run-id <run-id> --resume require

# 保留 accepted round-000..N，删除后续候选和 optimized 盲测，再从 N+1 继续
# 适用于 alternating_schema_description 与 two_stage_alternating_schema_description
python -m test.textgrad_validation --config <config.yaml> --run-id <run-id> --resume-from-round N

# 只重跑失败的主流程文档；保留成功产物和冻结候选
python -m test.textgrad_validation --config <config.yaml> --run-id <run-id> --retry-failed-primary

# 只从既有产物重建报告
python -m test.textgrad_validation --config <config.yaml> --run-id <run-id> --report-only

# 新建独立 run，一次性运行完整盲测 direct baseline；不训练、不生成 optimized 臂
python -m test.textgrad_validation --config <nocache-config.yaml> --run-id <baseline-run-id> --blind-baseline-only

# 可选的基于 Judge 分数的自适应重抽：低于阈值时最多抽取3次
python -m test.textgrad_validation --config <nocache-config.yaml> --run-id <adaptive-run-id> --blind-baseline-only --retry-below-score 60

# 用当前配置的 Judge 重评既有两臂；LABEL 隔离不同 Judge/提示词设置
python -m test.textgrad_validation --config <judge-config.yaml> --run-id <run-id> --judge-only <LABEL>

# 同一 LABEL 默认复用已完成项；确需覆盖时显式指定
python -m test.textgrad_validation --config <judge-config.yaml> --run-id <run-id> --judge-only <LABEL> --judge-force

# 可选：只重评指定 arm 和指定文档；--judge-doc 可重复
python -m test.textgrad_validation --config <judge-config.yaml> --run-id <run-id> --judge-only <LABEL> --judge-arm optimized --judge-doc <DOC_ID_1> --judge-doc <DOC_ID_2> --judge-force

# 从冻结 prediction 计算不调用模型的严格辅助指标
python -m test.textgrad_validation --config <config.yaml> --run-id <run-id> --metrics-only

# 从既有主评分、Judge-only和严格指标生成完全离线的配对统计报告
python -m test.textgrad_validation --config <config.yaml> --run-id <run-id> --statistics-only
```

```bash
# 离线汇总已有抽取产物中的 token 使用量；不调用 API，也不修改 run。
python -m test.textgrad_validation --config <config.yaml> --run-id <run-id> --usage-only

# 生成公开 artifact release；不调用 API，且不复制原始 PDF
python -m test.textgrad_validation --config <config.yaml> --run-id <run-id> --artifact-release

# 生成完整的脱敏实验代码包；不调用 API，也不复制数据、结果或缓存
python -m test.textgrad_validation --config <config.yaml> --code-release
```

```bash
# 离线检查冻结 prompts / schema 是否精确复制训练 Gold 中的高风险事实。
python -m test.textgrad_validation --config <config.yaml> --run-id <run-id> --leakage-audit
```

### 固定盲测集的1/2/3-shot消融

`train_ids` 始终定义一个有序的固定3篇训练池，`train_count`（或命令行
`--train-count`）决定实际启用其前几篇。`train_count: 1/2` 时，训练池中未启用的
文档既不参与训练，也不会被放回盲测，因此1/2/3-shot始终共享原来的17篇盲测，
可以直接比较配对增量。未提供显式 `train_ids` 时只允许3-shot自动划分。

```bash
python -m test.textgrad_validation --config test/textgrad_validation/config_two_stage2.yaml --train-count 1 --run-id v2-goldnorm-focus-sol-two-stage-train1-01
python -m test.textgrad_validation --config test/textgrad_validation/config_two_stage2.yaml --train-count 2 --run-id v2-goldnorm-focus-sol-two-stage-train2-01
```

先用 `--preflight` 可免费核验三种设置的盲测ID完全一致。run manifest、配置快照、
缓存指纹和 output.log 中的原始命令都会记录实际 `train_count`；不同shot数必须使用
不同 run-id。

`description_only`、单阶段交替优化与两阶段训练都会在 baseline `round-000` 结束后、以及每个候选 round 已评估并确定接受/回滚后立即持久化 checkpoint。因而意外终止后可用相同 `run-id` 加 `--resume require`：恢复逻辑读取最后一个完整 round 的候选轨迹和最佳状态，从下一 round 继续。正在执行、尚未完成的那一轮不被视为可复用结果；其局部原始产物可供诊断，但不会进入 checkpoint。Ctrl+C 会设定 run 级取消事件并取消尚未开始的任务；两阶段的 stage-2 内层使用 daemon 线程池、调用 `shutdown(wait=False, cancel_futures=True)`，因此不会等待被网络层卡住的 worker。已发出的 HTTP 请求不能由 Python 强制杀死，但不再阻塞 CLI 退出。

### 7.1 “继续训练状态”与“冻结最佳候选”必须分开

TextGrad 在训练中将最后一个有效候选作为下一轮更新的起点；它可能低于更早轮的训练均分。该状态只服务于继续优化，**不**决定盲测。单阶段交替优化与两阶段都会在训练结束、恢复、缩短轮数或补跑后，从完整候选轨迹中重新选择训练均分最高的有效 joint candidate，并读取该候选目录内成套的 schema-description、抽取提示词（两阶段还包括 evidence / resolve pair）和 schema。

冻结文件、盲测 gate 与 optimized 臂必须引用同一最高分候选。若恢复时重新选择的候选与既有 optimized 臂的冻结状态不同，系统只清除 optimized 文档产物并重跑该臂；baseline 不依赖训练候选，因而保留。这样既避免无谓重跑，也避免将不同训练轮的 optimized 分数混入同一报告。

### 7.2 失败重试与缓存边界

`--retry-failed-primary` 仅用于主流程已完成后的失败文档恢复：它保留冻结的候选、成功的 baseline / optimized 文档和配对设计，只重跑失败臂，并将重试前后 retry / proxy 策略写入 `meta/events.jsonl`。普通单阶段复用冻结抽取提示词；`alternating_schema_description` 分别复用冻结的 base schema 与 selected schema；两阶段还复用冻结的 evidence / resolve 提示词。它不能用于改变提示词、schema、模型或训练轨迹。除 checkpoint 中的非 `valid` 状态外，必要产物缺失也会触发补跑：有效 prediction 要求 `prediction.json`、`validation.json` 和 `judge.result.json`；schema-invalid 但仍允许 judge 的 prediction 则要求原始响应、`validation.json` 和 `judge.result.json`。这种恢复以 `required_artifact_missing` 记入事件。控制台先给出重试前的全量配对统计，再逐篇显示旧分数→新分数，最后给出重试后的全量 `valid_paired`、两臂均分和 `mean_paired_delta`；评估函数自身打印的 `mean=... (k/k docs)` 只描述本轮补跑子集。

`cache.enabled: false` 时，每个抽取、裁判和审计调用都会重新请求模型，适合正式公平比较；此时 checkpoint 仍会复用已完成的训练候选和文档产物，但不会从共享 API 缓存读取结果。任何会改变输入指纹的改动都应使用新 `run-id`。

## 8. 运行产物与排障入口

```text
results/<run-id>/
├── meta/manifest.json            # 数据、划分、schema、模型与请求格式指纹
├── meta/config.snapshot.json     # 脱敏配置快照
├── meta/events.jsonl             # CLI 启动命令、重试、审计等事件
├── meta/frozen_protocol/         # run 创建时的文件型提示词原文与哈希索引
├── prompts/                      # 冻结 prompt；两阶段含 prompt pair
├── schemas/                      # 联合 schema-description 模式的基线/最终 schema
├── training/                     # 各轮候选和逐文档结果
├── blind_test/baseline/          # 基线臂
├── blind_test/optimized/         # 最优候选臂
├── judge_only/<label>/           # 可选：当前配置 Judge 对冻结 prediction 的隔离重评分
├── deterministic_metrics/        # 可选：不调用模型的严格辅助指标
├── statistics/                   # 可选：配对CI、检验、Judge一致性及指标相关性
├── analysis/                     # 既有产物的确定性差异分析
├── gold_audit/                   # 可选后置审计
├── blind_test_summary.json       # 配对主结论、失败数、可信配对统计
└── report.md                     # 人类可读报告
```

### 8.1 启动协议快照

新 run 创建 manifest 时，会立即把配置中实际引用的全部**文件型提示词**复制到
`meta/frozen_protocol/prompts/`。`index.json` 为每份文件记录逻辑角色、原始路径、run 内相对路径与
SHA-256，覆盖 extraction initial/system、Judge、Gold audit，以及按模式启用的 schema-description、
evidence、resolve、coverage-plan 和 evidence-routing 提示词。未启用模式的可选提示词不会写入；
运行时动态生成的请求 payload 和第三方库内部模板也不属于文件型协议快照。

这个快照回答“该 run 启动时究竟使用了哪些提示词”。根目录 `prompts/` 则保存训练候选最终冻结的
baseline/最优提示词，两者不可互相替代。启动快照只在 manifest 首次建立时创建，resume、report-only
和 retry 均不得覆盖；历史 run 缺少快照时也不会用当前仓库文件回填，以免制造错误的实验 provenance。

通过 CLI 启动的主实验会在 `output.log` 写入可复制的 `[run] command: python -m ...`，并以 `cli_invocation` 事件写入 `meta/events.jsonl`。该命令只用于复现审计，不参与 manifest、缓存指纹或评分。

每篇文档目录的常用文件：

| 文件 | 含义 |
| --- | --- |
| `extraction.response.txt` | 模型原始响应，绝不作为“已通过校验”的证据 |
| `prediction.json` | 通过最终 schema 校验时的 prediction |
| `validation.json` | 最终解析/Schema 校验状态和错误 |
| `extraction.metadata.json` | 调用元数据；两阶段包含原始/去重后 manifest 数量、`stage1_manifest_warnings`、`stage_validation_warnings` 与 `stage_alignment_warnings`（如有） |
| `extraction.failure.json` | 硬失败的阶段、尽可能保留的原始输出、validation errors 和诊断上下文 |
| `judge.result.json` | Gold 裁判分数、反馈及可选字段定位 |

排障顺序：先看 `extraction.failure.json`；若没有硬失败但分数不可信，检查 `validation.json` 与 `stage_validation_warnings`；再对照原始响应、`judge.result.json` 和 Gold。不要只凭控制台中的一个总分判断优化有效性。

### 8.1 Judge-only：多裁判敏感性分析而不重抽取

Judge-only 只读取既有 `blind_test/baseline|optimized/documents/*`，按各 arm 冻结 schema 重建 prediction，并用当前 config 的 Judge 重新评分。它不会调用 extractor，不会改变主 `judge.result.json`、checkpoint、冻结候选或 `report.md`。运行时终端逐篇显示 arm、进度、文档 ID、score 与 `(reuse)` 标记；缺 Gold、缺 prediction 或缺 arm 目录会明确显示跳过原因，结束时输出 paired 均值、delta 与 wins/ties/losses。同样的信息追加到独立的 `judge_only/<label>/output.log`。`summary.json` 记录 `judge_model`、Judge prompt SHA-256、`include_pdf`、`include_error_locations`、两臂均分和配对统计；逐篇结果保存在 `judge_only/<label>/documents/<doc>/<arm>.judge.json`。若某一篇 Judge 输出不符合结果契约、无法解析或调用失败，程序不会终止整个批次：该篇的错误、尝试次数和 arm 写入 `<arm>.judge.failure.json`，其他文档照常完成；不带 `--judge-force` 再运行同一 label 时，成功结果复用、failure/missing 项自动补跑。`--judge-force` 会在单篇重评开始前先移除该篇旧结果，因此强制重评中途失败也不会悄悄复用历史分数。

`--judge-arm baseline|optimized|both` 可限制重评臂，默认 `both`；重复使用 `--judge-doc DOCUMENT_ID` 可限制文档集合。筛选只影响当前 label 的 Judge-only 产物。单臂运行没有 paired delta，但 `summary.json` 仍报告该臂的文档数和均分。

不同 Judge 模型的推荐做法是每个设置使用不同 label，例如 `sol-pdf-v1`、`terra-nopdf-v1`。随后比较各 Judge 对逐篇 delta 的方向一致率、均值、排序相关与结论符号，而不是把多 Judge 分数简单平均成一个“更真”的分数。

### 8.2 确定性辅助指标的能力边界

`--metrics-only` 保留 v1 strict leaf-overlap 与 v2 property-tuple 指标，并新增用于论文复核的 entity-aligned、unit-normalized、schema-aware slot PRF。实体首先按 schema 中的稳定身份字段一对一对齐，身份格式不一致时依次使用名称/缩写与确定性记录相似度；性质只能在对应实体内匹配。每个实体、性质记录、数值/单位和关键测试条件分别贡献一个等权 slot，未对齐记录的全部 slot 仍计入预测或 Gold 分母。数值采用 1% 相对容差，并按固定表统一温度、频率、时间、压力、长度、体积、质量、百分比和波数；条件缺失只损失对应条件 slot，而不会把整个正确数值判错。输出包括 `aligned_entity_*`、`property_detection_*`、`value_unit_*`、`condition_slot_*` 和总 `schema_aware_slot_*` 的逐篇、macro 与 micro PRF。

这些指标故意保持透明且无需模型：显式支持的等价单位（如 `10 kHz` 与 `10000 Hz`、`25 °C` 与 `298.15 K`）可判同；未登记的英文/中文同义词及自由描述改写不会自动判同。它适合检查 Judge 波动、样品归属、覆盖变化和结果可复现性，不适合取代语义评分。它不进入 TextGrad loss、候选接受或冻结，也不输出 JSON exact match。

### 8.3 完全离线的配对统计

`--statistics-only` 读取冻结的主盲测配对分数、所有已完成的 `judge_only/<label>/documents.csv` 和确定性逐篇指标，不创建任何 API 客户端。它在 `statistics/` 写入 `summary.json`、`per_document.csv` 与 `report.md`。主裁判、每个附加 Judge，以及 entity alignment、property detection、value/unit、condition 和总 schema-aware slot F1 均报告 delta 均值、中位数、胜/平/负、固定种子的10,000次 paired bootstrap 95% CI，以及双侧符号翻转 permutation p-value；非零配对不超过20个时进行全枚举，因此17篇盲测得到精确检验。多 Judge 之间报告逐篇 delta 的 Spearman 相关和正/平/负方向一致率，同时计算 Judge delta 与全部确定性指标 delta 的探索性相关。该统计不改变任何 prediction、Judge结果、候选选择或主报告。

新运行在 `meta/api_calls.jsonl` 逐 attempt 落盘 extractor/Judge/optimizer/schema-patch/audit 的状态、模型、协议、usage、请求耗时与 retry backoff。`--usage-only` 优先汇总该账本，分别给出 logical calls、physical attempts、retry attempts、unknown-usage failures、token、累计服务耗时、等待时间及公开价估算；并按 operation 分组。并行请求的累计服务耗时不等于 run wall-clock。老 run 没有账本时仍使用 `extraction.metadata.json` 的兼容统计，缺失调用不能重建。统计命令本身完全离线，不消费 token。

`--artifact-release` 输出到 `test/artifact_release/<run-id>/`。打包器采用公开白名单，复制 Gold JSON、预测、冻结协议和分析产物；不复制 PDF、`.env` 或 `output.log`，并从公开 config snapshot 递归删除 `base_url`、`proxy`、`api_key_env`。每次生成都会重建目录并写 `MANIFEST.json`、`SHA256SUMS.txt`，便于上传匿名 artifact 服务或录用后归档到带 DOI 的仓库。

`--code-release` 输出到 `test/code_release/`。它复制完整实验源码、prompts、schema、脱敏 YAML 配置、测试与依赖清单，同时排除 data、results、artifact_release、缓存、PDF、`.env`、服务 URL、proxy 和密钥环境变量名。它应与某个固定 run 的 `--artifact-release` 成对发布：代码包支持重跑，artifact 包支持直接核验论文表格。

`--leakage-audit` 从该 run 的 `meta/manifest.json` 确定训练 Gold，只扫描最终 `blind_test/optimized/` 的 prompts 与 schema。它报告 DOI、带数字/连字符的标识符、长字符串和精确数值等高风险训练事实是否以**完全相同的字符串**出现在冻结工件中；同时比较初始与最终 schema 去除 `description` 后的结构是否相同，并列出 description 改动路径。输出写入 `leakage_audit/summary.json` 与 `leakage_audit/report.md`。命中项只是人工复核项，通用领域措辞不作为泄漏证据；未命中只能说明未发现精确复制，不能证明没有任何语义层面的训练事实记忆。该命令完全离线，不重跑模型。

### 8.4 公平内部基线与消融

| 基线/方法 | 实现位置 | 只改变什么 |
| --- | --- | --- |
| Direct zero-shot | 单阶段 run 的 baseline 臂 | 什么都不优化 |
| Prompt-only TextGrad | `optimization_mode: single` | 抽取 prompt |
| OPRO Prompt-only | `optimization_mode: opro_prompt_only` / `config_opro.yaml` | 抽取 prompt；历史 prompt 与客观分数驱动的黑盒候选生成 |
| GEPA Prompt-only | `optimization_mode: gepa_prompt_only` / `config_gepa.yaml` | 抽取 prompt；官方GEPA反思变异与instance Pareto选择 |
| MIPROv2 instruction-only | `optimization_mode: mipro_v2_instruction_only` | 抽取instruction；bootstrapped/labeled demos均为0 |
| MIPROv2 instruction+demo | `optimization_mode: mipro_v2` | instruction + 最多1个真实PDF文本→Gold labeled demo |
| 无 Schema 直抽 | `optimization_mode: schema_free_direct` | 无 contract、无示例、无优化的自由 JSON 抽取 |
| 固定 3-shot 直抽 | `optimization_mode: few_shot_direct` | contract + 三个固定 labeled examples，无优化 |
| Description-only TextGrad | `optimization_mode: description_only` / `config_description_only.yaml` | schema description；抽取 prompt 固定 |
| Joint single-stage | `alternating_schema_description` | 抽取 prompt + schema description |
| Fixed two-stage | two-stage run 的 baseline 臂 | 仅使用固定两阶段架构 |
| Optimized two-stage | 同一 run 的 optimized 臂 | evidence + resolve + schema-description 三个提示变量 |

固定 two-stage 已是 two-stage run 内的 baseline，不额外增加重复模式。正式消融必须保持训练 3 篇、盲测 17 篇、模型、Gold、Judge、PDF 开关、max iterations、cache 策略一致；优先比较同一 run 的配对臂，跨 run 时核对 manifest。

### 8.5 Two-stage execution-only 消融

`--execution-ablation` 不训练新候选，也不改变冻结的 schema 结构、description、evidence prompt 或 resolve prompt。它从既有 two-stage run 中读取训练均分最高的完整冻结状态，并在固定盲测集上比较：

- `manifest_unbounded`：stage 1 生成一次有序 identity manifest，stage 2 将整份 manifest 作为一个 batch 解析；
- `manifest_bounded_5`：逐篇复用上一个 arm **完全相同的 stage-1 manifest**，仅将其切成至多 5 个 identity 的 batch 再解析。

共享 manifest 是公平性的关键：否则两次随机的实体发现会把 stage-1 方差混入 execution topology 的比较。batch size 与复用 manifest 的哈希均进入 extraction cache 指纹，因此不会误命中普通 two-stage 盲测缓存。两臂的逐篇 prediction/Judge 产物写入 `execution_ablation/<arm>/documents/`，总体配对结果写入 `execution_ablation/summary.json`，完成事件写入 `meta/events.jsonl`。

```bash
python -m test.textgrad_validation --config test/textgrad_validation/config_two_stage2.yaml --run-id v2-goldnorm-focus-sol-two-stage-03 --execution-ablation
```

配置必须与源 run 的 immutable inputs 一致。该命令不产生 TextGrad/schema-patch 调用，但会执行新的 stage-2 extraction 与 Judge；unbounded arm 还会为每篇文档执行一次 stage 1，以产生供两臂共享的 manifest。

该专用模式支持本地断点补跑，即使配置为 `cache.enabled: false`，重复执行时也会复用 `execution_ablation/` 内已成功且可读取的 Judge 结果，只重跑失败或缺失文档；这不会启用跨 run 共享缓存。若 stage 2 的 `聚合物` 数组含字符串等非对象元素，系统原样保留该元素、记录 warning 并交给最终 schema validation/Judge，而不会因本地身份诊断调用 `.get()` 崩溃或偷偷修复模型输出。

### 8.6 固定 manifest 的 Stage-2 replay

`--stage2-replay LABEL` 从同一 run 的 `execution_ablation/manifest_unbounded/documents/*/extraction.metadata.json` 读取已保存的 `stage1_output`，跳过 Stage 1，只重新执行 bounded-5 Stage 2 和 Judge。重复使用 `--stage2-doc DOCUMENT_ID` 可仅选择少量盲测文档。该模式强制绕过共享 extraction/Judge cache，避免把历史响应误当成新调用；同一 label 下已完成的本地产物仍可用于中断续跑。

```bash
python -m test.textgrad_validation --config test/textgrad_validation/config_two_stage2.yaml --run-id v2-goldnorm-focus-sol-two-stage-03 --stage2-replay current-manifest-r1 --stage2-doc 0a47c679ec028585da44a38eb89ad3ff --stage2-doc 4d05e51f5798c06e05e56db328473f23 --stage2-doc 4edbc5672c679244aaab6aa121d15107
```

结果隔离写入 `execution_ablation/stage2_replay/<LABEL>/`。`summary.json` 明确记录 `manifest_source=execution_ablation/manifest_unbounded` 和 `historical_manifest_replay=false`：旧主运行没有保存完整历史 Stage-1 manifest，因此该实验检验的是“固定当前 manifest 后的 Stage-2 重复性”，不能表述为对 8 月历史 manifest 的精确重放。

OPRO 与 Prompt-only TextGrad 共享相同可变空间：schema 与 description 固定，只优化 extraction prompt。每轮 OPRO 元提示包含此前全部候选的提示词、训练均分和逐篇客观分数，不读取裁判的自然语言反馈，并只生成一个新候选；最终仍按三篇训练文献的最高均分冻结，再进入同一 17 篇配对盲测。这样既保持 OPRO 的分数轨迹搜索特征，也避免把它混成 GEPA/TextGrad 式反思优化；比较的是候选更新算法，而不是 schema trick 或两阶段架构。

论文中的重复运行要求每次生成独立的 extraction、Judge 与 optimizer 轨迹。因此 `config_opro.yaml`、`config_gepa.yaml`、`config_mipro_v2.yaml` 和 `config_mipro_v2_instruction_only.yaml` 默认关闭共享缓存；仅更换 `run-id` 或使用 `--resume never` 并不能在缓存开启时保证统计独立性。固定 3-shot 与 fresh direct 配置同样关闭缓存。

### 8.5 无 Schema 与固定 3-shot 直接基线

```bash
python -m test.textgrad_validation --config test/textgrad_validation/config_schema_free_direct.yaml --run-id v2-goldnorm-focus-sol-schema-free-direct-01 --blind-baseline-only
python -m test.textgrad_validation --config test/textgrad_validation/config_few_shot_direct.yaml --run-id v2-goldnorm-focus-sol-fixed-3shot-direct-01 --blind-baseline-only
```

两个模式都只能与 `--blind-baseline-only` 一起运行，不创建 training 或 optimized 目录。输出包括逐篇 `prediction.json`、`judge.result.json`、`blind_baseline_documents.json` 和 `blind_baseline_summary.json`。

`--retry-below-score SCORE`是默认关闭的自适应重抽协议：第一次达到阈值即固定；否则执行第二次，第二次达到阈值则采用第二次；仍未达到阈值时执行第三次，第三次达到阈值则采用第三次，否则选取三次分数的中位数对应产物。该参数既可与`--blind-baseline-only`组合，也可与`--skip-blind-baseline`组合以只对optimized盲测臂应用规则。所有尝试保存在对应的`adaptive_attempts/`目录，最终选中产物复制到标准baseline或optimized路径；逐篇摘要或checkpoint记录尝试分数和选中轮次。由于该模式使用Gold/PDF Judge分数控制重抽与选择，应与固定次数、非自适应的主评估分开报告。已有attempts可用`--reselect-score-retries SCORE`离线按当前规则重建最终产物，不会发起API调用。

- `schema_free_direct`：抽取 payload 只有 `extraction_prompt`，没有 schema 或 demonstrations；模型可生成自由 JSON。评分端不例外：与主方法一样接收 schema、Gold 与 PDF，并按主 PDF 语义协议评分。返回后仍按目标 schema 本地校验，仅将其作为结构可消费性的辅助指标；为避免在第二次请求泄露 schema 路径，schema-free 输出不进入 schema 修复回合。它因此是可比较的“无 schema 抽取”基线；其 prediction 可单独报告 schema-valid 率与严格路径指标。

若要单独审计自由 JSON 是否因“内容看似合理”而被主语义 Judge 高估，可使用 `config_schema_free_direct_content_audit.yaml` 重评**已有** Schema-free prediction。该审计只向 Judge 提供 schema、Gold 和 prediction，不提供 PDF，避免源文献内容被误当成 prediction 已抽取内容并减少输入 token。Judge 返回结构化缺失计数；本地程序据此确定性执行分数上限（主要实体/路线/数据块缺失最高 69，样品级合并压缩最高 79，至少两个独立实质问题最高 84），同时保存模型原始分数和实际应用的上限。它仍不评价字段、层级或 JSON 格式，因此不改变主方法的评分协议与历史结果。审计分数应标为 Schema-free content-coverage audit，不能替代主方法分数：

```bash
python -m test.textgrad_validation --config test/textgrad_validation/config_schema_free_direct_content_audit.yaml --run-id v2-goldnorm-focus-sol-schema-free-direct-01 --judge-only sol-schema-free-coverage-audit --judge-force
```
- `few_shot_direct`：抽取 payload 包含原始 schema、固定 extraction prompt，以及 `train_ids` 指定的三条 `{document_id, document_text, gold_json}`。`document_text` 由 `pypdf` 确定性提取，Gold 被压缩为稳定 JSON 字符串；三篇示例只来自冻结训练池，17 篇盲测 Gold 永不进入请求。该模式强制 `train_count: 3` 且三个 ID 唯一。它与主方法使用同一 schema-aware Judge，但其每次推理输入显著更长，必须单列 token/成本。

两份配置使用不同的 `request_format_version` 且默认 `cache.enabled: false`，防止与 zero-shot、TextGrad 或历史缓存混用。运行前可先执行：

```bash
python -m test.textgrad_validation --config test/textgrad_validation/config_schema_free_direct.yaml --preflight
python -m test.textgrad_validation --config test/textgrad_validation/config_few_shot_direct.yaml --preflight
```

GEPA模式调用官方 `gepa.optimize`，通过项目适配器将每篇文献的归一化分数、分项分数和Judge反馈转换为reflective dataset；搜索组件仅为`extraction_prompt`，预算用`max_metric_calls=(max_iterations+1)*3`控制。GEPA 0.1.4 要求适配器显式提供可选的 `propose_new_texts` 属性；本项目适配器实现该钩子，并通过既有 judge/optimizer 客户端把反思记录转换为一个新的 JSON prompt 候选。早期缺少此钩子的 GEPA run 会反复记录“未提出新候选”，其 optimized 臂等同 baseline，不能作为有效外部基线。MIPROv2 instruction-only调用官方`dspy.MIPROv2`，但将`max_bootstrapped_demos=0`和`max_labeled_demos=0`，避免把三篇Gold直接作为盲测推理示例。完整`mipro_v2`模式则允许最多1个labeled demo：输入是由`pypdf`确定性提取的训练PDF全文，输出是对应Gold JSON；最终冻结prompt会携带DSPy选中的demo进入17篇盲测。两种MIPRO模式均不使用bootstrapped demo，因为外部PDF执行器没有可重放的DSPy teacher trace。

可选依赖：

```bash
pip install gepa==0.1.4 dspy==3.3.1
```

建议在独立虚拟环境运行外部基线：DSPy 3.3.1的依赖链可能升级`openai`、`litellm`、`pydantic`等共享包，而本实验主体此前验证的是`openai==1.109.1`。当前代码的DSPy适配器使用自定义`BaseLM`直接调用项目Responses客户端，不依赖LiteLLM发请求，但安装器层面的版本变化仍应通过`python -m pytest -q test/tests`复核。GEPA可单独只安装`gepa==0.1.4`。DSPy 3.3.1 在 `max_bootstrapped_demos=0`、`max_labeled_demos>0` 且 few-shot candidates 大于3时会从空区间 `randint(1, 0)` 采样；full MIPROv2 适配器因此把 labeled-only 的 few-shot候选限制为其3个内置集合，同时保持 instruction candidates 与试验轮数不变，避免偷偷引入 bootstrapped demo。

完整`mipro_v2`是额外的few-shot推理范式，不应与prompt-only方法解释成同预算比较。当前三篇demo的“PDF文本字符数/Gold JSON字符数”约为`62k/216k`、`47k/47k`、`32k/44k`，即使最多只选1个，推理输入仍可能显著增加并触发网关上下文限制。论文必须单列其输入token、优化调用数、盲测推理token和失败率；若发生context-length/413错误，不应截断Gold后继续冒充同一方法。

对于已经有独立canonical baseline、只需要生成新optimizer结果的single-stage外部基线，
可在正常训练命令后添加`--skip-blind-baseline`。训练集上的`candidate-000`仍会执行，
因为它是优化搜索的起点；该选项只跳过17篇盲测文档的baseline抽取与Judge，因而输出
没有run内paired delta。例如：

```bash
python -m test.textgrad_validation --config test/textgrad_validation/config_gepa.yaml --run-id table2-gepa-s2 --resume never --skip-blind-baseline
```

Judge-only可直接运行：

```bash
python -m test.textgrad_validation --config test/textgrad_validation/config_judge_terra.yaml --run-id <existing-run-id> --judge-only terra
python -m test.textgrad_validation --config test/textgrad_validation/config_judge_gemini.yaml --run-id <existing-run-id> --judge-only gemini
```

## 9. 结果报告的最低要求

一份可用于比较的报告至少应说明：

1. baseline / optimized 的模型、schema、裁判、PDF 开关和训练集是否一致；
2. 盲测文档数、有效配对数、两臂校验失败数；
3. `mean_paired_delta` 及可信配对版本的统计；
4. wins / ties / losses 与至少一个提升、一个退步、一个失败或不可信 case；
5. 是否出现 native schema fallback、repair、重试或 resume，它们是否影响可比性。

## 10. 从 0 到 1：关键难点、设计取舍与解决方案

这一节保留当前实现背后的问题链，供复现实验、撰写方法部分和解释结果时使用。它不是按日期罗列的开发日志，而是“问题 → 风险 → 设计 → 剩余边界”的决策记录。

### 10.1 Gold 与 schema 先于优化：否则分数没有可解释性

**问题。** 早期 Gold 的同义字段、扁平单位格式、可选字段和 schema 的严格契约并不完全一致。模型即使抽到了论文中的内容，也可能因键名、值对象或 required 规则不一致而整篇 validation 失败。

**风险。** 若直接优化抽取提示词，TextGrad 学到的可能只是“迎合某个偶然格式”，而不是提升文献抽取；不同 run 的分数也不可比较。

**设计。** 先将 Gold 归一化到 schema 契约：统一可比字段名；无损转换可确定的历史单位表示；对 Gold 实际存在、但 schema 未覆盖的溯源字段和缺失值形式作适度契约扩展。保留数据备份与归一化记录。随后把 schema 作为固定输出合同，允许优化的只有 description，而不是任意改变字段结构。

**边界。** Gold/schema 改动会改变任务定义，必须新建 run id；旧结果不能与新 Gold 的结果直接比较。

### 10.2 为什么联合优化 schema description 与抽取提示词

**问题。** 仅优化抽取提示词时，模型常常知道“要抽什么”，却不知道某个仓库字段在何种论文表达下应如何映射；而任由 TextGrad 改动整个 schema 又会破坏 JSON 合同。

**设计。** `alternating_schema_description` 将可优化的 schema description 与抽取提示词分开：前者只提出 description 补丁，后者约束抽取策略；二者基于同一已接受候选的裁判反馈更新，并与候选 schema 联合评估、联合接受或回滚。结构、字段名、类型和 required 规则不允许被优化器任意改写。

**为什么不是只靠固定 prompt。** 目标字段含义来自仓库 schema，而不同论文对工艺、性能和表征的表达差异大。description 补丁提供的是可审计、可冻结的“字段语义适配”；抽取提示词提供跨字段的执行策略。两者共同作用，且盲测用于约束训练集过拟合。

### 10.3 如何让 TextGrad 的反馈可学习而不记忆训练论文

**问题。** 只给总分，优化器不知道该修哪里；只给 Gold 的具体值，又会诱导 prompt 记住训练文献中的名称、样品编号或数值。

**设计。** Judge 反馈要求同时表达：

- 当前样本的根因级差异，作为模型识别失败模式的锚点；
- 可跨论文迁移的抽取规则，作为实际更新方向；
- 低分维度与最高影响问题的优先级。

优化器接收两者，但其约束明确禁止把样品特有名称、编号、数值或原句写回 prompt。这样既避免“只有空泛原则，学不到合成/表格展开等结构性失败”，又保留盲测对记忆化的最终否决权。

**边界。** 只有 3 篇训练文献时仍存在过拟合风险；提示词异常膨胀、训练分提高而盲测下降时，应报告为过拟合信号，而不是挑选最好的一次结果。

### 10.4 为什么评分允许 PDF，但 Gold 仍是主目标

**问题。** 仅比较 prediction 与 Gold 时，Gold 外内容无法区分“PDF 有依据的额外抽取”和“无依据编造”；但让 PDF 成为主要目标又会让任务范围随裁判自由扩张。

**设计。** `include_pdf: true` 采用受限 PDF 核验：Gold 决定应抽取范围和正确目标，PDF 仅复核争议项、Gold 外新增内容、疑似幻觉和疑似 Gold 问题。PDF 支持但 Gold 未收录的内容不加分也不扣分；PDF 无法确认时仍以 Gold 为准。

**结果解释。** 因而含 PDF 与不含 PDF 的评分是不同的评估设置，应在结果表中明确标注，不应将两者的绝对分数混为同一量纲。

### 10.5 为什么校验失败不再一律终止评分

**问题。** 大型嵌套 JSON 中，模型可能仅在一个数组元素、单位对象或 required 键上不满足 schema，但其余抽取内容具有可评价价值。早期“一处校验失败即整篇失败”把格式失误放大为内容全失。

**设计。** 区分“可继续评估的表示差异”和“无法信任的流程失效”：

- 单阶段对可确定的解析/required 问题实施受限修复，并始终保留原始响应；
- 两阶段对可解析的 stage schema 差异记录 warning、继续合并和 Gold 裁判，但绝不改写 stage 内容；
- 不可解析或缺少最小结构才硬失败；identity 对齐、重复与覆盖错误保留为 warning，使其内容后果可由 judge 和报告反映。

**报告要求。** 可解析但 invalid 的分数必须标记为不完全可信；主结论同时提供两臂都 schema-valid 的可信配对统计，避免把格式不可靠的结果混入核心结论。

### 10.6 为什么需要两阶段 identity-first，而不是简单分块/OCR

**问题。** 对高样品数或表格主导论文，单次 PDF→JSON 常出现“实体折叠”：模型把几十个同族样品概括成少量家族记录。这通常不是输出 token 截断，而是模型在一次任务中主动压缩结构。

**备选方案。** 按 PDF 页分块、先 OCR 后抽取、或多轮自由式 agent 都能降低单次上下文压力，但会带来页间实体归属、重复合并、证据漂移和不可控工作流问题。

**设计。** 两阶段先从完整 PDF 提取 identity 清单，再以每批 5 个 manifest 条目做 resolve。科研字段 `身份标识` 可能是重复的裸缩写，不能承担内部主键；系统因此按清单顺序分配仅供调度的 `manifest_id`，不进入最终 JSON。这样分批边界来自实体，而不是脆弱的页码或段落边界，并能将覆盖异常作为可报告质量信号。文献信息只由 stage 1 提供，避免 batch 0 同时承担 metadata 与聚合物内容而引发 schema 冲突。

**边界。** 两阶段增加调用次数、成本和失败点；它是独立的工程变量。因此论文比较应分别报告“单阶段 vs 两阶段”的增益，以及各自的 TextGrad 增益，不能将两者混为纯 prompt 优化贡献。

### 10.7 结构化输出护栏为何需要本地校验和协议降级

**问题。** 本地 JSON Schema 能发现错误，但在模型生成完成后才发现；将 schema 原生传给网关可提前约束，却受到不同 OpenAI-compatible/Gemini 实现的子集限制。

**设计。** 对 Responses/Terra 传入动态 schema，采用 `strict: false` 以适配真实 DSL 的 optional fields；对 Gemini 转换到其 `responseSchema` 子集。无论服务端是否接受，最终都由本地 JSON Schema 校验。Gemini 的复杂 stage 2 schema 被拒绝时，进行一次受控 JSON-mode fallback，而不是把一次兼容性 400 误判为整个文档内容失败。

**边界。** fallback 降低了结构约束强度，必须写入日志并纳入运行解释。它不是网络重试，也不能证明模型输出有效。

### 10.8 缓存、resume 与网络失败如何不污染实验结论

**问题。** API 调用昂贵且会受 429/5xx/超时影响；但不受控的缓存可能让不同 schema 或不同 prompt 的候选复用旧预测，得到虚假的“优化提升”。

**设计。** 调用指纹包含角色、文档哈希、schema、提示词、模型配置和请求格式。manifest 冻结数据与实验条件；冻结提示词和已成功文档可在恢复/失败重试时复用。网络重试只针对暂态错误，确定性 4xx 不盲目重试。`request_format_version` 用于隔离协议或 schema 传递方式改变后的缓存与产物。

**边界。** 正式横向实验宜关闭共享缓存或确保所有候选的输入指纹严格隔离；报告中必须区分模型失败和抽取质量退步。

### 10.9 为什么“最后有效轮”不能替代“训练最高分轮”

**问题。** 联合优化的每一轮只要可评估就能成为下一轮的更新起点；后续轮可能因提示词膨胀、局部策略漂移或模型调用波动而低于此前均分。若直接冻结内存中的最后有效状态，训练表显示的最高分候选和实际 optimized 盲测提示词会不一致。

**设计。** 候选轨迹是唯一选择依据：训练结束或恢复后，系统在有效 joint candidates 中按训练均分取最大值（并读取该目录完整的 schema / prompt pair），再冻结、执行 blind-test gate 与运行 optimized 臂。优化器的 live state 只用于续训。

**恢复语义。** 若旧 run 在修复前按末轮生成过 optimized 产物，重新运行会检测冻结状态差异，保留 baseline、失效旧 optimized 产物并重跑；报告不会混用两个候选的结果。

### 10.10 如何把“Judge 波动”与“抽取波动”拆开

**问题。** 过去若想换 Judge 模型或评分提示词，只能重新跑完整抽取，既浪费成本，也把抽取随机性和裁判随机性混在一起；只看一个 LLM Judge 又容易被质疑标尺漂移。

**设计。** Judge-only 固定已有 baseline/optimized predictions，对每个 Judge 配置隔离重评分并记录模型、prompt 哈希和 PDF 开关。另用 metrics-only 计算无需模型的严格叶子与实体身份指标。这两套后验评估都不改变训练 loss、候选选择或原主评分，从而把“方法生成了什么”与“不同评价器怎么看”分开。

**边界。** 严格指标不会识别单位换算或同义表达，多 Judge 也不自动产生真值。论文中应报告结论方向的一致性、相关性和分歧 case，而不是挑选最有利的 Judge 或把不同标尺直接平均。

### 10.11 当前仍应诚实报告的限制

1. 数据集仅 20 篇、训练仅 3 篇，方差与训练集选择效应不可忽略；应报告多 run 或多 seed，而不是只报告最优 run。
2. LLM Judge 仍有随机性和标尺漂移；`temperature: 0`、明确 rubric、score breakdown 和配对设计只能缓解，不能消除。
3. 两阶段 identity manifest 本身也可能漏/错，因而其失败率和覆盖诊断是方法结果的一部分，不应只筛选成功案例。
4. Gemini 复杂 stage 2 `responseSchema` 的网关兼容性仍不完整；fallback 是可运行性方案，不是等价的硬约束。
5. schema-valid 不等于科学事实正确，Gold/PDF 审计和人工 case study 仍是必要补充。

## 11. 开发问题台账（保留的详细历程）

本节是实现过程中的原始问题台账。它与第 10 节的设计总结并存：第 10 节解释“为什么采用当前架构”，本节保留“具体在哪些失败上得到这个结论”。其中的 run、文档 ID 和分数是当时的诊断证据，不应替代当前 run 的正式结果。

### A. Gold 与 schema 契约

| 问题 | 当时现象 | 已采用的处理 |
| --- | --- | --- |
| 表征键命名不统一 | 同一 `¹H NMR` 在 20 篇 Gold 中出现 `氢核磁_1H_NMR`、`氢谱_1H_NMR`、`核磁_1H-NMR` 等变体；固定 schema 不接受这些额外键。 | 进行 A/B 档归一化：变体键改为规范键；保留 `data_backup_normalize_*` 与归一化日志。 |
| Gold 有、schema 无的位置索引 | 多篇 Gold 带 `位置索引` 溯源对象，schema 未定义它，导致 Gold 自身无法通过同一契约。 | 在 `聚合物.items`、`工艺流程.items` 增加可选位置索引，并按 Gold 实测类型定义子键。 |
| 值/范围对象 required 过严 | `应变` 等对象要求单值、最小、最大、单位同时存在；模型省略不存在的子项就使整篇 invalid。 | 将 required 调整为与 Gold 表示一致的最小字段集；不把“未报告的范围端点”强行伪造出来。 |
| 数值、字符串与空值混用 | Gold 中大量 `单值` 为 number，部分体积、当量比、温度为纯字符串或空串。 | 对确有 Gold 证据的字段适度允许 `string/number` 或 `object/string`；不泛化放宽所有字段。 |
| 历史单位格式扁平 | 少数文献以 `摩尔量` + `摩尔量单位` 表示，schema 使用 `{单值, 单位}`。 | 对可确定的 508 处旧表示做无损转换，并保留备份。 |
| PDI、后处理等真实可选性 | 部分 Gold 缺 PDI，后处理有时为数组而不是字符串。 | 将契约调整为真实数据可表达的可选/联合类型，而不是要求模型编造占位事实。 |

### B. 裁判、反馈与分数稳定性

| 问题 | 当时现象 | 已采用的处理 |
| --- | --- | --- |
| Judge 反馈偶发空或格式不完整 | 单次 judge 解析失败会使完整候选不可用。 | 只对 judge 输出解析失败按角色重试；结果解析允许附加 `reasoning` 等非必要键。 |
| 总分掩盖弱维度 | 工艺分从 27 降到 0 时，性质分上涨仍可能抬高总分；例如曾在 `4aa3…`、`0a9e…` 观察到。 | judge 输出 `score_breakdown` 时，反馈向 TextGrad 同时呈现各维度得分，要求优先修广泛低分维度。 |
| 反馈只剩泛化规则 | 早期剥离了当前样本差异，只保留抽象 GENERAL_RULE；模型学不到表格折叠、共享工艺展开等具体结构模式。 | 反馈同时保留 `CURRENT_ERROR` 模式锚点与 `GENERAL_RULE`；优化约束禁止将专属事实写入 prompt。 |
| PDF 裁判过宽或任务漂移 | 允许 PDF 直接推翻 Gold 会使不同裁判设置的绝对分不可比。 | Gold 固定为评分目标；PDF 仅做受限核验，Gold 外但 PDF 支持的内容不加不扣。 |
| 无效 prediction 的分数被误读 | schema-invalid JSON 仍可能具有大量正确内容，judge 分数既有价值又不够可靠。 | 控制台、`validation.json`、盲测 CSV 与 summary 分别记录 invalid；报告并列所有有效配对与可信配对统计。 |
| 换 Judge 必须重跑抽取 | 抽取随机性与裁判标尺变化混在一起，且多花大量 PDF 调用。 | 新增 Judge-only：冻结 prediction，按 label 隔离重评分并记录 Judge 配置摘要，原分数不覆盖。 |
| 只有 LLM Judge 难以复核 | 相同结果可能因裁判模型不同出现分差，缺少完全可复现的参照。 | 新增 metrics-only：严格叶子/身份 PRF、schema-valid 与 exact match 仅作辅助，不进入优化。 |

### C. TextGrad 与 schema-description 联合优化

| 问题 | 当时现象 | 已采用的处理 |
| --- | --- | --- |
| 交替优化难归因 | schema 阶段和 extraction 阶段分别打分，某次增减无法归因到真实的组合。 | 改为联合候选：候选 schema + 更新后的抽取提示词只评估一次，作为一个原子状态接受或回滚。 |
| `min_accept_delta` 丢掉可用更新 | 有效候选因总分未超过人为阈值而被回滚。 | 移除基于小分差的硬接受门槛；非法或缺失关键评估才回滚，最终仍由盲测判断。 |
| schema prompt 被改坏协议 | TextGrad 曾让 schema-prompt 输出与系统协议冲突的 `scope` 数组。 | schema 与抽取使用独立优化器；schema 侧添加结构/路径/description-only 约束。 |
| schema patch 路径不稳定 | 提案曾缺 `.properties`、带多余根 `.properties`、丢父路径、使用包装键前缀或命中同名字段；过去单条坏路径会让整份候选 `proposal_failed`。 | 补丁解析先做受限归一化、真实路径候选消歧、包装键剥离与闭集路径选择；任何通过 fallback 找到的节点立即改写为 schema 中的 canonical path，避免“校验能过、写入又失败”。Optimizer 最终应用前再逐条校验，作为所有运行入口共用的第二道防线。仍无法定位时只剔除该条，其他有效改动继续评估。纯 `description_only` 若全部被剔除，则记录 `no_valid_patch`、输出原因并继续下一轮；联合模式仍评估同时更新的 extraction/evidence/resolve prompt。候选绝不凭空创建字段或路径。 |
| 同名字段语义不足 | 内外层均名为 `反应条件`，description 完全相同，优化器无法区分列表与条目描述。 | 为内层字段补充差异化 description，只改变语义说明，不改变 JSON 结构。 |
| “反幻觉”导致覆盖塌方 | TextGrad 变得过严，丢失整段样品、流程或性质；曾出现 `29d…` 丢 PIC-3、`4d05…` 工艺全空。 | 抽取约束同时强调“不得编造”和“不得为避免幻觉牺牲 schema 可表示的样品/步骤/性质覆盖”。 |
| 必填发射约束被删 | 优化后的 description 把“固定输出、无值留空”改为“明确报告才保留”，导致 required 键消失。 | 对 base schema 及补丁后 schema 重注入 required emission 语句；结构不变，优化器不得删除该保证。 |
| 联合优化收益无法拆分 | prompt 与 schema description 同轮变化，盲测提升不能判断主要来自哪一变量。 | 新增 `description_only` 消融：沿用同一联合候选/patch 校验框架，但不为抽取 prompt 建 loss 或 optimizer，保证其逐字冻结。 |

### D. 抽取校验与受限 repair

| 问题 | 当时现象 | 已采用的处理 |
| --- | --- | --- |
| 解析错误与缺 required 键混为同一种 repair | “只修语法”的 repair 无法补齐缺失键，导致调用白费。 | 区分失败类型：解析错误仅修语法；缺 required 时按错误路径补 schema 规定的空值；其他违规只做合规化尝试。 |
| 嵌套 required 补不全 | 如 `测试速率.单值` 缺失时，只补父对象仍然 invalid。 | `fill_missing_required` 递归按 schema 补嵌套 required；先走确定性补键，再决定是否调用 LLM repair。 |
| 修复隐藏了原始质量 | 仅存最终 prediction 时无法判断模型原始是否合格。 | 原始响应固定写入 `extraction.response.txt`；repair/validation 元数据与最终 prediction 分开保存。 |
| 高样品数文献实体折叠 | `00bc2129` 的 54 个 PI 曾被抽成一个家族记录加少量示例；响应长度未达到上限，说明是主动压缩而非截断。 | 这是引入 identity-first 两阶段的直接证据，而不是继续无限加长单阶段提示词。 |

### E. 两阶段流程的具体失败与修复

| 问题 | 当时现象 | 当前策略 |
| --- | --- | --- |
| batch 0 同时承担文献信息与聚合物 | stage 2 batch 0 要满足 metadata 与聚合物两个不同的 schema 责任，容易出现多个必填错误。 | 文献信息仅从 stage 1 的 `metadata_evidence` 合并；batch 0 与其余 batch 只负责本批 identity 的聚合物。 |
| 每批 identity 太多 | 一批 10 个 identity 时，模型重复记录、漏记录和字段串位增多，validation errors 集中爆发。 | 将 `BATCH_SIZE` 固定为 5；这是工程常量而非待优化参数，保证候选间可比。 |
| stage 1 被 prompt 推向最终内容 | evidence/index prompt 可能输出内容字段，和中间 schema 冲突。 | 固定协议要求 stage 1 只输出 metadata evidence 与 identity 清单；最终字段解析留给 stage 2。 |
| stage 2 字段语义放错 | 模型用英文别名、把标量写成 `{min,max,unit}`、或者把别批 identity 放入本批。 | resolve 系统提示词列出 exact-field/标量规则；本地批对齐检查将 request、covered 与 records 的差异写为 warning，并由 judge 计入质量。 |
| 普通 schema 误差被硬失败放大 | 例如单一原料数组项类型不符，早期会停止整篇两阶段流程。 | 可解析且保有最小对象/列表结构时写 `stage_validation_warnings` 后继续；不修复、不掩盖，最终让 judge 和报告反映。 |
| identity/最小结构不可信 | stage 1 可重复列出同一 stub；covered identities 也可能缺失、重复或跨批。 | 同 identity key 的 stub 稳定去重，且忽略仅由 `位置索引`造成的差异，因为同一材料可在不同表格、图片和正文位置重复出现；名称、样本形态或结构特征等身份语义字段冲突时才保留多个条目，并由本地 `manifest_id` 路由。coverage 差异写 warning 后继续。只有不可解析或缺少最小对象/数组才硬失败，失败产物尽可能落盘 raw 与 diagnostics。 |

### F. 网络、协议、缓存与报告工程

| 问题 | 当时现象 | 已采用的处理 |
| --- | --- | --- |
| `APIError` 基类未重试 | 流式中途 `server_is_overloaded` 等错误不一定属于预期子类。 | 将暂态 API 基类纳入统一重试；429、5xx、超时、连接错误使用退避。 |
| 413 被无意义重试 | Chat 图片协议的大 PDF 超过网关请求体限制，等待多轮后仍失败。 | 将 413 列为确定性非重试错误，并在诊断中说明请求体风险。 |
| chat 空 content 无线索 | 推理模型耗尽输出预算后 `delta.content` 为 0，最终只显示 `Invalid JSON`。 | 写 `_empty_content_diagnostic`：delta 键、reasoning 痕迹、chunk 数和 finish reason。 |
| 网关不支持 Responses | 部分端点对 `/v1/responses` 返回 404，只支持 chat + image。 | 加 `chat_completions_pdf_images` 回退协议；PDF 渲染为 PNG，但不把它与原生 PDF 协议混为同一成本/限制。 |
| 原生 schema 兼容性不一致 | Terra strict schema 与真实 optional DSL 不兼容；Gemini 复杂 stage 2 `responseSchema` 可返回 400。 | Terra 用 `strict:false`；Gemini 首次 stage 2 兼容性 400 后仅一次无 schema JSON-mode fallback，并在本次客户端生命周期内记忆该 scope，后续 stage 2 不再发送必然失败的 schema；stage 1 与本地校验始终保留。 |
| 缓存导致候选看似提升 | 若 cache key 不含候选 schema hash，变更 schema 后可能复用旧预测。 | 缓存指纹包含候选 schema、提示词、文档、角色、模型和请求格式；正式比较可关闭 cache。 |
| report-only 被 config 漂移拦住 | 已完成 run 后补配置默认字段，纯读报告重建却因 manifest 指纹失败。 | report-only 仅基于既有产物重建，并按优化模式读取对应 checkpoint。 |
| 盲测均值混入失败/不可信配对 | 只看全部均值时，单侧 invalid 或失败会掩盖真实增益。 | `blind_test_summary.json` 和 report 并列全部与 trusted 配对的数量、均值、中位数、胜平负和失败数。 |

### G. 训练集选择与尚未完成的研究问题

| 观察/问题 | 证据与当前取舍 |
| --- | --- |
| 三篇训练集也需覆盖执行风险 | 当前训练集采用 `00bc2129`（高样品数表格）、`25ef8d4f`（电化学多阶段/多指标）、`0ba22b43`（共享继承工艺）。它比仅按复杂度取前三更贴近实际失败模式；代价是最难样本进入训练，盲测绝对分会改变，报告必须说明。 |
| 训练最优不一定盲测最优 | 曾见提示词由约 6935 膨胀到 9296 字符、训练均分提高但盲测更差。当前不把“最佳训练轮”解释为泛化证明；后续可研究长度正则、较少迭代与多 seed。 |
| 两阶段不是纯 TextGrad 改进 | 分批 identity 解析本身就是工程干预，可能比提示词更新带来更大增益。未来报告需做单阶段 baseline、两阶段 baseline、各自 optimized 的消融，而不是把全部变化归给 TextGrad。 |
| Gemini stage 2 原生 schema 仍有限制 | 当前已实现按客户端/阶段的能力记忆：首个 stage 2 schema 400 后，后续 stage 2 直接 JSON-mode，以减少重复 400。仍需在报告中披露该 run 的 stage 2 缺少服务端 schema 护栏，且下一次新运行会重新探测能力。 |
| 小数据集的统计不确定性 | 20 篇数据、3 篇训练意味着随机调用与训练集选择均有影响。应优先报告配对分布、case study、失败率和多次独立 run，而不是单个最大均值。 |

## 12. 离线测试

```bash
python -m pytest -q test/tests
```

测试不调用外部模型，覆盖数据发现、schema 转换、校验、裁判、两阶段 batch 对齐/合并、客户端协议和运行恢复。


### 2.4 四变量 evidence routing：`two_stage_evidence_routing_schema_description`

这是独立于上述两阶段模式的新方法，现有 three-variable two-stage、静态 `coverage_plan` 模式及其缓存/结果均不改变。它将 Stage 1.5 命名为 **evidence routing**，因为其职责是把 stage-1 的实体清单路由到后续需要读取的字段组、证据锚点与显式共享关系，而不是要求模型“覆盖”某个 Gold 清单。

```text
schema-description prompt ─┐
evidence/index prompt ─────┼─ 同一训练反馈 → 四个独立 TextGrad 更新 → 原子接受/回滚
routing prompt ────────────┤
resolve prompt ────────────┘
```

routing map 只由 PDF 与 identity manifest 生成，按当前 batch 过滤后传给 resolve；它不进入最终 prediction，也不看 Gold。`round-000` 已是完整四提示词基线，optimized 臂则冻结训练均分最高的整套四变量候选。因此该实验测量的是“增加并优化 evidence routing”而不是把一个事后规则加在 optimized 臂上。每个候选目录及冻结目录都会保存 `prompt.evidence-routing.txt`；恢复器从最高分候选目录读取四份原文，不会拿最后一轮覆盖它。对于在该工件写入修复前中断、且只留下 routing 哈希的历史 run，系统会拒绝伪恢复：原文不可由哈希重建，必须以新 run 重跑训练。checkpoint、`--resume require`、`--resume-from-round`、`--report-only` 和 `--retry-failed-primary` 均会携带该工件。

引入动机是：identity manifest 只能回答“有哪些实体”，却未指定每个实体应优先检索哪些字段组、哪些证据可能共享；长 PDF 的实体折叠、共享工艺遗漏与“只报告测试存在”的空值记录容易在这一步丢失。它也是一个需检验的假设而非已证实结论：新增一次 PDF 调用，会增加成本和方差，可能无增益。建议在同一 split/model/cache 条件下同时比较普通 two-stage、固定 routing 和可学习 routing。

运行：

```bash
python -m test.textgrad_validation --config test/textgrad_validation/config_two_stage_evidence_routing.yaml --run-id v2-goldnorm-focus-sol-two-stage-evidence-routing-01
```
