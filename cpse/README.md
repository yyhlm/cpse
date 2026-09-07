# `test/`：PDF→JSON 的 TextGrad 金标验证实验

> 最新实验结果总表、统计检验、消融、外部基线与投稿结论见
> [`CURRENT_RESULTS_20260827.md`](CURRENT_RESULTS_20260827.md)。该文件是当前结果口径的统一入口；
> 各实验必须使用各自运行内的配对 baseline 计算增益，不跨 run 替换 baseline。

`test/` 是一个自包含的科研 PDF 信息抽取实验：20 篇 PDF 与人工 Gold JSON 构成固定数据集，在固定 schema、固定训练/盲测划分和固定裁判设置下，检验 TextGrad 优化是否能提升抽取质量。

实验结论只看盲测集中同一文档的配对差值 `optimized_score - baseline_score`；训练分只用于选择候选，不可单独作为有效结论。

## 先选实验模式

| 模式 | 配置示例 | 适用情况 | 优化变量 |
| --- | --- | --- | --- |
| prompt-only 消融 | `config_prompt_only.yaml` | 与联合方法做同模型、同训练集、无缓存的公平比较 | 仅抽取提示词；schema description 冻结 |
| schema-description-only 消融 | `config_description_only.yaml` | 判断收益是否来自 schema 语义描述，而不是抽取 prompt 改写 | 仅 schema description；抽取提示词冻结 |
| 单阶段联合优化 | `config_final.yaml`、`config.yaml` | PDF 可在一次调用中直接完成完整 JSON 抽取 | 抽取提示词 + schema description |
| 两阶段联合优化 | `config_two_stage_terra.yaml`、`config_two_stage_gemini.yaml` | 样品很多、表格密集，单次抽取容易把多个实体折叠或漏掉 | schema description + evidence/index 提示词 + resolve 提示词 |

单阶段流程为：

```text
PDF → 完整 JSON → 校验 → 裁判
```

两阶段流程为：

```text
PDF → stage 1：元数据证据 + identity 清单
    → stage 2：按 identity 分批 resolve
    → 合并 → 最终 JSON → 校验 → 裁判
```

两阶段不是为了“多一步就更好”，而是为高样品数论文提供更细的实体覆盖约束。两种模式应使用相同的 Gold、裁判配置和盲测集进行公平比较。

## 快速开始

所有命令在仓库根目录运行。密钥放在 `test/textgrad_validation/.env`；配置只引用环境变量名，不应写入真实密钥。

训练样本数量消融支持 `--train-count 1|2|3`。显式 `train_ids` 表示固定且有序的
3篇训练池；1/2-shot 只启用其前N篇，其余训练池文档不会进入盲测，因此三种设置
始终共享同一17篇盲测集。不同shot数必须使用不同 `run-id`。

```bash
# -----------------1、单阶段-------------------
# 单阶段：Sol 示例
python -m test.textgrad_validation --config test/textgrad_validation/config_final.yaml --run-id sol-single-01

# 单阶段：Terra 示例
python -m test.textgrad_validation --config test/textgrad_validation/config.yaml --run-id terra-single-01

# 单阶段 + schema patch：Sol 示例

# 单阶段 + schema patch：Terra 示例

# 只优化 schema description、冻结抽取 prompt 的公平消融
python -m test.textgrad_validation --config test/textgrad_validation/config_description_only.yaml --run-id sol-description-only-01

# 只优化 extraction prompt、冻结 schema description 的公平消融
python -m test.textgrad_validation --config test/textgrad_validation/config_prompt_only.yaml --run-id sol-prompt-only-01

# 两阶段 + schema patch：Terra / Responses
python -m test.textgrad_validation --config test/textgrad_validation/config_two_stage_terra.yaml --run-id terra-two-stage-01

# 两阶段 + schema patch：Gemini generateContent
python -m test.textgrad_validation --config test/textgrad_validation/config_two_stage_gemini.yaml --run-id gemini-two-stage-01
```

先用冒烟运行检查接口、密钥、schema 和产物目录：

```bash
python -m test.textgrad_validation --config test/textgrad_validation/config_two_stage_terra.yaml --run-id smoke-two-stage-01 --smoke --blind-doc 0b30cafa06dfd1c3860911a2fdaf7f0b
```

完整配置、协议兼容性、失败语义和产物说明见 [textgrad_validation/README.md](textgrad_validation/README.md)。
论文消融、辅助指标、多 Judge 与独立重复的逐行命令见 [EXPERIMENT_COMMANDS.md](EXPERIMENT_COMMANDS.md)。外部 GEPA 基线依赖其适配器的 `propose_new_texts` 钩子；若日志显示未提出任何新候选且 optimized 完全复用 baseline，则该 run 是接口失败而非 GEPA 的负结果，不得纳入比较。

## 实验边界与核心规则

1. **数据与划分**：`data/` 中每个 stem 必须同时具有 PDF 与 Gold JSON。训练集为 3 篇，盲测集为其余文档；显式 `train_ids` 会被写入 manifest。
2. **Gold 是评分目标**：裁判始终以 Gold 为主目标。`evaluation.include_pdf: true` 时，PDF 仅用于核验争议、Gold 外新增内容和疑似 Gold 问题；它不扩大“必须抽取”的范围。
3. **训练与盲测隔离**：训练文档只用于优化，提示词和候选 schema 冻结后才允许盲测。无论优化器内部最后停在哪一有效轮，冻结与 optimized 盲测始终使用训练均分最高的有效候选及其配套 schema / prompt；绝不按最后一轮盲测。
4. **不以无效输出掩盖问题**：原始响应和校验结果总会落盘。可解析但 schema 不一致的结果仍可供裁判评估，但其得分会标为不完全可信；最终报告同时给出全部配对和两臂均通过校验的可信配对统计。
5. **单阶段与两阶段的修复策略不同**：单阶段可对明确的解析/必填字段问题进行受限修复并保留原始响应；两阶段不重写 stage 输出。两阶段的可解析 schema 不一致、identity 覆盖不一致和可恢复的 stage-1 重复 stub 都记录 warning 后继续；只有不可解析、缺少最小阶段结构或 API 最终失败才硬失败。
6. **结论必须配对解释**：同时报告有效配对数、校验失败数、`mean_paired_delta`、逐篇 delta 和 case study，而不能只引用训练均分。
7. **schema-description 补丁容错**：补丁路径必须最终指向当前 schema 的真实 `description` 路径。路径异常时先在同名真实字段构成的闭集中消歧；仍无法定位时仅剔除该条，保留其余合法 patch 参与本轮候选评估，绝不凭空创建字段或路径。
8. **语义判断与契约校验分工**：Judge 与 schema patch LLM 负责理解 prediction、Gold、反馈及完整 schema 的语义，不用本地字符串精确比较或固定 top-k 白名单替代。确定性代码只验证 patch 指向真实 `description`，并保证字段名、类型、层级、数组形式和 required 关系不变。
9. **schema patch 双层容错**：Runner 优先对错误路径做闭集候选纠正，并将 fallback 定位的缩写路径改写为 schema 的 canonical path；Optimizer 在最终应用前再次逐条校验。单条无效 patch 只被剔除，不连带拒绝同一候选中的有效修改；纯 description-only 候选全部无效时记为 `no_valid_patch` 并继续下一轮，联合模式则仍评估同时更新的其他提示词。状态和原因写入日志及 checkpoint。

## 可选 coverage-plan 两阶段消融

`optimization_mode: two_stage_coverage_plan_schema_description` 是普通两阶段的并列新模式：Stage 1 先生成 metadata 与 identity manifest；新增的固定 Stage 1.5 仅基于 PDF 和 manifest 生成“每个实体应覆盖哪些字段组、可选证据锚点及显式共享关系”的中间 `coverage_plan`；Stage 2 只接收当前 5 个 manifest 条目的 plan 子集。它不进入最终 JSON，也不是第四个 TextGrad 优化提示词，因此可独立检验中间接口是否改善长文献的实体覆盖。若 plan 不可用，记录 warning 后退化为普通 two-stage resolve，不会凭空修复或中止文献。

配置和命令：

```bash
python -m test.textgrad_validation --config test/textgrad_validation/config_two_stage_coverage_plan.yaml --run-id v2-goldnorm-focus-sol-two-stage-coverage-plan-01
```
## 两阶段的关键契约

- stage 1 只产生 `metadata_evidence` 与 `identity_manifest`，不输出最终内容 JSON。
- stage 2 每批固定处理 **5 个 manifest 条目**（代码常量，不是 yaml 配置项）。stage 1 的科研字段 `身份标识` 不被当作内部主键；系统按首次顺序生成仅供调度的 `manifest_id`，它只出现在 stage-2 请求中，绝不进入最终 prediction。stage 2 返回本批的 `covered_identities` 与聚合物记录；文献信息由 stage 1 合并。
- stage 1 完成后，stage 2 的独立 batch 可并行；`max_parallel_calls` 是 stage 1、stage 2 batch 与 judge 共享的全局 API 在途上限，不是“每篇论文各自的 batch 上限”。
- identity 覆盖是质量诊断：请求 identity、`covered_identities` 与聚合物记录不一致（重复、漏掉或跨批混入）会写入 warning 并交给 judge；stage 1 的完全等价重复 stub 会稳定去重，身份字段冲突的同 key 条目会保留为不同 manifest 条目。不会借助 Gold 补造 DOI、序号或科研标识。
- stage schema 的一般类型/字段问题是 warning，不会自动修复或停止后续裁判；最终 JSON 的 `validation.json` 是结果是否严格符合总 schema 的唯一依据。

## 协议与原生 JSON schema

| 协议 | `api_protocol` | 两阶段结构化输出行为 |
| --- | --- | --- |
| OpenAI Responses / Terra 网关 | `responses` | stage 1 与 stage 2 动态传入 JSON Schema；为兼容真实 DSL 的可选字段，使用非严格模式。|
| Gemini generateContent | `gemini_generate_content` | 请求 `responseSchema`。当前 stage 1 可用；复杂 stage 2 schema 被部分网关以 HTTP 400 拒绝时，本次运行记忆该能力缺失：仅首次回退一次，后续 stage 2 batch 直接 JSON mode，stage 1 不受影响。|
| Chat Completions 图片回退 | `chat_completions_pdf_images` | 将 PDF 渲染为页面图片；用于不支持 PDF 文件输入的端点，成本和请求体风险更高。|

Gemini 的降级是已知兼容性措施，不表示最终输出一定合规。若日志出现 `gemini-schema-fallback`，应在报告中记录 stage 2 的服务端 schema 护栏已在该 run 内关闭；stage 1 仍保留其已验证可用的护栏。

## 目录

```text
test/
├── data/                         # 20 对 PDF + Gold JSON
├── schema.json                   # 仓库字段式 schema DSL
├── results/<run-id>/             # 每次实验的完整可审计产物
├── tests/                        # 不调用 API 的离线单元测试
└── textgrad_validation/          # 实验实现、配置与提示词
```

`results/<run-id>/` 中最重要的文件：

```text
meta/manifest.json                # 输入、划分、模型及请求格式指纹
meta/config.snapshot.json         # 脱敏配置快照
meta/events.jsonl                 # 启动命令、重试、审计等事件
meta/frozen_protocol/             # run 创建时使用的全部文件型提示词及 SHA-256 索引
prompts/                          # 冻结的基线/最优提示词（两阶段包含对应 prompt pair）
schemas/                          # 交替优化模式中冻结的 schema
training/                         # 每轮候选的原始响应、校验与裁判结果
blind_test/                       # baseline 与 optimized 两个盲测臂
blind_test_summary.json           # 主要配对结论及可信配对统计
judge_only/<label>/               # 可选：冻结 prediction 的隔离重评分
deterministic_metrics/            # 可选：完全无模型的严格辅助指标
analysis/                         # 不调用模型的确定性差异分析
report.md                         # 人类可读汇总
```

每个**新 run** 在建立 manifest 时，都会把当前配置实际引用的全部文件型提示词复制到
`meta/frozen_protocol/prompts/`，并在 `meta/frozen_protocol/index.json` 记录逻辑角色、原始路径、
run 内相对路径和 SHA-256。范围包括抽取 initial/system、Judge、Gold audit，以及配置启用时的
schema-description、evidence、resolve、coverage-plan 和 evidence-routing 提示词。该目录记录的是
实验启动时的完整协议输入，与训练后写入 `prompts/` 的 baseline/最佳候选用途不同。

快照只在 run 首次创建时写入，`--resume` 和失败重试不会用后来修改过的源文件覆盖它。因此以后即使
仓库中的提示词继续演化，也能从对应 run 恢复当时的原文。旧 run 若没有这个目录，程序不会拿当前
提示词伪装成历史版本；此类历史内容只能从当时的外部备份恢复。

两阶段文档目录中的 `extraction.metadata.json` 还会记录 `raw_manifest_count`、去重后的 `manifest_count`、`stage1_manifest_warnings`、stage schema warning 与覆盖 warning。真正失败时的 `extraction.failure.json` 会尽可能保留失败阶段、原始模型响应和诊断上下文，便于复盘而不改变输出。

## 续跑与重试

```bash
# 恢复未完成的 run；manifest 必须匹配
python -m test.textgrad_validation --config <config> --run-id <run-id> --resume require

# 仅保留已接受的 0..N 轮并从 N+1 重训；适用于两种 alternating 模式
python -m test.textgrad_validation --config <config> --run-id <run-id> --resume-from-round N

# 仅重建报告，不调用模型
python -m test.textgrad_validation --config <config> --run-id <run-id> --report-only

# 仅重试失败的主流程文档，保留成功产物与冻结提示词（记得cache：false）
python -m test.textgrad_validation --config <config> --run-id <run-id> --retry-failed-primary

# 新建独立 run，一次性运行 17 篇 direct baseline；不训练、不生成 optimized 臂
python -m test.textgrad_validation --config <nocache-config> --run-id <baseline-run-id> --blind-baseline-only

# 只用当前配置的 Judge 重新评分既有 baseline/optimized prediction；不重新抽取、不覆盖原分数
python -m test.textgrad_validation --config <judge-config> --run-id <run-id> --judge-only <judge-label>

# 强制重写同一 judge-label 的重评分结果
python -m test.textgrad_validation --config <judge-config> --run-id <run-id> --judge-only <judge-label> --judge-force

# 完全不调用模型，从既有 prediction 计算严格确定性辅助指标
python -m test.textgrad_validation --config <config> --run-id <run-id> --metrics-only

# 完全不调用模型，生成脱敏、无 PDF、带 SHA-256 清单的公开 artifact 包
python -m test.textgrad_validation --config <config> --run-id <run-id> --artifact-release

# 完全不调用模型，生成可复现实验代码包；不含数据、结果、缓存和服务配置
python -m test.textgrad_validation --config <config> --code-release

# 完全不调用模型，统计配对置信区间、显著性、Judge一致性及与严格指标的相关性
python -m test.textgrad_validation --config <config> --run-id <run-id> --statistics-only
```

```bash
# 仅统计已有抽取工件中的 token；不发 API 请求、不改结果。
python -m test.textgrad_validation --config <config> --run-id <run-id> --usage-only
```

```bash
# 仅审计冻结 prompt / schema 是否精确复制训练 Gold 中的高风险事实；不发 API 请求。
python -m test.textgrad_validation --config <config> --run-id <run-id> --leakage-audit
```

变更 Gold、总 schema、模型、提示词、训练集、协议或 `request_format_version` 后，使用新的 `run-id`，不要把不同实验条件混入一个 run。网络重试覆盖 429、5xx、超时和连接错误；一般 4xx 不重试。Gemini `responseSchema` 的 400 是唯一受控例外：会执行一次无 schema 降级请求。

恢复时会重新从候选轨迹选择训练均分最高的有效候选，而不是相信旧的“最后有效轮”冻结文件。若这个选择与既有 optimized 臂不一致，系统会保留训练和 baseline 产物、清除 `blind_test/optimized/` 并用重新冻结的最佳候选重跑 optimized 臂，避免混合不同提示词条件。`--retry-failed-primary` 则不改变冻结候选，只重跑失败的 baseline / optimized 文档并在 `meta/events.jsonl` 记录重试策略；交替模式会分别复用冻结的 base schema 与 selected schema，两阶段还会复用冻结的 evidence / resolve 提示词。它不仅检测 checkpoint 的失败状态，也检测必要产物是否缺失：有效预测需要 `prediction.json`、`validation.json`、`judge.result.json`；允许 schema-invalid 的预测则需要原始响应、`validation.json`、`judge.result.json`。手动删除产物后可直接用该命令补跑，事件会标记 `required_artifact_missing`。控制台会同时打印重试前后的**全量**有效配对数、baseline/optimized 均分与配对增益，并逐篇显示失败项的旧分数→新分数；其中阶段内部的 `mean=... (1/1 docs)` 仅是本次重跑子集的执行进度，不是完整盲测结论。

单阶段 `alternating_schema_description` 与两阶段 `two_stage_alternating_schema_description` 都会在 round-000、以及每个后续 round 完成评估并确定接受/回滚后立即写入 `meta/checkpoint.json`。因此新运行被终止后可直接以同一命令加 `--resume require` 继续；它会从已持久化的最后一轮之后开始，而不是重跑已完成的训练轮。Ctrl+C 会设置 run 级取消事件、取消尚未开始的任务；两阶段内部 batch 使用 daemon 线程池并不等待卡住的网络调用。已经进入 API 调用的请求无法被 Python 强制中止，但不会阻塞主流程退出或污染 checkpoint。

## 评分复核与无模型辅助指标

`--judge-only LABEL` 只读取一个既有 run 的冻结 prediction、对应 arm schema 与 Gold，再使用**当前配置中的 Judge**重新评分。结果隔离写入 `judge_only/<LABEL>/`，包含逐文档两臂结果、`documents.csv`、`summary.json` 和独立 `output.log`；终端会显示启动规模、逐篇 score、`(reuse)` 复用标记、跳过原因及最终 paired 汇总。原有 `blind_test/**/judge.result.json`、checkpoint、冻结提示词和主报告均不修改。同一 label 默认断点复用，`--judge-force` 才会覆盖。单篇 Judge 返回格式异常、解析失败或临时调用错误不会中止整批：该篇写为 `<arm>.judge.failure.json`，其余篇照常落盘；下一次不带 `--judge-force` 的同 label 命令会复用成功的 `<arm>.judge.json`，仅补跑 failure/missing 项。若使用 `--judge-force`，该篇旧结果会在重评前清除，避免一次强制重评中途失败后误复用旧分数。summary 同时记录 Judge 模型、Judge prompt 哈希、`include_pdf` 和定位开关，适合用多个 Judge 模型做敏感性分析。

仓库已提供可直接使用的多Judge配置：

```bash
python -m test.textgrad_validation --config test/textgrad_validation/config_judge_terra.yaml --run-id <run-id> --judge-only terra
python -m test.textgrad_validation --config test/textgrad_validation/config_judge_gemini.yaml --run-id <run-id> --judge-only gemini
```

`--metrics-only` 完全不创建 API 客户端，输出到 `deterministic_metrics/`。除保留旧的 strict-leaf 和 property-tuple 指标外，当前主辅助指标是 **entity-aligned、unit-normalized、schema-aware slot PRF**：先以 schema 身份字段、名称/缩写和确定性记录相似度进行一对一实体对齐；随后只在已对齐实体内匹配性质；最后把实体、性质、数值/单位和测试条件拆为独立 slot 计数。数值采用固定 1% 相对容差，并显式换算温度、频率、时间、压力、长度、体积、质量、百分比和波数单位。未对齐实体或性质的所有 slot 仍进入分母，错误样品归属不能靠全局值重合获得分数。输出同时给出 entity、property detection、value/unit、condition 四个分项，以及 macro/micro 总 PRF。自由文本仅做 NFKC、空白和大小写规范化，不用脆弱的同义词规则冒充语义理解；该指标完全可复现，但仍不能替代独立 Judge 或人工 PDF 审计。JSON exact match 不统计。

`--statistics-only` 同样完全离线，读取主 `blind_test_documents.csv`、已有 `judge_only/*/documents.csv` 与 `deterministic_metrics/documents.csv`，输出到 `statistics/`。它报告配对 delta 的均值、中位数、胜/平/负、固定种子的 paired bootstrap 95% CI，以及双侧符号翻转 permutation p-value；17 篇时枚举全部 `2^17` 种符号组合。相同统计现在也覆盖 entity alignment、property detection、value/unit、condition 与总 schema-aware slot F1。若存在多 Judge 结果，还报告逐篇 delta 的 Spearman 相关和方向一致率，并探索主/Judge delta 与全部确定性指标的相关性。相关性因样本仅17篇，应作为探索性稳健性分析而非独立主结论。

`--usage-only` 是离线账务核对工具，不重跑、不调用 API，也不改变主结果。新运行由 Responses 客户端把每个物理 attempt 追加到 `meta/api_calls.jsonl`，包括 extractor、Judge、TextGrad update、schema patch、audit、失败、retry、usage、请求耗时与 backoff；汇总区分逻辑调用和物理 attempts，失败但无 usage 的请求标为 unknown，绝不按零成本处理。成本使用带日期的公开列表价快照生成 `estimated_reference_cost_usd`，实际账单仍以服务端账单为准。历史 run 若没有该账本，命令仍回退到 `extraction.metadata.json` 的旧版可核验 extraction 范围，无法凭空恢复 Judge/optimizer/retry 成本。论文可以只把主实验写成 `primary-run cost profile`，其他消融至少说明调用协议/逻辑调用规模，不得把主实验成本暗示成所有方法的统一成本。

`--artifact-release` 将既有 run 整理到 `test/artifact_release/<run-id>/`：公开 Gold JSON、冻结 prompts/schemas、split/config/checkpoint、prediction、Judge 输出、确定性指标、统计与 leakage audit；原始 PDF、`.env`、API URL、proxy、密钥环境变量名和运行日志被排除。`MANIFEST.json` 与 `SHA256SUMS.txt` 为每个文件提供内容哈希。上传前仍需确认 Gold 标注的许可，并在匿名审稿阶段使用符合会议匿名规则的托管链接。

`--code-release` 输出到仓库根目录的 `code_release/`，用于提供完整可执行代码：实验包、提示词、脱敏 YAML 配置、schema、离线测试和依赖清单均会保留；数据、Gold、历史结果、缓存、PDF、`.env`、服务 URL、proxy 与密钥环境变量名均会排除。论文发布时应同时提供代码包和对应 run 的 artifact 包：前者用于复现，后者用于核验论文中的固定结果。

`--leakage-audit` 是训练事实复制审计：从 run manifest 中读取训练 Gold，检查最终 optimized prompt/schema 是否包含 DOI、标识符、长文本或精确数值等高风险训练事实的精确字符串，并验证 schema 去除 description 后的结构不变。结果写入 run 的 `leakage_audit/`。它不调用 API；“无精确命中”只能作为反记忆证据的一部分，不能替代人工检查或证明没有语义层面的记忆。

## 公平内部基线矩阵

论文实验应在同一数据划分、Gold、模型、Judge、PDF 开关、轮数、并发/重试和 cache 策略下报告以下对照：

| 对照 | 读取位置/运行模式 | 可归因变量 |
| --- | --- | --- |
| Direct zero-shot | 任一单阶段 run 的 `blind_test/baseline` | 无优化 |
| Prompt-only TextGrad | `optimization_mode: single` 的 optimized 臂 | 只改抽取 prompt |
| OPRO Prompt-only | `opro_prompt_only` | 历史prompt与客观分数驱动，不读取自然语言反馈 |
| GEPA Prompt-only | `gepa_prompt_only` | 官方GEPA适配器，使用富执行反馈反思演化prompt |
| MIPROv2 instruction-only | `mipro_v2_instruction_only` | 官方DSPy MIPROv2，demonstrations全部关闭 |
| MIPROv2 instruction+demo | `mipro_v2` | instruction与最多1个训练PDF文本→Gold labeled demo |
| 无 Schema 直抽 | `schema_free_direct` | 抽取请求不含 schema、字段清单或示例；模型从 PDF 自行组织自由 JSON |
| 固定 3-shot 直抽 | `few_shot_direct` | schema + 固定三篇训练 PDF 文本→Gold；不训练、不优化任何变量 |
| Description-only TextGrad | `config_description_only.yaml` 的 optimized 臂 | 只改 schema description，抽取 prompt 固定 |
| Prompt + description | `alternating_schema_description` 的 optimized 臂 | 两个变量联合更新 |
| Fixed two-stage | two-stage run 的 `blind_test/baseline` | 固定 evidence/resolve/schema 的架构效应 |
| Optimized two-stage | 同一 two-stage run 的 `blind_test/optimized` | 架构 + 三提示词/schema-description 优化 |

固定 two-stage 已天然包含在每个 two-stage run 的 baseline 臂中，所以没有另建一个重复调用的模式。比较时优先使用同一 run 内的配对差；跨 run 比较则必须核对 manifest 与 Judge-only 的评分设置一致。

### Two-stage execution-only 消融

若要把“manifest-conditioned bounded execution”的作用从提示词/schema-description 优化中单独拆出，可在一个已完成的 two-stage run 上运行：

```bash
python -m test.textgrad_validation --config test/textgrad_validation/config_two_stage2.yaml --run-id v2-goldnorm-focus-sol-two-stage-03 --execution-ablation
```

命令复用训练均分最高的冻结状态，不重新训练。它先生成逐篇 manifest，再让 `manifest_unbounded` 与 `manifest_bounded_5` 共享同一份 manifest，仅改变 stage-2 是否按 5 个 identity 分批。结果隔离保存在 `execution_ablation/`；详细协议和缓存规则见 [textgrad_validation/README.md](textgrad_validation/README.md)。

外部基线配置分别为`config_opro.yaml`、`config_gepa.yaml`、`config_mipro_v2_instruction_only.yaml`和`config_mipro_v2.yaml`。GEPA/MIPROv2是可选依赖：`gepa==0.1.4`、`dspy==3.3.1`。完整MIPROv2会把选中的训练PDF文本和Gold带入盲测提示词，输入成本与prompt-only方法不同，必须作为few-shot参照单列，而不能用于归因schema-description收益。针对DSPy 3.3.1在“0 bootstrapped + labeled-only”设置下超过3个few-shot候选会触发空区间采样的问题，适配器固定使用3个内置few-shot候选，但仍保留配置对应数量的instruction候选，不改变无bootstrap的基线定义。

新增的两条“能力来源”基线不调用优化器，只生成 `blind_test/baseline`：

```bash
python -m test.textgrad_validation --config test/textgrad_validation/config_schema_free_direct.yaml --run-id v2-goldnorm-focus-sol-schema-free-direct-01 --blind-baseline-only
python -m test.textgrad_validation --config test/textgrad_validation/config_few_shot_direct.yaml --run-id v2-goldnorm-focus-sol-fixed-3shot-direct-01 --blind-baseline-only
```

`schema_free_direct` 用于回答“完全不给结构契约，强模型能抽到多少语义信息”。虽然配置仍引用 `schema.json` 以冻结同一数据版本和 3/17 划分，但抽取 payload 与该模式 Judge payload 都不含 schema。专用 Judge 按语义对齐，不因自由 JSON 的键名或层级与 Gold 不同扣分。因此它与主 Judge 的绝对值不应被解释为纯粹的同一量尺差值；论文应同时报告其协议差异，并以 case study 检查实际语义覆盖。

`few_shot_direct` 用于回答“直接把三篇 Gold 当作上下文，是否已足以替代优化”。三篇训练 PDF 先由 `pypdf` 确定性转为全文文本，再与完整 Gold JSON 一起放入每次盲测请求；目标 PDF 仍单独上传，目标 Gold 从不进入请求。当前三组示例实际约 44.8 万字符，属于高输入成本 ICL 基线，应同时报告 token/调用成本，不能与短提示词优化方法视为等成本。两条配置默认关闭共享缓存，避免把历史直接抽取混入新协议。

## 离线测试

```bash
python -m pytest -q test/tests
```

测试不调用 API，覆盖数据集、schema 转换、校验、裁判、两阶段合并与运行编排。


## 四变量 evidence-routing 两阶段实验

`optimization_mode: two_stage_evidence_routing_schema_description` 在原有 two-stage 的三个可学习变量（schema-description、stage-1 evidence/index、stage-2 resolve）之外，将 Stage 1.5 的 **evidence-routing prompt** 也作为第四个 TextGrad 变量。它只使用 PDF 与 stage-1 `identity_manifest`，生成实体→字段组/证据锚点/显式共享关系的中间路由图；每个 stage-2 batch 只接收自己的子图。路由图不是最终 schema 字段、不进入 Gold Judge，也不包含 Gold 事实。

该模式的 round-000 是四提示词完整流程的未优化 baseline；每一轮四个变量基于同一训练反馈分别更新，再与候选 schema 一起作为原子候选接受或回滚。训练结束、恢复或缩短轮数时，系统冻结训练均分最高候选的四份提示词并进行 paired blind test。每个候选目录都持久化 schema/evidence/evidence-routing/resolve 四份原文，恢复时按候选目录恢复，而不是用最后一轮覆盖最高分候选；缺少任一原文的历史 run 不能被语义等价地还原，必须新建 run 重跑训练。额外调用会增加成本与方差，因此它应作为方法假设，与普通 two-stage 和固定 routing 进行同条件消融，而不能预设为必然提升。

```bash
python -m test.textgrad_validation --config test/textgrad_validation/config_two_stage_evidence_routing.yaml --run-id v2-goldnorm-focus-sol-two-stage-evidence-routing-01

# 中断后继续，复用所有已完成 round 与其四份候选提示词
python -m test.textgrad_validation --config test/textgrad_validation/config_two_stage_evidence_routing.yaml --run-id v2-goldnorm-focus-sol-two-stage-evidence-routing-01 --resume require
```
