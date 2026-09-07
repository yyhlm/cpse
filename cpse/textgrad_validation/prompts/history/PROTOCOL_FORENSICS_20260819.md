# `v2-goldnorm-focus-sol-two-stage-03` 协议核验

核验对象：主运行 `test/results/v2-goldnorm-focus-sol-two-stage-03` 首次启动时使用的固定系统提示词与核心执行逻辑。

## 证据强度

- **确定**：`meta/config.snapshot.json` 固定了提示词路径、模型、训练文档、`include_pdf=true`、`include_error_locations=false`、`optimization_mode=two_stage_alternating_schema_description`。
- **确定**：会话中的实际 `apply_patch` / `Get-Content` 时间线可以恢复三个系统提示词的版本边界。
- **限制**：该旧 run 创建时尚未保存 `meta/frozen_protocol/`，manifest 也没有系统提示词哈希；仓库没有对应日期的代码提交。因此不能声称已经恢复整个 Python 源码的逐字快照。

## 固定系统提示词

### Evidence

主运行使用 `evidence_system.v2_identity_dedup.txt`，SHA-256 为 `a0e5e8c189c02eb44a03faad9bbf38d2c9de85e6b64f6b9e6284180772637df7`。该结论同时由补丁时间线和历史 extraction cache 哈希支持。

### Resolve

主运行时已经具备以下规则：

- stage 2 接收 PDF、候选 schema、`covered_identities` 与 stage-1 identity stubs；
- `manifest_id` 仅用于内部路由，不进入最终 JSON；
- 每个 supplied manifest entry 生成一个完整聚合物记录并保持顺序；
- schema 是字段、拼写、类型和嵌套的 exhaustive allow-list；
- 标量字段不能输出范围对象；
- `文献信息` 不由 stage 2 输出，而从 stage 1 确定性合并；
- shared procedure 可以传播，但不得捏造步骤、用量或身份。

这与当前 `prompts/resolve_system.txt` 的基线文本一致；当前文件 SHA-256 为 `85ea523d3b1f81b40f0fa022f5886365b060849863a7d567a6f555e13281daa2`。2026-08-22 曾临时加入更强的 identity-boundary 规则，2026-08-23 撤回；该临时版本不属于 2026-08-19 主运行。

### Schema-description patch

主运行使用本目录中的 `schema_description_system.v0_main_20260819.txt`。它只有三条约束：只返回 description patch、只能改既有 description 字符串、不得写入文档专属事实。当前文件中的第四条“MINIMAL and FOCUSED / at most 15 patches”直到 2026-08-23 才出现，因此不属于主运行。

重建版本按 LF 结尾计算的 SHA-256 为 `00f990fbe28f097e65e40e5d1aeda632ea5d403c0aafca014c36b8f750b7dc78`。

### Judge

主运行配置使用 `judge_with_pdf_system.txt`，PDF 参与争议、Gold 外内容和幻觉核验，`include_error_locations=false`。主实验时期已采用当前四分项语义评分框架：`document_sample=10`、`process=30`、`properties=50`、`characterization=10`，并使用 79/89/84 等总分上限以及 90–99/100 的严格门槛。

同时，主实验协议明确把 `CURRENT_ERROR` 和 `GENERAL_RULE` **都提供给优化器**：前者作为具体错误锚点，后者作为跨论文规则。它不是早期“只向优化器传 GENERAL_RULE”的版本。当前 Judge 文件 SHA-256 为 `65951b381172fa2421748882d7bea207392f098c5bcf7379a70ca2ae9d70a428`；2026-09-01 曾短暂切换为结构敏感版本，随后已撤回，不能把那次临时协议与主结果混用。

## 当时的核心代码行为

由 2026-08-18 至 2026-08-19 的补丁、测试和运行日志可确认：

1. stage 1 输出 `metadata_evidence + identity_manifest`；stage 2 不再重复输出文献信息。
2. manifest 默认按最多 5 个 entry 切成不重叠 batch。
3. stage 2 每批读取 PDF，并按 manifest entry 做完整记录 resolution。
4. stage 输出的 raw response 与 validation errors 已能在失败产物中落盘。
5. 可解析但最终 schema 不一致的 merged prediction 会保留、警告并继续交给 Judge；不可解析或无法完成最低限度合并的输出才失败。
6. 两阶段训练按 round 写 checkpoint，允许中断后恢复；训练文档并行，stage-2 batch 也已加入并行执行。
7. 冻结状态应从具有有效均分的 joint candidates 中选择最高训练均分，而不是机械选择最后一轮。主 run 后续发生过恢复和 optimized 臂补跑，因此最终目录是“同一冻结训练状态上的恢复后完整结果”，并非一次不中断进程的原样目录。

## 后续代码不属于原始主运行

以下能力是之后加入的，不能反推为 2026-08-19 已存在：四变量 coverage-plan/evidence-routing、Judge-only、多模型复评、外部 OPRO/GEPA/MIPROv2 基线、确定性 slot metric、execution-only ablation、run 级完整 `frozen_protocol` 快照，以及 2026-09-03/04 对 manifest 冲突与宽容合并的进一步修订。

## 关于“后面的实验是否都使用更新后版本”

不能笼统回答“是”。每次进程启动时会读取当时磁盘上的固定提示词；旧版共享缓存则由请求指纹决定是否复用。2026-08-22 至撤回前启动的运行可能使用实验性 identity-boundary 版本；2026-08-23 之后的新运行使用撤回后的语义版本；schema-patch 的稀疏限制也从 2026-08-23 起才生效。只有新增 `meta/frozen_protocol/index.json` 之后创建的 run，才能直接逐文件核对实际字节。
