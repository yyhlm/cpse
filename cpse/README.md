# CPSE

`cpse/` 是随论文发布的实验源码包：PDF→JSON 的科学文献结构化抽取，用 TextGrad 在少量标注下自动适应抽取提示词与 schema 字段描述（结构契约保持），并支持身份先行的两阶段执行以缓解高实体密度长文档的记录折叠。

发布的 `cpse/` 只含实现、提示词、schema 与脱敏配置模板。**不含** PDF、Gold 标注、运行结果、缓存、密钥、端点 URL 与代理设置。

## 快速开始

1. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```
2. 准备数据集：把已授权的 PDF/Gold 对放到 `cpse/data/`。每个数据 stem 需同时有 `.pdf` 与 `.json`（Gold）。
3. 配置：复制 `cpse/textgrad_validation/config.example.yaml` 为工作目录下的 `config.yaml`，填入你自己的模型名与端点；密钥放入 `.env`（`TEXTGRAD_EXTRACTOR_API_KEY` 等，配置只引用环境变量名，不写真实密钥）。
4. 运行（示例，改为你的 `config.yaml` 与 run-id）
   ```bash
   python -m cpse.textgrad_validation --config config.yaml --run-id my-run-01
   ```
   先用 `--smoke --blind-doc <doc>` 冒烟验证接口、密钥与产物目录。

## 配置与模式

`config.example.yaml` 的 `optimization_mode` 可选：

- `single`：只优化抽取提示词；
- `description_only`：只优化 schema 字段描述（冻结抽取提示词）；
- `alternating_schema_description`：抽取提示词 + 字段描述联合优化；
- `two_stage_alternating_schema_description`：联合优化 + 身份先行的两阶段执行（高实体密度长文档）。

`cache.enabled` 为 `false` 时每次评估都强制真实调用模型，用于公平对比；`true` 可加快调试。`max_iterations` 是同 run-id 续跑时唯一可安全调整的字段。

## 复现边界

- **金标准未随包分发**，请使用已授权副本，并保持与论文一致的 `train_ids` / `train_count` / `evaluation.include_pdf`。
- 论文中结论以盲测集内同一文档的配对差值（`optimized_score - baseline_score`）为准，训练分只用于选择候选。
- 两阶段契约：stage 1 只产出元数据证据与 identity 清单；stage 2 每批处理至多 5 个 manifest 条目并绑定证据；最终经确定性合并与校验后输出 JSON。

## 产物

每个 run 的结果写在 `output_root/<run-id>/`，含训练轮次、盲测配对、逐文档评分、checkpoint 与统计。详细配置/协议/产物说明见 `textgrad_validation/` 内注释。