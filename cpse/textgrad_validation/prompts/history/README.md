# 固定系统提示词历史恢复

这些文件由本机 Codex 会话中保存的 `apply_patch` 记录重建，用于恢复旧实验协议；不会自动替换当前活动提示词。

- `v0_initial`：最初版本。
- `v1_metadata_schema`：2026-08-18 将 `metadata_evidence` 约束到 supplied 文献信息 schema。
- `v2_identity_dedup`：2026-08-18 增加重复身份与禁止虚构唯一 ID 的约束。
- `v3_identity_boundary_experimental`：2026-08-22 临时增加阶段/形态边界规则；该规则于 2026-08-23 撤回。

## 已核验 SHA-256

| 版本 | SHA-256 |
| --- | --- |
| `v0_initial` | `f26776227fb7f4c33c202ca7f54f1b84d29d360120b9cf5ccd891f13f67ffcf6` |
| `v1_metadata_schema` | `4be27d558c11ebe4074edb7e690f52d168301f6f5f0202728af933091836fdec` |
| `v2_identity_dedup` | `a0e5e8c189c02eb44a03faad9bbf38d2c9de85e6b64f6b9e6284180772637df7` |
| `v3_identity_boundary_experimental` | `fca90ce221fa2f903552e84af4977b64fbb0f270e3f75092842b2a4130b9be6b` |

`v2-goldnorm-focus-sol-two-stage-03` 首次运行于 2026-08-19；按补丁时间线，它使用的是 `v2_identity_dedup`。该版本的哈希也与历史 extraction cache 中出现的 `a0e5e8...` 完全一致。当前活动文件的哈希为 `43db15...`；其文本主体与 v2 相同，但在撤回实验性段落时留下了额外空行，因此原始字节哈希不同。要复现旧 run，应使用归档的 v2 文件，而不是当前文件的视觉等价副本。

恢复依据：`C:\Users\28172\.codex\sessions\2026\07\16\rollout-2026-07-16T23-23-51-019f6b86-d703-78e2-957d-5658f2bc2cbf.jsonl` 中第 3949、5954、7583、7673 条补丁记录。精确复现实验前，必须以历史 artifact 中记录的 SHA-256 匹配对应文件；不能仅凭文件名推断。

主两阶段实验其余协议的核验结论见 `PROTOCOL_FORENSICS_20260819.md`。其中 `schema_description_system.v0_main_20260819.txt` 是已逐字重建的主实验 schema-patch 固定系统提示词；resolve、Judge 与代码状态按证据强度分别记录，未把当前代码直接冒充成历史源码快照。

2026-09-04 恢复主提示词目录到 2026-08-19 可证实协议前，将后期加入稀疏补丁约束的版本保存为 `schema_description_system.v1_sparse_pre_restore.txt`。当前 `evidence_system.txt` 恢复为 `evidence_system.v2_identity_dedup.txt` 的字节内容，`resolve_system.txt` 与 `judge_with_pdf_system.txt` 经核验无需修改。
