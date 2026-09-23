# 2026-09-19：N=4 严格 frozen-Fisher JVP 校准

## 实验任务

固定每轮 rollout anchor、真实 rollout 前缀分布和全词表权重，通过参数 JVP 测量
`Delta^T F0 Delta / 2`、`u^T F0 u / 2` 与 `Delta^T F0 u`，再与精确 KL 比较有限步
二阶余项。每步只保存约化标量，不保存完整参数方向、JVP 向量或 Fisher 矩阵。

## 与其他实验的关系

[logits 几何 KL 校验实验](../20260919_logits_kl_geometry_validation/README.md)使用有限 log-prob
差分代理输出几何，不能等同于严格的参数空间 JVP。本实验专门校准这一解释边界，不用于
选择 KL 预算、最大 N 或 controller。

## 实验配置

| 项目 | 设置 |
| --- | --- |
| 共同基线 | Qwen3-8B-Base、全参数 GRPO、固定 `1e-6`、无 warmup、4×A800、TP=2 |
| N | 4 |
| Probe | 1 rollout / 4 updates；8 prompts，只作工程校验 |
| 正式校准 | 10 rollouts / 40 updates |
| 测量上下文 | 每轮 32 prompts，每个所选 response 最多 8 个位置 |
| 保存 | JVP/Fisher 约化标量和测量耗时；不保存完整方向或矩阵 |

## 启动、时间与状态

提交脚本为 [`submit_n4_jvp_calibration.sh`](scripts/submit_n4_jvp_calibration.sh)。正式任务
依赖 8B probe，probe 依赖 FSDP-JVP 兼容性检查。2026-09-23 远端核验只发现
`fsdp-jvp-compat` job 166375 `FAILED` 和 job 166912 `CANCELLED`，未发现
`outputs/fisher_jvp_n4/` 正式输出。因此该实验尚未形成可同步的正式原始数据，不能标记
为运行完成。

本实验尚未提出或确认结果分析，因此不创建 `docs/EXPERIMENT_RECORD.md`。

## 产物入口

- [提交脚本](scripts/submit_n4_jvp_calibration.sh)
- [返回实验索引](../README.md)
