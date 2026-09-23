# 2026-09-19：N=1/4/16/32 logits 几何 KL 校验

## 实验任务

在每轮固定训练侧 anchor 和真实 rollout 前缀上，记录精确全词表 KL、KL 三点交叉项、
固定-anchor有限 log-prob 差分二次量、AdamW `update_norm` 与 gradient norm，检验这些
logits/log-prob 几何量在不同复用窗口下能否重构或逼近相邻与累计 KL。

N=1、4、16、32 使用同一测量实现和共同训练口径，只改变 rollout 支持的连续更新次数，
因此归为同一实验的四个运行设置：N=1 是无轮内交叉项控制，N=4 是主校验，N=16/32
用于扩大 age 和累计位移范围。这里的“校验”使用已经观测到的各步 logits，是回顾性
几何验证，不等于只用前四步信息预测未来 KL。

## 与其他实验的关系

[Gradient/KL 回顾分析](../20260918_gradient_kl_validation/README.md)发现累计 KL 不能由相邻
KL 直接相加，但旧日志无法判断负残差是否对应固定-anchor输出几何中的方向抵消。本实验
补充有限 log-prob 差分几何和实际更新范数。它沿用
[无 warmup N=4/8 实验](../20260909_full_kl_no_warmup_n4_n8/README.md)的模型、优化器和
单次 update 数据量。

[严格 frozen-Fisher JVP 校准](../20260919_fisher_jvp_n4/README.md)额外要求参数到 logits
的 JVP，测量对象和工程路径不同，仍作为独立实验保留。

## 共同配置

| 项目 | 设置 |
| --- | --- |
| 模型 / 算法 / 数据 | Qwen3-8B-Base，全参数 GRPO，DAPO-Math-17K |
| 每次 update 数据量 | 256 prompt group × 8 responses；2048 trajectories；`ppo_epochs=1` |
| 学习率 / warmup / seed | `1e-6 / 0 / 1` |
| Runtime | 4×A800-80GB、TP=2、`sis_offload` |
| 全词表测量 | 每轮 64 prompts，每个所选 response 最多 8 个有效位置 |
| 测量开关 | `FULL_KL_EXPERIMENT=1`、`FULL_KL_GEOMETRY=1` |
| 保存 | 不保存正式运行终点权重；`kl_contexts/*.pt` 不同步到本地 |

## 运行设置、时间与状态

| 设置 | 角色 | Rollouts / updates | 能力评测 update | Slurm job | 集群时间与状态 |
| --- | --- | ---: | --- | ---: | --- |
| N=1 | 无轮内交叉项控制 | 96 / 96 | 0/32/64/96 | 167543 | 2026-09-21 23:15:10—09-23 00:07:20；`COMPLETED / 0:0` |
| N=4 probe | 工程验收 | 1 / 4 | 关闭 | 166277 | 2026-09-20 01:39:35—02:28:20；`COMPLETED / 0:0` |
| N=4 | 主校验 | 24 / 96 | 关闭 | 166278 | 2026-09-20 02:28:39—23:07:50；`COMPLETED / 0:0` |
| N=16 | 大 age 压力设置 | 6 / 96 | 0/32/64/96 | 167220 | 2026-09-21 05:46:28—23:15:01；`COMPLETED / 0:0` |
| N=32 | 更大 age 压力设置 | 3 / 96 | 0/32/64/96 | 167221 | 2026-09-21 06:58:49—09-22 00:31:16；`COMPLETED / 0:0` |

对应提交脚本为 [`submit_n1_control.sh`](scripts/submit_n1_control.sh)、
[`submit_n4_fisher_alignment.sh`](scripts/submit_n4_fisher_alignment.sh)和
[`submit_n16_n32_geometry.sh`](scripts/submit_n16_n32_geometry.sh)。远端输出目录仍保留
作业提交时的 `fisher_alignment_*` 名称，不因本地文档合并而改名。

截至 2026-09-23，本地已按明确 job ID 同步四个正式设置的最终 96 条逐 update JSONL、
resolved config、run manifest、训练统计、已有逐题评测、TensorBoard event 和 Slurm
日志；N=4 probe 也一并归档。模型权重、optimizer checkpoint 与 `kl_contexts/*.pt`
按项目规范不复制到本地。

## 产物入口

- [N=4 已确认结果](docs/EXPERIMENT_RECORD.md)；其结论范围不自动扩展到 N=1/16/32
- 原始产物：`raw/n1_job167543/`、`raw/n4_job166278/`、
  `raw/n16_job167220/`、`raw/n32_job167221/`
- N=4 工程 probe：`raw/n4_probe_job166277/`
- [派生表](tables/)
- [N=4 确认图表](figures/)
- [分析、绘图、提交脚本](scripts/)
- [统一同步脚本](../scripts/sync_remote_raw_artifacts.sh)
- [返回实验索引](../README.md)
