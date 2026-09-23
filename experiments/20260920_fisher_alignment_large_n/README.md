# 2026-09-20：N=16/32 Fisher alignment 压力测试

## 实验任务

保持模型、优化器、每次 update 数据量和全词表测量口径不变，把连续复用窗口扩展到
N=16 和 N=32，采集更大 age 与累计 KL 范围下的有限差分二次量、`rho_eff`、实际
更新范数和能力评测原始数据。

## 与其他实验的关系

本实验直接扩展[N=4 Fisher alignment](../20260919_fisher_alignment_n4/README.md)的
测量范围，用于检查 N=4 上贴近精确 KL 的有限差分近似在更大位移下是否仍成立。
[N=1 控制实验](../20260921_fisher_alignment_n1_control/README.md)使用相同测量实现提供
无轮内交叉项的同步刷新控制。

## 实验配置

| 项目 | N=16 | N=32 |
| --- | ---: | ---: |
| 真实 optimizer updates | 96 | 96 |
| 完整 rollout 周期 | 6 | 3 |
| prompt groups / rollout | 4096 | 8192 |
| trajectories / update | 2048 | 2048 |
| MATH500 评测 update | 0/32/64/96 | 0/32/64/96 |

共同配置为 Qwen3-8B-Base、全参数 GRPO、seed 1、固定学习率 `1e-6`、无 warmup、
`ppo_mini_batch_size=256`、`rollout.n=8`、`ppo_epochs=1`、4×A800、TP=2、
`sis_offload`。全词表测量每轮使用 64 prompts、每个所选 response 最多 8 个位置。

## 启动、时间与状态

两组使用 [`submit_n16_n32_geometry.sh`](scripts/submit_n16_n32_geometry.sh)独立提交，无
作业依赖。

| N | Slurm job | 完成状态 | 耗时 |
| ---: | ---: | --- | ---: |
| 16 | 167220 | `COMPLETED / 0:0` | 17:28:33 |
| 32 | 167221 | `COMPLETED / 0:0` | 17:32:27 |

逐 update JSONL、0/32/64/96 逐题评测、TensorBoard event 和 Slurm 日志已归档。
目前只完成原始产物归档，尚未提出或确认结果分析，因此不创建
`docs/EXPERIMENT_RECORD.md`。

## 产物入口

- [原始产物](raw/)
- [提交脚本](scripts/submit_n16_n32_geometry.sh)
- [返回实验索引](../README.md)
