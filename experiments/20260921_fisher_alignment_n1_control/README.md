# 2026-09-21：N=1 无交叉项同步控制

## 实验任务

每轮 rollout 后只执行一次 optimizer update，使训练数据使用前始终为 `policy_age=0`，
验证 `K(1)=d(1)`、累计方向与交叉项为零，以及相同训练预算下的同步刷新能力基线。

## 与其他实验的关系

本实验复用[N=4 Fisher alignment](../20260919_fisher_alignment_n4/README.md)的测量实现，
但消除了连续多次更新形成的轮内交叉项；它与
[N=16/32 压力测试](../20260920_fisher_alignment_large_n/README.md)共同覆盖 N=1、4、16、32
的测量范围。它仍可能包含 vLLM rollout 与训练侧模型的引擎 gap，但 full-vocabulary KL
使用训练侧 anchor，不把该 gap 混入 `K(1)`。

## 实验配置

| 项目 | 设置 |
| --- | --- |
| 模型 / 算法 / 数据 | Qwen3-8B-Base，全参数 GRPO，DAPO-Math-17K |
| N / 训练量 | N=1；96 rollouts / 96 updates |
| 每次 update 数据量 | 256 prompt group × 8 responses；2048 trajectories |
| 学习率 / warmup / seed | `1e-6 / 0 / 1` |
| Runtime | 4×A800、TP=2、`sis_offload` |
| 全词表 KL | 每轮 64 prompts，每个所选 response 最多 8 个位置 |
| 能力评测 | MATH500 sampled Avg@1；update 0/32/64/96 |
| 保存 | 不保存终点权重 |

## 启动、时间与状态

2026-09-21 使用 [`submit_n1_control.sh`](scripts/submit_n1_control.sh)提交 Slurm job
`167543`。远端运行已完成，由用户于 2026-09-23 确认。当前本地
[`raw/job167543/`](raw/job167543/)仍是 2026-09-22 的运行中快照，只覆盖 update 1–87
和评测 0/32/64，最终 96 条记录、update-96 评测和结束日志尚待同步。

目前尚未提出或确认结果分析，因此不创建 `docs/EXPERIMENT_RECORD.md`。

## 产物入口

- [当前原始快照](raw/job167543/)
- [提交脚本](scripts/submit_n1_control.sh)
- [统一同步脚本](../scripts/sync_remote_raw_artifacts.sh)
- [返回实验索引](../README.md)
