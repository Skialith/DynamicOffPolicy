# 2026-09-21：N=1 logits 几何边界控制

## 实验任务与关系

在每次 update 后立即刷新 rollout，验证 age=1 时相邻 KL 与累计 KL、有限差分 step 与
current 二次量应重合，并确认没有既有累计方向时交叉项和方向量的边界处理。

本实验是 [logits 几何 KL 校验实验族](../README.md)的 N=1 子实验。它与 N=4/16/32
使用同一测量实现，但独立记录，因为它有 96 个不同 anchor，且第 3、4 个方向比较在
数学上不适用。

## 配置

共同模型、优化器、单次 update 数据量和测量设置见[实验族 README](../README.md)。本组
设置为 `N=1`、`train_batch_size=256` prompt groups、96 rollouts / 96 updates、seed 1；
MATH500 sampled Avg@1 在 update 0/32/64/96 评测。

## 启动、时间与状态

| 项目 | 内容 |
| --- | --- |
| 提交脚本 | [`scripts/submit_n1_control.sh`](scripts/submit_n1_control.sh) |
| Slurm job | 167543 |
| 运行时间 | 2026-09-21 23:15:10—2026-09-23 00:07:20 |
| 状态 | `COMPLETED / 0:0`；2026-09-23 已核验 |
| 远端输出 | `outputs/fisher_alignment_n1_control/.../20260921-231510_job167543_g4_tp2_seed1/` |

## 产物

- 原始快照：`raw/job167543/`
- [确认后的分析记录](docs/EXPERIMENT_RECORD.md)
- 派生表：`tables/`
- 图表：`figures/`
- N=1 专项边界检查脚本：[`scripts/analyze_n1_control.py`](scripts/analyze_n1_control.py)
- 四组统一分析脚本：[`../scripts/analyze_kl_approximation.py`](../scripts/analyze_kl_approximation.py)
