# 2026-09-21：N=16 logits 几何大 age 校验

## 实验任务与关系

把连续复用窗口扩大到 age 16，检验 N=4 上观察到的有限差分 KL 近似和负方向对齐是否
仍成立，以及近似误差是否随 age 系统增大。

本实验是 [logits 几何 KL 校验实验族](../README.md)的 N=16 子实验。它与其他三组使用
相同实现和 96 次真实 update，但只有 6 个 rollout/anchor，因此单独记录并明确较小的
每-age 重复数。

## 配置

共同设置见[实验族 README](../README.md)。本组为 `N=16`、
`train_batch_size=4096` prompt groups、6 rollouts / 96 updates、seed 1；MATH500 sampled
Avg@1 在 update 0/32/64/96 评测。

## 启动、时间与状态

| 项目 | 内容 |
| --- | --- |
| 提交脚本 | [`../scripts/submit_n16_n32_geometry.sh`](../scripts/submit_n16_n32_geometry.sh) |
| Slurm job | 167220 |
| 运行时间 | 2026-09-21 05:46:28—23:15:01 |
| 状态 | `COMPLETED / 0:0`；2026-09-23 已核验 |
| 远端输出 | `outputs/fisher_alignment_large_n/.../20260921-054628_job167220_g4_tp2_seed1/` |

## 产物

- 原始快照：`raw/job167220/`
- [确认后的分析记录](docs/EXPERIMENT_RECORD.md)
- 派生表：`tables/`
- 图表：`figures/`
- 四组统一分析脚本：[`../scripts/analyze_kl_approximation.py`](../scripts/analyze_kl_approximation.py)
