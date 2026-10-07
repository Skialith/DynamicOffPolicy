# 2026-09-21：N=32 logits 几何最大 age 校验

## 实验任务与关系

把连续复用窗口扩大到 age 32，检验有限差分 KL 几何在本实验已测最大 age 上是否出现
明显失效，并观察累计变化中的方向抵消。

本实验是 [logits 几何 KL 校验实验族](../README.md)的 N=32 子实验。它与其他三组使用
相同实现和 96 次真实 update，但只有 3 个 rollout/anchor；逐 age 结果因此只是三个
anchor 的描述性统计，独立记录且不与其他 N 合并推断。

## 配置

共同设置见[实验族 README](../README.md)。本组为 `N=32`、
`train_batch_size=8192` prompt groups、3 rollouts / 96 updates、seed 1；MATH500 sampled
Avg@1 在 update 0/32/64/96 评测。

## 启动、时间与状态

| 项目 | 内容 |
| --- | --- |
| 历史提交脚本 | `../scripts/submit_n16_n32_geometry.sh`（已退役，见[取回方式](../README.md#历史脚本复现)） |
| Slurm job | 167221 |
| 运行时间 | 2026-09-21 06:58:49—2026-09-22 00:31:16 |
| 状态 | `COMPLETED / 0:0`；2026-09-23 已核验 |
| 远端输出 | `outputs/fisher_alignment_large_n/.../20260921-065849_job167221_g4_tp2_seed1/` |

## 产物

- 原始快照：`raw/job167221/`
- [确认后的分析记录](docs/EXPERIMENT_RECORD.md)
- 派生表：`tables/`
- 图表：`figures/`
- 历史统一脚本：`../scripts/analyze_kl_approximation.py`；按[历史脚本复现](../README.md#历史脚本复现)取回
