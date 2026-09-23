# 2026-09-19：N=4 logits 几何主校验

## 实验任务与关系

在四步复用窗口内检验有限 log-prob 差分二次量对相邻/累计全词表 KL、精确三点交叉项
和归一化方向量的近似程度，并记录 AdamW `update_norm` 供补充换算分析使用。

本实验是 [logits 几何 KL 校验实验族](../README.md)的主子实验；N=1 提供边界控制，
N=16/32 检验更大 age。4-update probe 只验证工程链路，不混入正式统计。

## 配置

共同设置见[实验族 README](../README.md)。正式组为 `N=4`、
`train_batch_size=1024` prompt groups、24 rollouts / 96 updates、seed 1；MATH500 benchmark
关闭，使用 16 个训练集 prompt 的辅助 sampled Avg@1 在 update 48/96 检查运行状态。
probe 使用相同 N 和 runtime，只运行 1 rollout / 4 updates。

## 启动、时间与状态

| 运行 | Slurm job | 时间 | 状态 |
| --- | ---: | --- | --- |
| 工程 probe | 166277 | 2026-09-20 01:39:35—02:28:20 | `COMPLETED / 0:0` |
| 正式运行 | 166278 | 2026-09-20 02:28:39—23:07:50 | `COMPLETED / 0:0` |

提交脚本为 [`scripts/submit_n4_fisher_alignment.sh`](scripts/submit_n4_fisher_alignment.sh)。
截至 2026-09-23，两项状态及正式运行 96 条逐 update 记录均已核验。

## 产物

- 正式原始快照：`raw/job166278/`
- 工程 probe：`raw/probe_job166277/`
- [确认后的分析记录](docs/EXPERIMENT_RECORD.md)
- 派生表：`tables/`
- 图表：`figures/`
- N=4 补充分析脚本：[`scripts/analyze_results.py`](scripts/analyze_results.py) 与
  [`scripts/plot_formula_validation.py`](scripts/plot_formula_validation.py)
- 四组统一分析脚本：[`../scripts/analyze_kl_approximation.py`](../scripts/analyze_kl_approximation.py)
