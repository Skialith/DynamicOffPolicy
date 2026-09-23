# 2026-09-19：N=4 Fisher alignment 测量

## 实验任务

在每轮固定 anchor 与真实 rollout 前缀上，同时记录 AdamW `update_norm`、KL 三点交叉项
和固定-anchor有限 log-prob 差分 Fisher 量，检验 `rho_eff` 是否对应输出几何中的方向
抵消，并比较 gradient norm 与实际更新范数的换算稳定性。

## 与其他实验的关系

[Gradient/KL 回顾分析](../20260918_gradient_kl_validation/README.md)发现累计 KL 不能由相邻
KL 直接相加，但旧日志无法判断负残差是否对应同一固定-anchor几何中的方向抵消。本实验
沿用[无 warmup N=4/8 实验](../20260909_full_kl_no_warmup_n4_n8/README.md)的 N=4 配方，
新增有限差分几何和实际更新范数测量。

## 实验配置

| 项目 | 设置 |
| --- | --- |
| 模型 / 算法 / 数据 | Qwen3-8B-Base，全参数 GRPO，DAPO-Math-17K |
| N / 正式训练量 | N=4；24 rollouts / 96 updates |
| 每次 update 数据量 | 256 prompt group × 8 responses；`ppo_epochs=1` |
| 学习率 / warmup / seed | `1e-6 / 0 / 1` |
| Runtime | 4×A800-80GB、TP=2、`sis_offload` |
| 全词表测量 | 每轮 64 prompts，每个所选 response 最多 8 个有效位置 |
| 能力评测 / 权重 | 关闭；本实验只验证动力学公式 |

## 启动、时间与状态

正式作业依赖 4-update integration probe 成功后启动。

| 用途 | Slurm job | 状态 | 耗时 | 记录 |
| --- | ---: | --- | ---: | ---: |
| integration probe | 166277 | `COMPLETED / 0:0` | 00:48:45 | 4/4 |
| 正式 N=4 | 166278 | `COMPLETED / 0:0` | 20:39:11 | 96/96 |

提交脚本：[`submit_n4_fisher_alignment.sh`](scripts/submit_n4_fisher_alignment.sh)。
远端正式输出位于 `outputs/fisher_alignment_n4/20260919-182654/`。

## 产物入口

- [实验结果](docs/EXPERIMENT_RECORD.md)
- [原始产物](raw/)
- [派生表](tables/)
- [确认图表](figures/)
- [分析与绘图脚本](scripts/)
- [返回实验索引](../README.md)
