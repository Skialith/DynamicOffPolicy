# 2026-09-02：首轮动态 Staleness Pilot

## 实验任务

从同一个 update-100 checkpoint 分出 N=4、N=8、N=16 三条支线，各执行 96 次真实
optimizer update，观察增大 rollout 复用窗口后能力、sampled-token staleness、PPO
裁剪、训练信号和实际耗时的变化。

## 与其他实验的关系

这是本仓库第一轮系统性 staleness 对照。它使用 sampled-token proxy，且没有保留三个
终点权重，因而不能直接校准全词表 KL 或补算终点模型间 KL。后续
[10-step warmup 全词表 KL 实验](../20260903_full_kl_warmup/README.md)和
[无 warmup 全词表 KL 实验](../20260909_full_kl_no_warmup_n4_n8/README.md)针对这些限制
补充了固定口径的直接 KL 测量。

## 实验配置

| 项目 | 设置 |
| --- | --- |
| 模型 / 算法 / 数据 | Qwen3-8B-Base，全参数 GRPO，DAPO-Math-17K |
| 起点 / 终点 | 共享 update 100 checkpoint；三组均更新至 196 |
| N / 真实更新数 | N=4/8/16；各 96 次 update |
| 每次 update 数据量 | 256 prompt group × 8 responses；2048 trajectories |
| 学习率 / seed | `1e-6` / 1 |
| Runtime | 4×A800-80GB、TP=2、`sis_offload` |
| 能力评测 | 历史 MATH500 greedy Avg@1；共同检查点 116/132/148/164/180/196 |
| 权重 | 只保留共享 update-100 完整 checkpoint；三条 update-196 分支未保留终点权重 |

## 启动、时间与状态

| 分支 | Slurm job | 完成时间（集群时间） | 状态 |
| --- | ---: | --- | --- |
| N=4 | 151476 | 2026-09-01 19:10:13 | `COMPLETED / 0:0` |
| N=8 | 150042；与 anchor 串行 | 2026-09-01 17:21:33 | `COMPLETED / 0:0` |
| N=16 | 151477 | 2026-09-02 20:42:27 | `COMPLETED / 0:0` |

远端持久化根目录：

```text
/data/run01/scyb980/cyt/src/verl-staleness/outputs/dynamic_staleness/20260830-174008_job150042_g4_tp2_seed1
```

## 产物入口

- [实验结果](docs/EXPERIMENT_RECORD.md)
- [历史研究笔记与旧预案](docs/HISTORICAL_NOTES.md)
- [原始产物](raw/)
- [能力图](figures/math500_branches_by_optimizer_step.png)
- [整理后的评测点](tables/math500_greedy_avg1.csv)
- [绘图脚本](scripts/plot_math500_branches.py)
- [返回实验索引](../README.md)
