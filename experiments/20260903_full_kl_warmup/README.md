# 2026-09-03：10-step warmup 全词表 KL 实验

## 实验任务

在每次真实 update 数据量固定的条件下，测量 N=4、N=8 的相邻模型 KL、相对 rollout
anchor 的累计 KL、gradient norm、能力、token 数和耗时；随后补跑 N=16 作为大窗口
扩展。三组各执行 96 次真实 update。

## 与其他实验的关系

本实验承接[首轮 Staleness Pilot](../20260902_staleness_pilot/README.md)，把 sampled-token
proxy 扩展为固定真实 rollout 前缀上的全词表 KL。后续
[无 warmup N=4/8 实验](../20260909_full_kl_no_warmup_n4_n8/README.md)去掉前 10 次更新的
warmup，并改为 N=4、N=8 各运行 24 个完整 rollout 周期，因此两次实验的学习率和训练
预算口径不能混用。

## 实验配置

| 项目 | 设置 |
| --- | --- |
| 模型 / 算法 / 数据 | Qwen3-8B-Base，全参数 GRPO，DAPO-Math-17K |
| N / 真实更新数 | N=4/8/16；各 96 次 update |
| Prompt mini-batch / responses / PPO epochs | `256 / 8 / 1` |
| 学习率 | 峰值 `1e-6`；前 10 次真实 update warmup |
| Seed / Runtime | 1；4×A800-80GB、TP=2、`sis_offload` |
| 能力评测 | MATH500 sampled Avg@1；update 0/48/96 |
| 全词表 KL | 真实 rollout 前缀；每轮固定 anchor 与前缀；旧模型到新模型方向 |

## 启动、时间与状态

| 分支 | Slurm job | 启动时间 | 运行结果 |
| --- | ---: | --- | --- |
| N=4 | 154027 | 2026-09-03 20:08 | 96 updates 完成；`COMPLETED` |
| N=8 | 154028 | 2026-09-04 15:32 | 96 updates 完成；`COMPLETED` |
| N=16 | 157388 | 2026-09-08 | 96 updates 与 0/48/96 评测完成；旧终点比较脚本随后拒绝 N=16，Slurm 最终标记失败 |

N=16 的收尾错误不影响已归档训练统计，但它没有生成与 N=4/8 相同的终点双向 KL
比较。本地归档核验日期为 2026-09-22。

## 产物入口

- [实验结果](docs/EXPERIMENT_RECORD.md)
- [原始产物](raw/)
- [派生表](tables/)
- [确认图表](figures/)
- [汇总与绘图脚本](scripts/)
- [返回实验索引](../README.md)
