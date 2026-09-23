# 2026-09-09：无 warmup、等 24 轮 N=4/8 全词表 KL 实验

## 实验任务

在无 warmup、每次真实 update 数据量固定的条件下，让 N=4 和 N=8 各运行 24 个完整
rollout 周期，收集轮内相邻/累计全词表 KL、gradient norm、每轮能力评测和运行时间。
N=4 执行 96 次 update，N=8 执行 192 次 update。

## 与其他实验的关系

本实验修正[10-step warmup 实验](../20260903_full_kl_warmup/README.md)的学习率和轮数口径，
用于观察相同数量的轮内曲线，不用于直接声称 N=4 与 N=8 训练预算相等。
[Gradient/KL 回顾分析](../20260918_gradient_kl_validation/README.md)直接使用本实验的
`kl_updates.jsonl`，检验 gradient norm、相邻 KL 和累计 KL 的关系。

## 实验配置

| 项目 | 设置 |
| --- | --- |
| 模型 / 算法 / 数据 | Qwen3-8B-Base，全参数 GRPO，DAPO-Math-17K |
| Seed | 1 |
| 学习率 / warmup | `1e-6 / 0` |
| Prompt mini-batch / responses / PPO epochs | `256 / 8 / 1` |
| Rollout batch | `256*N` prompt group |
| Prompt / response 最大长度 | `1024 / 3072`，截断 `right` |
| KL coefficient / PPO clip / entropy | `1e-3 / 0.2 / 0` |
| Runtime | 4×A800-80GB、TP=2、`sis_offload` |
| 能力评测 | MATH500 sampled Avg@1；训练前及每轮结束 |
| 全词表 KL | 每轮 64 prompts，每个所选 response 最多 8 个有效位置 |
| 保存 | 终点 HF 权重；不保存 Adam 状态 |

锁定配置的提交脚本为
[`rerun_experiment2_20260909.sh`](scripts/rerun_experiment2_20260909.sh)。

## 启动、时间与状态

两组于 2026-09-09 22:59:22（北京时间）独立提交，互无依赖。

| 分支 | Slurm job | 更新 / rollout | 完成时间（北京时间） | 状态 |
| --- | ---: | ---: | --- | --- |
| N=4 | 158523 | 96 / 24 | 2026-09-11 18:51:27 | `COMPLETED / 0:0` |
| N=8 | 158524 | 192 / 24 | 2026-09-12 11:43:33 | `COMPLETED / 0:0` |

两组预定记录和终点 HF 权重均已生成。日志在最终评测后的退出阶段包含 DataLoader/
vLLM 报错，但 Slurm 成功且所需产物完整。

## 产物入口

- [实验结果](docs/EXPERIMENT_RECORD.md)
- [N=4 原始产物](raw/n4/)
- [N=8 原始产物](raw/n8/)
- [TensorBoard 事件](raw/tensorboard/)
- [Slurm 日志](raw/slurm/)
- [确认图表](figures/)
- [分析与绘图脚本](scripts/)
- [返回实验索引](../README.md)
