# 2026-10-06：正常训练 batch 的 Fisher K/B 短程多轮实验族

## 任务与关系

用户指定 N=4 执行 5 轮 rollout，N=8 执行 3 轮 rollout，观察多个 rollout anchor
上的 K/B 测量。每次 update 保持正常训练的数据量和更新配方，不沿用 190665 的
mini-batch=8。两组分别执行 20、24 次真实 optimizer update，训练量不相等。

沿用 [2026-10-05 实验族](../20261005_fisher_kqb_n4_n8/README.md) 的模型、资产、
训练配方及分层 Fisher 测量；前置 [190665 工程验收](../20261005_fisher_kqb_n4_n8/n8/README.md#190665-终态与工程验收)
已通过。本次将“等 96 updates”改为用户指定的短程多轮设置，另建实验族，不覆盖原设置。

## 共同配置与口径

- Qwen3-8B-Base，全参数 GRPO，DAPO-Math-17K 的 SIS 预处理数据，seed=1；从同一
  初始模型分别训练，`resume_mode=disable`，运行中保留各自 optimizer/scheduler 状态。
- 每次真实 update 使用 256 prompt groups × 8 responses = 2048 trajectories，
  `ppo_epochs=1`；rollout batch 为 256×N prompts，各 mini-batch 分别更新一次。
- 固定 LR=1e-6，warmup=0，长度 1024/3072，right 截断；KL loss coefficient=1e-3，
  clip low/high=0.2/0.2，entropy=0。训练生成 temperature=1、top_p=1、top_k=-1。
- 每组 4×A800-80GB、TP=2、`sis_offload`、legacy FSDP+vLLM；每卡 actor/infer
  dynamic batch=16384 tokens，vLLM utilization=0.7、batched tokens=8192、seqs=512；
  8 个独立 Ray Math-Verify reward workers。
- 本次不测 MATH500 或其他能力验证：`ENABLE_BENCHMARK_EVAL=0`、
  `VAL_BEFORE_TRAIN=false`、`TEST_FREQ=-1`，包括终点也不评测。训练 reward 正常计算。
- 每轮固定 64 prompts，每个所选 response 最多 8 个有效位置；prompt 等权再按位置
  等权，全词表。K/B 使用同一 FP32 eval 测量模型和 q_t，KL/F_z 为 FP64。
  `select_rollout_positions` 及保存前缀、权重的逻辑沿用旧实现：每个入选 prompt
  随机选一条有效 response，从有效 response token 中无放回选最多 8 个位置；使用
  每个 token 前一位置的 logits。保留至最晚测量位置的全部因果前缀，仅去掉 padding
  和无关的未来后缀，没有缩短所测位置之前的上下文。
- 保留已通过的分层 JVP/VJP、层输入 CPU 保存和单层激活 offload。每轮 anchor 估计
  λ，每个 age 测量 K、实际累计位移及 B；不计算逐 age 的 Q，仍用 FVP
  做 anchor 谱估计。Power Iteration residual≤1e-3，cap=200。
  单次测量 timeout 从 6 小时改为 8 小时，进程组 timeout 为 10 小时；收敛门槛不变。
- `SAVE_FREQ=-1`，不持久保存训练后 HF 权重或 Adam checkpoint；运行中 Adam 状态
  不重置。临时 anchor/current/cache 放 job-local scratch，轮末清理；复用已有基座。

```text
Δ_(t,a) = θ_(t+a) - θ_t
K_t(a) = E_qt KL(π_t || π_(t+a))
B_hat_t(a) = 1/2 λ_hat_max(F_t) ||Δ_(t,a)||²
```

B_hat 是 anchor 处谱估计构造的候选界，不是有限位移真实 KL 的严格上界。
`FULL_KL_ACTOR_MEASUREMENT=0` 暂停旧 BF16 actor 的相邻/累计 KL，
`HVP_COMPUTE_QUADRATIC=0` 暂停逐 age 的 Q；K 只采用 FP32 模型的
`hvp_cumulative_kl`，不与历史 BF16 actor K 混用。测量在 update 之间暂停训练并借用同四卡，
不改变 GRPO loss 或 optimizer 更新，但不能把带测量的总耗时当作原训练配方吞吐。

终点 20/24 不构成等训练预算对照。本次只采集测量数据；正式分析问题、指标和
图表另行确认，不创建分析 Record。

## 子实验与入口

| 设置 | rollout / updates / trajectories | 入口 |
| --- | --- | --- |
| N=4 | 5 / 20 / 40960 | [n4/README](n4/README.md) |
| N=8 | 3 / 24 / 49152 | [n8/README](n8/README.md) |

共用 [提交脚本](scripts/submit_experiment.sh) 每次调用只提交一个设置，统一调用
`examples/dynamic_staleness/submit_slurm.sh anchor`。设置特有的 job、时间、状态及产物
只记录在各子实验 README；按用户要求直接提交，不建立定时监控。
