# 2026-10-05：等 96 次更新的 Fisher K/Q/B 实验族

## 任务与关系

用户确认两组均执行 96 次真实 optimizer update，N=4 为 24 个完整 rollout 周期，
N=8 为 12 个周期；每次更新的数据量相同。不是沿用旧的“等 24 轮”训练量比较。
同时测量固定 rollout anchor、固定真实前缀分布 q_t 上的三项：

```text
Δ_(t,a) = θ_(t+a) - θ_t
K_t(a) = E_qt KL(π_t || π_(t+a))
Q_t(a) = 1/2 Δ_(t,a)^T F_t Δ_(t,a)
B_hat_t(a) = 1/2 λ_hat_max(F_t) ||Δ_(t,a)||²
```

F_t 是 frozen-anchor、全词表 KL 在 θ_t 处的 Hessian/Fisher。每轮在更新前用精确
FVP 和幂迭代估计 λ，每个 age 再用实际累计位移方向做一次 FVP 得到 Q；不构造
或保存完整 Fisher，不使用差分。当前修复候选按层计算 J^T F_z Jv，与 anchor 处
的 KL HVP 数学上相同；原始双反向 HVP 保留为小模型数值参考。B_hat 是谱估计构造的候选界，
不是有限位移 KL 的严格上界，也不是已获得最大特征值上界证书。

沿用 [165 小模型验证](../20261004_kl_hvp_exact_vs_fsdp_fd/README.md) 的数学路线；
[8B 四卡短前缀探针](../20261004_exact_kl_hvp_8b_4gpu/README.md) 只验证了独立
模型的 HVP 和谱迭代。本实验新增真实训练权重/累计位移、真实 rollout 前缀、FSDP
offload 与独立测量模型间切换，必须先通过集成 probe。

## 共同口径与实现

- 两组同一 Qwen3-8B-Base 初始资产，seed=1，全参数 GRPO；不恢复 optimizer 状态。
- 每次更新固定 256 prompt groups × 8 responses，ppo_epochs=1；两组各 196608 条
  trajectory。rollout batch 分别 1024/2048 prompt groups，轨迹不反复训练。
- 固定 LR=1e-6、warmup=0；其他正式训练配方和 sampled MATH500 评测沿用共同基线。
- 每轮选择 64 个 prompt group，各选择一个有效 response 和最多 8 个有效位置，
  prompt 等权、再按有效位置等权；同一轮 K/Q/B 使用完全相同 q_t 和全词表。
- 三项均使用独立 FP32、eval、eager-attention、TF32 关闭的按层四卡测量模型；
  log-softmax/KL/F_z 计算 FP64，方向/Jv/Fv 为 FP32，向量标量乘积 FP32 后 FP64 累加。
  原有 BF16 训练侧 KL 字段保留，不与新 `hvp_*` 字段混作同精度测量。
  整网双反向的 GPU 图、CPU offload 图及 decoder 前向重算均在真实长前缀上 OOM
  （工程记录见 n8 README）。当前候选在独立非 FSDP 模型上逐层做参数/输入 JVP，
  得到全参数方向的 Jv，再乘 F_z，最后逐层重算并做普通 VJP 得到 J^T F_z Jv。
  切层后只保存 detached 的层输入到 CPU，VJP 最多保留一层的普通反传图，并原精度
  offload 该层 activations；不保留整网二阶图、不冻结 embedding/head/norm。
  参数方向通过 dot_h_l = D_h f_l dot_h_(l-1) + D_theta_l f_l v_l 传播，不是只对
  输入求导。逐层 forward AD 不经过 FSDP，也不依赖 FSDP 输入 JVP 补丁。
  anchor 零梯度校验同样逐层 VJP。第一轮 FVP 记录逐前缀/层的 CPU 保存图与 RSS。
  数学对象、q、完整前缀、全词表和精度不变；真实内存效果仍须 GPU probe 验收。
  逐层 AD 使用 PyTorch 2.8 的 [jvp](https://docs.pytorch.org/docs/2.8/generated/torch.func.jvp.html)
  与 [functional_call](https://docs.pytorch.org/docs/2.8/generated/torch.func.functional_call.html)；
  官方接口并不保证所有算子都有 forward AD，因此保留 CPU 与四卡逐参数对照门槛。
- 每轮只有 anchor λ；幂迭代残差 `||Fv-λv||/||Fv|| <= 1e-3` 后提前结束，
  200 次/单次测量 6 小时为异常限额，触顶未收敛则失败，不沿用不合格 λ。
- 每次 update norm 来自分片参数更新前后真实差值，累计 norm 来自 θ_(t+a)-θ_t；
  不是 raw gradient norm、不是各步 norm 求和，也不是 AdamW 位移重构。
- FSDP 只负责普通训练和 CPU full-state 导出；Fisher 导数在独立非 FSDP 进程执行。
  暂停训练并 offload actor/optimizer，再借用同四卡；此测量开销单独记录，不声称
  保持原基线吞吐。
  `FULL_KL_HVP=1` 仍是这套 K/Q/B 测量入口；`FULL_KL_JVP=0` 禁用的是旧 FSDP
  functional-JVP 路径，不表示独立逐层 backend 不能使用 forward AD。
- `SAVE_FREQ=-1`：不保存终点 HF 权重或 Adam 状态。临时 anchor/current 权重与概率
  cache 只放 `/tmp/ds-$SLURM_JOB_ID/tmp/fisher-hvp/`，每轮正常完成后清理；异常强杀时
  本地 scratch 可能需要检查，但不会写入持久权重目录。基座资产不改写。

每个独立设置的提交、时间、状态和产物只记在子实验 README。当前不建立正式分析
记录、不拟合或绘图，数据完成后再确认分析方案。

## 设置入口

| 设置 | 真实 updates / rollout 周期 | 入口 |
| --- | --- | --- |
| N=4 正式实验 | 96 / 24 | [n4/README](n4/README.md) |
| N=8 正式实验、前置集成 probe | 96 / 12；probe 为 8 / 1 | [n8/README](n8/README.md) |

共用提交脚本：[submit_experiment.sh](scripts/submit_experiment.sh)。
统一 Slurm 入口仍为 `examples/dynamic_staleness/submit_slurm.sh anchor`。
正式作业以 `afterok:<probe-job-id>` 排队：只有 probe 训练结束且逐步测量验收通过
才可启动。源代码版本、作业状态与依赖核验记入各子实验。
