# 动态 staleness 探究实验

这套代码实现同步、colocated 的“rollout 一次，顺序更新 N 次”，不是 rollout/train 并行的 AReaL 式异步训练。独立代码基线为官方 verl `v0.7.1`。

## 1. Pilot 设计

第一阶段只做单 seed 的探索性实验：

| phase | N | optimizer update 范围 | 新增 verl outer step | prompts / rollout |
|---|---:|---:|---:|---:|
| shared anchor | 4 | 0 → 100 | 25 | 1024 |
| branch N=4 | 4 | 100 → 196 | 24 | 1024 |
| branch N=8 | 8 | 100 → 196 | 12 | 2048 |
| branch N=16 | 16 | 100 → 196 | 6 | 4096 |

若 196 update 时趋势仍不清晰，各分支延长到 292 update；分别再增加 24、12、6 个 outer step。运行 `plan_experiment.py` 可重新计算并校验这些数字。

核心控制变量是 learner optimizer update：每次 update 始终使用 256 个 prompt group，每个 prompt 有 8 条 response，所以一次 update 是 2048 条 response trajectories。verl 的用户配置 `actor.ppo_mini_batch_size=256` 使用 prompt 单位；legacy FSDP worker 内部才乘以 `rollout.n` 并按 data-parallel size 分片。

共同 anchor 能低成本回答“训练到 update 100 后增大 N 是否立刻恶化”，但单一晚期分支不能单独证明“晚期比早期更能容忍”。若 pilot 有信号，再在 update 0/25/50 增加相同的 N 分支点并使用 2–3 个 seed。

## 2. 固定配置

- Qwen3-8B-Base、GRPO；题目来自 DAPO-Math-17K，但正式 Parquet 只保留 17,917 个唯一样本
- 训练与推理统一采用 SIS/Qwen-Math prompt：system 要求最终答案写入 `\\boxed{}`，user 仅放原题
- `math_sis` reward 要求存在 `\\boxed{}`，并用 Math-Verify 做数值与代数等价判断；reward 为 0/1
- reward 固定使用 `remote` manager 和 8 个独立 Ray worker。Math-Verify 的超时实现依赖
  `signal.alarm()`，不能放进 `dapo` manager 的线程池；后者会把正确答案静默记为 0
- learning rate `1e-6`，constant scheduler，warmup 10（与 SIS 8B GRPO recipe 对齐）
- mini prompt batch `256`，responses/prompt `8`，PPO epochs `1`
- prompt/response 最大长度 `1024/3072`
- symmetric clip `0.2/0.2`，KL loss coefficient `1e-3`，entropy coefficient `0`
- FSDP2 + vLLM、单机 colocated；本地基线为 2×A800-80GB、rollout TP=2
- vLLM `gpu_memory_utilization=0.7`
- `trainer.balance_batch=false` 与 actor `shuffle=false`，确保一个 global batch 被稳定地切成 N 个互斥 mini-batch
- legacy FSDP actor 的 dynamic batch 会在 DP process group 内同步 micro-batch 数。否则各 rank
  因 response 长度不同而执行不同次数的 forward，FSDP2 会在 `_ALLGATHER_BASE` 等待并触发 NCCL timeout

`gpu_memory_utilization` 是 vLLM 模型执行器可使用的显存比例，主要决定 KV cache/推理侧预算；它不是 FSDP 训练阶段的显存比例。colocated 模式会在训练更新期间 sleep rollout engine。若 0.7 启动 OOM，先降到 0.6；若稳定且 KV cache 吞吐受限，再考虑上调。

### 2.1 并行云执行参数与 SIS 对齐

公开的 SIS Qwen3-8B GRPO recipe 使用 actor/ref dynamic batch、每卡 token 上限 `16384`、
vLLM `max_num_batched_tokens=8192`（其 verl 默认值）和显式 `max_num_seqs=512`。早期资源
排查阶段，本项目曾将对应值保守设为 `8192/8192/4096/256`，目的是先缩小单次 kernel 和
显存峰值、定位首次 Adam OOM 与不同 rank 的 dynamic-batch forward 次数不一致；它们不是
根据实验目标选择的超参数。4 卡正式作业 149315 已证明流程稳定，但 actor MFU 约 12%，且
每张卡显存仍有较大余量，因此 A800 profile 现对齐为：

| 参数 | A800 默认值 | 作用 |
|---|---:|---|
| `ACTOR_MAX_TOKENS_PER_GPU` | 16384 | actor 每个 dynamic micro-batch 的 token 上限；越大则 forward/backward 次数通常越少，但 activation 峰值越高 |
| `INFER_MAX_TOKENS_PER_GPU` | 16384 | old/ref/rollout-log-prob 每个 dynamic micro-batch 的 token 上限；越大通常减少纯前向次数 |
| `ROLLOUT_MAX_BATCHED_TOKENS` | 8192 | vLLM 单个 scheduler iteration 可处理的 token 数；影响 prefill/decode 吞吐和临时显存 |
| `ROLLOUT_MAX_NUM_SEQS` | 512 | vLLM 同时调度的序列上限；影响并发度和 KV-cache 压力 |
| `GPU_MEMORY_UTILIZATION` | 0.7 | vLLM 执行器/KV cache 的显存预算比例，不控制训练 actor 的显存 |
| `UPDATE_WEIGHTS_BUCKET_MEGABYTES` | 3072 | FSDP 权重同步到 vLLM 时的分桶大小；大桶减少传输调用次数，但增加瞬时 staging 显存 |

前四项只是执行打包上限，不改变 `256 prompts/update × 8 responses`、prompt 顺序、loss、
学习率、`policy_age` 或 optimizer-step 口径；dynamic batch 的划分变化只可能带来常规的浮点
归约微差。anchor 与由其恢复的所有 branch 必须使用同一个 runtime profile 和同一组执行上限。

脚本提供三个显式 runtime profile：

- `memory_safe`（默认）：已验证的 FSDP2 `offload_policy=true`，parameter/gradient/optimizer
  都走 CPU offload；最稳，但 actor update 受 CPU 搬运限制。
- `sis_offload`：对齐 SIS recipe 的 legacy FSDP `param_offload=true + optimizer_offload=true`；
  actor update 时状态回到 GPU，预期更快，但需按 GPU 数验证第一次 Adam 峰值。
- `gpu_resident`：FSDP2 全 GPU 常驻；潜在最快、OOM 风险最高，只用于资源探针后再决定。

SIS 公开启动示例中的 warmup 10 和 `data.truncation=right` 已对齐，但该示例不是 Figure 3
的 N=4/8/16 复现实验脚本。论文 Table 4 明确给出 math GRPO 的 KL loss coefficient 为
`1e-3`、mini-batch size 为 256、global batch size 为 `256×N`，因此正式配置以论文表格为准。
公开脚本的 `batch_size=512` 与 `ppo_mini_batch_size=256` 实际只产生 2 次 mini-batch update，
即有效 N=2；脚本中的小写 `n=8` 是 responses/prompt，不是 staleness N。本实验为制造可控的
N 次连续 update，必须保持 `train_batch_size=256×N`，否则 N=4/8 的 `policy_age` 结构会被
破坏。SIS fork 额外支持
`use_policy_clip=false`，但当前独立基线 verl v0.7.1 没有该配置；本实验保留标准 GRPO 的
`0.2/0.2` clipping，同时继续记录落在 clip window 外的 token，不能用扩大 clip 阈值冒充
“关闭 clip”。评测仍保留 pilot 既定的 MATH500 greedy Avg@1，而不改成 SIS 的多 benchmark
采样评测，以免同时改变训练系统和能力曲线口径。

## 3. 指标口径

每个真实 optimizer update 都写入 `staleness_metrics.jsonl`，横轴不是 verl outer step，而是绝对 optimizer update。一次 sequential optimizer update 就是本文所说的一个 gradient step；verl 自身的 `global_step` 是一次 rollout 加随后 N 次 update 的 outer step。`policy_age=0..N-1` 表示该 mini-batch 相对本轮 rollout policy 已经过了多少次更新。

同一个统计量分三种口径，避免把实现 gap 误判成 staleness：

- `update_*`：当前 train-mode policy 对 rollout 后、任何 update 前预计算的同为 train-mode anchor。它隔离 sequential optimizer update 造成的漂移，是主分析口径。
- 无前缀：当前 train-mode policy 对 verl 为 PPO 重算的 eval-mode `old_log_probs`。这是 PPO surrogate 实际使用的 ratio/clip 口径。
- `rollout_*`：当前 train-mode policy 对 vLLM 采样时的 log probability。它包含推理/训练引擎 gap 和后续 optimizer drift，是端到端“陈旧 token”口径。

主要观察：

- `update_sampled_kl`：纯更新漂移的采样 `KL(anchor || current)`；可能受正负 log-ratio 抵消，不能单独使用。
- `log_ratio_abs_mean`、`sequence_abs_log_ratio_sum_mean`：不抵消的局部漂移，以及 SIS 理论里的 response 累计绝对 log-ratio 对应量。
- `ratio_p05/p95/p99`、`ratio_second_moment`：importance ratio 尾部和二阶矩。
- `clipfrac_low/high/total`：落在 PPO clip window 外的 token 比例。
- `pg_clipfrac`：结合 advantage 符号后，PPO surrogate 中真正触发 clipping 的 token 比例。
- `ess_fraction`：importance weights 的归一化 effective sample size，越低越危险。
- `grad_norm`、verl 自带的 reward 和 entropy：用于判断漂移是否已经转化为优化不稳或能力变化。

`age=0` 的 `update_ratio` 应为 1、`update_sampled_kl/update_clipfrac` 应为 0，这是每轮自检点。无前缀和 `rollout_*` 在 age 0 可以非零；它们分别量化 train/eval 数值 gap 和 vLLM/FSDP 引擎 gap。

为得到严格可比的 `update_*`，指标代码会在每轮 optimizer update 前额外做一次 train-mode anchor forward，并在 CPU 保存 token log probabilities；因此它会增加一次全 batch 前向和约 `4 bytes × response token 数` 的主存占用。这是 pilot 的有意测量成本。

### 3.1 Benchmark eval 口径

正式 anchor/branch 已启用在线 MATH500 Avg@1。训练与评测使用相同的 SIS `\\boxed{}` 输出协议；评测使用固定 500 题、greedy decoding（`temperature=0`、`n=1`），在每个完整的 rollout→N 次 update 周期结束后运行，不插入同一批 stale rollout 的 N 次连续更新中间。资源 probe 和 smoke 不做 benchmark eval。

SIS Figure 3 的横轴是 `Gradient Step`，即本项目的真实 `optimizer.step()`，不是 verl outer/global step。论文展示 Qwen3-8B-Base 在 N=4/8/16 下的 AIME24、AIME25 accuracy，并说明小数据集使用 Avg@32，但没有说明“每 10 步 eval”这一采样频率。因此不能简单设置 `trainer.test_freq=10`：它按 verl outer step 计数，在 N=4/8/16 下分别隔 40/80/160 次 optimizer update，曲线不可比。

正式 pilot 收集：

- `staleness_metrics.jsonl`：每个 optimizer update 一条。
- `eval_metrics.jsonl`：以绝对 `optimizer_step` 为键的 MATH500 accuracy/reward；恢复重算时同 step 自动覆盖。
- `eval_generations/optimizer_step_XXXX.jsonl`：500 道题的 prompt、生成答案、ground truth、accuracy 和 optimizer step，便于复核判分。

anchor 先评 update 0，之后得到 4、8、…、100；branch 不重复 update 100，N=4/8/16 分别得到 104、108、… / 108、116、… / 116、132、…、196。绘图统一使用 `eval_metrics.jsonl` 的绝对 optimizer step；三条 branch 在 116、132、148、164、180、196 严格对齐。后续与 SIS Figure 3 对齐的正式报告再补 AIME24/AIME25 Avg@32。

## 4. 环境和资产

当前首选 `/home/ymy/yes/envs/SkyRL`：它已有 torch 2.8、vLLM 0.11、Ray、FlashAttention、tensordict，并与 verl v0.7.1 的依赖范围相容。通过 `PYTHONPATH` 导入本地 verl，表示解释器直接执行这个 clone 中的源码，但第三方依赖仍来自 SkyRL；editable install 只是把同一个源码路径注册进环境。新建环境则额外隔离所有第三方包和版本。

SkyRL 技术上可以用 `conda create --clone` 复制后删除旧环境来达到“改名”，但 conda 没有可靠的原地 rename；脚本、Ray runtime 或 shebang 还可能保存旧绝对路径。因此这里不改共享 SkyRL 名称，而是在本仓库使用 `.venv` overlay。它的源码改动只影响本 clone，删除 `.venv` 即可恢复。

先下载小得多的数据集：

```bash
examples/dynamic_staleness/prepare_assets.sh dataset
examples/dynamic_staleness/prepare_assets.sh math500
examples/dynamic_staleness/prepare_assets.sh sis
```

`dataset` 保留下载得到的原始 Parquet；`sis` 校验其中 100 份 UUID 顺序完全一致，只抽取一份
17,917 题写入 `assets/datasets/SIS-Math-GRPO/data/train.parquet`，不会覆盖原文件。同时生成
`assets/datasets/MATH-500/data/math500-sis.parquet`。两份正式数据都标记为 `math_sis`，从而统一
使用 boxed 数学等价 scorer；DAPO/GSPO 算法实现不受影响。

`sis` 还会生成 `pilot-seed1-u0292.parquet`：它用固定 seed 模拟原始“100 份副本整体 shuffle”
的随机前缀，但只物化 update 0→292 实际需要的 `292×256=74,752` 条 prompt。正式训练对此文件
设置 `data.shuffle=false`，因此 shared anchor 与任意 N=4/8/16 branch 在相同 optimizer update
范围内读取严格相同的 prompt 序列；update 196 使用其前 50,176 条。换 seed 时必须用相同 seed
重新生成 schedule，不能只修改训练命令中的 `SEED`。

MATH500 固定使用 `HuggingFaceH4/MATH-500` revision `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`，转换脚本会检查必须恰好为 500 条并生成 verl/DAPO scorer 可直接读取的 Parquet。`ln02` 已有外网 DNS，可在登录节点完成一次下载；正式 Slurm 作业仍从 `assets/datasets/MATH-500/data/math500.parquet` 离线读取，避免重复缓存和网络波动。

主实验必须使用准确的 Base checkpoint；现有 `/home/ymy/cyt/models/Qwen3-8B` 是 `Qwen/Qwen3-8B`，不能代替 Base。smoke 使用已下载到本仓库 assets 的官方 `Qwen3-0.6B-Base`。下载正式 Base：

```bash
examples/dynamic_staleness/prepare_assets.sh model
```

先运行轻量隔离环境脚本。它复用 SkyRL 已有的 CUDA 大包，只在本 clone 的 `.venv` 中补齐依赖，且不会修改共享 SkyRL：

```bash
examples/dynamic_staleness/create_overlay_env.sh
```

只有 overlay 环境实际启动失败时，再运行 `create_fresh_env.sh`。后者从空的 Python 3.12 conda prefix 安装 torch 2.8/cu128、vLLM 0.11、FlashAttention 2.8.3 和本 clone；两个脚本都继承当前 shell 的代理变量，不在仓库中保存代理凭据。

## 5. 运行顺序

在并行云 Slurm 集群运行时，不要沿用本机 Python 绝对路径，也不要直接在登录节点启动训练。
上传、集群环境、A800/5090 profile、`sbatch` 提交和配额检查见
[`并行云环境说明`](../../集群使用指导/PARACLOUD_ENV_SETUP_ZH.md)。`run_staleness.sh` 可通过 `N_GPUS=1/2/4/8` 和
`ROLLOUT_TP_SIZE` 配置单机卡数；Slurm 包装脚本会把申请卡数同步传给 FSDP、vLLM 和 Ray。

先做两段最小 smoke；第一段验证一次 rollout 内 N=4 的逐 update 指标和 checkpoint，第二段验证从 checkpoint 改为 N=8：

```bash
examples/dynamic_staleness/run_smoke.sh
```

正式 anchor：

```bash
examples/dynamic_staleness/run_staleness.sh anchor
```

只做最小的 N=4 update 0→100、随后 N=8 update 100→196 时，可在同一个 Slurm allocation
内自动串行执行。第一段成功并校验完整 checkpoint 后才启动第二段；第二段失败时 update 100
checkpoint 仍然保留：

```bash
PARTITION=gpu_a800 N_GPUS=8 GPU_PROFILE=a800 ROLLOUT_TP_SIZE=2 SEED=1 \
  examples/dynamic_staleness/submit_slurm.sh n4_then_n8
```

该入口不会运行 N=4/N=16 branch；branch N=8 默认不保存完整 checkpoint，只写指标和评测结果。

按服务器现有两张 GPU 串行运行三个分支：

```bash
examples/dynamic_staleness/run_pilot_branches.sh
```

单独运行一个分支或延长到 292 update：

```bash
ANCHOR_CKPT=outputs/dynamic_staleness/20260829-153012_job123456_g8_tp2_seed1/anchor_n4_u0100/global_step_25 \
REUSE_N=8 TARGET_OPTIMIZER_STEP=292 \
examples/dynamic_staleness/run_staleness.sh branch
```

通过 `submit_slurm.sh` 提交时，每次作业会生成唯一的运行实例名：

```text
20260829-153012_job123456_g8_tp2_seed1
└── 日期时间_作业号_GPU数_TP数_seed
```

结果目录示例为 `outputs/dynamic_staleness/<运行实例>/anchor_n4_u0100/`；即使同一天用相同配置重复提交也不会覆盖。TensorBoard 使用相同实例前缀，run 名为 `<运行实例>_anchor_n4_u0100`，可统一查看：

```bash
tensorboard --logdir tensorboard_log --port 6006
```

续跑同一运行实例时显式设置原来的 `RUN_INSTANCE_TAG`，或者用 `OUTPUT_INSTANCE_ROOT` 指向确定的实例目录；不要用“最新目录”猜测 checkpoint。

默认 checkpoint 策略只在 anchor 最后一步保存 update 100 的完整可恢复 checkpoint；branch 默认 `SAVE_FREQ=-1`，只保留 staleness/eval 指标和逐题生成，不保存 8B 权重。若某条 branch 明确需要在 196 后恢复并延长到 292，提交该 branch 时显式设置 `SAVE_FREQ=1000000`，它会在该 branch 最后一步额外保存一份完整 checkpoint。

两卡对 8B full-parameter Adam 是这个 pilot 的主要资源约束。实测 legacy FSDP1 的 rollout 已完成，但第一次 Adam 状态创建达到约 76.5 GiB/卡并 OOM。这不是 rollout batch/序列长度造成的；即使把它们缩得很小，模型、梯度、Adam state 与 step 临时张量仍决定峰值。

这里必须区分两个名字相近但语义不同的开关：

- `optimizer_offload=true` 只是 verl 的手动换入换出：`update_actor` 开始时把已有 Adam state 搬回 GPU，结束后再搬回 CPU。它不是真正的 optimizer-only CPU update，也不能解决第一次 `optimizer.step()` OOM。
- `actor.strategy=fsdp2` 且 `offload_policy=true` 是当前两卡上已验证能明显降低显存的路径，但 PyTorch 会一起 offload parameter、gradient 和 optimizer，并非“参数常驻 GPU、只下放 optimizer”。正式脚本暂以它作为 memory-safe 默认；CPU 内存实测约 78 GiB/rank，update 会显著变慢。

当前 PyTorch FSDP/verl 组合没有可直接打开的 optimizer-only CPU-update 模式。TorchAO 的 `CPUOffloadOptimizer` 官方只支持单 GPU，并且不兼容内置 LR scheduler/gradient clipping，因此不应未经验证接入两卡主实验。若必须同时满足“参数留 GPU、Adam state 留 CPU”，应另开 Megatron/DeepSpeed ZeRO-Offload 工程分支，不与当前 FSDP pilot 混跑。先用 FSDP2 跑一个完整 outer step 测时，再决定 8B 是否适合在本机推进到 100 update；否则先在较小 Base 模型完成 N/metric 筛选，再到更多 GPU 上做 8B 验证。

下载准确的 8B Base 后，用下面的受限探针完成一次 N=4 outer step。它保留正式的 `1024/3072` 长度、GRPO/KL/reference 和 FSDP2 offload，只把 mini prompt batch 降到 1，因此适合验证第一次 Adam、checkpoint 与单步耗时，不可作为 N 实验结果：

```bash
examples/dynamic_staleness/run_8b_resource_probe.sh
```

2026-08-30 的 4×A800 集成探针（Slurm 149313）已通过：作业 `COMPLETED 0:0`，
`old_log_prob=4.80s`，连续完成 optimizer step 1–4；reward mean 为 `0.34375`，4 个
`grad_norm` 均非零，`policy_age=0,1,2,3` 各写入一条 `staleness_metrics.jsonl`。这验证了
remote Math-Verify、FSDP2 dynamic-batch DP 同步、首次 Adam 和 staleness 管线。它仍不是
256 prompt/update 的正式吞吐或显存结论；下一次正式 anchor 必须从 update 0 重新运行。

该探针退出清理时可能打印 `Exception ignored in atexit ... DataLoader worker ... Killed`；若
Slurm 状态为 `COMPLETED`、ExitCode 为 `0:0` 且 4 条 metric 完整，这是 Ray/DataLoader
退出阶段的非致命清理信息，不是训练失败。

探针通过后再运行 256 prompt/update 的 anchor。先关闭 Unreal 等占用 GPU1 算力的进程；此前一次 GPU-only 附加探针在 Adam 之前被该进程长期阻塞，已手动终止，不能据此声称 `foreach=false` 已经通过。

若要测试显存边缘但更快的 GPU-only 路径，可显式关闭 offload；必须单独观察第一次 update 和完整长度的峰值，不能直接批量开跑：

```bash
FSDP_STRATEGY=fsdp FSDP_OFFLOAD_POLICY=false \
FSDP_PARAM_OFFLOAD=false FSDP_OPTIMIZER_OFFLOAD=false \
GPU_MEMORY_UTILIZATION=0.6 \
examples/dynamic_staleness/run_staleness.sh anchor \
  'actor_rollout_ref.actor.optim.override_optimizer_config={foreach:false}'
```

对 `sis_offload` 或 `gpu_resident`，先用 8 卡正式 batch 但只跑一个 N=4 outer step：

```bash
PARTITION=gpu_a800 N_GPUS=8 GPU_PROFILE=a800 RUNTIME_PROFILE=sis_offload \
  JOB_NAME=staleness-8g-sis-offload-probe \
  examples/dynamic_staleness/submit_slurm.sh performance_probe
```

该 probe 保留 `256 prompts/update`、8 responses、`1024/3072` 长度和 4 次连续 update，
但关闭 MATH500 eval、checkpoint 保存并写入独立 `outputs/performance_probe/`，因此能验证首次
Adam、NCCL 和真实吞吐，又不会混入正式实验曲线。通过后，正式 anchor/branch 只需沿用同一个
`RUNTIME_PROFILE`；不要从 `memory_safe` anchor 切到另一 profile 后再把差异完全归因于 N。

如果只需确认 4 卡能否创建 SIS-offload 的 Adam state，不做吞吐测试，使用更小的单 update
探针：

```bash
PARTITION=gpu_a800 N_GPUS=4 GPU_PROFILE=a800 RUNTIME_PROFILE=sis_offload \
  JOB_NAME=staleness-4g-sis-adam \
  examples/dynamic_staleness/submit_slurm.sh adam_probe
```

它执行一次真实 `optimizer.step()`，会完整创建每个 rank 的 Adam state，但不跑 MATH500、
不保存 checkpoint。该探针通过只能证明 optimizer state 能放下；正式 batch 的 activation
峰值和吞吐仍由正式作业首个 outer step 观察。

若正式 N=4→N=8 作业应仅在 Adam probe 成功后启动，使用 Slurm `afterok` dependency：

```bash
PARTITION=gpu_a800 N_GPUS=4 GPU_PROFILE=a800 RUNTIME_PROFILE=sis_offload \
  SLURM_DEPENDENCY=afterok:<adam_job_id> \
  JOB_NAME=staleness-g4-sis-n4-then-n8 \
  examples/dynamic_staleness/submit_slurm.sh n4_then_n8
```

dependency 作业会提前进入队列，但在 probe 成功前不会占用 GPU；probe 失败或被取消时，正式
作业不会启动。查看时同时查询两个 job id，并以 `Dependency`/`DependencyNeverSatisfied` 区分
“正常等待前置作业”与“前置作业失败”。

## 6. 汇总

安装 `matplotlib` 后：

```bash
RUN_DIR=outputs/dynamic_staleness/20260829-153012_job123456_g8_tp2_seed1
python examples/dynamic_staleness/plot_staleness.py \
  anchor="$RUN_DIR/anchor_n4_u0100/staleness_metrics.jsonl" \
  N4="$RUN_DIR/branch_n4_u0100_0196/staleness_metrics.jsonl" \
  N8="$RUN_DIR/branch_n8_u0100_0196/staleness_metrics.jsonl" \
  N16="$RUN_DIR/branch_n16_u0100_0196/staleness_metrics.jsonl" \
  --eval-inputs \
  anchor="$RUN_DIR/anchor_n4_u0100/eval_metrics.jsonl" \
  N4="$RUN_DIR/branch_n4_u0100_0196/eval_metrics.jsonl" \
  N8="$RUN_DIR/branch_n8_u0100_0196/eval_metrics.jsonl" \
  N16="$RUN_DIR/branch_n16_u0100_0196/eval_metrics.jsonl" \
  --output "$RUN_DIR/pilot_dashboard.png"
```

除了图，还会导出原始合并 CSV 和按 `policy_age` 聚合的 median/q90 CSV。初期不预设绝对阈值；先以 N=4 continuation 的 late-window 分布作为内部基准，再决定 metric-adaptive N 的 EMA 门限，避免把任意阈值写进结论。

## 7. 论文对应

- [SIS: Turning Off-Policy Tokens On-Policy](https://arxiv.org/abs/2607.04728)：采用同一组 math 配置和 N 定义；Figure 3 按 gradient step 绘制 AIME24/AIME25 accuracy，附录指出训练动力学重点展示前 400 个 gradient steps。
- [Prosperity before Collapse / M2PO](https://arxiv.org/abs/2510.01161)：说明 stale 训练下 clip ratio 与 importance-weight 二阶矩比单一均值更关键。
- [Staleness–Learning Rate Scaling Laws](https://arxiv.org/abs/2607.01083)：给出局部 staleness 风险随 `N × learning-rate × update magnitude` 增长的理论动机。
- [Asynchronous RLHF](https://arxiv.org/abs/2410.18252)：提供“可容忍多少 off-policyness”的系统背景；本实验刻意不引入并行异步这一额外变量。
