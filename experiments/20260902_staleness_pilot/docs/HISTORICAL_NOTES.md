# 首轮动态 Staleness Pilot：历史研究笔记与旧预案

> 本文保存实验早期的作业索引、实现记录、研究推导和后续预案，仅用于追溯历史。
> 它不是当前实验配置或待执行任务；当前共同口径、资源约束和研究主线以
> [AGENTS.md](../../../AGENTS.md) 为准。

本文件由原实验记录中的历史部分拆出，未据此生成新的派生结果。

## 首轮作业索引（截至 2026-08-31 的历史记录）

正式 4 卡 `N=4 -> N=8` 串行作业为 Slurm `150042`，运行实例为
`20260830-174008_job150042_g4_tp2_seed1`。它使用 `sis_offload`、TP=2 和 seed 1，从 update 0 开始。旧失败作业 `149151/149194` 和已取消的旧配置作业 `149315` 不得用于实验结论。

`150042` 的 N=4 anchor 已于 2026-08-31 完成 update 0→100，共有 100 条 staleness 指标和
update 0,4,...,100 的 26 次 MATH500 评测，未出现 OOM、NCCL timeout 或 Traceback。完整可
恢复 checkpoint 位于 `anchor_n4_u0100/global_step_25`，约 92 GiB，状态为 optimizer step 100、
rollout step 25；作业随后已在同一 allocation 内进入 N=8 branch。

从同一 `global_step_25` 提交的 4 卡 N=4 branch 为 Slurm `151476`，N=16 branch 为
`151477`；二者于 2026-08-31 提交时因 `Priority` 排队，无 dependency，分别写入
`branch_n4_u0100_0196` 和 `branch_n16_u0100_0196`。这些是提交时的状态，不代表目前仍在
排队或训练；更新运行结论时须核对对应 job id、运行实例和实际产物，不猜测“最新目录”。

## 首轮探索性实验设计

| 阶段            | N   | optimizer update 范围 | 新增 verl global step |
| ------------- | ---:| -------------------:| -------------------:|
| shared anchor | 4   | 0 → 100             | 25                  |
| branch N=4    | 4   | 100 → 196           | 24                  |
| branch N=8    | 8   | 100 → 196           | 12                  |
| branch N=16   | 16  | 100 → 196           | 6                   |

N=4、N=8 与 N=16 从同一个 update 100 checkpoint 出发，各新增 96 次 update，
到 update 196。update 292 保留为可选延长终点，而非自动执行的下一步；先分析已有轨迹、
明确新增实验要回答的问题及 checkpoint 保存条件。使用 196/292 是为了保证从 update 100
开始的剩余更新数能同时被 4、8、16 整除。所有曲线按绝对 optimizer update 和累计 prompt
数比较，不能只按 verl `global_step` 比较。

shared checkpoint 分支可以回答“update 100 后立即增大 N 是否安全”。如果 pilot 出现信号，
还需要在 update 0/25/50 增加 matched early branches，并增加 2–3 个 seed，才能严格支持
“后期比早期更能容忍 staleness”的结论。固定 N 分支用于校准指标与漂移预算；在线
adaptive-N controller 应在测量口径、阈值与预测能力得到验证后再实现。

## 首轮实现与计数口径

当前 pilot 使用 verl 的普通 GRPO 和 colocated GPU，采用可控的“rollout 后多次 update”流程：

```text
policy π_rollout 生成一个 global rollout batch
                    ↓
              划分为 N 个 mini-batch
                    ↓
         按顺序执行 N 次 optimizer.step()
                    ↓
             更新后的 policy 再 rollout
```

这使第 `a` 个 mini-batch 的 `policy_age=a`，其中 `a=0,...,N-1`。改变 `N` 就能人为控制
同一轮数据的最大陈旧程度。这里复用的是 rollout 策略版本，N 个 mini-batch 是划分后的
不同数据，并非每条 trajectory 反复训练 N 次。异步 rollout/train overlap 可作为后续
外部有效性实验，不与本 pilot 的可控 staleness 混在一起。

计数口径：

- 一个研究中的 `update` 是一次真实的 `optimizer.step()`。
- 一个 verl `global_step` 是一次 rollout 加随后 `N` 次 update。
- `ppo_mini_batch_size=256` 表示 256 个 prompt group。
- `rollout.n=8`，所以每个 optimizer update 使用 2048 条 response trajectory。
- `ppo_epochs=1`，`train_batch_size=256*N`，因此改变 N 时每次 update 的数据量不变。
- 同样 96 次 update 下，N=4/8/16 分别刷新 rollout 24/12/6 次，但总 prompt group 和 response
  trajectory 数不因此减半。效率结论应使用实际耗时、token 数和吞吐，不能仅凭刷新次数。

## 首轮基础配置快照

- Model：Qwen3-8B-Base
- Train data：DAPO-Math-17K
- Algorithm：GRPO
- Learning rate：`1e-6`，warmup `10`
- Prompt mini-batch：`256`
- Responses per prompt：`8`
- Max prompt / response length：`1024 / 3072`
- PPO epochs：`1`
- KL loss coefficient：`1e-3`（以 SIS 论文 Table 4 的 math GRPO 配置为准）
- Clip low / high：`0.2 / 0.2`
- Entropy coefficient：`0`
- Engine：FSDP/FSDP2 + vLLM，并行云单节点 4×/8×A800-80GB colocated
- vLLM `gpu_memory_utilization=0.7`
- A800 执行上限与 SIS 8B recipe 对齐：actor/infer dynamic batch 每卡 `16384` token、vLLM`max_num_batched_tokens=8192`、`max_num_seqs=512`。这些只影响执行打包和吞吐，不改变prompts/update、responses/prompt、loss 或 optimizer-step 口径。
- 脚本保留 `memory_safe`、`sis_offload`、`gpu_resident` 三种 runtime profile。正式作业`150042` 固定使用已通过 4 卡首次 Adam 探针的 `sis_offload`：legacy FSDP，actor parameter与 optimizer 在训练阶段外手动 offload，reference parameter offload；不是 FSDP2`offload_policy=true`。同一 anchor/branch 链必须保持 profile、GPU 数和执行上限不变。
- 数据截断使用 `right`。SIS fork 的 `use_policy_clip=false` 在当前独立 verl v0.7.1 基线中不存在，因此保留标准 GRPO `clip=0.2/0.2`；`train_batch_size` 仍为 `256×N`，不能照搬SIS 公开通用启动示例的 512，否则只会得到有效 N=2，并破坏受控
  `policy_age=0,...,N-1`。脚本中的 `rollout.n=8` 是 responses/prompt，与 staleness N 无关。
- Math-Verify reward：`remote` manager，8 个独立 Ray reward workers

缩小 batch、模型或长度的运行只用于 smoke/resource probe；正式比较需要明确记录实际配置。

## 首轮指标与分析口径

### 四类 KL 的研究用途

以下比较应明确相同的上下文分布 q；表中的 KL 指在这些上下文上比较两策略的 next-token
分布，不是直接比较两段各自生成的答案文本。

| 类型       | 策略对                           | 回答的问题                         |
| -------- | ----------------------------- | ----------------------------- |
| 单步 KL    | 同一分支 `π_t` 与 `π_(t+1)`        | 一次 optimizer update 引起多少策略变化？ |
| 同轮累计 KL  | 本轮起点 `π_t` 与 `π_(t+a)`        | age=a 时的 staleness gap 有多大？   |
| 分支累计 KL  | 共同 `π_100` 与各自 `π_196`        | 各分支偏离共同起点多少？                  |
| 分支终点间 KL | `π_196^(N=4)` 与 `π_196^(N=8)` | 两种更新方式最终得到的策略有多不同？            |

终点间 KL 可报告两个方向；它衡量差异，不能单独判断优劣或中途稳定性。完整 next-token
分布 KL 与 sampled-token KL/proxy 必须分开命名；记录采样来源、KL 方向、token/sequence
权重、长度处理及评估模式。相邻策略应在同一组前缀上测量，固定 probe 用于跨 update/
stage 比较；本轮 rollout 前缀用于检查实际 stale 数据，两种分布的结果不能混为一条曲线。

### 已有逐 update 日志

每次 optimizer update 都写入 `staleness_metrics.jsonl`。其中 KL/ratio/clip 来自该次
`optimizer.step()` 之前的 forward；尽管日志的 optimizer step 编号在完成更新后记账，
这些值并不是更新后策略的测量。指标分成三组：

- `update_*`：当前 train-mode policy 相对本轮任何 update 之前的 train-mode anchor；用于隔离sequential update 本身造成的漂移，是主要分析口径。
- 无前缀指标：当前 train-mode policy 相对 verl 为 PPO 重算的 `old_log_probs`；对应 PPO surrogate 实际使用的 ratio 和 clipping。
- `rollout_*`：当前 policy 相对 vLLM rollout 时的 log probability；包含推理/训练引擎 gap和后续 update drift，对应端到端陈旧 token。

`update_*` 是相对同轮起始 anchor 的累计漂移，不是相邻更新 KL。各 age 使用的
mini-batch 也不同，不能将相邻两条 `update_*` 相减来恢复单步 KL。真正的单步 KL
需要在同一组上下文上比较更新前后策略；目前日志也没有记录实际参数位移或 Fisher。

重点曲线：

- sampled KL、mean absolute log-ratio；
- ratio mean、p05/p95/p99、second moment；
- clip fraction low/high/total 与真正触发 policy-gradient clipping 的 `pg_clipfrac`；
- importance-weight ESS fraction；
- response 累积 absolute log-ratio；
- gradient norm、reward、entropy，以及已有的 MATH500 accuracy。

每轮 `policy_age=0` 时，`update_ratio_mean` 应接近 1，update-only KL/clip 应接近 0。核心
判断是比较 early/late window 中指标随 `policy_age` 的增长速度，并观察增大 N 后 reward、
梯度和能力是否仍稳定，而不是仅凭平均 KL 下结论。

终点模型 KL 需要相应权重或为指定估计预先保留的充分概率数据；在新 benchmark 上生成
答案则需要模型权重。仅凭当前汇总 JSONL 和生成文本不能补算模型间 KL。启动新的
分支/测量运行前先决定保存完整可恢复 checkpoint、weights-only snapshot 还是特定
probe 概率，不默认已有终点模型。

## 已完成的代码

- `examples/dynamic_staleness/run_staleness.sh`：anchor/branch 统一入口和 update/global-step 换算。
- `examples/dynamic_staleness/run_pilot_branches.sh`：从共享 checkpoint 串行运行 N=4/8/16。
- `examples/dynamic_staleness/run_8b_resource_probe.sh`：正式长度下的一轮 8B 资源探针。
- `verl/trainer/ppo/staleness_metrics.py`：token-weighted KL、ratio、clip、ESS 和尾部统计。
- `verl/workers/actor/dp_actor.py`：每个 mini-batch 的 train-mode anchor 与三组指标采集。
- legacy FSDP actor 的三个 dynamic-batch 入口都在 DP process group 内同步 micro-batch 数，
  防止不同 response 长度导致各 rank 的 FSDP forward 次数不一致。
- `math_sis` 使用 `remote` reward manager；若误放入 DAPO 线程池会 fail-fast，不再把
  Math-Verify 的线程异常静默记为 reward 0。
- `verl/trainer/ppo/ray_trainer.py`：绝对 optimizer counter、逐 update JSONL、checkpoint 状态和
  改变 N 后的恢复检查，以及按绝对 optimizer update 写入的在线 eval JSONL。
- `examples/dynamic_staleness/prepare_math500.py`：将固定 revision 的 MATH500 转为 DAPO scorer
  兼容的 500 题 validation Parquet。
- `examples/dynamic_staleness/plot_staleness.py`：合并曲线、原始 CSV 和按 policy age 聚合统计。
- `examples/dynamic_staleness/README.md`：完整运行说明和资源约束。

## 已完成的验证

- Qwen3-0.6B-Base 已完成 N=4 anchor、checkpoint 保存、改成 N=8 恢复并继续训练。
- 已验证 optimizer step 从 1–4、再从 5–12 连续记录，`policy_age` 和 `reuse_n` 正确。
- 已验证 PPO ratio 与 vLLM rollout ratio 可以分别记录；age 0 的 update-only ratio 为 1。
- staleness 数值单元测试：`4 passed`。
- shell syntax、Python compile、Hydra dry-run 和 `git diff --check` 已通过。
- StatefulDataLoader checkpoint 在改变 rollout batch size 后仍从同一 sampler 游标继续；各分支在相同 optimizer-update 数内消费相同的 prompt 序列。
- 4×A800、Qwen3-8B-Base、正式 `1024/3072` 长度的 N=4 integration probe（Slurm 149313）已 `COMPLETED 0:0`：`old_log_prob=4.80s`，optimizer step 1–4 连续完成，reward mean`0.34375`，四个 grad norm 均非零，age 0–3 的 JSONL 完整。
- 已定位并修复两次正式 4 卡失败（149151/149194）：legacy FSDP dynamic batch 未传 DP group，一个 rank 少执行一次 forward，最终在 `_ALLGATHER_BASE` 等待 600 秒后 NCCL timeout。
- 已定位并修复原 MATH500 全 0：正确答案在主线程得 1，但 DAPO reward manager 在线程池调用Math-Verify 时触发 `signal.alarm()` 限制并被旧 scorer 静默吞掉。

## 已解决问题清单

- **可联网准备、离线训练**：`ln02` 已有外网 DNS；模型仍集中保存到集群持久化 assets，Slurm训练使用离线变量，避免重复下载和缓存膨胀。
- **缓存和磁盘占用**：Ray、vLLM、Triton、TorchInductor、CUDA、Numba 等运行缓存重定向到`/tmp/ds-$SLURM_JOB_ID/`；不再长期占用 `$HOME` 或 `run`。
- **重复数据**：移除无必要的 100 份重复 DAPO Parquet 和 Hugging Face datasets 缓存；GRPO使用一份 17,917 题的去重训练集，并按 seed 生成确定性 prompt 序列。
- **SIS 数学预处理对齐**：GRPO 训练集和 MATH500 使用 SIS `system + user` prompt 与严格`\boxed{}` 输出要求；DAPO/GSPO 路径未被修改。
- **Math-Verify 全零**：reward manager 改为独立 Ray process；线程环境误用会 fail-fast，不再静默把正确答案记为 0。正式作业 update-0 MATH500 Avg@1 已得到非零结果。
- **4 卡 rank/NCCL 等待**：legacy actor/ref 的三个 dynamic-batch 入口都传入实际 DP group，通过 `all_reduce(MAX)` 统一 micro-batch 数，修复不同 response 长度导致的 FSDP forward 次数不一致和 `_ALLGATHER_BASE` 600 秒超时。该实现不写死 4 卡，可扩展到 8 卡；8 卡真实 probe仍须完成。
- **输出覆盖与 TensorBoard 命名**：输出和 TensorBoard run 均包含日期时间、job id、GPU/TP、seed 和 phase，同一天多次实验不会互相覆盖。
- **checkpoint/eval 口径**：仅 anchor update 100 保存完整可恢复 checkpoint；在线 MATH500在完整 rollout→N 次 update 后评测，不插入同一批 stale update 中间。
- **4 卡集成与 Adam 验证**：Slurm `149313` 已完成正式长度的非零 reward/gradient probe；`sis_offload` 首次 Adam 探针 `150022` 已 `COMPLETED 0:0`，峰值 GPU allocated/reserved约 `38.16/53.97 GiB`。正式作业 `150042` 已连续完成 N=4 anchor 的 100 个 optimizer update，未复现旧 NCCL timeout。
- **已出现的 pilot 信号**：`150042` 的 MATH500 Avg@1 从 update 0 的 `0.694` 上升到
  update 100 的 `0.832`，该 anchor 阶段最佳为 update 100。比较 update 1–24 与 73–96的六个 rollout window，age 3 的 update absolute log-ratio 均值从 `0.01325` 降到`0.00562`，clip fraction 从 `0.00930` 降到 `0.00458`，grad norm 均值从 `0.0807`
  降到 `0.0441`。这只是单 seed、固定 N=4 的探索性信号，不能代替 matched branch 和多seed 结论。
- 8B legacy FSDP 已在两卡复现第一次 Adam state 创建 OOM；4 卡 `sis_offload` 已通过首次Adam 和正式 batch 长时间训练。8 卡仍未做真实 probe，不能仅凭实现不写死 world size 就宣称已经验证。
- 历史记录中，4 卡正式配置 actor MFU 约 `36%`，明显高于旧 full-policy offload 作业约 `12%`；
  代价是 CPU/GPU 手动搬运仍占用相当时间。是否切换 8 卡或 `gpu_resident` 必须另做 probe，不能在同一 anchor/branch 对比链中改变 runtime profile 或 GPU world size。
- 正式 anchor/branch 已在每个完整 rollout→N 次 update 周期后运行 MATH500 Avg@1；anchor 额外记录 update 0 基线。评测使用 greedy decoding，只保存 `eval_metrics.jsonl` 和逐题生成，不在同一批 stale rollout 的 N 次连续更新中间插入。默认仅 anchor update 100 保存完整可恢复checkpoint，branch 不保存完整 checkpoint。在线 adaptive-N controller 与 AMC/AIME evaluation 尚未实现。

## 本地整理图表

当前正式图表只保留由
[`plot_math500_branches.py`](../scripts/plot_math500_branches.py)生成的
[`math500_branches_by_optimizer_step`](../figures/math500_branches_by_optimizer_step.png)。
此前 8 张没有现存绘图代码的整理图已移出 `figures/`；对应的逐题评测、逐 update
staleness、TensorBoard event 和 Slurm 日志已经从远端完整归档到 `raw/`，后续需要时应
先补绘图代码再生成。

## 首轮后形成的研究笔记（2026-09-02）

### 1. 梯度信息 → 实际参数位移 → 策略漂移

“参数量变化”统一表述为“参数值的变化幅度/参数位移”，不是模型参数个数变化。
令 `θ_t` 为完成 t 次 optimizer update 后的参数，单次实际更新为：

```text
u_t = θ_(t+1) - θ_t
Δθ_(t,a) = θ_(t+a) - θ_t = Σ_(i=0)^(a-1) u_(t+i)
```

普通 SGD 下 `u_t = -η_t g_t`；Adam 类优化器还依赖梯度的一阶、二阶矩，实际位移不能由当前 gradient norm 直接换算。测量应区分裁剪前/后的梯度、学习率和实际 optimizer
更新；若采用 AdamW，还要计入权重衰减。[Adam 原文，Algorithm 1](https://arxiv.org/pdf/1412.6980)

在约定的状态/回答前缀分布 q 上，小范围参数变化的平均 KL 有局部二阶近似：

```text
D_q(π_θ || π_(θ+Δθ)) ≈ 1/2 · Δθᵀ F_(θ,q) Δθ
K_t(a) ≈ 1/2 · Δθ_(t,a)ᵀ F_(θ_t,q_t) Δθ_(t,a)
```

`F_(θ,q)` 是对应分布下的 Fisher 信息矩阵；第二式在本轮固定 `q_t`、局部近似仍有效时
使用。它提供解释和预测的研究起点，不要求直接构造 8B 模型的完整 Fisher 矩阵。
[TRPO 原文，附录 C](https://arxiv.org/pdf/1502.05477)

需要分别验证三段关系：梯度信息能否预测实际位移；位移能否预测单步 KL；多步更新方向
如何累积或抵消。由上述局部近似可见，多步累计 KL 含方向间的交叉项，不能直接等于
单步 KL 之和，也不能只用 `阈值 / 单步平均 KL` 宣称求得最大 N。当前日志中的
gradient norm 是候选预测特征，实际参数位移和真正的相邻更新 KL 尚需专门测量。

### 2. 借鉴 TRPO 的策略变化预算

TRPO 的理论动机涉及约束：

```text
D_KL^max(π_old, π_new) = max_s D_KL(π_old(·|s) || π_new(·|s)) ≤ δ
```

这里的 `max` 是对状态取最大 KL，不是最大更新次数、最大 N 或样本中最大的 log-ratio。
实际 TRPO 用平均 KL 近似难以直接处理的逐状态约束；重点阅读第 3–4 节及附录 C 中的
局部近似、步长确定与实际约束检查。[TRPO 原文](https://arxiv.org/pdf/1502.05477)

本项目借鉴的是“策略变化预算”思想：TRPO 选择一次更新能走多远，本项目探索旧 rollout
策略能支持多少次连续更新。当前仍使用 GRPO，不因引入 KL 阈值就改为 TRPO，也不能
直接继承 TRPO 的单调改进保证。

### 3. Threshold 与 maximal 分开定义

- `δ`：允许的策略漂移预算。必须明确 KL 方向、上下文分布、估计方法及 token/sequence
  聚合口径；它不等于 PPO 的 `clip=0.2`，也不等于 KL loss coefficient `1e-3`。
- `N_max(t)`：在该预算下，本轮能够容忍的最大连续更新数。它是受训练阶段和更新轨迹
  影响的量，不预设随 t 单调增加。

令 t 为本轮 rollout 时已完成的 optimizer update 数，固定本轮用于测量的前缀分布
`q_t`，先用同一引擎的策略定义 update-only 漂移：

```text
K_t(a) = E_(s~q_t) [D_KL(π_(θ_t)(·|s) || π_(θ_(t+a))(·|s))]
N_max(t) = max {n ∈ {1,...,N_cap}: K_t(a) ≤ δ，对所有 a=0,...,n-1}
```

`N_cap` 是预先设定的探索上限。这一定义与当前“使用数据前 `policy_age=0,...,N-1`”
的口径一致，要求所有中间 age 都满足预算；不假设累计 KL 单调。若还要约束最后一次
更新后的策略，需另检查 `K_t(n)`，不能将更新前 age=n-1 的日志当作该终点值。
端到端容忍度还要用真实 rollout log probability 检查引擎 gap，不能只看 update-only。

研究顺序先固定 `δ`，检验同一漂移预算在后期是否支持更大的 N；之后再单独研究阈值本身
是否需要随阶段变化为 `δ_t`。不要同时调整两者而失去对变化来源的解释。

阈值需结合能力表现、ratio 尾部、clip fraction、ESS 和梯度稳定性进行校准，并在未用于
调阈值的运行上验证。平均 KL 达标本身不是训练安全的充分条件，有限样本的最大值也不
等于全状态空间的 `D_KL^max`。目前尚未确定阈值数值或验证 gradient norm 预测器。

`N_max(t)` 是控制目标，不是当前能精确在线求出的量。先用固定 N 轨迹离线观察边界，
再研究“预测下一步漂移—实际校验—接近预算时刷新 rollout”的在线估计。若 N=4/8
均未触及阈值，只能说明已观测范围满足该条件，不能确定真实最大值。在线候选集可先
沿用 `{4,8,16}`；候选均不满足预算时应允许提前刷新，不能强行选择 N=4。提前刷新对
剩余 mini-batch 的处理及实际 prompts/update、采样开销必须明确记录，不能假定仍与
固定 N 实验完全等价。

## 当时的后续预案（历史参考）

1. 整理首轮 N=4/8 的 96 次新增 update 对照，核对实际配置、日志覆盖和对应产物。优先
   分析同轮累计漂移随 age 的增长、early/late 差异及能力变化；N=16 若已有结果，经核对后
   可作为扩展校准数据。保留已知作业索引，不沿用历史排队状态安排新作业。
2. 阅读 TRPO 第 3–4 节、附录 C 和 Adam 更新规则，明确参数位移、平均/最大 KL、漂移
   预算及最大陈旧度之间的关系。先提出可验证的预测关系，不把 gradient norm 下降
   直接解释成更大的安全 N。
3. 设计最小测量实验，补充实际参数位移与同一组前缀上的单步 KL，并检查它们对累计
   漂移的预测能力。记录测量开销；这些是待实现项，不是已有逐 update 日志的功能。
4. 先校准固定 `δ` 下的经验容忍范围，结合尾部指标与能力表现检验；随后用 matched
   early/late branches 和额外 seed 验证阶段差异。必要时探索更大的 age，但未观察到
   超阈值点时不能宣称已找到 maximal，也不自动延长到 update 292。
5. 在上述测量与校准之后，研究在线预测、逐步校验和刷新 rollout 的动态 N 规则，再
   评估真实训练稳定性、最终能力及吞吐收益。阶段相关阈值 `δ_t` 与真正异步 overlap
   分别作为后续扩展，不与首次固定预算的验证混在一起。
6. 在启动任何新分支、续训或终点 KL 测量前明确保存策略。当前分支默认 `SAVE_FREQ=-1`，
   不保存终点完整 checkpoint 或 BF16 weights；需要续训时应在启动前设为终点保存完整
   checkpoint，只需离线评测时应实现/启用 weights-only snapshot。不能假定第一次实验
   已有所有终点权重。8 卡性能 probe 保留为独立工程任务，不改变同一 anchor/branch
   对照链的 GPU 数、runtime profile 或执行上限。

以下原有开发检查命令作为历史操作索引保留；本次仅复核已有实验结果，没有提交训练或重跑这些开发检查。

常用检查：

```bash
bash -n examples/dynamic_staleness/*.sh
.venv/bin/python -m pytest tests/trainer/ppo/test_staleness_metrics.py -q
DRY_RUN=1 examples/dynamic_staleness/run_staleness.sh anchor >/tmp/staleness-config.txt
git diff --check
```
