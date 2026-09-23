# 动态 Off-policy / Staleness：核心研究与实验约束

更新：2026-09-23。本文件只保留长期研究主线、共同计数口径、实验文档规范、资源约束
和当前基础配置。具体实验的设置、启动与状态写入实验 README；只有确认分析后，数值
结果和图表解读才写入对应 `EXPERIMENT_RECORD.md`。本文件不累积单次运行日志。

- 共同基线实验：[无 warmup、等 24 轮 N=4/8 实验](experiments/20260909_full_kl_no_warmup_n4_n8/README.md)。
- 实验入口：[实验索引](experiments/README.md)。
- 操作资料：[运行说明](examples/dynamic_staleness/README.md)与[并行云环境说明](集群使用指导/PARACLOUD_ENV_SETUP_ZH.md)。历史命令需与本文件的当前配置核对。

## 科研命题

在每次真实 optimizer update 的训练数据量固定时，研究能否利用一个 rollout 周期前段
已经观测到的学习动力学，预测继续使用同一行为策略数据时的未来策略漂移，并据此动态
选择该行为策略支持的连续更新次数 N。目标是在相同训练 update、trajectory/token 和
模型能力约束下减少 rollout 刷新开销，同时不增加校准后的 staleness 越界风险。

这个命题包含两个需要分别验证的问题：前段观测能否对尚未发生的轮内 KL 峰值提供额外
预测能力；使用该预测选择 N 后，能否在等训练数据预算下提高端到端训练效率而不损害
稳定性和能力。前者是预测问题，后者是策略效果问题，不能由同一次相关性分析替代。

## 核心 idea

当前顺序是：测量轮内相邻与累计 KL → 检验早期 gradient/update 信息是否能预测后续
漂移 → 形成带不确定性的未来峰值预测 → 结合能力与稳定性校准 KL 预算 → 动态选择 N。
在线 controller 是后续验证对象，不是当前测量实验的前提。

核心假设是后期单步策略漂移可能减小，从而在同一 KL 预算下允许更大的 N；不预设
KL、峰值或可容忍 N 随训练单调变化。stage 按真实更新编号或明确 checkpoint 描述，
同时记录学习率和观测分布，避免把调度变化或前缀变化归因于纯粹的阶段效应。

### 固定每次 update 的数据量

固定 `ppo_mini_batch_size=256` 个 prompt group、`rollout.n=8` responses/prompt，
每次真实 update 使用 2048 条 trajectory；`ppo_epochs=1`。一次 rollout 的 batch 为
`train_batch_size=256*N` 个 prompt group，并划分成 N 个不同 mini-batch，各训练一次。

因此改变 N 时不缩小 mini-batch。N 从 4 增至 8 时，一次 rollout 准备的数据量翻倍，
随后连续执行 8 次同样数据量的更新；复用的是行为策略版本，不是把同一 trajectory
重复训练 N 次。

### 轮内累计 KL、峰值与预算

令 t 为本轮 rollout 起点的真实更新编号，π_t 为该策略版本的训练侧 anchor。在本轮
固定的前缀集合和权重 q_t 上直接测量旧模型到新模型的全词表 KL：

```text
K_t(a) = E_(h~q_t) [KL(π_t(·|h) || π_(t+a)(·|h))]
M_post_t(N) = max {K_t(a): a=1,...,N}
M_pre_t(N)  = max {K_t(a): a=0,...,N-1}
M_post_t(4→N) = max {K_t(a): a=5,...,N}
M_post_t(N) = max(M_post_t(4), M_post_t(4→N))
```

主图采用更新后 `age_after=1,...,N`，包含终点；使用旧数据前的预算采用
`policy_age=0,...,N-1`。声明约束范围后再计算容忍度，不混用两种峰值。这里的 max
沿轮内更新 age 取值，每个 K 本身仍是前缀平均 KL，不是对状态取最大值。

比较较长窗口分支的前 4 步与后续步骤时，固定该轮 anchor 和前缀。这能测量延长窗口
带来的额外漂移，但不等于已经验证每 4 步刷新一次的 N=4 分支。解释刷新行为的影响需
另做匹配起点、optimizer/scheduler 状态和更新数据安排的分支对照。

先固定候选 KL 预算 δ，再研究各 stage 满足整个 age 范围约束的最大已测 N。δ 不等于
PPO clip 参数、KL loss coefficient 或 N_cap；当前未确定 δ。“预算内”还需能力、
ratio 尾部、clip fraction 和 ESS 校准。有限运行的观测最大值只是经验先验，不是普适
安全上界；只测到 N=8 满足条件，不代表已找到 maximal。

若在第 4 步决定是否继续延长，只能使用截至第 4 步可获得的信息预测未来
`age=5,...,N_cap` 的峰值，并重点检验低估导致的越界。预测时比较“只用训练进度”
“加入近期 KL”“再加入梯度或位移信息”的效果，在未参与拟合的运行上验证；不能使用
未来 gradient norm、KL 或评测输入。最终规则应选择满足未来峰值高分位约束的最大 N：

```text
N* = max {n : Q_(1-alpha)(max_(a=5,...,n) K_t(a) | I_t,4) <= delta}
```

`I_t,4` 表示第 4 次更新结束时已经获得的信息；`delta` 和置信水平需由数据校准，不预先
绑定到一次运行中的示例数值。

## 当前推理假设

候选解释链条是：stage → 梯度与实际参数更新 → 单步/累计 KL → 可容忍的更新间隔。
令 `θ_t` 为完成 t 次真实 optimizer update 后的参数，在固定上下文分布 q 上：

```text
u_t = θ_(t+1) - θ_t
θ_(t+a) = θ_t + Σ_(i=0)^(a-1) u_(t+i)
Δθ_(t,a) = θ_(t+a) - θ_t
d_t ≈ 1/2 · u_tᵀ F_(θ_t,q) u_t
D_q(π_(θ_t) || π_(θ_(t+a))) ≈ 1/2 · Δθ_(t,a)ᵀ F_(θ_t,q) Δθ_(t,a)
```

最后两式仅为小参数位移下的局部二阶近似，F 是对应的 Fisher 信息矩阵。普通 SGD 有
`u_t=-η_t g_t`，Adam 类更新还依赖梯度的一阶、二阶矩；gradient norm 不能直接换算
实际位移，多步累计 KL 也不能等于相邻 KL 之和。事后相关不等于额外预测能力。

不直接拟合完整 Fisher，而是先检验两个可由实测 KL 构造的有效标量：

```text
c_eff_i = d_i / (eta_i^2 ||g_i||^2)
rho_eff_i = (K_i - K_(i-1) - d_i) / (2 sqrt(K_(i-1) d_i))
```

`c_eff` 表示 gradient norm 到相邻 KL 的有效换算，`rho_eff` 表示新更新与此前累计位移
在局部 KL 几何中的有效对齐。AdamW 下 `c_eff` 同时吸收预条件、动量、weight decay
和 Fisher 方向曲率，不能解释为纯曲率；若能低成本记录真实 `update_norm=||u_i||`，
应将它作为比 raw gradient norm 更接近参数步幅的候选特征。

验证分两层：先代入已经观测到的后续 gradient norm，检验其与 adjacent KL 的换算
关系能否跨 age 泛化；这只是 oracle-gradient 回顾性检验。真正的前瞻预测还必须只用
前四步信息预测未来 gradient/update、`c_eff`、`rho_eff` 和累计 KL 峰值，并按完整
rollout 或独立运行留出验证。

### 论文依据的使用边界

- [TRPO 第 3–4 节与附录 C](https://arxiv.org/pdf/1502.05477)：借鉴“允许策略变化多少”
  的 KL 预算和局部 Fisher 近似。`D_KL^max` 的 max 是对状态取最大值，实际算法用平均
  KL 近似；本项目研究既定优化器下旧 rollout 能支持多少步，不把 GRPO 改称 TRPO，
  也不直接继承其改进保证。
- [Learning Dynamics of LLM Finetuning 第 2–3 节](https://arxiv.org/html/2407.10490v4)：
  借鉴“每次更新如何改变输出概率，以及影响如何多步积累”的分析方式。其分解不能缩减
  成单一 gradient norm 关系；SFT/DPO 结论迁移到 GRPO staleness 场景仍需验证。

## 共同实验口径

- 一个研究 update 是一次真实 `optimizer.step()`；一个 verl `global_step` 是一轮
  rollout 加随后 N 次 update。分别记录，禁止混用。
- 等 rollout 周期只用于收集相同数量的轮内曲线，不等于等训练量。跨 N 或分裂后的
  策略效果比较首先对齐真实更新数和 prompt group/trajectory 数；当前每次 update 的数据量固定，因此它们等价。response 长度会变化，实际训练 token 需另外核对，并在差异明显时按共同累计 token 补充比较。首要预算不是相同 rollout 周期；rollout 次数、生成开销、总耗时和吞吐作为效率结果另行报告。
- 校准出 KL 预算后，策略对照按预先声明的 `M_pre` 或 `M_post` 口径，在等训练数据预算下报告最坏峰值和按 update/trajectory 加权的越界率；不同策略经历的 rollout 周期数不同，不能直接比较“有多少个周期越界”。
- KL 必须明确两端策略、方向、前缀分布和聚合权重。当前使用每轮真实 rollout 的已保存前缀，轮内固定、轮间重选，按 prompt 等权再按有效位置等权，并记录实际位置数。跨 stage 隔离模型漂移需另用统一 probe q；换轮后的曲线断点不表示模型回退。
- 相邻模型 KL、相对本轮 rollout anchor 的累计 KL、两分支间 KL 分开记录；全词表 KL
  与 sampled-token proxy 分开。`update_*` 是更新前相对 train-mode anchor 的累计漂移，`rollout_*` 还包含引擎 gap；不能用相邻日志相减恢复相邻模型 KL。
- 正式逐步分析以 `kl_updates.jsonl` 为准。TensorBoard 的 `full_kl/age_XX/*` 横轴是 rollout/global step，`full_kl_by_update/*` 横轴是真实更新数；tag 中的 age 不是横轴。

共同基线测量实验中，N=4 执行 96 次更新、N=8 执行 192 次更新，均为 24 个完整 rollout
周期；每组使用 4 张 A800。学习率从第一次真实更新起固定为 `1e-6`，无 warmup；
MATH500 sampled Avg@1 在训练前和每轮完成后同步评测。相邻/累计 KL、gradient norm逐次更新记录。历史各 96 步、10 步 warmup、0/48/96 评测及 greedy 结果保持各自口径，不与本轮混合。具体结果见[当前实验记录](experiments/20260909_full_kl_no_warmup_n4_n8/docs/EXPERIMENT_RECORD.md)。

训练 token 与评测或测量重复计算量分开；训练周期、能力评测、KL 直接计时、权重保存
和作业总耗时分别报告。KL 计时已包含在训练周期中，不重复相加；先核实吞吐是每卡还是
全卡合计，再比较刷新频率的收益。

## 实验文档与命名规范

实验目录统一命名为 `experiments/YYYYMMDD_<topic>_<variant>/`，使用小写
`snake_case`。日期放在最前，表示实验正式建立日期；GPU 实验通常取首次正式提交日期，
纯离线分析取开始分析日期。目录建立后不再因补跑或结果变化改名。Slurm job ID 不进入
实验目录名。同一设置的 probe、正式运行和重跑属于同一实验，分别放在
`raw/<phase>_job<id>/`；若核心研究问题或主要对照改变，则建立新实验目录。README
标题使用 `YYYY-MM-DD：实验名称`。

若一个研究问题包含多个独立提交、可独立解释的正式设置（例如 N=1/4/16/32），建立一个
带日期的实验族目录，并在其中按小写 `snake_case` 建立子实验。实验族 README 只保存
共同问题、共同配置、公式口径、设置关系和索引；每个子实验必须各自拥有 `README.md`、
`raw/` 和经确认后才建立的 `docs/EXPERIMENT_RECORD.md`。设置特有的 job、时间、配置与
结果不得只写在实验族文档中，也不得把多个子实验的结果合并成一份 Record。跨设置分析
若被单独确认，放入实验族的 `comparisons/<analysis_name>/`，不覆盖各子实验记录。

每个独立实验或实验族中的子实验首先只建立 `README.md`，写清：

1. 实验任务；
2. 与其他实验的关系及产生关系的具体原因；没有直接关系时不强行填写；
3. 模型、数据、N、更新数、seed、学习率、资源、评测、测量和保存等实验配置；
4. 提交脚本、job ID、提交/开始/结束时间、依赖、远端输出和带核验日期的运行状态；
5. `raw/`、`scripts/`、权重及其他已有产物入口。

子实验 `README.md` 是该设置和启动信息的唯一记录位置。Slurm 完成只表示运行完成，不自动
产生分析结论。原始产物可以先归档并更新 README；没有分析要求时，不创建
`docs/EXPERIMENT_RECORD.md`，也不预写结果。

只有用户明确要求并确认分析方案后，才创建 Markdown 格式的
`docs/EXPERIMENT_RECORD.md`。该文件只记录分析问题与数据范围、结果、图表、结论边界
和复现方式，不重复实验配置、启动过程或历史计划。数值明细放入 `tables/*.csv` 或
`tables/*.json`；图放入 `figures/*.png`，需要矢量版本时同时保存 PDF。Record 中嵌入或
链接确认后的 PNG，并注明图要回答的问题、数据口径、主要观察、绘图脚本和 PDF。

已经被当前规范取代但仍需保存的旧研究笔记，单独放入 `docs/HISTORICAL_NOTES.md`，
开头注明它不是当前配置或待执行任务。单实验或子实验按实际产物创建，不补空目录：

```text
README.md  实验任务、关系、配置、启动、时间、状态和产物入口
docs/      仅在已有确认分析时保存 EXPERIMENT_RECORD.md；必要时保存 HISTORICAL_NOTES.md
scripts/   采集、提交、同步、汇总和绘图脚本
raw/       原始快照、配置、日志、逐步/逐题数据，保留来源且不覆盖
tables/    可复现生成的 CSV、JSON 等派生表
figures/   可复现生成的 PNG、PDF 图表
```

实验族的最小结构为：

```text
YYYYMMDD_<topic>/
  README.md                  共同问题、公式、共同配置和子实验索引
  scripts/                   跨设置共用脚本
  <setting>/README.md        该设置的配置、启动、时间、状态和产物入口
  <setting>/raw/             该设置的原始产物
  <setting>/docs/            仅保存该设置经确认后的实验记录
  <setting>/tables|figures/  该设置可复现的派生产物
```

分析仍按以下流程执行：Codex 先提出候选研究问题、比较口径、指标和图表；用户修改或
补充；Codex 评估数据支持与混杂因素；用户确认最终方案后，才生成派生表、图表和正式
结论。未经确认不新增派生指标或把探索性判断写成正式结论。

分析脚本按自身位置定位实验根目录，从 `raw/` 只读输入，写入 `tables/`、`figures/` 或
`docs/`，不依赖启动目录。搬动文件时同步更新脚本和文档链接；原始数值不因重新分类而
改写，派生结果应能由脚本复现。

## Git 维护规范

本项目使用 Git 维护。每次代码、配置、脚本或文档变更都应形成提交；未完成检查和提交
的改动不视为交付完成。

- 修改前先检查工作区，保留并区分已有的用户改动；一次提交只包含一个内聚变更。
- 只暂存本次任务相关文件，提交前检查 staged diff，并运行与改动风险相称的静态检查或
  测试；提交信息简洁说明变更目的。
- 代码、配置、README、分析记录、可复现脚本、派生表和正式图表进入 Git。各实验的
  `raw/`、远端权重与 optimizer checkpoint、重复代码快照、缓存和密钥不进入 Git。
- 不覆盖或改写已有提交历史，不使用破坏性 reset；除非用户明确要求，不向远端 push。
- 完成任务时报告提交哈希；若因检查失败或外部阻塞无法提交，明确说明未提交的原因。

## 实验资源与使用约束

8B 正式训练和 GPU integration probe 在并行云 Slurm 集群执行；本地只做代码编辑、
静态检查和结果分析。登录节点只做准备、提交、读日志和文件整理，不运行训练或
Ray/vLLM GPU resource probe。

```bash
ssh scyb980\@NMCC-N46H1\@ssh.paracloud.com -p 2222
```

集群仓库：`/data/run01/scyb980/cyt/src/verl-staleness`。

- 统一通过 `examples/dynamic_staleness/submit_slurm.sh` 提交到 `gpu_a800`。4 卡
  `sis_offload` 已验证；单个 8 卡训练任务需独立真实 probe，两组各 4 卡并行不等于
  验证了 8 卡 world size。同一对照保持 GPU 数和 runtime profile 一致。
- `N_GPUS` 与 Slurm 卡数、FSDP world size、Ray 一致，`ROLLOUT_TP_SIZE=2` 整除
  `N_GPUS`。跨 world size 恢复前另验 FSDP checkpoint 与 StatefulDataLoader 恢复。
- `ln02` 可联网按固定 revision 准备 assets；训练使用持久化模型、数据和离线变量。
  Ray、vLLM、Triton、TorchInductor、CUDA、Numba 等可释放计算缓存放
  `/tmp/ds-$SLURM_JOB_ID/`，权重、JSONL、逐题生成和 TensorBoard 放 `/data/run01`。
- 输出实例名编码日期时间、job id、GPU 数、TP、seed 和 phase；不覆盖旧结果或猜测
  “最新目录”。提交前核查持久化空间、测量产物和权重保存策略；旧 branch 默认
  `SAVE_FREQ=-1`，不能假设已有终点权重。
- `sis_offload` 在训练阶段外手动 offload actor parameter/optimizer，并 offload
  reference parameter，不是 FSDP2 `offload_policy=true`。其他 profile 另做 probe。
- 保留 legacy FSDP dynamic-batch 三个入口在实际 DP group 内同步 micro-batch 数的
  修复；Math-Verify 不放入 DAPO 线程池。缩小模型、batch 或长度的运行明确标为 probe。
- SIS 数学预处理使用 `system + user` 和严格 `\boxed{}` 输出要求。当前独立 verl
  v0.7.1 保留标准 GRPO clip，不照搬 SIS fork 的 `use_policy_clip=false`，也不照搬
  `train_batch_size=512` 破坏 N 的定义；`rollout.n=8` 不是 staleness N。

## 基础配置

| 项目                                                       | 当前共同基线                                                           |
| -------------------------------------------------------- | ---------------------------------------------------------------- |
| 模型 / 算法 / 数据                                             | Qwen3-8B-Base，全参数 GRPO，DAPO-Math-17K，沿用 SIS 数学预处理                |
| 学习率 / warmup                                             | 第一次真实更新起固定 `1e-6`；`LR_WARMUP_STEPS=0`                            |
| Prompt mini-batch / responses / PPO epochs               | `256 / 8 / 1`                                                    |
| Rollout batch                                            | `256*N` 个 prompt group                                           |
| Prompt / response 最大长度                                   | `1024 / 3072`，截断 `right`                                         |
| KL loss coefficient / PPO clip low、high / entropy        | `1e-3 / 0.2、0.2 / 0`；KL coefficient 与 δ 区分                       |
| Runtime                                                  | 4×A800-80GB、TP=2、`sis_offload`、legacy FSDP + vLLM colocated      |
| vLLM memory utilization / batched tokens / max sequences | `0.7 / 8192 / 512`                                               |
| Actor / infer dynamic batch                              | 每卡 `16384` token                                                 |
| Reward                                                   | Math-Verify `remote` manager，8 个独立 Ray reward workers            |
| 能力评测                                                     | MATH500 sampled Avg@1：`n=1, temperature=1, top_p=0.95, top_k=-1` |
| 评测时点                                                     | 训练前一次、每轮 N 次更新完成后一次；`TEST_FREQ=1`，单位为 rollout                    |
| 全词表 KL                                                   | `FULL_KL_EXPERIMENT=1`；每轮 64 prompts，每个所选 response 最多 8 个有效位置    |
| 保存策略                                                     | 终点 HF 权重；不保存 Adam 状态，不能当作完整续训 checkpoint                         |

当前配方显式覆盖 warmup 和评测频率，不依赖旧脚本默认值；锁定参数见
[共同基线脚本](experiments/20260909_full_kl_no_warmup_n4_n8/scripts/rerun_experiment2_20260909.sh)。
