# 动态 Off-policy / Staleness：核心研究与实验约束

更新：2026-10-07。本文件只保留研究命题、核心 idea、共同计数口径、文档与代码隔离规范
和必要基础信息。具体实验的设置、启动与状态写入实验 README；只有确认分析后，数值
结果和图表解读才写入对应 `EXPERIMENT_RECORD.md`。本文件不累积单次运行日志。

- 本地研究资料入口：[实验索引](experiments/README.md)与[集群使用说明](集群使用指导/PARACLOUD_ENV_SETUP_ZH.md)。这些相对链接属于本地档案仓库。
- 训练代码入口：[Skialith/DynamicOffPolicy](https://github.com/Skialith/DynamicOffPolicy)。具体运行配方以对应实验 README 和远端提交脚本为准。

## 科研命题

在每次真实 optimizer update 的训练数据量固定时，研究能否利用一个 rollout 周期前段
已经观测到的学习动力学，预测继续使用同一行为策略数据时的未来策略漂移，并据此动态
选择该行为策略支持的连续更新次数 N。目标是在相同训练 update、trajectory/token 和
模型能力约束下减少 rollout 刷新开销，同时不增加校准后的 staleness 越界风险。

这个命题包含两个需要分别验证的问题：前段观测能否对尚未发生的轮内 KL 峰值提供额外
预测能力；使用该预测选择 N 后，能否在等训练数据预算下提高端到端训练效率而不损害
稳定性和能力。前者是预测问题，后者是策略效果问题，不能由同一次相关性分析替代。

## 核心 idea

当前顺序是：直接测量轮内累计 KL → 用 anchor Fisher 的 λmax 与实际累计参数位移
解释局部漂移尺度 → 检验前段信息对未来位移和 KL 峰值的预测能力 → 校准不确定性、
能力与稳定性预算 → 动态选择 N。在线 controller 是后续验证对象，不是测量前提。

候选解释链条是：stage → anchor 处的方向曲率与 optimizer 实际位移 → 局部 Fisher
二次型及其谱界 → 真实累计 KL 与近似余项 → 可容忍的更新间隔。不预设 λmax、KL、
峰值或可容忍 N 随训练单调变化；stage 按真实更新编号或明确 checkpoint 描述，同时
核对学习率和观测分布，避免将调度变化或前缀变化归因于纯粹的阶段效应。

### 固定每次 update 的数据量

共同口径为每次真实 update 使用 256 个 prompt group、每个 prompt 8 条 response，
即 2048 条 trajectory；`ppo_epochs=1`。一次 rollout 准备 `256*N` 个 prompt group，
划分成 N 个不同 mini-batch，各训练一次。

改变 N 时不缩小 mini-batch；复用的是行为策略版本，不是把同一 trajectory 重复训练
N 次。缩小 batch 的工程运行必须标为 probe，不与正式训练量混用。

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

更新后口径 `age_after=1,...,N` 包含终点；使用旧数据前的口径为
`policy_age=0,...,N-1`。声明约束范围后再计算容忍度，不混用两种峰值。这里的 max
沿轮内 age 取值，每个 K 本身仍是前缀平均 KL，不是对状态取最大值。

比较较长窗口的前段与后续步骤时，固定该轮 anchor 和前缀。这能测量延长窗口带来的
额外漂移，但不能替代更频繁刷新 rollout 的分支对照。验证刷新策略需匹配起点、
optimizer/scheduler 状态和更新数据安排。

先固定候选 KL 预算 δ，再研究各 stage 满足整个 age 范围约束的最大已测 N。δ 不等于
PPO clip 参数、KL loss coefficient 或 N_cap；当前未确定 δ。“预算内”还需能力、
ratio 尾部、clip fraction 和 ESS 校准。有限运行的观测最大值不是普适安全上界；
只测到某个 N 满足条件，不代表已找到 maximal。

## 当前推理假设：λmax 与实际参数位移

令 θ_t 为完成 t 次真实 optimizer update 后的参数，F_t 为同一 anchor 和 q_t 上的
全参数 Fisher。它等于 frozen-anchor 全词表 KL 在 θ_t 处对参数的 Hessian：

```text
u_i = θ_(i+1) - θ_i
Δ_(t,a) = θ_(t+a) - θ_t = Σ_(i=0)^(a-1) u_(t+i)
F_t = E_(h~q_t, y~π_t) [s_t(h,y) s_t(h,y)^T]
s_t(h,y) = ∇_θ log π_θ(y|h) |_(θ=θ_t)

Q_t(a) = 1/2 · Δ_(t,a)^T F_t Δ_(t,a)
B_t(a) = 1/2 · λ_max(F_t) ||Δ_(t,a)||²
Q_t(a) <= B_t(a)
K_t(a) = Q_t(a) + R_t(a)

B_hat_t(a) = 1/2 · λ_hat_max(F_t) ||Δ_(t,a)||²
```

λmax 描述 anchor 处最敏感参数方向的局部曲率；实际累计位移描述 optimizer 已经走了
多远。二者共同提供漂移尺度，不能只用 λmax 或 raw gradient norm 判断允许的 N。
Adam 类更新依赖动量、预条件与 weight decay，gradient norm 不能直接换算实际位移；
累计位移 norm 也不能由各步 norm 相加恢复，相邻 KL 之和不等于累计 KL。

上述谱不等式约束的是局部二次型 Q。真实有限位移 KL 还含余项 R，R 没有预设符号或
已校准上界；anchor 曲率也不能代表整个参数路径上的曲率。因此 B_hat 是候选漂移尺度
或候选界，不是有限位移 KL 的严格安全上界，不能未经校准直接拿来替代 KL 预算。

不构造完整 Fisher；通过 FVP 与谱迭代估计 λmax。固定模型、q_t 与 anchor 后再改变
迭代方向，不能在迭代中更新 anchor。Rayleigh quotient 和特征方程残差用于数值检查；
残差达标只说明找到近似特征对，不自动证明它就是全局最大特征值，也不提供最大
特征值上界证书。迭代未收敛时不将结果作为合格 λ 使用，门槛与资源上限写入实验配置。

当前测量以每轮 anchor 的 λ、直接累计 KL K 和真实累计位移构造的 B_hat 为主。
必要时另测 Q，区分方向曲率、谱界松紧和有限位移余项；Q 是诊断量，不要求每个实验
都计算。各量必须使用同一策略 anchor、前缀分布、参数版本与聚合权重；不同精度或
模型执行路径的 KL 不混作同一测量。

### 求导实现与已否决路线

当前 FVP 在独立、非 FSDP 的测量模型中，逐层传播参数/输入 JVP，再作用全词表
softmax Fisher，最后逐层重算并做普通 VJP，得到 `J^T F_z Jv`。它与 anchor 处的
KL HVP 数学对象相同；embedding、norm、head 等全参数参与，不冻结参数或使用 TOP-K。
只保留 detached 层输入和单层反传图，避免保留整网二阶图。

- 更新后有限 logits/log-prob 差分代理 KL、Fisher 长度和对齐量已否决：得到新策略
  输出后直接计算 KL，不将事后输出差分作为未来预测或参数 JVP 的替代。
- 训练侧 legacy FSDP 内直接 functional 参数 JVP 已注释停用，保留最初逻辑考量。
- 参数扰动、logits 中心差分近似 JVP 已注释停用，列为最后 JVP 实现备选；它与
  更新前后有限输出差分代理是不同方法。
- KL 梯度中心差分近似 FVP/HVP 未通过既有数值对照，不能作为当前谱估计主实现。
- 整网 double-backward 保留作小模型精确 AD 参考；整网图 CPU offload、前向重算
  是该路线的内存处理，不改变 Fisher 目标。长前缀测量使用当前逐层实现。

### 从测量到前瞻预测

事后用实际累计位移计算 B_hat，只能说明已发生的漂移关系。即使使用已观测到的未来
gradient/update 后能复现后续 KL，也只是 oracle 回顾性检验，不是在线预测。

若在第 4 步决定是否继续延长，只能使用截至第 4 步已获得的 anchor λ、近期 KL、
gradient/update 和累计位移信息，预测未来 `age=5,...,N_cap` 的位移及 KL 峰值。
固定 anchor 的 λ 估计可以作为特征；未来位移、曲率变化和近似余项仍需预测与校准，
不能读入未来 gradient、KL 或评测输入。对比“只用训练进度”“加入近期 KL”“再加入
λ 与位移信息”的额外预测能力，并按完整 rollout 或独立运行留出验证，重点检查低估
导致的越界。最终候选规则为：

```text
N* = max {n : Q_(1-alpha)(max_(a=5,...,n) K_t(a) | I_t,4) <= delta}
```

`I_t,4` 表示第 4 次更新结束时已经获得的信息；这里的 `Q_(1-alpha)` 是条件分位数，
与 Fisher 二次型 Q_t 不同。δ 和置信水平由能力、稳定性及预测误差校准，不绑定一次
运行的示例值；λ/B_hat 的解释能力和动态 N 的策略收益分别验证。

### 论文依据的使用边界

- [TRPO 第 3–4 节与附录 C](https://arxiv.org/pdf/1502.05477)：借鉴 KL 预算和局部
  Fisher 近似。其 `D_KL^max` 的 max 是对状态取最大值，实际算法用平均 KL 近似；
  本项目研究既定优化器下旧 rollout 支持多少步，不直接继承其改进保证。
- [Learning Dynamics of LLM Finetuning 第 2–3 节](https://arxiv.org/html/2407.10490v4)：
  借鉴更新如何改变输出概率以及多步积累的分析方式，不能缩减成单一 gradient norm
  关系；SFT/DPO 结论迁移到 GRPO staleness 场景仍需验证。

## 共同实验口径

- 一个研究 update 是一次真实 `optimizer.step()`；一个 verl `global_step` 是一轮
  rollout 加随后 N 次 update。分别记录，禁止混用。
- 策略效果首先对齐真实更新数与 prompt group/trajectory 数；response 长度变化时
  另核对训练 token，并在差异明显时按共同累计 token 补充比较。等 rollout 周期只
  用于采集轮内曲线，不等于等训练量；rollout 次数、生成开销与吞吐另作效率结果。
- 按预先声明的 M_pre 或 M_post，在等训练数据预算下报告峰值和按 update/trajectory
  加权的越界率；不同策略经历的周期数不同，不直接比较越界周期的个数。
- KL 明确两端策略、方向、前缀分布和聚合权重。当前使用真实 rollout 保存的因果
  前缀，轮内固定、轮间重选，prompt 等权再按有效位置等权。跨 stage 隔离模型漂移
  需统一 probe q；换轮后的曲线断点不表示模型回退。
- 相邻 KL、相对 rollout anchor 的累计 KL、两分支间 KL 分开；全词表 KL 与
  sampled-token proxy 分开。训练侧漂移与 rollout 引擎 gap 分开，不能用相邻日志
  相减恢复相邻模型 KL。
- 逐步分析以 `kl_updates.jsonl` 为准。TensorBoard 的 `full_kl/age_XX/*` 横轴是
  rollout/global step，`full_kl_by_update/*` 横轴是真实更新数；tag 中的 age 不是横轴。
- 训练 token、能力评测和测量重复计算量分开。训练周期、测量、评测、保存及作业总
  耗时分别报告，已包含的开销不重复相加；核实吞吐是每卡还是全卡合计。

## 实验文档与命名规范

实验目录使用 `experiments/YYYYMMDD_<topic>_<variant>/`，小写 snake_case；日期为
正式建立日期，目录不因补跑或结果变化改名，Slurm job ID 不进入目录名。同一设置的
probe、正式运行和重跑放在各自 `raw/<phase>_job<id>/`。研究问题或主要对照改变时
另建实验。README 标题为 `YYYY-MM-DD：实验名称`。

多个可独立提交、解释的设置放在实验族下。实验族 README 只保留共同问题、公式、
共同配置和子实验索引；每个子实验各有 README、raw 与经确认后才建立的 Record。
设置特有的启动、状态和结果只写在对应子实验。经确认的跨设置分析另放
`comparisons/<analysis_name>/`，不覆盖子实验记录。

每个独立设置首先只建立 README，写清任务、与其他实验的具体关系、完整配置、提交
脚本/job ID/时间/依赖/远端输出/带核验日期的状态，以及已有产物入口。Slurm 完成不
自动产生分析结论；未要求分析时不建立 `docs/EXPERIMENT_RECORD.md`。

只有用户明确要求并确认分析方案后才建立 Record。它只记录分析问题与数据范围、结果、
图表、结论边界和复现方式，不重复配置或启动过程。数值明细放 tables，图放 figures；
Record 嵌入确认后的 PNG，注明问题、口径、观察、绘图脚本及必要的 PDF。被取代但需
保留的笔记放 `docs/HISTORICAL_NOTES.md`，开头注明不是当前配置或待执行任务。

```text
README.md  实验任务、关系、配置、启动、时间、状态和产物入口
docs/      经确认的 EXPERIMENT_RECORD.md；必要时保存 HISTORICAL_NOTES.md
scripts/   本地分析/绘图/归档脚本；远端提交脚本以代码仓库链接引用
raw/       原始产物，保留来源且不覆盖，不进入 Git
tables/    可复现生成的 CSV、JSON 等派生表
figures/   可复现生成的 PNG、PDF 图表
```

按实际产物建目录，不补空目录。分析流程仍为：提出候选问题、口径、指标和图表 → 用户
修改补充 → 评估数据支持与混杂 → 用户确认 → 生成派生表、图表和正式结论。未经确认
不新增派生指标或把探索性判断写成正式结论。

分析脚本按自身位置定位实验根目录，从 raw 只读输入，写 tables、figures 或 docs，
不依赖启动目录。搬动文件同步更新脚本与文档链接；原始数值不因重新分类而改写。
引用远端实现时记录代码 commit，避免代码更新改变旧实验的复现口径。

## Git 与代码隔离规范

本地研究档案和远端训练代码使用两个独立 Git 仓库；本地档案仓库不配置代码仓库的
origin，也不将档案推到代码 GitHub。AGENTS.md 在两边保留长期研究与协作边界，具体
实验档案只在本地仓库管理，训练实现只在远端代码仓库管理。

- 本地 `/Users/Workspace/dynamicStaleness`：论文、实验 README/Record、本地分析脚本、
  派生表、正式图表与必要操作资料进入本地 Git；raw 保留来源但不进入 Git。本地不
  维护训练框架源码，遗留未提交文件保留并单独说明。
- 远端 `/data/run01/scyb980/cyt/src/DynamicOffPolicy` 与 GitHub
  `Skialith/DynamicOffPolicy`：训练源码、配置、必要提交/测量脚本、回归测试和代码
  使用说明进入 Git；不收本地研究档案、论文、图表和分析脚本。
- 修改前检查对应仓库工作区，区分并保留已有用户改动；只暂存本次任务相关文件，
  检查 staged diff，运行风险相称的检查，一次提交只包含一个内聚变更。
- 两个仓库的改动分别提交并报告各自哈希；不覆盖或改写已有历史，不使用破坏性 reset。
  向代码 GitHub push 需有用户授权；本地档案提交不触发任何远端同步。
- 模型、optimizer checkpoint、环境、重复代码快照、缓存和密钥不进入任一仓库。
  原始资产与既有作业部署保留；代码同步只更新代码工作目录，不搬运实验产物。

## 必要基础与资源约束

基础研究对象为 Qwen3-8B-Base、全参数 GRPO、DAPO-Math-17K，沿用 SIS 数学预处理。
保持标准 GRPO clip，KL loss coefficient 与研究预算 δ 分开。学习率、长度、测量规模、
评测与保存策略以各实验 README 和提交脚本为准，不将某次设置视为全部实验默认值。

8B 正式训练与 GPU integration probe 在并行云 Slurm 执行；本地做资料维护与分析。
登录节点只做准备、提交、读日志和文件整理，不运行训练或 Ray/vLLM GPU resource probe。

```bash
ssh scyb980\@NMCC-N46H1\@ssh.paracloud.com -p 2222
```

训练代码目录为 `/data/run01/scyb980/cyt/src/DynamicOffPolicy`。共享模型、数据与环境
在 `/data/run01/scyb980/cyt/src/verl-staleness`，通过忽略的 assets/.venv 符号链接复用。
已有作业继续使用原部署；更新代码前核对工作区和使用该目录的作业，记录运行 commit。

- 统一用远端 `examples/dynamic_staleness/submit_slurm.sh` 提交。正式配方沿用已验证的
  A800 四卡 legacy FSDP + vLLM、sis_offload；改变 world size/runtime 另做真实 probe，
  两组四卡并行不等于验证了单个八卡训练。GPU 数与 FSDP/Ray world size 一致，
  rollout TP 能整除 GPU 数。
- sis_offload 在训练阶段外手动 offload actor parameter/optimizer 与 reference，
  与 FSDP2 offload_policy 区分。Fisher 测量借用训练 GPU，其开销单独记录。
- 训练使用按固定 revision 准备的持久化离线资产。可释放缓存与临时测量快照放
  `/tmp/ds-$SLURM_JOB_ID/`；原始数据、日志及明确要求保存的权重放持久目录。
- 不覆盖旧输出或猜测最新目录；提交前核对空间和保存策略。没有终点权重或 Adam
  状态时，不声称已获得可续训 checkpoint。
- 保留 legacy FSDP dynamic-batch 在实际 DP group 内同步 micro-batch 数的修复；
  Math-Verify 不放入 DAPO 线程池。缩小模型、batch 或长度的运行明确标为 probe。
