# 动态 Off-policy / Staleness：核心研究与实验约束

更新：2026-10-10。本文件只保留研究命题、核心 idea、共同计数口径、文档与代码隔离规范
和必要基础信息。具体实验的设置、启动与状态写入实验 README；只有确认分析后，数值
结果和图表解读才写入对应 `EXPERIMENT_RECORD.md`。本文件不累积单次运行日志。

- 本地研究资料入口：[实验索引](experiments/README.md)与[集群使用说明](集群使用指导/PARACLOUD_ENV_SETUP_ZH.md)。这些相对链接属于本地档案仓库。
- 训练代码入口：[Skialith/DynamicOffPolicy](https://github.com/Skialith/DynamicOffPolicy)。具体运行配方以对应实验 README 和远端提交脚本为准。

## 科研命题

在每次真实 optimizer update 的训练数据量固定时，保留旧策略 rollout 中的固定因果
前缀和该策略在这些位置上的全词表分布，在后续更新、甚至跨 rollout 周期后，用当前
模型前向直接计算相对旧策略的 KL。研究这种延迟观测能否以足够低的额外成本，筛选
值得尝试更大更新间隔 N 的时机，并经真实大 N 运行校准后逐级调整 N。

当前优先验证两个问题：

1. 保存旧 anchor 的前缀和输出分布，再做当前模型前向与 KL 聚合，存储、内存和
   端到端训练暂停成本是否可接受。
2. 较小 N 训练路径上的跨轮 KL，是否能有效筛选真实较大 N 运行中可接受的窗口。
   首个候选关系为 N=4 跨两轮的八步漂移与真实 N=8 轮内漂移的对应关系。

随后单独验证反馈规则的策略效果：在相同真实更新数、trajectory/token 和模型能力
约束下，能否减少 rollout 刷新开销、提高端到端训练效率，同时不增加校准后的
staleness 越界风险。观测成本、候选信号的有效性和策略收益分别验证。

## 核心 idea

当前顺序是：保留旧 anchor 的前缀与全词表分布 → 验证缓存和额外前向成本 → 在
较小 N 下直接观测跨轮 KL → 与真实较大 N 运行对照 → 校准经验阈值和测量触发规则
→ 在 rollout 边界逐级调整下一轮 N → 验证等训练数据预算下的效率、稳定性和能力。

Fisher、参数 norm 与 KL 的数学拟合路线暂缓，当前测量和控制不以这些量为前提。
不预设 KL、轮内峰值或可容忍 N 随训练单调变化；stage 按真实更新编号或明确
checkpoint 描述，同时核对学习率和观测分布。

### 固定每次 update 的数据量

共同口径为每次真实 update 使用 256 个 prompt group、每个 prompt 8 条 response，
即 2048 条 trajectory；`ppo_epochs=1`。一次 rollout 准备 `256*N` 个 prompt group，
划分成 N 个不同 mini-batch，各训练一次。

改变 N 时不缩小 mini-batch；复用的是行为策略版本，不是把同一 trajectory 重复训练
N 次。缩小 batch 的工程运行必须标为 probe，不与正式训练量混用。

### 旧 anchor 缓存与直接 KL

令 t 为旧 anchor 的真实更新编号，π_t 为训练侧策略，q_t 为其真实 rollout 中选定
的前缀和聚合权重。缓存 token IDs、预测位置、mask、权重，以及这些位置的旧策略
全词表 logits 或 log-probabilities；后续对同一前缀做当前模型前向，计算
`KL(π_t || π_current)`。旧 rollout 缓存用于观测，不作为重复训练同一 trajectory 的理由。

沿用既有 K 测量的选择口径：64 个 prompt group，每组一条有效 response，每条最多
8 个有效 response-token 位置；取预测该 token 的前一位置输出，保留完整因果前缀
与全词表，prompt 等权、组内有效位置等权。调整采样规模需写入对应实验配置，不
静默改变 response/position 选择、前缀或权重。

一个 anchor 的前缀、位置和权重在其整个观测窗口内固定；新 rollout 可另选新 anchor
的前缀，但不能覆盖仍需跨轮比较的旧缓存。保存所选位置的全词表分布，不要求保存
整批 rollout 全部 token 的 logits、旧模型权重、KV cache 或反传图。

当前模型使用完整因果前缀做无求导前向，不重新生成 response；同一 response 的多个
测量位置尽量共用前向。KL 不做 top-k/top-p 截断，不以 sampled-token log-prob 差或
有限 logits 差分二次量代替；log-softmax 和 KL 聚合至少用 FP32。两端模型精度和
执行路径应一致，跨路径复用分布须先验证数值差异不会影响阈值判断。

### 跨轮漂移与真实大 N 的区别

以 N=4 为例，π_old=π_t，第一轮结束为 π_(t+4)，第二轮结束为 π_(t+8)。保留 q_t
和 π_t 的分布到第二轮后，可直接得到旧 anchor 到八步后模型的 KL。它属于中间
刷新过 rollout 的 N=4 训练路径，不等同于连续八步使用 π_t 行为策略数据的真实
N=8 路径；后四步的数据、行为策略概率和 optimizer 更新可能不同。

```text
a = current_optimizer_update - anchor_optimizer_update
K_t^path(a) = E_(h~q_t) [KL(π_t(·|h) || π_(t+a)^path(·|h))]
```

`path` 明确刷新和训练数据安排。跨轮的 a=8 表示相对旧 anchor 已经过八次真实更新，
不能记作当前 rollout 的 `policy_age=8`。测量记录明确 anchor update、current
update、当前 rollout 的行为策略版本，以及该比较属于哪条训练路径。

较小 N 路径上的跨轮 KL 是升级候选信号，不是未执行的大 N 路径的实测结果或安全
上界。两者没有预设大小关系；先检验候选信号与真实大 N 轮内 KL 的对应关系，再
用于反馈规则。分支对照需匹配起点、optimizer/scheduler 状态、前段更新数据及后段
prompt 安排，明确后段行为策略刷新差异。

### KL 峰值、经验阈值与反馈规则

对已经声明的训练路径和固定 q_t，区分更新后与使用旧数据前的口径：

```text
M_post_t^path(A) = max {K_t^path(a): a=1,...,A}
M_pre_t^path(A)  = max {K_t^path(a): a=0,...,A-1}
```

轮内窗口取 A=N；跨轮窗口另声明 A 与 path。这里的 max 沿 anchor age 取值，每个
K 本身仍是前缀平均 KL，不是对状态取最大值。相邻 KL 之和不等于累计 KL。
轮内更新后口径为 `age_after=1,...,N`，使用旧数据前的口径为 `policy_age=0,...,N-1`；
声明约束范围后再计算容忍度，不混用两种峰值或跨轮 anchor age。

只测终点 K_t(8) 不能排除 age=5、6、7 的更高峰值。首轮验证应覆盖后段逐 update
测量，以评估稀疏观测会遗漏的峰值；后续测量频率依据成本和该验证结果确定。稀疏
采样的最大值须标为已测位置最大值，不将其写成整个窗口峰值或普适安全上界。

threshold 可先采用预先声明的经验候选值，结合已接受的基线和真实大 N 试运行校准。
δ 不等于 PPO clip 参数、KL loss coefficient 或 N_cap；数值、约束口径与适用设置
写入实验 README，当前不指定全实验通用值。KL 达标还需核对能力、ratio 尾部、
clip fraction 和 ESS。只测到某个 N 达标，不代表已找到最大可容忍 N。

规则先在 rollout 边界决定下一轮 N。候选流程为：按固定频率或近期已获得的 KL/
训练状态触发跨轮测量 → 连续观测满足升级条件 → 少量真实大 N 试运行 → 根据其
实测 KL 和训练表现保持或降级。可检验不同的升级/降级阈值以减少来回切换；等级、
触发频率、连续窗口数和阈值均为待验证设置，不提前固定为结论。

决定时只能使用已经获得的信息，不能回填尚未发生的 KL。事后观测用于后续决策，
不能撤销已经发生的越界，也不等于第 4 步已预测出后四步漂移。规则需按完整 rollout
或独立运行留出验证，分别检验候选信号的误判和真实升级后的越界、能力及效率。

### 成本与实现优先级

新方向复用直接 KL 路径，仅需分布缓存、当前模型无求导前向与聚合。新测量路径
不得调用 Fisher/FVP/HVP、分层 JVP/VJP、谱迭代、测量用参数位移或 Q/B 计算，
也不依赖 `measure_training_fisher.py`、`layerwise_fisher.py` 等求导入口。新分支与
worktree 实际删除这些实现、训练接入、配置及旧提交/probe 入口，不仅关闭开关；
历史实现保留在原主 worktree 和 Git 历史中。正常训练的 backward/optimizer 不受影响。

新路线不提供历史 JVP/HVP 配置；启动脚本拒绝残留的旧求导环境设置，旧 Hydra
配置不再受支持。验证代码清理和调用链后，再用真实 GPU probe 确认无求导前向的
KL、成本和数值一致性。是否采用独立测量模型，以该验证为准。

分别测量首次分布计算与保存、缓存字节数和内存/显存峰值、额外前向与 KL 聚合、
以及实际训练暂停。若实现涉及模型快照、装载、offload/reload、分布传输或多卡同步，
这些成本均计入端到端暂停；区分每个 anchor 的首次缓存成本与后续重复测量成本，
并报告摊销后的训练开销。
既有 K/B 报告包含的装载、位移等工作不能直接当作一次额外 KL 前向成本。

等真实更新数下，增大 N 不减少应生成的 trajectory 总量；潜在收益来自减少刷新、
模型切换及改善批处理效率。比较实测端到端收益与新增观测开销，不把 N 加倍解释
为生成 token 或 rollout 生成耗时自动减半。缓存保留数量和释放时机写入实验配置，
原始日志和需归档的前缀数据按既有 raw 规范保存。

### 暂缓路线

Fisher 矩阵、λmax、FVP/HVP、norm→KL 拟合、Q/B/R 数学解释和基于未来位移
预测的 controller 暂缓。已有实现保留在原主 worktree 与 Git 历史，实验档案保留；
新方向不继续优化或默认运行这些测量，只有明确重新启用时才恢复相应验证要求。暂缓不改写已有实验
结论，也不改变已提交作业。先前否决的有限输出差分代理与梯度差分 FVP 不因转向
而恢复为当前路线。

## 共同实验口径

- 一个研究 update 是一次真实 `optimizer.step()`；一个 verl `global_step` 是一轮
  rollout 加随后 N 次 update。分别记录，禁止混用。
- 策略效果首先对齐真实更新数与 prompt group/trajectory 数；response 长度变化时
  另核对训练 token，并在差异明显时按共同累计 token 补充比较。等 rollout 周期只
  用于采集轮内曲线，不等于等训练量；rollout 次数、生成开销与吞吐另作效率结果。
- 按预先声明的 M_pre 或 M_post，在等训练数据预算下报告峰值和按 update/trajectory
  加权的越界率；不同策略经历的周期数不同，不直接比较越界周期的个数。
- KL 明确两端策略、方向、训练路径、前缀分布和聚合权重。当前使用真实 rollout
  保存的因果前缀，各 anchor 的整个观测窗口内固定，新 rollout 可另选新前缀；
  prompt 等权再按有效位置等权。跨 stage 隔离模型漂移需统一 probe q；换 anchor
  后的曲线断点不表示模型回退。
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

实验结束后可直接将日志、配置、JSON/JSONL、逐 age 报告和 TensorBoard 原始产物
归档到本地对应实验的 `raw/<phase>_job<id>/`，无需另行确认；保留来源且不覆盖原件，
不复制模型权重、Adam 状态或临时大张量。

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

远端仓库主目录为 `/data/run01/scyb980/cyt/src/DynamicOffPolicy`；跨轮 KL 路线使用
分支 `codex/delayed-anchor-kl` 与独立 worktree
`/data/run01/scyb980/cyt/src/DynamicOffPolicy-delayed-kl`，新路线的代码与提交在该
worktree 中进行，原目录保留 Fisher 基线。共享模型、数据与环境在
`/data/run01/scyb980/cyt/src/verl-staleness`，通过忽略的 assets/.venv 符号链接复用。
已有作业继续使用原部署；更新代码前核对工作区和使用该目录的作业，记录运行 commit。

- 统一用远端 `examples/dynamic_staleness/submit_slurm.sh` 提交。正式配方沿用已验证的
  A800 四卡 legacy FSDP + vLLM、sis_offload；改变 world size/runtime 另做真实 probe，
  两组四卡并行不等于验证了单个八卡训练。GPU 数与 FSDP/Ray world size 一致，
  rollout TP 能整除 GPU 数。
- sis_offload 在训练阶段外手动 offload actor parameter/optimizer 与 reference，
  与 FSDP2 offload_policy 区分。额外 KL 前向借用训练 GPU，其端到端开销单独记录。
- 训练使用按固定 revision 准备的持久化离线资产。可释放缓存与临时测量快照放
  `/tmp/ds-$SLURM_JOB_ID/`；原始数据、日志及明确要求保存的权重放持久目录。
- 不覆盖旧输出或猜测最新目录；提交前核对空间和保存策略。没有终点权重或 Adam
  状态时，不声称已获得可续训 checkpoint。
- 保留 legacy FSDP dynamic-batch 在实际 DP group 内同步 micro-batch 数的修复；
  Math-Verify 不放入 DAPO 线程池。缩小模型、batch 或长度的运行明确标为 probe。
