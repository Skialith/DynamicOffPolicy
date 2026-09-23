# 第二轮：N=4 / N=8 / N=16，96 次全参数更新与 KL 测量

本文保留该轮历史设计与配置。后续无 warmup、等 rollout 周期数的 N=4/8 实验见
[当前实验入口](../../experiments/20260909_full_kl_no_warmup_n4_n8/README.md)；当前共同约束以 AGENTS.md 为准。

更新日期：2026-09-06。状态：代码已实现，CPU 针对性测试与配置解析通过。N=4/8 已完成；
按用户最新决定，以相同口径新增 N=16 四卡正式作业；156023 在训练前因 NCCL CUDA OOM
失败，替代作业为 156172。独立 GPU 集成验证未执行，
不能记为通过。具体状态见
[第二轮运行记录](FULL_PARAMETER_KL_RUN_RECORD.md)。
资源约束见 [AGENTS.md](../../AGENTS.md)，首轮结果见
[首轮实验记录](../../experiments/20260902_staleness_pilot/docs/EXPERIMENT_RECORD.md)。

## 1. 当前范围

本轮观察 N=4/8/16 各自的相邻模型 KL，同时收集本轮 rollout 起点到当前模型的累计 KL。
N=4/8 终点双向 KL 已完成；N=16 完成后保存终点权重，再与既有终点做共同前缀比较。
暂不实现 maximal 搜索、动态 N controller、梯度预测模型或
新增全量参数位移测量；已有 gradient norm 和训练诊断继续保留。

按最新讨论，原先的 100 次更新改为 96 次，不再开发非完整尾周期。全程固定 probe 不作为
本轮前提：使用真实 rollout 的已保存前缀，同一次比较复用相同 token IDs，不要求不同
策略重新生成相同回答。

| 项目 | N=4 | N=8 | N=16 |
| --- | ---: | ---: | ---: |
| 实际 optimizer updates | 96 | 96 | 96 |
| 完整 rollout 周期 | 24 | 12 | 6 |
| 每次 update 的 prompt group | 256 | 256 | 256 |
| 每个 prompt 的 response | 8 | 8 | 8 |
| rollout batch 的 prompt group | 1,024 | 2,048 | 4,096 |
| 实际训练 prompt group 总数 | 24,576 | 24,576 | 24,576 |
| 实际训练 trajectory 总数 | 196,608 | 196,608 | 196,608 |
| 相邻 KL 记录数 | 96 | 96 | 96 |
| MATH500 sampled Avg@1 检查点 | 0 / 48 / 96 | 0 / 48 / 96 | 0 / 48 / 96 |

三组保持 Qwen3-8B-Base 全参数更新、DAPO-Math-17K、GRPO、ppo_epochs=1、4×A800-80GB、
TP=2、sis_offload、长度 1024/3072 和现有 GRPO loss 配置不变。

本轮代码采用以下配置，运行是否完成以运行记录和作业产物为准：

- 共同起点：同一固定 revision 的 Base 权重，各自初始化 optimizer/scheduler，seed=1，
  使用相同 prompt 顺序。提交前锁定模型和数据路径，不默认接首轮 update 100 checkpoint。
- 学习率：峰值 1e-6，10 次真实 optimizer update 线性 warmup，计入 96 次更新，之后恒定。
- MATH500：do_sample=true、n=1、temperature=1.0、top_p=0.95、top_k=-1，沿用数学模板、
  长度上限和 Math-Verify。top_p=0.95 来自 SIS 公开脚本；论文 Table 4 是 0.7，两者不同，
  本轮按公开脚本拟定并在提交前冻结，不混入历史 greedy@1 曲线。
- KL：所选前缀上的全词表 next-token KL，temperature=1.0、不做 top-p/top-k 截断，与能力
  评测采样设置分开。模型精度沿用基线，log-softmax、概率与 KL 累加至少用 FP32。

若改为 checkpoint 恢复，须重新明确绝对 update 编号和 optimizer/scheduler 状态；
不能在恢复已完成 warmup 的状态后又默认热身 10 步。

## 2. warmup 的来源与计数

SIS 公开 [8B GRPO 脚本](https://github.com/liyu199809/SIS/blob/main/examples/train/math_simple_rl/train_qwen3_8b_grpo.sh)
设置 lr_warmup_steps=10，但参数名不定义计数单位。核对公开
[SIS FSDP worker](https://github.com/liyu199809/SIS/blob/main/verl/verl/workers/fsdp_workers.py)
可见，update_actor 在整个 actor.update_policy(data) 返回后调用一次 actor_lr_scheduler.step()；
[actor](https://github.com/liyu199809/SIS/blob/main/verl/verl/workers/actor/dp_actor.py) 内部按
mini-batch 执行 optimizer update。本地 worker 也沿用了这个周期级推进位置。因此不能把
公开配方中的 10 个 scheduler step 直接说成 10 次 mini-batch 参数更新，也不能据此确认
论文 Figure 3 的全部运行配置。

本轮拟显式采用 optimizer-update 计数，第 s 次更新实际使用：

    lr_used(s) = 1e-6 * min(s / 10, 1), s=1,...,96

第一更新用 1e-7，第十更新达到 1e-6；第 11–96 次恒定。检查初始化的 0/1 索引，不能只
移动 scheduler.step() 而不核对第一步 LR。每次实际 optimizer.step() 后推进 scheduler，
新模式取消周期末的重复推进，并记录本次真正使用的 LR，而非下一步的 LR。

非有限梯度或 AMP 跳步不能计作成功更新，也不推进 scheduler。若未完成约定的 96 次更新，
报告验收失败，不自动补采样或改变 N。新计数模式显式启用，旧实验默认行为保持原口径；
禁止把旧周期级 scheduler state 不加说明地恢复进新模式。

## 3. 三组 KL 的定义与测量数据

令 π_s^(N) 为分支 N 完成 s 次更新后的策略，τ 为本轮 rollout 的起点 update。
H_τ^(N) 为本轮真实 rollout 中选出的前缀集合，q_τ^(N) 为声明的聚合权重。

### 一轮内复用上下文

建议每轮抽取一组真实 rollout 前缀，在该轮全部 N 次更新中复用，使一次测量前向能同时
支持相邻与累计 KL。下一轮重新选择，不建立额外固定题库，不增加训练 rollout。

正式运行每轮全局抽取 64 个 prompt group、每题 1 条 response、每条最多 8 个有效回答
位置，最多 512 个上下文。三组取样规则和规模一致；prompt 等权、题内位置等权，记录短
回答的有效位置数和权重。缩小 batch 的 probe 若不足 64 个有效 group，则全部选入；
它不作为正式测量精度或吞吐的结论。

保存 prompt/response 标识、token IDs、mask、position IDs、所选位置和权重。用独立测量
seed 选取已有数据，不消费训练 RNG。比较时进行 teacher-forced 前向评分而非重新 generate；
因果 mask 保证每个位置只观察其前缀。

### 相邻 KL

    d_s^(N) = E_(h~q_τ^(N)) [KL(π_(s-1)^(N)(·|h) || π_s^(N)(·|h))]
    s=τ+1,...,τ+N

每组保留 0→1 到 95→96 的 96 个值。新 rollout 到来后先用 θ_τ 在新 H_τ 上评分，再执行
第一次更新，不能把上轮不同上下文的概率缓存拿来计算新轮的相邻 KL。

### 本轮累计 KL

    K_τ^(N)(a) = E_(h~q_τ^(N)) [KL(π_τ^(N)(·|h) || π_(τ+a)^(N)(·|h))]
    a=0,...,N

每次更新后记录 K_τ(a)，K_τ(0) 用作自检。分别记录更新前 policy_age=a-1 和更新后
age_after=a；K_τ(N) 是周期末终点，不是该轮使用旧数据前的 age=N。累计 KL 直接比较
两端分布，不是单步 KL 求和，也不是相邻旧日志相减。

### 终点双向 KL

从两分支保存的 rollout 前缀中各抽取 32 个 prompt 的已选回答位置，合成共同 H_end，记录来源 update 与
分支权重。顺序加载两个第 96 步模型，用同一测量入口对整个 H_end 分别评分：

    D_end(4→8) = E_(h~q_end) [KL(π_96^(4)(·|h) || π_96^(8)(·|h))]
    D_end(8→4) = E_(h~q_end) [KL(π_96^(8)(·|h) || π_96^(4)(·|h))]

两方向使用同一 H_end 和权重，不是各自在自己的前缀上算一个方向后直接并列。可离线计算，
但必须保存终点权重，不能默认 branch 的 SAVE_FREQ=-1 会保存模型。

全词表指每个选定前缀对所有词表项求和，不是覆盖所有训练 token 或全部可能状态。
每轮 q_τ 可以变化，三组轨迹也会分化，曲线反映各自真实 rollout 分布上的漂移，不能单独
证明排除观测分布影响后的 stage 效应。本轮不因此扩展固定跨 stage probe 或 maximal 实验。

## 4. 最小代码改动

### A. 入口与配置

- 在 run_staleness.sh 增加本轮测量配置入口/开关，绑定 96、N、update 级 scheduler、
  KL 与 sampled eval；新字段在对应 Actor/optimizer 配置中正式声明，保留旧实验默认值。
- 复用现有 anchor 的 fresh-start 能力分别启动三组，不用 n4_then_n8 或历史 branch
  恢复流程冒充共同 Base 起点；仍经 submit_slurm.sh 提交，不重构提交系统。
- trainer.total_training_steps 仍是 rollout 数，分别为 24/12/6；scheduler 的 96 是
  optimizer-update 数，两层不能混用 total_steps。
- 当前 TEST_FREQ 仍按周期计数，N=4/8/16 分别设 12/6/3，配合起点评测得到 0/48/96。

### B. scheduler 与真实更新

- verl/workers/fsdp_workers.py：把同一 scheduler 实例及推进模式传给 actor，checkpoint
  manager 仍保存该实例；新模式取消周期末 step，不能出现两个各自推进的 scheduler。
- verl/workers/actor/dp_actor.py：在完整 optimizer update 边界记录 lr_used、实际执行状态
  和 update 编号，并推进 scheduler；保留已有梯度裁剪和全局 grad_norm 口径。
- 测试新旧两种推进模式，不增加旧 checkpoint 的自动计数换算功能。

### C. 全词表前向与缓存

- 在 actor 中增加测量专用的选定位点全词表 log-prob 前向。现有返回值是答案 token 的
  log_probs，不能直接计算全词表 KL，也不能用 top-k 截断冒充 full-vocabulary。
- 纯 KL 数学和加权聚合放入小型辅助模块，独立测试。按微批/位置分块控制 B×L×V 峰值，
  不保存全部历史 logits，不常驻第二套 8B 训练模型。
- 本轮缓存 anchor 和 previous 概率；每次更新后评分，得到 d_s 和 K_τ(a)，再替换 previous。
  anchor 本轮不变，下轮重建。缓存可放 CPU，必须校验上下文和位置对应关系。
- 测量使用固定模式和无梯度前向，恢复原模式/RNG，不触碰未完成的梯度累积，不调用
  vLLM rollout 或刷新行为策略版本。
- 新测量前向每次只处理一条去 padding 序列，按真实 DP group 的 all_reduce(MAX) 对齐
  执行次数；没有选中样本的 rank 也执行占位前向。已有三个 dynamic-batch 入口及其 AST
  同步检查不变，另有四卡测试覆盖不同 rank 选中 0/1/2/3 条回答的情况。

### D. 记录、评测、保存与分析

- trainer 接收逐 update 记录，由单一写入者持久化，不将 N 次更新平均成一条。
  分布式按真实上下文权重归并，不能在样本量不同时直接平均 rank 均值。
- kl_updates.jsonl 至少记录 run/seed/N、相对/绝对 update、rollout 编号、起点 τ、
  policy_age/age_after、lr_used、context_set_id、上下文数、相邻/累计 KL、聚合口径和
  测量耗时。已有 update_* / rollout_* sampled-token 指标分开命名。
- TensorBoard 同时保留两种视图：`full_kl/age_XX/*` 继续用 rollout/global step 观察固定
  age 的跨周期变化；`full_kl_by_update/{adjacent_kl,cumulative_kl,grad_norm,lr_used,policy_age}`
  使用真实 optimizer step，形成连续 1–96 曲线。只向 TensorBoard backend 追加后一组，
  不改变常规训练指标的横轴，也不向 console/W&B 重复发送。
- 已完成的 N=4/8 从各自 `kl_updates.jsonl` 回填连续 tags，不重跑训练；每个 tag 必须恰有
  96 个 event，event step 为 1–96，数值与 JSONL 逐点一致。
- MATH500 保存逐题采样回答、0/1 reward、解码参数和独立 eval seed，汇总 sampled Avg@1；
  不拼接历史 greedy 曲线，不把 best@1 以外的 best@k 指标误用作 Avg@k。
- 在线采样评测仅在完整周期边界执行，并验证不改变训练 rollout 的 RNG。只恢复 trainer
  的 Python/Torch RNG 不等于恢复 vLLM worker 采样状态；若在线接口无法隔离，改为保存
  对应权重后独立评测，先核查空间，不默默改变训练随机过程。
- 通过现有 checkpoint 管线的 save_contents=[hf_model] 保存终点权重、tokenizer/config，
  不保存 optimizer state；该输出不承诺断点续训。在线评测使用每条 request 独立 seed；
  Python/NumPy/Torch 的测量与验证 RNG 也恢复。独立评测若需第 48 步权重，另计空间。
- N=4/8 作业分别校验自己的 96 条 KL 和 0/48/96 评测产物后登记 ready；后完成的
  作业在自己的四卡 allocation 中顺序加载两个模型、自动计算终点双向 KL，无额外等待作业。
- N=16 使用同一训练入口独立保存 96 条 KL、三次评测和终点 HF 权重；完成后先做相同
  产物验收，再在共同前缀上补充与 N=4/8 终点的双向 KL，不改变训练期间的测量分布。
- 离线终点 KL 与绘图入口复用同一 KL 数学/前向协议，产出相邻曲线、累计数据、终点双向
  KL 和 0/48/96 sampled Avg@1，不增加预测模型。

已从集群取得本轮所需的匹配 trainer/config/agent-loop 源码，并确认原有 actor/worker/
启动脚本与本地一致。本地和集群均无 Git 元数据；部署前保留被替换源码备份，不用上游文件
覆盖已有 staleness/eval 修复。本地仍为部分源码副本，运行验证在集群进行。

## 5. 实施顺序与验收

1. 配置与 scheduler：已有 CPU 单测验证 N=4/8 的 96 个 lr_used 完全一致；N=16 复用同一
   update 级 scheduler，配置展开确认 96 次更新和 6 个周期。检查首步、第十步、跨周期、
   跳步行为及新旧模式是否重复推进。
2. KL 数学与取样：验证小词表已知分布、自 KL 近零、方向性、mask/位置对齐、短回答权重、
   分块与分布式聚合一致性，以及换轮缓存切换。
3. Actor 与记录：本次直接检查正式作业首个完整 rollout 周期的逐 update 记录、KL 有限性、
   LR 和计数，以及换轮 anchor 切换。小模型测量开关对照未执行，不宣称已验证 GPU 上
   更新结果与 RNG 完全一致。
4. 评测与保存：只在 0/48/96 评测，n=1 且 do_sample=true；验证平均 reward、eval RNG 隔离，
   终点保存重载和双向 KL 全链路。
5. 独立 4 卡集成 probe：按用户决定取消，不再作为本次正式运行的前置条件；保留脚本供
   需要排错时使用。首次 Adam 显存、DP 同步和测量耗时直接在正式运行前几步观察。
6. 三组正式作业无前置依赖：各 96 次实际更新、24/12/6 个完整周期、各 96 条相邻和
   更新后累计 KL、0/48/96 sampled Avg@1。用户接受首次实际运行暴露问题后
   可能需要修复并重跑；保留配置和明确输出路径，禁止覆盖旧结果。

入口：FULL_KL_EXPERIMENT=1，经 submit_slurm.sh anchor 提交；N=4/8 另设相同的
KL_PAIR_DIR，启用产物验收和自动终点比较。N=16 使用独立输出实例，完成后再验收并补充
终点比较。kl_probe 脚本保留但本次不执行，其中包括
更新/RNG 的 FSDP 对照、小模型 N=4/8 全链路及 8B 正式长度短测。实际状态见运行记录。
