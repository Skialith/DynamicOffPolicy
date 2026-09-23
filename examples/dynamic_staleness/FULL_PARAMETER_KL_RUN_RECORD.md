# 第二轮全参数 KL：实现与运行记录

更新：2026-09-06。研究协议见 [实验计划](FULL_PARAMETER_KL_PLAN.md)。

## 已完成的实现和 CPU 验证

- 选定真实 rollout 前缀上的全词表 KL；逐 update 同时记录相邻与本轮累计 KL。
- 每轮 64 个 prompt group × 1 条 response × 最多 8 个位置；保存 token、位置和权重。
- 各分支 96 次真实更新；10 次 update warmup，首步 1e-7、第十步 1e-6，之后恒定。
- sampled Avg@1，MATH500 update 0/48/96，T=1、top_p=0.95、n=1；保存逐题结果和 seed。
- 训练/评测逐 request 采样种子分离，测量/验证保存并恢复进程 RNG。
- HF 终点权重保存；后完成的作业自动在共同前缀上计算两个终点的双向 KL。
- 8 项 CPU 单元测试通过：已知 KL、方向/自 KL、分块、两组 LR、短回答/聚合权重、
  RNG、前缀/预测位置对齐、多卡调用结构、非有限梯度不推进 scheduler，以及实际
  trainer 的上下文序列化和逐 update 记录。非有限梯度测试中的 nan 警告是预期输入。
- 已有三个 dynamic-batch 入口的同步检查通过；shell/Python 语法检查通过。
- Hydra dry-run：N=4/8/16 分别为 24/12/6 周期，test_freq 分别为 12/6/3。

## TensorBoard 双视图修正

- 原 `full_kl/age_XX/*` 的横轴是 rollout/global step，age 是 tag 的组成部分；保留该组，
  用于比较固定 policy age 在不同 rollout 周期的变化。
- 新增 `full_kl_by_update/{adjacent_kl,cumulative_kl,grad_norm,lr_used,policy_age}`，只写入
  TensorBoard backend，每条 event 使用记录中的真实 `optimizer_step`，不改变训练过程、
  JSONL、常规 rollout 指标或其他 logger backend。
- N=4/8 已从各自 `kl_updates.jsonl` 回填。EventAccumulator 验收：两条 run 均为 5 个新 tag，
  每个 tag 96 个 event，steps 严格为 1–96，数值逐点与 JSONL 一致。
- 修正后的 8 项 CPU 定向测试通过；集群源码备份为
  `logs/code_snapshots/tensorboard-by-update-before-20260906/`，当前源码快照为
  `logs/code_snapshots/tensorboard-by-update-20260906.tar.gz`。

## 四卡集成验证

- Slurm 作业：154018，名称 fullkl-validation-g4，分区 gpu_a800，4×A800，TP=2。
- 日志：`/data/run01/scyb980/cyt/src/verl-staleness/logs/slurm/fullkl-validation-g4-154018.out`。
- 2026-09-03 14:19 CST，按用户要求在 PENDING 状态取消，Slurm 确认为 CANCELLED by 2506；
  未实际执行，不能称为四卡验证通过。不再是本次正式作业的前置条件。
- 原计划检查顺序（本次未执行）：四卡 tiny FSDP 测量开关对照（含空样本 rank）→ Qwen3-0.6B N=4/8 各
  16 update、0/8/16 小样本评测和 HF 保存/终点 KL → 8B、1024/3072 长度、N=4 的
  4 次真实更新（小 batch，不保存 8B probe 权重）。
- 脚本保留供排错使用；本次未获得 FULL_KL_PROBE_PASSED。

## 存储和源码

- 首次提交前账户配额 250 GiB，已用约 125.4 GiB；不是共享文件系统整体剩余空间。
- 正式作业只存终点 HF 权重，不保存 Adam 状态，不删除任何历史实验产物。
- 原始源码备份：集群 `logs/code_snapshots/kl96-before-20260903/`。
- 本轮源码快照：集群 `outputs/full_kl_pairs/20260903-124556_seed1_g4_tp2/code_snapshot.tar.gz`。
- 两组共同 Base：`assets/models/Qwen3-8B-Base/`，revision
  `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`；共同训练数据
  `assets/datasets/SIS-Math-GRPO/data/pilot-seed1-u0292.parquet`。
- 终点文件不包含完整续训状态；任务中途失败需重新启动 fresh 分支，不能伪装成恢复。

## 正式作业

两个四卡作业已提交，gpu_a800、TP=2、sis_offload、seed=1；每组 96 次真实全参数更新，
Slurm 时限 24 小时。原 afterok:154018 依赖已按用户最新授权移除，直接用正式作业的
前几步检查运行；用户接受可能需要修复并重跑。未重新提交或改变两个正式作业的训练配置。

| 分组 | 作业号 | Rollout 周期 | 日志（集群仓库相对路径） |
| --- | --- | --- | --- |
| N=4 | 154027 | 24 | `logs/slurm/fullkl-n4-g4-154027.out` |
| N=8 | 154028 | 12 | `logs/slurm/fullkl-n8-g4-154028.out` |

共同结果目录：
`/data/run01/scyb980/cyt/src/verl-staleness/outputs/full_kl_pairs/20260903-124556_seed1_g4_tp2/`。
两组各自训练目录在启动时生成，包含启动时间、job id、GPU/TP/seed；训练产物验收后，
共同目录的 ready_n4.json / ready_n8.json 保存准确路径，endpoint_kl.json 保存终点双向 KL。
状态历史：2026-09-03 12:49 CST，两组为 PENDING (Dependency)。
2026-09-03 14:19 CST，scontrol 已确认 154027 / 154028 的 Dependency=(null)、
EligibleTime=2026-09-03T14:19:20，均仍为 PENDING；不再受验证作业阻塞。两组各申请
gres/gpu=4，尚无实际训练步可检查。后续首个周期检查 update 计数、warmup、相邻/累计 KL
和非有限值/显存/多卡错误；排队与依赖清空不等于运行验证通过。

## 完成状态与 N=16 扩展

- N=4（154027）于 2026-09-04 15:31:55 完成，Slurm `COMPLETED / 0:0`；96 条 KL、
  0/48/96 评测和终点 HF 权重通过产物验收。训练目录：
  `/data/run01/scyb980/cyt/src/verl-staleness/outputs/dynamic_staleness/20260903-200830_job154027_g4_tp2_seed1/full_kl_n4_u0096/`。
- N=8（154028）于 2026-09-05 09:33:15 完成，Slurm `COMPLETED / 0:0`。训练目录：
  `/data/run01/scyb980/cyt/src/verl-staleness/outputs/dynamic_staleness/20260904-153215_job154028_g4_tp2_seed1/full_kl_n8_u0096/`。
- N=4/8 的共同终点比较已完成：`D_KL(N4 || N8)=0.003147946263751314`，
  `D_KL(N8 || N4)=0.0035637222167339295`；64 个 prompt、512 个位置、全词表、T=1。
  产物为共同结果目录下的 `endpoint_kl.json`。
- N=16 四卡正式作业 156023 于 2026-09-06 15:52:20 提交，分区 gpu_a800、TP=2、
  sis_offload、seed=1、无依赖、时限 24 小时。96 次真实更新对应 6 个完整 rollout 周期，
  训练 batch 为 4096 个 prompt group，每次 update 仍为 256 个 prompt group × 8 responses；
  test_freq=3，对应 update 0/48/96。日志：`logs/slurm/fullkl-n16-g4-156023.out`。
- 156023 于 2026-09-06 16:45:49 启动，16:47:18 以 `FAILED / 1:0` 结束。失败发生在
  reference FSDP 初始化，NCCL 报 `Cuda failure 2 'out of memory'`；尚未开始 rollout 或
  optimizer update，因此没有可用的 N=16 实验结果，且与 TensorBoard 记录修改无关。
- 替代作业 156172 于 2026-09-06 17:48:26 按原 N=16 配置提交，并包含修正后的连续
  TensorBoard tags。分区 gpu_a800、4 卡、TP=2、sis_offload、seed=1、无依赖、时限 24 小时；
  日志：`logs/slurm/fullkl-n16-g4-156172.out`。截至 17:48 CST 为 `PENDING (Priority)`，
  scontrol 确认 `Dependency=(null)`、`gres/gpu=4`。
- N=16 提交前账户根目录使用 200,987,918,130 bytes（约 187.2 GiB）/ 250 GiB；既有
  N=4、N=8 目录各约 31 GiB。N=16 仍只保存一份终点 HF 权重，不保存 Adam 状态。
