# 2026-10-05：N=8 等 96 次更新 Fisher K/Q/B

## 任务与关系

按照用户确认设置，完成 96 次真实 update，同时逐 age 测量 K、Q、B_hat。
公式、精度与口径见 [实验族 README](../README.md)。与 N=4 对照按更新数及 trajectory
数对齐，本设置共 12 轮；不是旧基线的 N=8、192 updates。

## 配置

| 项目 | 正式设置 |
| --- | --- |
| 模型 / 数据 / 算法 | Qwen3-8B-Base；DAPO-Math-17K SIS 预处理；全参数 GRPO |
| N / rollout batch / update mini-batch | 8 / 2048 prompts / 256 prompts；8 responses/prompt，ppo_epochs=1 |
| updates / trajectory 总量 | 96 / 196608；实际训练 token 另记 |
| seed / LR / warmup | 1 / 1e-6 固定 / 0 |
| 长度 / KL loss / PPO clip / entropy | 1024+3072；1e-3；0.2/0.2；0 |
| 资源 / runtime | 4×A800-80GB；TP=2；sis_offload；legacy FSDP+vLLM |
| 打包 / vLLM | actor、infer 每卡 16384 token；utilization=0.7，batched=8192，seqs=512 |
| reward / 评测 | 8 Ray Math-Verify workers；MATH500 sampled Avg@1，训练前和每轮后 |
| KL 前缀 | 每轮 64 prompts×每个所选 response 最多 8 位置，固定轮内 q；全词表 |
| λ / Q / B_hat | 每轮 anchor λ；每 age 精确 HVP；残差 1e-3，cap=200，超限失败 |
| 保存 | 原始前缀、逐 update 标量、评测生成、计时、显存、配置和日志；不保存训练后权重或 Adam |

## 前置集成 probe

同一 8B、四卡、完整 1024/3072 长度上限、真实 rollout 与完整 64-prompt q，执行一轮
N=8。仅训练 mini-batch 缩为 8 prompts（rollout batch=64），8 次 update；禁用 MATH500
benchmark 与权重保存。190665 实际仍触发末轮训练集小验证，见终态记录。
它验证 export/offload/子进程四卡/HVP/reload 与八个 age 的对接，
不是能力、训练吞吐或正式曲率结论。逐步 K/Q/B 缺失、λ 未收敛、首步分片 norm 与
完整差值不一致或 scratch 残留都会使 probe 失败；正式两组设置 `afterok` 依赖。

## 提交与状态

部署：`/data/run01/scyb980/cyt/src/fisher_kqb_20261005/`；复用原仓库 Python 与模型
资产，不复制基座。入口：

```bash
bash experiments/20261005_fisher_kqb_n4_n8/scripts/submit_experiment.sh probe
SLURM_DEPENDENCY=afterok:<probe-job-id> bash experiments/20261005_fisher_kqb_n4_n8/scripts/submit_experiment.sh n8
```

probe 时限 24 小时，正式时限 7 天；时限不是可靠完成时间预测。成功提交也不表示
集成通过，正式作业的启动必须满足新 probe 的 `afterok` 依赖。

### 首次集成 probe：188038

提交 2026-10-05 14:41:35；开始 14:41:52；结束 14:54:30，Asia/Shanghai，
elapsed 12:38。2026-10-05 重新核验 `FAILED 1:0`；实际 4 GPU、32 CPU、502400 MiB
主存。部署源码为 `27aac9b`。日志：独立部署下
`logs/slurm/fisher-kqb-integration-g4-188038.out`。

真实 rollout、64-prompt 前缀保存、FSDP full-state 导出及 offload 后进入独立测量。
第一个 anchor、第一次 HVP 的第二次 `autograd.grad` 在 GPU 3 OOM，追加申请 2.32 GiB
时只余 667.56 MiB；该测量进程已占 74.98 GiB，PyTorch allocated 63.31 GiB、reserved
未用 11.18 GiB。没有完成一次幂迭代或真实 optimizer update，不记为通过。

确定输出实例：`20261005-144152_job188038_g4_tp2_seed1_hvp_integration_probe`，其下
`fisher_kqb_probe_n8_u0008/` 位于 `n8/raw/integration_probe/`。持久产物约 7.6 MiB，
不含训练权重；异常路径会清理私有 anchor/cache，不改写基座资产。用户用量中断发生
在提交后；当时正式两组尚未排队。

修复方向：保持全词表、64 prompts、长度上限和 FP32/FP64 精度，二阶图 saved tensors
放普通 CPU memory（不累积大规模锁页缓存），但已常驻 GPU 的权重/方向不重复拷贝；第一次梯度值完成
contraction 后立即释放，测量子进程启用 expandable segments。仍为精确双反向 AD，
不使用差分、截短正式测量或降低精度。
新作业首次 anchor 在八十亿参数测量前，以四卡微型 Qwen3 比较常驻图 HVP 与 offload
HVP（逐参数 atol=1e-6、rtol=1e-5）；不一致则直接失败。这是新 GPU 搬运路径的
数值自检，不代替真实 8B 的显存与全 age 集成验收。

复测准备：最终选择性 offload 的六项训练测量 CPU 检查通过；既有三项精确 HVP
回归检查通过；编译、shell 语法与 diff 检查通过。GPU 搬运和真实 8B 的显存是否
通过，以重提交作业为准，不由 CPU 检查推出。

### 修复重提与正式排队

源码 `3ece08a`；核心文件部署摘要与本地一致，显式加载 miniforge3 和 CUDA 12.8。
2026-10-05 18:27:40（Asia/Shanghai）核验：

| phase | job ID | 提交时间 | 开始时间 | 依赖 / 当前状态 |
| --- | --- | --- | --- | --- |
| 修复后集成 probe | 188641 | 18:18:06 | 18:18:31 | 无；`RUNNING` |
| N=8、96 updates 正式 | 188660 | 18:20:16 | 尚未开始 | `afterok:188641`；`PENDING (Dependency)` |

probe 在 `d1n41a28g03`，实际 4×A800-SXM4-80GB、32 CPU、502400 MiB 主存。
已进入 anchor=0、age=0 的测量子进程。四卡、2200 参数微型 Qwen3 的 offload 与
常驻图 HVP 数值对照通过（atol=1e-6、rtol=1e-5）。尚无完整 8B HVP 或残差收敛
完成记录；整个集成 probe 不记为通过。
probe 完成全部 8 次真实 update、K/Q/B 及清理验收后，正式作业才可启动。正式请求
四卡，尚未实际分配节点；结束时间与退出状态未知。

probe 确定实例：`20261005-181832_job188641_g4_tp2_seed1_hvp_integration_probe`，
位于 `n8/raw/integration_probe/`，其下 `fisher_kqb_probe_n8_u0008/` 为实际输出。
两份日志分别为部署下 `logs/slurm/fisher-kqb-integration-g4-188641.out` 和
`logs/slurm/fisher-kqb-n8-g4-188660.out`。正式设置为 12 轮、96 updates，不沿用
probe 的 mini-batch=8。提交时账户空间约 38.72 GB/268.44 GB；没有复制基座或把
测量权重写入持久目录，`SAVE_FREQ=-1`。

### 188641 主存失败与后续取消

2026-10-05 19:01 核验：188641 在 18:56:04 结束，elapsed 37:33，
`OUT_OF_MEMORY 0:125`。Slurm batch MaxRSS 为 514451976 KiB，接近实际分配的
502400 MiB（约 490.6 GiB）主存；测量子进程被 SIGKILL，日志记录 batch oom_kill。
不是旧 probe 的 CUDA allocation 错误。真实 8B 完成了至少第一个前缀的 HVP
（1/64，79.96 秒），但没有完整 64-prefix FVP、幂迭代 λ、age_00 报告或结束验收。
仅凭旧日志不能断言是单条长前缀峰值还是跨前缀累积。

19:04:19，按用户授权取消尚未启动的 N=8 正式任务 188660；Slurm 确认为
`CANCELLED by 2506`。N=4 的取消信息见其 README。不重新排队正式任务。

针对性修复：在 eval 下为 decoder forward 显式使用 PyTorch 非重入 checkpoint，
反传重算前向中间量；仍保留精确双反向、全参数、全词表、64 prompts、完整长度上限
与 FP32/FP64，CPU offload 仅保存仍需保留的导数图。加入逐前缀、逐求导阶段的进程
RSS 和 saved tensors 存活/峰值字节诊断。微型 CPU/GPU 对照须检查每个参数的 HVP
一致性和保存图减少；真实 8B 主存与八个 age 验收仍以新 probe 为准。

2026-10-05 19:21，修复的三个 Python 文件编译与 diff 检查通过，提交入口 shell
语法通过。候选代码及微型测试已同步到无活动作业的独立部署；CPU 数值回归尚未
取得结果，SSH 网关连续两次报告 `server ssh.paracloud.com not responding`。
不能记作测试通过，也尚未重提 GPU probe。正式两组保持取消状态。

### 连接恢复、CPU 回归与重提：188840

2026-10-05 20:36:06，SSH 状态查询恢复正常，确认正式 188659/188660 仍为取消。
环境没有 pytest，使用 Python 内置 unittest 执行 `test_training_fisher.py` 七项和
`test_exact_kl_hvp.py` 三项；十项全部通过，测试执行 0.296 秒（不含环境导入）。
包括重算前后逐参数 HVP 对照（atol=1e-6、rtol=1e-5）、eval 保持、显式 Fisher、
真实位移 norm、未收敛拒绝，以及保存图释放检查。32-token 微型模型的 saved tensor
hook 字节计数峰值从 189024 降至 142208，约减少 24.8%，结束后计数为零；这是微型
保存图计数，不是进程 RSS 或真实 8B 内存节省率。Python 编译、shell 语法、diff 检查通过。

| 项目 | 记录 |
| --- | --- |
| probe job | 188840，`fisher-kqb-integration-g4` |
| 测量源码 | `c282f3c`，按层非重入重算与 CPU 导数图存储；部署后 CPU 测试通过 |
| 提交 / 开始 | 2026-10-05 20:41:08 / 20:41:08，Asia/Shanghai |
| 依赖 | 无；本次只提交 probe，不重提正式实验 |
| 状态核验 | 2026-10-05 20:42:37：`RUNNING`，尚未结束；不能将运行中的 ExitCode 0:0 视作通过 |
| 实际资源 | `d1n41a28g03`；4×A800-SXM4-80GB、32 CPU、502400 MiB 主存 |
| 日志 | 部署下 `logs/slurm/fisher-kqb-integration-g4-188840.out` |

沿用同一集成设置：N=8、mini-batch=8、8 次真实更新、64 prompts×最多 8 个位置、
完整长度上限、全参数/全词表、FP32/FP64、残差 1e-3/cap=200、SAVE_FREQ=-1。
没有新增模型资产或保存训练权重。重提时账户父目录用量约 38.73 GB、限额 268.44 GB。
真实四卡 startup 数值自检和完整 8B 长前缀/八个 age 的内存与收敛验收，以本作业日志
和 `hvp_validation.json` 为准；CPU 测试不代替集成通过。

确定输出实例：`20261005-204108_job188840_g4_tp2_seed1_hvp_integration_probe`，
位于 `n8/raw/integration_probe/`，实际输出为其下 `fisher_kqb_probe_n8_u0008/`。
第一轮 FVP 的 `hvp_memory` 逐前缀记录 forward/两次 backward/结束阶段及主存，
用来检查重算后峰值是否降低、每条二阶图是否释放，不保存大向量快照。

### 188840 长前缀主存失败与逐层精确 Fisher 修复

2026-10-05 22:44:31，连接恢复后重新核验：188840 于 21:08:55 结束，elapsed
27:47，`OUT_OF_MEMORY 0:125`；Slurm batch MaxRSS=514450844 KiB，触及约
490.6 GiB 主存配额。不是 SSH 超时、CUDA 显存错误或算法已收敛。

本作业四卡 2200 参数的 checkpoint/offload HVP 启动数值自检通过；真实 8B 完成
第一轮 FVP 的前 16/64 个前缀（709.84 秒），各 row_done 的 saved CPU 图计数均
为零。第 8 个 1724-token 前缀的第一反传保存图已有 166908097568 字节，随后能
释放。第 17 个前缀长 2801 tokens，已完成 forward，但尚未记录 first_backward_done
便被 SIGKILL。定位为长前缀的整网导数图峰值问题；日志不支持把它解释为未释放的
16 份完整图，也不排除 allocator 和训练父进程占用增加总主存。没有完整 FVP、λ、
age_00 报告或真实 optimizer update，集成仍未通过。

当前修复改变求导实现而不改变 Fisher 目标：独立非 FSDP 模型逐层参数/输入 JVP
得到 Jv，用全词表 softmax Fisher 乘法，再逐层重算并普通 VJP 得到 J^T F_z Jv。
仅保存 CPU 上 detached 层输入和单层普通反传图，不保留整网 double-backward 图。
这不是依赖 FSDP 参数 JVP 支持，也不是只对输入求导；embedding/head/norm 全部
参与，仍保持全参数、64 prompts、完整前缀、全词表、FP32/FP64 和原收敛门槛。
anchor 零梯度检查也走逐层 VJP；K 的直接计算、实际累计位移和幂迭代逻辑不变。

候选数值回归在集群现有 PyTorch 2.8 环境、隐藏 CUDA 的 CPU 小模型上执行：
九项训练集成 unittest 通过（0.682 秒），三项原 HVP unittest 通过（0.005 秒）。
包括逐层 Fv 对照原双反传每个参数、共享 embedding/head、非零 KL 梯度、加权
FVP/二次型、权重与 eval 不变。逐参数 atol=1e-6、rtol=1e-5。32-token 微型模型
保存图 hook 峰值为整网 189024、前向 checkpoint 142208、逐层 VJP 47840 字节，
结束后计数归零；不是 8B RSS 节省率，不代替真实四卡验证。

截至 2026-10-05，本节为工程修复与测试记录，尚未有新的完整 8B probe 通过。正式 188659/188660
保持取消，不重提正式任务。188840 原始日志已归档，未下载模型权重。

2026-10-05 23:08（Asia/Shanghai），候选修复提交为 `3f291ca`。三个核心代码/测试
文件已同步到独立部署，并在该部署完成上述十二项 CPU 数值检查。随后同步新版文档、
核查源码摘要/账户空间/旧任务状态时，网关再次连续返回
`ssh NMCC-N46H1:22: ssh: handshake failed: EOF`；放宽至连接 60 秒、保活 30 秒×6，
并等待后重查仍失败。这是握手阶段错误，不是 GPU probe 超时。新版文档远端同步
未成功，保留本地已提交记录。尚未执行新的提交命令、没有新 job ID；连接恢复后须
先重新核查部署和持久空间，再只提交一个 probe，不重排正式任务。

### 逐层 Fisher 四卡 probe：190665

2026-10-06 10:55:10（Asia/Shanghai），连接恢复，远端三个核心代码/测试文件的
SHA-256 与本地 `3f291ca` 完全一致，账户用量 38735335449 字节、限额
268435456000 字节（约 38.74/268.44 GB）。队列没有活动任务；188840 仍为主存
OOM，正式 188659/188660 仍是取消。随后同步此前未成功同步的文档和现有提交入口，
只执行一次 `SLURM_DEPENDENCY= ... submit_experiment.sh probe`，没有重排正式任务。

| 项目 | 记录 |
| --- | --- |
| job / 源码 | 190665，`fisher-kqb-integration-g4`；逐层 Fisher 修复 `3f291ca` |
| 提交 / 开始 | 2026-10-06 10:56:29 / 10:56:51，Asia/Shanghai |
| 核验状态 | 2026-10-06 21:13:21：`COMPLETED 0:0`；现有 K/Q/B 集成验收通过，收尾异常见下文 |
| 结束 / elapsed | 2026-10-06 21:09:04 / 10:12:13，Asia/Shanghai |
| 依赖 / 时限 | 无依赖；24 小时，调度时限不是完成时间预测 |
| 实际资源 | `d1n41a28g03`；4×A800-SXM4-80GB、32 CPU、502400 MiB 主存 |
| 日志 | 部署下 `logs/slurm/fisher-kqb-integration-g4-190665.out` |

启动日志确认 N=8、mini-batch=8、rollout 64 prompt groups、8 responses/prompt，
计划 8 次真实 optimizer update、一轮 rollout；MATH500 benchmark 关闭，`save_freq=-1`。
沿用 64 prompts×最多 8 位置、完整 1024/3072 长度上限、全参数/全词表、FP32/FP64、
残差 1e-3/cap=200 和测量 6 小时限额。本次完整测量 K/Q/B，未提前取消 Q；λ 和
Q 共用逐层 FVP，只有 λ 通过而 Q 单独失败时，才考虑用户允许的 KL/λ/norm/B 降级。
不新增持久化基座副本、不保存训练后 HF 权重或 Adam；anchor/current/cache 仍仅放
作业私有 `/tmp/ds-190665/tmp/fisher-hvp/`，按原异常/轮末路径清理。

确定输出实例：`20261006-105651_job190665_g4_tp2_seed1_hvp_integration_probe`，
位于部署下 `n8/raw/integration_probe/`，实际输出为其下
`fisher_kqb_probe_n8_u0008/`。启动时验收要求为四卡数值对照、完整 64-prefix FVP、
anchor λ 残差收敛、八个 age 的 K/Q/B、首步 norm 对照与 scratch 清理；运行中的
`ExitCode=0:0` 不表示完成。正式两组保持取消，本次没有设置后续自动启动依赖。

#### 190665 终态与工程验收

2026-10-06 21:13:21 核验 Slurm 终态、唯一日志和标量产物，随后用现有
`verify_training_fisher.py` 的 `validate` 函数只读复核归档数据。远端
`hvp_validation.json` 为 `passed=true`，包含 8 updates、一轮 N=8、9 份 age 报告；
本地复核坐标、完整性、精度、B 公式和首步 norm 对照通过。本地没有计算节点 scratch，
不把本地函数的清理字段当作远端清理证据。

| 验收项 | 原始证据 |
| --- | --- |
| 四卡数值自检 | 2200 参数微型 Qwen3：逐层 FVP 对照整网 double backward 通过；atol=1e-6、rtol=1e-5 |
| 完整 FVP | 12 次幂迭代 + 8 次位移 FVP 均完成 64/64 前缀；512 位置，最长前缀 3140 tokens |
| anchor λ | 2803.3018877023183；12 次迭代，残差 5.478276571843372e-4 < 1e-3，`converged=true` |
| anchor 一阶检查 | 64 个前缀全部检查；最大行梯度 norm=1.1455512153857286e-14 |
| 更新与 age | `kl_updates.jsonl` 恰有 8 条真实 update，anchor=0、age_after=1..8；age_00..08 报告齐全，K/Q/B 有限且与训练记录一致 |
| 参数 / 精度 | 8190735360 全参数、全词表；参数及导数 FP32，KL/log-softmax/F_z FP64；点积 FP32 元素乘法、FP64 累加，TF32 关闭 |
| 位移 / 公式 | Δ 来自 current-anchor 的实际全参数差值；Q=0.5ΔᵀFΔ（代码以归一化方向求 FVP 再乘 norm²），B_hat=0.5λ_hat‖Δ‖² |
| 首步 norm 对照 | 分片 update norm=0.0790590785657796，完整 Δ norm=0.0790590784850943；相对差约 1.02e-9，满足 1e-5 门槛 |
| 快照与 checkpoint | 计算节点结束验收以 `--scratch /tmp/ds-190665/tmp/fisher-hvp` 检查，`transient_snapshots_removed=true`、`no_persistent_checkpoint=true`；远端输出无 global_step_*、模型权重、Adam 或 anchor/current 文件 |

本批原来失败的 2801-token 长前缀以及最长 3140-token 前缀均通过。逐层导数路线解决了
这批 q 上的整网二阶图 OOM，λ/Q 共用的完整 FVP 和 actor 导出/offload/恢复/八步测量链路
已跑通，不需要暂去 Q。它不证明所有 4096-token 前缀或正式 mini-batch=256 的整体资源
都已验证，也不是 8B 上对显式 Fisher 最大特征值的独立误差对照。

资源与耗时：Slurm batch MaxRSS=444338260 KiB（约 423.75 GiB），分配主存为
490.625 GiB；不能把测量子进程日志中最大 VmHWM 70.48 GiB 当作整个 batch 的主存。
单层 saved CPU 图 hook 峰值约 3.83 GiB，所有已记录 row_done 的图存活计数为零；
这也不等于进程 RSS 归零。测量子进程各卡 peak allocated 约
41.70/31.46/31.46/43.10 GiB，最大 peak reserved 约 43.70 GiB，不代表训练/vLLM 峰值。
完整 FVP 用时约 1580–1679 秒；age_00 测量报告用时 21396.33 秒（5:56:36），
已接近单次测量 6 小时限额。age=1..8 每次报告约 1715–1769 秒（28.6–29.5 分钟），
不含父进程导出/恢复开销。正式实验前仍需核对资源和时间预算，不能仅凭本次成功忽略限额。

收尾与评测偏差：八步测量后，日志记录完成训练进度、最终小验证和最终指标；随后出现
vLLM engine core died / Ray DataLoader worker Killed traceback，再输出验收通过。
`run_staleness.sh` 的主训练命令返回成功后才执行 gate，Slurm 为 `COMPLETED 0:0`，
日志没有 CUDA OOM 或 batch oom_kill 记录。异常没有阻止八步产物与清理验收，但仅凭
日志不能确定 kill 来源，不能宣称退出过程无异常。`benchmark_eval=0` 仍沿用了
FULL_KL 的 `test_freq=6`，trainer 的末轮分支触发 16 条训练集验证；不是 MATH500，
不解释其分数为正式能力结果。本次未修改代码或补交任务。

K 的数值口径也需分开：age=8 的 FP32 测量为 K=0.013921867624118777、
Q=0.04950024539114931、B_hat=104.73657285932848；训练侧 BF16 actor 的
`cumulative_kl` 为 0.001303203858338557。后续 K/Q/B 比较应使用同一测量模型的
`hvp_cumulative_kl`，不能混用两个 K。这一差异的原因尚未拆分，内部验收通过不等于
已确定模型精度差异的误差大小。B_hat 不是有限位移真实 KL 的严格上界；幂迭代残差
通过也不是 λ_max 上界证书。本 probe 不产生正式拟合或长期安全结论。

原件保留在上述唯一输出目录及 Slurm 日志路径。2026-10-06 仅将日志、
`kl_updates.jsonl`、`hvp_validation.json`、`run_manifest.json`、`eval_metrics.jsonl`
和九份 age 标量报告不覆盖地归档至 `raw/integration_probe_job190665/`（raw 不入 Git）；
没有复制前缀张量、anchor/current/cache、大向量、模型权重或 Adam。
`run_manifest.json` 的 endpoint_model 只是预设路径，实际没有该 checkpoint。
正式 188659/188660 保持取消，不自动重提；本次结束验收完成后停止定时监控。

首次集成提交前 CPU 自检：五项通过，包括 FP32 权重往返、全参数微型 Qwen3 二阶反传、
加权 HVP 对照显式 Fisher、因果前缀选择、实际参数差值 norm 和失败门槛。静态编译与
shell 语法检查通过；不将 CPU 自检等同于真实四卡训练集成已通过。
既有三项 HVP 回归测试通过；正式 N=8 配置预览通过，确认 12 轮、96 updates、
mini-batch=256、四卡可见、warmup=0 和 save_freq=-1。

## 产物

远端根：部署下 `experiments/20261005_fisher_kqb_n4_n8/n8/raw/integration_probe/` 和
`n8/raw/formal/`；每作业唯一实例，以日志的 `output=` 为准。
`kl_updates.jsonl` 含三项和 norm；`hvp_diagnostics/start_*/age_*.json` 保存逐 age 标量、
显存和 anchor 幂迭代 history；`hvp_validation.json` 是结束验收。probe 与正式不混合。

- [188038 原始失败日志](raw/integration_probe_job188038/fisher-kqb-integration-g4-188038.out)：已归档，raw 不入 Git。
- [188641 原始失败日志](raw/integration_probe_job188641/fisher-kqb-integration-g4-188641.out)：已归档，raw 不入 Git。
- [188840 原始失败日志](raw/integration_probe_job188840/fisher-kqb-integration-g4-188840.out)：已归档，raw 不入 Git。
- [190665 原始日志](raw/integration_probe_job190665/fisher-kqb-integration-g4-190665.out)、[逐 update 标量](raw/integration_probe_job190665/kl_updates.jsonl)、[结束验收](raw/integration_probe_job190665/hvp_validation.json)：已归档，raw 不入 Git。
