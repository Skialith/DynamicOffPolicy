# 2026-10-04：8B 四卡精确 KL-HVP 工程探针

## 实验任务与关系

用户确认使用正式基座 Qwen3-8B-Base，在并行云四张 A800 上验证全参数 frozen-anchor
KL 的二阶自动微分和 Power Iteration 是否可运行。沿用
[165 小模型验证](../20261004_kl_hvp_exact_vs_fsdp_fd/README.md)的数学路线，去掉同时保存
多个参考方向、多个 FVP 和大向量快照的诊断开销；先完成一次 HVP，再继续谱迭代。

这是独立测量模型的规模迁移探针，不是 legacy FSDP actor 的 double backward 支持测试，
也不验证与 vLLM 共驻的训练集成。模型参数固定，Power Iteration 只改变方向向量。
不使用参数 JVP、有限差分、TOP-K、量化或参数冻结。

## 配置

| 项目 | 设置 |
| --- | --- |
| 模型 | 已有离线 Qwen3-8B-Base，全部参数参与；eager attention、eval、无 cache、无 checkpointing |
| 前缀分布 | 四个固定数学 prompt，各取末尾两个有效位置；prompt/位置等权，全词表 KL |
| 资源 | 并行云 `gpu_a800`，4×A800-SXM4-80GB；作业 187752 实际分配 32 CPU、502400 MiB 主存；一进程按层跨卡，非 FSDP、非 rollout TP |
| 精度 | FP32 参数与方向；FP64 输出 log-softmax/KL、标量归约累计；关闭 TF32 |
| 谱算法 | Power Iteration，最多 20 次 HVP，相对特征残差阈值 `1e-3` |
| seed | 20261005，各 GPU 的方向生成器用 seed 加 GPU 编号 |
| 训练/评测 | 0 optimizer updates；N、学习率、能力评测不适用；无 rollout |
| 自检 | anchor 一阶梯度接近零；HVP/Rayleigh/residual 有限、Rayleigh 为正；参数版本及 anchor 输出不变 |
| 保存 | 原始环境、输入 token、逐步标量、耗时、每卡显存、Slurm 日志；不保存模型或参数规模向量 |

上表描述首个工程探针；后续残差收敛补验的预算与 seed 单独记录，不回改首个作业设置。
短前缀只验证算子与内存路径，不代表正式 rollout 前缀上的 Fisher 数值。单 seed、有限
迭代的残差达标不是最大特征值的严格证明，也不是单卡参考误差校验。运行逻辑验收和谱
收敛状态分开记录。当前不建立正式分析记录或派生图表。

## 提交与状态

### 残差收敛补验：2026-10-05

用户要求改为残差达标后结束。保持同一 checkpoint、前缀/聚合、精度和 HVP 计算，使用
`--require-convergence` 严格验收：残差 `<=1e-3` 才计为成功；200 次迭代及每个作业
20 分钟只是异常兜底，达到上限仍未收敛会失败，不返回“成功”。未收敛的标量诊断仍
保留。原作业 187752 及其 raw 不改写。

使用 seed `20261005` 和 `20261015` 两个随机初始方向，GPU 生成器 seed 范围不重叠；
两次串行四卡运行，不同时占八卡。第二个方向只核验最大特征值估计对初值是否敏感，
两个估计一致不是已证明整个谱的最大值。残差本身检验某个特征方程，而非与单卡真值
比较。三项 CPU 自检通过，包括严格模式触顶必须失败；本地编译与 shell 语法检查通过。
部署探针 SHA256 为 `adbfa03edac0f3c539772ccff789501666c59db178caf8495478468d875f41d3`。

| seed | job ID | 提交时间（Asia/Shanghai） | 依赖 | 开始 → 结束（Asia/Shanghai） / 状态 |
| --- | --- | --- | --- | --- |
| 20261005 | 187793 | 2026-10-05 13:05:21 | 无 | 13:05:51 → 13:07:06；`COMPLETED 0:0`，第 30 次迭代残差达标 |
| 20261015 | 187794 | 2026-10-05 13:05:21 | `afterok:187793` | 13:07:21 → 13:08:36；`COMPLETED 0:0`，第 30 次迭代残差达标 |

提交仍用统一 `exact_hvp_probe` 入口，额外参数 `--require-convergence`；环境设
`HVP_POWER_STEPS=200`、`HVP_POWER_TOLERANCE=0.001`、`TIME_LIMIT=00:20:00`。
两个 `RUN_CONFIG_TAG` 分别为 `g4_mp4_seed20261005_convergence` 和
`g4_mp4_seed20261015_convergence`，日志分别是独立部署目录下的
`logs/slurm/exact-hvp-conv-s20261005-187793.out`、
`logs/slurm/exact-hvp-conv-s20261015-187794.out`。

2026-10-05 13:08:47 核验：两次均完成，Slurm elapsed 各 1 分 15 秒，严格收敛模式
达标后提前结束，没有触及 200 次或 20 分钟兜底。实际各分配四张 A800、32 CPU、
502400 MiB 主存，运行时间不重叠；节点均为 `d1n41a28g03`。数值保存在各自原始
`exact.json`，不创建正式分析记录或跨模型精度结论。

确定的输出实例分别为 `20261005-130551_job187793_g4_mp4_seed20261005_convergence`
和 `20261005-130721_job187794_g4_mp4_seed20261015_convergence`，均位于独立部署目录
的 `experiments/20261004_exact_kl_hvp_8b_4gpu/raw/`。仍只保存三个小型 JSON 和一份日志，
不保存模型、梯度、迭代大向量；原作业 187752 的所有原始文件保持不变。
两组原始文件已按确定实例同步到本地，分别共 73,260 / 73,275 bytes，不进入 Git。
本地核验两组均 `converged=true`、`logic_passed=true`、参数未改变；环境确实使用严格
模式、200 次上限及 `1e-3` 阈值，输入 token 和模型路径相同。seed 20261005 的前
20 次 Rayleigh/residual 与原作业逐值一致。两初值估计的数值接近只作初值敏感性的
工程核验，不解释为解析真值误差或 Fisher 全谱最大值的严格证书。

### 已完成工程探针：2026-10-05 重试

2026-10-05 12:51:31（Asia/Shanghai）核验：写入配额阻塞未重现，五个小脚本/自检文件
通过 rsync 完成部署，传输约 27 KB；远端 SHA256 与本地一致，shell 语法检查通过。
没有复制、删除或覆盖基座权重，也没有修改原训练仓库。本次明确不调用 `torch.save`、
`save_pretrained` 或 checkpoint 保存；只保存三个小型 JSON 和 Slurm 文本日志。
离线加载使用 `local_files_only=True`，不下载第二份模型。已有基座资产保留原样。

| 项目 | 作业 187752 |
| --- | --- |
| 提交入口 | 独立部署目录中的 `examples/dynamic_staleness/submit_slurm.sh exact_hvp_probe` |
| 提交 / 开始 / 结束 | 2026-10-05 12:48:44 / 12:48:52 / 12:50:05，Asia/Shanghai；Slurm elapsed 为 1 分 13 秒 |
| 状态 | `COMPLETED`，退出码 `0:0`；完成全部 20 次精确 HVP；运行逻辑验收通过，谱收敛验收未达标 |
| 节点 / GPU | `d1n41a28g03`，4×A800-SXM4-80GB |
| CPU / 主存 | Slurm 实际分配 32 CPU、502400 MiB；原计划的 SBATCH 环境变量没有体现为对应请求值，以实际分配为准 |
| 时限 / 依赖 | 20 分钟 / 无；实际结束时间来自 sacct，不采用运行中显示的预计时限终点 |
| 环境 | Python 3.12.14、PyTorch 2.8.0+cu128、Transformers 4.57.1、CUDA 12.8 |
| 代码 | `67cf3ce` 中的探针实现，本次未修改计算逻辑 |
| 输出实例 | `20261005-124852_job187752_g4_mp4_seed20261005_exact_probe` |

远端部署：`/data/run01/scyb980/cyt/src/exact_kl_hvp_probe_20261004/`。
确定的远端输出为上述部署目录下的
`experiments/20261004_exact_kl_hvp_8b_4gpu/raw/20261005-124852_job187752_g4_mp4_seed20261005_exact_probe/`。
日志：`logs/slurm/exact-kl-hvp-8b-g4-187752.out`；计算缓存：`/tmp/ds-187752/`。
三个 JSON 和 Slurm 日志已按同名实例同步至本地 `raw/`，总计 50,801 bytes。
远端测量输出目录仅有 `environment.json`、`batch.json`、`exact.json`；另存上述 Slurm
日志。没有权重、checkpoint、梯度或参数规模方向向量文件。

原始自检中 `logic_passed=true`、`parameters_unchanged=true`，anchor 输出未改变；完成
20 次 HVP 及迭代，不存在 OOM 或 double-backward 算子报错。`converged=false`，最后
相对特征残差仍高于预设 `1e-3`，不将退出码 0 等同于最大特征值已收敛。数值明细留在
原始 `exact.json`；本次没有增加迭代预算或补提交作业，也不创建正式分析记录。

### 历史准备阻塞：2026-10-04

2026-10-04 23:46（Asia/Shanghai）核验：尚未提交，没有 job ID；没有执行 8B GPU
计算。部署脚本时写入明确返回 `Disk quota exceeded (122)`，因此停在提交前。公共文件
系统的剩余空间不能代表本账户剩余配额；登录节点未安装 `quota` 命令，当前没有核验出
配额上限。只读查看原仓库 `outputs` 约 155 GB，没有删除任何模型、checkpoint 或原始
日志，也没有改写原训练脚本。需用户指定允许清理的目标或扩容后才能继续提交。

远端仓库是无 `.git` 的部署快照；本次建立独立部署目录
`/data/run01/scyb980/cyt/src/exact_kl_hvp_probe_20261004/`，传输受配额阻止，不能当作完成
部署。网关 SFTP 传输未完成，后续改用集群文档的 rsync；配额错误来自 rsync receiver。
本地 Python 编译、三个 shell 脚本语法和 `git diff --check` 已通过。
在已有并行云 Python 环境通过输入流执行两项微型 CPU 单元测试，均通过；没有启动模型
训练或 GPU 运算，也没有为自检写入新文件。登录节点自检仅核对数学实现，不代表 8B
算子路径已经验证。

统一入口：[`submit_slurm.sh`](../../examples/dynamic_staleness/submit_slurm.sh)
的 `exact_hvp_probe` target，执行
[`run_exact_kl_hvp_probe.sh`](../../examples/dynamic_staleness/run_exact_kl_hvp_probe.sh)。
核心探针为 [`verify_exact_kl_hvp.py`](../../examples/dynamic_staleness/verify_exact_kl_hvp.py)。

确认的已有模型资产：
`/data/run01/scyb980/cyt/src/verl-staleness/assets/models/Qwen3-8B-Base`。
Python 使用原部署环境 `.venv/bin/python`，已确认 PyTorch 2.8.0+cu128、Transformers
4.57.1，与 165 小模型的关键软件版本一致；原始输出存 `/data/run01`，缓存走 Slurm 的
`/tmp/ds-$SLURM_JOB_ID/`。

配额解除、上述四个部署文件同步完成后的提交设置：

```bash
cd /data/run01/scyb980/cyt/src/exact_kl_hvp_probe_20261004
export PARTITION=gpu_a800 N_GPUS=4 GPU_PROFILE=a800
export PYTHON_BIN=/data/run01/scyb980/cyt/src/verl-staleness/.venv/bin/python
export MODEL_PATH=/data/run01/scyb980/cyt/src/verl-staleness/assets/models/Qwen3-8B-Base
export SEED=20261005 RUNTIME_PROFILE=exact_layer_parallel
export RUN_CONFIG_TAG=g4_mp4_seed20261005_exact_probe
export OUTPUT_ROOT="$PWD/experiments/20261004_exact_kl_hvp_8b_4gpu/raw"
SBATCH_CPUS_PER_TASK=4 SBATCH_MEM=96G TIME_LIMIT=00:20:00 JOB_NAME=exact-kl-hvp-8b-g4 \
    bash examples/dynamic_staleness/submit_slurm.sh exact_hvp_probe
```

统一包装器中的 rollout TP 默认值仅是未使用的训练配置，本探针实际是一进程四卡按层
放置，不启动 rollout、Ray、NCCL/FSDP process group。CPU 自检入口为
[`test_exact_kl_hvp.py`](../../tests/workers/actor/test_exact_kl_hvp.py)，核对分块向量标量操作
以及一个非线性小模型的 KL-HVP 与显式 `J^T (diag(p)-pp^T) J v`。

## 产物入口

- [实验索引](../README.md)
- [作业 187752 原始归档](raw/20261005-124852_job187752_g4_mp4_seed20261005_exact_probe/)：已同步，不进入 Git。
- [残差收敛补验 187793](raw/20261005-130551_job187793_g4_mp4_seed20261005_convergence/)：已同步，不进入 Git。
- [第二初值补验 187794](raw/20261005-130721_job187794_g4_mp4_seed20261015_convergence/)：已同步，不进入 Git。
