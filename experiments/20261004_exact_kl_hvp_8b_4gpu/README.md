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
| 资源 | 并行云 `gpu_a800`，4×A800-80GB、4 CPU、96 GiB 主存；一进程按层跨卡，非 FSDP、非 rollout TP |
| 精度 | FP32 参数与方向；FP64 输出 log-softmax/KL、标量归约累计；关闭 TF32 |
| 谱算法 | Power Iteration，最多 20 次 HVP，相对特征残差阈值 `1e-3` |
| seed | 20261005，各 GPU 的方向生成器用 seed 加 GPU 编号 |
| 训练/评测 | 0 optimizer updates；N、学习率、能力评测不适用；无 rollout |
| 自检 | anchor 一阶梯度接近零；HVP/Rayleigh/residual 有限、Rayleigh 为正；参数版本及 anchor 输出不变 |
| 保存 | 原始环境、输入 token、逐步标量、耗时、每卡显存、Slurm 日志；不保存模型或参数规模向量 |

短前缀只验证算子与内存路径，不代表正式 rollout 前缀上的 Fisher 数值。单 seed、有限
迭代的残差达标不是最大特征值的严格证明，也不是单卡参考误差校验。运行逻辑验收和谱
收敛状态分开记录。当前不建立正式分析记录或派生图表。

## 提交与状态

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
4.57.1，与 165 小模型环境一致；原始输出存 `/data/run01`，缓存走 Slurm 的
`/tmp/ds-$SLURM_JOB_ID/`。GPU 开始/结束时间及实际输出实例在提交后补充，不提前填写。

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
- 原始产物：作业完成后按确定实例归档至本实验 `raw/`，不进入 Git。
