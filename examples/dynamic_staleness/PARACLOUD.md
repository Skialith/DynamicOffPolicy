# 并行云单机多卡运行说明

这份说明适用于当前 dynamic staleness pilot 的单节点 1–8 卡调试。代码仍是同步 colocated 的
FSDP2 + vLLM 流程，不需要改成多进程 `torchrun`，也不需要第一次就配置跨节点 NCCL。

## 1. 上传内容与目录

登录节点只负责上传、安装和提交任务，训练通过 Slurm 在计算节点运行。建议在本地仓库的上级目录执行：

```bash
rsync -av --info=progress2 \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='outputs/' \
  --exclude='assets/' \
  verl-staleness/ 用户名@登录地址:~/run/src/verl-staleness/
```

模型和数据单独传输，断线后可继续：

```bash
rsync -av --partial --info=progress2 \
  verl-staleness/assets/ 用户名@登录地址:~/run/src/verl-staleness/assets/
```

不要上传本机 `.venv`：虚拟环境含绝对路径和平台相关二进制文件，应在集群重新创建。
上传前先确认本地存在 `assets/models/Qwen3-8B-Base/config.json`；当前脚本要求准确的 Base
checkpoint，不能用 Qwen3-8B instruct/non-Base 权重替代。若缺失，在可访问 Hugging Face 的机器上先运行
`examples/dynamic_staleness/prepare_assets.sh model`。

## 2. 存储配额先决条件

先在登录节点检查：

```bash
getfattr -n ceph.quota.max_bytes ~/run
getfattr -n ceph.dir.rbytes ~/run
du -sh ~/run/*
```

默认 50 GiB 只够代码、环境、数据和部分模型，不足以稳妥保存 Qwen3-8B full-parameter Adam
checkpoint。正式 anchor 及三个 branch 都需要从可恢复 checkpoint 分叉；开始正式训练前应向平台申请
更大的 `~/run` 配额。建议先申请至少 250 GiB，并根据一次实际 checkpoint 的 `du -sh` 再决定是否
扩到 400 GiB 以上。配额未扩大时，8B probe 默认不保存 checkpoint。

## 3. 创建集群环境

以下命令都在登录节点执行，环境放在共享的 `~/run`，计算节点可直接读取：

```bash
cd ~/run/src/verl-staleness
module load miniforge3/24.11
ENV_PREFIX=$PWD/.venv examples/dynamic_staleness/create_fresh_env.sh
```

这套环境使用 Python 3.12、PyTorch 2.8/cu128、vLLM 0.11 和 FlashAttention 2.8.3。A800 节点的
驱动可以运行 cu128 wheel；5090 必须先用下面的 smoke 实测 PyTorch、FlashAttention、vLLM 和
NCCL 是否都支持平台镜像。计算节点不联网，所以依赖必须在登录节点预先安装完。

## 4. 提交顺序

先用 `sinfo` 确认账号真实可用的队列名。`PARTITION` 必须按账号分别填写；不要仅凭能看到队列就
假定有权限。

### 小模型 smoke

先验证环境、两段训练、checkpoint 恢复和改变 N：

```bash
cd ~/run/src/verl-staleness
PARTITION=gpu_a800 N_GPUS=1 GPU_PROFILE=a800 \
  examples/dynamic_staleness/submit_slurm.sh smoke
```

5090 账号示例（队列名按 `sinfo` 实际结果替换）：

```bash
PARTITION=你的5090队列 N_GPUS=1 GPU_PROFILE=rtx5090 \
  examples/dynamic_staleness/submit_slurm.sh smoke
```

查看状态和日志：

```bash
squeue -u "$USER"
tail -f logs/slurm/staleness-smoke-作业号.out
```

### 8B 正式长度资源探针

smoke 通过后，优先申请 8 卡单机。探针只把 mini prompt batch 缩到 1，保留 8B、1024/3072
长度和第一次 Adam update：

```bash
PARTITION=gpu_a800 N_GPUS=8 GPU_PROFILE=a800 \
  examples/dynamic_staleness/submit_slurm.sh probe
```

```bash
PARTITION=你的5090队列 N_GPUS=8 GPU_PROFILE=rtx5090 \
  examples/dynamic_staleness/submit_slurm.sh probe
```

A800 profile 默认 rollout TP=2；5090 profile 默认 TP=1，避免把 vLLM rollout 建立在高速卡间互联
假设上。两者都只是起始值，最终以日志中的 OOM、NCCL 报错、GPU 利用率和单步时间为准。

### 正式 anchor

只有 probe 完成且 checkpoint 配额足够后再提交：

```bash
PARTITION=gpu_a800 N_GPUS=8 GPU_PROFILE=a800 \
  examples/dynamic_staleness/submit_slurm.sh anchor
```

如果 8 卡 GPU 显存充足，可先用一个受限 probe 验证关闭 FSDP2 CPU offload 的快路径，不能直接
用于完整实验：

```bash
PARTITION=gpu_a800 N_GPUS=8 GPU_PROFILE=a800 \
FSDP_OFFLOAD_POLICY=false REF_PARAM_OFFLOAD=false \
  examples/dynamic_staleness/submit_slurm.sh probe
```

branch 需要共享 anchor checkpoint 的绝对路径：

```bash
PARTITION=gpu_a800 N_GPUS=8 GPU_PROFILE=a800 \
ANCHOR_CKPT=$PWD/outputs/dynamic_staleness/anchor_n4_u0100/global_step_25 \
REUSE_N=8 TARGET_OPTIMIZER_STEP=196 \
  examples/dynamic_staleness/submit_slurm.sh branch
```

N=4/8/16 三个 branch 建议串行提交，避免同一账号同时排队和争用存储。完成调试后用 `scancel 作业号`
取消不再需要的任务；正式训练使用 `sbatch`，不依赖 SSH 会话持续在线。

两个账号不能拼成一个跨账号 16 卡 Slurm 作业。一个 seed 的 anchor 和 branches 最好保持相同的
GPU 型号、卡数和软件环境；A800 账号优先承担主实验，5090 账号可先做兼容性探针，稳定后再运行
完整的独立 seed。不要把不同硬件上的运行时间直接当作 staleness 方法差异。

## 5. 常见失败的判断顺序

1. 日志没有生成：作业通常仍在 `PD` 排队，先看 `squeue`/`parajobs` 的 REASON。
2. Python 包或模型缺失：回登录节点补环境和资产，不要让计算任务在线安装或下载。
3. CUDA/FlashAttention 架构错误：保留完整报错和 `nvidia-smi -L`、PyTorch 版本，优先在 5090
   一卡 smoke 中解决。
4. vLLM 初始化 OOM：先下调 `GPU_MEMORY_UTILIZATION=0.45`，再减少
   `ROLLOUT_MAX_NUM_SEQS`，不要先改正式实验 batch 定义。
5. 第一次 Adam update OOM：增加卡数或启用 `FSDP_OFFLOAD_POLICY=true`；同时确认 CPU RAM 配额。
6. NCCL/P2P 错误：保存 `nvidia-smi topo -m` 与完整 NCCL 日志，再针对 5090 节点互联配置处理。
