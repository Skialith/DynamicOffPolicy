# 并行云 Slurm 集群环境一次性部署指南

本文用于在并行云 N46H2 Slurm 集群上，从一个能够联网的登录节点开始，部署并验证 dynamic staleness 实验环境。命令以 A800 队列和以下目录为例：

```text
代码与环境：~/run/cyt/src/verl-staleness
持久缓存：  ~/run/cyt/src/verl-staleness/.cache/paracloud/hf-datasets
临时缓存：  ${SLURM_TMPDIR:-/tmp}/ds-$SLURM_JOB_ID
模型：      ~/run/cyt/src/verl-staleness/assets/models
数据：      ~/run/cyt/src/verl-staleness/assets/datasets
输出：      ~/run/cyt/src/verl-staleness/outputs
日志：      ~/run/cyt/src/verl-staleness/logs/slurm
```

所有大文件都放在 `~/run`。`$HOME` 只保留 shell、SSH 等小型配置。

## 1. 已验证范围

下面这套组合已经在单张 NVIDIA A800-SXM4-80GB 上完成端到端 smoke：

- Slurm 作业正常分配 GPU，最终状态为 `COMPLETED 0:0`；
- Qwen3-0.6B-Base 完成 N=4 anchor；
- 保存模型、optimizer、RNG、scheduler 和 staleness 状态；
- 从 anchor checkpoint 恢复后改为 N=8；
- optimizer step 从 0 连续运行到 12；
- vLLM rollout、FSDP2 更新、FlashAttention、Ray、TensorBoard 和 staleness 指标均可运行。

这说明环境、代码、单卡 A800 执行链路和 checkpoint 恢复已经配置完成。正式 Qwen3-8B 实验仍需依次完成 8B 模型下载、8B resource probe 和目标卡数的显存/CPU 内存验证。5090 需要单独运行 smoke，不能只根据 A800 结果推定兼容。

## 2. 固定软件版本

| 组件 | 版本 |
|---|---|
| OS / 架构 | Linux x86_64 |
| Python | 3.12 |
| CUDA module | 12.8 |
| PyTorch | 2.8.0+cu128 |
| torchvision | 0.23.0+cu128 |
| vLLM | 0.11.0 |
| FlashAttention | 2.8.3 |
| Ray | 2.58.0 |
| verl | 本仓库 editable install，版本 0.7.1 |
| Transformers | 4.57.1 |
| tokenizers | 0.22.2 |
| huggingface-hub | 0.36.2 |
| tensordict | 0.10.0 |
| NumPy | 1.26.4 |
| PyArrow | 25.0.1 |
| hydra-core | 1.3.5 |
| codetiming | 1.4.0 |
| math-verify | 0.9.0 |

一键脚本会固定兼容性关键版本，并安装与 Python、Torch ABI 匹配的 FlashAttention release wheel。

## 3. 集群前提

### 3.1 节点职责

- 登录节点：上传代码、下载依赖和模型、创建环境、查看日志、提交作业。
- GPU 计算节点：执行训练；运行时不依赖公网。
- 正式计算统一通过 `sbatch` 提交，不能在登录节点直接训练。

本指南假设当前登录节点能够访问：

```text
conda.anaconda.org
pypi.org
download.pytorch.org
github.com
huggingface.co
```

### 3.2 队列与配额

```bash
sinfo

getfattr -n ceph.quota.max_bytes "$HOME"
getfattr -n ceph.dir.rbytes "$HOME"
getfattr -n ceph.quota.max_bytes "$HOME/run"
getfattr -n ceph.dir.rbytes "$HOME/run"
```

平台的 `$HOME` 默认只有 1 GiB，不适合安装环境。`~/run` 默认可能只有 50 GiB；完整 8B 模型、环境和多份 full-parameter Adam checkpoint 建议至少申请 250 GiB，正式多分支实验应根据一次 checkpoint 的实际大小继续扩容。

### 3.3 可用 module

```bash
module avail miniforge
module avail cuda
```

本文验证使用：

```text
miniforge3/25.11.0-1
cuda/12.8
```

## 4. 上传并展开代码

先在集群创建目录：

```bash
mkdir -p "$HOME/run/cyt/src" "$HOME/run/cyt/logs"
```

从本地上传源码压缩包：

```bash
rsync -avP -e 'ssh -p 2222' \
  dynamic-staleness-paracloud-src.tar.gz \
  'scyb980@NMCC-N46H1@ssh.paracloud.com:~/run/cyt/'
```

登录集群后校验并展开：

```bash
cd "$HOME/run/cyt"
sha256sum dynamic-staleness-paracloud-src.tar.gz
tar -xzf dynamic-staleness-paracloud-src.tar.gz -C "$HOME/run/cyt/src"
cd "$HOME/run/cyt/src/verl-staleness"
```

如果使用 `rsync` 直接同步目录，不要传本机 `.venv`、`outputs`、`.git` 和缓存目录。Python 环境包含平台二进制与绝对路径，必须在集群重新创建。

## 5. 一次创建 Python/CUDA 环境

在能够联网的登录节点执行：

```bash
cd "$HOME/run/cyt/src/verl-staleness"

conda deactivate 2>/dev/null || true
module purge
module load miniforge3/25.11.0-1
module load cuda/12.8

module list
which conda
python --version
nvcc --version

mkdir -p "$HOME/run/cyt/logs"
set -o pipefail

ENV_PREFIX="$PWD/.venv" \
CONDA_BIN=conda \
examples/dynamic_staleness/create_fresh_env.sh \
  2>&1 | tee "$HOME/run/cyt/logs/create_env.log"
```

`create_fresh_env.sh` 完成以下操作：

1. 将 conda、pip、XDG 和 Hugging Face 安装缓存放到仓库的 `.cache/setup`；
2. 在 `.venv` 创建 Python 3.12 conda prefix；
3. 从 PyTorch cu128 index 安装 Torch 2.8.0 和 torchvision 0.23.0；
4. 安装 vLLM 0.11.0；
5. editable 安装当前 verl 源码及 `vllm,math` extras；
6. 固定已验证的 Transformers、Ray、TensorDict 等关键版本；
7. 根据 Python tag、Torch 版本和 C++11 ABI 自动选择 FlashAttention 2.8.3 wheel；
8. 运行 `pip check` 和仓库环境预检。

脚本可在网络中断后重新执行，会复用已经创建的 `.venv` 并补齐剩余步骤。

### GitHub wheel 无法直接下载时

在其他可联网的 Linux x86_64 机器下载下面的文件并上传到 `~/run/cyt/wheels`：

```text
flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

校验本次已验证 wheel：

```text
SHA256 f25da18657a87fc83dc1bfb8b7751b82246e9db355510226b674fd437c34b5fb
```

然后重新运行安装脚本：

```bash
cd "$HOME/run/cyt/src/verl-staleness"
set -o pipefail

FLASH_ATTN_WHEEL="$HOME/run/cyt/wheels/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" \
ENV_PREFIX="$PWD/.venv" \
CONDA_BIN=conda \
examples/dynamic_staleness/create_fresh_env.sh \
  2>&1 | tee "$HOME/run/cyt/logs/create_env.log"
```

## 6. 下载模型和数据

完整实验需要三个资产：

```text
assets/models/Qwen3-0.6B-Base
assets/models/Qwen3-8B-Base
assets/datasets/DAPO-Math-17k/data/dapo-math-17k.parquet
assets/datasets/MATH-500/data/math500.parquet
```

下载 smoke 模型与数据集：

```bash
cd "$HOME/run/cyt/src/verl-staleness"

HF_BIN="$PWD/.venv/bin/hf" \
  examples/dynamic_staleness/prepare_assets.sh smoke-model

HF_BIN="$PWD/.venv/bin/hf" \
  examples/dynamic_staleness/prepare_assets.sh dataset

HF_BIN="$PWD/.venv/bin/hf" \
  examples/dynamic_staleness/prepare_assets.sh math500
```

下载正式 8B Base 模型：

```bash
HF_BIN="$PWD/.venv/bin/hf" \
  examples/dynamic_staleness/prepare_assets.sh model
```

也可以一次下载全部资产：

```bash
HF_BIN="$PWD/.venv/bin/hf" \
  examples/dynamic_staleness/prepare_assets.sh all
```

检查资产：

```bash
test -f assets/models/Qwen3-0.6B-Base/config.json
test -f assets/models/Qwen3-8B-Base/config.json
test -f assets/datasets/DAPO-Math-17k/data/dapo-math-17k.parquet
test -f assets/datasets/MATH-500/data/math500.parquet
```

正式实验必须使用 `Qwen/Qwen3-8B-Base`，不能用 instruct checkpoint 替代。

## 7. 登录节点最终检查

```bash
cd "$HOME/run/cyt/src/verl-staleness"

timeout 300 .venv/bin/python -m pip check

PYTHONPATH="$PWD" timeout 300 .venv/bin/python \
  examples/dynamic_staleness/check_env.py \
  --repo-root "$PWD"
```

预期关键输出：

```text
torch          2.8.0+cu128
vllm           0.11.0
ray            2.58.0
tensordict     0.10.0
transformers   4.57.1
tokenizers     0.22.2
huggingface-hub 0.36.2
flash-attn     2.8.3
preflight passed
```

登录节点没有 GPU 时，`CUDA devices 0` 是正常结果。GPU 检查必须由 Slurm 计算节点完成。

保存环境快照：

```bash
mkdir -p logs/environment
.venv/bin/python -m pip freeze > logs/environment/pip-freeze.txt
module list 2> logs/environment/modules.txt
```

## 8. Slurm 运行环境

`slurm_job.sh` 会为每个作业自动设置以下内容：

- Hugging Face datasets 的 Parquet→Arrow 转换缓存持久写入仓库 `.cache/paracloud/hf-datasets`；
- Ray、vLLM、Triton、TorchInductor、Torch extensions、CUDA、Numba、Matplotlib 和 XDG 缓存写入计算节点本地的 `${SLURM_TMPDIR:-/tmp}/ds-$SLURM_JOB_ID`；
- 作业结束后节点本地缓存会释放，不作为实验结果；模型、checkpoint、JSONL、TensorBoard event 和 Slurm 日志仍保存在 `~/run`；
- `VLLM_NO_USAGE_STATS=1`；
- `TMPDIR` 位于上述作业本地目录，避免 Ray Unix socket 路径超过 107 字节；
- `PYTHONNOUSERSITE=1`，避免加载 `$HOME` 下的用户包；
- 使用仓库 `.venv/bin/python` 和当前 clone 中的 verl 源码。

每次提交还会生成唯一运行实例名，例如：

```text
20260829-153012_job123456_g8_tp2_seed1
```

结果写入 `outputs/<实验类型>/<运行实例>/...`。日期时间区分同一天的重复运行，Slurm job id 避免同秒启动冲突，GPU/TP/seed 用于识别配置。TensorBoard run 名使用同一实例前缀，例如 `20260829-153012_job123456_g8_tp2_seed1_anchor_n4_u0100`，因此重复实验不会覆盖。查看全部实验：

```bash
tensorboard --logdir tensorboard_log --port 6006
```

需要恢复同一实例时，使用日志中的 `run_instance=...` 和明确的 checkpoint 路径；不要用“最新目录”猜测。若确实要把后续作业写回同一实例，可显式设置 `RUN_INSTANCE_TAG`，也可用 `OUTPUT_INSTANCE_ROOT` 指定完整实例目录。

因此重新登录后不需要重复手工导出缓存变量。提交前只需加载 module 并进入仓库：

```bash
module purge
module load miniforge3/25.11.0-1
module load cuda/12.8
cd "$HOME/run/cyt/src/verl-staleness"
```

## 9. 提交单卡 A800 smoke

```bash
cd "$HOME/run/cyt/src/verl-staleness"

PARTITION=gpu_a800 \
N_GPUS=1 \
GPU_PROFILE=a800 \
TIME_LIMIT=01:00:00 \
JOB_NAME=staleness-smoke \
examples/dynamic_staleness/submit_slurm.sh smoke
```

若管理员尚未处理某个异常节点，可临时排除：

```bash
SBATCH_EXCLUDE=节点名 \
PARTITION=gpu_a800 \
N_GPUS=1 \
GPU_PROFILE=a800 \
TIME_LIMIT=01:00:00 \
examples/dynamic_staleness/submit_slurm.sh smoke
```

查看排队与日志：

```bash
squeue -u "$USER" \
  -o "%.18i %.12P %.30j %.10T %.10M %.20R"

LOG=$(ls -t logs/slurm/staleness-smoke-*.out | head -n 1)
tail -F "$LOG"
```

作业结束后检查：

```bash
sacct -j 作业号 \
  --format=JobID,JobName,State,ExitCode,Elapsed,NodeList
```

成功标准：

```text
State=COMPLETED
ExitCode=0:0
anchor Training Progress=100%
branch Training Progress=100%
branch 成功加载 anchor 的 optimizer、RNG 和 scheduler
outputs/smoke_staleness/<运行实例>/smoke_anchor_n4_u0004/global_step_1 存在
outputs/smoke_staleness/<运行实例>/smoke_branch_n8_u0004_0012/global_step_2 存在
```

Ray 和 DataLoader 在进程退出时可能记录 worker 被终止的信息。判断是否成功以训练进度、checkpoint 和 Slurm 的 `COMPLETED 0:0` 为准。

## 10. 正式运行顺序

### 10.1 查看资源

```bash
sinfo
```

单节点可以申请 1–8 张卡。卡数越多，通常排队时间越长。

### 10.2 8B 资源探针

smoke 通过并且 8B Base 已下载后：

```bash
PARTITION=gpu_a800 \
N_GPUS=8 \
GPU_PROFILE=a800 \
TIME_LIMIT=02:00:00 \
examples/dynamic_staleness/submit_slurm.sh probe
```

探针保留 8B 和正式序列长度，只缩小 prompt batch，用于验证第一次 Adam update、CPU 内存、GPU 显存和单步时间。探针不是正式 N 比较结果。

### 10.3 Anchor

probe 完成且 checkpoint 配额足够后：

```bash
PARTITION=gpu_a800 \
N_GPUS=8 \
GPU_PROFILE=a800 \
TIME_LIMIT=24:00:00 \
examples/dynamic_staleness/submit_slurm.sh anchor
```

### 10.4 Branch

```bash
PARTITION=gpu_a800 \
N_GPUS=8 \
GPU_PROFILE=a800 \
TIME_LIMIT=24:00:00 \
RUN_INSTANCE_TAG=20260829-153012_job123456_g8_tp2_seed1 \
ANCHOR_CKPT="$PWD/outputs/dynamic_staleness/20260829-153012_job123456_g8_tp2_seed1/anchor_n4_u0100/global_step_25" \
REUSE_N=8 \
TARGET_OPTIMIZER_STEP=196 \
examples/dynamic_staleness/submit_slurm.sh branch
```

把示例实例名替换为 anchor 日志中的实际 `run_instance`。N=4、8、16 branch 共享同一个 anchor checkpoint，并显式复用这个 `RUN_INSTANCE_TAG`，从而写入同一实验组下的不同 branch 子目录。一个 seed 应保持相同 GPU 型号、卡数和软件环境。

### 10.5 Benchmark eval

模型上传完成后，正式实验确实应补能力评测曲线，但不要直接设置 `trainer.test_freq=10`。SIS Figure 3 的横轴是 gradient step，也就是本项目的真实 optimizer update；verl 的 `test_freq` 却按“rollout 一次再更新 N 次”的 outer/global step 计数。因此 `test_freq=10` 对 N=4、8、16 分别代表 40、80、160 次 optimizer update，无法公平比较。

正式 anchor/branch 已启用在线 MATH500 Avg@1：固定 500 题，greedy decoding（`temperature=0`、`n=1`），每个完整 rollout→N 次 update 周期结束后评测。评测不插入同一批 stale rollout 的连续 N 次 optimizer update 中间，因此不会改变 `policy_age=0…N-1` 的受控流程。

- resource probe 不做 benchmark eval；
- anchor 先评 update 0，随后评 4、8、…、100；
- branch 不重复 update 100，N=4/8/16 分别每 4/8/16 个 optimizer update 评测；
- `eval_metrics.jsonl` 按绝对 optimizer step 保存 accuracy，`eval_generations/optimizer_step_XXXX.jsonl` 保存500道题的答案与判分；
- 与 SIS Figure 3 对齐的正式结果后续再做 AIME24/AIME25 Avg@32；
- SIS 论文没有声明 Figure 3 是“每 10 步 eval”，所以不要把图上曲线密度当作已公开的评测频率。

默认只在 anchor 结束时保存 update 100 的完整模型/optimizer/scheduler/RNG checkpoint。branch 默认不保存完整 checkpoint，只保留 staleness/eval JSONL 和 TensorBoard event；如果某个 branch 明确要从 196 延长到 292，提交它时设置 `SAVE_FREQ=1000000`，在 branch 末尾额外保存一份可恢复 checkpoint。

## 11. 5090 账号

相同的共享环境可以作为起点，但必须先做独立 smoke：

```bash
PARTITION=实际5090队列名 \
N_GPUS=1 \
GPU_PROFILE=rtx5090 \
TIME_LIMIT=01:00:00 \
examples/dynamic_staleness/submit_slurm.sh smoke
```

`rtx5090` profile 默认使用 rollout TP=1 和较低的 vLLM 显存比例。只有 smoke、FlashAttention、vLLM、NCCL 和多卡通信都通过后，才能把 5090 用于完整的独立 seed。两个账号不能拼成一个跨账号的 16 卡 Slurm 作业。

## 12. 常用诊断

### 作业没有日志

```bash
squeue -u "$USER"
parajobs
```

`PD` 表示仍在排队，查看 `NODELIST(REASON)`。

### GPU 节点异常

先做最小 CUDA 分配测试：

```bash
srun \
  --nodes=1 \
  --ntasks=1 \
  --gpus=1 \
  -p gpu_a800 \
  --time=00:05:00 \
  .venv/bin/python -c \
  'import socket,torch; print(socket.gethostname()); print(torch.cuda.get_device_name(0)); print(torch.ones(1, device="cuda"))'
```

如果空闲 GPU 仍报告 busy/unavailable 或 ECC 错误，应保存 `hostname`、`nvidia-smi` 和作业号，临时使用 `SBATCH_EXCLUDE` 并报告管理员。

### 存储配额

```bash
getfattr -n ceph.quota.max_bytes "$HOME"
getfattr -n ceph.dir.rbytes "$HOME"
getfattr -n ceph.quota.max_bytes "$HOME/run"
getfattr -n ceph.dir.rbytes "$HOME/run"
```

只有可复用的 Hugging Face datasets Arrow 缓存放在 `~/run`；其余编译/运行缓存位于计算节点本地并随作业释放。模型、checkpoint、TensorBoard event、指标 JSONL 和 Slurm 日志仍需要足够的 `~/run` 配额。

### 取消作业

```bash
scancel 作业号
```

## 13. 最短执行清单

首次配置：

```text
检查配额与队列
→ 上传并展开代码到 ~/run
→ module load miniforge3 与 cuda/12.8
→ create_fresh_env.sh
→ prepare_assets.sh
→ 登录节点 check_env
→ sbatch 单卡 smoke
→ sacct 确认 COMPLETED 0:0
→ 8B probe
→ 正式 anchor/branches
```

以后重新登录：

```bash
module purge
module load miniforge3/25.11.0-1
module load cuda/12.8
cd "$HOME/run/cyt/src/verl-staleness"
sinfo
```

随后直接使用 `submit_slurm.sh` 提交，不需要重建环境。
