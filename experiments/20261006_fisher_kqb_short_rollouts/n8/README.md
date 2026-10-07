# 2026-10-06：N=8 正常训练 batch 的 3 轮 Fisher K/B

## 任务与关系

完成 3 个 rollout 周期、24 次真实 optimizer update，逐 age 测量 K/B。共同公式和
实现见 [实验族 README](../README.md)。与 N=4 的 5 轮设置共同观察多轮测量，两组终点
训练量不同。本次不做能力评测。

## 配置

| 项目 | 设置 |
| --- | --- |
| 模型 / 数据 / 算法 | Qwen3-8B-Base；DAPO-Math-17K SIS 预处理；全参数 GRPO |
| N / rollout / updates | 8 / 3 / 24 |
| rollout batch / update mini-batch | 2048 / 256 prompt groups；8 responses/prompt，ppo_epochs=1 |
| 训练总量 | 6144 prompt groups、49152 trajectories；实际 token 另记 |
| seed / LR / warmup | 1 / 固定 1e-6 / 0；同初始模型，训练不恢复 checkpoint |
| 长度 / KL loss / clip / entropy | 1024/3072，right；1e-3；0.2/0.2；0 |
| 资源 / runtime | 4×A800-80GB，TP=2，sis_offload，legacy FSDP+vLLM |
| 打包 / vLLM | actor/infer 每卡 16384 token；utilization=0.7，batched=8192，seqs=512 |
| reward / 评测 | 8 Ray Math-Verify workers；不做 MATH500 或其他能力评测，val_before_train=false、test_freq=-1 |
| 测量 | 64 prompts×最多 8 位置；全参数、全词表、FP32/FP64；每轮 λ、每 age K/B；暂停旧 actor KL 和逐 age Q |
| 谱门槛 / timeout | residual≤1e-3，cap=200；测量 8 小时、进程组 10 小时 |
| 保存 | SAVE_FREQ=-1；不保存训练后权重或 Adam，临时测量快照轮末清理 |

不评测；每轮 age_after=1..8。完整结束验收预期 24 条
`kl_updates.jsonl` 和 27 份 age 报告（3 份 anchor、24 份更新后报告）。

## 提交、时间与状态

在现有独立部署 `/data/run01/scyb980/cyt/src/fisher_kqb_20261005/` 同步本次脚本和 README
后，从部署根运行：

```bash
bash experiments/20261006_fisher_kqb_short_rollouts/scripts/submit_experiment.sh n8
```

| 项目 | 记录 |
| --- | --- |
| 前置验收 | 190665 已通过全 8 ages K/Q/B；该 probe 的训练 mini-batch 为 8，正常 256 的整体资源仍需运行核验 |
| job / 提交、开始、结束时间 | 未提交，尚无 job ID 或运行时间 |
| 计划依赖 / 时限 | 默认无 Slurm 依赖；可显式传入 SLURM_DEPENDENCY；时限 7 天 |
| 请求资源 | gpu_a800、4 GPU；实际 CPU、主存、节点待分配后核验 |
| 核验状态 | 2026-10-07：已部署 K/B-only；13 项 CPU 单元测试、实际 Hydra 配置、脚本语法和 diff 检查通过；待提交 |

2026-10-07 提交前核查：账户无在队列作业；Ceph 配额 268435456000 bytes、
已用 38755453154 bytes。保留当前 CPU offload。CPU 检查执行
`CUDA_VISIBLE_DEVICES= PYTHONPATH="$PWD" <asset-python> -m unittest tests.workers.actor.test_training_fisher tests.workers.actor.test_exact_kl_hvp`，13 项通过；真实 Hydra `--cfg job` 核对正常 batch、轮数及 K/B-only 开关通过。
测量 8 小时和作业 7 天均为异常时限，不是完成时间预测。结束后仍需核对正常训练、
逐步测量、残差、scratch 清理和收尾异常。

## 产物入口

远端输出根为部署下 `experiments/20261006_fisher_kqb_short_rollouts/n8/raw/formal/`，
实例编码时间/job/GPU/TP/seed/phase，实际路径以日志 `output=` 为准；实验名为
`fisher_kqb_short_n8_u0024`。Slurm 日志为部署下
`logs/slurm/fisher-kqb-short-n8-g4-<job-id>.out`。

运行后包含 `kl_updates.jsonl`、`hvp_diagnostics/start_*/age_*.json`、结束验收
`hvp_validation.json`、配置、计时和 TensorBoard；无评测逐题生成和 Q 指标。当前仅已有本 README
及共用提交脚本，没有 raw 或派生产物，不建立空目录。
