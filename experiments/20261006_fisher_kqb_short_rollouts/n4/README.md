# 2026-10-06：N=4 正常训练 batch 的 5 轮 Fisher K/Q/B

## 任务与关系

完成 5 个 rollout 周期、20 次真实 optimizer update，逐 age 测量 K/Q/B。共同公式和
实现见 [实验族 README](../README.md)。与 N=8 的 3 轮设置共同观察多轮测量，两组终点
训练量不同；共同完整周期评测时点为 update 0、8、16。

## 配置

| 项目 | 设置 |
| --- | --- |
| 模型 / 数据 / 算法 | Qwen3-8B-Base；DAPO-Math-17K SIS 预处理；全参数 GRPO |
| N / rollout / updates | 4 / 5 / 20 |
| rollout batch / update mini-batch | 1024 / 256 prompt groups；8 responses/prompt，ppo_epochs=1 |
| 训练总量 | 5120 prompt groups、40960 trajectories；实际 token 另记 |
| seed / LR / warmup | 1 / 固定 1e-6 / 0；同初始模型，训练不恢复 checkpoint |
| 长度 / KL loss / clip / entropy | 1024/3072，right；1e-3；0.2/0.2；0 |
| 资源 / runtime | 4×A800-80GB，TP=2，sis_offload，legacy FSDP+vLLM |
| 打包 / vLLM | actor/infer 每卡 16384 token；utilization=0.7，batched=8192，seqs=512 |
| reward / 评测 | 8 Ray Math-Verify workers；MATH500 sampled Avg@1，训练前与每轮后 |
| 测量 | 64 prompts×最多 8 位置；全参数、全词表、FP32/FP64；每轮 λ、每 age K/Q/B |
| 谱门槛 / timeout | residual≤1e-3，cap=200；测量 8 小时、进程组 10 小时 |
| 保存 | SAVE_FREQ=-1；不保存训练后权重或 Adam，临时测量快照轮末清理 |

评测 update 为 0、4、8、12、16、20；每轮 age_after=1..4。完整结束验收预期 20 条
`kl_updates.jsonl` 和 25 份 age 报告（5 份 anchor、20 份更新后报告）。

## 提交、时间与状态

在现有独立部署 `/data/run01/scyb980/cyt/src/fisher_kqb_20261005/` 同步本次脚本和 README
后，从部署根运行：

```bash
bash experiments/20261006_fisher_kqb_short_rollouts/scripts/submit_experiment.sh n4
```

| 项目 | 记录 |
| --- | --- |
| 前置验收 | 190665 已通过全 8 ages K/Q/B；该 probe 的训练 mini-batch 为 8，正常 256 的整体资源仍需运行核验 |
| job / 提交、开始、结束时间 | 未提交，尚无 job ID 或运行时间 |
| 计划依赖 / 时限 | 默认无 Slurm 依赖；可显式传入 SLURM_DEPENDENCY；时限 7 天 |
| 请求资源 | gpu_a800、4 GPU；实际 CPU、主存、节点待分配后核验 |
| 核验状态 | 2026-10-06：脚本语法、实际训练入口参数预览和 README 链接检查通过；尚未部署或提交 |

提交前核查部署源码、队列和持久空间；保留当前 CPU offload，不同时切换新的内存后端。
测量 8 小时和作业 7 天均为异常时限，不是完成时间预测。结束后仍需核对正常训练、
逐步测量、残差、scratch 清理和收尾异常。

## 产物入口

远端输出根为部署下 `experiments/20261006_fisher_kqb_short_rollouts/n4/raw/formal/`，
实例编码时间/job/GPU/TP/seed/phase，实际路径以日志 `output=` 为准；实验名为
`fisher_kqb_short_n4_u0020`。Slurm 日志为部署下
`logs/slurm/fisher-kqb-short-n4-g4-<job-id>.out`。

运行后包含 `kl_updates.jsonl`、`hvp_diagnostics/start_*/age_*.json`、结束验收
`hvp_validation.json`、评测逐题生成、配置、计时和 TensorBoard。当前仅已有本 README
及共用提交脚本，没有 raw 或派生产物，不建立空目录。
