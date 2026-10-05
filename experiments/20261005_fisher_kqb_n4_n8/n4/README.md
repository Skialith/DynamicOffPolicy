# 2026-10-05：N=4 等 96 次更新 Fisher K/Q/B

## 任务与关系

按照用户确认设置，完成 96 次真实 update，同时逐 age 测量 K、Q、B_hat。
公式、测量精度和共同口径见 [实验族 README](../README.md)。与 N=8 对照按更新数及
trajectory 数对齐，不按 rollout 数对齐；本设置共 24 轮。

## 配置

| 项目 | 设置 |
| --- | --- |
| 模型 / 数据 / 算法 | Qwen3-8B-Base；DAPO-Math-17K SIS 预处理；全参数 GRPO |
| N / rollout batch / update mini-batch | 4 / 1024 prompts / 256 prompts；8 responses/prompt，ppo_epochs=1 |
| updates / trajectory 总量 | 96 / 196608；实际训练 token 另记 |
| seed / LR / warmup | 1 / 1e-6 固定 / 0 |
| 长度 / KL loss / PPO clip / entropy | 1024+3072；1e-3；0.2/0.2；0 |
| 资源 / runtime | 4×A800-80GB；TP=2；sis_offload；legacy FSDP+vLLM |
| 打包 / vLLM | actor、infer 每卡 16384 token；utilization=0.7，batched=8192，seqs=512 |
| reward / 评测 | 8 Ray Math-Verify workers；MATH500 sampled Avg@1，训练前和每轮后 |
| KL 前缀 | 每轮 64 prompts×每个所选 response 最多 8 位置，固定轮内 q；全词表 |
| λ / Q / B_hat | 每轮 anchor λ；每 age 精确 HVP；残差 1e-3，cap=200，超限失败 |
| 保存 | 原始前缀、逐 update 标量、评测生成、计时、显存、配置和日志；不保存训练后权重或 Adam |

## 提交与状态

独立部署：`/data/run01/scyb980/cyt/src/fisher_kqb_20261005/`；复用原训练仓库的 Python
环境与已有资产，不复制第二份基座。提交入口：

```bash
SLURM_DEPENDENCY=afterok:<probe-job-id> bash experiments/20261005_fisher_kqb_n4_n8/scripts/submit_experiment.sh n4
```

当前尚未提交正式作业；前置 probe 188038 在第一次 HVP 的第二次反传 OOM，失败。
修复和 probe 详情见 [N=8 的状态记录](../n8/README.md)；正式作业只依赖后续通过的 probe。
时限 7 天是资源兜底，不是估算完成时间。

提交前 CPU 自检：五项通过，包括真实 FP32 state_dict 往返、全参数微型 Qwen3
double backward、逐前缀加权 HVP 与显式 Fisher 一致、真实 update norm、因果位置
选择以及未收敛/缺失字段必须拒绝。静态编译与 shell 语法检查通过；这不替代 GPU probe。
既有三项 HVP 回归测试通过；真实入口的 N=8 配置预览已核对 12 轮、96 updates、
mini-batch=256、四卡可见、warmup=0 和 save_freq=-1。

## 产物

远端输出根：独立部署下 `experiments/20261005_fisher_kqb_n4_n8/n4/raw/formal/`，
每个作业采用时间/job/GPU/TP/seed/phase 唯一实例。以实际日志中的 `output=` 为准，
不猜“最新目录”。`kl_updates.jsonl` 含三项和 norm；`hvp_diagnostics/start_*/age_*.json`
含 scalar/memory 与 anchor 幂迭代 history；`hvp_validation.json` 为完成验收。
