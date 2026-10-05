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
N=8。仅训练 mini-batch 缩为 8 prompts（rollout batch=64），8 次 update；关闭能力
评测与权重保存。它验证 export/offload/子进程四卡/HVP/reload 与八个 age 的对接，
不是能力、训练吞吐或正式曲率结论。逐步 K/Q/B 缺失、λ 未收敛、首步分片 norm 与
完整差值不一致或 scratch 残留都会使 probe 失败；正式两组设置 `afterok` 依赖。

## 提交与状态

部署：`/data/run01/scyb980/cyt/src/fisher_kqb_20261005/`；复用原仓库 Python 与模型
资产，不复制基座。入口：

```bash
bash experiments/20261005_fisher_kqb_n4_n8/scripts/submit_experiment.sh probe
SLURM_DEPENDENCY=afterok:<probe-job-id> bash experiments/20261005_fisher_kqb_n4_n8/scripts/submit_experiment.sh n8
```

当前准备提交；probe 时限 24 小时，正式时限 7 天。实际 job ID、时间、依赖和核验
状态在提交后补充；时限不是可靠完成时间预测。

提交前 CPU 自检：五项通过，包括 FP32 权重往返、全参数微型 Qwen3 二阶反传、
加权 HVP 对照显式 Fisher、因果前缀选择、实际参数差值 norm 和失败门槛。静态编译与
shell 语法检查通过；不将 CPU 自检等同于真实四卡训练集成已通过。
既有三项 HVP 回归测试通过；正式 N=8 配置预览通过，确认 12 轮、96 updates、
mini-batch=256、四卡可见、warmup=0 和 save_freq=-1。

## 产物

远端根：部署下 `experiments/20261005_fisher_kqb_n4_n8/n8/raw/integration_probe/` 和
`n8/raw/formal/`；每作业唯一实例，以日志的 `output=` 为准。
`kl_updates.jsonl` 含三项和 norm；`hvp_diagnostics/start_*/age_*.json` 保存逐 age 标量、
显存和 anchor 幂迭代 history；`hvp_validation.json` 是结束验收。probe 与正式不混合。
