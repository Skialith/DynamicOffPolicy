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
2026-10-05 18:22:05（Asia/Shanghai）核验：

| phase | job ID | 提交时间 | 开始时间 | 依赖 / 当前状态 |
| --- | --- | --- | --- | --- |
| 修复后集成 probe | 188641 | 18:18:06 | 18:18:31 | 无；`RUNNING` |
| N=8、96 updates 正式 | 188660 | 18:20:16 | 尚未开始 | `afterok:188641`；`PENDING (Dependency)` |

probe 在 `d1n41a28g03`，实际 4×A800-SXM4-80GB、32 CPU、502400 MiB 主存。
当前日志仍在模型/rollout 引擎初始化，尚无 8B HVP 或残差收敛完成记录；不记为通过。
probe 完成全部 8 次真实 update、K/Q/B 及清理验收后，正式作业才可启动。正式请求
四卡，尚未实际分配节点；结束时间与退出状态未知。

probe 确定实例：`20261005-181832_job188641_g4_tp2_seed1_hvp_integration_probe`，
位于 `n8/raw/integration_probe/`，其下 `fisher_kqb_probe_n8_u0008/` 为实际输出。
两份日志分别为部署下 `logs/slurm/fisher-kqb-integration-g4-188641.out` 和
`logs/slurm/fisher-kqb-n8-g4-188660.out`。正式设置为 12 轮、96 updates，不沿用
probe 的 mini-batch=8。提交时账户空间约 38.72 GB/268.44 GB；没有复制基座或把
测量权重写入持久目录，`SAVE_FREQ=-1`。

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

- [188038 原始失败日志](raw/integration_probe_job188038/fisher-kqb-integration-g4-188038.out)：已归档，raw 不入 Git。
