# 实验目录

实验目录按 `YYYYMMDD_<topic>_<variant>` 命名。每个实验从根目录 README 进入：README
记录任务、关系、配置、作业和时间；只有已经确认分析的实验才有
`docs/EXPERIMENT_RECORD.md`。共同研究口径与文档规范见 [AGENTS.md](../AGENTS.md)。

## 原始产物与同步

各实验的 `raw/` 保存从并行云复制的逐 update JSONL、逐题评测、resolved config、
run manifest、TensorBoard event 和 Slurm 日志。模型权重、optimizer checkpoint 与
`kl_contexts/*.pt` 不复制到本地，也不进入 Git。原始产物按确定的 job ID 和输出实例
同步，不使用“最新目录”匹配，且不删除已有本地文件。

统一同步入口：

```bash
experiments/scripts/sync_remote_raw_artifacts.sh
```

## 实验索引

| 实验 | 运行与本地归档状态 | 结果记录 | 入口 |
| --- | --- | --- | --- |
| 2026-10-06：正常训练 batch 的 Fisher K/B 短程多轮 | N=4 五轮/20 updates、N=8 三轮/24 updates；10-07 已提交两组 K/B-only 作业，等待调度；无 MATH500、旧 actor KL 或逐 age Q | 尚未要求分析 | [实验族入口](20261006_fisher_kqb_short_rollouts/README.md) |
| 2026-10-05：等 96 次更新 Fisher K/Q/B N=4/8 | 10-06 probe 190665 已完成 K/Q/B 工程验收；原 96-update 两组保持取消 | 尚未要求分析 | [实验族入口](20261005_fisher_kqb_n4_n8/README.md) |
| 2026-10-04：8B 四卡精确 KL-HVP 工程探针 | 187793/187794 串行完成，两个初值均第 30 次迭代残差达标；无新增权重 | 尚未要求正式分析 | [README](20261004_exact_kl_hvp_8b_4gpu/README.md) |
| 2026-10-04：小模型精确 KL-HVP 与 FSDP 梯度差分对照 | 165 对照完成；0.6B 跨卡/单卡精确 AD 通过，固定尺度差分数值验收失败 | 原始自检已归档，尚未要求正式分析 | [README](20261004_kl_hvp_exact_vs_fsdp_fd/README.md) |
| 2026-09-24：Frozen-Fisher 谱估计兼容性验证 | 本地公式与静态检查通过；jobs 169831/169835 排队中 | 尚未要求分析 | [README](20260924_fisher_spectral_compatibility/README.md) |
| 2026-09-19：N=1/4/16/32 logits 几何 KL 校验实验族 | 四个子实验均完成 96 updates；原始产物和记录按设置分开 | [N=1](20260919_logits_kl_geometry_validation/n1_control/docs/EXPERIMENT_RECORD.md) · [N=4](20260919_logits_kl_geometry_validation/n4_main/docs/EXPERIMENT_RECORD.md) · [N=16](20260919_logits_kl_geometry_validation/n16_large_age/docs/EXPERIMENT_RECORD.md) · [N=32](20260919_logits_kl_geometry_validation/n32_large_age/docs/EXPERIMENT_RECORD.md) | [实验族入口](20260919_logits_kl_geometry_validation/README.md) |
| 2026-09-19：N=4 严格 frozen-Fisher JVP 校准 | 兼容性作业失败/取消；未产生正式运行原始数据 | 尚未要求分析 | [README](20260919_fisher_jvp_n4/README.md) |
| 2026-09-18：Gradient norm 与轮内 KL 回顾分析 | 本地 CPU 分析完成 | [结果](20260918_gradient_kl_validation/docs/EXPERIMENT_RECORD.md) | [README](20260918_gradient_kl_validation/README.md) |
| 2026-09-09：无 warmup、等 24 轮 N=4/8 | 作业 158523/158524 完成，原始产物已归档 | [结果](20260909_full_kl_no_warmup_n4_n8/docs/EXPERIMENT_RECORD.md) | [README](20260909_full_kl_no_warmup_n4_n8/README.md) |
| 2026-09-03：10-step warmup 全词表 KL | N=4/8/16 均完成 96 updates；N=16 收尾脚本失败不影响训练记录 | [结果](20260903_full_kl_warmup/docs/EXPERIMENT_RECORD.md) | [README](20260903_full_kl_warmup/README.md) |
| 2026-09-02：首轮动态 Staleness Pilot | N=4/8/16 分支均完成 update 100→196 | [结果](20260902_staleness_pilot/docs/EXPERIMENT_RECORD.md) | [README](20260902_staleness_pilot/README.md) |

历史 N=4/8/16 各 96 步测量计划和当时的执行记录仍在
[原计划](../examples/dynamic_staleness/FULL_PARAMETER_KL_PLAN.md)及其关联文档中。

## 已确认图表与生成脚本

这里只列出当前仓库中有明确生成脚本的正式图表；图表的数据口径和结果解释以各实验的
`EXPERIMENT_RECORD.md` 为准。

| 实验 | 图表 | 生成脚本 |
| --- | --- | --- |
| 首轮 Staleness Pilot | `math500_branches_by_optimizer_step.{png,pdf}` | [生成脚本](20260902_staleness_pilot/scripts/plot_math500_branches.py) |
| 10-step warmup N=4/8 | `01`--`04` KL/gradient 图 | [生成脚本](20260903_full_kl_warmup/scripts/plot_kl_and_grad.py) |
| 10-step warmup N=4/8 | `05`--`06` KL-by-age 图 | [生成脚本](20260903_full_kl_warmup/scripts/plot_kl_by_age.py) |
| 10-step warmup N=4/8 | `07` 累计 KL 对齐图 | [生成脚本](20260903_full_kl_warmup/scripts/plot_cumulative_kl_aligned.py) |
| 10-step warmup N=4/8 | `08` 训练 reward 图 | [生成脚本](20260903_full_kl_warmup/scripts/plot_training_reward.py) |
| 10-step warmup N=4/8 | rollout block 与 peak KL 图 | [生成脚本](20260903_full_kl_warmup/scripts/plot_rollout_block_kl.py) |
| 无 warmup N=4/8 | 能力曲线 | [按 update](20260909_full_kl_no_warmup_n4_n8/scripts/plot_ability_by_optimizer_step.py)、[按 rollout](20260909_full_kl_no_warmup_n4_n8/scripts/plot_ability_by_rollout.py) |
| 无 warmup N=4/8 | rollout block 与 peak KL 图 | [生成脚本](20260909_full_kl_no_warmup_n4_n8/scripts/plot_rollout_block_kl.py) |
| 无 warmup N=4/8 | N=4/N=8 KL 与 gradient norm | [生成脚本](20260909_full_kl_no_warmup_n4_n8/scripts/plot_rollout_block_kl_grad_norm.py) |
| logits 几何 KL 校验（N=1/4/16/32） | 各子实验的 `kl_approximation_2x2` 与 `kl_approximation_error_by_age` | [统一生成脚本](20260919_logits_kl_geometry_validation/scripts/analyze_kl_approximation.py) |
| logits 几何 KL 校验（N=4 补充图） | `kl_formula_validation_2x2.{png,pdf}` | [生成脚本](20260919_logits_kl_geometry_validation/n4_main/scripts/plot_formula_validation.py) |
