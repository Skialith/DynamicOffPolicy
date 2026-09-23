# 2026-09-24：Frozen-Fisher 谱估计兼容性验证

## 实验任务

在不启动新的 N=16 训练前，验证当前 legacy FSDP 栈能否提供 Power Iteration 所需的
矩阵向量积 `F0 v`。本实验只回答工程兼容性与数值自检问题，不拟合 KL，也不据此选择
KL 预算或 rollout 更新次数。

并行检查两条路线：

1. 将 `KL(pi_anchor || pi_theta)` 作为普通 loss，用 `theta0 +/- epsilon v` 两点的一阶
   梯度中心差分近似 `F0 v`；该路线不调用 JVP 或二阶 autograd。
2. 在 legacy FSDP 中启用 `use_orig_params=True`，重试精确 `torch.func.jvp`，检查它是否
   消除此前的 `TensorWrapper` storage 错误。

若第 1 条小模型检查通过，再单独提交 Qwen3-8B 探针，测量 1--2 次 Power Iteration 的
显存、耗时、参数恢复误差和有限差分尺度稳定性。小模型通过不等于 8B 路径已经可用。

## 与其他实验的关系

[N=4 严格 frozen-Fisher JVP 校准](../20260919_fisher_jvp_n4/README.md)的 legacy FSDP
兼容性作业 166375 在 `torch.func.jvp` 进入 FSDP lazy init 时失败。本实验不重启该正式
校准，而是先排除 `use_orig_params=True` 的精确路径，并验证不依赖 JVP 的 Fisher-HVP
回退路径。

仓库已有的 `verify_fsdp_central_difference.py` 只验证真实 update 方向上的 logits
中心差分与 Fisher 二次型；它没有生成 `F0 v`，因此不能直接运行 Power Iteration。

## 实验配置

| 项目 | 设置 |
| --- | --- |
| 第一阶段模型 | 两层 toy MLP，只作 FSDP 与自动微分兼容性检查 |
| 分布式路径 | 4×A800、legacy FSDP、`use_orig_params=False/True` 分别检查 |
| Fisher-HVP | frozen-anchor 全类别 KL；梯度中心差分；默认 `epsilon=1e-2` |
| 数值自检 | 两种 FVP 路线一致性、能量恒等式、对称性、参数精确恢复、4 次 Power Iteration |
| 第二阶段模型 | Qwen3-8B-Base；仅在第一阶段通过后提交 |
| 保存 | Slurm 日志与约化标量；不保存 Fisher、完整方向或模型 checkpoint |

## 启动、时间与状态

第一阶段提交入口为
[`submit_compatibility_probes.sh`](scripts/submit_compatibility_probes.sh)。截至
2026-09-24，本地静态检查和非 FSDP 双精度公式自检通过，集群作业尚未提交。

本实验尚未提出结果分析方案，因此不创建 `docs/EXPERIMENT_RECORD.md`。

## 产物入口

- [兼容性提交脚本](scripts/submit_compatibility_probes.sh)
- [Fisher-vector-product 探针](../../examples/dynamic_staleness/verify_fsdp_fisher_vector_product.py)
- [精确参数 JVP 探针](../../examples/dynamic_staleness/verify_fsdp_parameter_jvp.py)
- [返回实验索引](../README.md)
