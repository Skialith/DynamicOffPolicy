# DynamicOffPolicy

本仓库管理 Qwen3-8B-Base 全参数 GRPO 的训练源码，基于 verl 0.7.1。
当前 `codex/delayed-anchor-kl` 分支研究纯前向跨轮 KL：缓存旧 rollout 的固定因果
前缀和全词表分布，在后续真实更新后直接计算旧策略到当前模型的 KL。

本分支已移除 JVP/HVP/Fisher 测量实现、训练回调、参数位移计算、相关配置和旧
提交/probe 入口。原主 worktree 和 Git 历史保留旧路线。正常训练反向传播、
标准 GRPO clip、逐 update KL/staleness 记录与 FSDP micro-batch 同步修复保留。

## 入口与运行

- [跨轮 KL 使用说明](examples/dynamic_staleness/DELAYED_ANCHOR_KL.md)
- [提交配方](examples/dynamic_staleness/submit_delayed_kl.sh)：固定每次 update 的
  256 prompt groups × 8 responses，提供 N=4/8 观测配方。
- [Slurm 统一入口](examples/dynamic_staleness/submit_slurm.sh)：8B 训练和 GPU probe
  只在 Slurm 计算节点执行。
- [测试说明](tests/README.md)、[研究与协作约束](AGENTS.md)

新 worktree 为 `/data/run01/scyb980/cyt/src/DynamicOffPolicy-delayed-kl`，
原主 worktree 为 `/data/run01/scyb980/cyt/src/DynamicOffPolicy`。共享模型、数据
和 Python 环境在 `/data/run01/scyb980/cyt/src/verl-staleness`，通过忽略的
`assets` 和 `.venv` 符号链接复用。

N=4 路径上的跨轮八步 KL 尚需与真实 N=8 路径对照；当前不自动升级 N。

## 仓库分工

训练实现、配置、提交脚本和回归测试在本代码仓库管理。本地研究档案仓库
`/Users/Workspace/dynamicStaleness` 单独保存论文、实验 README/Record、分析脚本、
派生表和图表，以运行 commit 引用代码。原始产物、权重和环境不进入 Git。
修改前检查工作区并保留用户改动；获得用户授权后才向
[Skialith/DynamicOffPolicy](https://github.com/Skialith/DynamicOffPolicy) push。

基础框架来自 [verl](https://github.com/verl-project/verl) 0.7.1。许可证与第三方
声明见 [LICENSE](LICENSE) 和 [Notice.txt](Notice.txt)。
