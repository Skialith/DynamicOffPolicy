# DynamicOffPolicy

研究固定每次 optimizer update 数据量时，能否利用 rollout 前段的学习动力学预测未来
策略漂移，并动态选择行为策略支持的连续更新次数 N。当前代码基于 verl 0.7.1，采用
Qwen3-8B-Base、GRPO 和 DAPO-Math-17K；科研口径及资源约束见 [AGENTS.md](AGENTS.md)。

## 代码与实验入口

| 目录 | 内容 |
| --- | --- |
| `verl/` | 完整训练框架；本项目的逐 update KL、staleness 和 Fisher 测量接入训练流程 |
| `examples/dynamic_staleness/` | 数据准备、环境检查、Slurm 提交及测量工具；见[运行说明](examples/dynamic_staleness/README.md) |
| `experiments/` | 按日期组织的实验配置、提交脚本和已确认分析；见[实验索引](experiments/README.md) |
| `tests/` | 框架测试和本项目 KL/staleness 回归测试 |
| `docs/`、`docker/`、`scripts/` | verl 的使用文档、运行环境和辅助脚本 |
| `集群使用指导/` | 并行云运行与环境说明 |

有限 logits/log-prob 差分代理已退役。直接全词表 KL、独立参数 JVP 和当前 Fisher K/B
测量保留；旧实验产物与原有 Git 历史用于追溯。退役说明见[历史实验族](experiments/20260919_logits_kl_geometry_validation/README.md)。

## 本地、GitHub 与集群

GitHub 主仓库为 [Skialith/DynamicOffPolicy](https://github.com/Skialith/DynamicOffPolicy)，
默认分支为 `main`。本地工作目录为 `/Users/Workspace/dynamicStaleness`；集群正式 Git
工作目录为 `/data/run01/scyb980/cyt/src/DynamicOffPolicy`。

本地完成修改、检查和提交后推送：

```bash
git status --short
git push origin main
```

集群在工作区干净且没有使用该目录的运行作业时更新：

```bash
cd /data/run01/scyb980/cyt/src/DynamicOffPolicy
git status --short
git pull --ff-only origin main
```

集群现有 `/data/run01/scyb980/cyt/src/verl-staleness` 保存共享 `assets/` 与 `.venv/`。
新 Git 目录通过本地符号链接复用它们，均由 `.gitignore` 排除。已提交的 Fisher 作业
继续使用 `/data/run01/scyb980/cyt/src/fisher_kqb_20261005`；这些旧部署目录不覆盖正在
执行的源码。后续作业从正式 Git 目录提交，提交前记录 `git rev-parse HEAD` 和工作区
状态到对应实验 README。

Git 保存代码、配置、文档、复现脚本和已确认派生产物。`raw/`、模型/optimizer 权重、
运行日志、缓存、环境和凭据留在 Git 之外；原始产物仍按[实验同步说明](experiments/README.md#原始产物与同步)归档。

## 来源

基础框架来自 [verl](https://github.com/verl-project/verl) 0.7.1。本次补齐的基础源码取自
2026-10-07 集群当前 Fisher 部署，保留现有项目定制文件及全部本地提交历史。许可证和
第三方声明见 [LICENSE](LICENSE) 与 [Notice.txt](Notice.txt)。
