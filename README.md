# DynamicOffPolicy

本仓库只管理远端训练源码、配置、必要提交/测量脚本、回归测试和代码使用说明。
代码基于 verl 0.7.1，研究 Qwen3-8B-Base 全参数 GRPO 的动态 off-policy / staleness。
研究 idea、共同口径和协作边界见 [AGENTS.md](AGENTS.md)。

## 仓库分工

| 仓库 | 位置与内容 |
| --- | --- |
| 远端训练代码 | `/data/run01/scyb980/cyt/src/DynamicOffPolicy`；与本 GitHub 仓库同步 |
| 本地研究档案 | `/Users/Workspace/dynamicStaleness`；独立本地 Git，保存论文、实验文档、分析脚本、派生表和图表，不配置本代码仓库的 origin |

本地不维护训练框架源码。具体实验设置和结果在本地档案中记录，并引用远端代码 commit；
代码仓库不包含论文、实验 README/Record、原始数据、表格、图表、权重或运行环境。
两边各自保留 AGENTS.md，作为共同研究与协作边界。

## 代码入口

| 路径 | 用途 |
| --- | --- |
| `verl/` | 训练框架、逐 update KL/staleness 记录与 Fisher 测量接入 |
| `examples/dynamic_staleness/` | 环境检查、数据准备、Slurm 提交与测量工具 |
| `experiments/<experiment>/scripts/` | 对应设置的远端提交与测量脚本；分析/绘图脚本在本地档案仓库 |
| `tests/trainer/ppo/`、`tests/workers/actor/` | 本项目的 KL/staleness/Fisher 回归测试 |
| `scripts/`、`pyproject.toml`、`requirements*.txt` | 框架辅助工具、安装与依赖配置 |

有限 logits/log-prob 差分代理已删除，真实全词表 KL 和当前 Fisher K/B 测量保留。
训练侧 legacy FSDP functional 参数 JVP 已注释停用，保留最初逻辑考量；参数扰动、
logits 中心差分近似 JVP 也已注释停用，列为最后实现备选。
当前使用独立非 FSDP 模型的逐层参数/输入 JVP → softmax Fisher → 逐层 VJP，
见 [layerwise_fisher.py](examples/dynamic_staleness/layerwise_fisher.py)。
旧 JVP/FVP 联测入口拒绝启动，整网 double-backward 只保留为小模型数值参考。

训练内 Fisher 测量不设置子进程时间上限。`HVP_POWER_STEPS=0`（默认）持续迭代至
残差满足 `HVP_POWER_TOLERANCE`，正整数只用于显式指定的有限迭代 probe。
等待测量时，各训练 rank 每秒同步就绪状态，因此通信超时不再充当测量时限。
N=4/8 提交配方的 `TIME_LIMIT=0` 不设置人为作业时限。残差门槛可在提交时指定：

```bash
HVP_POWER_TOLERANCE=0.01 bash experiments/20261006_fisher_kqb_short_rollouts/scripts/submit_experiment.sh n4
```

门槛越大越容易提前收敛；该相对残差是 `||Fv-λv|| / ||Fv||`，并非 λ 的相对误差。

## 运行与代码同步

8B 训练和 GPU probe 在并行云 Slurm 执行，统一入口为
[submit_slurm.sh](examples/dynamic_staleness/submit_slurm.sh)。例如现有 Fisher K/B 设置的
提交入口为 [submit_experiment.sh](experiments/20261006_fisher_kqb_short_rollouts/scripts/submit_experiment.sh)；
具体参数由本地实验 README 和对应提交脚本记录。

共享模型、数据与 Python 环境在 `/data/run01/scyb980/cyt/src/verl-staleness`，
代码目录通过忽略的 `.venv`、`assets` 符号链接复用。已有作业继续使用原部署。
更新代码前核对工作区与使用该目录的作业，在可联网登录节点执行：

```bash
cd /data/run01/scyb980/cyt/src/DynamicOffPolicy
git status --short
git pull --ff-only origin main
```

代码在远端编辑、检查和提交；获得用户授权后再 push 到
[Skialith/DynamicOffPolicy](https://github.com/Skialith/DynamicOffPolicy)。
本地档案仓库不参与代码 push/pull。两边原有历史保留，不改写已发布提交。

基础框架来自 [verl](https://github.com/verl-project/verl) 0.7.1，保留本项目定制实现。
许可证与第三方声明见 [LICENSE](LICENSE) 和 [Notice.txt](Notice.txt)。
