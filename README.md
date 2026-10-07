# DynamicOffPolicy

本仓库管理动态 off-policy / staleness 研究所需的源码、配置、必要脚本和回归测试。
代码基于 verl 0.7.1，当前使用 Qwen3-8B-Base、GRPO 和 DAPO-Math-17K；
每次 optimizer update 的训练数据量固定，研究用 rollout 前段信息预测未来漂移并选择 N。
研究口径与运行约束见 [AGENTS.md](AGENTS.md)。

## 代码入口

| 路径 | 用途 |
| --- | --- |
| `verl/` | 训练框架、逐 update KL/staleness 记录与 Fisher 测量接入 |
| `examples/dynamic_staleness/` | 环境检查、数据准备、Slurm 提交与测量工具 |
| `experiments/<experiment>/scripts/` | 各实验需要的提交和复现脚本 |
| `tests/trainer/ppo/`、`tests/workers/actor/` | 本项目的 KL/staleness/Fisher 回归测试 |
| `scripts/`、`pyproject.toml`、`requirements*.txt` | 框架辅助工具、安装与依赖配置 |

有限 logits/log-prob 差分代理已删除。直接全词表 KL 和当前 Fisher K/B 测量保留。
实验文档、原始数据、表格、图表、权重和运行环境保留在本地或原有远端位置，
不随当前代码树同步。已有本地提交历史保留。

JVP 实现状态（2026-10-07）：

| 路线 | 状态与代码入口 |
| --- | --- |
| 训练侧 legacy FSDP 内直接 functional 参数 JVP | 已注释停用，保留最初逻辑考量：[actor](verl/workers/actor/dp_actor.py)、[兼容性探针](examples/dynamic_staleness/verify_fsdp_parameter_jvp.py) |
| 参数扰动后用 logits 中心差分近似 JVP | 已注释停用，列为最后 JVP 实现备选：[扰动探针](examples/dynamic_staleness/verify_fsdp_central_difference.py)；不同于已删除的更新前后输出差分 KL 代理 |
| 独立非 FSDP 测量模型的逐层参数/输入 JVP → softmax Fisher → 逐层 VJP | 当前保留实现：[layerwise_fisher.py](examples/dynamic_staleness/layerwise_fisher.py)；embedding、norm、head 等全参数参与 |

旧 JVP/FVP 联测提交入口会明确拒绝启动，不提交 GPU 作业。独立 KL 梯度中心差分对照
保留在 [compare_kl_hvp.py](experiments/20261004_kl_hvp_exact_vs_fsdp_fd/scripts/compare_kl_hvp.py)。
它通过两个普通 backward 近似 HVP；整网 double-backward 通过二阶自动微分计算同一
anchor Fisher，CPU 图 offload 与前向重算只改变存储/执行方式。后者保留作小模型参考，
当前长前缀测量使用逐层 JVP/VJP，避免保留整网二阶图。

## 运行

8B 训练和 GPU probe 在并行云 Slurm 执行。统一入口为
[submit_slurm.sh](examples/dynamic_staleness/submit_slurm.sh)，具体参数由实验提交脚本固定。
例如当前 Fisher K/B 短程设置使用
[submit_experiment.sh](experiments/20261006_fisher_kqb_short_rollouts/scripts/submit_experiment.sh)。
已有作业继续使用原部署目录；后续作业从代码目录提交，提交前记录所用 commit。

集群共享模型、数据与 Python 环境仍在 `/data/run01/scyb980/cyt/src/verl-staleness`。
正式代码目录通过 `.venv` 和 `assets` 符号链接复用它们；这些链接不进入 Git。

## 代码管理与同步

GitHub：[Skialith/DynamicOffPolicy](https://github.com/Skialith/DynamicOffPolicy)，分支 `main`。
本地代码与档案所在目录：`/Users/Workspace/dynamicStaleness`。
集群代码目录：`/data/run01/scyb980/cyt/src/DynamicOffPolicy`。

本地修改代码、检查并提交后：

```bash
git push origin main
```

首次在其他位置检出当前代码，可使用浅克隆；它只下载当前代码及这一提交：

```bash
git clone --depth 1 https://github.com/Skialith/DynamicOffPolicy.git
```

更新集群代码前核对工作区与使用该目录的作业，在可联网登录节点执行：

```bash
cd /data/run01/scyb980/cyt/src/DynamicOffPolicy
git status --short
git pull --ff-only origin main
```

基础框架来自 [verl](https://github.com/verl-project/verl) 0.7.1，保留本项目已有定制实现。
许可证与第三方声明见 [LICENSE](LICENSE) 和 [Notice.txt](Notice.txt)。
