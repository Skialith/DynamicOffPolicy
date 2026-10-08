# Tests and probes

本目录区分回归测试、独立检查和需要 Slurm GPU allocation 的 probe。

| 位置 | 用途 |
| --- | --- |
| `trainer/ppo/test_*.py` | KL 前缀、权重、update 计数和 staleness 指标回归 |
| `workers/actor/test_*.py` | 精确 HVP 与逐层 Fisher 的 CPU 数值回归 |
| `checks/` | 启动配置与已生成实验产物的独立检查 |
| `probes/` | FSDP 集成、资源、吞吐和 FVP 存储 probe |
| `probes/historical/` | 已停用的 legacy FSDP JVP/FVP 路线与历史输出检查 |

普通回归测试仍放在原有模块对应目录；GPU probe 不作为普通 pytest 测试运行。
Probe 在 Slurm 计算节点执行，登录节点只准备和提交。统一提交入口保持为
`examples/dynamic_staleness/submit_slurm.sh`，已有 target 名称保持不变。
例如 FVP 存储 probe 的专用提交入口现在为：

```bash
bash tests/probes/submit_fvp_storage_probe.sh
```

历史 shell 入口仍明确拒绝启动；functional 参数 JVP 的最初考量和中心差分 JVP 的
最后备选保留原注释。历史输出检查用于旧产物，不作为当前训练的验收入口。

`examples/dynamic_staleness/` 保留训练启动、资产准备与当前测量实现。
`verify_exact_kl_hvp.py` 中的函数仍被当前 Fisher 测量共用；
`verify_training_fisher.py` 仍被训练启动脚本和回归测试共用。
这两个文件以及测量实现内的自检留待后续拆分，不在本轮调整它们的接口。
