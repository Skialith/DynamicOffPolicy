# Tests and probes

| 位置 | 用途 |
| --- | --- |
| `trainer/ppo/test_*.py` | 直接 KL、跨轮缓存、因果前缀、权重、计数及 staleness 回归 |
| `checks/` | 已生成训练产物的独立检查 |
| `probes/` | Slurm 上的 FSDP 集成、资源与吞吐检查 |

本分支已删除 Fisher/JVP/HVP 数值测试、历史 probe 和求导测量入口。旧版本可在
原主 worktree 和 Git 历史中查看。GPU probe 不作为普通 pytest 测试运行，统一
通过 `examples/dynamic_staleness/submit_slurm.sh` 提交。

跨轮 KL CPU 回归：

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH="$PWD" .venv/bin/python tests/trainer/ppo/test_delayed_anchor_kl.py
```

`delayed_kl_probe` 先检查四卡 FSDP 空/不等测量 rank、禁用求导 API、参数与 RNG
不受观测影响，再运行缩小 batch/长度的 Qwen3-8B 两轮集成。它验证调用链和
产物，不能代替正式 batch 的成本与阈值实验。
