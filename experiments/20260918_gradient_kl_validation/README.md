# 2026-09-18：Gradient norm 与轮内 KL 回顾分析

## 实验任务

使用既有 N=4/N=8 逐更新日志，回顾性检验 gradient norm 与 adjacent KL 的换算、
累计 KL 的非加性，以及只使用前四步信息预测未来峰值的探索性留出结果。本实验没有
新的 GPU 训练或测量作业。

## 与其他实验的关系

输入来自[无 warmup、等 24 轮 N=4/8 实验](../20260909_full_kl_no_warmup_n4_n8/README.md)。
本目录 `raw/n4/`、`raw/n8/` 保存对应 `kl_updates.jsonl` 的逐字节副本；新增的是分析
方法和派生结果，不是新的训练证据。该分析暴露的累计 KL 非加性解释缺口，随后由
[logits 几何 KL 校验实验](../20260919_logits_kl_geometry_validation/README.md)继续测量。

## 分析设置与时间

| 项目 | 设置 |
| --- | --- |
| 输入 | N=4：24 轮/96 updates；N=8：24 轮/192 updates |
| 数据口径 | 每轮固定 anchor 与真实 rollout 前缀；轮间更换前缀 |
| 分析脚本 | [`scripts/analyze.py`](scripts/analyze.py) |
| 执行资源 | 本地 CPU |
| 完成日期 | 2026-09-18 |

## 产物入口

- [分析结果](docs/EXPERIMENT_RECORD.md)
- [输入副本](raw/)
- [派生表与汇总](tables/)
- [分析脚本](scripts/analyze.py)
- [返回实验索引](../README.md)
