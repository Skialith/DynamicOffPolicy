# 2026-09-03：10-step warmup 全词表 KL 实验结果

实验设置、作业时间和原始产物入口见[实验 README](../README.md)。本记录只保留已经完成
的结果汇总、图表口径和结论边界。

两组均为 4×A800、TP=2、seed=1，各完成 96 次真实 optimizer update。能力评测采用 MATH500 sampled Avg@1，每个检查点 500 题；时间格式为 HH:MM:SS。

| 指标 | N=4 | N=8 |
| --- | ---: | ---: |
| 作业号 | 154027 | 154028 |
| 真实 optimizer updates / rollout 周期 | 96 / 24 | 96 / 12 |
| MATH500 sampled Avg@1，update 0 | 50.0% | 49.8% |
| MATH500 sampled Avg@1，update 48 | 77.6% | 77.8% |
| MATH500 sampled Avg@1，update 96 | 82.2% | 78.8% |
| 训练 trajectory 数 | 196,608 | 196,608 |
| 实际 response token 数 | 257,872,854 | 238,305,586 |
| 实际 prompt token 数（按 trajectory 计） | 23,679,240 | 23,679,240 |
| 训练 token 合计 | 281,552,094 | 261,984,826 |
| 训练周期累计耗时 | 19:15:14 | 17:51:36 |
| 平均每次 update 耗时 | 722.02 秒 | 669.75 秒 |
| 逐步全词表 KL 测量计时 | 845.09 秒（00:14:05） | 770.59 秒（00:12:51） |
| KL 测量计时 / 训练周期耗时 | 1.22% | 1.20% |
| Update 48、96 能力评测计时合计 | 117.32 秒 | 106.13 秒 |
| 终点 HF 权重保存计时 | 45.72 秒 | 45.92 秒 |
| 平均吞吐（4 卡合计） | 4,061.97 token/s | 4,074.66 token/s |
| Slurm 作业总耗时 | 19:23:25 | 18:01:00 |

## 终点双向 KL

共同的 64 个 prompt、512 个前缀位置，全词表 KL，temperature=1；两方向使用相同前缀和聚合权重。

| 方向 | KL |
| --- | ---: |
| N4 → N8 | 0.003147946 |
| N8 → N4 | 0.003563722 |

终点双向 KL 由作业 154028 执行，额外耗时约 78.9 秒；依据 endpoint.lock 到 endpoint_kl.json 的修改时间差估算，包含前缀准备、模型加载和评分，不是独立记录的精确计时。

## 计数与计时口径

- 训练 token 是实际非 padding token；总量来自各周期 perf/total_num_tokens，response 来自 96 条全局 response_mask token_count，prompt 为二者之差。Prompt 按每条 trajectory 计数，包含同题 8 条 response 对应的重复 prompt；不含能力评测、KL 评分的重复计算量，也不是各次前向/反向计算 token 数的总和。
- 不通过 response_length/mean 反推精确 token 数：N=8 的浮点均值反推总数比全局计数多 4 个 token。
- 训练周期耗时为 sum(timing_s/step)，包含 rollout、训练侧评分、更新、全词表 KL 测量、权重同步和终点 HF 保存；不含周期外能力评测、初始化及终点分支间 KL。
- KL 测量计时 = 每轮 anchor_seconds 只计一次 + 96 条 measurement_seconds；该时间已包含在训练周期耗时和 update_actor 中，不能再次相加。计时含首轮自 KL 检查；不含未单独计时的前缀选择/落盘等操作。这是日志计时汇总，不是关闭测量开关后的对照净开销。
- Update 48、96 的能力评测计时单独记录；update 0 未单独记录完整计时，不将该项当作全部三次评测耗时。
- Slurm 总耗时包含完整作业过程；N=8 额外执行一次终点双向 KL。

## 当前对比

N=8 的训练周期耗时减少 7.24%，训练 token 数减少 6.95%，按实际 token 归一化后的吞吐提高 0.31%。总耗时缩短主要伴随 token 数下降，不能把全部耗时差解释为减少 rollout 刷新带来的吞吐提升。
Update 96 的 sampled Avg@1：N=4 高 3.4 个百分点；update 48 两组相差 0.2 个百分点。这是每组一个训练 seed 的观测结果。

## 确认图表

| 图表 | 记录内容 | 生成脚本 |
| --- | --- | --- |
| `01`--`04` KL/gradient 曲线 | 相邻 KL、rollout-anchor 累计 KL，以及 N=4/N=8 gradient norm；横轴为真实 optimizer update | [`plot_kl_and_grad.py`](../scripts/plot_kl_and_grad.py) |
| `05`--`06` KL-by-age | 每个 rollout 周期单独成线；横轴为更新后 `age_after` | [`plot_kl_by_age.py`](../scripts/plot_kl_by_age.py) |
| [`07_cumulative_kl_vs_optimizer_step_aligned.png`](../figures/07_cumulative_kl_vs_optimizer_step_aligned.png) | N=4/N=8 按真实 update 对齐的累计 KL，每个 rollout 周期独立成段 | [`plot_cumulative_kl_aligned.py`](../scripts/plot_cumulative_kl_aligned.py) |
| [`08_training_reward_vs_optimizer_step.png`](../figures/08_training_reward_vs_optimizer_step.png) | 训练 reward 随真实 update 的变化 | [`plot_training_reward.py`](../scripts/plot_training_reward.py) |
| [`rollout_block_cumulative_kl.png`](../figures/rollout_block_cumulative_kl.png) | 固定各轮 anchor 后的累计 KL 轮内曲线 | [`plot_rollout_block_kl.py`](../scripts/plot_rollout_block_kl.py) |
| [`rollout_peak_cumulative_kl.png`](../figures/rollout_peak_cumulative_kl.png) | 每轮累计 KL 峰值 | [`plot_rollout_block_kl.py`](../scripts/plot_rollout_block_kl.py) |

编号 `01`--`08` 的图同时保存 PDF。所有曲线使用原始值、无平滑；累计 KL 直接比较
rollout 起点模型和当前模型，不是相邻 KL 的和。每轮固定 anchor 与前缀，轮间更换，
因此不跨 rollout 周期连线。

## 原始依据与复现

获取时间（UTC）：2026-09-05T02:42:00.552907+00:00。
原始日志标量、评测、计时记录、Slurm 状态与计数函数源码保存在
[`raw/raw_final_results_snapshot.json`](../raw/raw_final_results_snapshot.json)；
[`scripts/summarize_final_results.py`](../scripts/summarize_final_results.py)生成 CSV 和未舍入
JSON，本记录摘录其中的确认值。逐步 KL 数值与曲线使用
[`raw/raw_kl_snapshot.json`](../raw/raw_kl_snapshot.json)。
