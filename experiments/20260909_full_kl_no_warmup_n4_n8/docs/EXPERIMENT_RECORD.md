# 2026-09-09：无 warmup、等 24 轮 N=4/8 实验结果

实验设置、作业时间和原始产物入口见[实验 README](../README.md)。本记录只保留已经讨论
确认的图表和结果解释，不新增派生指标。

## 数据范围

- N=4：24 个 rollout、96 次真实 update；N=8：24 个 rollout、192 次真实 update。
- 每轮累计 KL 相对该轮训练侧 anchor，并在轮内固定的真实 rollout 前缀上测量。
- 能力评测为 MATH500 sampled Avg@1，训练前和每轮结束各一次。
- 原始输入分别位于 [`raw/n4/`](../raw/n4/) 和 [`raw/n8/`](../raw/n8/)。

## 确认图表

| 图表 | 回答的问题 | 生成脚本 |
| --- | --- | --- |
| [按真实 update 对齐的能力](../figures/ability_by_optimizer_step.png) | 在共同训练 update 处比较 N=4/N=8 能力 | [`plot_ability_by_optimizer_step.py`](../scripts/plot_ability_by_optimizer_step.py) |
| [按 rollout 周期的能力](../figures/ability_by_rollout_cycle.png) | 展示等 24 轮时两组实际训练量不同 | [`plot_ability_by_rollout.py`](../scripts/plot_ability_by_rollout.py) |
| [轮内累计 KL](../figures/rollout_block_cumulative_kl.png) | 展示每轮固定 anchor 下的 KL-age 形状 | [`plot_rollout_block_kl.py`](../scripts/plot_rollout_block_kl.py) |
| [每轮累计 KL 峰值](../figures/rollout_peak_cumulative_kl.png) | 比较峰值随训练进度的变化 | [`plot_rollout_block_kl.py`](../scripts/plot_rollout_block_kl.py) |
| [N=4 KL 与 gradient norm](../figures/rollout_block_cumulative_kl_grad_norm_n4.png) | 同步展示轮内 KL 和瞬时 gradient norm | [`plot_rollout_block_kl_grad_norm.py`](../scripts/plot_rollout_block_kl_grad_norm.py) |
| [N=8 KL 与 gradient norm](../figures/rollout_block_cumulative_kl_grad_norm_n8.png) | 同上；保留 N=8 的 age 5--8 | [`plot_rollout_block_kl_grad_norm.py`](../scripts/plot_rollout_block_kl_grad_norm.py) |

累计 KL 图和峰值图另有同名 PDF。所有曲线使用原始值、无平滑；跨 rollout 周期不连线。

## 讨论结论

1. 每轮累计 KL 都相对该轮自己的训练侧 anchor，并在轮内固定的 `q_t` 上测量；换
   rollout 后 anchor 和前缀集合一起更新。因此只连接同一周期内的 age 点，周期之间
   断线。等 24 个 rollout 周期用于收集曲线，不表示 N=4 与 N=8 训练量相同。

2. 在 N=8 分支内部、同一 anchor 和 `q_t` 下定义

   ```text
   E_t = max {K_t(a): a=1,...,4}
   T_t = max {K_t(a): a=5,...,8}
   ```

   24 个周期中有 18 个满足 `T_t > E_t`，两者 Pearson 相关为 0.9408。前四步对后四步
   有预测信号，但 `E_t` 不能直接作为未来峰值上界。本实验没有确定任何数值 KL 阈值。

3. MATH500 sampled
   Avg@1 在共同真实 update 48 时，N=4/N=8 分别为 0.788/0.784；在共同 update 96
   时分别为 0.806/0.802。等 24 个 rollout 周期的终点分别为 0.806/0.830，但 N=8
   使用了两倍 update。当前结果只说明额外更新没有在这项单 seed、sampled Avg@1
   测评上带来成比例收益；训练前 0.490/0.532 的差异也表明几个百分点的变化不能作为
   强能力结论。

4. `grad_norm_i` 是一次更新的瞬时量，累计 `K_t(a)` 包含此前全部更新，二者的同步图
   只能描述现象。局部关系应先用单独记录的 adjacent KL 检验，再研究多步
   累积。当前 gradient norm 均低于 norm clip 1.0；这与 PPO ratio clip 是两件事。
   长期科研命题、动力学假设与切换规则统一见 [`AGENTS.md`](../../../AGENTS.md)。
