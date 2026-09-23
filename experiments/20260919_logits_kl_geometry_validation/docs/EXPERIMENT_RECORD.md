# 2026-09-19：logits 几何 KL 校验实验记录

实验设置、测量字段、作业信息和原始产物入口见[实验 README](../README.md)。本记录目前
包含已经确认的 N=1 控制分析和 N=4 几何分析；4-update probe 仅用于工程验收，N=16/32
尚未确认分析方案，均不进入本记录的统计结果。

## N=1 控制结果

### 分析范围

本次只检验 job 167543 的 96 条逐 update 记录是否满足 N=1 应有的测量恒等式，以及
age=1 时固定-anchor有限 log-prob 差分二次量对精确全词表 KL 的近似误差。不比较能力、
不评价 N 的策略效果，也不外推到多步交叉项或未来 KL 预测。

分析由 [`analyze_n1_control.py`](../scripts/analyze_n1_control.py)从
`raw/n1_job167543/kl_updates.jsonl` 只读生成。逐步明细见
[`n1_control_per_update.csv`](../tables/n1_control_per_update.csv)，机器可读汇总见
[`n1_control_validation.json`](../tables/n1_control_validation.json)。

### 数据与恒等式闭合

| 检查 | N=1 结果 |
| --- | ---: |
| 记录完整性 | optimizer step 1–96 连续；96 个不同 `context_set_id`；每条 `age_after=1`、`policy_age=0` |
| 有效上下文与权重 | 每轮 510–512 个位置；`weight_sum=1` |
| `adjacent_kl` 与 `cumulative_kl` 最大绝对差 | `0` |
| `previous_cumulative_kl` 最大绝对值 | `0` |
| 精确 KL 三点交叉项 / 恒等式误差最大绝对值 | `0 / 0` |
| 有限差分累计项 / 交叉项最大绝对值 | `0 / 0` |
| 有限差分平方展开闭合误差最大绝对值 | `4.34e-19` |
| self-KL 最大绝对值 | `0` |
| `rho_eff` / 有限差分 cosine 的 defined 数 | `0 / 0`（共 96 条） |

N=1 每轮在一次更新后立即刷新，因此更新前相对 anchor 的累计方向为零。数据中精确和
有限差分交叉项均为零，`rho_eff` 与 cosine 的值虽以零占位，但相应 defined flag 全部
为 false；这些占位值不能解释成“方向正交”。这说明边界条件和未定义量处理符合设计。

### age=1 局部近似

精确 `cumulative_kl` 的中位数为 `4.4864e-4`，第 95 百分位为 `1.0587e-3`，完整范围为
`2.8413e-4`–`2.6534e-3`。N=1 下有限差分 step/current 二次量只在浮点舍入量级内不同，
因此它们对 adjacent/cumulative KL 的误差分布相同：

| 指标 | 结果 |
| --- | ---: |
| 预测/精确 KL 中位比 | `1.00110` |
| 有符号相对误差中位数 | `+0.110%` |
| 绝对相对误差中位数 | `0.526%` |
| 绝对相对误差第 90 / 95 百分位 | `1.239% / 1.310%` |
| 最大绝对相对误差 | `2.529%` |
| typical error factor | `1.00527` |

因此，在本次单 seed 的 96 个 age=1 测量点和上述 KL 范围内，有限 log-prob 差分二次量
对精确单步 KL 的典型误差约为半个百分点，且没有观察到测量恒等式或 N=1 边界条件失效。
这个结果只验证单步局部代理和记录实现，不能证明 age>1 时近似仍成立，也不能支持
N=1 优于其他 N、预测未来峰值或设定 KL 预算。该问题是数值闭合检查，表格比曲线更直接，
因此本阶段不新增图表。

## N=4 结果

### 分析范围

分析比较精确相邻/累计全词表 KL、固定-anchor有限差分二次能量、`rho_eff`、有限差分
cosine，以及 gradient norm/AdamW `update_norm` 到相邻 KL 的换算稳定性。所有派生统计
由 [`scripts/analyze_results.py`](../scripts/analyze_results.py) 从正式运行原始 JSONL
只读生成。

### 最简结果

派生统计仅使用正式运行的 `kl_updates.jsonl`。`typical factor` 定义为相对目标或中位数
的绝对对数误差中位数取指数；IQR factor 为 `q75/q25`。

| 问题 | 结果 |
| --- | --- |
| 原始记录是否闭合 | KL 三点恒等式最大绝对误差 `8.67e-19`；有限差分平方展开最大绝对误差 `4.34e-19`；96 个 `update_norm` 全部为正 |
| `rho_eff` 是否对应有限差分方向 | age 2–4 共 72 点：`rho_eff` 中位数 `-0.47634`，有限差分 cosine 中位数 `-0.47447`；两者 MAE `0.00417`、最大绝对差 `0.01487`、Pearson `r=0.99725` |
| 负号是否一致 | 72/72 的 `rho_eff` 和有限差分 cosine 都为负 |
| 有限差分二次能量是否贴近精确 KL | 相邻 KL 的近似/精确中位比 `1.00392`、典型误差因子 `1.00538`；累计 KL 为 `1.00694`、`1.00786` |
| 实际位移换算是否更稳定 | `d/(eta G)^2` 的典型波动因子 `1.2766`、IQR factor `1.5889`；`d/||u||^2` 分别为 `1.2481`、`1.6328`。按 age 的系数中位数最大/最小比由 `1.1156` 降至 `1.0519` |

按 age 的完整结果见 [`formula_alignment_by_age.csv`](../tables/formula_alignment_by_age.csv)、
[`local_quadratic_by_age.csv`](../tables/local_quadratic_by_age.csv)和
[`conversion_stability.csv`](../tables/conversion_stability.csv)。一页式机器可读汇总为
[`summary.json`](../tables/summary.json)，由
[`scripts/analyze_results.py`](../scripts/analyze_results.py)从 `raw/` 只读生成。

确认后的主图为
[`kl_formula_validation_2x2.png`](../figures/kl_formula_validation_2x2.png)（另有
[`PDF`](../figures/kl_formula_validation_2x2.pdf)）：上排直接比较有限差分二次量与实测
相邻/累计 KL；左下给出有符号相对误差及其中央 95% 区间；右下比较
`rho_eff` 与有限差分 cosine。图中只画理论参考线 `y=x`，不使用回归拟合线。绘图由
[`scripts/plot_formula_validation.py`](../scripts/plot_formula_validation.py)从正式运行的原始
JSONL 复现。

### 解释

第一，`rho_eff` 与有限差分 cosine 在所有 age 层都高度一致，且符号完全一致。结合
二次能量对精确 KL 约 0.5%–0.8% 的典型误差，可以把旧日志中持续为负的
`rho_eff` 更具体地解释为：在本实验的固定 anchor、固定前缀和全词表输出几何下，
新一步变化与此前累计变化呈负对齐，因而产生方向抵消。此前“相邻 KL 不能直接相加”
的观察获得了直接几何支持。

第二，这个结论仍限于**有限 log-prob 差分代理**。它说明输出分布的局部几何足以解释
当前步长下的 KL 交叉项，但没有直接测量参数空间 `J_0 Delta`、`J_0 u`，不能替代严格
JVP frozen-Fisher 校准，也不能单独区分参数非线性与 Fisher 漂移。

第三，实际 AdamW `update_norm` 的确减小了按 age 的系数中位数漂移，也略微改善了
典型偏差；但整体 IQR 反而略宽。因此本次单 seed 结果不支持“`update_norm` 换算显著
优于 raw gradient norm”的强结论。它仍是更接近理论链条的变量，但预测价值需要独立
运行和未来峰值任务验证。
