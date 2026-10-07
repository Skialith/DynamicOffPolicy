# 2026-09-19：logits 几何 KL 校验实验族

本路线于 2026-10-07 否决并退役：既然已经获得更新后策略输出，就直接计算全词表
KL，有限 logits/log-prob 差分不再作为 KL、Fisher 长度或对齐量的代理。以下内容记录
历史实验；原始产物、已确认表格和图表保持原样，运行实现、配置开关和专用脚本已删除。

## 研究任务与目录关系

本实验族在每轮固定训练侧 anchor 和真实 rollout 前缀上，比较精确全词表 KL 与
固定-anchor有限 log-prob 差分二次量，回答这种局部几何在多大的策略复用 age 内仍能
近似相邻 KL、累计 KL 及两段位移的交叉项。

N=1、4、16、32 使用相同测量实现和共同训练口径，但各自拥有独立的配置、作业、原始
数据和实验记录，因此拆成四个子实验：

| 子实验 | 角色 | Rollouts / updates | README | 已确认分析 |
| --- | --- | ---: | --- | --- |
| N=1 | 无既有累计方向的边界控制 | 96 / 96 | [设置](n1_control/README.md) | [记录](n1_control/docs/EXPERIMENT_RECORD.md) |
| N=4 | 主校验 | 24 / 96 | [设置](n4_main/README.md) | [记录](n4_main/docs/EXPERIMENT_RECORD.md) |
| N=16 | 扩大 age | 6 / 96 | [设置](n16_large_age/README.md) | [记录](n16_large_age/docs/EXPERIMENT_RECORD.md) |
| N=32 | 最大已测 age | 3 / 96 | [设置](n32_large_age/README.md) | [记录](n32_large_age/docs/EXPERIMENT_RECORD.md) |

四个子实验相关，是因为它们共同检验同一组 KL 几何公式；结果不合并成一份记录，是
因为不同 N 的 rollout 数、anchor、训练轨迹和统计重复数不同，不能把 384 行数据当成
同一批独立样本。跨 N 只做描述性对照，不作配对或因果比较。

本实验承接 [Gradient/KL 回顾分析](../20260918_gradient_kl_validation/README.md)：旧日志
发现累计 KL 不能由相邻 KL 直接相加，但不能判断负残差是否对应输出几何中的方向抵消。
它沿用[无 warmup N=4/8 实验](../20260909_full_kl_no_warmup_n4_n8/README.md)的模型、
优化器和单次 update 数据量。[严格 frozen-Fisher JVP 校准](../20260919_fisher_jvp_n4/README.md)
测量参数到 logits 的 JVP，测量对象和工程路径不同，仍是独立实验。

## 共同配置

| 项目 | 设置 |
| --- | --- |
| 模型 / 算法 / 数据 | Qwen3-8B-Base，全参数 GRPO，DAPO-Math-17K |
| 每次 update 数据量 | 256 prompt group × 8 responses；2048 trajectories；`ppo_epochs=1` |
| 学习率 / warmup / seed | `1e-6 / 0 / 1` |
| Runtime | 4×A800-80GB、TP=2、`sis_offload` |
| 全词表测量 | 每轮 64 prompts，每个所选 response 最多 8 个有效位置 |
| 测量开关 | `FULL_KL_EXPERIMENT=1`、`FULL_KL_GEOMETRY=1` |
| 权重 | 远端保存终点 HF 权重；本地不复制权重、optimizer checkpoint 或 `kl_contexts/*.pt` |

## 四组比较的计算与数学意义

以下先写单个固定上下文；实际日志先在词表上精确求和，再按实验约定的 prompt/有效位置
权重聚合。设本轮 anchor、当前更新前和当前更新后的策略分别为
`p₀=π_t`、`p₋=π_(t+a-1)`、`p₊=π_(t+a)`，并定义：

```text
d_a       = KL(p₋ || p₊)                         # adjacent_kl
K_(a-1)   = KL(p₀ || p₋)                         # previous_cumulative_kl
K_a       = KL(p₀ || p₊)                         # cumulative_kl
C_a       = K_a - K_(a-1) - d_a                 # kl_three_point_cross
```

`C_a` 是精确的 KL 三点交叉项；当时的代码也直接用
`Σ_v (p₀(v)-p₋(v))(log p₋(v)-log p₊(v))` 计算它并检查恒等式。若 `C_a<0`，
这一步与此前累计变化在 KL 意义下发生抵消，所以累计 KL 小于两段 KL 的直接相加。

再令 `Δ=log p₋-log p₀`、`s=log p₊-log p₋`，并在 anchor 分布下中心化：
`x̃=x-E_(p₀)[x]`。记录的有限差分二次量是：

```text
A_fd = 1/2 E_(p₀)[Δ̃²]                           # frozen_fisher_fd_cumulative
D_fd = 1/2 E_(p₀)[s̃²]                           # frozen_fisher_fd_step
T_fd = 1/2 E_(p₀)[(Δ̃+s̃)²]                     # frozen_fisher_fd_current
X_fd = E_(p₀)[Δ̃ s̃]                             # frozen_fisher_fd_cross
T_fd = A_fd + D_fd + X_fd
```

这里的 “frozen Fisher” 指在 anchor 分布 `p₀` 下冻结权重后，用**有限 log-prob 差分**
构造的 Fisher 型二次几何；它不是参数空间的严格 JVP。四个比较分别是：

1. `D_fd` 对 `d_a`：检验一步有限差分能量是否近似精确相邻 KL。age>1 时前者仍由
   `p₀` 加权，而后者由 `p₋` 加权，因此误差也反映 anchor 与当前策略的分布漂移。
2. `T_fd` 对 `K_a`：直接检验从 anchor 到当前策略的累计 KL 二阶近似。
3. `X_fd` 对 `C_a`：检验二次几何是否重构精确三点交叉项；符号决定累计变化是相互
   增强还是抵消。
4. `cos_fd=X_fd/[2√(A_fd D_fd)]` 对
   `rho_eff=C_a/[2√(K_(a-1)d_a)]`：去除两段变化的尺度，检验方向对齐量。

age=1 时 `p₋=p₀`，没有既有累计方向，所以第 3、4 项记为不适用；日志中的零只是
占位值，不能解释为方向正交。

## 误差指标

前两项的精确值严格为正，报告 `预测/精确` 中位比、绝对相对误差的中位数与 P95，
以及最大误差。第 3 项可正、可负且可能接近零，不使用不稳定的 `X_fd/C_a-1`，改用
原始 MAE、Pearson、符号一致率及
`(X_fd-C_a)/[2√(K_(a-1)d_a)]` 的绝对值。第 4 项直接报告 `|cos_fd-rho_eff|`
的中位数、P95、最大值、Pearson 和符号一致率。

这些统计描述已观测轨迹上的回顾性近似程度，不是置信区间，也不构成未来 KL 预测。
同一 rollout 内的行共享 anchor 和上下文集合，不能视为 IID 重复；每个 age 的实际重复
数分别只有 N=1/4/16/32 的 96/24/6/3。

## 公共产物

- [统一远端同步脚本](../scripts/sync_remote_raw_artifacts.sh)
- [返回实验索引](../README.md)

## 历史脚本复现

专用提交和分析脚本已从当前代码树删除，可从退役前提交 `211675f` 取回。统一分析
入口当时为 `scripts/analyze_kl_approximation.py`，N=4 补充图入口为
`n4_main/scripts/plot_formula_validation.py`。在独立目录解出历史实验族，再放入原有
`raw/` 输入即可复现已有分析；当前配置不再接受旧的 `FULL_KL_GEOMETRY` 开关。

```bash
mkdir -p /tmp/logits-kl-history
git archive 211675f experiments/20260919_logits_kl_geometry_validation | tar -x -C /tmp/logits-kl-history
```
