# 2026-10-04：小模型精确 KL-HVP 与 FSDP 梯度差分对照

## 实验任务

在用户指定的 165 服务器上，对同一固定模型、参数方向和前缀分布，比较：

1. 非 FSDP、跨 GPU 按层放置模型，用 frozen-anchor KL 的两次反向求导得到精确 HVP；
2. legacy FSDP 真正分片，局部分片参数在 `theta0 +/- epsilon v` 两点做普通 backward，
   用梯度中央差分近似同一个 HVP。

二者的目标均为 anchor 处全参数、全词表的 Fisher，不是训练 loss 的一般 Hessian。
anchor 概率固定且 detach；没有 optimizer、训练更新或 rollout。Power Iteration 只改变
方向向量，不更新 anchor。这里的“精确”指自动微分而非差分，不表示无浮点误差。

## 与其他实验的关系

[2026-09-24 兼容性 probe](../20260924_fisher_spectral_compatibility/README.md)尝试了
legacy FSDP 参数 JVP 和差分路线。本次不重试原 JVP 设置，而是比较两个独立的可行路径。
采用独立实现，不直接使用 HessFormer 公共源码；差分仅作为此次对照，不改变精确路线
优先、差分最后备选的研究顺序。

## 实验配置

| 项目 | 设置 |
| --- | --- |
| 阶段 | 随机初始化微型 Qwen3 自检，然后已有 Qwen3-0.6B-Base checkpoint |
| 微型模型 | 2 层、hidden=32、intermediate=64、4 query heads、2 KV heads、vocab=128，tied embedding/head |
| 前缀 | 微型模型：固定 2×8 token；0.6B：固定四个数学 prompt；每个 prompt 取末尾两个位置 |
| 聚合 | prompt 等权，两个有效位置等权；每个位置使用全部 vocabulary |
| 资源 | 165，2×A800-80GB，GPU 0/1；精确路线一进程按层跨卡，差分路线两 rank FULL_SHARD |
| 精度 | FP32 模型；输出 log-softmax/KL 用 FP64，向量内积 FP64 累计；关闭 TF32，eager attention、eval、无 cache |
| seed | 模型/前缀 20261004；方向 20261005/20261006 |
| 训练 | 0 optimizer updates；N、学习率、能力评测不适用 |
| 差分尺度 | 从 `1` 至 `1e-5` 的 11 个尺度，方向全局单位范数；每次由 base 精确恢复 |
| 谱算法 | 同一初始向量的 Power Iteration；记录每步 Rayleigh quotient 和相对特征残差 |
| 自检 | anchor 一致性、FVP 相对误差/余弦、PSD、对称性、扰动确实改变输出、base 精确恢复 |
| 保存 | 原始 JSON、stdout/stderr、环境清单；参考大向量仅远端保存，不同步或进入 Git |

时间和显存仅是小模型工程诊断，不能直接外推 8B；两种模型放置方式不同，不据此作纯算法
速度排名。正式分析方案尚未确认，不建立 `docs/EXPERIMENT_RECORD.md` 或派生图表。

### 单卡数值参考补验

用户进一步要求检查残差与实际特征值误差的关系。单卡补验将完整 0.6B 模型放在 GPU 0，
不使用 FSDP 或按层跨卡；沿用最终两卡运行保存的参数方向和前缀，验证 anchor 与随机
方向 FVP 一致性。仍是 FP32 模型、FP64 输出 KL，不构造完整 Fisher 矩阵。
Power Iteration 阈值收紧为 `1e-5`，使用 seed `20261005/20261007` 两个初始方向。
原两卡作业的 `1e-3` 阈值不回改。各方法相对该单卡数值参考的误差写入原始自检 JSON；
单卡参考不是解析真值，也不单凭残差宣称已证明求到了全局最大特征值。

残差定义为 `||Fv - lambda v|| / ||Fv||`，其中 `v` 全局单位范数；它衡量特征方程的
一致性，不是与另一个方法作比较。单卡补验启动脚本为
[`run_single_reference_165.sh`](scripts/run_single_reference_165.sh)。
补验实例 `single_reference_20261004_200135_GNV6FU` 于 20:01:35 开始、20:02:46 结束，
两个 seed 均完成、收敛并通过 anchor/FVP 一致性检查，退出码 0；只使用 GPU 0。
原始标量分别在 `seed_20261005/exact.json`、`seed_20261007/exact.json`，误差相对于
原两卡实例 `qwen06_20261004_194558_cnh3Ei`，不改写其历史文件。
20:05:45 核验两张 GPU 均约 21 MiB、0% utilization，无计算进程。

## 启动、时间与状态

2026-10-04：165 SSH 已恢复；GPU 0/1 各使用约 21 MiB，未见计算进程。已有环境
`/home/ymy/cyt/dynamicoffpolicy/verl-staleness/.venv/bin/python`：Python 3.12.13、
PyTorch 2.8.0+cu128、Transformers 4.57.1。已有模型资产：
`/home/ymy/cyt/dynamicoffpolicy/verl-staleness/assets/models/Qwen3-0.6B-Base`。

按用户明确指定，本次小模型实验在 165 执行，不提交 Slurm，不运行 8B。启动脚本为
[`run_compare_165.sh`](scripts/run_compare_165.sh)，先 `tiny`，再 `qwen06`；脚本在启动前
检查现有 GPU 计算进程，使用两个 GPU，串行执行精确和差分路径，无后台任务。

确定的远端根目录：`/home/ymy/cyt/dynamicoffpolicy/kl_hvp_probe_20261004/`。
以下时间均为 Asia/Shanghai，状态核验日期为 2026-10-04；本次没有 Slurm job ID。

| raw 实例名 | 开始 → 结束 | 设置与运行状态 |
| --- | --- | --- |
| `tiny_20261004_194011_TaZwPj` | 19:40:11 → 19:40:30 | 首次 FP32-KL 检查；精确通过，差分在原 epsilon 范围内验收失败 |
| `tiny_20261004_194213_up1BWQ` | 19:42:13 → 19:42:32 | 扩展 epsilon 扫描；FP32-KL 两路线通过 |
| `qwen06_20261004_194303_JSG5o8` | 19:43:03 → 19:43:44 | FP32-KL 精确路径的 anchor-gradient 自检失败，未启动差分阶段 |
| `tiny_20261004_194538_JwJAnn` | 19:45:38 → 19:45:58 | 最终 FP64-KL 配置；两路线通过，退出码 0 |
| `qwen06_20261004_194558_cnh3Ei` | 19:45:58 → 19:47:28 | 最终 FP64-KL 配置；精确 AD 通过，FSDP 差分数值验收失败，退出码 1 |

最终 0.6B 的退出码 1 来自显式数值验收，不是 OOM、FSDP 前后向不兼容或 SSH 中断。
所有失败实例保留，不覆盖或删除。`exact.json` 和 `fd.json` 包含实际 FVP 误差、每步
特征残差、尺度扫描、耗时和显存。差分迭代自身 residual 达标不自动等于真实 Fisher
特征值准确；验收同时检查相对精确参考的误差。

输出 KL 改用 FP64 的原因是保持 anchor 处一阶梯度接近零，而不是放宽失败阈值。
两次最终运行的模型参数仍为 FP32。没有冻结参数块、使用 TOP-K 或修改训练主流程。
2026-10-04 运行后核验：两张 GPU 均回到约 21 MiB、0% utilization，无计算进程。

复现（把本目录 scripts 复制到上述远端根目录的 scripts 后，在 165 执行）：

```bash
bash scripts/run_compare_165.sh tiny
bash scripts/run_compare_165.sh qwen06
bash scripts/run_single_reference_165.sh
```

每次自动建立新的 raw 实例，JSON 用独占创建防止覆盖。精确 `.pt` 参考包含大向量，
仅留在远端 raw；本地只归档 JSON、日志和启动/完成文本，均不进入 Git。

## 产物入口

- [独立探针](scripts/compare_kl_hvp.py)
- [165 启动脚本](scripts/run_compare_165.sh)
- [最终微型模型原始自检](raw/tiny_20261004_194538_JwJAnn/)
- [最终 0.6B 原始自检](raw/qwen06_20261004_194558_cnh3Ei/)
- [单卡精确补验原始自检](raw/single_reference_20261004_200135_GNV6FU/)
- [实验索引](../README.md)
