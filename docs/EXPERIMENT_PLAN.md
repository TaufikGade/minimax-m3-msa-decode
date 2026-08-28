# MiniMax M3 MSA 小 Batch Decode 实验设计（两人 / 两天）

## 1. 目标与最终交付

研究问题：在 B300 上，MiniMax M3 MSA 的小 batch decode 是否值得从现有 Triton split-K 实现迁移到专用 kernel？

本项目不预设必须获得正向加速。最终结论可以是：

1. 某些小 batch 形状值得实现专用 kernel；
2. 只值得做轻量 Triton 优化，不值得新增 CUDA/CUTLASS 路径；
3. 收益上限低于工程成本，不值得继续优化。

两天内必须完成：

- 可复现的 benchmark 脚本；
- Triton 基线及性能数据；
- 至少一组 Nsight Systems / Nsight Compute 证据；
- 对六个讨论点给出“结论 + 证据”；
- 一个挑战实验，包括失败或负收益结果；
- 原始 CSV/JSON 数据、报告和 10 分钟答辩材料。

> 当前仓库快照没有任务书中提到的 `harness/`，需要自行补一个最小 benchmark harness。

## 2. 硬件选择

### 结论：主实验只用 B300

B300 是本题两天内的唯一主线平台，原因如下：

- 上游对照路径是 `fmha_sm100` CUTLASS kernel；
- batch=16 的 crossover 是在 SM100 路径上得到的；
- cluster、mbarrier 和 TMA 的讨论以 SM100 为中心；
- 同一张卡上比较 Triton、CUTLASS 和挑战实现，才能避免跨 GPU 对比失真。

5090 只用于 B300 排队期间开发 Python harness 或检查基本正确性。两天时间内不做完整的 5090 性能矩阵，也不把 5090 数据放入主结论。

### B300 机时应急预案

三个 B300 窗口均视为**必须提前预约**，不是临时建议。第一次上机前必须在 5090 或无卡环境完成 dry-run：参数解析、数据生成、结果落盘、统计和 Nsight 命令均应可直接执行，B300 机时只用于正式运行。

触发条件与降级策略：

- Day 1 正式测量窗口延迟超过 2 小时：立即缩减为 TP1、FP8、batch=1/8/16，优先得到 baseline、CUTLASS crossover 和一个 Nsight Systems trace；
- Day 1 结束仍无 B300 数据：暂停 kernel 实现，完成 harness、正确性、理论分析和文档考证；报告明确标记“SM100 实测证据不足”，不得用 5090 性能外推 B300 结论；
- Day 2 最终复测窗口延迟超过 2 小时：冻结已有 B300 数据，不再增加挑战变体，只复测最终候选与 baseline；
- 完全无法获得 B300：提交复现工具、5090 功能验证和分析框架，但结论限定为“当前证据不足以判断 B300 是否值得实现”，不能声称验证了 batch=16 crossover。

历史 B300 数据只能作为旁证，并明确来源、软件版本和不可比因素；不能和本次 5090 数据拼接成主性能曲线。

## 3. 范围控制

### 3.1 固定真实形状

- top-k blocks：16；
- page/block size：128；
- head dimension：128；
- query heads / KV heads：TP1 和 TP4 对应的真实本地形状；
- decode query length：主实验固定为 1；
- KV cache：FP8 为主，BF16 只做一组控制实验；
- batch：1、4、8、16，附加 batch=32 用来确认 crossover 趋势。

### 3.2 两天内明确不做

- 不实现完整、通用、可上游合并的 CUDA kernel；
- 不覆盖所有 dtype、head dimension、top-k 和 page size；
- 不跑完整模型端到端吞吐；
- 不同时优化 lightning indexer；
- 不在 5090 和 B300 上各跑一套完整矩阵；
- 除非 profile 证明 merge 是主要瓶颈，否则不做 cluster/mbarrier 融合。

测量范围默认是：**top-k 已生成之后的 sparse attention decode**。Indexer 时间单独标注，不混入 kernel 主结果。

## 4. 阶段一：复现与测量

### 4.1 最小正确性验收方案

在开始挑战实验前，将本节发给另一组审查。

参照实现分两层：

1. 根据 `topk_idx` 和 `block_table` gather 选中 KV，使用 PyTorch FP32 实现 attention；
2. 使用仓库中的 Triton 实现作为第二参照。

最小测试按风险分级，不做全部维度的笛卡尔积。

一级必测组合：

- 非 128 整数倍 sequence length；
- 随机离散 page；
- 乱序 top-k；
- per-token FP8 scale；
- batch=1 和 batch=16；
- 检查输出中的 NaN 和 Inf。

二级验证组合（一级通过后执行）：

- batch=4、8；
- sequence length=127、128、2048；
- 连续 page；
- 升序 top-k；
- scalar FP8 scale。

二级项目采用单因素变化，不与其他维度做全排列。

所有随机输入固定并记录 seed。Triton、CUTLASS 和挑战实现必须复用完全相同的输入 tensor、page table 和 top-k 顺序。

误差指标：

- max absolute error；
- max relative error；
- cosine similarity；
- `torch.testing.assert_close`。

候选容差不直接作为最终口径。先测 Triton 相对 FP32 reference 的误差分布，再要求挑战实现的误差不显著差于 Triton。BF16 可先用 `rtol=1e-2, atol=1e-2` 作为起点。

正确性中止条件：若 Triton baseline 出现 NaN/Inf、地址错误，或相对 FP32 reference 的误差明显超过预期量级，立即停止性能实验，先排查 reference、scale 口径、mask 和输入布局。禁止带着未解释的 baseline 错误继续比较性能。

### 4.2 Benchmark 方法

每个形状执行：

- 预热 100 次；
- 正式测量至少 500 次；
- 使用 CUDA event 计时并正确同步；
- 报告 median、p95 和标准差；
- 预分配所有 workspace，计时区间内禁止创建 tensor；
- 首次 JIT/plan 时间单独记录，不计入 warm latency；
- Triton、CUTLASS 和挑战实现使用相同输入。

正式矩阵前，选择 batch=16 的固定配置，执行 10–20 轮相互独立的测量，每轮包含完整预热和正式计时，计算 run-to-run 变异系数：

$$
CV = \frac{\sigma_{run}}{\mu_{run}}
$$

后续只有当性能差异同时大于预设阈值和至少两倍噪声底线时，才称为稳定收益：

$$
\Delta_{significant} > \max(\Delta_{threshold}, 2CV)
$$

环境噪声控制：

- 运行前确认没有其他计算进程占用 GPU；
- 若集群规则和权限允许，锁定 GPU graphics clock，并记录命令和频率；
- 若不允许锁频，记录每轮实际 clocks、temperature、power 和 throttling reason，并增加独立重复；
- 不擅自关闭集群统一配置的 auto-boost；
- 不同实现交错或随机顺序运行，避免温度和功耗状态与实现顺序绑定；
- 自动保存环境 manifest：GPU/driver/CUDA/PyTorch/Triton/CUTLASS/vLLM commit、`nvidia-smi` 摘要和 Python package 版本。

需要分开记录：

- Triton partial kernel；
- Triton merge kernel；
- 两个 kernel 合计；
- host dispatch / kernel 间隙；
- CUTLASS kernel；
- CUTLASS metadata 或 plan 更新；
- CUDA Graph replay。batch=1、4 下 Triton 与 CUTLASS 各至少测一组，是必测项；若某实现无法 capture，记录失败原因和生产影响。

### 4.3 主实验矩阵

| 维度 | 配置 |
|---|---|
| GPU | B300 |
| batch | 1、4、8、16、32 |
| decode query length | 1 |
| TP shape | TP1、TP4 |
| KV dtype | FP8 |
| page 分布 | 随机离散为主，连续 page 做控制 |
| 实现 | Triton；强制 CUTLASS；挑战版本 |

BF16 只测 batch=1、8、16 的 TP1，避免实验矩阵膨胀。

### 4.4 Profile 计划

Nsight Systems 对 batch=1、8、16 各抓一次，回答：

- partial、merge 和 launch gap 分别占多少；
- batch 增大后 CUTLASS 在哪里超过 Triton；
- CUDA Graph 是否显著改变结论。

其中 batch=1 的 Triton/CUTLASS 需要同时保留 eager 与 CUDA Graph trace，以区分 host launch overhead 和 device kernel 本体。

Nsight Compute 只抓最关键的三个 case：

- Triton，batch=1；
- Triton，batch=16；
- CUTLASS，batch=16。

重点指标：

- DRAM throughput 和实际传输 bytes；
- L2 hit rate；
- achieved occupancy；
- active warps / active CTAs；
- registers per thread；
- Tensor Core pipe utilization；
- kernel grid、wave 数和执行时间。

## 5. 阶段二：分析框架

报告中的每个小节必须使用固定格式：

> **结论：** 一句话回答问题。  
> **证据：** 给出测量、公式、profile 截图或文档依据。  
> **局限：** 说明该证据尚不能证明什么。

### 5.1 Arithmetic intensity 与 Tensor Core

选中 token 数为：

$$
N = 16 \times 128 = 2048
$$

对一个 KV head，设 GQA group size 为 $$g$$，QK 与 PV 的总 FLOPs 近似为：

$$
F \approx 4gNd
$$

其中：

$$
d = 128
$$

TP1 下通常有：

$$
g = \frac{64}{4} = 16
$$

因此每个 KV head 的计算量约为：

$$
F \approx 4 \times 16 \times 2048 \times 128
  \approx 16.8\ \text{MFLOPs}
$$

FP8 K/V 的最低读取量约为：

$$
B_{KV} \approx 2Nd = 512\ \text{KiB/KV head}
$$

若 K/V 能在整个 GQA group 内理想复用，则理想 arithmetic intensity 上限约为：

$$
AI_{ideal}
\approx \frac{4gNd}{2Nd}
= 2g
\approx 32\ \text{FLOP/byte}
$$

实际还需加入 top-k、block table、scale、partial output、LSE 和 merge 的流量。因此最终必须同时报告理论 AI、Nsight 实测 bytes 和 achieved FLOP/s，不能只根据代码中存在 `tl.dot` 就断言 Tensor Core 有价值。

### 5.2 partial 与 merge 是否值得融合

先测量 merge 及其相关 launch gap 在总时间中的占比：

$$
R_{merge} =
\frac{T_{merge} + T_{launch\ gap}}
{T_{partial} + T_{merge} + T_{launch\ gap}}
$$

如果假设能够完全消除 merge，则乐观加速上限为：

$$
S_{max} =
\frac{T_{total}}
{T_{total} - T_{merge} - T_{avoidable\ traffic}}
$$

决策规则仅在差异超过第 4.2 节测得的噪声底线时生效：

- 占比低于 5%：不做融合，直接形成负结论；
- 占比高于 15%：允许投入半天做融合或近似融合原型；
- 5%–15%：优先做低风险 Triton 调参，融合只写可行性分析。

### 5.3 TMA 与两级间接寻址

访问路径是：

```text
topk_idx → logical block → block_table → physical page → K/V
```

需要验证的假设是：静态 TMA tensor map 不能直接表达完整的两级运行时 gather。两天内不实现复杂动态 descriptor，只做以下证据：

- 查阅 CUDA Programming Guide / CUTLASS 文档；
- profile 连续 pages 与随机 pages 的延迟和 L2 hit rate；
- 说明普通 global/vector load 或“先解析 page，再搬运 tile”的可行路径；
- 估算动态 descriptor 成本是否可能超过单个 page 的搬运收益。

### 5.4 FP8 scale

严格采用仓库 `test_sparse_attn_fp8_scale.py` 的语义。比较 scalar 与 per-token scale 后回答：

- scale load 是否进入关键访存路径；
- scale 应在 K/V load 后立即应用，还是可以与转换/计算融合；
- per-token scale 是否显著增加寄存器或访存压力。

### 5.5 batch=16 crossover

不能把源码中的门槛直接当作结论。强制在 batch=1、4、8、16、32 运行 CUTLASS。下面的线性式只作为零阶解释模型，不默认 GPU 延迟随 batch 线性变化：

$$
T(B) = T_{fixed} + B \times T_{request}
$$

拟合后必须检查 residual。若 batch=8→16 附近出现系统性残差或 grid wave 数跳变，则改用按 wave 区间分段的模型：

$$
T(B) = T_{fixed,k} + B \times T_{request,k},
\qquad B \in \text{wave region } k
$$

若数据点不足以可靠分段，则直接报告实测 crossover 位于哪两个 batch 档位之间，并注明不确定性为至少一个测试档位，而不是给出伪精确的交点。

重点回答：

- CUTLASS 是否固定成本更高、单位请求成本更低；
- 小 batch 是否因为 grid/wave 太小而无法发挥 Tensor Core；
- metadata/plan 更新是否影响 crossover；
- TP1 和 TP4 的 crossover 是否一致。
- crossover 附近是否与 grid/wave quantization 的台阶一致。

## 6. 阶段三：挑战与止损规则

### 6.1 默认挑战：改良 Triton

两天时间下，默认不从零写完整 CUDA kernel。挑战按以下顺序进行：

1. 做一次有预算上限的配置 sweep：`num_warps`、`num_stages`、`NUM_TOPK_CHUNKS` 和少量合法 block 配置；
2. 为 batch=1、4、8 分别寻找最佳 split 数；
3. 检查当前 `BLOCK_SIZE_H` 是否浪费线程或寄存器；
4. 减少 partial/LSE workspace 或不必要的中间读写；
5. 只有 merge 占比超过 15% 时，尝试融合或单 kernel 原型。

配置 sweep 最多占用 45 分钟或 24 个候选配置，以先到者为准。先在 batch=1、8 上筛选，再把前两名复测到 batch=4、16；禁止开启无界 autotune 导致大量 JIT 编译。

每个实验只改变一个因素，并记录：

- 正确性；
- latency；
- registers；
- occupancy；
- 失败原因。

### 6.2 半天止损点

在第一天结束前作出 go/no-go 决定。

继续实现挑战 kernel，至少需要满足以下一项：

- merge 和可避免流量占总延迟超过 15%；
- Triton Tensor Core/occupancy 明显偏低，且存在明确的特化修正方法；
- 最佳 split 配置相对基线已有至少 5% 且超过两倍噪声底线的稳定收益；
- roofline 与实测之间存在至少约 20% 的可信空间。

否则停止复杂实现，转向“不值得做”的完整证据链。

### 6.3 负结论的完成标准

若挑战没有加速，需要同时给出：

1. Triton 当前时间分解；
2. 完全消除 merge 后的理论加速上限；
3. 实测 roofline 或硬件利用率；
4. CUTLASS 在小 batch 下的 occupancy/grid 证据；
5. 专用 kernel 必须处理的复杂度：paged KV、两级索引、FP8 scale、softmax、reduction、TP shape 和 CUDA Graph；
6. 为什么预计收益不足以覆盖实现与维护成本。

最终结论句式：

> 在 B300、TP__、batch≤__ 下，即使完全消除 __，理论加速上限也只有 __%；实测 __ 已达到估算上限的 __%。考虑到专用 kernel 还需处理 __，因此建议 __。

## 7. 两人具体分工

### 成员 A：复现、正确性与数据负责人

第一天上午：

- 建立最小 harness；
- 生成 paged KV、block table 和 top-k；
- 实现 FP32 reference；
- 接通 Triton baseline；
- 完成最小正确性测试。

第一天下午：

- 跑 Triton/CUTLASS 主矩阵；
- 输出统一 CSV；
- 跑 Nsight Systems；
- 计算 median、p95 和 crossover 曲线。

第二天：

- 对成员 B 的挑战版本做独立正确性验收；
- 复跑全部关键性能数据；
- 整理图表、环境和复现命令；
- 负责答辩中的实验方法、正确性和性能结果。

### 成员 B：分析、profile 与挑战负责人

第一天上午：

- 阅读 Triton partial/merge 和 CUTLASS dispatch；
- 推导 FLOPs、bytes 和理想 AI；
- 列出六个讨论点的待验证假设；
- 准备 Nsight Compute 命令与指标表。

第一天下午：

- 对 batch=1/16 Triton 和 batch=16 CUTLASS 做 Nsight Compute；
- 分析 merge 占比、grid/wave、occupancy、带宽和 Tensor Core；
- 与成员 A 共同完成 go/no-go 决定；
- 选择一个挑战方向。

第二天上午：

- 实现并测试 Triton split 数/launch 配置优化；
- 若证据充分，再尝试最小融合原型；
- 每 90 分钟保留一个可测版本，禁止整上午只写未运行代码。

第二天下午：

- 完成六个“结论 + 证据”；
- 量化收益上限与工程成本；
- 负责答辩中的瓶颈分析、挑战结果和最终建议。

### 共同责任

- 开始优化前共同确认验收方案；
- 第一天结束前必须做 go/no-go 决策；
- 成员 A 不参与挑战实现的性能调参，保证独立复测；
- 成员 B 不单方面修改 tolerance 或删减失败 case；
- 所有主图只使用同一台 B300、同一软件环境的数据。
- 上机前完成 dry-run；任何 5090 结果不得外推为 B300 主结论。

## 8. 两天时间表

| 时间 | 成员 A | 成员 B | 里程碑 |
|---|---|---|---|
| Day 1 09:00–10:00 | 环境与 harness | 源码路径与理论模型 | 基线能运行 |
| Day 1 10:00–12:00 | reference 与正确性 | Nsight 指标和讨论点假设 | 验收方案冻结 |
| Day 1 13:00–15:00 | 主矩阵 benchmark | Nsight Compute | 第一版数据 |
| Day 1 15:00–17:00 | CUTLASS crossover | profile 分析 | 找到瓶颈 |
| Day 1 17:00–18:00 | 数据复核 | 收益上限估计 | go/no-go 决策 |
| Day 2 09:00–12:00 | 独立验证脚本 | 挑战实现 | 至少一个可测版本 |
| Day 2 13:00–15:00 | 完整复测与画图 | 挑战迭代或负结论 | 最终数据冻结 |
| Day 2 15:00–17:00 | 报告实验部分 | 报告分析部分 | 报告完成 |
| Day 2 17:00–19:00 | 联合制作答辩 | 联合制作答辩 | 两轮计时排练 |

必须提前预约以下三个 B300 窗口：

1. Day 1 上午：环境验证；
2. Day 1 下午：完整基线和 profile；
3. Day 2 下午：最终冻结复测。

任何窗口延迟超过 2 小时，立即执行第 2 节的范围降级，不等待到窗口结束才调整计划。

## 9. 10 分钟答辩结构

控制在 7 页正文：

1. **问题与最终结论（45 秒）**：先回答值不值得做；
2. **形状与现有实现（60 秒）**：Triton split-K + merge、CUTLASS 门槛；
3. **验收和测量方法（60 秒）**：reference、容差、计时范围；
4. **基线与 crossover（90 秒）**：batch-latency 主图；
5. **瓶颈证据（120 秒）**：时间分解、roofline、Nsight；
6. **挑战及结果（120 秒）**：加速、失败或收益上限；
7. **六点结论表与建议（105 秒）**：每点一句结论和一条证据。

备份页放置：

- FP8 tolerance 与 scale；
- TP1/TP4 差异；
- 连续/随机 page 对比；
- Nsight 原始指标；
- TMA 两级寻址依据；
- 失败的挑战配置。
- 为什么不做 5090 完整性能矩阵；
- CUDA Graph 是否改变小 batch 排名；
- 时钟、独占状态和 run-to-run 噪声。

## 10. 最终检查清单

- [ ] 主实验全部来自 B300；
- [ ] benchmark 范围明确，不把 indexer 混入 decode；
- [ ] 正确性方案在优化前冻结；
- [ ] 随机 seed 固定，所有实现复用同一输入；
- [ ] baseline 正确性异常时已停止性能测试并排查；
- [ ] Triton baseline 已复现；
- [ ] CUTLASS 被强制测到 batch<16；
- [ ] batch=1、4、8、16、32 有统一图表；
- [ ] partial、merge 和 launch gap 已分解；
- [ ] batch=1、4 的 Triton/CUTLASS CUDA Graph replay 已测，或 capture 失败已有记录；
- [ ] run-to-run 变异系数已测，收益超过噪声底线；
- [ ] GPU 独占状态已确认；允许时锁频，否则已记录 clocks/power/temperature；
- [ ] 至少三个关键 case 有 Nsight Compute 数据；
- [ ] 六个讨论点均有“结论 + 证据 + 局限”；
- [ ] 挑战实验有正确性和性能数据；
- [ ] 无正收益时给出定量收益上限；
- [ ] 原始数据、环境和复现命令已保存；
- [ ] 自动生成的环境 manifest 已与 CSV 一起归档；
- [ ] B300 延迟时已按应急预案缩减范围，没有使用 5090 外推 SM100 结论；
- [ ] 答辩不超过 10 分钟。
