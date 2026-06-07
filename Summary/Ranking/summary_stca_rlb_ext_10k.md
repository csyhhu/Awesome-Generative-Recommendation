# Make It Long, Keep It Fast：端到端 10k 超长用户行为建模（Douyin 推荐）

来源：arXiv: [2511.06077](https://arxiv.org/abs/2511.06077)（WWW 2026，ByteDance）

## 研究动机
短视频推荐（如抖音）需要利用用户非常长的行为历史来提升排序质量，但在线系统同时受限于严格的时延与计算/带宽成本。传统做法往往采用两阶段范式：先检索出与目标相关的一小段历史（或截断），再做精排。这样做能省成本，却破坏了“端到端长序列训练/推理”的能力，也会丢失大量可学习信号。

作者目标是：在不破坏在线预算的前提下，把端到端长序列建模扩展到 10k 级别历史长度，并让模型质量随序列长度与模型容量呈现可预测的、类似“缩放定律”的单调提升。

## 核心方法：STCA + RLB + 长度外推训练
作者提出一个端到端长序列推荐框架，由三部分组成：

### 1) STCA：Stacked Target-to-History Cross Attention
关键思想是把注意力结构从“历史-历史自注意力”改为“目标驱动的跨注意力”。具体做法是：每一层都用**目标 item**作为唯一 query，对整个历史作为 key/value 进行 attention（single-query cross attention），从而把与历史长度相关的复杂度从自注意力的二次量级降到线性量级（$O(L)$ vs $O(L^2)$）。

此外，作者利用“单 query”这一结构特性，对计算顺序进行重排，避免显式物化长度为 $L$ 的投影 $XW_K$ / $XW_V$ 等中间张量，进一步降低长度相关的 FLOPs 与显存压力。

STCA 还采用多层堆叠，并用**target-conditioned query fusion**把低层对目标的细粒度信息逐层注入到高层推理中，提升长上下文下的表达能力。

### 2) RLB：Request Level Batching（请求级批处理/复用）
仅靠 STCA 虽然能让“单 target 的序列编码”按 $O(L)$ 缩放，但在线日志里通常同一用户在一次请求/会话中会对应多个候选目标。如果仍然对每个 target 独立编码同一个长历史，会使得**带宽与 I/O/激活复制**成为瓶颈（即便 FLOPs 线性了，系统开销仍会随 $L$ 放大）。

RLB 的系统层做法是：把同一用户的多个 targets 聚成一个用户 micro-batch（大小为 $m$），对用户历史路径（user/history representation）**只计算一次**，然后对所有 targets 复用，并在请求级别聚合梯度。作者同时强调：这种 regrouping 不改变损失定义，因此保持经验风险估计的 unbiasedness（无偏）。

实验给出的系统收益包括：
- 历史长度为 $L=512$：带宽减少 **77%**
- 历史长度为 $L=2k$：带宽减少 **84%**
- 训练吞吐：相对点对点（point-wise）基线，获得 **2.2x** 吞吐提升；配合 kernel 优化最高到 **5.1x**
- 可训练序列最大长度：在相同基础设施下可提升约 **8x**

### 3) 长度外推训练：Train Sparsely / Infer Densely
在线推理希望用 $L_{infer}=10k$，但如果训练也用完整 10k，会导致显存/吞吐随 token 数线性恶化，训练成本难以承受。

作者采用“训练稀疏、推理稠密”的长度外推策略：
- 推理固定使用 $L_{infer}=10k$
- 训练平均长度设为 $L_{train}^{avg}=2k$（外推比约 $\rho_{extra}=5$）
- 训练阶段使用 **Stochastic Length (SL)**：对每个样本随机截断到 $L_{train}\in[L_{min}, L_{max}]$（$L_{max}\le L_{infer}$）
- 长度采样使用 **U-shaped Beta 分布**：混合短窗口与偶发长窗口，让模型获得可外推的训练课程（curriculum）
- 在子序列选取上，经验上保留“最近的时间后缀（temporal suffix）”最有效
- 为处理变长带来的 GPU 负载不均，加入 batch-level load balancing，并用 ragged/索引式实现避免 padding 开销

作者还在结论中给出一个关键观点：在较小 $\\alpha$ 的 Beta 形状与约 20% sequence sparsity 的设置下，能实现约 10k 窗口增益的 ~80%，但训练成本只有全量训练的约三分之一量级。

## 实验结论
### 离线（Douyin offline）：在移除 TWIN(10k) 的情况下仍获最大收益
离线实验对比多个序列编码器（single-layer target attention、DIN、Transformer、自注意力体系、HSTU 等）。为了让对比更保守，基线都额外加入检索式块 TWIN(10k)；而作者的方法则**移除** TWIN(10k)，完全依赖端到端长序列建模（STCA + RLB + Ext）。

在 matched compute 设置下，作者方法对三个目标（finish/skip/head）的改进幅度为：
- Finish：$\Delta$AUC **+0.49**，$\Delta$NLL **-1.16**
- Skip：$\Delta$AUC **+0.71**，$\Delta$NLL **-1.14**
- Head：$\Delta$AUC **+0.39**，$\Delta$NLL **-1.41**

同时，作者报告与设计一致的归因：STCA 在全历史上做“精确 softmax attention 且 $O(L)$ 代价”，RLB 复用了用户侧长历史编码，外推训练提供了对长上下文的校准尾部（calibrated tail）。

### 在线（Douyin & Douyin Lite）：一月部署，跨分层稳定增益
作者把 STCA + RLB + 外推训练部署到抖音与抖音 Lite，并用单 query 的 target->history encoder 替换原有带 TWIN(10k) 的检索式特征（其它模块保持不变），观察一月 A/B：

以“所有用户（All Users）”为例：
- Douyin：
  - 30-day Activeness：**+0.1161%**
  - App Stay Time：**+0.9266%**
  - Finish：**+3.3454%**
  - Comment：**+1.5678%**
  - Like：**+1.8282%**
- Douyin Lite：
  - 30-day Activeness：**+0.1281%**
  - App Stay Time：**+0.8467%**
  - Finish：**+4.2275%**
  - Comment：**+2.6167%**
  - Like：**+2.3828%**

作者同时给出内部成本核算：移除 TWIN 会带来 GPU 成本增加（约 **+33%**）但 CPU 成本降低（约 **-16%**），综合约 **+17%**，相对线上收益可接受。

## 可复用的工程启示
1. 让注意力“以目标为中心”可以显著降低历史长度维度的复杂度，同时仍能端到端反传全历史。
2. 系统层复用（RLB）能把“线性 FLOPs 优化”转化为真实的带宽/IO/吞吐收益，避免 $L$ 放大时成本转移到数据与激活层。
3. 用训练阶段的随机长度与校准尾部，实现长上下文外推，是在预算约束下把 10k 能力落地到生产的关键。

## 结论摘要
这篇工作给出了一套面向生产的端到端长序列推荐配方：STCA 把长历史建模从二次复杂度压到线性复杂度，RLB 在请求级别复用用户侧长历史编码降低系统瓶颈，长度外推训练则用“训练稀疏/推理稠密”在不显著增加训练成本的情况下获得 10k 级别效果。离线与在线实验都显示稳定、可预测的收益。

