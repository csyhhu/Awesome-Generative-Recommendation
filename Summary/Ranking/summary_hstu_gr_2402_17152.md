# Actions Speak Louder than Words：面向生成式推荐的万亿参数顺序转导器（HSTU）

- 论文：["Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations"](https://arxiv.org/abs/2402.17152)
- 代码（论文给出的实现）：`[facebookresearch/generative-recommenders](https://github.com/facebookresearch/generative-recommenders)`

## 1. 论文想解决什么问题

工业级推荐（DLRM 体系）常见特征：**特征极其异构且高基数（亿/十亿级 ID、交叉特征、计数/比率等）**、数据流式更新、每天处理数十亿到数百亿用户行为。但作者指出：尽管数据规模巨大、特征工程复杂，**主流 DLRM 在工业界往往"算力扩展不灵"**（加大算力/参数并不稳定带来收益）。

作者借鉴 Transformer 在语言/视觉的成功，尝试把推荐问题"重新表述（reformulate）"为更像大模型可扩展的形式：**把用户行为视为一种新"模态"，把推荐任务做成"序列转导（sequential transduction）+ 生成式训练"的问题**，并围绕这一表述设计可在长序列、动态大词表下高效训练/推理的新架构与系统方法。

## 2. 核心范式：从 DLRM 到 Generative Recommenders（GRs）

### 2.1 统一异构特征：把"特征工程"压进序列

作者的做法是把 DLRM 的稀疏/类别特征尽可能**顺序化（sequentialize）**，并把它们合并进一个统一时间序列：

- **稀疏/类别特征**：选择最长的主时间序列（通常是用户与内容交互得到的内容序列）作为骨架；其他变化慢的时间序列（例如人口属性、关注列表等）先压缩（只保留分段最早项）再合并进主序列，从而不会显著拉长序列。
- **稠密/数值特征**：很多数值特征本质上是对一些类别维度（topic、location 等）做聚合的统计量。作者观点是：当这些类别维度被顺序化编码后，配合**足够表达力的序列转导模型 + 目标感知（target-aware）**，可以逐步减少甚至移除大量显式数值特征（用"算力/序列长度"换"特征工程"）。

### 2.2 把排序与召回都写成"序列转导任务"

作者把推荐系统的两个核心子任务都表述为"给定输入 token 序列，输出 token（或输出为空）的转导"：

- **排序（ranking）**：通过把"内容 \Phi_i"与"用户动作 a_i"交错排列（interleave），让模型在自回归（causal）下对 a_i 做预测，从而天然地支持目标感知（候选 item 作为 token 进入上下文，"交互"可以发生在编码阶段而不是最后 softmax）。
- **召回（retrieval）**：学习 p(\Phi_{i+1}\mid u_i) 或等价的下一内容预测（结合正/负反馈定义），以适配大规模候选集与工业训练方式。

### 2.3 训练方式的关键改变：从"曝光样本"到"生成式训练"

工业 DLRM 往往以（user/request，candidate，label）三元组为单位做流式训练；若直接做长序列自注意力会导致训练复杂度过高。作者提出用**生成式训练**摊销 encoder 成本：以较低频率（近似 1/n_i）为一个用户序列"发射"训练样本，使得整体训练复杂度从直觉上的 O(N^3)（按用户 token 数累计）降低一个 O(N) 因子到更可承受的规模。

## 3. 关键架构：HSTU（Hierarchical Sequential Transduction Unit）

作者认为标准 Transformer（特别是 softmax attention）在推荐的"非平稳、动态大词表、强度信号重要、长序列稀疏"场景下不理想，因此提出 HSTU。它可以理解为：**面向推荐重新设计的"自注意力编码器块"**，并在工程上对长序列显著提速。

### 3.1 点式聚合注意力（Pointwise aggregated attention）

HSTU 把 softmax 归一化注意力替换为**点式（pointwise）激活/归一化的注意力**（文中用 SiLU 等非线性 + LayerNorm 稳定训练）。动机包括：

- 推荐里"出现次数/强度"本身是强特征；softmax 的归一化会抹平强度信息，不利于建模"喜欢程度强弱"等。
- 流式动态词表下，softmax 的一些性质在实践中更容易带来不稳定/效果损失。

作者用合成数据（Dirichlet Process 模拟"rich get richer"的流式行为）展示 softmax 注意力与其方案存在显著差距（文中举到最高 44.7% 的 HR 差距）。

### 3.2 门控（U）与"用一个块替代 DLRM 的多种模块"

HSTU 每层包含：

- 投影得到 Q,K,V,U（其中 U 用于门控）
- 注意力聚合得到 A(X)V(X)
- 再通过 \text{Norm}(A(X)V(X)) \odot U(X) 做"交互 + 条件计算"的效果

作者将其类比为同时覆盖 DLRM 的三类核心计算：**特征提取（池化/目标感知注意力）、特征交互（类似 FM/DCN 的交互效果）、表示变换（类似 MoE gating 的条件计算）**，从而用一个统一的顺序转导块替代大量异构手工组件。

#### 3.2.1 深入理解门控 U

##### 数学形式

HSTU 一层的完整计算流程如下（对应论文 Sec. 3, Eq. 1–3）：

**Step 1 — 联合投影：**

\[
U(X), V(X), Q(X), K(X) = \operatorname{Split}\bigl( \phi_1 (f_1(X)) \bigr)
\]

其中 \(f_1(\cdot)\) 是可学习的线性变换（或小型 MLP），\(\phi_1(\cdot)\) 为逐点激活函数（通常为 SiLU），\(\operatorname{Split}\) 沿最后一维将激活后的张量切为四块，分别得到门控张量 \(U\)、值 \(V\)、查询 \(Q\)、键 \(K\)。

**Step 2 — 点式聚合注意力：**

\[
A(X)V(X) = \phi_2\bigl( Q(X) K(X)^{\mathsf{T}} + \mathrm{rab}^{p,t} \bigr)\, V(X)
\]

其中 \(\mathrm{rab}^{p,t}\) 为融合位置与时间差（分桶）的相对注意力偏置，\(\phi_2(\cdot)\) 为逐点激活（替代 softmax），不对整行做归一化，从而保留"同一信号出现多次，聚合值越大"的强度信息。

**Step 3 — 门控与输出：**

\[
Y(X) = f_2\Bigl( \operatorname{Norm}\bigl( A(X)V(X) \bigr) \odot U(X) \Bigr)
\]

\(\operatorname{Norm}\) 为 LayerNorm，\(\odot\) 为逐元素（Hadamard）乘法，\(f_2(\cdot)\) 为输出线性/MLP 变换。**这里没有独立的 FFN 子层。**

##### 门控 U 的核心作用与动机

**① 替代传统 FFN，大幅降低激活内存**

传统 Transformer 每个 block 包含 Self-Attention + FFN（两层线性：升维 4d → 激活 → 降维 d）。FFN 的中间隐藏维度（通常是嵌入维度的 4 倍）是激活内存的主要来源。HSTU 的做法是：

- 将 Q、K、V、U 合并为一个联合投影，取代分别投影，再通过一次轻量输出线性 \(f_2\) 完成整个 block；
- 每层线性层数从约 **6 个**（Q、K、V 投影 × 3 + 输出投影 × 1 + FFN × 2）降至约 **2 个**（联合投影 + 输出投影）；
- 激活内存（bfloat16 下）从约 **33d 降至 14d**，使得相同硬件可训练约 **2 倍更深**的模型。

这是 HSTU 能扩展到万亿参数规模的关键工程条件。

**② 门控即"条件计算"：选择性地放大/抑制注意力信号**

\(U(X)\) 与 \(\operatorname{Norm}(A(X)V(X))\) 做逐元素乘法，本质上是一个**信息门控**：
- 若 \(U\) 的某维度值较大（经 SiLU 激活后为正），该维度的注意力信号被**放大**（"门开"）；
- 若接近零或为负，则被**抑制或屏蔽**（"门关"）。

这与 GLU（Gated Linear Unit）的思路一致：门控机制让模型学会"根据上下文选择性输出"——给定当前位置的表示，模型通过 \(U\) 自主决定注意力聚合结果中哪些维度值得前传、哪些可以忽略。这比 FFN 对所有位置的统一非线性变换更加灵活且参数更高效。

**③ 与点式注意力的协同：天然实现特征交互**

关键洞察：\(\operatorname{Norm}(A(X)V(X)) \odot U(X)\) 这个**逐元素乘法**（哈达玛积）本身就蕴含了类似于 FM（Factorization Machine）或 DCN 的**二阶特征交叉**效果。门控向量 \(U\) 由输入 \(X\) 本身计算得到（与 Q、K、V 同源），因此门控乘法实际上是**让注意力聚合结果与输入自身进行交互**——这等价于在序列的每个位置上做了一次轻量化的 self-cross，无需额外引入 DLRM 中显式的特征交互模块（如 FM、DCN-v2、Deep Cross 等）。

**④ SwiGLU、U Projection 与 DCN 三者的对比：同一公式骨架的三个变体**

SwiGLU、HSTU 的 U Projection、DCN Cross Layer，三者都可以归入 **GLU（Gated Linear Unit）范式**——核心都是 `gate ⊙ value → output` 的骨架，差异在于 gate/value 的来源和类型。

**公式并排对照：**

| 方法 | 公式 | 骨架展开 |
|---|---|---|
| **SwiGLU** | \((xW_1 \odot \text{SiLU}(xW_2)) W_3\) | \(\text{SiLU}(xW_2) \;\odot\; xW_1 \;\cdot\; W_3\) |
| **DCN Cross** | \(x_0 \odot (W_l x_l) + b_l + x_l\) | \(1 \;\odot\; (W_l x_l \text{ 即标量}) \cdot x_0 \;+\; \text{残差}\) |
| **DCN-V2 Cross** | \(x_0 \odot (W_l x_l) + b_l + x_l\) | \(1 \;\odot\; (W_l x_l \text{ 矩阵版}) \cdot x_0 \;+\; \text{残差}\) |
| **HSTU U-Proj** | \(f_2(\text{Norm}(\text{Attn}(\cdot)) \odot \text{SiLU}(XW_u))\) | \(\text{SiLU}(XW_u) \;\odot\; \text{Norm}(\text{Attn}(\cdot)) \;\cdot\; f_2\) |

**统一骨架 `gate ⊙ value → output`：**

```
               ┌──── gate ────┐   ┌─────── value ───────┐   ┌─ out ─┐   ┌ 残差 ┐
SwiGLU(x)   =  SiLU(x·W₂)     ⊙       x·W₁                 ·   W₃
DCN(x)      =        1        ⊙   (w^T x_l) · x₀          +    b  +   x_l
DCN-V2(x)   =        1        ⊙   (W_l x_l)  · x₀          +    b  +   x_l
U-Proj(X)   =  SiLU(X·W_u)    ⊙  Norm( Attn(·) )          ·   f₂
```

**三维对比：**

| 维度 | SwiGLU | DCN / DCN-V2 | HSTU U-Proj |
|---|---|---|---|
| **gate 来源** | 独立权重 \(W_2\)，逐 token 本地 | **无 gate**（或视为 gate=1），乘法另一端由 \(x_0\) 充当 | 与 QKV 共享底层投影的 \(W_u\)，逐 token 本地 |
| **value 支路** | \(xW_1\)：本地线性投影 | DCN：标量 \((w^T x_l)\)；DCN-V2：矩阵 \((W_l x_l)\)，均与 \(x_0\) 交互 | **全序列 self-attention** 聚合的上下文表示 |
| **输出投影** | \(W_3\) | 无（直接 \(+b+x_l\) 残差） | \(f_2\) |
| **跨 token 交互** | ❌ 仅本地 | ❌ 仅本地（纯特征交叉，不涉及时序） | ✅ **通过 Attention 聚合全序列** |
| **gate 的角色** | 控制哪些特征维度被放大 | 无显式 gate，\(x_0\) 决定交叉方向 | 基于本地表示，选择性保留/过滤全局上下文 |
| **参数量 / 层** | \(3d^2\)（W₁,W₂,W₃） | DCN: \(2d\)；DCN-V2: \(d^2 + 2d\) | 约 \(2d^2\)（联合投影 + f₂） |
| **本质语义** | 非线性门控 FFN | 显式多项式特征交叉 | **上下文感知的门控交叉** |

**三者的设计逻辑差异：**

- **SwiGLU**：关注"这个 token 的哪些特征重要"——纯粹的 per-token 门控激活，目标是替代 ReLU/GeLU 提升 FFN 表达力。
- **DCN**：关注"这些特征之间如何交叉"——显式构造多项式交叉项，目标是高效建模特征交互（尤其推荐场景）。
- **HSTU U-Proj**：同时做到上述两者——`U` 是 per-token 门控（类似 SwiGLU），哈达玛积产生交叉（类似 DCN），但交叉的对象是**注意力聚合后的全局序列上下文**而非本地特征，因此比前两者多了一层"时序/上下文感知"。

**⑤ 类比 DLRM 三大功能的统一解释**

| DLRM 组件 | 传统实现 | HSTU 中的对应 |
|---|---|---|
| **特征提取** | 各类池化/Embedding Lookup + MLP | 点式注意力（A(X)V(X)）：对序列上下文做自适应加权池化 |
| **特征交互** | FM / DCN-v2 / Deep Cross 等显式交叉层 | 门控哈达玛积 \(\text{Norm}(A(X)V(X)) \odot U(X)\)：乘法即交叉 |
| **表示变换** | MoE gating / 多层 FFN | \(U\) 本身就是门控（类似 MoE 的 router），决定哪些信号被增强/抑制 |

##### 一句话总结

**门控 U 不是简单地在 QKV 上多加一个投影，而是 HSTU 架构的核心设计支点：它用一个轻量的门控乘法同时替代了 FFN（降内存）和显式特征交互模块（统一异构组件），并通过与点式注意力的协同，让推荐模型在长序列、动态大词表、强度信号敏感的工业场景下实现了可扩展的 scaling law。**

### 3.3 面向长序列的工程优化：稀疏/不规则（ragged）注意力核 + 相对偏置融合

长序列推荐数据的一个关键性质是：**序列长度分布偏斜、长序列样本天然稀疏**。作者利用这一点把注意力计算重写成不同形状的 grouped GEMM，并实现融合 kernel（包括相对注意力偏置 rab 的融合），使得 HSTU 在 GPU 上更接近 memory-bound 的高吞吐。

### 3.4 Stochastic Length（SL）：进一步利用稀疏减少训练成本

在训练时对序列做随机长度/稀疏化（由参数 \alpha 控制），大量场景可以移除 70%~80% token 但指标下降很小（文中示例：NE 变化不超过约 0.002 的量级），从而显著降低训练代价，并优于一些既有的长度外推/稀疏化方案。

## 4. 端到端推理：M-FALCON（关键在"目标感知下也能批量化"）

排序阶段往往需要对成千上万候选打分。常见直觉是：目标感知（target-aware）意味着要"一个候选一次前向"，代价 O(m n^2 d)。作者提出 **M-FALCON**，核心是：

- **改 attention mask**：把 b_m 个候选一起拼到序列末端，同时禁止候选之间相互注意力（mask 掉），使得每个候选的输出只依赖历史而不依赖其他候选。这样一次前向就能得到 b_m 个候选的打分。
- **微批（microbatching）**：把 m 个候选分成 \lceil m/b_m\rceil 个 microbatch，扩展到上万候选。
- **KV 缓存**：历史部分的 K,V 可跨 microbatch、甚至跨请求复用，只需为末端候选重新计算必要张量，从而把大量重复算子摊薄。

论文给出一个关键对比：即使 GR 模型 FLOPs 复杂度远高于 DLRM（文中举例达到 **285×**），通过 HSTU + M-FALCON，端到端吞吐仍可做到 **1.50×~2.99× 更高 QPS**（在 1024/16384 候选打分的设置下）。

## 5. 实验与结论要点（按论文叙述口径）

- **公开数据集（传统全量 shuffle、多 epoch 的序列推荐设置）**：HSTU 在 MovieLens、Amazon Books 等上相对 SASRec 有明显 HR/NDCG 提升，且扩大模型后提升更显著（论文表格报告最高到 **NDCG +65.8%** 的量级）。
- **工业级流式设置（100B 级训练样本规模的对比实验口径）**：HSTU 在检索（log perplexity）与排序（NE）上优于 Transformer/改进版 Transformer++，并带来更好的稳定性与更低显存占用。
- **长序列效率**：在 8192 序列长度上，HSTU 相比基于 FlashAttention2 的 Transformer 报告 **5.3×~15.2×** 加速。
- **线上 A/B**：作者报告 HSTU-based GR 在多个产品面部署，线上指标提升 **12.4%**，并达到 **1.5T 参数**规模。
- **可扩展性/Scaling law**：作者声称 GR 的质量随训练算力呈幂律（power-law）增长，跨约三个数量级，最高到接近 GPT-3/LLaMa-2 训练算力量级（按其"流式训练按年归一化"的口径），而传统 DLRM 更易出现平台期。

## 6. 我认为最值得记住的启示（偏"方法论"）

- **把推荐重新"语言化"不是关键，关键是把工业推荐的异构特征与目标感知需求，重写成"可扩展的序列转导问题"**：交错 token（内容/动作）+ generative training + 统一序列特征空间，形成端到端可扩展训练范式。
- **架构与系统共同决定能否扩展**：HSTU（注意力形式/门控/核优化）解决"能训练长序列"，M-FALCON 解决"能在目标感知下做大候选高吞吐推理"，两者一起才让"万亿参数推荐模型"在工业推理预算下成立。
- **"减少特征工程、增加算力与序列长度"是一条路线**：作者的叙事本质是把大量手工特征（尤其是密集统计特征）交给更强的序列模型去"涌现式"学习，并希望像 LLM 一样得到更稳定的 scaling 行为。

