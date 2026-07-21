# SlimPer: Make Personalization Model Slim and Smart

## 用户综合理解

> 本文的输入是用户序列+稀疏密集特征+候选item组成的序列，为避免HSTU中每一层都要进行一次full attention. 本文给每个候选Item生成一个固定维度的表达，执行多层target attention. 这个target attention在文中被称为SlimPer, 其实是一系列query attention. 最终使用这个表达用于概率预估。

### 理解分析

**整体方向正确**，但有以下细节需要澄清：

| 用户理解 | 是否正确 | 澄清说明 |
|----------|----------|----------|
| 输入是用户序列+稀疏密集特征+候选item | ✅ 正确 | 输入确实包含这三类特征 |
| 组成的序列 | ⚠️ 不完全准确 | 特征不是"组成一个序列"，而是分别处理为不同的token类型 |
| 避免HSTU中每一层都要进行一次full attention | ✅ 正确 | 这是SlimPer的核心动机之一 |
| 给每个候选item生成一个固定维度的表达 | ✅ 正确 | 每个item有一个知识库$\mathcal{X}$（64×256） |
| 执行多层target attention | ⚠️ 不完全准确 | SlimPer的attention与传统target attention（如DIN）有本质区别：SlimPer是迭代精化的三步循环（Select-Match-Refine），而非一次性attention聚合 |
| 这个target attention在文中被称为SlimPer | ❌ 不准确 | SlimPer是完整架构名称，不是一种target attention |
| 其实是一系列query attention | ✅ 正确 | 每一层的Select步骤从知识库生成query向量，对用户侧tokens执行attention |
| 最终使用这个表达用于概率预估 | ✅ 正确 | 最终知识库$\mathcal{X}^L$用于生成任务logits和概率 |

### 修正后的理解

> SlimPer 的输入是用户序列、稀疏特征、密集特征和候选item。为避免 HSTU 中每一层都要执行 full self-attention（$\mathcal{O}(N^2)$），本文给每个候选 item 生成一个固定维度的知识库表达（64×256），执行多层的 Select-Match-Refine 迭代精化。每一层的 Select 步骤使用从知识库派生的 query 向量对用户侧 tokens 执行 attention，这类似于一系列 query-based attention。最终使用精化后的知识库表达用于概率预估。

---

## 论文信息

- **标题**: SlimPer: Make Personalization Model Slim and Smart
- **作者**: Siqi Wang, Xianjie Chen 等 (Meta Platforms, Inc.)
- **arXiv**: https://arxiv.org/abs/2607.12281

## 核心观点

Transformer 架构在工业推荐系统中越来越流行，但存在一个根本的设计不匹配：LLM 需要逐 token 的自回归预测，因此需要维护与序列长度成正比的大型中间张量；而推荐系统只需要为每个 `<user, item>` 对产生单一的相关性分数，不需要 token 级别的监督。

SlimPer 提出了一个全新的范式：将个性化排序重新表述为**对紧凑统一的 `<user, item>` 知识库的迭代精化**。

## 设计原则

1. **Slim**：固定大小的知识库（$K \times d$），每层成本 $\mathcal{O}(N)$，与输入序列长度解耦
2. **Complete Access**：每一层都直接查询完整的原始用户侧 tokens，不丢失信息
3. **ROO-Aware**：用户侧 tokens 只计算一次，在同一次请求的所有候选 item 间共享

## 架构详解

### SlimPer 层：Select-Match-Refine 循环

每层执行三个步骤：

#### Step 1: Token Selection（基于全模态注意力的 token 选择）

- 查询向量 $\mathbf{Q}$ 从当前知识库 $\mathcal{X}^k$ 派生：$\mathbf{Q} = \mathcal{L}(\mathcal{X}^k) \in \mathbb{R}^{q \times d}$
- 用户侧所有输入 tokens 作为 keys 和 values
- 稀疏特征使用 MLP 作为注意力核（参数化），序列特征使用标准缩放点积注意力（非参数化）
- 得到两个注意力输出：$\mathbf{R}_s$（稀疏特征）和 $\mathbf{R}_e$（序列特征）

#### Step 2: Explicit Multifaceted Dot-Product Relevance Matching（显式多面点积相关性匹配）

- 从知识库派生多面模板 $\mathbf{T} = \mathcal{L}(\mathcal{X}^k) \in \mathbb{R}^{t \times d}$
- 计算匹配分数：$\boldsymbol{\lambda}_s = \mathrm{DotProduct}(\mathbf{R}_s, \mathbf{T})$，$\boldsymbol{\lambda}_e = \mathrm{DotProduct}(\mathbf{R}_e, \mathbf{T})$

#### Step 3: Refine the Knowledge Base（精化知识库）

- 将相关性分数归一化后，通过 MLP 更新知识库：

$$
\mathcal{X}^{k+1} = \mathcal{X}^{k} + \mathrm{MLP}_{\mu}\Big(\mathrm{Concat}\big(\mathrm{RMSNorm}(\boldsymbol{\lambda}_{s}), \, \mathrm{RMSNorm}(\boldsymbol{\lambda}_{e}), \mathcal{L}(\mathcal{X}^{k}), \, \mathbf{D}\big)\Big)
$$

- 密集特征 $\mathbf{D}$ 在更新时直接拼接

### 多模态 Tokenization

| 模态 | 输入类型 | Tokenization 方法 |
|------|----------|-------------------|
| 稀疏 | 高基数类别 ID | 嵌入池化（Embedding Pooling） |
| 序列 | 用户交互记录 | 事件建模（Event Modeling） |
| 密集 | 连续数值 | MLP 投影（Dense Processing） |

**上下文编码**：每个事件 token 额外编码其邻域信息 $\mathbf{e}_{\text{context}}^{(t)} = \Psi_{\text{context}}(x_{t-w+1}, \ldots, x_{t})$，捕捉局部行为连续性。

### ROO（Request-Only Optimization）

用户侧特征与非用户侧特征显式分离，用户侧 tokens 在同一次请求的所有候选 item 间共享，大幅减少内存和计算开销。

## 复杂度分析

| 模型 | 每层计算复杂度 | 每层内存复杂度 | 用户-Item 交互容量 |
|------|----------------|----------------|--------------------|
| DLRM | $\mathcal{O}(L \cdot N \cdot 16)$ | $\mathcal{O}(L \cdot 64)$ | $\mathcal{O}(L \cdot N \cdot 64)$ |
| ROO-aware Transformer (HSTU) | $\mathcal{O}(L \cdot (N+1)^2 / B)$ | $\mathcal{O}(L \cdot N / B)$ | $\mathcal{O}(L \cdot N)$ |
| Non-ROO-aware Transformer | $\mathcal{O}(L \cdot N^2)$ | $\mathcal{O}(L \cdot N)$ | $\mathcal{O}(L \cdot N)$ |
| **SlimPer** | $\mathcal{O}(L \cdot N \cdot q)$ | $\mathcal{O}(L \cdot K)$ | $\mathcal{O}(L \cdot N \cdot K)$ |

SlimPer 的用户-Item 交互容量是 Transformer 类模型的 $K$ 倍（$K=64$），因为它将所有交互能力都用于知识 base-token 相关性匹配，而不是用户侧 token 之间的交互。

## 信息论分析

1. **瓶颈是容量充足的**：推荐任务只需要 $\tau$ 个标量预测，远低于 $K \times d$ 知识库的表示能力
2. **迭代访问防止不可逆损失**：每一层都直接 cross-attend 到原始输入 tokens，而不是仅依赖前一层的压缩表示，打破了 Data Processing Inequality 的限制

## 实验结果

### 离线性能（Instagram Reels 和 Feed）

| 表面 | 配置 | NE 改善 | QPS 变化 | 内存变化 |
|------|------|---------|----------|----------|
| Reels | 2k events | -0.51% | +11.0% | -9.32% |
| Reels | 5k events | -0.80% | -10.0% | +4.59% |
| Feed | 1k events | -0.49% | +12.5% | -18.12% |
| Feed | 4k events | -0.94% | -16.07% | +2.04% |

### A/B 测试

- 在 Instagram Reels 和 Feed 全量上线
- 多个主要参与度指标获得统计显著提升
- 总体影响约为典型显著发布的 10 倍
- 生态保护指标无回归
- GPU 资源保持中性

### FLOPs 对比

| 序列长度 | Baseline (HSTU) | SlimPer | 比率 |
|----------|-----------------|---------|------|
| 2048 | ~24 GFLOPs | ~3 GFLOPs | ~8× |
| 4096 | ~74 GFLOPs | ~5 GFLOPs | ~16× |
| 6144 | ~150 GFLOPs | ~6 GFLOPs | ~25× |

### 消融实验

- **知识库大小 K**：K=64 提供最佳质量-效率平衡；K=4 时 QPS +23.2% 但 NE 退化 +1.2%
- **层数 L**：7 层表现优秀；9 层继续获得 -0.12% NE 提升但成本增加
- **上下文编码**：引入上下文编码带来约 -0.15% NE 提升，事件类型比内容更重要

## 可解释性

SlimPer 的注意力机制提供了内在可解释性：

- **低层**：关注整个历史的广泛上下文（前 4k 个事件）
- **中层**：聚焦于较近期的事件（前几百个）
- **高层**：进一步收窄到最近的 ~400 个事件

这种层次化注意力行为符合迭代精化设计：早期层收集广泛证据，后期层使用最新、最具影响力的行为信号精化预测。

## 关键贡献

1. **新范式**：将判别式推荐表述为对紧凑 `<user, item>` 知识库的迭代精化，深度与输入序列长度解耦
2. **统一多模态架构**：使用标准构建块（MLP、QKV attention、线性投影、RMSNorm）统一处理稀疏、密集和序列特征
3. **ROO 感知**：扩展 ROO 到所有用户侧特征模态，减少冗余计算
4. **内在可解释性**：每一层的注意力权重直接关联推荐结果与特定用户交互历史

## 超参数配置

| 超参数 | Feed | Reels |
|--------|------|-------|
| 层数 L | 5 | 7 |
| 知识库大小 K | 64 | 64 |
| 隐藏维度 d | 256 | 256 |
| 查询槽 q | 16 | 16 |
| 模板槽 t | 32 | 32 |

## 与现有工作的关系

- **Perceiver/Perceiver IO**：类似的瓶颈架构，但缺少推荐特定的归纳偏置
- **HSTU**：仅在用户历史建模中使用 ROO，SlimPer 扩展到所有用户侧特征
- **DLRM/Wukong**：固定大小特征的交互学习，SlimPer 提供了对原始历史事件的直接访问

## 未来方向

1. 扩展到无固有顺序的证据源（如跨表面参与信号）
2. 在早期排序阶段部署，利用更大候选集的摊销计算优势
3. 探索更具表达力的匹配和精化算子

---

## 讨论问答

### Q1：本文是否解释了 Transformer 架构在工业推荐系统的使用方式？改进的基础框架是什么？

**A：**

是的，本文详细解释了 Transformer 架构在工业推荐系统中的应用现状：

**Transformer 在推荐系统中的使用方式**：
1. **序列特征建模**：如 HSTU、Interformer，用于处理变长的用户交互历史序列
2. **多模态统一建模**：如 OneTrans、HHFT、RankMixer，用于统一处理不同特征模态

**为什么 LLM 需要逐 token 的自回归预测**：
- LLM 的预训练目标是预测下一个 token（Next Token Prediction）
- 为了实现这个目标，模型需要为每个位置维护完整的隐藏状态表示
- 中间张量大小与序列长度 $N$ 成正比：$\mathcal{O}(N \times d)$
- 计算成本是二次的：$\mathcal{O}(N^2)$（自注意力）

**推荐系统不需要这个设计**：
推荐系统是判别式任务，只需要输出一个标量相关性分数，不需要为每个输入 token 预测下一个 token。这就是论文所说的 "design premise misaligned with the task"（设计前提与任务不匹配）。

**本文改进的基础框架**：
论文改进的是 **HSTU + Wukong 的混合架构**：
- HSTU：处理变长用户交互历史序列（使用 ROO）
- Wukong：处理固定大小的稀疏/密集特征

这是 Instagram 当前生产环境使用的最终阶段排序模型。

---

### Q2：用例子说明本文的创新方法（训练和推理角度）

**场景设定**：用户 Alice 在 Instagram 上有：
- 稀疏特征：喜欢的类别（摄影、旅行、美食），关注的创作者 ID
- 序列特征：最近 2000 条交互事件（点赞、观看、分享、跳过等）
- 密集特征：当前时间、设备类型、历史 CTR 等
- 候选 Item：100 个待排序的帖子

**基线方法（HSTU + Wukong）**：

> 注意：HSTU 已经是 ROO-aware 的，用户历史序列 tokens 在同一次请求的所有候选 item 间共享。

训练阶段：
1. **用户历史建模（HSTU）**：对 2000 个事件 tokens 执行自注意力，每层 $\mathcal{O}(2000^2)$ 计算，维护 2000 × d 的中间张量，最终输出一个聚合的用户表示
2. **固定特征建模（Wukong）**：对稀疏特征（3 tokens）和密集特征进行交互学习
3. **Late Fusion**：将 HSTU 的输出和 Wukong 的输出拼接，通过 MLP 得到最终预测分数

推理阶段：
1. 用户历史 tokens 只计算一次，在 100 个候选 item 间共享（ROO）
2. 但每层仍需维护 2000 × d 的中间张量（HSTU）
3. 稀疏特征和密集特征的处理未完全 ROO-aware

**SlimPer 创新方法**：

> **关键澄清**：
> - 知识库 $\mathcal{X}$ 是**每个 `<user, item>` 对一个**，不是每个用户一个！它从 item 侧特征初始化
> - 🟢 用户侧 tokens（2003 个）在整个 7 层过程中**保持不变**，只有🔴知识库被迭代更新
> - 与 target attention（如 DIN）的区别：DIN 是一次性 attention 聚合，SlimPer 是多次迭代精化
> - 🟢 表示所有层中保持不变的变量；🔴 表示每一层都会变化的变量

训练阶段：
1. **Tokenization**：🟢 稀疏特征 → 3 个 tokens；🟢 序列特征 → 2000 个事件 tokens；🟢 密集特征 → 1 个向量（用户侧 tokens 计算一次，不再更新）
2. **初始化知识库**：对每个候选 item，从 item 侧特征初始化 🔴 $\mathcal{X}^0 \in \mathbb{R}^{64 \times 256}$（固定大小！）
3. **第 1~7 层 Select-Match-Refine**：每一层都直接访问 🟢 原始的 2003 个用户侧 tokens，迭代更新 🔴 知识库（始终保持 64 × 256）
4. **输出预测**：从 🔴 最终知识库生成任务 logits

推理阶段（三重优势体现）：

**优势 1：扩展 ROO 到所有用户侧特征**
- 用户历史 tokens（2000 个）只计算一次 ✓（HSTU 也做到了）
- 用户稀疏特征 tokens（3 个）只计算一次 ✓（HSTU + Wukong 未完全做到）
- 所有用户侧 tokens 在 100 个候选 item 间共享

**优势 2：固定大小知识库替代 N 大小中间张量**
- HSTU：每层维护 2000 × 256 的中间张量 → 7 层共 7 × 2000 × 256 = 3.5M 参数
- SlimPer：每层维护 64 × 256 的知识库 → 7 层共 7 × 64 × 256 = 114K 参数（约 30× 更少）

**优势 3：重定向交互容量到用户-Item 相关性匹配**
- HSTU：$\mathcal{O}(2000^2)$ 计算主要用于用户侧 token 之间的交互（自注意力）
- SlimPer：$\mathcal{O}(64 \times 2003)$ 计算全部用于知识库与用户侧 tokens 的相关性匹配

**与 Target Attention（DIN）的本质区别**：

| 维度 | Target Attention (DIN) | SlimPer |
|------|----------------------|---------|
| 核心对象 | 用 item 作为 query，attention 聚合用户历史 | 维护固定大小知识库，迭代精化 |
| 迭代次数 | 单次 attention 聚合 | 多次（5-10 层）迭代精化 |
| 信息访问 | 只能访问一次用户历史 | 每一层都直接访问原始用户侧 tokens |
| 模态支持 | 主要用于序列特征 | 统一处理稀疏、序列、密集三种模态 |
| 交互类型 | 仅注意力加权聚合 | Select → Match → Refine 三步循环 |

**复杂度对比**：
- HSTU：$\mathcal{O}(7 \times 2000^2 / B) = \mathcal{O}(28M / B)$（B 为批大小）
- SlimPer：$\mathcal{O}(7 \times 2003 \times 16) = \mathcal{O}(224K)$（约 125× 更少，当 B=5 时）

> 注：上述 SlimPer 复杂度主要考虑 Select 步骤（$\mathcal{O}(q \times N)$），Match 步骤（$\mathcal{O}(q \times t)$，$t=32$）相对可忽略。完整每层复杂度为 $\mathcal{O}(q \times N + q \times t)$。

---

### Q3：Select-Match-Refine 的输入输出是什么？

> **颜色标记说明**：🟢 表示在所有层中保持不变的变量；🔴 表示每一层都会变化的变量

**整体输入输出**：

层输入：
- 🔴 上一层的知识库 $\mathcal{X}^k \in \mathbb{R}^{K \times d}$（$K=64$, $d=256$）
- 🟢 用户稀疏特征 tokens $\mathbf{S} \in \mathbb{R}^{|S| \times d}$（所有层共享，不更新）
- 🟢 用户序列特征 tokens $\mathbf{E} \in \mathbb{R}^{N \times d}$（所有层共享，不更新）
- 🟢 密集特征向量 $\mathbf{D} \in \mathbb{R}^{D}$（所有层共享，不更新）

层输出：
- 🔴 更新后的知识库 $\mathcal{X}^{k+1} \in \mathbb{R}^{K \times d}$
- 🔴 可选：当前层的任务嵌入 $\mathbf{P}_p^k$（用于多任务预测）

---

**Step 1: Select（Token Selection）**

输入：
- 🔴 查询来源：$\mathcal{X}^k \in \mathbb{R}^{64 \times 256}$（当前知识库，每一层变化）
- 🟢 Keys/Values：$\mathbf{S} \in \mathbb{R}^{|S| \times 256}$（稀疏特征，所有层不变）和 $\mathbf{E} \in \mathbb{R}^{2000 \times 256}$（序列特征，所有层不变）

处理：
1. 🔴 线性投影生成查询：$\mathbf{Q} = \mathcal{L}(\mathcal{X}^k) \in \mathbb{R}^{16 \times 256}$（$q=16$，每一层变化，因为 $\mathcal{X}^k$ 变化）
2. 🔴 对稀疏特征：$\mathbf{R}_s = \Phi_s(\mathbf{Q}, \mathbf{S}) \cdot \mathbf{S} \in \mathbb{R}^{16 \times 256}$（MLP 注意力核，每一层变化）
3. 🔴 对序列特征：$\mathbf{R}_e = \Phi_e(\mathbf{Q}, \mathbf{E}) \cdot \mathbf{E} \in \mathbb{R}^{16 \times 256}$（缩放点积注意力，每一层变化）

输出：
- 🔴 $\mathbf{R}_s \in \mathbb{R}^{16 \times 256}$（从稀疏特征中选择的证据，每一层变化）
- 🔴 $\mathbf{R}_e \in \mathbb{R}^{16 \times 256}$（从序列特征中选择的证据，每一层变化）

---

**Step 2: Match（Relevance Matching）**

输入：
- 🔴 当前知识库：$\mathcal{X}^k \in \mathbb{R}^{64 \times 256}$（每一层变化）
- 🔴 选择的证据：$\mathbf{R}_s \in \mathbb{R}^{16 \times 256}$ 和 $\mathbf{R}_e \in \mathbb{R}^{16 \times 256}$（每一层变化）

处理：
1. 🔴 线性投影生成模板：$\mathbf{T} = \mathcal{L}(\mathcal{X}^k) \in \mathbb{R}^{32 \times 256}$（$t=32$，每一层变化）
2. 🔴 点积匹配：$\boldsymbol{\lambda}_s = \mathbf{R}_s \cdot \mathbf{T}^\top \in \mathbb{R}^{16 \times 32}$（每一层变化）
3. 🔴 点积匹配：$\boldsymbol{\lambda}_e = \mathbf{R}_e \cdot \mathbf{T}^\top \in \mathbb{R}^{16 \times 32}$（每一层变化）

输出：
- 🔴 $\boldsymbol{\lambda}_s \in \mathbb{R}^{16 \times 32}$（稀疏特征与模板的相关性分数，每一层变化）
- 🔴 $\boldsymbol{\lambda}_e \in \mathbb{R}^{16 \times 32}$（序列特征与模板的相关性分数，每一层变化）

---

**Step 3: Refine（Knowledge Base Refinement）**

输入：
- 🔴 当前知识库：$\mathcal{X}^k \in \mathbb{R}^{64 \times 256}$（每一层变化）
- 🔴 相关性分数：$\boldsymbol{\lambda}_s \in \mathbb{R}^{16 \times 32}$ 和 $\boldsymbol{\lambda}_e \in \mathbb{R}^{16 \times 32}$（每一层变化）
- 🟢 密集特征：$\mathbf{D} \in \mathbb{R}^{1024}$（所有层不变）

处理：
$$
\mathcal{X}^{k+1} = \mathcal{X}^k + \mathrm{MLP}_\mu\Big(
    \mathrm{Concat}\big(
        \mathrm{RMSNorm}(\boldsymbol{\lambda}_s),
        \mathrm{RMSNorm}(\boldsymbol{\lambda}_e),
        \mathcal{L}(\mathcal{X}^k),
        \mathbf{D}
    \big)
\Big)
$$

输出：
- 🔴 更新后的知识库：$\mathcal{X}^{k+1} \in \mathbb{R}^{64 \times 256}$（每一层变化）

---

### Q4：Select-Match-Refine 是否替代了 Transformer self-attention？好处是什么？

**是的，Select-Match-Refine 确实替代了传统 Transformer 的 self-attention 形式，但它不仅仅是替代——更是一种范式转变。**

**核心差异**：

| 维度 | Transformer (HSTU) | SlimPer (Select-Match-Refine) |
|------|-------------------|-------------------------------|
| **中间表示** | 每一层维护 $N \times d$ 的 token-level hidden states | 每一层只维护 $K \times d$ 的知识库（$K=64$） |
| **信息流动** | 前一层的 hidden states → 后一层的 input | 原始用户侧 tokens 在每一层都被直接访问 |
| **交互类型** | $\mathcal{O}(N^2)$ 用户侧 token 之间的自注意力 | $\mathcal{O}(K \times N)$ 知识库与用户侧 tokens 的 cross-attention |
| **容量分配** | 大部分计算用于用户-用户交互 | 全部计算用于用户-Item 相关性匹配 |

**好处不仅仅是"不用计算多层 hidden representation"**：

1. **深度与序列长度解耦**：层数增加时，内存保持 $\mathcal{O}(L \times K)$（$K$ 是常数），而非 $\mathcal{O}(L \times N)$
2. **无损信息访问**：每一层都直接访问原始用户侧 tokens，不存在不可逆信息损失（打破 Data Processing Inequality）
3. **交互容量重定向**：$\mathcal{O}(K \times N)$ 计算全部用于用户-Item 相关性匹配，而非用户侧 token 间交互
4. **天然支持 ROO**：同一次请求的所有候选 item 共享用户侧 tokens，只需为每个 item 维护各自的知识库

---

### Q5：输出得到的知识库怎么用于概率计算？

**知识库 $\mathcal{X}$ 本身不是概率，而是 `<user, item>` 相关性的紧凑表示。概率计算通过以下两种方式实现：**

**方式一：各层任务嵌入求和（推荐方式）**

每一层可以选择输出一个任务嵌入，代表该层对某个任务的理解：

$$
\mathbf{P}_p^k = \mathrm{MLP}_p(\mathcal{L}(\mathcal{X}^k))
$$

然后将所有层的任务嵌入求和，再通过最终线性投影得到 logits：

$$
\hat{y}_p = \mathrm{Linear}\left(\sum_{k=1}^{L} \mathbf{P}_p^k\right)
$$

最后通过 Sigmoid 得到概率：

$$
P(\text{engagement}_p) = \sigma(\hat{y}_p)
$$

**方式二：仅使用最终层知识库（推断）**

> 注：论文中只明确描述了方式一（各层嵌入求和），方式二是基于模型结构的合理推断。

也可以只使用最后一层的知识库 $\mathcal{X}^L$ 进行预测：

$$
\hat{y}_p = \mathrm{Linear}(\mathrm{MLP}_p(\mathcal{L}(\mathcal{X}^L)))
$$

**示例流程（多任务场景）**：

```
X^0 → X^1 → X^2 → X^3 → X^4 → X^5 → X^6 → X^7
       │     │     │     │     │     │     │
       P^1   P^2   P^3   P^4   P^5   P^6   P^7
       │     │     │     │     │     │     │
       └─────┴─────┴─────┴─────┴─────┴─────┘
               │
               ▼
         Σ P_p^k (对每个任务 p)
               │
               ▼
         Linear(Σ P_p^k) → logit_p
               │
               ▼
         σ(logit_p) → P_p (概率)
```

**关键设计思想**：使用各层嵌入求和是因为不同层捕获不同层次的信息（低层捕获广泛上下文，高层捕获精细偏好），求和相当于让模型学习每层对最终预测的贡献权重，类似于残差连接，允许梯度更好地反向传播。

---

### Q6：Step 2 Match 中的证据是什么？维度为什么是确切的数字？

**证据是什么？**

Step 2 Match 的证据是 Step 1 Select 输出的 $\mathbf{R}_s$ 和 $\mathbf{R}_e$：
- $\mathbf{R}_s \in \mathbb{R}^{q \times d}$（$q=16$, $d=256$）：从稀疏特征中通过注意力选择的证据
- $\mathbf{R}_e \in \mathbb{R}^{q \times d}$（$q=16$, $d=256$）：从序列特征中通过注意力选择的证据

**直观理解**：
- Step 1 Select 像是"从用户历史中挑选出与当前 item 最相关的线索"
- 这些线索（$\mathbf{R}_s$ 和 $\mathbf{R}_e$）就是 Step 2 Match 的"证据"
- Match 步骤计算这些证据与 item 模板之间的匹配程度

---

**维度为什么是确切的数字？**

是的，这些数字是论文中给定的超参数，通过实验确定：

| 超参数 | Feed | Reels | 含义 |
|--------|------|-------|------|
| $L$（层数） | 5 | 7 | 迭代精化的次数 |
| $K$（知识库大小） | 64 | 64 | 知识库的槽位数 |
| $d$（隐藏维度） | 256 | 256 | 每个 token 的向量维度 |
| $q$（查询槽） | 16 | 16 | Select 步骤中查询向量的数量 |
| $t$（模板槽） | 32 | 32 | Match 步骤中模板向量的数量 |

**为什么是这些特定数值？**

1. **$K=64$**：消融实验表明，$K=64$ 提供最佳的质量-效率平衡（$K=4$ 时 QPS +23.2% 但 NE 退化 +1.2%）
2. **$d=256$**：工业推荐系统中常见的 embedding 维度
3. **$q=16$**：实验确定的超参数（注意：$q=K/4$ 是数值巧合，论文未明确定义为比例关系）
4. **$t=32$**：实验确定的超参数（注意：$t=K/2$ 是数值巧合，论文未明确定义为比例关系）

**维度之间的关系**：
- $\mathbf{Q} \in \mathbb{R}^{q \times d}$：从知识库生成 $q$ 个查询向量
- $\mathbf{R}_s, \mathbf{R}_e \in \mathbb{R}^{q \times d}$：注意力输出，与查询数量一致
- $\mathbf{T} \in \mathbb{R}^{t \times d}$：从知识库生成 $t$ 个模板向量
- $\boldsymbol{\lambda}_s, \boldsymbol{\lambda}_e \in \mathbb{R}^{q \times t}$：点积匹配分数矩阵

---

### Q7：Step 1 中线性投影生成查询是怎么来的？每一层会变化吗？

**查询向量是怎么生成的？**

根据论文第 249-253 行，查询向量 $\mathbf{Q}$ 通过 segment-wise linear projection 从当前知识库 $\mathcal{X}^k$ 派生：

$$
\mathbf{Q} = \mathcal{L}(\mathcal{X}^k) \in \mathbb{R}^{q \times d}
$$

**生成过程**：
1. **输入**：当前层的知识库 $\mathcal{X}^k \in \mathbb{R}^{64 \times 256}$
2. **操作**：通过可学习的线性变换 $\rho \in \mathbb{R}^{q \times K}$（$q=16$, $K=64$）
3. **输出**：$\mathbf{Q} = \rho \cdot \mathcal{X}^k \in \mathbb{R}^{16 \times 256}$

**直观理解**：知识库 $\mathcal{X}^k$ 包含了 64 个槽位的信息，线性投影 $\mathcal{L}$ 从中提取出 16 个"查询方向"，每个方向代表模型想要从用户历史中寻找的一种信息模式。

---

**查询向量每一层会变化吗？**

**是的，$\mathbf{Q}$ 每一层都会发生变化！**

因为 $\mathbf{Q}$ 是从知识库 $\mathcal{X}^k$ 派生的，而知识库在每一层都会被更新（$\mathcal{X}^k \rightarrow \mathcal{X}^{k+1}$），所以 $\mathbf{Q}$ 也会随之变化。

**这是 SlimPer 的核心设计**：
- 随着知识库逐步精化，查询向量也会变得更加精准
- 早期层的查询可能比较宽泛（"寻找与狗相关的内容"）
- 后期层的查询可能更加精细（"寻找最近没有跳过的狗视频"）

---

**各层变量变化情况总结**：

| 变量 | 是否变化 | 说明 |
|------|----------|------|
| 🟢 $\mathbf{S}$（稀疏特征 tokens） | ❌ | 所有层共享，不更新 |
| 🟢 $\mathbf{E}$（序列特征 tokens） | ❌ | 所有层共享，不更新 |
| 🟢 $\mathbf{D}$（密集特征） | ❌ | 所有层共享，不更新 |
| 🔴 $\mathcal{X}^k$（知识库） | ✅ | 每一层被更新 |
| 🔴 $\mathbf{Q}$（查询向量） | ✅ | 从 $\mathcal{X}^k$ 派生，随知识库变化 |
| 🔴 $\mathbf{R}_s, \mathbf{R}_e$（选择的证据） | ✅ | 依赖 $\mathbf{Q}$，随查询变化 |
| 🔴 $\mathbf{T}$（模板向量） | ✅ | 从 $\mathcal{X}^k$ 派生，随知识库变化 |
| 🔴 $\boldsymbol{\lambda}_s, \boldsymbol{\lambda}_e$（匹配分数） | ✅ | 依赖 $\mathbf{R}$ 和 $\mathbf{T}$，随证据和模板变化 |
