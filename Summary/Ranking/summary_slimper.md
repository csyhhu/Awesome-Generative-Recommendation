# SlimPer: Make Personalization Model Slim and Smart

## 用户综合理解

[2026.08.04] 本文的输入是用户序列 + 稀疏密集特征 + 候选item组成的序列，为避免HSTU中每一层都要进行一次full attention. 本文给每个候选Item生成一个固定维度的表达，执行多层target attention. 这个target attention在文中被称为SlimPer, 其实是一系列query attention. 最终使用这个表达用于概率预估。

[2026.09.19] 本文motivation是更新item侧表达。Method综合来说：对每个item, 除了本身的1-d embedding, 再加上一个2-d tensor embedding, 对其他token (用户序列，ROO特征，用户 x item特征)做两种target attention, 之后对两个的结果做融合，最后对自身做更新。进行N层，在这个过程中，其他token不做变化。

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

SlimPer 的输入是用户序列、稀疏特征、密集特征和候选item。为避免 HSTU 中每一层都要执行 full self-attention（$\mathcal{O}(N^2)$），本文给每个候选 item 生成一个固定维度的知识库表达（64×256），执行多层的 Select-Match-Refine 迭代精化。每一层的 Select 步骤使用从知识库派生的 query 向量对用户侧 tokens 执行 attention，这类似于一系列 query-based attention。最终使用精化后的知识库表达用于概率预估。

---

## 论文信息

- **标题**: SlimPer: Make Personalization Model Slim and Smart
- **作者**: Siqi Wang, Xianjie Chen 等 (Meta Platforms, Inc.)
- **arXiv**: https://arxiv.org/abs/2607.12281

**内部对照材料**（用于本文档"论文 vs 内部落地"的交叉核对）：
- 内部技术分享 deck《SlimPer: Make Personalization Model Arch Slim and Smart》（Xianjie Chen，2025-12，Cross-app Core Modeling Council 分享）
- 内部推广帖《[Model Scaling] Scaling IG's Recommendations Models towards lifetime user understanding》（Jordan Edwards，2025-12-17）
- 内部可解释性落地帖《Explainable Model Predictions from Slimper Using FIND》（Misael Manjarres 等）
- 内部数据分析帖《[Slimper User Value Understand] Phase 1: Cross-Surface UIH Deep Dive and Opportunities》（Yilin Qi）

> 下文中用 【论文】 标注公开论文的表述、用 【内部】 标注仅出现在内部材料中的信息，二者出现分歧的地方会额外用 ⚠️ 标注。

---

## Motivation

### 【论文】叙事：Transformer 架构与推荐任务的设计前提不匹配

论文的第一性原理论证是：LLM 依赖逐 token 的自回归监督，因此必须在每一层维护与序列长度 $N$ 成正比的大型中间张量（$\mathcal{O}(N \times d)$），代价是 $\mathcal{O}(N^2)$ 的自注意力计算。而推荐系统是判别式任务，每个 `<user, item>` 对只需要输出 $\tau$ 个标量相关性分数，不存在 token 级别的监督信号。论文认为，业界把 Transformer 架构直接搬到推荐系统（HSTU、Interformer 处理序列；OneTrans、HHFT、RankMixer 统一多模态）继承了一个不该继承的设计前提，由此提出"把个性化排序重新表述为对紧凑知识库的迭代精化"这一新范式。

### 【内部】叙事：三代架构演进 + 三个经验性假设驱动，而非从零设计

内部材料给出的动机链条完全不同，是一条渐进式的工程演化史，而不是从理论出发的重新设计：

1. **SparseNN（2016–2024）**：IG 个性化模型的基础框架，擅长处理稀疏/密集特征（配合 Dot Compress++/DCPP、Wukong、DotEncoder 等 OverArch 交互技术），但无法消费动态长度的序列特征，2024 年开始遇到扩展瓶颈。
2. **HSTU 引入序列建模（2024–H1 2025）**：MRS 与 IG 合作，通过数据/UIH 特征改造、ROO 范式、HSTU 注意力机制，首次证明序列建模（1k）对 IG Reels 有实质性收益，H1 2025 进一步把序列长度从 1k 扩展到 2k。
3. **H2'25 遇到新瓶颈**：① 计算低效——HSTU 和 SparseNN 两套架构并行、逻辑重复，维护成本高；② 事件特征 scale-up 边际收益递减——2k 序列长度往上再加事件特征，NE 收益和成本比开始变差；③ 模态连接不足——三类模态（sparse/dense/event）是"松散拼接"而非联合建模，错过了跨模态交互的机会。
4. **SlimPer 的立项**是针对以下三个假设做验证：
   - **H1**：dense/sparse 模态在序列 scale 之后是否还重要？——是，SparseNN/Wukong/DotEncoder 的 OverArch 交互仍在贡献增量。
   - **H2**：是否需要把三种模态高效地连起来一起 scale？——是，当前架构各模态独立 scale、缺乏联合建模。
   - **H3**：序列建模是否还有更高效的资源分配方式？——是，HSTU 沿用了 LLM 式 vanilla transformer 的 $\mathcal{O}(N^2)$ self-attention，把大量计算花在了"用户-用户"token 交互上，但 IG 模型的主任务是 user-item 多任务预测，这部分资源分配并非最优。

5. **⚠️ 差异**：论文中"用户-用户交互是低价值连接、用户-item 交互是高价值连接"的核心论证（信息论 & 交互容量分析），在内部 deck 里其实是 H3 假设验证后的产物，且明确定性为"reduce 低价值连接、enlarge 高价值连接"，配的是工程化的 Big-O 表格，而不是论文里更抽象的 Data Processing Inequality / mutual information 论证——内部叙事本质上更偏工程直觉，论文把它包装成了更严谨的信息论语言。

6. **⚠️ 差异**：论文完全没有提及的驱动力——**对标 TikTok**。内部帖子明确把 UIH scaling（2k→5k→10k→16k→30k）的路线图和 ROI 分析，锚定在"TikTok 已经在用 10k~20k 未采样序列 + target attention，甚至用另一路 target attention 把最长 1 年（~百万级事件）的终身序列压缩到约 1.5 万长度作为补充信号"这一竞对情报上。SlimPer 之所以要解决 O(N²) 瓶颈，直接目的就是要有能力把序列长度堆到能对标 TikTok 的量级。

---

## Method

### 整体流程

模型整体上仍是论文里描述的 Tokenization → 知识库初始化 → L 层 Select-Match-Refine → 多任务输出，但内部 deck（Slide 20–23）给出了生产环境里的真实张量形状，比论文附录的抽象超参数表（$K$=64, $d$=256, $q$=16, $t$=32）更具体：

| 模块 | 内容 | 【内部】真实张量形状（Reels LSR，5k UIH 版本） |
|------|------|------|
| Tokenization - 用户稀疏 | Embedding Pooling | $\mathbf{S}\in\mathbb{R}^{B_{RO}\times\sim150\times256}$ |
| Tokenization - 用户序列（UIH） | Event Modeling | $\mathbf{E}\in\mathbb{R}^{B_{RO}\times\sim5000\text{–}6000\times256}$ |
| Tokenization - 用户密集 | MLP 投影 | $\mathbf{D}\in\mathbb{R}^{B\times\sim1000}$ |
| Tokenization - Item 稀疏 | Embedding Pooling | $\in\mathbb{R}^{B\times\sim30\times256}$ |
| 知识库 $\mathcal{X}^k$ | Select-Match-Refine 的载体 | $\mathbb{R}^{B\times64\times256}$（每个候选 item 一份） |
| Query（Select 步骤） | 从 $\mathcal{X}^k$ 线性投影 | $\mathbb{R}^{B_{NRO}\times16\times256}$ |
| Dot Product 输出（Match 步骤） | $\lambda_s,\lambda_e$ flatten | $\mathbb{R}^{B\times(32\times16)}=\mathbb{R}^{B\times512}$ |

其中 $B_{RO}$（Request-Only 侧，即用户侧）和 $B_{NRO}$（Non-Request-Only 侧，即候选 item 侧）的区分，就是 ROO 设计在张量层面的直接体现：用户侧张量按 request batch 只算一次，候选 item 侧张量按 100+ 个候选分别维护。这与论文 Table 1 里 $\mathcal{O}(N/B)$ 的 tokenization 成本一一对应。

### Select-Match-Refine 三步（公式 + 真实实现）

**Step 1 Select**：$\mathbf{Q}=\mathcal{L}(\mathcal{X}^k)\in\mathbb{R}^{16\times256}$，对稀疏 token 用 MLP 注意力核、对序列 token 用标准缩放点积注意力，分别取回 $\mathbf{R}_s,\mathbf{R}_e\in\mathbb{R}^{16\times256}$。【内部】图上把这一步的 attention 权重直接标注为 `Attention Weights = Dot(Q, K)`，与 HSTU 处理 UIH 的方式完全一致——这一步本身就是把 HSTU 的 QKV attention 原样搬进来，用于从序列 token 里检索证据。

**Step 2 Match**：从 $\mathcal{X}^k$ 派生模板 $\mathbf{T}\in\mathbb{R}^{32\times256}$，与 $\mathbf{R}_s,\mathbf{R}_e$ 做显式点积，得到 $\boldsymbol{\lambda}_s,\boldsymbol{\lambda}_e\in\mathbb{R}^{16\times32}$。

> ⚠️ **关键差异**：论文把这一步呈现为"全新的多面模板显式相关性匹配机制"。但内部 deck（Slide 6）明确说得很直白："Reformulated the popular OverArch technique Dot Compress++ (DCPP) as QKV attention + Dot Product to make it ROO-aware"——也就是说 Match 步骤本质上是把 SparseNN 里已经生产多年的 DCPP 特征交叉方式，重新表述成 QKV attention 形式，使其满足 ROO 约束（用户侧只算一次、跨候选共享），而不是一个从零发明的组件。

**Step 3 Refine**：$\mathcal{X}^{k+1}=\mathcal{X}^k+\mathrm{MLP}_\mu(\mathrm{Concat}(\mathrm{RMSNorm}(\boldsymbol{\lambda}_s),\mathrm{RMSNorm}(\boldsymbol{\lambda}_e),\mathcal{L}(\mathcal{X}^k),\mathbf{D}))$。【内部】图上标注为 `Norm & Concat & MLP` 接 `Add & Norm`，与论文公式一致。

> ⚠️ **关键差异（每层是否真的能访问"完整"原始 token）**：论文把"Complete Access：每一层都直接查询完整的原始用户侧 tokens"列为三大设计原则之一，并作为"无信息损失"论证的核心支撑。但可解释性落地帖明确提到：Reels LSR **MB7**（论文报告 5k 结果对应的那个生产 checkpoint）实际是**奇偶层交替处理 UIH**——奇数层只处理奇数索引事件、偶数层只处理偶数索引事件；要到 **MB8** 才会让每一层看到全部 10k UIH。也就是说论文强调的"每层完整访问"设计原则，在实际部署的那一版里是打了折扣的工程折中，并非严格成立，MB8 才是真正意义上的完整实现。

### 关于知识库瓶颈与 Contextual Encoding：补丁还是加分项？

论文信息论章节主张"迭代访问打破了 Data Processing Inequality，不存在不可逆信息损失"，随后把 Contextual Encoding（编码每个事件邻域的上下文）作为 Section 5.4.1 的一项消融改进呈现，显得像是一个可选的锦上添花组件。

> ⚠️ **关键差异**：内部 deck（Slide 12）明确承认一个限制：64×256 的 x-tensor 容量有限，QKV attention 检索到的只是"整体全局用户上下文"，很难捕捉每个事件周围的局部依赖（"It is hard to capture subtle local context from each event token's surrounding events"）。正因为知识库瓶颈天生就学不到局部上下文，才需要专门设计 Contextual Encoding 模块去做局部窗口特征工程加以补救。换句话说，Contextual Encoding 在内部视角里是**修补**知识库瓶颈已知短板的必要组件，而不是论文呈现方式所暗示的可选加分项。

### 复杂度对比（论文 Table 2，与内部 deck 数字一致）

| 模型 | 每层计算复杂度 | 每层内存复杂度 | 用户-Item 交互容量 |
|------|----------------|----------------|--------------------|
| SparseNN（不 ROO-aware） | $\mathcal{O}(L\cdot N\cdot16)$ | $\mathcal{O}(L\cdot64)$ | $\mathcal{O}(L\cdot N\cdot64)$ |
| HSTU（IG Prod 25'H1，ROO-aware） | $\mathcal{O}(L\cdot(N+1)^2/B)$ | $\mathcal{O}(L\cdot N/B)$ | $\mathcal{O}(L\cdot N)$ |
| HSTU w/ Sparse Attention（sliding window） | $\mathcal{O}(L\cdot N+L\cdot N\cdot W/B)$ | $\mathcal{O}(L\cdot N/B)$ | $\mathcal{O}(L\cdot N)$ |
| **SlimPer**（全模态 ROO-aware） | $\mathcal{O}(L\cdot N\cdot16)$ | $\mathcal{O}(L\cdot64)$ | $\mathcal{O}(L\cdot N\cdot64)$ |

内部 deck 的旁注补充了论文没有的两个具体结论：SlimPer 比 HSTU（2k~10k、5 层配置下）**内存效率高 6×~30×**；SparseNN 和 SlimPer 在 user-item 交互学习上花的容量是 HSTU 的 **64 倍**（对应 $K=64$）。

### ROO-Aware 的范围

【论文】把 ROO 从 HSTU 原来只覆盖用户历史序列，扩展到了所有用户侧模态（稀疏、密集、序列）。【内部】deck 用更明确的 RO（Request-Only，用户侧）/ NRO（Non-Request-Only，候选 item 侧）划分来表达同一件事，并强调这是"Hypothesis 1: User & Item Features are not mixed together"——用户侧和 item 侧特征在 tokenization 阶段就被严格分开，从根源上保证了共享/不共享的边界清晰，这是 ROO 能落地的前提条件，论文中没有单独强调这一点。

---

## A Detailed Case

**场景设定**：用户 Alice 在 Instagram 上有：
- 稀疏特征：喜欢的类别（摄影、旅行、美食），关注的创作者 ID
- 序列特征：最近 2000 条交互事件（点赞、观看、分享、跳过等）
- 密集特征：当前时间、设备类型、历史 CTR 等
- 候选 Item：100 个待排序的帖子

> 为了贴近生产真实规模，下面同时给出论文摘要参数（Alice 场景，$N$=2000）和内部生产真实规模（Reels LSR 5k UIH 版本，$N\approx5000$）两组数字做对照。

**Step 0：Tokenization（ROO，每个 request 只算一次）**

| 特征 | Alice 场景（$N$=2000） | 【内部】生产规模（5k UIH） |
|------|------------------------|------------------------------|
| 用户稀疏 token | 3 个 token（喜欢的类别、关注的创作者） | $\sim$150 个 token |
| 用户序列 token | 2000 个事件 token | $\sim$5000–6000 个事件 token |
| 用户密集向量 | 1 个向量 | $\sim$1000 维向量 |
| Item 侧稀疏 token（100 个候选各自独立） | 每候选若干 token | $\sim$30 个 token/候选 |

这四类 token 一旦算好，在整个 7 层循环里**只读不写**，且在 100 个候选 item 之间共享（ROO）。

**Step 1：知识库初始化**

对 100 个候选 item 中的每一个，从其 item 侧特征线性投影出 $\mathcal{X}^0\in\mathbb{R}^{64\times256}$。知识库是"每个 `<user,item>` 对一个"，而不是每个用户一个——100 个候选对应 100 份独立知识库，共享同一份用户侧 token。

**Step 2：第 1~7 层 Select-Match-Refine**（内部生产的 Reels LSR MB7 实际是 6 层循环 + 输出层，论文 Reels 超参数表标注 $L$=7，二者接近但不完全对齐，可能对应不同 checkpoint）

对每一层 $k$：
1. Select：$\mathbf{Q}=\mathcal{L}(\mathcal{X}^k)\in\mathbb{R}^{16\times256}$，分别对 3 个稀疏 token（用 MLP 注意力核）和 2000 个序列 token（用缩放点积注意力，⚠️MB7 版本里实际只访问其中奇/偶索引的一半）做 attention，取回 $\mathbf{R}_s,\mathbf{R}_e\in\mathbb{R}^{16\times256}$。
2. Match：从 $\mathcal{X}^k$ 派生模板 $\mathbf{T}\in\mathbb{R}^{32\times256}$，与 $\mathbf{R}_s,\mathbf{R}_e$ 做点积，得到 $\boldsymbol{\lambda}_s,\boldsymbol{\lambda}_e\in\mathbb{R}^{16\times32}$——这一步是 DCPP 的 ROO 化版本。
3. Refine：归一化后 concat 上 $\mathcal{L}(\mathcal{X}^k)$ 和密集向量 $\mathbf{D}$，过 $\mathrm{MLP}_\mu$，残差加回知识库，得到 $\mathcal{X}^{k+1}\in\mathbb{R}^{64\times256}$。

**Step 3：输出预测**

每一层可选输出任务向量 $\mathbf{P}_p^k=\mathrm{MLP}_p(\mathcal{L}(\mathcal{X}^k))$，7 层求和后线性变换、sigmoid，得到每个候选 item 在 $\tau$ 个任务（reshare、like、comment、skip……）上的概率。

**与基线（HSTU + SparseNN/Wukong）的对比**

| 维度 | 基线（HSTU + SparseNN，Late Fusion） | SlimPer |
|------|--------------------------------------|---------|
| 序列建模每层中间张量 | $N\times256$（$N$=2000 时约 2000×256） | 固定 64×256，与 $N$ 无关 |
| 稀疏/密集特征交互方式 | SparseNN 的 DCPP OverArch，独立于序列模块 | DCPP 被重构为 QKV+Dot Product，与序列模块共享同一知识库 |
| 用户侧 token 复用 | HSTU 已 ROO-aware，但 SparseNN 侧未完全 ROO-aware | 全模态 ROO-aware，所有用户侧 token 都只算一次 |
| 每层是否看到完整历史 | 是（自注意力覆盖全部 $N$） | 论文声称是；MB7 生产版本实际奇偶交替，MB8 起才完整 |
| 可解释性落地方式 | 缺乏统一的 attention 归因手段 | 通过内部 FIND 工具在单请求级别物化 attention map，定位到具体历史事件 |

这个对照说明：SlimPer 相对基线的改进，与其说是发明了一种全新的算法范式，不如说是把 HSTU 的 QKV 检索机制和 SparseNN 的 DCPP 交互机制，在 ROO 约束下重新组织进同一个固定大小的知识库里，用一套统一的 Select-Match-Refine 循环取代了"HSTU 算序列 + SparseNN 算稀疏密集 + 后融合"的松散拼接结构。

---

## 追问 Q&A（架构细节澄清）

### Q8：为什么"用户密集"Tokenization 模块的输出维度是 $B$？这个模块的输入是什么？

**输入不止是"用户"密集特征。** 论文 Table 1 对 Dense 模态的定义本身就是三类信号的混合：user-level 统计量（如实时 CTR）、**item quality 指标（如 reshare rate）**、以及上下文属性（设备、时间）。Appendix 的 D-P 模块把这些 raw dense 值拼接后过一个 MLP：$\mathbf{D}=\mathrm{MLP}(\mathrm{Concat}(d_1,\ldots,d_{|\mathcal{D}|}))$，其中就包含了 item 专属和 user×item 交叉的信号，不是纯用户侧的量。

**为什么输出是 $B$、不是去重后的 $B_{RO}$**：内部 deck 的张量图上，Dense Features 明确标注为 `[B, ~1000]`，而不是像 User Sparse / User Event 那样标注成 `[B_RO, ...]`。原因正是上面这点——Dense 里混入了随候选 item 变化的 item-level / cross-level 信号，天然无法像纯用户侧的 Sparse/Event token 那样被 ROO 去重、在候选 item 间共享一份。所以这个模块只能在"每个 `<user,item>` 样本"的粒度上跑，$B$ 就是这批样本的总数——规模确实随候选 item 数量增长，你的直觉是对的。

> ⚠️ 这也暴露了论文"ROO 已扩展到所有用户侧模态"这句话的一个隐藏边界：这个说法在 Sparse 和 Event（序列）两条流上干净成立，但 Dense 流因为混入了 item/cross 信号，天生没法完全去重——内部张量形状用 `[B, ...]` 而非 `[B_RO, ...]` 印证了这一点，论文没有单独说明这个例外。

### Q9："对稀疏 token 用 MLP 注意力核、对序列 token 用标准缩放点积注意力"分别是什么？这是不是做了两次 Target Attention？

论文 Eq.(4) 给出的公式：

$$\Phi(\mathbf{Q},\mathbf{K})=\begin{cases}\mathrm{MLP}(\mathrm{Concat}(\mathcal{L}(\mathbf{Q}),\mathcal{L}(\mathbf{K}))), & \text{稀疏特征}\\ \mathrm{Softmax}(\gamma \mathbf{Q}\mathbf{K}^\top), & \text{序列特征}\end{cases}$$

- **MLP 注意力核（稀疏）**：不是点积，而是把 Query、Key 分别线性投影后拼接，丢进一个小 MLP，直接输出一个学出来的"匹配分数"。这和 DIN 的 activation unit 几乎是同一套思路——DIN 把 target embedding 和 behavior embedding 拼接/交叉后过 MLP 得到 attention 权重，而不是用点积相似度。因为稀疏 token 数量少而固定（内部数据约 150 个以内），$q\times|S|$ 次 MLP 前向的开销可以接受，用参数化核换取更强的非线性匹配表达力。
- **标准缩放点积注意力（序列）**：就是 Transformer 原始公式 $\mathrm{Softmax}(QK^\top/\sqrt d)$（论文写成温度系数 $\gamma$），和 HSTU/标准 self-attention 是同一套非参数化相似度度量。序列 token 数量大（数千到上万），逐对 MLP 代价太高，点积可以整体走一次矩阵乘法，GPU 友好。
- **是否是两次 Target Attention**：是的。本质上同一层里做了两次 Target Attention——Query 都来自当前知识库 $\mathcal{X}^k$，一次查稀疏特征池、一次查序列历史池，只是针对两种模态的 token 规模选了不同 kernel（参数化 MLP vs 非参数化点积）。和经典 DIN 的区别在于：DIN 通常只用 target item 的原始 embedding 做一次性 attention；SlimPer 是用不断被 Refine 的 $\mathcal{X}^k$ 做 query，在 $L$ 层里重复做了 $L\times2$ 次这种 target attention，每一次的 query 都比上一次更"懂"这个 `<user,item>` 对。

### Q10：Match 中派生的模板 $\mathbf{T}$ 是可学习参数吗？

**不完全是——投影权重可学习，但模板向量本身是随输入动态生成的，不是一份静态的可学习向量库。**

$\mathbf{T}=\mathcal{L}(\mathcal{X}^k)$ 里的 $\mathcal{L}$ 是 segment-wise 线性投影 $\beta=\rho\alpha$，$\rho\in\mathbb{R}^{32\times64}$。$\rho$ 是反向传播训练出来的可学习参数矩阵，这部分和普通线性层权重无异。但 $\mathbf{T}$ 这个"结果"本身，是把**当前这个 `<user,item>` 样本、当前这一层的知识库状态** $\mathcal{X}^k$ 过一遍 $\rho$ 算出来的，会随样本、随层变化——这和"学一份固定的模板/原型词表，所有样本共享同一批模板向量"（比如 VQ 的 codebook，或 memory network 里持久化的 memory matrix）是两种不同设计：SlimPer 没有维护跨样本共享的静态模板库，模板永远是"临时生成"的。

论文附录还特别说明："we use this module ($\mathcal{L}$) multiple times, each with independent learnable parameters"——Step 1 生成 Query 用的 $\rho_Q$ 和 Step 2 生成模板用的 $\rho_T$，输入都是同一个 $\mathcal{X}^k$，但是两套完全独立的可学习权重。

### Q11：Item 侧稀疏 token 原本有特征吗？会变成 dense embedding 吗？和知识库的 embedding 有什么区别？

**原始特征**：有。按 Table 1 的模态定义，item 侧稀疏特征是 item 自己的高基数类别 ID——类比用户侧"最喜欢的创作者/类别"，item 侧对应的是 item 的类别 ID、作者/创作者 ID、话题标签 ID 等（内部 deck 标注为 `Item Sparse Features [B, ~30, 256]`，约 30 个 token/候选）。

**是否变成 dense embedding**：是。这些原始类别 ID 经过和用户侧完全相同的 Embedding-Pooling（E-P）模块——查表 + sum/mean pooling——变成固定数量的 $d$=256 维连续向量。Tokenization 之后，item 侧稀疏特征在数值形式上已经和用户侧稀疏 token 一样，都是 dense embedding。

**和知识库 embedding 的区别**：

| 维度 | Item 侧稀疏 token | 知识库 $\mathcal{X}^k$ |
|------|--------------------|--------------------------|
| 语义 | "这个 item 本身是什么"——身份类信息，与具体哪个用户无关 | "这个 `<user,item>` 对目前的相关性理解"——关系类信息，因用户而异 |
| 是否跨层更新 | ❌ 不更新，$L$ 层里始终不变（🟢，只读） | ✅ 每层残差更新（🔴），$\mathcal{X}^0\to\mathcal{X}^L$ |
| 在架构中的角色 | 只作为 $\mathcal{X}^0$ 的初始化来源（$\mathcal{X}^0=\mathcal{L}(\mathbf{S}_{\text{in}})$），Select 阶段之后不再被查询 | 既是被 Query 的来源（$Q=\mathcal{L}(\mathcal{X}^k)$），也是被更新写入的目标 |
| 是否用户特定 | 同一 item 对所有用户的 embedding 完全相同（标准 embedding 表查值） | 同一 item 对不同用户会演化出完全不同的轨迹，即使 $\mathcal{X}^0$ 相同 |

简单说：item 侧稀疏 token 是"原材料"，知识库是"用这份原材料打底、再不断吸收用户证据后炖出来的汤"——$\mathcal{X}^0$ 那一刻两者还很接近（只差一次线性投影），但从 $\mathcal{X}^1$ 开始就分道了。

### Q12：K 是什么？

**$K$ 就是知识库 $\mathcal{X}$ 的槽位数（slot 数量）**——知识库张量的形状是 $\mathcal{X}^k\in\mathbb{R}^{K\times d}$，$K$=64、$d$=256 是论文默认配置（内部生产 Reels/Feed 也都用 $K$=64）。可以把它理解成"这个 `<user,item>` 对的中间表达，允许用多少个独立的向量位来存放"——$K$ 越大，知识库能同时记住的"证据面"越多，表达力越强，但每层的 Refine/Match 计算和内存也随 $K$ 线性增长。

$K$ 在全文里对应几件事：
- 是论文 Table 2 复杂度分析里"每层内存 $\mathcal{O}(L\cdot K)$"、"用户-item 交互容量 $\mathcal{O}(L\cdot N\cdot K)$"两处 $K$ 的来源；
- 是 Section 5.4.2 消融实验里被扫描的超参数（$K\in\{4,8,16,32,64\}$），实验里 query 数 $q$、模板数 $t$ 都按 $K$ 的比例联动缩放（如 $K$=4 时 $q$=1, $t$=2）；
- 是本节 Q13/Q14 讨论"退化到极小知识库"时的核心变量——$K=1$ 意味着知识库只剩 1 个槽位，退化成"单个不断被 refine 的向量"。

注意 $K$ 和 $q$（Select 步骤里的 query 数，默认 16）、$t$（Match 步骤里的模板数，默认 32）是三个不同但相关的超参数：$K$ 是知识库本身的容量，$q$、$t$ 是知识库每次生成 query/模板时"抽取"出的子集大小，三者共同决定了每层的计算量（$\mathcal{O}(q\times N)$ 主导）和表达力上限。

### Q13：一个明显的 baseline——在 HSTU 里只更新 item token、历史 token 算一次后冻结，复杂度和内存都同阶，论文/内部有做这个实验吗？

**没有找到直接对应、明确命名的实验**——但可以从三个角度间接回答：

1. 这个 baseline 本质上是 **SlimPer 自己在 $K=1$ 时的退化情况**，也基本等价于论文 Related Work 里明确讨论过的 Perceiver/Perceiver IO（"a learned latent array that iteratively cross-attends into raw inputs"），只是 Perceiver 的 latent 通常不止 1 个。论文把 Perceiver 列为最接近的已有架构，区别只在于推荐特定的归纳偏置（显式匹配、ROO、模态相关 kernel），并没有报告和它的直接数值对比。
2. 更接近这个 baseline 单层版本的是 Related Work 里的 DIN/DIEN（"target-aware attention over user interaction histories"，通常只做一次而非迭代 refine）。论文没有把 DIN/DIEN 拉进 Table 3 做数值对比，只在文字上做了定性区分。
3. **可以间接推算这个 baseline 的表现**：论文自己的知识库大小消融（$K\in\{4,8,16,32,64\}$）显示 $K$ 越小 NE 越差且单调下降——$K=4$ 时已经 NE 退化 +1.2%（尽管 QPS +23.2%）。按 Table 2 最右列"user-item 交互容量"的口径，$K=1$ 时交互容量会退化成 $\mathcal{O}(L\cdot N)$，和 HSTU 之类 transformer baseline 完全同阶——也就是论文反复强调的"$K\times$ 优势"在 $K=1$ 时会完全消失。换句话说，论文自己的理论框架已经预测了这个 baseline 会失去 SlimPer 的核心卖点，只是没有真的跑出这一行数字。

> ⚠️ **内部材料里有一个方向类似但不完全相同的实验**：FIND 可解释性帖的评论区提到团队测试过"gated attention"（Diff D88083665），把 attention 输出当作一个在 Preproc 阶段算一次、之后所有层共用的固定 embedding（而不是每层重新算）——这个思路和"冻结历史 token 表示"在精神上接近，但它应用在 Contextual Encoding 的滑窗 self-attention 模块上，不是应用在主干的 Select-Match-Refine 循环上，团队反馈"没看到大的提升"，且归因于用了固定窗口 + 所有层共享同一个 attention 输出，而不是归因于"冻结"本身。

**结论**：你提出的这个 baseline（HSTU 架构下只更新 item token、历史 token 只算一次不再更新）是一个论文和内部材料都没有直接跑过的、真实存在的空白对照实验——现有证据（K 消融的单调趋势 + Perceiver/DIN 的定性讨论）都指向"复杂度确实同阶，但预期质量会明显低于 $K=64$ 的 SlimPer"，但没有一行实测数字直接证明这一点，值得作为一个可以自己补的实验来验证。

### Q14：追问澄清——我说的 baseline 是"HSTU 里只有 item token 更新，其他（历史）token 不更新"，不是笼统的"缩小 SlimPer 的 K"，这种实验做过吗？

这个澄清很关键，值得把两种表述的关系说清楚：**"HSTU 里只让 item token 更新、历史 token 冻结"和"SlimPer 收缩到 $K=1$"其实是同一个架构，只是从两个不同方向收敛到的**——

- 从 HSTU 出发做减法：去掉历史 token 之间的 self-attention（即历史 token 不再跨层更新，永远停留在 tokenization 那一刻的表示），只留一个专门的 item token，每层通过 cross-attention 从这个冻结的历史 token 池里检索信息、更新自己——这样 item token 的状态从 $[1,d]$ 演化到下一层的 $[1,d]$，历史 token 池始终是同一份 $[N,d]$，只读不写。
- 从 SlimPer 出发做减法：把知识库槽位数从 $K$=64 收缩到 $K$=1，$\mathcal{X}^k$ 就变成了单个 $[1,d]$ 向量；$\mathbf{S}$、$\mathbf{E}$（用户侧稀疏/序列 token）本来就是 🟢 全程只读、不跨层更新的——这一点在 SlimPer 里从 $K$=64 到 $K$=1 都不会变。

两条路径落地后是**完全等价的架构**：一个不断被更新的单一状态向量，每层 cross-attend 进一个从头到尾不变的原始 token 池。差异只剩下两个实现细节，不影响这个"是否做过这个精确实验"的问题本身：① SlimPer 版本还会做一次显式的 Match（点积匹配模板）再 Refine，HSTU 版本可能直接用 attention 输出过 MLP 更新；② SlimPer 对稀疏 token 用 MLP 核、序列 token 用点积核两条并行流，HSTU 原本只处理序列这一条流。

**所以答案不变，但可以给出更精确的最接近证据**：论文消融表里最接近这个设定的一行是 $(K,q,t)=(4,1,2)$——注意 $q=1$，也就是说知识库虽然还留了 4 个槽位，但**每一层只用 1 个 query 向量去检索历史**（这已经非常接近"只有一个 item token 在做检索"的设定），结果是 NE 退化 +1.2%、QPS 提升 +23.2%。这是论文和内部材料里唯一一个把"检索端"收缩到 1 的真实实测数据点，可以作为你提议的这个 baseline 的一个偏乐观的下界参考（因为知识库状态本身还有 4 个槎位，比真正的 $K=1$ 略宽裕）。但**严格按"HSTU 架构 + 历史 token 完全冻结 + 只有 1 个 item token 跨层更新"这个确切设定命名的实验，论文和我读到的内部材料里都没有直接报告**，仍然是一个真实的空白点。

### Q15：如果 $K>1$，那其实就是 Multi Token Item，即用多个 token 表达一个 item？

**在 $\mathcal{X}^0$ 这一刻，是的；但随着层数推进，这个说法需要修正一下——它会从"多 token 表达 item"逐渐变成"多 token 表达 `<user,item>` 关系"。**

**为什么在初始层是准确的**：$\mathcal{X}^0=\mathcal{L}(\mathbf{S}_{\text{in}})$ 是纯粹由 item 侧稀疏特征线性投影出来的（Q11 已经讨论过），也就是把 item 原本 $\sim$30 个稀疏 token 重新组合、压缩/扩展成 64 个槎位——这确实就是"用多 token（64 个）表达一个 item"的做法，和一些 recsys/检索里"多向量表示"的思路是同一类设计（比如 ColBERT 用多个 token 级向量表示一篇文档、而不是一个 pooled 向量；MIND/ComiRec 用多个"兴趣向量"表示一个用户）。

**为什么从 $\mathcal{X}^1$ 开始需要修正**：知识库每一层都会通过 Refine 吸收 Select/Match 检索到的用户侧证据（$\boldsymbol{\lambda}_s,\boldsymbol{\lambda}_e$），残差写回 64 个槎位。所以到了 $\mathcal{X}^k$（$k>0$），它已经不再是纯粹描述"这个 item 是什么"，而是描述"这个 `<user,item>` 对目前被理解到什么程度"——64 个槎位承载的是**关系状态**，不只是**item 身份**。严格说，"Multi Token Item"这个说法只对 $\mathcal{X}^0$ 精确，层数越深越应该叫"Multi Token `<user,item>` Relevance State"。

**⚠️ 一个容易高估的地方：64 个槎位并不是显式设计成 64 个"独立可解释的 facet"。** 对比 MIND/ComiRec 这类显式的"多兴趣"模型，它们通常会加入动态路由（capsule routing）或自注意力提取 + 多样性约束，专门强制不同兴趣向量分化去覆盖不同的行为簇；也对比 Slot Attention（Locatello et al. 2020，一个和 SlimPer 结构上非常接近但论文没有引用的先例：$K$ 个可学习 slot 迭代 cross-attend 进原始输入，且 slot 之间做 softmax 竞争）——SlimPer 没有类似的多样性约束或 slot 间竞争机制。Query（$q$=16）和模板（$t$=32）都是用**同一个** segment-wise 线性层 $\mathcal{L}$ 把 64 个槎位**混合**投影出来的（$\rho\in\mathbb{R}^{q\times K}$ 或 $t\times K$，会对全部 64 行做加权组合），并不是"第 1 个槎位专门负责美食兴趣、第 2 个槎位专门负责旅行兴趣"这种显式划分。64 个槎位之间是否会自发分化出可解释的"facet"，是训练过程中隐式涌现的（类似 transformer 的注意力头会自发分化专长而不需要显式监督），论文和内部材料都没有给出"拆解 64 个槎位分别学到了什么"的分析——可解释性章节只分析了对**原始输入 token**（历史事件）的 attention 权重，没有分析知识库内部 64 个槎位各自的语义。

**支持"多 token 表示确实有用"的间接证据**：$K$ 消融实验里 $K$ 从 4 单调增加到 64，NE 持续变好，这至少说明"更多并行槎位=更强的容量/能同时保留更多互不相同的证据面"这个直觉方向是对的，只是不能过度解读成"每个槎位对应一个人类可读的兴趣类别"。

---

## 与《Interest Cache Roadmap》对照：SlimPer 在方法对比表里的定位

内部《Interest Cache Roadmap》文档用一张表对比了 TokenMinds、Personas、FOUNDv2、RecGPTv3（归为"Discrete User Representation"一类）、LLaTTE（归为"Dense Representation"一类）等长历史压缩方法，列出 Input / Training Methods / Output / How to use / Intermediate Performance Validation / Resource Consumption 六列。**SlimPer 被单独放进了"Non-user Representation"这一类别下，且该行在文档里是空的**——下面按这张表的三个关键列，把 SlimPer 的答案补上，同时说明为什么它会被归到"Non-user"而不是"Discrete/Dense User Representation"。

### Q16：SlimPer 的序列数据形式是怎样的，用了什么特征？举一个例子

【内部】明确给出了具体字段，不需要推测。序列是**结构化的多字段事件序列**（不是文本、也不是 RQ-VAE 离散 SID），每个事件由若干原始字段组成：`media_id`（内容 ID）、`author_id`（作者/创作者 ID）、`explicit_engagement_type`（显式动作：like/reshare/comment/exit...）、`implicit_watch_time`（隐式观看时长）、`event timestamp`（事件时间戳）等元数据。

内部数据来源上，最终喂给 SlimPer 的 UIH 由三条 DataFM 原始序列合并去重而成：**IG Public**（全部隐式+显式互动，30 天回看，如视频观看时长/完播）、**IG Public Sparse**（仅显式互动如 like/reshare/exit/reply/comment，400 天回看）、**IG Video Positive**（vvp95/vvp100 等视频完播信号，90 天回看），经 Merge-Dedup、Flip、Truncate 后得到最终长度 2k~10k 的序列。

**举例**（原文和内部 deck 给出的原始例句）：一条事件可以是——

```
{
  media_id: "reel_98213",
  author_id: "creator_5521",
  explicit_engagement_type: "reshare",
  implicit_watch_time: 20s,
  event_timestamp: "xxxxxxxx"
}
```

即"用户观看某条 Reel 20 秒后将其转发，发生在 10 分钟前"——这正是论文 Table 1 和内部 deck Slide 20 里给出的原文例子。这类事件会经过 Event-Modeling（E-M）模块编码成 $1\times256$ 的 token：$\mathbf{E}^{(t)}=\mathrm{MLP}_E^s(e_{\text{sparse}}^{(t)})+e_{\text{temporal}}^{(t)}+\mathrm{MLP}_E^e(e_{\text{type}}^{(t)})+e_{\text{context}}^{(t)}$，四路分别编码内容、时间（位置编码+delta-time）、动作类型、局部上下文，求和融合成一个向量。

> 与表格里其他方法的对比：TokenMinds/Personas/RecGPTv3 都是把历史转成**文本**再喂给 LLM，FOUNDv2 是把每条记录转成 Qwen3 Embedding 再用 RQ-VAE 离散化成 SID token；SlimPer 完全没有走"文本化"或"离散 SID 化"这条路，序列全程保持**结构化多字段特征**，直接用传统 embedding 表 + MLP 编码，这是它和表里其它方法最根本的输入形式差异。

### Q17：SlimPer 最终产出的压缩表达是怎样的？举一个例子

最终产出是知识库 $\mathcal{X}^L\in\mathbb{R}^{64\times256}$（16384 个浮点数），但**关键是它不是一个"用户表达"，而是一个"user-item 对"表达**，且是**每次请求临时算出来、不落盘、不跨请求复用**的中间张量。

**举例**：Alice 打开 Reels，这次请求带来 100 个候选 Reel。SlimPer 会对每一个候选独立跑一遍 7 层 Select-Match-Refine，产出 100 份不同的 $\mathcal{X}^7\in\mathbb{R}^{64\times256}$（每份代表"Alice 的历史证据 + 这一个具体候选 Reel 的相关性理解"）。这 100 份 $\mathcal{X}^7$ 之间没有共享、下一次 Alice 刷新 Feed 请求新的 100 个候选时，会重新算 100 份全新的 $\mathcal{X}^0\to\mathcal{X}^7$，上一次算出来的完全不会被保留或复用。

> 这正是内部文档把 SlimPer 单独归为"**Non-user Representation**"（而不是 TokenMinds/FOUNDv2/RecGPTv3/Personas 所在的"Discrete User Representation"，也不是 LLaTTE 所在的"Dense Representation"）的原因：那几个方法的输出都是**以用户为单位**、离线算好后可以**缓存（cache）、跨下游任务/跨未来请求复用**的持久化 artifact（比如 FOUNDv2 的 220-token SID 序列存 8.2G/2000 万用户，LLaTTE 的 Cached Upstream User Dense Embedding 存 7.63T）；而 SlimPer 的 $\mathcal{X}$ 从诞生起就是和某一次具体请求、具体候选 item 绑定的一次性中间激活，天生不可缓存、不可复用，"压缩"发生在**模型内部**而不是产出一个可以独立存储的"用户兴趣"artifact。

### Q18：SlimPer 的压缩表达有没有 intermediate evaluation？还是完全 end-to-end？

**完全 end-to-end，没有 intermediate evaluation。**

对比表格里其他方法：TokenMinds 用"decoder output 做检索"来评估中间表达质量、Personas 用"生成兴趣与用户点击兴趣的相似度/用户是否认可"来评估、RecGPTv3 靠人工标注判断、LLaTTE 用"upstream NE 变化 / downstream NE 变化的迁移比"来评估——**这些方法都需要 intermediate evaluation，本质原因是它们的表达是被单独产出、缓存、再喂给下游模型的，中间有一次"artifact 生成"和"下游消费"的解耦，所以需要一个独立指标先判断这份 artifact 本身好不好，否则要等一整个下游排序实验周期才能知道效果**（文档 Summary 部分自己也点出了这一点："Evaluation of intermediate representation is a weak form of downstream task"——即中间评估本质上是下游任务评估的一个弱替代品）。

SlimPer 没有这个"生成 artifact → 缓存 → 下游消费"的两阶段结构：知识库 $\mathcal{X}$ 只是排序模型内部的一个中间激活，从 $\mathcal{X}^0$ 到最终的 reshare/like/comment 概率预测都在同一次前向传播、同一套端到端反向传播里完成，唯一的监督信号就是最终的多任务 NE/交叉熵损失。因为不存在一个独立于下游任务的、可以拿出来单独打分的"压缩表达"，所以**根本不需要**也没有设计 intermediate evaluation——论文和内部材料里唯一接近"中间产出分析"的是 FIND 工具对 attention 权重的可解释性分析（Q&A 前文已讨论），但那是训练完成后的**事后诊断/归因分析**，不是训练过程中用来验证或指导表达质量的一个独立指标，和表格里其它方法的 Intermediate Performance Validation 不是同一类东西。
