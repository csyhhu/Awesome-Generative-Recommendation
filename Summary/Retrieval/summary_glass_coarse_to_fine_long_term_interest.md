# GLASS: Coarse-to-Fine Long-term Interest Modeling for Generative Recommendation — 论文总结

## 综合理解（讨论总结）

**我的理解**：
1. **基础架构**：本文基于经典的 TIGER 架构做 SID augmented Retrieval。Encoder-Decoder 架构，输入为 SID，输出为 retrieved items' SID。
2. **SID-Tier**：先使用 SID_1 embedding 计算兴趣直方图，作为 encoder 输入的最后一个 token。
3. **逐个物品解码**：decoder 依次解码 SID1…3。每生成一个 item，先生成它的第一个 SID（简称 c_0）。
4. **每一步的 attention**：decoder 的 Q 为上一轮的 decode 结果，KV 为用 c_0 从历史序列中 hard search 出来的 item embedding 做 pooling 的结果。最终产生 SID1..3。

**点评**：整体框架正确：TIGER 式 Encoder-Decoder，加上"SID-Tier 管 sid₁、SHS 管 sid₂/₃"的分工。以下几处需要修正或补充。

| 原表述 | 判断 | 修正 / 补充 |
|---|---|---|
| 基于 TIGER 的 Encoder-Decoder，输入和输出都是 SID | ✅ 正确 | 输入是短序列（最近 50 条）中每个物品的 3 个 SID token，再加上 $e_{tier}$。<br>本质是"**长期历史增强**的生成式召回"。 |
| 输出为 retrieved items' SID | ⚠️ 易混淆 | 输出是**下一个推荐物品**的 SID。<br>"retrieved"指推荐系统的召回阶段，不是 hard search 检索出的历史物品。<br>后者只作上下文，不会被输出。 |
| 用 SID_1 embedding 计算兴趣直方图 | ❌ 不准确 | 用的不是 decoder 中可学习的 sid₁ token embedding，而是**原型** $\tilde{h}_k$：全局桶内所有物品**多模态 embedding** 的均值。<br>计算方式：<br>• 原型与长期历史各物品的多模态 embedding $h_v$ 求余弦相似度；<br>• 分 tier 计数，每个码字得到一个直方图；<br>• 64 个码字的直方图拼接后过 MLP，得到 1 个 token。<br>这一步可离线预计算。 |
| 作为 encoder 输入的最后一个 token | ✅ 正确 | $O^{enc}=\text{Encoder}([E_{short}; e_{tier}])$。 |
| 每个 item 先生成 c_0，再生成后续 SID | ✅ 正确 | 一个物品占 3 步解码：BOS→ĉ₀→ĉ₁→ĉ₂。<br>外层是 beam search，同时维护 K 条候选路径。 |
| decoder 的 Q 为上一轮的 decode 结果 | ⚠️ 需细化 | Q 是对**当前物品内**已生成前缀 $[t_{bos};\hat c_0;\dots;\hat c_{q-1}]$ 做因果 Self-Attn 后的隐状态 X。<br>不是"上一个物品"的结果。<br>预测下一个码字用的是最后位置的隐状态。 |
| KV 为 hard search 出的 item embedding 的 pooling | ❌ 两处不对 | **(1) 不是 pooling。** KV 是检索物品完整 SID 的 **token 级 embedding 序列**（长度 $3\|H_{ret}\|$）。加权由 Cross-Attn 的注意力 $\alpha_j$ 按当前前缀动态完成。<br>"平均池化历史"恰恰是论文批评 DualGR 的做法。<br>**(2) 不是唯一的 KV 来源。** 每层 decoder 有**两路并行 Cross-Attn**：<br>• 短期路：KV 为 Encoder 输出（短序列 + $e_{tier}$）；<br>• 检索路：KV 为 $E_{ret}$。<br>两路经逐维门控 $\mathbf g$ 融合：$Z=(1-\mathbf g)\odot Z_{short}+\mathbf g\odot Z_{ret}$。 |
| "每一次生成中"都用 c_0 检索 | ⚠️ 需细化 | **生成 c_0 本身时还没有 key**，检索分支为空，只用短期路（长期信息来自 $e_{tier}$）。<br>检索分支只在生成 sid₂、sid₃ 时生效：每条 beam 用**自己的** ĉ₀ 检索，生成 sid₃ 时复用同一个桶，不重新检索。 |
| （未提及） | 补充 | **Sparsity-Aware Augmentation**：用户桶平均只有 $L/\|\mathcal C_0\|$ 条，长尾兴趣的桶更稀。<br>因此用两种方式补足检索上下文：<br>• **语义近邻扩充（SN）**：命中太少时并入近邻码字的历史；<br>• **压缩一级码本（CR）**：让每个桶装更多物品。<br>两者中 CR 效果更好。 |

**修正后的一句话版本**：GLASS = TIGER，加上两处改动：
- **Encoder 端**：多一个由"码字原型 × 长期历史多模态 embedding"的兴趣直方图得到的 token，用来增强 sid₁ 的预测；
- **Decoder 端**：每层改为双路 Cross-Attn 加门控。生成 sid₂/₃ 时，额外 attend 到"用该 beam 的 ĉ₀ 在长期历史中 hard search 出的同桶物品的 SID token 序列"。相当于对每条 beam 跑一次 SIM（GSU = 按 ĉ₀ 取桶，ESU = Cross-Attn）。

---

## 基本信息

- **论文标题**: Coarse-to-Fine Long-term Interest Modeling for Generative Recommendation
- **方法名**: GLASS（**G**enerative recommendation with **L**ong-term interest vi**A** **S**ID-Tier and **S**emantic Search）
- **arXiv ID**: 2602.05663
- **会议**: RecSys '26
- **作者**: Shiteng Cao, Junda She, Bin Zeng（共同一作）, Ji Liu, Chengcheng Guo, Kuo Cai, Qiang Luo, Ruiming Tang, Han Li, Kun Gai, Zhiheng Li, Cheng Yang
- **单位**: 清华大学深圳国际研究生院 / 北京邮电大学 / 快手
- **关键词**: Generative Recommendation, Long Sequence Modeling, Semantic ID, SIM

---

## 核心问题

生成式推荐（GR）把推荐建模为对层次化语义 ID（SID）的自回归生成。但：

1. **长序列难以使用**：Attention 的二次复杂度使 GR 一般只能输入短序列（如最近 50 条），丢失了用户长期历史中的丰富信号。
2. **已有方案忽视 SID 的层次性**：DualGR 等工作把 SIM（GSU+ESU 两阶段检索）引入 GR，但对所有 SID 层一视同仁。而 GR 本质是 **coarse-to-fine** 的生成：
   - **前缀（sid₁）**决定宽泛的语义类别，需要 **cluster 级偏好** 来锁定正确分支；
   - **后缀（sid₂, sid₃）**在类内定位具体物品，需要 **item 级偏好** 来区分细节。
3. **召回阶段的 query 缺位**：排序阶段的 SIM 有明确 target item 作为检索 query；召回阶段没有 target，传统长序列召回方法（ULIM、LongRetriever、MISS）又是基于固定 embedding 相似度设计的，无法直接迁移到 GR。

**核心洞察**：SID 前缀和后缀需要**不同方式**注入长期信号——
- 生成 sid₁ 时尚无路径，需**主动注入**全局长期兴趣；
- sid₁ 生成后，它本身就是一个"潜在 target"，可作为 **hard search 的 key** 去长期历史中检索相关行为。

此外作者观察到 **误差累积 / rank degradation**：vanilla GR 在生成 sid₁ 之后，ground-truth 在 beam 中的排名往往变差而非变好。

---

## 方法：GLASS 框架

整体采用 **Encoder-Decoder**（类 TIGER）架构，SID 由 RQ-VAE 对多模态 embedding 量化得到，3 层码本。

### 1. SID-Tier：增强首个 token（sid₁）的预测

利用"一级码本规模很小（如 64）"这一特性，在召回阶段也能做**用户长期历史 × 候选语义码**的交叉特征（传统召回在海量 item ID 上无法做 cross feature）。可离线预计算。

- **Prototype 对齐**：对一级码本中每个码字 k，取所有 c₀(v)=k 的物品多模态 embedding 均值作为原型 $\tilde{h}_k$，解决离散码字与连续 item embedding 的语义错位。
- **兴趣直方图**：对每个码字 k，计算 $\tilde{h}_k$ 与长期历史中每个物品的余弦相似度，把 [-1,1] 划为 N 个 tier，统计落入各 tier 的物品数，得到 $t_k \in \mathbb{R}^N$（刻画兴趣**强度分布**，而非单一均值）。
- **全局注入**：拼接所有码字的直方图 $x \in \mathbb{R}^{K_0 \cdot N}$ → MLP → $e_{tier} \in \mathbb{R}^d$，作为一个额外 token 拼到短序列 embedding 末尾送入 Encoder：
  $O^{enc}_{short} = \text{Encoder}([E_{short}; e_{tier}])$
  - **正文没写、但图 1(b) 画了 Norm**：直方图先归一化成比例（0.25、0.10…）再进 MLP。原始计数随历史长度线性增长，推测是按历史长度归一化。

> 思路与淘宝 SimTier（MAKE）一脉相承，只是把"target item"替换为"一级码字原型"，从而在召回场景可用。

### 2. Semantic Hard Search（SHS）：用生成的 sid₁ 校准后缀生成

- **ĉ₀ 是怎么解码出来的**（讨论补充）：标准 TIGER 式 Encoder-Decoder 自回归解码，起始 token 是 **BOS**（论文记作 $t_{bos}$），不是 CLS。
  - **输入**：第 q 步的 decoder 输入为 $H^{dec}_q = [t_{bos}; \hat{c}_0; \dots; \hat{c}_{q-1}]$。第 0 步只有 $t_{bos}$。
  - **单层计算**：Self-Attn（带因果 mask）→ 对 Encoder 输出 $O^{enc}_{short}$ 做 Cross-Attn → FFN。
  - **Encoder 输出**：$O^{enc}_{short}$ 已包含短序列和 $e_{tier}$，所以长期全局兴趣只能通过 SID-Tier 这条路影响 sid₁。
  - **输出头**：取 $t_{bos}$ 位置的最终隐状态 $h^{dec}_0$，经第 0 层码本的线性头得到 $\mathcal{C}_0$ 上的分布，beam search 保留 top-K 个 ĉ₀。
  - **BOS 与 CLS 的区别**：CLS 是双向编码中用来汇聚整段序列的表示；BOS 只是自回归生成的起点，它的输出直接用来预测下一个 token。
- **SHS 何时介入**：
  - **ĉ₀ 确定之后**，才能用它检索 $H_{ret}$，所以 SHS 实际只作用于 sid₂、sid₃ 的生成。
  - **第 0 步论文没写清楚**：此时检索分支为空，推测是 mask 掉，或由门控 $g \to 0$ 退化为只用短期分支。
  - **训练时论文也没明说**：teacher forcing 下推测用 ground-truth $c_0$ 做检索。
  - **推理时**：beam 中每条路径的 ĉ₀ 不同，需要**按 beam 分别检索**，得到各自的 $E_{ret}$，再做 Cross-Attn。
  - **Encoder 只算一次**：Encoder 输出在所有 beam 间共享。
  - 另外，原文写的 "$t_{bos}=H_q^{dec}$" 是笔误。
- **语义检索**：解码生成 $\hat{c}_0$ 后，以其为 key 在长期历史中做 hard search，取所有一级码相同的历史物品：
  $H_{ret} = \{c(v_i) \mid v_i \in H_{long}, c_0(v_i) = \hat{c}_0\}$
  相当于给 decoder 提供一个按显式语义意图过滤的 RAG 上下文窗口，噪声远小于稠密相似度检索。
  - **符号说明**（讨论补充）：
    - $\mathcal{C}_0$ 是第一层**码本**，即码字集合，例如 64 个码字。
    - $c_0(v)$ 是物品 v 的第一层**码字**，也就是 sid₁。论文的 $c_j$ 从 0 开始编号，$sid_t$ 从 1 开始，所以 $c_0 = sid_1$。
    - 稀疏性小节里写的 $|\mathcal{C}_1|=128$ 也是指第一层，是论文的符号不一致。
    - **桶（bucket）**：指共享同一个一级码字 k 的物品集合，论文稀疏性小节称之为 "SID bucket"，在 RQ-VAE 中就是第一层聚类的一个簇。
      - **全局桶**：$\mathcal{V}_k = \{v \in \mathcal{V} \mid c_0(v) = k\}$，物品库中所有一级码字为 k 的物品，共 $|\mathcal{C}_0|$ 个桶。SID-Tier 的原型 $\tilde{h}_k$ 就是在全局桶上做均值得到的。
      - **用户桶**：$\mathcal{B}_k(u) = H_{long}^{id} \cap \mathcal{V}_k$，即用户 u 长期历史中落在第 k 个桶里的物品。
      - **与检索的关系**：SHS 的检索结果就是用户桶，$H_{ret} = c\big(\mathcal{B}_{\hat{c}_0}(u)\big)$。"每个 sid₁ 平均命中 7.8 条"说的是用户桶的平均大小 $L/|\mathcal{C}_0|$。
  - **ĉ₀ 本身就是 sid₁，不是 item ID**：decoder 每一步输出的都是 SID token。
  - **全程不需要"先得 item ID 再查 SID"**：历史物品的 SID 是 tokenizer 离线算好的，检索就是逐条比较"历史物品的 $c_0(v_i)$ 是否等于 ĉ₀"，属于整数相等匹配（可预先建倒排：码字 → 历史物品列表）。
  - **item ID 只在最后出现**：三层 SID 全部解码完后，才通过 SID→item 映射表得到 item ID。
  - **解码过程中如何使用过滤后的长期历史**（讨论补充，beam size 为 K）：
    外层就是普通的 beam search，SHS 没有改搜索流程，改的是每一步的**打分函数** $P(c_q \mid \cdot)$。过滤后的历史只充当 Cross-Attn 的 K/V（RAG 式上下文）。它不是候选，不会被直接输出，也不限制输出空间，论文中没有 copy/pointer 机制。

    **① Step 0：生成 sid₁，此时不用检索。** decoder 输入为 $[t_{bos}]$，检索分支为空，只依赖 Encoder 输出（短序列和 $e_{tier}$）：
    $$s_0^{(b)} = \log P\big(\hat{c}_0^{(b)} \mid H_{short}, e_{tier}\big),\quad b=1,\dots,K$$
    在 $\mathcal{C}_0$ 上保留 top-K 个 $\hat{c}_0^{(b)}$，形成 K 条 beam。

    **② 按 beam 做 hard search，取出该 beam 对应的桶。** 每条 beam 以自己的 $\hat{c}_0^{(b)}$ 为 key，从用户长期历史中取出同一桶的物品，并 Lookup 成 embedding 序列：
    $$H_{ret}^{(b)} = \big\{c(v) \mid v \in \mathcal{B}_{\hat{c}_0^{(b)}}(u)\big\},\qquad E_{ret}^{(b)} = \text{Lookup}\big(H_{ret}^{(b)}\big) = [\mathbf{e}_1;\dots;\mathbf{e}_M] \in \mathbb{R}^{M\times d},\ M = 3|H_{ret}^{(b)}|$$
    其中每个检索到的物品贡献 sid₁、sid₂、sid₃ 三个 token。
    - ĉ₀ 相同的 beam 共享检索结果，最多只需 $\min(K,|\mathcal{C}_0|)$ 次检索。
    - 各 beam 的 M 不同，batch 化时需要 padding + mask。

    **③ Step 1：生成 sid₂，过滤后的历史在这一步起作用。** decoder 输入为 $[t_{bos}; \hat{c}_0^{(b)}]$。每层先做 Self-Attn 得到 $X$，再并行做两路 Cross-Attn。检索分支为：
    $$\mathbf{Z}_{ret}^{(b)} = \text{softmax}\Big(\frac{X W_Q (E_{ret}^{(b)} W_K)^\top}{\sqrt{d}}\Big) E_{ret}^{(b)} W_V \;=\; \sum_{j=1}^{M} \alpha_j\, \mathbf{e}_j W_V$$
    也就是说，$\mathbf{Z}_{ret}$ 是**桶内历史码字 embedding 的加权和**，权重 $\alpha_j$ 由当前 beam 状态（编码了 BOS→ĉ₀）决定，表示"用户在这个大类下历史上偏好哪些细分码"。之后与短期视图门控融合：
    $$\mathbf{Z}_{context} = (1-\mathbf{g}) \odot \mathbf{Z}_{short} + \mathbf{g} \odot \mathbf{Z}_{ret}^{(b)}$$
    经 FFN、堆叠 L 层后，取最后位置的隐状态 $\mathbf{h}_1$，用 $\mathcal{C}_1$ 的线性头 $W^{(1)}$ 打分：
    $$P\big(c_1 = a \mid \hat{c}_0^{(b)}, H_{short}, H_{ret}^{(b)}\big) = \text{softmax}\big(W^{(1)}\mathbf{h}_1\big)_a$$
    - **为什么能抬高"历史上偏好的细分码"**：忽略 FFN 和残差做线性近似，码字 a 的 logit 中来自检索分支的贡献约为
      $$\Delta\text{logit}_a \approx \mathbf{w}_a^\top\Big(\mathbf{g} \odot \sum_j \alpha_j\, \mathbf{e}_j W_V\Big)$$
      这相当于桶内历史码字对候选码字 a 的一次**加权投票**。与历史中高频 sid₂ 语义相近的码字 logit 被抬高。门控 $\mathbf{g}$ 决定这次投票的话语权：桶内历史少或噪声大时 $\mathbf{g}$ 小，退回短期分支。
    - **beam 得分**：累加为 $s_1^{(b,a)} = s_0^{(b)} + \log P(c_1=a\mid\cdot)$，在 $K\times|\mathcal{C}_1|$ 个候选中重新选 top-K。

    **④ Step 2：生成 sid₃，复用同一桶。** decoder 输入为 $[t_{bos}; \hat{c}_0; \hat{c}_1]$，key 仍是该 beam 的 ĉ₀，所以 $E_{ret}^{(b)}$ 不变，不重新检索。
    - **注意力会软性收窄（推断）**：此时 query 同时编码了 ĉ₀ 和 ĉ₁，注意力权重 $\alpha_j$ 可以自然地集中到桶内 sid₂ 与 ĉ₁ 相同的历史物品上。虽然没有用 (sid₁, sid₂) 做硬检索，但能通过注意力实现**软性收窄**，同时避免了二级桶过稀的问题。
    - 打分后累加得分 $s_2 = s_1 + \log P(c_2 \mid \cdot)$，选出最终 top-K 条完整 SID。完整路径概率即论文 RQ4 中的式子：
      $$P(\text{path}\mid\cdot) = \underbrace{P(sid_1 \mid H_{short}, e_{tier})}_{\text{无检索}}\cdot \prod_{t=2}^{3} P\big(sid_t \mid sid_{<t},\ \underbrace{H_{short}, H_{ret}(sid_1)}_{\mathcal{C}\ \text{随 sid}_1\ \text{动态扩充}}\big)$$

    **⑤ 映射回物品。** 通过 SID→item 映射得到推荐列表。不对应真实物品的无效 SID 需要过滤，论文未提及如何处理。

    - **例子**：
      - **设定**：ĉ₀=17 对应"篮球"，用户该桶内有 30 条历史，其中 20 条 sid₂=5（NBA 集锦），8 条 sid₂=88（街球）。短序列中最近没有篮球行为。
      - **无 SHS**：$P(sid_2)$ 只能依赖短序列，接近大盘热门分布。
      - **有 SHS**：$\sum_j \alpha_j \mathbf{e}_j$ 偏向码字 5、88 的 embedding，它们的 logit 被抬高。生成 sid₃ 时，若 ĉ₁=5，注意力进一步聚焦到那 20 条历史上。
    - **两层影响**：
      - **beam 内**：把 sid₂、sid₃ 的分布从"大盘先验"拉向"该用户在该大类下的细粒度偏好"。
      - **beam 间**：得分是累积 log-prob。某个大类下历史丰富且偏好一致时，后续分布更尖锐，路径得分更高，更容易在 top-K 剪枝中存活。历史稀疏的大类则因 $\mathbf{g}$ 小而影响有限。
      - 因此 SHS 也间接决定了不同大类之间的最终排序，CRP 实验中 rank degradation 减轻就体现在这里。
    - **与 SIM 的对应关系**：
      - beam 前缀 ĉ₀ 对应 SIM 中的 **target 类目**；
      - 按 ĉ₀ 过滤历史（取桶）对应 **GSU 的 hard search**；
      - decoder 状态对 $E_{ret}$ 做 Cross-Attn 对应 **ESU 的 target attention**；
      - 本质上是**对每条 beam 跑一次 SIM**，用 beam 前缀解决召回阶段没有 target 的问题。
  - **训练细节（论文未说明，属推断）**：
    - **训练方式**：teacher forcing 下，`[BOS, c₀, c₁]` 三个位置并行计算，$H_{ret}$ 用 ground-truth $c_0$ 检索。
    - **必须 mask BOS 位置的检索分支**：BOS 位置负责预测 $c_0$，若它能看到按 GT $c_0$ 检索出的 $E_{ret}$，就会**泄露标签**。所以只能让 $c_0$、$c_1$ 两个位置（预测 sid₂、sid₃）attend 到 $E_{ret}$。
    - **训练与推理的检索 key 不一致**：训练用 GT $c_0$ 检索，推理用预测的 ĉ₀ 检索。ĉ₀ 预测错误的 beam 会拿到"错误大类"的历史，但这条 beam 本身也是错误路径，所以影响有限。
    - **其他未交代的实现细节**：$E_{ret}$ 是否加位置或时间 embedding、是否按时间排序、长度是否截断，论文均未说明。
- **双路 Cross-Attention + 自适应门控融合**：
  - **位置**（讨论补充）：在 **Decoder 的每一层**中，用它替换标准 Transformer decoder 原有的单路 Cross-Attn。Encoder 不变。
  - **单层结构**：Self-Attn → [两路并行 Cross-Attn → Gate] → FFN，共堆叠 L 层（实验中为 4 层）。
  - **作用范围**：decoder 的每个位置都会经过这个模块，但只有 ĉ₀ 确定后（预测 sid₂、sid₃ 时）检索分支才有内容。
  - $Z_{short} = \text{CrossAttn}_\theta(X, O^{enc}_{short})$：短期视图（即时意图）
  - $Z_{ret} = \text{CrossAttn}_\phi(X, E_{ret})$：检索视图（与当前语义相关的长期偏好）
    - **$E_{ret}$ 不经过 Encoder**：它直接对检索到的 SID 做 embedding Lookup。
    - **长度**：每个检索到的物品贡献 3 个码字 token，所以 $E_{ret}$ 长度为 $3|H_{ret}|$。
    - **正文与图 1(c) 不一致**：图中对 Item1/Item3/Item40 做 Lookup，且该表标 ❄（冻结），Decoder 输入的 Lookup 标 🔥。可能检索分支实际用的是冻结的 item 级 embedding（长度 $|H_{ret}|$）。作者的匿名代码链接已过期（repository_expired），无法核实。
  - $g = \sigma([Z_{short}; Z_{ret}]W_g)$，$Z_{context} = (1-g)\odot Z_{short} + g \odot Z_{ret}$
    - 论文正文称 g 为 "scalar"，但 $W_g \in \mathbb{R}^{2d\times d}$，实际是**逐维门控向量**。
  - 门控相当于"自校验"模块：检索结果少/噪时自动退回短期分支。区别于 DualGR 在输入层用路由（DBR）做硬选择。
- 之后接 FFN、堆叠 L 层，按标准 NLL 训练各层 codeword。
- **各层 SID 的使用情况**（讨论补充）：本文 SID 只有 3 层（m=3，码本 [64,128,128]），不存在 sid₄。
  - **检索 key**：只用 sid₁。
  - **检索内容**：检索到的历史物品的**完整 SID**（sid₁、sid₂、sid₃）都作为 $E_{ret}$ 的 token 参与 Cross-Attn。所以 sid₂、sid₃ 被当作"内容"用上了，只是没有当作 key。
  - **检索集不随解码收窄**：生成 sid₂ 和 sid₃ 时用的是同一个 $H_{ret}$，没有在 ĉ₁ 确定后用 (sid₁, sid₂) 进一步缩小检索范围。
  - **为什么不用前两层做 key**：前两层组合有 64×128 个桶，1000 条历史平均每桶约 0.1 条，稀疏性会更严重。第 3 节的稀疏性问题本身就说明了这一点。
  - **SID-Tier 不用 SID 本身**：它只用一级码字原型和历史物品的**原始多模态 embedding** $h_v$。

### 3. Sparsity-Aware Augmentation：解决检索结果过稀

问题：长度 1000 的序列、一级码本 128 时，每个 sid₁ 平均只命中约 7.8 条历史，信息增益很小。两种方案见下文。

> **讨论补充：7.8 只是均匀假设下的理论均值 L/K，不代表码本分布均匀。**论文原文就写的是 "Theoretically"。
> - **单个用户的历史分布高度偏斜**：用户兴趣集中，少数桶有几十上百条，大多数桶为 0。
> - **全局码本利用率也不均匀**：RQ-VAE 常见码本坍缩和长尾问题。
> - **偏斜带来两种结果**：
>   - 生成的 ĉ₀ 多是用户主兴趣，往往命中稠密桶，SHS 收益较大；
>   - 命中长尾或探索性兴趣时，检索结果很稀，甚至为空。
> - **这正是 SN、CR 和自适应门控要处理的情况**。RQ3 中"门控值随检索长度上升"也间接说明，实际检索长度的方差很大。
> - **实验配置下的均值**：一级码本为 64、平均历史长度约 987，理论均值约 15 条。
>
> 两种方案：

- **Semantic Neighbor Augmentation（SN，松弛匹配）**：基于 RQ 码字 embedding 预计算每个一级码的 top-k 近邻；当 $|R(sid_1)| < \tau$ 时，把近邻码对应的历史物品并入检索集。
- **Codebook Resizing（CR）**：训练时压缩一级码本大小，让每个桶装更多物品。作者指出，常规观点认为改码本大小只是在层间重新分配预测难度，但在 hard search 框架中这会带来质变——"**一级负责密度，后续层负责精度**"。

---

## 实验

### 设置
- **数据集**：TAOBAO-MM（18.9 万用户，1 万物品，平均历史长度 987）、KuaiRec（3790 用户，10728 物品，平均历史长度 884）。仅保留正反馈；最近 50 条为短序列，其余为长期历史。
- **配置**：embedding 维度 96，码本 [64,128,128]，SID-Tier MLP 256，4 层 encoder/decoder。
- **基线**：ID 类（Caser、HGN、GRU4Rec、SASRec、Bert4Rec、S³-Rec、HSTU）；SID 类（TIGER、DualGR）。

### 主结果（RQ1）

| 数据集 | 指标 | 最强基线 (DualGR) | GLASS | 相对提升 |
|---|---|---|---|---|
| Taobao-MM | R@1 | 0.0306 | 0.0372 | +21.6% |
| Taobao-MM | R@10 | 0.1126 | 0.1284 | +14.0% |
| Taobao-MM | N@3 | 0.0448 (TIGER) | 0.0582 | +29.9% |
| Taobao-MM | N@20 | 0.0738 | 0.0893 | +21.0% |
| KuaiRec | R@10 | 0.1292 | 0.1345 | +4.1% |
| KuaiRec | N@10 | 0.0821 | 0.0855 | +4.1% |

- 在 Taobao-MM 上提升更大，作者归因于其多模态特征（监督对比学习得到）质量更高；KuaiRec 多模态信息仅来自简单文本编码。
- 相对 DualGR 的优势：(1) SID-Tier 做了 target-aware 的交叉，而非平均池化；(2) 双路 cross-attn + gate 融合短/长期，而非单路 cross-attn。

### 消融（RQ2）
- **SID-Tier**：Recall@5 从 0.0702 → 0.0842；teacher-forcing 下 acc₁ 0.2410 → 0.2485，推理时 P₁（首码 top-1 正确率）0.1470 → 0.1587——印证前缀依赖：首码准了，整体就准。
- **SHS**：主要提升 acc₂ 和 P₂；两者结合在 Recall@10 / NDCG@10 上最好，呈互补关系。
- **稀疏增强**：CR 效果最显著（Recall@10 峰值来自 GLASS(mode=CR)）；SN 在无 CR 时有效，但与 CR 叠加时略降——码本已经很粗时再引入邻居会模糊兴趣边界、引入噪声。
  - 读图近似值（R@10 / N@10）：CR 0.128 / 0.078，SN 0.127 / 0.078，CR+SN 0.124 / 0.066，None 0.112 / 0.078。CR+SN 在 NDCG 上掉得尤其明显。
  - 从门控实验的两组配置推断：CR 对应一级码本 64（[64,128,128]，即主配置），不做 CR 对应 [128,128,128]。

### 门控分析（RQ3）
- 固定权重（0.5）在一级码本小的时候有效，码本变细后反而掉点：检索序列短时信噪比低，噪声会直接扭曲 beam search 的累积概率。
- 学到的平均门控值随检索序列长度单调上升（[128,128,128] 和 [64,128,128] 均如此）——模型学会"检索越多越信任长期分支"。
- **但变化幅度很小**（读图）：检索长度 0→30，平均 g 只从 0.576 升到 0.581（码本 128），或从 0.532 升到 0.540（码本 64）。检索为空时 g 仍有 0.53~0.58，论文没说空检索时 $Z_{ret}$ 是什么。这只是对 d 维取平均，个别维度可能变化更大；但仅从图看，"按检索质量动态开关"的效果比文字描述弱。

### SHS 的理论解释（RQ4）
- 提出 **Conditional Rank Progression（CRP）**：在 ground-truth 前缀存活到 d-1 层的样本上，统计第 d 层 ground-truth 前缀在 beam 中的平均排名。
- Beam=20 时，vanilla GR 第 2 层 CRP 上升 1.25，SHS 仅 0.98（缓解 22%）；第 3 层从 0.19 降到 0.13（缓解 31%）。
- 解释：$P(path) = P(sid_1|ctx)\prod_{t\ge2} P(sid_t|sid_{<t}, \mathcal{C})$，baseline 中 $\mathcal{C}$ 是静态的，无法在细粒度层拉开 GT 与干扰项的概率差；SHS 在 t>1 时动态扩充 $\mathcal{C}$ 为 {context, context'}，用个性化细粒度偏好校准转移概率。

### 线上 A/B（RQ5）
- 快手短视频平台，实验组/对照组各 3%（约 1200 万用户），两周。
- App 总使用时长 **+0.056%**，人均使用时长 **+0.052%**（95% 置信），召回率绝对提升 **3 个百分点**。

---

## 个人点评 / Takeaways

1. **"前缀与后缀区别对待"是核心贡献**：sid₁ 用全局兴趣直方图（无 query 的 cross feature），sid₂/₃ 用 sid₁ 作 query 做 hard search——巧妙解决了召回阶段 SIM 没有 target 的问题：**GR 的已生成前缀天然就是 target 的粗粒度代理**。
2. **SID-Tier = 码本版 SimTier**：小码本使"用户 × 所有候选类别"的交叉特征变得可枚举，是 SID 范式下做召回交叉特征的一个通用思路。
3. **码本大小在 hard search 下有了新含义**：一级码本不再只是"预测难度分配"，还决定检索桶的密度。这对设计面向长序列的 SID tokenizer 有指导意义（与 [[summary_semantic_id_prefix_ngram_stability]] 等关注前缀结构的工作可对照）。
4. **局限**：
   - 公开数据规模较小（物品仅 1 万量级，KuaiRec 仅 3790 用户），ID 基线配置偏弱；
   - SHS 只用 sid₁ 作 key，未探索用 sid₁+sid₂ 做更细粒度的逐层检索；
   - Hard search 需要在每条 beam 上分别检索并做 cross-attn，推理开销与 beam 大小相关，论文未详细报告线上延迟；
   - 线上收益（+0.05% 时长）相对温和；
   - 门控的平均值几乎不随检索长度变化（变化 < 0.01）；$E_{ret}$ 的实现正文与图不一致，代码链接已失效。

> **HTML 插图**：[glass-coarse-to-fine-long-term-interest.html](https://www.internalfb.com/collab-files/view/1aovkdM6ctzaz_ryjE-w_iAnTThx4s60k)
