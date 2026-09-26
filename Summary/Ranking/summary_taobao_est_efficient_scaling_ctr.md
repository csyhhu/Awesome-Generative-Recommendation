# EST: Towards Efficient Scaling Laws in Click-Through Rate Prediction via Unified Modeling

- **机构**: 阿里巴巴（淘宝天猫集团）
- **arXiv**: 2602.10811
- **关键词**: CTR 预测、Scaling Law、统一建模（Unified Modeling）

## 综合理解（用户提出，基本正确，有两处需澄清）

> 本文对于给定的非行为特征/用户短期行为序列/候选相关行为序列，进入建模时，为了减少计算量，只计算非行为序列对行为序列的 target attention，更新非行为序列的表达。对于行为序列的表达，提前使用对应的多模态内容序列计算自相关矩阵（self-attention matrix），并用该矩阵多次聚合行为序列。

**总体判断：方向正确**，抓住了 EST 的两个核心模块（LCA 只算 N→B 单向交叉注意力更新 N；CSA 用内容特征算好的固定矩阵反复聚合 B），但有两处表述需要精确化：

1. **"target attention" 用词需澄清**：LCA 与传统 target attention（DIN/TWIN/MUSE 等，用**单个候选 item embedding**作为唯一 query、对行为序列做一次 attention 后**池化成定长向量**，属于"早期聚合"预处理步骤）并不完全一样——LCA 是用**全部非行为 token $\mathbf{N}$**（不止候选 item，还包括用户画像等多种特征）作为多个 query，每个 query 有**独立的投影参数**，且**不对行为序列做池化压缩**（$\mathbf{B}$ 整体保持 token 序列形态），并且这一 cross-attention 在 **每一层**都重新计算一次（而不是只做一次早期聚合）。详见下文"LCA vs 传统 Target Attention"。
2. **"自相关矩阵"的计算对象是内容特征而非行为 token 本身**：$\mathbf{G}_\gamma=\mathbf{M}_\gamma\mathbf{M}_\gamma^\top$ 是基于**冻结的多模态内容特征** $\mathbf{M}_\gamma$（而不是行为 ID token $\mathbf{B}_\gamma$）算出的相似度矩阵；算出后**只算一次**（可与其它特征处理并行预计算），但会被**每一层复用**，去聚合当层的 ID token 表达 $\mathbf{B}_\gamma^{(s-1)}$，因此确实是"用同一个矩阵在多层上反复聚合演化中的行为表达"，这点理解正确。

## 背景与动机

受 LLM scaling law 成功的启发，工业界近年积极探索 CTR 模型的可扩展架构（WuKong、RankMixer、HiFormer、MTGR、OneTrans 等）。但工业 CTR 场景有严格的算力/延迟约束（单次请求要在毫秒级对上千候选打分），而标准 Transformer 的计算量随模型规模和**序列长度**同时增长，导致直接照搬 LLM 式全量 self-attention 不可行。

CTR 模型的输入通常包含三类：
- **非行为特征** $\mathcal{N}$（用户画像、候选 item 属性等，数量少、信息密度高）；
- **用户短期行为序列** $\mathcal{B}_u$；
- **候选相关行为序列** $\mathcal{B}_c$（从超长生命周期行为中，通过 GSU/MUSE 等方法动态检索出与候选 item 相关的子序列）。

现有两类做法都存在信息瓶颈：
1. **层次化建模（Hierarchical Modeling）**（WuKong、RankMixer、HiFormer 等）：用 DIN/LONGER 等模块把行为序列先压缩成定长向量再与非行为特征联合建模，属于"早期聚合"，丢失了细粒度、token 级别的信号；
2. **（部分）统一建模（Unified/Partially-Unified Modeling）**（MTGR、OneTrans）：把用户行为 $\mathcal{B}_u$ 与非行为特征 $\mathcal{N}$ 放进同一序列联合建模，并利用"用户侧计算可在同一请求的多个候选间复用"来提效；但**候选相关行为 $\mathcal{B}_c$ 因无法跨候选复用，仍被池化压缩成定长摘要**，造成信息损失，因此只是"部分统一"。

本文提出 **EST（Efficiently Scalable Transformer）**，目标是实现**完全统一建模**——把所有原始输入（$\mathcal{N}$、$\mathcal{B}_u$、$\mathcal{B}_c$ 及其内容特征）不经有损聚合地组织进单一序列，同时通过架构设计保证计算效率。

**Q: 本文的重点是内容特征的引入还是三种特征（近期行为/长期行为/非行为）的融合？** 从标题（"Efficient Scaling Laws...via Unified Modeling"）、摘要和引言的行文看，**核心贡献是"三种特征的完全统一建模"这一架构问题**（如何在算力约束下把 $\mathcal{N}$、$\mathcal{B}_u$、$\mathcal{B}_c$ 无损地放进同一序列并保持可扩展性），对应两条洞察中的**洞察一（信息密度不对称）→ LCA** 是解决这一核心问题的主模块（决定了能否做到"完全统一"而不牺牲效率）。内容特征的处理（洞察二 → CSA）是**服务于同一目标的第二个、相对次要的模块**——用于在已经统一的行为序列内部进一步引入内容语义、补足 ID embedding 的信息，属于锦上添花而非论文的立论出发点；消融实验也显示 LCA 独立贡献了 72% 的算力节省（性能几乎不掉），而 CSA 在此基础上只带来 +0.14% 的增量提升。因此，**三种特征的统一建模（尤其是候选相关行为 $\mathcal{B}_c$ 不再被池化压缩）才是本文的重点和主要创新点，内容特征的接入方式是为支撑这一统一框架而设计的配套技术**。

## 两条关键洞察

论文先分析 CTR 输入与 LLM token 序列的本质区别，得到两条指导架构设计的洞察：

### 洞察一：信息密度的不对称性（Asymmetry in Information Density）

与 LLM 中相对同质的 token 序列不同，CTR 输入中非行为特征 $\mathcal{N}$ **数量少、信息密度高**（类似 query），行为序列 $\mathcal{B}$ **数量大、单个 token 信息密度低**（类似冗长的 context）。

**实验验证**：将行为 token $\mathbf{B}$ 与非行为 token $\mathbf{N}$ 拼成一条序列做标准 self-attention，可视化注意力矩阵并按 query-key 类型分块（$\langle\mathbf{N},\mathbf{B}\rangle$、$\langle\mathbf{B},\mathbf{B}\rangle$ 等）：
- 用 **effective rank**（$\text{erank}(\mathbf{H})=\frac{1}{\max(m,n)}\|\mathbf{H}\|_F^2/\|\mathbf{H}\|_2^2$，值越高说明信息越多样、冗余越少）衡量各分块，发现只有 $\langle\mathbf{N},\mathbf{B}\rangle$（非行为特征作为 query，行为序列作为 key/value）分块的 erank 明显更高，其余分块普遍出现 **rank collapse**（不同 query 的注意力分布高度同质化）；
- **分块掩码实验**：屏蔽 $\langle\mathbf{B},\mathbf{B}\rangle$、$\langle\mathbf{B},\mathbf{N}\rangle$、$\langle\mathbf{N},\mathbf{N}\rangle$ 三个分块，性能几乎不变（ΔAUC/ΔGAUC 约 -0.01%~-0.03%）；而屏蔽 $\langle\mathbf{N},\mathbf{B}\rangle$ 分块则显著掉点（-0.20%/-0.20%）。

**结论**：真正有信息量的交互是"非行为特征查询行为序列"这一方向，行为-行为、行为-非行为（反向）、非行为-非行为等交互大多冗余。这直接启发了 **Lightweight Cross-Attention (LCA)**：只保留高价值的 $\langle\mathbf{N},\mathbf{B}\rangle$ 交互，砍掉其余冗余的自注意力计算。

### 洞察二：模态特定先验（Modality-specific Priors）

行为序列中的 item 往往是"离散 ID + 图像/文本等内容特征"的多模态记录。直接把内容特征当作普通 token embedding 融入模型效果有限；更有效的方式是把内容特征当作**相似度先验（relational prior）**去引导 ID 空间内的交互（呼应 Courier、SimTier 等前作发现）。

**实验对比三种多模态接入方式**（ΔAUC/ΔGAUC，相对纯 ID 模型）：

| 方法 | ΔAUC | ΔGAUC |
|---|---|---|
| Side Information（内容特征与 ID embedding 拼接） | +0.44% | +0.89% |
| Semantic ID（内容特征离散化为聚类语义 ID，当类别特征用） | +0.47% | +0.64% |
| **SimTier**（用内容特征计算目标与历史行为的相似度分布） | **+1.18%** | **+1.58%** |
| SimTier + Side Information（组合） | +0.95% | +1.47% |

SimTier（相似度先验）显著优于直接拼接类方法；有趣的是 SimTier 与 Side Information 组合反而比单用 SimTier 略差，说明不同接入范式在同一优化目标下可能相互干扰，难以协调。**结论**：内容特征需要专门的利用机制（相似度关系），而非简单 token 级拼接。这启发了 **Content Sparse Attention (CSA)**：用内容相似度动态引导行为序列内部的稀疏交互。

**Q: 本文如何处理内容特征？** 具体做法（见下文 CSA 一节）：对行为序列 $\mathcal{B}_u,\mathcal{B}_c$ 中每个 item，查表得到其**预训练、冻结**的内容特征 $\mathbf{M}_u,\mathbf{M}_c$（训练时不更新，避免高维稠密向量与离散 ID embedding 联合优化的不稳定性、保留预训练语义）。内容特征**不会**被当作普通 token 直接拼进统一序列参与可学习的注意力/FFN 计算，而是仅用来计算行为序列内部的相似度矩阵 $\mathbf{G}_\gamma=\mathbf{M}_\gamma\mathbf{M}_\gamma^\top$，作为**无需训练、无反向传播**的固定注意力权重，去聚合序列内的 **ID token**（$\mathbf{O}_{\mathbf{B}_\gamma}=\mathbf{G}_\gamma\mathbf{B}_\gamma$，即 CSA），并做逐行 top-K（$K=5$）稀疏化以把复杂度降到线性。换言之，内容特征的角色是"相似度先验（relational prior）"，用来引导 ID 空间内的交互，而不是作为独立的语义 token 参与建模——这一选择正是上面消融实验（Side Information/Semantic ID/SimTier 对比）驱动的。

## 方法：EST 架构

EST 将输入 $\mathcal{X}=(\mathcal{B}_u, \mathcal{B}_c, \mathcal{N})$ 通过特征专属 tokenizer 映射为统一维度 $d$ 的 token：
- **非行为 token** $\mathbf{n}_i = \text{MLP}_i(n_i)$，逐特征独立 MLP（不聚合），拼成 $\mathbf{N}\in\mathbb{R}^{L_N\times d}$；支持后续新增特征时 warm-start 即插即用，无需改架构；
- **行为 token**：$\mathcal{B}_u$、$\mathcal{B}_c$ 各自用专属 tokenizer 把每个 item 的 ID embedding 映射为 token，拼接得到 $\mathbf{B}=[\mathbf{B}_u,\mathbf{B}_c]\in\mathbb{R}^{L_B\times d}$；
- **内容特征**：对行为序列和候选 item 的内容特征（预训练、冻结）做查表，得到 $\mathbf{M}_u,\mathbf{M}_c$，训练时不更新（避免高维稠密向量与离散 ID embedding 联合优化的不稳定性，保留预训练语义）。

每层 EST block 的计算：

$$\mathbf{N}^{(s)} \leftarrow \text{LCA}(\text{Norm}(\mathbf{N}^{(s-1)}), \text{Norm}(\mathbf{B}^{(s-1)})) + \mathbf{N}^{(s-1)}$$
$$\mathbf{B}^{(s)} \leftarrow \text{CSA}(\text{Norm}(\mathbf{B}^{(s-1)})) + \mathbf{B}^{(s-1)}$$

**Q: 每一层都会同时进行 LCA 和 CSA 吗？** 是的。论文原文明确指出 backbone 的每一层"computes the proposed LCA and CSA in parallel"：在第 $s$ 层，LCA 用上一层的 $\mathbf{N}^{(s-1)}$ 和 $\mathbf{B}^{(s-1)}$ 更新 $\mathbf{N}^{(s)}$，CSA 用上一层的 $\mathbf{B}^{(s-1)}$ 更新 $\mathbf{B}^{(s)}$，两者**并行读取同一层输入、各自独立更新各自的输出**（LCA 不改 $\mathbf{B}$，CSA 不改 $\mathbf{N}$），互不依赖先后顺序；之后再各自过对应的 FFN。这一 LCA+CSA 组合在 $S$ 层中重复堆叠，两个序列的表达逐层演化，直到最后一层才对行为 token 做 mean-pooling 与非行为 token 拼接输出。

之后分别过 FFN（非行为 token 用 **逐 token 独立的 FFN**，行为 token 用**共享 FFN**）。堆叠 $S$ 层后，行为 token 做 mean-pooling 后与非行为 token 拼接，送入 MLP 输出点击概率，用交叉熵训练。

### Lightweight Cross-Attention (LCA)

只建模非行为 token（作为 query）与行为 token（作为 key/value）之间的交互：

$$\text{LCA}(\mathbf{N},\mathbf{B}) = \phi\Big(\text{Softmax}\big(\tfrac{\mathbf{Q_N}\mathbf{K_B}^\top}{\sqrt{d}}\big)\mathbf{V_B}, \mathbf{W}^O\Big)$$

其中 $\mathbf{W}^Q, \mathbf{W}^O$ 是**逐 token 独立参数**（每个非行为特征有自己的投影，捕捉特征专属语义），而 $\mathbf{W}^K,\mathbf{W}^V$ 在所有行为 token 间**共享**（保证可扩展性）。

**工程优势**：用户侧行为序列 $\mathbf{B}_u$ 与候选 item 无关，其 $\mathbf{B}_u\mathbf{W}^K, \mathbf{B}_u\mathbf{W}^V$ 可在一次请求中对所有候选**只算一次、复用**（user-candidate 解耦计算），显著降低推理延迟。

**复杂度对比标准 self-attention**：
- 投影：$O((L_N+L_B)d^2) \to O(L_Nd^2+L_Bd^2)$；
- 注意力：$O((L_N+L_B)^2d) \to O(L_NL_Bd)$（从平方降为双线性）。

由于工业场景 $L_B\gg L_N$，这一改动让模型能处理更长行为序列而不超延迟预算。

**Q: LCA 和普通的 Target Attention 有什么区别？** 传统 target attention（DIN、TWIN、MUSE 等相关工作里所用的机制）通常是：以**单个候选 item 的 embedding 作为唯一 query**，对行为序列做一次 attention，然后把结果**池化成一个定长向量**，作为"早期聚合"步骤在进入主干网络前把行为序列压缩掉——序列信息在此之后就丢失了，且这一操作通常只做一次。LCA 与之有三点本质区别：
1. **Query 不是单个候选 embedding，而是全部非行为 token $\mathbf{N}$**（用户画像、候选属性等所有特征各自成为一个 query），且每个非行为 token 有**逐 token 独立**的 $\mathbf{W}^Q,\mathbf{W}^O$（各自学习专属的查询/输出投影），而不是像传统 target attention 那样只有一套共享参数；
2. **不做池化压缩**：LCA 只更新 $\mathbf{N}$，行为序列 $\mathbf{B}$ 本身在 LCA 这一步保持原始 token 序列形态（不被压缩成向量），真正的降维只发生在堆叠完全部 $S$ 层之后的最终 mean-pooling；
3. **在每一层都重新计算**（见下文"每层是否都做 LCA+CSA"），而非只做一次性的早期聚合预处理，因此非行为特征对行为序列的关注会随层数加深而逐步精炼，形成深层交互；
4. **工程上支持 user-candidate 解耦复用**：$\mathbf{B}_u\mathbf{W}^K,\mathbf{B}_u\mathbf{W}^V$ 可在一次请求中对所有候选只算一次复用，这是把"计算复用"显式设计进 attention 机制、而不仅仅依赖池化来省算力。

简言之，LCA 可以理解为"把传统 target attention 中单一候选 query 泛化为多个（带专属参数的）非行为特征 query，且去掉池化环节、嵌入到多层可扩展 Transformer 中反复执行"的轻量级交叉注意力，而非传统意义上一次性生成用户兴趣向量的 target attention。

### Content Sparse Attention (CSA)

利用预训练冻结的内容特征 $\mathbf{M}_\gamma$（$\gamma$ 为 $\mathcal{B}_u$ 或 $\mathcal{B}_c$）计算序列内部相似度矩阵 $\mathbf{G}_\gamma=\mathbf{M}_\gamma\mathbf{M}_\gamma^\top$，作为**无需训练、无需反向传播**的固定注意力矩阵去聚合行为序列内的 ID token：

$$\mathbf{O}_{\mathbf{B}_\gamma} = \mathbf{G}_\gamma \mathbf{B}_\gamma$$

这让 ID token 之间"按内容相似度互相关注"，实现 ID 共现关系与内容语义关系的互补。由于 $\mathbf{G}_\gamma$ 完全由冻结特征计算，**可被所有层复用**，且计算与训练无关。

原始形式复杂度为 $O(L_{\mathbf{B}_\gamma}^2 d)$（平方），实践中对相似度矩阵做**逐行 top-K 稀疏化**（固定 $K=5$），只保留每个 item 最相似的 K 个邻居参与聚合，复杂度降为 $O(L_{\mathbf{B}_\gamma}Kd)$，随序列长度**线性增长**。

CSA 对 $\mathcal{B}_u$、$\mathcal{B}_c$ **分别独立计算**，以保持 user-candidate 解耦；相似度矩阵及 top-K 结果可与排序阶段的其它特征处理并行预计算，几乎不引入额外延迟。

**Q: CSA 中 $\gamma$ 是否是"由多模态特征组成"，$\mathbf{B}_{u/c}$ 是否随请求变化，是否是"给定一个 $\gamma$、多层重复稀疏聚合"？** 基本理解正确，细节澄清如下：
- $\gamma\in\{u,c\}$ 本身只是一个**类型下标**（标记当前处理的是用户短期行为还是候选相关行为），不是"由多模态特征组成"；真正由多模态内容特征组成、且训练时冻结不动的是 $\mathbf{M}_\gamma$（即 $\mathbf{M}_u$ 或 $\mathbf{M}_c$，对每个 $\gamma$ 各有一份）；
- $\mathbf{B}_u$（用户短期行为）和 $\mathbf{B}_c$（GSU/MUSE 从生命周期行为中按候选检索出的子序列）确实是**随每次请求的输入数据变化**的：$\mathbf{B}_u$ 对同一用户在同一请求内的所有候选是**共享不变**的（因此可做 user 侧计算复用），而 $\mathbf{B}_c$ 是**逐候选不同**的（因为检索依赖具体候选 item）；
- 实际计算顺序确实是"给定 $\gamma$（即先确定是处理 $\mathbf{B}_u$ 还是 $\mathbf{B}_c$），用对应的冻结内容特征 $\mathbf{M}_\gamma$ **一次性算出**相似度矩阵 $\mathbf{G}_\gamma$（及其 top-K 稀疏化），该矩阵可与其它特征处理并行预计算、**不随层数重复计算**；随后在 $S$ 层堆叠的每一层里，都用这个**同一个、固定不变**的 $\mathbf{G}_\gamma$ 去聚合**当前层**、已经过若干层演化的行为 token 表达 $\mathbf{B}_\gamma^{(s-1)}$"——即"矩阵算一次、聚合用 S 次"，而不是每层都重新计算相似度矩阵。

### 训练与部署实践

- **从生产模型热启动**：线上生产模型已经用数万亿样本训练多年，从零训练 EST 难以超越。做法是**稀疏参数（embedding 表）从历史生产模型初始化，稠密参数从零训练**；在数千亿样本上训练后，EST 相对生产模型提升 **GAUC +0.74%**，且随数据量增加持续提升；
- **多 epoch 训练 + one-epoch 现象应对**：为提升数据利用率采用多 epoch 训练，但推荐系统在第二个 epoch 开始常因过拟合掉点（one-epoch 现象）。借鉴前人做法，采用**异步范式**：每个 epoch 开始时，稀疏参数重置为初始的生产模型权重 $\theta_s^{(0)}$，而稠密参数则延续上一个 epoch 训练结果（仅第一个 epoch 随机初始化）。该策略额外带来 **GAUC +0.44%** 的提升。

## 实验结果

### 离线对比（RQ1）

数据集：淘宝生产日志，超 10 亿次曝光（6 天）；短期序列截断 300，生命周期行为序列截断 5000，GSU（MUSE）从中检索 top-50 作为 $\mathcal{B}_c$。评测指标 AUC/GAUC/参数量/推理 GFLOPs（batch=60）。所有可扩展模型统一为 6 层、token 维度 128、约 1 亿参数。

| 范式 | 方法 | AUC | ΔAUC | GAUC | ΔGAUC | Params(M) | GFLOPs |
|---|---|---|---|---|---|---|---|
| Base | MLP | 0.6895 | - | 0.6459 | - | 83.13 | 9.64 |
| 传统 DNN | AutoInt | 0.6907 | 0.18% | 0.6477 | 0.28% | 5.20 | 1.10 |
| 传统 DNN | DCNv2 | 0.6916 | 0.31% | 0.6482 | 0.35% | 15.18 | 1.49 |
| 层次化可扩展 | RankMixer | 0.6914 | 0.28% | 0.6482 | 0.35% | 136.84 | 15.46 |
| 层次化可扩展 | HiFormer | 0.6928 | 0.49% | 0.6493 | 0.53% | 113.78 | 14.17 |
| 统一可扩展 | MTGR | 0.6917 | 0.32% | 0.6474 | 0.23% | 118.79 | 10.57 |
| 统一可扩展 | OneTrans | 0.6921 | 0.38% | 0.6477 | 0.27% | 111.98 | 15.94 |
| 统一可扩展 | OneTrans-D（候选行为用DIN聚合） | 0.6941 | 0.68% | 0.6500 | 0.64% | 112.06 | 16.19 |
| 统一可扩展 | OneTrans-F（候选行为全量拼接） | 0.6955 | 0.87% | 0.6507 | 0.75% | 110.54 | 52.79 |
| **统一可扩展** | **EST** | **0.6963** | **0.99%** | **0.6515** | **0.87%** | 103.81 | 18.48 |

三点结论：① 带 token 专属参数的注意力（如 HiFormer）比普通门控/attention 更擅长建模异质特征交互；② 候选相关行为序列的充分建模至关重要——OneTrans/MTGR 因池化候选行为而不如 HiFormer，加入 DIN 聚合（OneTrans-D）或全量拼接（OneTrans-F）后明显提升，但全量拼接计算开销骤增；③ **EST 在效果和效率上双赢**：相对 OneTrans 提升 GAUC +0.59%（计算量相近），相对 OneTrans-F 提升 GAUC +0.12% 但计算量降低 65%。

### 消融实验（RQ2）

以 Full-Attention（完整 self-attention 做统一建模，其余设置与 EST 一致）为基准：

| 方法 | FA | LCA | CSA | ΔAUC | ΔGAUC | ΔGFLOPs |
|---|---|---|---|---|---|---|
| Full-Attention | ✓ | - | - | 0.6951/0.6506 (绝对值) | - | 66.27 (绝对值) |
| EST (仅LCA) | - | ✓ | - | -0.01% | -0.03% | -72.23% |
| EST (LCA+CSA) | - | ✓ | ✓ | +0.17% | +0.14% | -72.11% |

LCA 相比全量 self-attention **减少 72.23% 计算量，性能仅降 0.03%**，验证了"非行为-行为交互才是关键信息，其余交互冗余"的结论；在 LCA 基础上加入 CSA 进一步带来 **+0.14%** 的性能提升,而计算量几乎不变，验证了内容相似度稀疏建模的有效性。

### Scaling Law（RQ3）

沿两个维度扩展模型：**深度**（堆叠层数 $S$）和**宽度**（token 维度 $d$），推理 GFLOPs 从 5 到 50、参数量从 30M 到 0.3B。对 $\Delta\text{GAUC}$ 拟合幂律 $\Delta\text{GAUC}(X)=E\times X^\alpha$：

- 相对计算量 $C$：深度扩展 $\Delta\text{GAUC}=0.61\times C^{0.12}$，宽度扩展 $\Delta\text{GAUC}=0.68\times C^{0.10}$；
- 相对参数量 $P$：深度扩展 $\Delta\text{GAUC}=0.46\times P^{0.14}$，宽度扩展 $\Delta\text{GAUC}=0.63\times P^{0.08}$。

两个维度下 ΔGAUC 均随规模单调提升、呈幂律趋势；**深度扩展的增长曲线比宽度更陡**，说明加深层数更有利于挖掘复杂的特征交互。

### 线上 A/B 测试（RQ4）

部署于淘宝展示广告"全站推"（QuanZhanTui）场景，生产基线是已集成 CAN、SENET、SIM、**MUSE**、SimTier、**MAKE** 等多种 SOTA 模块的复杂 MLP 模型：

| 场景 | CTR | RPM |
|---|---|---|
| Guess What You Like（猜你喜欢） | +1.22% | +3.27% |
| Post-Purchase（购后推荐） | +2.01% | +2.66% |

两个场景均取得显著业务收益。

## 与团队前作的关系

本文与团队此前的 SimTier/MAKE（CIKM 2024, 2407.19467）和 MUSE（2512.07216）一脉相承，且在实验中直接复用/对比这些前作模块：
- **SimTier**（多模态相似度直方图）在本文中被用作所有 baseline 的标准内容特征接入方式（作为 $\mathcal{N}$ 中的一种非行为 token），并作为洞察二的实证依据之一；
- **MUSE** 的 GSU 被直接用作从生命周期行为中检索候选相关子序列 $\mathcal{B}_c$ 的模块；
- 线上生产基线本身已集成 MUSE、SimTier、MAKE，EST 是在此基础上进一步提升的**统一建模骨干网络**，解决的是"如何在算力约束下把所有异质输入（含 MUSE 检索出的候选行为）无损地放进同一 Transformer 序列并保持可扩展性"这一更上层的架构问题，与前作的"如何构造/接入多模态表征"是互补而非替代关系。

## 结论

论文针对工业 CTR 预测在严格延迟约束下追求 LLM 式 scaling law 的问题，从"信息密度不对称"和"模态特定先验"两个 CTR 与 LLM 的本质区别出发，提出了完全统一建模的 **EST** 架构：**Lightweight Cross-Attention (LCA)** 只保留非行为-行为这一高价值交互方向，将自注意力复杂度从平方降为双线性；**Content Sparse Attention (CSA)** 用冻结的内容相似度做免训练的稀疏行为间交互，复杂度随序列长度线性增长。两者共同实现了"不做有损早期聚合、同时兼顾效率"的完全统一建模。离线实验验证了 EST 在效果和效率上同时优于层次化建模与已有的（部分）统一建模方法，并展现出稳定的幂律 scaling 关系；线上 A/B 在淘宝展示广告的两个核心场景取得显著的 CTR 与 RPM 提升，为工业级 CTR 模型的高效可扩展架构提供了可复用的设计范式与部署经验。
