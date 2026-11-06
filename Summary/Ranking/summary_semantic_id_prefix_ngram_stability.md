# Enhancing Embedding Representation Stability in Recommendation Systems with Semantic ID

> **综合理解（讨论后总结）**
>
> 本文沿用 RQ-VAE 得到的 SID 结构，例如一个 item 表示为 (c_1, c_2, c_3)，但完全没有使用 RQ-VAE 中对应的码本 embedding，而是另外构造了一组层级 embedding：以前缀作为每一层的 key，第一层取 c_1，第二层取 [c_1,c_2]，第三层取 [c_1,c_2,c_3]。得到多层 embedding 后再做 pooling，作为 item 的表征。随着层数变多，key 的数量急剧增大，因此用取模（mod）来控制 key 的数量。
>
> **点评：整体正确，有几处需要修正或补充。**
> - ✅ **"沿用 SID 结构、不用码本 embedding"**：正确。推荐模型只用了 SID 的离散码来决定哪些 item 共享哪些 embedding 行（即 trie 划分结构），RQ-VAE 码本向量既不加载也不用于初始化；前缀 embedding 随机初始化，由推荐 loss 端到端训练。补充：RQ-VAE 由作者基于 CU 内容 embedding 自行训练，然后冻结使用。
> - ✅ **"以前缀为每层的 key"**：正确，这就是 prefix-ngram 的核心。每层 key 通过偏移公式 Σ K^{i-t}(c_t+1) − 1 映射到互不重叠的 ID 区间，因此 [5,17] 和 [8,17] 是两个不同的 key。
> - ❌ **"mean pooling"**：应为 **sum pooling**（论文 §3.1：e_f(x) = Σ e_{f_i}(x)）。层数 L 固定时，两者只差常数因子 1/L，效果上基本等价，但严格来说论文用的是 sum。sum 的写法也更贴合 RQ 的残差分解 z ≈ v_{c_1}+v_{c_2}+v_{c_3}：粗层学共性，细层学残差。
> - ⚠️ **"用 mod 减少 key"**：方向正确，需要补充三点：
>   1. 只有当某层基数超过表大小 H 时才取模。浅层（第 1 层 2048，第 2 层约 4.2M）可以直接放进表里、无碰撞；需要取模的主要是深层。
>   2. 理论基数是 K^l，但实际出现的 key 数不超过 item 数。深层取模后的随机碰撞仍然存在，只是有粗粒度层兜底，影响比直接哈希 raw ID 小。
>   3. 取模后，深层 key 在理论上可能与浅层 key 落到同一行。论文没有讨论这一点，也没有做消融。
> - ➕ **补充：层数与用法。** 生产配置是 RQ-VAE L=6、K=2048，但只用 **prefix-5gram**（前 5 层），表大小 H=O(50M)。离线实验用的是 L=3、prefix-3gram。层数越深、K 越大，NE 越好（2048×6 的 prefix-6gram 效果最好）。
> - ➕ **补充：为什么必须把前缀写进 key。** 残差量化的深层码只有在给定前缀时才有确定含义，而 DLRM 的 sparse 模块只做 pooling、不做上下文建模，所以前缀条件必须显式编码进 ID。这也是 trigram（只用完整 tuple）和 all bigrams（不以完整前缀为条件）明显更差的原因。

- **arXiv**: [2504.02137](https://arxiv.org/abs/2504.02137)（2025-04）
- **作者**: Carolina Zheng（Columbia，2024 Meta 实习）, Minhui Huang, Dmitrii Pedchenko, Kaushik Rangadurai, Siyu Wang, Gaby Nahum, Jie Lei, Yang Yang, Tao Liu, Zutian Luo, Xiaohan Wei, Dinesh Ramasamy, Jiyan Yang, Yiping Han, Lin Yang, Hangjun Xu, Rong Jin, Shuang Yang（AI at Meta）
- **场景**: Meta Ads Ranking（DLRM 类精排模型）
- **关键词**: Semantic ID, RQ-VAE, Prefix-ngram, 哈希碰撞, ID 漂移, 长尾/冷启动, A/A variance

---

## 1. 一句话总结

用内容 embedding + RQ-VAE 生成的层级 Semantic ID 替代随机哈希的 item ID，并提出 **prefix-ngram** 参数化（把每一层前缀都当作独立 token 查表后 sum-pool），在 Meta 广告精排中作为 **top sparse feature** 上线一年多，带来 **+0.15% 线上核心指标**，并将同素材副本的预测差异（A/A variance）降低 **43%**。

## 2. 问题：工业推荐中 item embedding 的三大数据难题

| 问题 | 现象（Meta Ads 数据） | 对随机哈希的影响 |
|---|---|---|
| **Item cardinality** | item 数远大于 embedding 表容量 H | 必须哈希，碰撞是随机的，相互冲突的梯度污染 embedding |
| **Impression skew** | top 0.1%（head）占 25% 曝光；接下来 5.5%（torso）占 50%；剩余 94.4%（tail）只占 25% | tail item 训练样本少，随机碰撞无法让相似的头部 item 向其迁移知识 |
| **ID drifting** | 初始 corpus 中一半 item 在 **6 天内**退出，同时等量新 item 进入 | 同一行 embedding 随时间代表不同 item → **embedding representation instability**；老 item 学到的知识丢失，新 item 的 embedding 本质上是随机的 |

核心观点：需要一个 **稳定的 ID 空间**——embedding 的含义不随时间漂移。Semantic ID 的 ID 空间是先验固定且有语义的，天然满足这一点：广告主下线旧广告、上线新广告时，产品大类通常不变，因此新旧广告的 Semantic ID 至少共享前缀。附录图显示 Semantic ID 空间下的点击分布偏斜显著小于 raw ID 空间。

## 3. 方法

### 3.1 Semantic ID 生成
1. 多模态内容理解（CU）模型（图文对齐预训练，类 CLIP；生产中 CU 模型在 CC100 预训练后在内部广告数据上微调）产出 item 内容 embedding。
2. 在近 3 个月的 item 内容 embedding 上训练 **RQ-VAE**：层数 L、每层码本大小 K，残差量化得到由粗到细的码序列 (c_1, …, c_L)。
   - 损失 = 重建损失 + β·commitment + codebook 损失（β=0.5）。
   - 直觉：c_1 = "食品类广告"，(c_1,c_2) = "披萨广告"，(c_1,c_2,c_3) = "英文披萨广告"。
3. 训练完后冻结 RQ-VAE 用于在线服务。

### 3.2 Token Parameterization（本文核心贡献）
Semantic ID → embedding 表行号的映射 p(c; H)，查出的多个 embedding 做 sum-pooling。超过 H 时取模，不同位置加偏移避免跨位置碰撞。

| 参数化 | 定义 | 说明 |
|---|---|---|
| Trigram / Fourgram | 完整 tuple 作为一个 ID | 只有最细粒度，基数极高 |
| All bigrams | 相邻两层 (c_i, c_{i+1}) 各一个 ID | 平铺、非层级 |
| **Prefix-ngram** | 对 i=1..n，用前缀 (c_1..c_i) 各生成一个 ID：Σ_{t=1}^{i} K^{i-t}(c_t+1) − 1 | **同时包含所有粒度的前缀**，粗粒度 token 由大量 item 共享 → 更多知识共享 |

**消融（Train NE gain vs. baseline，越负越好）：**

| RQ-VAE K×L | 参数化 | NE Gain |
|---|---|---|
| 2048×3 | Trigram | −0.028% |
| 2048×4 | Fourgram | −0.035% |
| 2048×4 | All bigrams | −0.091% |
| 512×3 | Prefix-3gram | −0.034% |
| 1024×3 | Prefix-3gram | −0.097% |
| 2048×3 | Prefix-3gram | −0.141% |
| 2048×5 | Prefix-5gram | −0.208% |
| 2048×6 | Prefix-6gram | **−0.215%** |

结论：① Prefix-ngram 最优，**把层级结构编码进查表方式是关键**（同为 2048×3，trigram −0.028% vs prefix-3gram −0.141%，约 5 倍）；② 层数越深越好；③ 码本越大越好。

## 4. 离线实验

**设置**：简化版 Meta 广告精排模型（保留全部 dense 特征和用户历史，sparse 模块只保留 target item，去掉约 O(100) 个其他 sparse 特征）；4 天数据、单 epoch 时序训练；在次日前 6 小时评估。RQ-VAE L=3、K=2048、prefix-3gram。

**Baselines**：
- **IE**（Individual Embeddings）：每个 item 独立一行，生产中不可行，仅作参照；未见过的 ID 使用随机初始化 embedding。
- **RH**（Random Hashing）：平均碰撞因子 3。
- **SemID**：碰撞因子同为 3。

### 4.1 分段分析（Eval NE）

| 分段 | RH | IE | SemID | vs RH | vs IE |
|---|---|---|---|---|---|
| Head（0.1% items） | 0.80105 | **0.80101** | 0.80108 | 0.00% | +0.01% |
| Torso（5.6%） | 0.83589 | 0.83583 | **0.83580** | −0.01% | −0.00% |
| Tail | 0.83904 | 0.83886 | **0.83872** | −0.04% | −0.02% |
| 训练中见过的 items | 0.82626 | 0.82612 | **0.82600** | −0.03% | −0.02% |
| **新 item（冷启动）** | 0.83524 | 0.83453 | **0.83180** | **−0.41%** | **−0.33%** |
| 全部 | 0.82663 | 0.82645 | **0.82621** | −0.05% | −0.03% |

- 收益集中在 **tail 和冷启动**；head 基本持平。
- **SemID 甚至优于无碰撞的 IE**，说明收益不只是"比随机聚类好"，而是语义知识共享本身有价值。冷启动 item 直接复用语义相似老 item 的已训练权重。

### 4.2 对分布漂移的敏感度
在训练数据上评估 NE[结束前 42–48h] − NE[最后 6h]（越小说明表征漂移影响越小）：RH 0.0083，IE 0.0074，**SemID 0.0073**。RH 受 ID drifting 影响最大（embedding 被新 item 覆盖后遗忘老 item），SemID 与 IE 一样稳定。

**长周期训练**：20 天 vs 4 天数据，RH Eval NE 提升 −0.18%，SemID 提升 **−0.23%** → SemID 更能从长时间窗口数据中获益。

### 4.3 表征空间分析
把 RH / SemID 视为对 IE embedding 的两种划分（碰撞因子 5），看簇内方差：RH 1.52e-3，SemID 小簇 1.31e-3，SemID top-1000 大簇 1.23e-3 → SemID 簇内更紧凑。簇间距离信号不一致（top-1000 大簇间距离仅 0.06），作者推测 RQ-VAE 在高密度区域放置了多个质心。

### 4.4 用户历史序列建模
用户历史覆盖 3 个月、item 基数超 10 亿，序列长度 O(100)。把用户历史的 item ID 从 RH 换成 SemID：

| 聚合模块 | Train NE Gain | Eval NE Gain |
|---|---|---|
| Bypass（无上下文化） | −0.056% | −0.085% |
| Transformer | −0.071% | **−0.110%** |
| PMA（32 个可学习 query） | −0.073% | −0.100% |

注意力模式分析：使用 SemID 后，**熵更低、对 padding 的注意力更少、对首个（最近）token 的注意力更高、自注意力更少** → 注意力更集中在高信号 item 上，说明 SemID 表征更稳定、更有意义。

## 5. 生产落地（Meta Ads，已上线 1 年以上）

- **RQ-VAE 配置**：L=6，K=2048，**prefix-5gram**，embedding 表 H=O(50M)；冻结 checkpoint 服务。
- **Serving 链路**：广告创建 → CU 模型出 embedding → RQ-VAE 计算 Semantic ID → 写入 **Entity Data Store** → 特征生成阶段为 target item 与用户交互历史的 raw ID 补充 Semantic ID → 请求到达时取预计算特征送入精排模型。
- **特征**：基于文本、图像、视频等不同内容 embedding 源，构建 **6 个 sparse 特征 + 1 个序列特征**；按特征重要性评估位列 **top sparse features**。

| 旗舰广告精排模型 | Train NE | Eval NE |
|---|---|---|
| + 6 个 sparse 特征 | −0.063% | −0.071% |
| + 1 个序列特征 | −0.110% | −0.123% |

（Meta Ads 中离线 NE > 0.02% 即视为显著。）多个广告精排模型合计 → **+0.15% 线上核心指标**。

### 5.1 语义相似 ⇒ 预测相似（线上 A/B）
给用户推荐集合 S 时，以 50% 概率把其中一个 item 替换为**共享 Semantic ID 前缀**的另一个 item，测 Click Loss Rate。前缀越深，点击损失越小（被注释掉的表格中的数值：随机替换 13.7% → 1-prefix 8.2% → 2-prefix 6.0% → 3-prefix 5.2%），单调下降。

### 5.2 A/A Variance
随机哈希下，同一素材的完全副本（不同 raw ID）会得到不同 embedding → 预测与投放结果不同，损害广告主信任。定义 AAR = 2(p(a1)−p(a2))/(p(a1)+p(a2)+ε)。加入 6 个 SemID sparse 特征后 **平均 AAR 降低 43%**，作者认为主要来自 tail item。

## 6. 洞察与评价

**亮点**
- 把 Semantic ID 的价值从"泛化更好"具体归因到 **长尾、冷启动、ID 漂移下的表征稳定性**，并用 IE 这一"无碰撞"上界证明语义共享本身有增益。
- **Prefix-ngram** 简单但有效：在判别式精排模型中使用 SID 时，查表粒度设计比 RQ-VAE 本身更关键（同一 RQ-VAE 下效果差约 5 倍）。
- 工业证据充分：生产配置、serving 链路、线上收益、A/A variance 这一业务层面的稳定性指标。

**局限**
- 离线实验只保留 target item 一个 sparse 特征，与真实生产环境差距较大；相对 IE 的提升幅度较小（−0.03%）。
- 表征空间分析结论不一致（簇间距离），缺乏更深入的分析。
- 没有讨论 RQ-VAE / CU 模型更新导致 SID 重新映射时的一致性问题（仅使用冻结 checkpoint）。
- 源码中有被注释掉的改进（**nested reconstruction loss**：Collision Rate@1 从 0.0447 降到 0.0146，重建损失略升；**多实体共享 SID embedding 表**：仅用 20% 模型大小保持效果），但未写进正文。

**与相关工作的关系**
- 基于 Singh et al. 2023（YouTube "Better Generalization with Semantic IDs"）和 TIGER（Rajput et al. 2024），但场景是**判别式精排中把 SID 当作 sparse/序列特征**，而非生成式检索。
- 与 Meta CU 团队面向自然流量（IGR、IFR、FBR 等）并行推进的 Semantic ID 工作是两条独立线（见下文）。

## 7. 内网相关资料（综合）

| 资料 | 作者 / 时间 | 要点 |
|---|---|---|
| [Meta AI Papers 预印本帖](https://fb.workplace.com/groups/831302610278251/permalink/27570558339259315/) | Minhui Huang，2024-11-18 | 投稿 **WWW 2025 Industry Track**；评论中 cc 了 CU Semantic ID 团队（Sreya Dutta Roy、Maksym Kuzyshyn、Jun Xiao、Chen Yuan） |
| [Analyzing Embedding Representation Stability and Source of Gains from Semantic ID](https://fb.workplace.com/groups/8000374759979655/permalink/9758624350821345/) | Carolina Zheng 等，2024-11，Tokenization Learning 群 | 实习总结。离线实验基于 **IGFM mini**（只保留 target ad ID sparse 特征 + previous conversions EBF）；RQ-VAE 为 **4 层 × 2048，基于 MARS 内容 embedding**（6–9 月广告），prefix-gram 参数化。结论与论文一致（新广告 −0.41% vs RH、−0.33% vs IE） |
| [Semantic IDs for Ads Ecosystem（设计文档）](https://docs.google.com/document/d/1jmBVPK3LcnEnKoIMVVPnxA9OmzPB3qIQ7eH7O6ohwPI) | Kaushik Rangadurai、Minhui Huang、Jiyan Yang，2023-12 | 动机数据：30 天窗口内 **2.6 亿**独立广告，广告活跃期中位数约 6 天，有时约 1000 个广告共用一行 embedding；IG CTR 7x 模型去掉 sparse arch 会损失 0.96% NE。远期规划：对所有实体（user/page/event）做 tokenization、HSNN 聚类、EBF tokenization |
| [Tokenization Learning – 2024H2 Tracking](https://docs.google.com/document/d/1atkIF0LZBwRoE3IKM9ZIGpCBVFnUAseCRU24D7GgDpE) | 团队执行跟踪 | 生产路径：对 12 个 AdID/EEK/EEK_NC 特征做重映射（18M 哈希，3-prefix MARS）比 v44 基线 **NE +0.05%**；重映射 49 个特征得到 0.06%，但 **QPS 从 1.9M 降到 0.6M**（工程代价明显）；简化版 CMFv2 在 7.5B 时 0.02% NE。同时探索 Fully Tokenized Model、PageID tokenization |
| [MRS Launch KB / arxiv sota / 2025](https://www.internalfb.com/wiki/MRS_Launch_KB/arxiv_sota/2025/) | wiki，2026-09 更新 | 本文排名第 4（13/17 分），状态为 **CANDIDATE**，即 MRS 考虑采纳；预估每个 surface（Reels/Feed/IG）需 1–2 个季度 |
| [Gradient Tokenization 26H1 Update](https://fb.workplace.com/groups/1033540429995021/permalink/28457375643851463/) | 2026-06-30 | 这条线已并入 Ads 的生成式推荐项目 **Gradient**。26H1 共 21 个 MC 提案，把 tokenization 覆盖率从 **5% 提到 19%**，带来 **0.26% eGAS**、冷启动 eval NE +0.22%；在 CMF 上把 27 个 ad-vocab 特征（1.7M 哈希）换成 CRDL RQ-VAE token 基本持平（+0.015%，在噪声内）。另在研究记忆（item ID）与泛化（SID）的权衡边界，以及 LLM-native / 生成式检索方向 |

**相关但独立的工作**
- [Semantic ID: A Tale of Three Perspectives](https://fb.workplace.com/groups/1176352800425411/permalink/1324083358985687/)（Jun Xiao，2024-12）：CU 团队面向自然流量 surface（IGR、IG/FB IFR、FBR、FB CFR）的 SID 工作，与 Ads 无关。
- [MPZCH 论文文档](https://docs.google.com/document/d/1iTOtkS9f_-qflXLMBNlgupFnA7jXIjqwbijHwDd3xpQ)（Ziliang Zhao）：另一篇讨论 embedding 碰撞的 Meta 论文（零碰撞哈希，已部署于 20+ 模型），思路互补：MPZCH 走"消除碰撞"，SemID 走"让碰撞有语义"。

**论文与内网资料对照**
- 离线实验：论文写 L=3 的 prefix-3gram，实习帖写 4 层 RQ-VAE + MARS embedding。内容 embedding 来源 **MARS** 在论文中被泛化成了"多模态图文基础模型"。
- 生产收益：论文写"+0.15% 线上核心指标"，但内网没找到对应的 launch 帖来独立确认这个数字（wiki 中的 0.15% 也是转述论文）。内网能查到的是离线 NE 增量（0.02%–0.06%）和后续 Gradient 项目的汇总数字（0.26% eGAS）。
- 工程代价：论文没提，内网跟踪文档显示大规模重映射会明显降低 QPS，说明特征数量与 serving 效率之间需要权衡。
- 覆盖范围：目前的证据都指向 **Ads Ranking**（IGFM/CMF 系列模型），没有找到这篇工作在自然流量 Feed/IG 上的上线记录。

## 8. 讨论记录

### Q1：prefix-ngram 是替代了原来的 SID embedding 吗？它和 motivation 怎么对应？

**澄清：prefix-ngram 没有替代 SID，它替代的是"SID → embedding 行"的查表方式。**

整个流程分三层：
1. **SID 本身**：RQ-VAE 输出的码 (c_1, …, c_L)，论文对这部分没有改动。
2. **Token parameterization** p(c; H)：把码映射到 embedding 表的若干行，**本文的贡献在这一层**。
3. **Embedding 表 + sum-pooling**：查出的多行加起来，就是 item 表征。

对比的是同一套 SID 的不同查表方式：trigram/fourgram（整个 tuple 只查一行）、all bigrams（相邻两层各查一行）、prefix-ngram（每一层前缀各查一行）。原始 SID 工作（Singh et al. 2023）也比较过 n-gram 类参数化，本文的 prefix-ngram 是在这之上的改进。

"替代"关系在两个场景下不同：
- **离线实验**：SemID 直接替代 target item 原来的随机哈希查表（RH → SemID，碰撞因子不变）。
- **生产**：表 "Baseline + 6 sparse features" 是在原有 ad ID 特征之外**新增**的 SID 特征。内网 2024H2 跟踪文档里也有把 AdID/EEK 等已有特征直接"重映射"成 SID 的实验，两种用法都有。

**prefix-ngram 与三个 motivation 的对应**

prefix-ngram 下 item 的表征是
e(x) = e[c_1] + e[c_1c_2] + e[c_1c_2c_3] + …

各层前缀的偏移量互不重叠：第 1 层落在 [0, K)，第 2 层落在 [K, K+K²)……所以取模之前，不同层之间不会碰撞。

| Motivation | 随机哈希 / trigram 的问题 | prefix-ngram 如何解决 |
|---|---|---|
| **Item cardinality**（H ≪ I） | trigram 只查完整 tuple，基数 K³≈8.6e9，必须取模，碰撞又变成随机的 | 粗粒度层基数小（第 1 层 2048 行，第 2 层约 4.2M 行），不用哈希就能放进表里。只有细粒度层需要取模，而细粒度层即使发生随机碰撞，也有粗粒度层的"语义底座"兜底 |
| **Impression skew**（94.4% 长尾） | tail item 自己那一行很少被更新，几乎等于随机初始化 | 粗粒度行被成百上千个 item（包括 head）共享，梯度充足、训练充分；细粒度行提供区分度。效果类似 n-gram 语言模型的 **backoff/平滑**：样本不足时靠粗粒度，样本充足时细粒度起作用。这也解释了为什么 head 基本持平（细粒度行几乎等于自己的独立 embedding，相当于记忆）、tail 获益最大 |
| **ID drifting**（半数广告 6 天内下线） | 新 ID 被哈希到和语义无关的行上；老 ID 下线后，学到的知识被新 ID 覆盖 | 新广告与已下线广告通常共享前缀，粗粒度行**已经训练好、可以直接用**。RQ-VAE 冻结后，每一行的含义等于一个固定的语义簇，不随 raw ID 的更替而漂移，也就是 **stable ID space** |

**为什么只查完整 tuple（trigram）或 all bigrams 明显更差？**
- **Trigram**：只有最细粒度，相当于"语义去重版的 IE"。兄弟簇 (c_1,c_2,x) 和 (c_1,c_2,y) 查的是完全独立的两行，RQ-VAE 的层级结构没有用上，知识共享只发生在同一个细粒度簇内部。同为 2048×3，trigram 只有 −0.028%，prefix-3gram 是 −0.141%。
- **All bigrams**：(c_2, c_3) 这一对没有以 c_1 为条件。而**残差量化中深层码的含义依赖于前面的码**（c_3 量化的是减去 v_{c_1}、v_{c_2} 之后的残差），所以相同的 (c_2,c_3) 出现在不同 c_1 下，语义可能完全不同，这里的共享反而是"伪语义碰撞"。
- **Prefix-ngram**：每个 token 都以完整前缀为条件，语义定义清晰。而且 sum-pooling 得到的 e[c_1]+e[c_1c_2]+e[c_1c_2c_3] 和 RQ-VAE 的残差分解 z ≈ v_{c_1}+v_{c_2}+v_{c_3} **形式同构**：相当于在推荐目标上重新学了一套"层级残差 embedding"，粗层学共性，细层学偏差。

**深度 L 和码本 K 越大越好**，可以理解为在"记忆 ↔ 泛化"之间加了更多刻度：越深越接近 IE（记忆），越浅越接近类目特征（泛化），模型通过各层的权重自行权衡。线上 Click Loss Rate 随前缀加深单调下降（13.7% → 8.2% → 6.0% → 5.2%），印证了每一层前缀都携带增量的预测信息。

**值得注意的隐患**：生产配置是 L=6、prefix-5gram、H=O(50M)。第 3 层起基数 K³ 已远超 H，所以第 3–5 层都要取模，**深层仍然存在随机碰撞**。不过深层实际出现的 tuple 数不超过 item 数，而且有粗粒度层兜底，影响远小于直接对 raw ID 做哈希。论文没有做消融来量化这部分影响。

### Q2：prefix-ngram 是不是给 [c_1,c_2] 学一个新 embedding、给 [c_1,c_2,c_3] 再学一个？它和 RQ-VAE 的多层 SID 怎么关联？

**是的。每个前缀都被当作一个全新的离散 ID，各自对应 embedding 表中独立的一行**，这些行随机初始化，只用推荐的 CE loss 端到端训练。

示例（K=2048，某 item 的 SID = (5, 17, 300)）：

| 前缀 | 生成的 ID（按 Table 1 公式） | embedding |
|---|---|---|
| [c_1] = [5] | (5+1) − 1 = **5** | e[5] |
| [c_1,c_2] = [5,17] | 2048·6 + 18 − 1 = **12305** | e[12305] |
| [c_1,c_2,c_3] = [5,17,300] | 2048²·6 + 2048·18 + 301 − 1 = **25202988** | e[25202988]（超过 H 时取模） |

item 表征 = e[5] + e[12305] + e[25202988]。

**与 RQ-VAE 多层 SID 的关系：只继承"结构"，不继承"向量"**
- RQ-VAE 的码本向量 v^l_k 位于**内容 embedding 空间**，在 prefix-ngram 中**完全没有被使用**：推荐模型不加载它们，也不用它们来初始化。
- SID 真正传递给推荐模型的是一棵**层级划分树（trie）**：第 l 层的每个前缀 = 树上的一个节点 = 一个语义簇。它决定了**哪些 item 共享哪些 embedding 行**，也就是碰撞结构。
- 因此 prefix-ngram 等价于：**给 trie 上的每个节点学一个 embedding，item 表征 = 从根到叶路径上所有节点 embedding 之和**。
- 这与 RQ 本身的形式同构：内容空间中 z ≈ v_{c_1} + v_{c_2} + v_{c_3}（同一棵树、按路径累加），推荐空间中 e(x) = e[c_1] + e[c_1c_2] + e[c_1c_2c_3]。区别在于：RQ 中第 l 层码本是**跨父节点共享**的（c_2=17 在不同 c_1 下对应同一个 v^2_{17}），而 prefix-ngram 中 [5,17] 和 [8,17] 是两行**完全独立**的 embedding，也就是每个节点单独拥有参数。

**为什么要把前缀显式编码进 ID，而不是按层各查一个 c_l（unigram）？**
- 残差量化的深层码 c_l 只有在给定前缀时才有确定含义。DLRM 的 sparse 模块只做 sum-pooling，**没有能力把 c_2 和 c_1 组合起来理解**，所以前缀条件必须手工写进 ID 里。
- 对比 TIGER 等生成式模型：它们用按位置区分的 unigram 词表，但由 Transformer 自回归地建模 c_l 与前缀的依赖关系，**条件关系由模型学出来**。prefix-ngram 可以看作在没有上下文建模能力的判别式模型里，用 ID 工程来"硬编码"这种条件关系。

**可能的延伸（论文未做）**
- 用 RQ 码本向量（经投影）初始化各层 embedding，把内容空间的先验迁移过来。
- 在 prefix embedding 之外保留 raw ID embedding（生产中新增而非替换的做法本质上就是这样），兼顾记忆与泛化。
- 参数组织上，论文写的是给不同位置加偏移、超过 H 时取模，从公式看像是所有层共用一个 ID 空间和一张表。深层节点数远超 H，可以考虑按层分配不同的表大小。

### Q3：综合理解与点评

用户的综合理解及逐条点评见文首「**综合理解（讨论后总结）**」。结论：整体正确；需要修正的一处是 pooling 方式为 **sum pooling** 而非 mean pooling；取模只作用于基数超过 H 的深层，浅层前缀无碰撞；生产使用 L=6 RQ-VAE 的前 5 层（prefix-5gram）。
