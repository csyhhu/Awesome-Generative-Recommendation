# LONGER：面向工业级推荐系统的超长序列建模 Transformer

来源：arXiv: [2505.04421](https://arxiv.org/abs/2505.04421)（RecSys '25，ByteDance）

## 综合理解与点评

> 本文把 Transformer 架构使用在推荐系统上。序列特征中，每个元素都构成一个 Token，元素对应的特征（属性、位置、时间特征）通过 concatenation 或 summation 方式组成 Token；非序列特征（包含 Target），多个特征（按照 Semantic）使用类似的方法组成 Token。然后进入 Transformer。为了避免序列特征过长导致的计算复杂，第一层使用非序列特征作为 Query，序列特征作为 KV，进行 Query Attention，得到长度为非序列特征 Token 数的序列。后面的层就使用这个序列做 full Attention。最后以 Target 部分的 Token 作为 Target 的最终表达，计算得分。

**点评：整体正确，抓住了 LONGER 的核心思想（全局/Target token 主导 Query + 第一层 cross 压缩序列 + 后续 self-attn + target token 出分）。但 4 处细节需要修正/补充：**

1. **"过程导致"应为"过长导致"**（疑似笔误）：是序列长度过长导致 $O(L^2)$ 复杂度爆炸，才需要第一层 cross 来压缩。
2. **第一层 Query 不只是非序列特征**：实际 $\mathbf{Q}=[\mathbf{G}; \mathbf{H}_S]$ = 全局 token + **从序列采样的 $k$ 个 token**；输出长度是 $m+k$，不是 $m$（非序列特征 token 数）。所以"得到长度为非序列特征 Token 数的序列"**不准确**——精神上对（target/全局是 Query 主导），但细节上 Query 还含从序列采样的 $k$ 个 token，输出长度 $m+k$ 而非 $m$。
3. **summation vs concatenation 的具体使用**：位置编码是 **add**（summation），时间差 side info 是 **concat**（拼接），最后过 MLP 投影到 $d$ 维。原概括用"concatenation 或 summation"是正确的高层表述，但没区分哪个用哪种方式。
4. **缺一个关键设计点**：第一层是 **Cross-Causal**（带因果 mask），让序列 token **看不到 target**——这是 KV Cache Serving 成立的前提（序列 K/V 与 target 解耦才能跨 candidate 缓存）。原概括未提 mask，但这点对理解"为什么推理能缓存用户序列"至关重要。

## 研究动机
在推荐系统中，超长用户行为序列同时承载长短期偏好，对精度与多样性至关重要，并能缓解信息茧房。然而传统方案存在三类瓶颈：
- **两阶段检索**（SIM、TWIN）：先从超长序列中检索 top-k，再做端到端短序列建模 → 上下游不一致、信息丢失；
- **预训练用户嵌入**（S3D、UE）：在源模型预训练后，将长序列压缩为单一 user embedding 传入下游 → 间接感知、丢失细粒度；
- **记忆增强模型**（MIMN、LMN、MARM）：用记忆槽或量化分解缓存长序列 → 命中率积累慢、近似误差。

近年来 HSTU、Wukong 等开始在推荐中验证 scaling law，但**GPU 高效的端到端超长序列建模**在工业规模仍欠探索。LONGER 的目标是在工业级时延与成本约束下，把端到端超长序列（长度 > $10^3$，工业实际做到 2000~10000）一次性落地，并验证推荐系统的 scaling law。

## 核心方法：LONGER = 全局 Token + Token Merge + 混合注意力 + 系统级优化

### 1) Global Tokens（全局 Token）（非序列特征 Non-Sequence Features）
在输入序列前拼接若干**全局 token**：目标 item 表征、可学习 CLS、UID embedding、高阶压缩的用户-物品交互特征等。
- **作用一**：作为信息锚点，桥接用户历史、上下文与候选 item；
- **作用二**：稳定长序列下的注意力分布，缓解"attention sink"（深层注意力过度聚焦早期 token 的现象，参考 StreamLLM）。全局 token 拥有完整注意力感受野。

### 2) Token Merge + InnerTrans（分组压缩 + 组内细化）
**Token Merge**：将相邻 $K$ 个 token 分组并压缩成更短序列，把 $O(L^2 d)$ 注意力复杂度降为 $O(L^2 d / K)$。FLOPs 比为：
$$\frac{\text{FLOPs}_{\text{Merge}}}{\text{FLOPs}_{\text{vanilla}}} = \frac{6dK + L/K}{6d + L}$$
工业典型配置 $L=2048, d=32, K=4$ 时，FLOPs 从 587M 降到 336M（**~42.8% 降幅**）。同时，token merge 引入了 $\Theta_{\text{merge}} = 12K^2 d^2 + 13Kd$ 的额外参数，**用压缩换参数扩展**，同时获得效率与表达力。

**InnerTrans**：在每组 $K$ 个 token 内部跑一个轻量 transformer，避免简单拼接导致的细粒度交互丢失。由于组内维度与长度都很小，开销有限：
$$\mathbf{M}_i = \text{TransformerBlock}\left([\mathbf{e}_{i}^1, \ldots, \mathbf{e}_{i}^K]\right)$$

### 3) LONGER 模型结构（Cross-Causal + Self-Causal 混合注意力）

#### 输入生成
- 序列 token 加入两种位置信息：(1) **绝对时间差特征**（每次交互与目标 item 的时间距离），(2) **可学习绝对位置编码**；
- 经 MLP 后形成 $\mathbf{R} \in \mathbb{R}^{(m+L)\times d} = [\mathbf{G}; \mathbf{H}]$（全局 token + 序列 token）；
- **Query 矩阵** $\mathbf{O} = [\mathbf{G}; \mathbf{H}_\mathbf{S}]$：由 $m$ 个全局 token 与 $k$ 个从全序列中**按策略采样**得到的序列 token 拼接而成。作者对比了 recent-k / uniform / learnable / 混合策略，**recent-$k$ 最优**。这是受 Perceiver/Q-Former 启发的"查询压缩"思路——只采样 40% 序列即可保留 95%+ 的性能提升，FLOPs 减半。

#### 第一层：Cross-Causal Attention
- Query = $\mathbf{O}$，Key/Value = 全序列 $\mathbf{R}$；
- 因果 mask 保证序列对候选 item 不可见（这一点是 KV Cache Serving 的前提）；
- 注意力后接 FFN。

#### 后续 N 层：Self-Causal Attention
- 在采样得到的 query 序列内部做自注意力，捕捉高阶依赖；
- 同样接 FFN，并堆叠 $N$ 次。

$$\underbrace{\text{CrossAttn}(\mathbf{O}, \mathbf{R})}_{\text{压缩长序列}} \longrightarrow \underbrace{\text{SelfAttn}(\cdot)\times N}_{\text{高阶交互}}$$

### 4) 训练与部署优化

#### (1) 训练框架 Jaguar：全同步 GPU 训练
- **统一存储**：dense + sparse 参数同步在 GPU 机器上更新，**取消外部 Parameter Server**；
- **分层稀疏嵌入存储**：高频特征 → HBM，中频 → CPU MEM，低频 → SSD，匹配推荐数据的访问分布；
- **软硬协同**：减少通信开销与参数搬运延迟，提升吞吐与收敛稳定性。

#### (2) 混合精度 + Recompute
- BF16/FP16 混合精度，对关键模块保留高精度；
- 反向传播时按声明重算前向激活，以算换存；
- 实测：**+18% 吞吐、-16% 训练时长、-18% 显存**（dense 层最高 -28%）。

#### (3) KV Cache Serving
- 候选打分时，用户序列的 K/V 投影**只算一次**并缓存；
- 对每个候选只计算其全局 token 与缓存用户序列之间的注意力；
- 把多候选打分时吞吐退化从 -40% 缩小到 **-6.8%**。

## 实验结果
### 离线实验（Douyin Ads CVR 预测）
数据集：5.2B 样本、130 天日志、123 天训练/7 天评估，48×A100 GPU。

| Model | AUC | LogLoss | ΔAUC | ΔLogLoss |
|---|---|---|---|---|
| Base | 0.83968 | 0.48758 | - | - |
| SumPooling | 0.84201 | 0.48538 | +0.28% | -0.45% |
| TWIN | 0.84472 | 0.48168 | +0.60% | -1.21% |
| DIN (Recent50) | 0.84698 | 0.47830 | +0.87% | -1.90% |
| DIN | 0.84982 | 0.47452 | +1.21% | -2.68% |
| HSTU | 0.84994 | 0.47490 | +1.22% | -2.60% |
| Transformer | 0.85111 | 0.47293 | +1.36% | -3.00% |
| **LONGER** | **0.85290** | **0.47103** | **+1.57%** | **-3.39%** |

LONGER 相对 Base 提升 **+1.57% AUC**，相对最强基线 Transformer 再提升 **+0.21% AUC**（工业上 0.1% 即被视为显著提升）。

### 消融实验（关键组件与查询配置）
| Configuration | FLOPs (×10⁹) | AUC | ΔAUC | ΔLogLoss |
|---|---|---|---|---|
| LONGER (w/o Merge, 2000) | 3.73 | 0.85111 | +1.36% | -3.00% |
| +TokenMerge4(Concat, 500) | 2.13 | 0.85232 | +1.51% | -3.31% |
| +TokenMerge8(Concat, 250) | 3.03 | 0.85291 | +1.58% | -3.48% |
| **+InnerTrans** | 3.52 | **0.85332** | **+1.63%** | **-3.50%** |

- **查询数量**：$k=100$ 是最优 trade-off 点（AUC 0.85290，只用 $k=250$ 的 54% FLOPs 即达到几乎相同效果）；
- **采样策略**：Recent 100（AUC 0.85290） > Uniform 100 > Recent50+Unif50 > Learnable 100（AUC 0.84946，最差）。说明用真实最近行为初始化 query 比可学习 token 更有效。

### Scaling Law 分析
统一形式：$y = \alpha x^\beta + \gamma$
- **序列长度**：随长度增加 AUC 单调提升，遵循幂律；深层模型收益更大但有 diminishing returns；
- **参数量**（固定 2 层、$L=2000$）：$R^2 = 0.987$，强幂律，无饱和迹象；
- **FLOPs**（变深度+长度，固定 $d=32$）：$R^2 = 0.967$，强幂律。

### 在线 A/B 测试
#### Douyin Ads（ADSS = 广告主得分，ADVV = 广告主价值）

| 广告形态 | ADSS | ADVV |
|---|---|---|
| Live Streaming | +1.063% | +1.168% |
| Short Video | +2.097% | +2.151% |
| Mall | +1.816% | +1.407% |

#### Douyin E-Commerce（Order/U、GMV/U）

| 形态 | Order/U | GMV/U |
|---|---|---|
| Live Streaming | +7.9222% | +6.5404% |
| Short Video | +4.6125% | +5.2771% |

直播带货的提升尤其显著。LONGER 已在字节数十个核心场景全量部署，影响数十亿用户。

## 工程启示
1. **Global Token + Causal Cross-Attn + KV Cache** 三件套是多候选打分场景下的经典组合：全局 token 作锚点、因果 mask 保证候选不可见、用户序列 K/V 缓存复用——三者协同把多候选推理成本压到可控。
2. **Token Merge 不是简单截断**：分组压缩 + 组内 InnerTrans，在降 FLOPs 的同时还能扩张参数，是"用算力换表达力"与"用参数换算力"的双向 trade-off 设计。
3. **Query 采样策略**：直接用 recent-$k$ 真实行为初始化 query，远优于可学习 token（与 Perceiver/Q-Former 的 learnable query 路线形成对比，推荐场景下"真实行为"信息量更高）。
4. **工业 scaling law 是可验证的**：序列长度、FLOPs、参数量三个维度均呈现 $R^2>0.96$ 的幂律，且未饱和——这给后续的算力投入提供了量化依据。
5. **全同步 GPU 训练 + 分层稀疏嵌入**（HBM/MEM/SSD）是绕开 PS 瓶颈、把 sparse 推荐搬到 GPU 上的实用路径。

## 结论摘要
LONGER 通过四类设计——全局 token 稳定注意力、Token Merge + InnerTrans 压缩长序列、Cross/Self-Causal 混合注意力、Jaguar 全同步训练 + 混合精度 + KV Cache Serving——把端到端超长序列建模（2000~10000 token）在工业级 GPU 上落地。在抖音广告与电商两大业务、Live/Short Video/Mall 多形态上，离线与在线 A/B 均取得显著正向收益，并验证了推荐系统的工业级 scaling law。目前已在字节数十个场景全量部署，服务数十亿用户。

---

## Q&A 讨论

### Q1：全局 Token 是每个特征一个 token 吗？

**不是"所有特征拼到 1 个 token"，也不是严格"一个特征一个 token"，而是"一个语义来源的聚合表征 = 1 个 token"。**

论文 Section 3.3 原文把全局 token 列为：target item representation tokens、learnable CLS tokens、UID embeddings、高阶压缩的用户-物品交互特征，且数量记为 $m$（"concatenating $m$ global tokens"）。由此可确定：
- 全局 token 有 $m$ 个，不是 1 个；
- 每个 token 对应"一个语义来源的聚合表征"：target item 的多稀疏特征（id/category/creator 等）先经 embedding 查表 + 融合（sum pooling 或 MLP）聚合成 **1 个 target representation → 1 个全局 token**；UID 1 个；CLS 1 个；高阶交互特征若干个。
- 论文未给出具体 $m$ 值，"每个来源聚合成 1 token"是基于工业实践和文本的合理推断。

### Q2：序列 Token 如何形成？是"特征 + 位置信息 拼接过 MLP 形成 Embedding"吗？

**基本正确，但要区分"加"和"拼接"两种位置注入方式。**

论文 Section 3.5.1 (Input Generation) 的流程：
1. **Item embedding**：历史 item 的稀疏特征经 embedding 查表 + 融合（论文未细化，工业上常用 sum/concat pooling）→ item embedding；
2. **加位置编码**：可学习的**绝对位置编码**——**加**到 item embedding（标准 transformer 习惯）；
3. **拼接 side info**：**绝对时间差特征**（target 时间 − 交互时间，"距离当前候选多远"）作为 side info **拼接**到 (item emb + pos emb)；
4. **过 MLP**：拼接后的高维向量经 MLP 投影到 $d$ 维 → 序列 token $\mathbf{h}_i \in \mathbb{R}^d$，全序列 $\mathbf{H} \in \mathbb{R}^{L \times d}$。

所以更精确的表述：**item 特征 embedding + 可学习位置编码（相加） + 时间差 side info（拼接）→ MLP → $d$ 维 token**。这与 STCA 论文里"video, action-type, position fused"的描述对应，LONGER 显式把时间差作为 side info 拼进去（STCA 消融里也有 time-delta side info，$+0.08\%$）。

### Q3：Target 信息如何引入？作为全局 Token？每个样本只对一个 Target 打分？

**(a) Target 如何引入？** → **作为全局 Token**。Section 3.3 明确把 "target item representation tokens" 列为全局 token 之一。Target 的特征先聚合成 representation，作为 1 个全局 token 拼到序列前。

**(b) 训练时每个样本只对一个 Target 打分？** → **基本是**。Problem Statement 里样本是 $(S_u, u_d, v, y)$，一个 (user, target) 对应一条样本、一个 BCE 损失。论文未提"同用户多 target 在一个 batch 内共享用户编码"的训练侧优化（这点与 STCA 的 RLB 形成对比）。

**(c) 推理时多候选共享？** → **是**。KV Cache Serving（Section 3.6.3）写明：用户序列 K/V 投影**算一次并缓存**，每个候选只算"自己的全局 token 与缓存用户序列"的注意力。所以推理时一次请求对多候选打分，用户序列只算一次。**但与 STCA 的 RLB 区别在于**：LONGER 只在**推理侧**做缓存复用，STCA 的 RLB 在**训练+推理两侧**都做共享，且训练侧连梯度都按 request 聚合保持无偏。

### Q4：LONGER 与 STCA+RLB+Ext 架构区别？都是 Cross+Full 混合吗？

**直觉只对了一半。两者都"先 cross、后 self/堆叠"，但混合成分与哲学不同，差别比看上去大。**

| 维度 | LONGER | STCA + RLB + Ext |
|---|---|---|
| 第一层 attention | Cross-Causal：Q=[$m$ 全局 + $k$ 采样] (长度 $m{+}k$)，KV=**全序列** $L$ → 把长序列**压缩**到 $m{+}k$ | Single-query Cross-Attn：Q=**只有 target**（1 个），KV=全序列 $L$ |
| 后续层 | Self-Causal × N：在压缩后的 $m{+}k$ token 上做**自注意力**（有二阶 token-token 交互） | **纯 cross-attn 堆叠** × M：每层都是 target→history，**完全没有 history-history 自注意力** |
| 是否真"混合" | ✅ 真·混合：Cross（压缩）+ Self（高阶） | ❌ 不是 Cross+Full 混合：**纯 Cross 堆叠** + target-conditioned query fusion；history self-attn 被完全移除 |
| 每层复杂度 | 第一层 $O((m{+}k) \cdot L)$；后续 $O((m{+}k)^2)$（短序列上） | **严格线性 $O(L)$**（单 query + 计算重排避免物化 $XW_K/XW_V$） |
| 序列压缩方式 | 靠 **Token Merge + Query 采样**（recent-$k$）把序列压短 | **不压缩序列**，靠单 query 让复杂度本身变线性，全序列保留 |
| History-History 二阶关系 | 压缩后 token 间有 self-attn 交互（原始全序列二阶靠 cross 隐式表达） | **完全放弃**，靠多层 cross 让 target 多次扫历史 + query fusion 补偿 |
| Target 引入 | 作为 1 个全局 token 拼到 query 前 | 作为唯一 query，每层通过 fusion 把低层输出拼回 query |
| 多 target 推理复用 | **仅推理侧** KV Cache（用户序列 K/V 缓存） | **训练+推理两侧** RLB（用户路径一次编码，同 request 多 target 共享，梯度按 request 聚合保持无偏） |
| 长度外推 | 未强调外推；部署 $L \approx 2000$ | **核心特性**：训练 $\sim$2k / 推理 10k，U-shaped Beta 采样，$\rho_{\text{extra}}=5$ |
| 训练框架 | Jaguar：全同步 GPU 训练，dense+sparse 统一 GPU 存储，分层 HBM/MEM/SSD | 未强调框架，依赖已有大规模基础设施 |
| 部署序列长度 | $\sim$2000 | $\sim$10000（5× 更长） |
| 位置编码 | 绝对时间差 side info + 可学习绝对位置 | 时间差 side info（消融 $+0.08\%$），未细说位置方案 |

**关键洞察**：

1. **两者都意识到"history-history 全自注意力太贵要砍"，但砍法不同**：
   - LONGER：保留压缩后 token 间的 self-attn，牺牲原始全序列的二阶关系（靠第一层 cross 隐式表达）；
   - STCA：完全移除 history-history，只留 target→history，靠多层堆叠 + query fusion 补偿表达力。
2. **Query 数量的根本差异**：LONGER 的 query 是 $m{+}k$ 个（含全局 + recent 采样），STCA 严格只有 target 1 个——这是 STCA 能做到严格 $O(L)$ 而 LONGER 第一层是 $O((m{+}k)L)$ 的根本原因。
3. **系统侧复用深度不同**：LONGER 只做推理 KV Cache；STCA 的 RLB 更彻底，训练侧也复用且保持无偏——这是 STCA 能把训练推到 2k、推理推到 10k 的关键系统支撑。
4. **长度哲学**：LONGER 走"压缩"路线（2000 级），STCA 走"线性 + 外推"路线（10000 级），STCA 的部署序列长度是 LONGER 的 5×。
5. **所以"都是 Cross+Full 混合"的直觉不成立**：LONGER 是 Cross+Self 真·混合；STCA 是**纯 Cross 堆叠**（没有 Full/Self-Attn over history），把它归为"混合"会掩盖其"以单 query 换线性复杂度"的核心设计。

### Q5：序列侧 Token 能否看到 Target token？

**不能。这是 KV Cache Serving 成立的设计前提。**

论文 Section 3.5.2 (Cross-Causal Attention) 原文：
> The causal mask design, on the one hand, maintains temporal relevance between sequence items. On the other hand, it ensures the invisibility from the sequence to the candidate item, enabling the KV Cache Serving mechanism.

具体 mask 设计：输入 $\mathbf{R}=[\mathbf{G}; \mathbf{H}]$（**全局 token 在前**，序列 token 在后），$\mathbf{Q}=\mathbf{O}=[\mathbf{G}; \mathbf{H}_S]$，因果 mask $M_{ij}=0$ if $j\geq i$, $-\infty$ otherwise。

- **Target 全局 token 作为 query**（位置 $i \leq m$）：能看到 K/V 中 $j \geq i$ 的所有位置 → **能看到全部序列 H**（含自己）；
- **序列 token 作为 query**（位置 $i > m$）：只能看到 $j \geq i$ → **看不到 $j \leq m$ 的 G（即 target）**。

**设计意图**：序列 token 的表征不依赖 target → 序列的 K/V 投影（$H W_K, H W_V$）与 target 无关 → **可以跨 candidate 缓存复用**。这是 KV Cache Serving 的根本。

**代价**：序列侧 token 的输出"不含 target 信息"——但不影响预测，因为**最终预测用的是 target 全局 token 的输出**（target 能看到序列，所以 target 输出蕴含 target-history 交互）。

### Q6：推理多候选共享 vs 训练单 target，是否在/离线不一致？

**严格说，数学上没有不一致；但训练效率不对称是 LONGER 的相对弱点。**

详细分析：
- **训练时**：每条样本 $(S_u, u_d, v, y)$ 独立 forward + BCE loss + backward，序列 H 的 $K_H, V_H$ 每次重算（论文未提训练侧 batch 内多 target 共享 H 编码）；
- **推理时**：$K_H, V_H$ 算一次缓存，多 candidate 共享；每个 candidate 只实时算自己的 $G_{\text{target}_c}$ 和"target query 与 $(K_{H,\text{cache}} + K_{G,\text{target}_c})$ 的 attention"。

**数学等价性证明**：因为序列 token 看不到 target（Q5），$K_H, V_H$ 与 target 无关 → 缓存复用与重算**结果完全一致** → 推理多 candidate 共享与训练单 target 处理在 forward 上**数学等价**，**没有在/离线 gap**。

**但效率不对称**：
- LONGER：训练 H 重算（或未提优化），推理 H 缓存——效率不对称但结果正确；
- STCA：训练+推理两侧都做 RLB，同 request 多 target 共享 H 编码，backward 时把多 target 梯度按 request 聚合保持无偏——这是 STCA 论文 explicit 强调的工程难点。

**结论**：用户担心的"在/离线不一致"在数学上**不成立**（结果等价）；但 LONGER 训练侧的效率是个相对弱点——论文未提训练侧 batch 内多 target 共享优化，可能确实在重算 H。这是 STCA RLB（训练推理双侧对齐）相对 LONGER 的关键工程优势。

### Q7：LONGER 第一层 cross 后续 full-attention？第一层使序列长度变短？

**对，第一层把序列从 $L/K$ 压到 $k$，后续在 $m+k$ 长度上做 causal self-attention。**

完整流程（结合 Token Merge）：
1. **Input**：序列 H 长度 $L$；
2. **Token Merge + InnerTrans**（input 阶段）：$L$ 个 token 分成 $L/K$ 组，每组 $K$ 个 token，经 InnerTrans 后合并成 $L/K$ 个 token → 长度 $L \to L/K$；
3. **第一层 Cross-Causal**：$\mathbf{Q}=[\mathbf{G}; \mathbf{H}_S]$（长度 $m+k$，$\mathbf{H}_S$ 是从 $L/K$ 长度序列采样 $k$ 个），$\mathbf{K}=\mathbf{V}=\mathbf{R}=[\mathbf{G}; \mathbf{H}_{\text{merged}}]$（长度 $m+L/K$）。输出长度 $=\mathbf{Q}$ 长度 $=m+k$。**这一层把 $L/K$ 长度的序列通过 query 采样 + attention 提取压缩到 $k$**；
4. **后续 N 层 Self-Causal**：在 $m+k$ 长度上做 causal self-attention，复杂度 $O((m+k)^2)$。

所以：
- "第一层 cross 后续 full-attention"**基本对**（更精确是 causal self-attention，但 $m+k \ll L$ 时 causal 与 full 差别可忽略）；
- "第一层使序列长度变短"**对**：从 $L/K$（甚至 $L$）压到 $k$，这是 LONGER 的核心压缩手段；
- **压缩不是简单截断**：每个采样 query token 通过 attention attend 整个全序列，吸收信息后成为压缩后的一个 token。

**与 STCA 的对比**：LONGER 用"采样 + cross"把序列物理压缩；STCA **不压缩序列**，靠单 query 让复杂度本身变线性，全序列保留到最后一层。

### Q8：STCA 中后面 history 如何更新？

**History 的 token embedding 在堆叠层中不更新；但每层的 $K_H, V_H$ 投影表征不同。**

STCA 每层都是 target→history cross-attention：
- 输入：target 表征 $T^{(l)}$，history $X$（原始 token embedding，**整个堆叠过程中不变**）；
- 计算：$T^{(l+1)} = \text{CrossAttn}(Q=T^{(l)}, K=X W_K^{(l)}, V=X W_V^{(l)})$；
- FFN：$T^{(l+1)} \leftarrow T^{(l+1)} + \text{SwiGLU}(T^{(l+1)})$；
- Query fusion：$T^{(l+1)} \leftarrow T^{(l+1)} + [\text{low layer outputs concat}]$。

关键点：
- **$X$ 在堆叠层中不变**：没有 self-attn 让 history token 互相影响，所以 history 的原始表征保持；
- **但 $K_H^{(l)} = X W_K^{(l)}, V_H^{(l)} = X W_V^{(l)}$ 每层不同**：每层用不同投影参数 $W_K^{(l)}, W_V^{(l)}$ 把 $X$ 投影成不同的"视图"，让 target 多次以不同视角扫描 history；
- 这相当于"target 用 M 个不同注意力头去多次扫描 history"，每次得到不同的 history-target 关系，再通过 query fusion 累积到 target 表征中。

**复杂度控制**：为了做到严格 $O(L)$，STCA 避免物化 $X W_K$ 和 $X W_V$（即不实际算出 $L \times d$ 的矩阵），而是通过计算重排让每层只算 target query（1 个）与 $X W_K^{(l)T}$ 的乘积。这正是 STCA 把复杂度从 $O(L^2)$ 降到 $O(L)$ 的核心。

**与 LONGER 的对比**：
- LONGER：history 在第一层 cross 后就被压缩到 $k$ 个 token，后续在压缩后序列上做 self-attn——**history 的"精华"在压缩后参与后续更新**；
- STCA：history 原始表征始终不变，**只通过 target 多次扫描 + 投影视角变化**来挖掘信息——history 不"进化"，但 target 对 history 的"理解"逐层加深。

### Q9（补充 Q4）：LONGER 与 STCA 的输出对比

| 维度 | LONGER | STCA |
|---|---|---|
| 最终输出 token 数 | $m+k$（全局 + 采样），但用于预测的是 **target 全局 token**（1 个/candidate） | **1**（只有 target query 的最终输出） |
| 输出维度 | target token 输出 $\in \mathbb{R}^d$ | $T_{\text{final}} \in \mathbb{R}^d$ |
| 物理含义 | target 经过 cross（吸收 history）+ self（与采样 token 交互）后的表征，蕴含"target 在该 user 历史下的相关性" | target 经过 M 次"扫描 history"（每次不同投影视角）+ query fusion 累积后的表征，蕴含"target 与 user 长序列的深层交互" |
| 最终分数 | target 输出过 prediction head（MLP + sigmoid）→ 标量分数 | $T_{\text{final}}$ 过 prediction head（MLP + sigmoid）→ 标量分数 |
| 分数数量 | **每个 candidate 1 个分数**（target 数量 = candidate 数） | **每个 candidate 1 个分数**（STCA 也是单 target 处理，多 target 时按 batch/request 处理） |
| 多余 token 的用途 | $m-1$ 个非 target 全局 token + $k$ 个采样 token 的输出，论文未明说用途，可能用于辅助 loss 或表征学习 | 无多余 token，所有输出都用于预测 |

**关键关系**：
- 两者最终都是"**每个候选 1 个分数**"——target 数量 = candidate 数 = 分数数；
- 但 LONGER 的输出 token 数远多于 1（$m+k$ 个），只有 target 全局 token 用于预测；STCA 的输出 token 数 = 1（严格），全部用于预测；
- 这反映了两者的设计哲学：LONGER 保留多 token 做高阶交互（self-attn），STCA 极简到单 token 做线性扫描。

### Q10：LONGER 与 STCA 在数据、特征组织及非架构层面的差异

**数据层面差异**

| 维度 | LONGER | STCA + RLB + Ext |
|---|---|---|
| 序列长度 | 训练 $L \approx 2000$，部署 $\sim$2000 | 训练 $\sim$2k，**推理 10k**（外推 5×，$\rho_{\text{extra}}=5$） |
| 训练数据规模 | 5.2B 样本，130 天日志 | 论文未明确样本量，部署 1 个月 A/B |
| 采样策略 | **采样 Query**：recent-$k$ / uniform / learnable，实验证明 **recent-$k=100$** 最优 | **采样序列长度**：U-shaped Beta 分布，训练时长度均值 2k，推理全长 10k |
| 业务/任务 | 抖音广告 CVR + 抖音电商（Live/Short Video/Mall 多形态） | 抖音主站视频推荐（finish/skip/head 三目标） |
| 基线对比 | Base, SumPooling, TWIN, DIN(Recent50), DIN, HSTU, Transformer | DIN, SASRec, HSTU, TWIN(10k retrieval-based) |

**特征组织层面差异**

| 维度 | LONGER | STCA |
|---|---|---|
| 序列 Token 形成 | item emb + **可学习绝对位置编码（add）** + **时间差 side info（concat）** → MLP → $d$ 维 token | video + action-type + position fused（论文未细化融合方式）+ **时间差 side info（消融 +0.08%）** |
| 位置信息 | 绝对时间差 + **可学习绝对位置编码**（两套都用） | 时间差 side info，位置编码方案未细说 |
| 全局/Target Token | **$m$ 个全局 token**：target representation + UID + CLS + 高阶压缩的用户-物品交互特征 | **极简**：target 是唯一 query，论文未明确提 UID/CLS/高阶交互等额外全局 token |
| Token Merge 机制 | **有**（核心组件）：相邻 $K$ 个 token 分组压缩 + 组内 InnerTrans 细化 | **无**（不压缩序列，靠单 query 让复杂度本身变线性） |
| Query Fusion | 未明确提 | **有**（核心组件）：target-conditioned query fusion，每层把低层输出拼回 query |

**除模型架构和 RLB 外的其他区别**

| 维度 | LONGER | STCA |
|---|---|---|
| 训练框架 | **Jaguar**：全同步 GPU 训练，dense+sparse 统一 GPU 存储，分层 HBM/MEM/SSD 嵌入，**取消外部 Parameter Server** | 依赖已有大规模基础设施，论文未强调框架创新 |
| 混合精度策略 | BF16/FP16 混合精度，关键模块保留高精度，**+18% 吞吐、-16% 训练时长、-18% 显存** | 论文未明确提混合精度细节 |
| 激活重计算 | Recompute 前向激活，以算换存（dense 层显存最高 -28%） | 论文未明确提 |
| 推理优化 | **KV Cache Serving**（仅推理侧，用户序列 K/V 缓存） | **RLB**（训练+推理双侧，更彻底，backward 梯度按 request 聚合保持无偏） |
| 长度外推策略 | 未强调外推，部署 $L \approx 2000$ | **核心特性**：train sparsely / infer densely，U-shaped Beta 采样做长度外推 |
| Scaling Law 验证 | 明确验证三维（序列长度 / 参数量 $R^2=0.987$ / FLOPs $R^2=0.967$）的幂律，未饱和 | 也验证 scaling law，但更关注"质量随长度单调提升"的工程价值 |
| 部署规模 | 字节数十个场景全量部署 | 抖音 + 抖音 Lite 部署 1 个月 |

**关键差异总结**

1. **数据采样哲学不同**：LONGER 在**空间维度采样**（从全序列采样 $k$ 个 query token，recent-$k$ 最优）；STCA 在**时间维度采样**（训练时采样序列长度，推理用全长）。两者是正交的采样维度，理论上可以组合（但两篇论文都没做）。
2. **特征组织的"重"vs"轻"**：LONGER 重（$m$ 个全局 token + Token Merge + InnerTrans + 双位置编码，特征工程更丰富）；STCA 轻（target 唯一 query + 时间差 side info，特征工程极简，靠架构深度 M 层 cross + query fusion 补偿）。
3. **训练框架是 LONGER 的独特贡献**：Jaguar 全同步 GPU 训练 + 分层稀疏嵌入（HBM/MEM/SSD）把 sparse 推荐搬到 GPU 上绕开 PS 瓶颈；STCA 论文未强调框架，可能依赖字节已有基础设施。
4. **推理优化的深度不同**：LONGER 仅推理侧 KV Cache；STCA 的 RLB 是**训练推理对称的共享机制**，训练侧 backward 梯度按 request 聚合保持无偏——这是 STCA 能把训练推到 2k、推理推到 10k 的关键系统支撑。RLB 不只是"训练侧的 KV Cache"，工程难度更高。
5. **长度哲学是根本分野**：LONGER 走"压缩"路线（Token Merge + Query 采样把 2k 压到 $m+k$，部署 2k）；STCA 走"线性 + 外推"路线（单 query 严格 $O(L)$ + train sparsely/infer densely，部署 10k）。这是两者最根本的路线选择差异，决定了后续所有架构/系统设计。
