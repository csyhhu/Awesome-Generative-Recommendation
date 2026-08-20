# The Case Against Generation for Retrieval: Discriminative Language Models as Effective Retrievers

> 来源：Meta，arXiv:2607.25346。骨干网络 Qwen3-0.6B。

## 综合理解

本文做 Retrieval，训练时使用 distillation 框架（Cross-Encoder 教师 → Two-Tower 学生）：

**Teacher 侧（Cross-Encoder）**
- 输入：文本化的 user + item，拼成一段 prompt；
- 输出：同一个 LLM 的 LM head 在两处复用：
  - **打分信号**：prompt 要求模型判断"item 是否与 user 相关、回答 yes/no"，取 next-token logits 中 yes 与 no 两个 token 的 **logit 差** $s_{\mathrm{CE}}=\ell_{\mathrm{yes}}-\ell_{\mathrm{no}}$，作为 Teacher Retrieval Score 用于蒸馏；
  - **辅助生成任务（NTP）**：在 user 条件下复述 item 文本 token，标准自回归 LM loss，**仅训练时用**，线上教师只出 yes/no 分。

**Student 侧（Two-Tower）**
- 输入：文本化的 user、item，分别独立送入**共享** LLM 编码器（两次 forward，user/item token 不交互）；
- 表达：各自取 **EOS token 隐状态**作表征 $\mathbf{z}_u, \mathbf{z}_i$；user tower 可加一步 Coconut latent reasoning 进一步精化（item tower 保持标准、可预算）；
- 分数：**点积** $s_{\mathrm{TT}}=\mathbf{z}_u^\top \mathbf{z}_i$ 作为 Student Retrieval Score；
- 蒸馏：用**候选集分数分布 KL** 把 Teacher 的相对排序偏好传给 Student；
- 线上：item 表征**离线预算 + ANN 加速**检索，user 表征在线算一次。

> 精化两处：① "两个头"实为同一个 LM head 在不同位置/目标下复用，yes/no 取的是 logit 差而非概率；② ANN 只用于线上 serving 检索加速，蒸馏训练时直接算点积 + KL，不走 ANN。

## 理解点评

对"综合理解"小节里那段原始描述的逐句勘误：

| # | 原描述 | 评价 | 说明 |
|---|---|---|---|
| 1 | 本文做 Retrieval，训练时使用 distillation 框架 | ✓ 正确 | — |
| 2 | Teacher 侧输入为文本化的 user, item | ✓ 正确 | 拼成一段 prompt |
| 3 | 输出有两个头 | ⚠️ 不准确 | 不是两个独立头，是**同一个 LM head**（到词表的输出投影）在两处复用：yes/no 在 prompt 末尾位置，NTP 在 item 文本段位置 |
| 4 | 一个头是判断为 yes 的概率，一个为判断为 no 的概率 | ⚠️ 两处不精确 | ① 同 #3，不是两个头；② 用的是 **logits** 不是概率。$s_{\mathrm{CE}}=\ell_{\mathrm{yes}}-\ell_{\mathrm{no}}$ 是 next-token logits 中两 token 的差值。logit 差 ≠ 概率差（差一个 softmax/温度缩放） |
| 5 | 其差值作为 Teacher Retrieval Score，用于 distillation | ✓ 正确 | — |
| 6 | 另一个头根据 user 输入，输出 item，是辅助生成任务 | ⚠️ 措辞误导 | "另一个头"同上不准确；本质是同一个 LM head 在 item 段做自回归 LM loss，让模型在 user 条件下复述 item 文本 token。**辅助生成任务**定性正确，仅训练时用、线上不出 |
| 7 | Student 侧输入同样为文本化的 user, item | ✓ 正确 | — |
| 8 | 两个内容分别进入 Student 模型 | ✓ 正确 | 两次独立 forward，user/item token 不交互 |
| 9 | 最后的 embedding 或者 EOS token 作为各自表达 | ⚠️ 不精确 | 不是"二选一"——EOS token 的末隐状态**就是**该序列的 embedding（pooled representation）。另外**漏了 user tower 的 Coconut latent reasoning**：user 表征不是直接取 EOS，而是先取末隐状态作连续 thought token $\mathbf{c}_u$，再过一遍 $[x_u;\mathbf{c}_u;\mathrm{EOS}]$ 取末隐状态。**item tower 才是直接取 EOS** |
| 10 | 然后使用 ANN 计算 Student Retrieval Score，用于 distillation | ✗ 实质性错误 | 把"训练蒸馏"和"线上检索"混在一起：<br>① Student Retrieval Score 是**点积** $s_{\mathrm{TT}}=\mathbf{z}_u^\top\mathbf{z}_i$，训练时**直接计算**；<br>② 蒸馏用候选集分数分布 **KL**，也是直接算；<br>③ **ANN 只在线上 serving** 做 kNN 检索加速（item 表征预算建索引，user 在线算一次后查 ANN 取 top-K） |

**汇总**：6 处正确，4 处需精化，1 处实质性错误。

**最关键的纠正**：

- **#10（必须改）**：ANN 是 serving-time 检索加速工具，与 distillation 训练无关。蒸馏时 teacher/student 分数都直接算（CE 是 logit 差，TT 是点积），再用 KL 拉齐候选集分布。ANN 只在"线上 item 表征预算→建索引→user 在线算一次→查 ANN 取 top-K"路径里出现。
- **#3/#4/#6 的"两个头"**：是同一个 LM head 复用——共用 $f_\theta^{\mathrm{CE}}$ 和同一个输出到词表的投影，NTP 只是一个 loss 项而非新模块，加 $\lambda_{\mathrm{ntp}}=0.5$ 权重即可。
- **#9 漏的 Coconut latent reasoning**：Student 侧 user tower 多算一步 latent 精化表达，是论文最终保留的创新点。原描述把它简化成"标准双塔 + EOS pooling"，漏掉了这个设计。

## 动机

在 LLM 时代，推荐系统大量探索将推荐重构为语言建模/生成问题（P5、TIGER、LLMRank、TALLRec、PLUM、OneRec 系列等），但这些**生成式或交互密集型 LLM 推荐器**在大规模检索场景下遭遇严重瓶颈：

1. **服务成本高昂**：自回归解码延迟大，难以满足实时检索的延迟预算。
2. **Grounding 问题**：生成模型输出的是离散文本 token，token→item 的翻译必须在模型外部完成，这种结构性分离会让对齐误差在推理后级联放大。
3. **难以并行**：输出序列长度 $N_{\mathrm{gen}}$ 增加，prefill 后需要 $N_{\mathrm{gen}}-1$ 步串行解码，延迟随序列长度增长。

与之相对，**Two-Tower（双塔）架构**仍是工业检索的基石：user/item 独立编码、item 表征可离线预算、ANN 检索高效。本文核心问题：**能否用 LLM 时代的语义深度振兴经典双塔，同时保留其生产级效率与可扩展性？** 作者明确主张"反对为检索而生成"（the case against generation for retrieval），即判别式 LLM 表征 + 因式分解双塔，是大规模推荐更实用、更经济的路径。

## 方法

整体采用 **Teacher–Student** 蒸馏框架：Cross-Encoder（CE）教师提供高质量排序信号，Two-Tower（TT）学生作为高效检索器吸收教师知识。两条改进路线相互独立又可叠加累积。

### 1. 改进 Cross-Encoder 教师

教师决定蒸馏软标签的上限，从三方面增强（Dec2Enc 最终被排除）：

- **Yes/No 输出头**：构造 prompt 让模型判断 item 是否与 user 相关、回答 "yes"/"no"，用 next-token logits 中两个标签的差值作为相关性分数 $s_{\mathrm{CE}}(u,i)=\ell_{\mathrm{yes}}-\ell_{\mathrm{no}}$。相比 projection head，verbalized 评分更贴合 LLM 的预训练分布。
- **User-Conditioned NTP 辅助损失**：对 item 文本做自回归预测，但**始终以 user 文本为条件**，使模型在用户偏好语境下学习 item 语义。$\mathcal{L}_{\mathrm{CE}}=\mathcal{L}_{\mathrm{con}}(s_{\mathrm{CE}})+\lambda_{\mathrm{ntp}}\mathcal{L}_{\mathrm{ntp}}$，$\lambda_{\mathrm{ntp}}=0.5$。NTP 解决了 yes/no head 在 Sports 数据集上单独使用反而退化的问题。
- **Dec2Enc（评估后排除）**：将 causal mask 换成双向 mask 的尝试**未稳定带来收益**，常同时降低 CE 与 TT，故最终方法不含此项。

### 2. 增强 Two-Tower 学生

- **Shared User-Item Encoder**：user 与 item 共享同一 LLM 编码器 $f_\theta$，参数共享促使双方嵌入同一语义空间，且参数量更少。
- **EOS Pooling**：取序列末尾 EOS token 的隐状态作为序列表征（$\mathbf{z}_x=\mathbf{h}_{x,L_x}$）。对 decoder-only LLM，因 causal attention 末 token 可聚合全序列信息，效果优于 mean pooling。
- **Cross-Dataset Transfer Learning**：先在全部 R 个数据集上 mid-train（$\mathcal{L}_{\mathrm{mid}}=\frac{1}{R}\sum_r \mathcal{L}_{\mathrm{con}}^{(r)}$），再在目标数据集 fine-tune，迁移可复用的行为/文本模式。
- **CE2TT 蒸馏（候选集分数分布蒸馏）**：对同一候选集 $\mathcal{C}_u$，用温度 $T$ 将 CE 与 TT 的分数归一化为分布，最小化两者 KL 散度 $\mathcal{L}_{\mathrm{KD}}=T^2\sum_u \mathrm{KL}(q_{\mathrm{CE}}\,\|\,p_\theta^{\mathrm{TT}})$。**只迁移相对排序偏好**，不直接对齐 EOS 表征（后者用 $L_2$ 匹配反而退化）。
- **Coconut 式 User-Tower 隐式推理**：只在 user tower 加一步 latent reasoning。先编码 user 文本（不追加 EOS）取末隐状态作连续 thought token $\mathbf{c}_u$，再以 $[x_u;\mathbf{c}_u;\mathrm{EOS}]$ 再过一次，取末隐状态为 $\mathbf{z}_u$。**item tower 不变**，item 表征仍可预算；user 表征更具表达力而检索效率不损。多步 latent 不带来增益且显存开销大，故仅用一步。

### 3. 效率与服务架构

- 在线只需一次 user 编码（开 latent reasoning 则多一步缓存解码）+ 对预算好 item 索引的 MIP 检索，替代迭代式生成。
- **服务架构**分三部分：item 表征 nearline 异步生成并实时索引；user 表征周期性重算并缓存到分布式 KV；在线层从 KV 取 user embedding 做 kNN（用 embedding 压缩加速），数百毫秒内返回候选给下游排序。

## 实验结果

### 公开数据集（Amazon Beauty / Sports / Toys，沿用 TIGER 协议）

骨干统一为 **Qwen3-0.6B**，而基线 ORT（OneRec-Think）用 **Qwen3-8B**。

- **SOTA 对比**：本文 CE 在 12 项指标-数据集组合中 **10 项最优**；TT 学生在三个数据集的 R@10 上**全部超越 ORT**（+4.3%、+31.6%、+35.3%），NDCG 不够一致。CE 在 Sports/Toys 的 R@10 较 ORT 提升 64.3%/53.5%。
- **CE 消融**：yes/no+NTP 在全部 12 组合上最优；yes/no head 单独在 Sports 上退化，NTP 补齐了该不一致。
- **TT 消融**（leave-one-out）：**CE2TT 最关键**，移除后 R@10 在三数据集分别下降 13.3%/23.1%/8.0%；TL 与 latent 提供较小但互补的增益，移除任一都会让 R@10 下降。完整模型在所有数据集的 R@10/N@10 均最优。
- 完整组合消融见附录：vanilla（分离编码+mean pooling）→ Shared+EOS 单项即把 Beauty R@10 从 0.0389 提到 0.0672。

### 内部生产数据（vs 重度调优的 DLRM Two-Tower）

主指标为 Normalized Entropy（NE，越低越好），报告相对 NE gain。

- **数据效率**：LLM-native TT 用**仅 0.5% 训练数据**即追平 DLRM NE；LLM-native CE 仅用 **0.15%** 数据即达 parity，且 NE 比 baseline 好 +2.25%。
- **分群分析**：tail items（底部 15%）NE +5.5%，head users（顶部 12%）NE +2.3%——语义表征在稀疏历史与长历史两端均有收益。
- **抗 staleness**：冻结 3 天评估，冻结 DLRM 在 ds+2/ds+3 分别 -2.68%/-4.30%，**冻结 LLM-Native CE 仍 +2.21%/+2.23%**，源于稳定 token 词汇表而非易变的 item ID 分布。
- **数据 scaling**：2x/3x 数据，DLRM +1.30%/+1.80%，LLM-native CE +1.59%/+2.19%，scaling 曲线更优。
- **服务效率优化**：数值特征 FSQ 压缩 +19.7% QPS（且 NE +0.3%）、深度剪枝 3/28 层 +10.6%、静态词表剪枝 +2.0%、FP8 量化 +13.6%（NE 退化约 0.003%）。
- **容量上限（超服务预算）**：latent reasoning +0.41% NE、Mixtral MoE(8 expert top-2) +0.91%、4B 骨干 +1.90%，显示明显 headroom。

## 意义

1. **对"为检索而生成"提出反例**：在 web-scale 检索场景，判别式 LLM 表征 + 经典双塔比生成式检索更实用、更经济——一次 user 编码 + 向量检索替代自回归解码，天然适配低延迟大候选集。
2. **方法论组合可叠加**：Shared+EOS、跨数据集迁移、CE2TT 候选集分数分布蒸馏、user-tower 单步 latent reasoning 各自独立有效，且消融显示 CE2TT 是最关键的迁移机制（匹配 EOS 表征反而有害）。
3. **工业落地的关键洞察**：LLM-native 模型以极小比例数据即追平 DLRM、对 staleness 鲁棒、scaling 更优，根源在于**稳定的语义 token 词汇表 vs 易变 item ID 依赖**。这为把 LLM-native 范式嵌入工业检索系统提供了可操作路径，而非单纯追求更大的生成模型。
4. **诚实排除无效设计**：Dec2Enc、item-tower latent、多步 latent、EOS 表征 $L_2$ 匹配均经实验评估后排除，体现了对"看似合理但无收益"设计的克制。

## 讨论 Q&A

### Q1：User/Item Tower 在 Teacher/Student 侧的结构与输入输出？Student 的输入是否是传统特征表达？

**结论：Teacher 与 Student 的输入都是文本（verbalized），Student 不使用传统特征向量表达。** 这是 "LLM-native" 的核心设计——所有特征先被语言化/Token 化再喂 LLM。

**Teacher（CE）侧**
- 输入：文本 prompt $x_{u,i}^{\mathrm{yn}}=\mathrm{Prompt}(x_u,x_i)$，要求模型判断相关性并回答 yes/no。
- 结构：单个共享 LLM 编码器，对 $[x_u;x_i]$ 做 joint encoding（user/item token 全 self-attention 交互）。
- 输出：next-token logits，$s_{\mathrm{CE}}=\ell_{\mathrm{yes}}-\ell_{\mathrm{no}}$；另带 user-conditioned NTP 头预测 item 文本。
- 不可预算：$\mathbf{h}_{u,i}$ 依赖具体 $(u,i)$ 对，无法做 ANN。

**Student（TT）侧**
- 输入：$x_u$ 与 $x_i$ 分别独立送入（两次 forward，user/item token 不交互）。
- 结构：user/item 共享同一 LLM 编码器 $f_\theta$。
  - Item tower（标准、可预算）：$f_\theta(x_i)\to$ EOS 隐状态 $\mathbf{z}_i$，离线预算 + ANN 索引。
  - User tower（Coconut 式单步 latent reasoning，不可预算但在线算一次）：
    1. $g_\theta(x_u)$（不追加 EOS）→ 末隐状态作连续 thought token $\mathbf{c}_u$；
    2. $g_\theta([x_u;\mathbf{c}_u;\mathrm{EOS}])$ → 末隐状态 $\mathbf{z}_u$。
- 输出：$s_{\mathrm{TT}}=\mathbf{z}_u^\top\mathbf{z}_i$（点积）。

**内部生产数据的"传统特征"如何处理**：同样全部 Token 化后送 LLM，不走 DLRM 的 ID embedding 拼 dense feature。
- 连续特征 → FSQ（64 bin）→ soft token；
- 遗留 dense embedding → Q-Former → 可学习 query token；
- ID/事件序列 → verbalize 成短文本片段；
- 整体 8K-token 窗口。

→ 这正是 LLM-native TT 在 0.5% 数据下追平 DLRM 的根因：语义 token 词汇表稳定可迁移，而 DLRM 的 item-ID embedding 易变。

### Q2：本文是否认为"Generative Retrieval 效果更好只是太贵"？有无离线证据证明 Teacher > Student？

**先纠正表述**：本文并未主张"生成式检索效果更好"。它把 OneRec-Think（Qwen3-8B）当作对手基线，证明自己的判别式方法（仅 Qwen3-0.6B）在多数指标上反超生成式检索。论文立场是"生成式有 grounding/延迟/串行解码等结构性问题，且效果并不更优"。

但"Teacher 是否比 Student 好"——**有明确的离线证据**，这正是蒸馏动机（teacher 是上界）：

**公开数据集（Table 1，同骨干 Qwen3-0.6B）**

| 数据集 | CE（Teacher）R@10 | TT（Student）R@10 | Teacher 优势 |
|---|---|---|---|
| Beauty | 0.0957 | 0.0825 | +16% |
| Sports | 0.0677 | 0.0542 | +25% |
| Toys | 0.1223 | 0.1078 | +13% |

CE 在 12 项指标-数据集组合中 10 项最优，TT 在 NDCG 上 "less consistent"。

**内部生产数据**
- LLM-native CE：NE 相比 DLRM baseline **+2.25%**；
- LLM-native TT：仅**追平** DLRM baseline（NE parity）；
- ⇒ CE 相对 TT 还有 +2.25% headroom，CE2TT 蒸馏就是要传递它（消融显示 CE2TT 是 TT 学生最关键组件，移除后 R@10 降 8%~23%）。

**既然 Teacher 更好为何还用 Student**：CE 不能上线做检索——对每个 $(u,i)$ 对都要 joint encoding，item 表征无法预算，线上对全候选各跑一次 forward 扛不住规模。TT 的 item 表征离线预算 + ANN，在线只算一次 user 编码（开 latent 多一步缓存解码）。

**逻辑链**：CE 离线更强（upper bound）→ 太贵不能上线 → 蒸馏到 TT → TT 逼近 CE 且可预算检索 → 仍反超生成式检索基线。

这与"生成式检索更好但太贵"是两回事：前者（CE vs TT）是判别式框架内的 teacher-student 权衡；后者（判别 vs 生成）论文给出的结论是判别式在效果和效率上都不输。

### Q3："另带 user-conditioned NTP 头预测 item 文本"——是不是有个单独的头根据 user 输入预测 item？

**不是单独的头。是同一个 CE 教师网络，共用 LLM 的 LM head（词表投影），NTP 只是一个辅助训练目标。**

$$\mathcal{L}_{\mathrm{ntp}} = -\sum_{(u,i)\in\mathcal{S}}\sum_{\ell=1}^{L_i-1}\log p_\theta\big(t_{i,\ell+1}\mid x_u, t_{i,1},\ldots,t_{i,\ell}\big)$$

训练时把序列拼成 `[user 文本 x_u][item 文本 t_{i,1..L_i}]`，**只在 item 段算标准自回归 LM loss**——给定用户上下文，让模型"复述/描述出这个 item 的文本"。共用 LLM 的输出头，并不是单独加一个网络。

- **yes/no 头**：prompt 末尾要求回答 yes/no，取两 token 的 logit 差作分数。判别式打分。
- **NTP 辅助**：item 段做语言建模。生成式辅助任务，**仅用于训练**——线上 CE 教师只出 yes/no 分数，从不真的"生成 item 文本"。

两个"头"是同一个 LM head 在不同位置/不同目标下复用，$\lambda_{\mathrm{ntp}}=0.5$ 控制辅助权重。

### Q4：输入都是文本，具体形式长什么样？是 {user_id, exposed item ids...} 那种吗？

**公开数据集（Amazon）：不用 user_id，也不把 item_id 当 embedding 查表，完全用 verbalized 元数据。**

**User 侧**（两段）：
1. Recent items（最近 5 个），每个格式化为 `[Brand] Title (LeafCategory, L2Category) Price Rating`
   例：`[Nike] Air Zoom Pegasus (Running Shoes, Athletic Footwear) $89.99 4.5`
2. Taste summary（≥3 历史才有）：top-3 L2 类目带百分比（≥15% 才报）、top-2 品牌、价格区间。
   例：`Top categories: Athletic Footwear 45%, Apparel 20%; Top brands: Nike, Adidas; Price range: $20–$100`

**Item 侧**：
- 核心字段：ASIN、title、leaf category、L2 category、brand、price、平均评分及评分数；
- Description snippet：商品描述前 100 字符（按词截断）；
- Also-bought signal：最多 3 个共买商品，`Also bought: [Brand] Title (LeafCat) | ...`。

输入完全是自然语言 + 结构化字段 verbalize，没有 `user_id=xxx`。ASIN 出现在 item 侧，但作为文本 token 而非 embedding lookup。

**内部生产数据**：结构化"块"序列，但同样最终 token 化：
- 整体 `[user block][item block][task block]`，8K-token 窗口；
- 连续特征 → FSQ（64 bin）→ soft token；
- 遗留 dense embedding → Q-Former → 一组可学习 query token；
- ID/事件序列 → verbalize 成短文本片段。

"ID"在内部是有的，但**不是 DLRM 那种 ID→embedding 查表**，而是把 ID/事件 verbalize 成文本片段送 LLM。这是 LLM-native 与 DLRM 在表征层最本质的差别。

### Q5：OneRec-Think 是不是和 Teacher 类似，都是直接 predict next token as exposed item？数据上加了更复杂思维？

**直觉部分对，但要分清"训练目标"和"线上推理"。**

**训练目标层面：确有相似**——两者都有"由 user 上下文预测 item token"的训练信号：
- OneRec-Think：Itemic Alignment 把 item 映到 LLM token 空间，含"序列偏好建模"+"项目密集描述"任务；
- 本文 Teacher：NTP 辅助损失让模型在 user 条件下复述 item 文本 token。

**线上推理层面：本质不同**：

| 维度 | OneRec-Think | 本文 Teacher (CE) |
|---|---|---|
| 推理范式 | **生成式检索**：自回归解码出 [推理文本 + item token 序列] | **判别式打分**：给定 (user, item) 对，输出 yes/no logit 差 |
| 是否生成 item | 是，item 由生成的 token 序列对齐回 item 库 | 否，item 外部给定，模型只打分 |
| 推理形态 | 多步串行解码，序列越长延迟越高 | 一次 forward 给分数，但每对 (u,i) 都要跑一次 |
| 骨干 | Qwen3-8B | Qwen3-0.6B |
| Reasoning 形态 | **显式 in-text reasoning**（CoT 风格，解码出可见推理文本）+ Think-Ahead 两阶段架构 | **隐式 latent reasoning**（Coconut 式连续 thought token，不解码文本）— 仅 Student (TT) 侧 |

**关键区别**：OneRec-Think 真的在线把 item token 一个个 decode 出来（生成式检索）；本文 Teacher 不生成 item，只对给定 item 打 yes/no 分（判别式 ranker）。即便两者训练时都有"由 user 预测 item"的信号，**线上用法完全不同**。

- OneRec-Think：`generate(reasoning + item_tokens)` → 检索结果
- 本文 Teacher：`score(user, item) → yes/no logit diff` → 排序分

这也是本文敢用 0.6B 反超 8B OneRec-Think 的根因——它不需要把"生成"塞进推理路径，避免自回归串行延迟。Teacher 太贵不能上线不是因为"生成"，而是因为**对每对 (u,i) 都要 joint encoding、item 无法预算**；这才是蒸馏到 TT 的动机。

Reasoning 形态也别忽略：OneRec-Think 是显式可解释的 in-text CoT（用 GRPO 强化训练推理质量）；本文是隐式省延迟的 Coconut latent（只一步、只在 user tower）。

### Q6：Coconut latent reasoning 是什么？本文如何用？

**原始 Coconut（Hao et al. 2024，Chain of Continuous Thought）**：把 Chain-of-Thought 中的**离散自然语言推理 token** 替换成**连续值的 latent token**，让模型在连续隐空间里"思考"，不必解码出可见文字推理步骤。

- 显式 CoT：`[Question] → "Let me think..." → [Answer]`，中间推理是可见文字 token，每步要 decode。
- Coconut：`[Question] → [连续 thought c₁] → [连续 thought c₂] → ... → [Answer]`，中间推理是连续 d 维向量，不 decode 成词。

**连续 thought token 怎么"塞进去"**：取 LLM 某次 forward 的**最后一个 hidden state**（d 维向量），直接当作一个"伪 embedding token"拼到输入 embedding 序列，再跑下一次 forward。该向量**不经过 unembedding**（到词表的投影），所以不对应任何具体单词。由于 LLM 的 hidden dim = embedding dim，维度天然对齐。原 Coconut 用多阶段课程学习训练（先用离散 CoT，再逐步替换成连续 latent）。

**本文用法（大幅简化版）**：只一步、只在 user tower、无课程学习。记 $g_\theta$ 为共享 LLM 编码器（pooling 前）：

- **第一步（生成 thought）**：$\mathbf{H}_u^{(0)} = g_\theta(x_u)$，**不追加 EOS**，取末隐状态 $\mathbf{c}_u = \mathbf{H}_u^{(0)}[-1]$。
- **第二步（用 thought 精化 user 表征）**：$\mathbf{H}_u^{(1)} = g_\theta([x_u;\,\mathbf{c}_u;\,\mathrm{EOS}])$，取末隐状态 $\mathbf{z}_u = \mathbf{H}_u^{(1)}[-1]$。

第二次 forward 的输入 embedding 序列：`[embed(x_u tokens), c_u (直接当 embedding 用), embed(EOS)]`。

**关键设计权衡（非对称）**：

| 设计点 | 本文选择 | 原因 |
|---|---|---|
| 只在 user tower | item tower 仍标准 EOS pooling | item 表征要离线预算 + ANN 索引；user 在线算一次就行 |
| 只一步 latent | 多步被附录排除 | 多步 latent 无增益，且显存开销大、计算图变深 |
| thought token 不 decode | 纯连续向量，不解码文字 | 省延迟，避免自回归串行解码 |
| 不在 item tower 加 latent | 附录试过无收益 | item 文本较简单（title/desc/price），无需多步精化 |

把"贵的那一步"只放在在线算一次的 user 侧，而必须离线预算的 item 侧保持标准——这是本文最巧的工程取舍。

**效果（偏弱）**：
- 公开数据集 leave-one-out：去掉 latent 后 R@10 在三数据集都下降（Beauty 0.0825→0.0794，Sports 0.0542→0.0530，Toys 0.1078→0.1063），但 **Beauty N@5 反而升一点**（0.0267→0.0269）——增益不绝对一致。
- 内部生产：+0.41% NE（residual-stream variant），相对其他组件增益偏小。

**与 OneRec-Think 的 reasoning 对比**：

| 维度 | OneRec-Think | 本文 user tower latent |
|---|---|---|
| 形态 | 显式 in-text CoT（解码可见推理文字） | 隐式 continuous latent（不 decode） |
| 训练 | RL（GRPO）训练推理质量 | 端到端随主任务训，无专门 RL |
| 步数 | 多步（推理链可长） | 固定一步 |
| 部署 | Think-Ahead 两阶段：离线生成 reasoning + 在线用前缀约束 | 在线多算一步 cached forward |
| 可解释性 | 高（推理可见） | 无（latent 不可读） |

**点评**：本文对 Coconut 的"借鉴"是非常浅的版本——原 Coconut 的多步连续推理 + 课程训练被简化成"一步 latent + 端到端训"。工程上零负担（就多一次 forward），但 +0.41% NE 和 Beauty N@5 反向的事实说明这个组件**更像是"凑数的微创新"而非关键设计**。论文最终保留它可能是为了 completeness，消融证据偏弱。

### Q7：EOS 是什么？为什么 embed(EOS) 能编码 user/item？为什么不用开头的 [CLS]？

**EOS = End Of Sequence**，词表里的一个特殊 token（Qwen3 里是 `<|endoftext|>` 或 `<|im_end|>`）。就是个普通词表条目，有 token ID，能像任何词一样查 embedding 矩阵得到它的 input embedding。

**关键区分：embed(EOS) ≠ EOS 位置的 hidden state。** 容易混的就是这里——

| 概念 | 是什么 | 用作表征吗 |
|---|---|---|
| **embed(EOS)** | EOS 的**输入 embedding**（embedding 矩阵查表的固定向量） | ❌ 不是 |
| **EOS 位置的 hidden state** | 整个序列过完 transformer 后，EOS 位置经所有层后的**输出隐状态** | ✅ 是这个 |

"EOS pooling" 指的是后者——取 **EOS 位置的 output hidden state**，不是 EOS 的 input embedding。EOS 只是个放在末尾的"哨兵 token"，本身不带 user/item 信息；带信息的是它**经过 transformer 后**的隐状态。

```
输入序列：[tok_1, tok_2, ..., tok_L, EOS]
                              ↓ 经过 N 层 transformer（含 self-attention）
输出隐状态：[h_1,   h_2,   ..., h_L,   h_EOS]
                                        ↑ 取这个作序列表征 z
```

$h_{\mathrm{EOS}}$ 能代表整个序列，是因为 **self-attention 让它"看见"了前面所有 token**。

**为什么用末尾的 EOS 而不是开头的 [CLS]**：这是 **causal attention mask** 决定的。

Decoder-only LLM（Qwen3 等）用 causal mask：位置 $i$ 的 token 只能 attend 到位置 $\le i$ 的 token。

| 位置 | 能 attend 到 |
|---|---|
| tok_1（开头） | 只能看 tok_1 ← **看不到后面任何东西** |
| tok_2 | tok_1, tok_2 |
| EOS（末尾） | tok_1, tok_2, ..., tok_L, EOS ← **能看到全部** |

所以：
- **末尾 token**（EOS）的 hidden state 能 attend 到前面所有 token → 能聚合整个序列 → 适合做表征。
- **开头 token**（[CLS]）的 hidden state 只能 attend 到自己 → 几乎不含序列信息 → 不适合。

这就是 decoder-only LLM 用 **EOS（末尾）pooling** 而非 [CLS]（开头）pooling 的根本原因。

**[CLS] 在什么情况下能用**：bidirectional encoder（BERT 这类）去掉 causal mask，所有 token 互相可见，[CLS] 放开头也能 attend 到后面所有 token。BERT 就是这么干的。

所以 [CLS] vs EOS 的选择**本质是 attention mask 决定**：
- causal mask（decoder）→ 用末尾（EOS）
- bidirectional mask（encoder）→ 用开头（[CLS]）也行，用末尾也行，用 mean 也行

**回到本文**：用 Qwen3-0.6B（decoder-only, causal mask），所以必须用 EOS pooling。论文明确写："EOS pooling is commonly used when adapting decoder-style LLMs into embedding models, since the final token can aggregate information from the preceding sequence under causal attention"。附录试过 Dec2Enc（去掉 causal mask 变 bidirectional，类似 [CLS] 思路），但未稳定带来收益，最终排除。

**回看 Q6 Coconut latent 第二步**：`[embed(x_u tokens), c_u, embed(EOS)]` 是**输入 embedding 序列**。$g_\theta$ 跑完后取 EOS 位置的 **output** hidden state 作 $\mathbf{z}_u$。$\mathbf{c}_u$ 是上一步末 hidden state 当伪 embedding 拼进**输入**；$\mathbf{z}_u$ 是 EOS 位置的**输出** hidden state，能 attend 到 user 文本 token + $\mathbf{c}_u$，所以聚合了"原始 user 信息 + 一步 latent 思考"两层信息。这也是 $\mathbf{z}_u$ 比直接 EOS pooling 多一点表达力的来源。

### Q8：那其实就是把文本化的 user/item 扔进模型后吐出来的第一个 token 的 embedding？

方向对了，但两点要纠正：

**① 不是"第一个"，是"最后一个"（EOS 位置）**

| 位置 | hidden state 能 attend 到 | 含序列信息 |
|---|---|---|
| 开头第一个（tok_1） | 只能看自己 | ❌ 几乎不含 |
| 末尾 EOS | 全部 token | ✅ 含整条序列 |

所以取的是末尾 EOS 位置的 hidden state，不是开头第一个。

**② 不是 token 的"embedding"，是该位置的 output hidden state**

```
输入端：[embed(tok_1), embed(tok_2), ..., embed(tok_L), embed(EOS)]   ← 查表得到的固定向量
                              ↓ transformer N 层（self-attention 让位置间互相"看"）
输出端：[h_1,        h_2,        ..., h_L,        h_EOS]            ← 经过处理的隐状态
                                                      ↑ 取这个当 user/item 表征
```

- "token 的 embedding" = 输入端从 embedding 矩阵查表得到的固定向量，与上下文无关；
- 我们取的是 transformer 跑完后该位置的**输出隐状态**，已过 N 层 self-attention + FFN，含整条序列信息。

**③ 关于"吐出来"——不是生成，是单次 forward**

这里不是 autoregressive 生成（模型不在 decode 新 token）。是一次 forward pass，所有位置 hidden state 同时算出，我们从输出里"挑"EOS 位置那个。

**有趣的联系**：在 decoder-only LLM 里，最后位置（EOS）的 hidden state 就是"如果让它继续生成，它用来预测下一个 token 的那个向量"——$h_{\mathrm{EOS}}$ 经过 LM head（到词表投影）就能预测下一个 token。所以 EOS pooling 的本质是**借用这个本该用来预测下一个 token 的 hidden state，把它当序列表征**。"吐出来的第一个 token"在直觉上接近，但更准确的说法是"用来生成下一个 token 的 hidden state"，不是真的 decode 出来的 token。

**一句话总结**：把 user/item 文本拼成 `[tok_1, ..., tok_L, EOS]`，**一次性 forward**，取**末尾 EOS 位置的输出 hidden state** 当这个 user/item 的 d 维表征，再点积算相似度。
