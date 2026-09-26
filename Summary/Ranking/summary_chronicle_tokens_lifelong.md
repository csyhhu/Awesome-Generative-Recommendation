# ChronicleRec: Pre-training Cacheable Chronicle Tokens for Lifelong User Interest Modeling

- arXiv: 2609.12375
- 机构: Tencent Inc. / University of New South Wales
- 场景: 超长用户行为序列建模（推荐 & 在线广告 CTR/CVR 预估）

## 一句话总结

把超长用户历史"预训练压缩一次"成一小组**按时间顺序排列、causal 编码**的可缓存 Chronicle Tokens，做到 target-independent、可复用于所有候选item打分，同时用 mask-and-predict 的"Chronicle Alignment"把压缩后的历史对齐到近期意图，在离线和在线实验中都逼近全量 attention 的效果但成本大幅降低。

## 背景与动机

- 超长行为序列（几千到几万条)对CTR/CVR预测很有价值，但直接全量 self-attention 计算量随长度平方增长，线上单次请求要对上百候选打分，成本不可接受；简单截断成短窗口又会丢失长期兴趣信号。
- 工业界主流方案是 **target-attention 检索式**（SIM、ETA、SDIM、TWIN 等）：为每个候选从长历史里检索出相关行为再精细attention。问题：
  1. **target-dependent**——针对每个候选都要重新计算，长序列建模成本随候选数线性增长；
  2. 硬检索（类别/embedding相似度）容易过度聚焦于表面相似的行为，忽略用户多尺度、多层次的兴趣结构。
- 最近出现的 **压缩范式**（如 VISTA）把历史一次性压缩成target-independent 的少量 token 并缓存复用，避免了逐候选重算。但 VISTA 把 query token 全部拼在序列末尾、用**双向**attention 读出，产生的是一个**无序的 token 包**（"sequence-in, tokens-out"），存在两个问题：
  1. 塌缩了历史的时间结构；
  2. 双向attention让每个query都能看到全部时间线，导致各token高度冗余而非互补。

## 核心方法：ChronicleRec

提出 "sequence-in, sequence-out" 范式：压缩后的输出本身也是一个**按时间顺序排列的短序列**（Chronicle Tokens），而不是无序的 latent 包。针对四个挑战 (C1-C4) 设计对应组件：

### 1. Recency-aware 多粒度 Merge（解决 C1：压缩粒度不均匀）
- 把历史切成 far / mid / near 三段（旧→新），near 段保留全分辨率（stride=1），mid 段中等粒度池化，far 段用大 kernel/stride 粗聚合。
- 用 mask 过的滑窗均值池化生成 merge token 序列 U，padding 位置不会污染合并结果。
- 因窗口到源位置的映射在给定最大长度下是静态的，整个 merge 可以向量化实现，开销可忽略。

### 2. Causal Query-Token Interleaving（解决 C2：不能"看到未来"）
- 引入 P 个可学习 query token，按锚点位置**穿插**在 merge 后的序列中间（越靠近现在锚点越密集，越远越稀疏），而不是像 VISTA 那样统一拼在末尾。
- 用轻量 Transformer + **causal mask** 编码混合序列：每个位置只能看到它之前的内容（padding 部分作为 key 也被 mask 掉，但为了防止全 mask 行导致 softmax 数值不稳定，自身位置始终可见）。
- 编码后在 query 位置处的输出即为 Chronicle Tokens：每个 token 精确summarize"到其锚点为止"的历史，具有明确的时间感受野。
- 论文也提了一个 Q-Former 风格的变体（先用strided卷积降采样,再用双向linear attention读出)作为压缩器的替代实现，用于对比实验。

### 3. Multi-Branch Multi-Horizon 压缩（解决 C3：单趟压缩被近期行为主导）
- 用 B 个并行分支，分支 j 在压缩前mask掉最近 δ_j 条行为（0=δ1<δ2<...<δB），迫使不同分支关注不同远近的时间窗口。
- 各分支有独立的 query token 和 encoder，输出 prefix C^(j)，按权重 w_j 缩放后拼接成最终的 Chronicle Token 集合。

### 4. Chronicle Alignment（解决 C4：压缩的是过去，但排序关心的是当下）
- 强制性的预训练阶段：对每个 horizon 分支，mask掉一段近期行为，把压缩后的 prefix 喂给轻量 target-attention 头去预测被mask掉的目标标签，多分支用加权 BCE 联合训练。
- 该目标迫使每个分支把压缩的历史投射到"预测近期"的表示空间，同时鼓励不同 horizon 保留互补信息。推理时去掉辅助头，只保留拼接后的 prefix 作为 Chronicle memory 送入下游 ranker。

### 下游排序与训练
- Ranking head：几层 Pre-LN self-attention + target-attention read-out(候选embedding作query),拼接后MLP打分。
- 两种使用模式：**compress-only**（只用 Chronicle Tokens 替代全部历史）和 **hybrid**（Chronicle Tokens 作为 prefix + 近期短窗口行为一起输入）。
- 两阶段训练：Stage 1 用 Chronicle Alignment 预训练压缩器；Stage 2 把压缩器（冻结或微调）接入排序头做端到端训练。所有对比方法共享同一个 ranking head，保证效果差异只来自压缩方式。

## 实验

**数据集**：
- KuaiRand-27K（公开短视频数据集，人均历史~11.8K条，最长22.8万条，标签用是否点击）
- Tencent AdLive（工业界直播广告数据，20万用户，人均历史~1070条）

**主结果（GAUC）**：

| Method | KuaiRand | AdLive |
|---|---|---|
| Full-Attn (全量attention上界) | **0.5601** | **0.8071** |
| Short-Attn (近100条窗口) | 0.5307 | 0.7950 |
| VISTA (单趟压缩baseline) | 0.5518 | 0.7995 |
| ChronicleRec (三段压缩) | 0.5536 | 0.8010 |
| Hybrid-ChronicleRec | 0.5541 | 0.8030 |
| Multi-ChronicleRec (完整版) | 0.5580 | 0.8034 |

Multi-ChronicleRec 在 KuaiRand 上距 Full-Attn 仅差 0.0021 GAUC，同时全面优于 VISTA 和 Short-Attn。

**Query位置&方向消融**（AdLive）：interleaved+causal 组合最优 (0.8034)，明显优于 end+causal (0.7890)、interleaved+bidirectional (0.7930)、end+bidirectional (0.7951)——验证了"穿插排布 + 因果编码"两者缺一不可。

**组件消融**（去掉后 GAUC 下降）：
- 去 recency-aware merge: -0.0076
- 去 causal mask: -0.0074
- 去 multi-branch: -0.0072
- 去 deep supervision: -0.0039
- 去 Chronicle Alignment: **-0.0086（降幅最大）**

**远距离信号分析**：mask掉最近1000条行为后仍有 0.5499 GAUC，远高于只看近期窗口的 Short-Attn(0.5307)，说明远期历史本身就蕴含丰富、自成体系的兴趣信号,是渐进式而非灾难性的性能下降。

**效率**：相比 Full-Attn（49.1GB显存，训练2997分钟），ChronicleRec仅需28.0GB / 620.3分钟（约4.8倍训练加速），Hybrid版本进一步降到24.8GB/614.3分钟；各方法参数量相近(~1.23B)，说明性能差异主要来自序列建模策略而非模型容量。

**Token质量分析**：
- Token相似度：VISTA的token间高度相似（冗余），ChronicleRec呈现清晰的时间分区结构（近/中/远区块内相关性高，跨区相关性低）。
- Target-attention可视化：multi-horizon版本的注意力分散在多个时间位置；去掉multi-horizon后注意力高度集中在最近token（约1/3权重集中在最近token上）。
- 累积互信息(MI)分析：causal+interleave 设计比 end+bidirectional 累积MI增长更持续，加上multi-horizon后整体MI最高，说明后续token能持续贡献额外的目标相关信息而非重复信息。

**线上A/B测试**：部署在微信朋友圈广告的 pCVR 预估场景（MixFormer架构，NS-Token/S-Token），Chronicle Tokens融入两条pathway。7天A/B测试GMV提升 **+1.61%**（95% CI: [0.678%, 2.547%]，统计显著）。

## 总结

ChronicleRec 把"超长序列建模"重新定义为一个"预训练-迁移"的表征压缩问题：通过 recency-aware 多粒度merge、causal query-token 穿插、multi-horizon多分支、以及 Chronicle Alignment 预训练目标，学到的 Chronicle Tokens 既保持时间结构、又target-independent可缓存复用，在离线（逼近全量attention）和线上（GMV+1.61%）均取得显著收益，同时把长序列建模成本与候选打分成本解耦。

## 补充问答

### Q1：Causal Query-Token Interleaving 的关键点是"穿插"，不是"加在后面"

- **对**：因果 mask 保证每个位置只能 attend 到它自己及之前的位置（padding key 也被 mask，为避免全 mask 行导致 softmax 数值不稳定，自身位置总是可见）；编码后在 query 锚点位置读出的 encoder state 即为 Chronicle Token：
  $$\mathbf{C}=\mathrm{Encoder}_{\text{causal}}\big([\mathbf{U};\mathbf{Q}]_{\text{interleaved}}\big)\big|_{\text{query positions}}$$
- P 个 query token 不是拼在序列**末尾**，而是按"锚点位置"**穿插**在 merge 后的序列 $\mathbf{U}$ 中间——锚点距当下越近越密集，距当下越远越稀疏。这一点是本质设计，而非细节：如果所有 query 都放在末尾，因果 mask 下每个 token 看到的"过去"范围完全相同（整段历史），P 个 token 只是初始化不同，天然冗余；穿插到不同锚点后，每个 token 才拥有**严格不同、明确的时间感受野**（"summarize exactly the history up to its anchor"），使 P 个 token 在时间维度上互补而非重复。论文的消融直接验证了这一点：AdLive 上 interleaved+causal（0.8034）明显优于 end+causal（0.7890），说明"穿插排布"本身（而不仅仅是因果 mask）是关键收益来源。

### Q2：Multi-Horizon 是"分支独立 encoder + 拼接"，不是"加权 pooling"

- **对**：存在 $B$ 组独立的 readout token，分支 $j$ 在压缩前 mask 掉最近 $\delta_j$ 条行为（$0=\delta_1<\delta_2<\dots<\delta_B$），逼迫不同分支关注不同远近的时间窗口。
- **需要修正**：多分支的输出不是"加权 pooling"成更少的 token，而是按权重**缩放后拼接（concatenate）**：
  $$\mathbf{C}=[w_1\mathbf{C}^{(1)};w_2\mathbf{C}^{(2)};\dots;w_B\mathbf{C}^{(B)}]\in\mathbb{R}^{(B\cdot P)\times d}$$
  最终 token 数是 $B\times P$（**变多**而不是压缩变少），每个分支的 $P$ 个 token 原样保留、只做标量缩放，再整体交给下游继续做 attention。
- **Encoder 是什么**：每个分支拥有"独立的 query token 和 encoder"——即各分支分别实例化一份与"Causal Query-Token Interleaving"一节结构完全相同的**轻量级因果 Transformer encoder**，参数互不共享，各自跑一遍完整的"recency-aware merge → 穿插 query → 因果编码"流程，而不是一个共享 encoder 处理所有分支。

### Q3：与 `summary_llatte_scaling_laws.md`（LLaTTE）的本质区别

LLaTTE 的 Query Token 表面上也是"拼接到序列 + Transformer 编码 + 取 query 位置输出"的 readout 范式（$\mathbf{X}_{\mathrm{input}}=[\mathbf{X}_{\mathrm{seq}};\mathbf{Q}]$，最终只读取 Query 位置状态），但和 ChronicleRec 有几处本质差异：

| 维度 | ChronicleRec | LLaTTE |
|---|---|---|
| Query 锚点位置 | **穿插**在序列不同时间位置，越靠近当下越密集 | 全部拼在序列**末尾** |
| 每个 token 的感受野 | 严格不同，由锚点决定"看到历史的多少" | 完全相同（因果 mask 下都能看到全部历史），差异只来自 query 的初始化/角色（如 user-only vs candidate-aware） |
| "多组 readout token" 的构造维度 | 显式**时间维度**：不同分支 mask 不同长度的近期行为 | **语义/角色维度**：不同 query 编码不同上下文（用户侧 vs 广告/请求侧），不按时间窗口切分 |
| 多 token 后处理 | 按权重缩放后**拼接**，token 数变多（$B\times P$），保留序列结构 | 所有 query final states 先 **Flatten + LoRA MLP** 投影压缩成一个（或少数几个）固定维度稠密向量，序列结构被压扁 |
| 下游消费方式 | 下游 ranker 对这组 token 做 self-attention + target-attention 读出，仍是"序列进、序列用" | 稠密向量作为 dense feature 送入 DHEN 式特征交叉网络，不再是可供 attention 的 token 集合 |
| 训练监督 | 专门的 Chronicle Alignment 预训练目标，强制每个 horizon 分支对齐"预测被 mask 的近期标签"，显式去冗余 | 无对应的对齐/去冗余辅助 loss，靠端到端 CTR/CVR 多任务 loss 塑形 |

一句话总结：**LLaTTE 的 readout token 本质上是"位置相同、看到信息相同、仅角色分工不同"的可学习种子，最终被压缩（flatten+MLP）成稠密向量，时间结构被丢弃；ChronicleRec 的 readout token 是"锚点不同、感受野不同"的结构化序列，显式保留时间轴，并用额外的 Alignment loss 强制多尺度信息互补，最终以序列形式（而非压缩向量）交给下游。** 用 ChronicleRec 自己的消融坐标系来说，LLaTTE 的做法正好对应其验证过效果更差的 "end + causal" 配置。

### Q4：序列数据形式 & 特征（对照 "Interest Cache Roadmap" 文档 Related Works 表的 Input 列）

- **数据形式**：一个用户的历史是一条按时间从旧到新排列的行为序列 $\mathcal{H}=(b_1,\dots,b_L)$，左侧 padding 到固定长度 $L_{\max}$（KuaiRand 为 2048，AdLive 为 4000），附带一个 0/1 有效性 mask 标记 padding 位置。整条序列中不包含文本，也不显式携带 timestamp 字段——"新旧"完全由序列内的**位置**决定（recency-aware merge 的 near/mid/far 分段边界也是按"最近 n 条"这种**位置**划分，而非按具体时间戳阈值）。
- **特征**：论文原文只写明每个行为 $b_i$ 是一个**类别特征 tuple**：item/video ID、author/advertiser ID、content tag（3 个字段），过 embedding 层后得到 $d$ 维向量；默认 $d_{\text{model}}=64$，其中 video/author/tag 三个 embedding 维度分别是 $64/16/16$（原文未说明三者是拼接还是相加投影到 64 维，可能是 concat 后过一层线性投影）。没有提及engagement 类型（点击/完播/点赞等多种行为类型)、播放时长、是否广告等其它 side feature——这点明显比同表里 IG 现有实现（68 列特征、6 类 selected feature、event_times/engagement_types 等）或 LLaTTE（sparse/dense/float 多种非序列特征）**简化很多**，更接近 "id-only" 的极简 schema。
- **举例**（论文未给出具体例子，以下按论文描述的 schema 推测构造一条合理样本）：

  ```
  History H (从旧到新，截断/左padding到 L_max=2048):
    b_1    = (video_id=V_003321, author_id=A_1187, tag_id=Comedy)
    b_2    = (video_id=V_009142, author_id=A_0456, tag_id=Food)
    ...
    b_1999 = (video_id=V_884213, author_id=A_2231, tag_id=Travel)
    b_2000 = (video_id=V_990211, author_id=A_2231, tag_id=Travel)   # near 段，最近的行为，全分辨率保留

  Candidate (target item):
    t = (video_id=V_120099, author_id=A_5567, tag_id=Sports)

  Label:
    y = is_click ∈ {0,1}
  ```

  与文档表格里其它方法对比：TokenMinds/RecGPTv3/Personas 的输入都带**文本**（视频文案、用户历史的自然语言描述），依赖 LLM 做 encoder；FOUNDv2 把"文本+common feature"过 Qwen3 Embedding 得到稠密向量再做 RQ-VAE；而 **ChronicleRec 和 LLaTTE 一样，输入是纯 ID 类别特征序列，不经过任何 LLM/文本环节**，这也是它能做到"轻量级压缩器 + 4.8× 训练加速"的前提之一。

### Q5：最终压缩表达 & 例子（对照 Output 列）

- **形式**：Chronicle Tokens 是一个**稠密（dense）、按时间顺序排列的向量序列** $\mathbf{C}\in\mathbb{R}^{P\times d}$（多分支版本是 $\mathbb{R}^{(B\cdot P)\times d}$），**不是**离散 SID、**不是**文本、也**不是**像 LLaTTE 那样 flatten 成的单一稠密向量——它是一个保留"多 token + 时间序结构"的矩阵，可以直接被下游当作一段可 attend 的短序列（prefix）使用。
- **具体维度**：默认单分支配置 $P=11$、$d=64$，即 $\mathbf{C}\in\mathbb{R}^{11\times 64}$：11 个 64 维向量，按锚点从"最远"到"最近"排列。效果最好的 Multi-ChronicleRec 用 $B=5$ 个分支（$\delta_j=\{0,100,300,600,1000\}$），每个分支各出 $P=11$ 个 token 并按权重 $w_j=\{1.0,0.5,0.4,0.3,0.2\}$ 缩放后拼接，最终 $\mathbf{C}\in\mathbb{R}^{55\times 64}$（55 个 token，而不是变得更少）。
- **举例**：对上面 Q4 例子里的用户，Multi-ChronicleRec 会产出一个 $55\times64$ 的浮点矩阵，可以理解成 55 条“记忆条目”：
  - 第 1 个 token（来自 $\delta_5=1000$ 分支、锚点在"1000 步之前"）：`[0.021, -0.183, 0.077, ..., 0.004]`（64 维），概括的是"1000 步之前的全部历史"；
  - 第 55 个 token（来自 $\delta_1=0$ 分支、锚点在序列末尾）：`[-0.056, 0.132, ..., 0.091]`（64 维），概括的是"几乎全部历史直到最新一条行为"。
  - 这 55×64 的矩阵作为一个**紧凑前缀（prefix）**被 cache 下来，下游 ranker 用候选 item 的 embedding 作 query，对这 55 个 token 做 target-attention 读出兴趣向量，再拼接候选 embedding 过 MLP 打分——不需要重新扫一遍原始的 2000+ 条历史。
- 对照文档里其它方法的 Output：TokenMinds/FOUNDv2/RecGPTv3 输出的是**离散 SID 序列**（+可选 pooled dense embedding），LLaTTE 输出的是**单个 pooled dense embedding 向量**；ChronicleRec 的输出本质上是这两者之外的第三种形态——**多个、有序、稠密的 token**，这正是它反复强调的 "sequence-in, sequence-out"（而非 "sequence-in, vector-out" 或 "sequence-in, tokens(离散/无序)-out"）。

### Q6：有没有 Intermediate Evaluation？

- **没有像文档里其它方法那样报告一个独立的、脱离下游任务的"中间效果指标"**。对照 Related Works 表的 "Intermediate Performance Validation" 列：TokenMinds 用"decoder 输出做 retrieval"、Persona 用"生成兴趣与用户点击兴趣的相似度/人工认可度"、RecGPTv3 用"人工标注打分"、LLaTTE 用"upstream NE 变化量 / downstream NE 变化量的迁移比"——这些都是**不需要接完整下游排序头、单独衡量压缩表示本身好坏**的指标。ChronicleRec **论文中没有这样一个单独报告的指标**；FOUNDv2 那一行写的 "No intermediate validation" 更贴近 ChronicleRec 的情况。
- **训练范式是"两阶段 pretrain-and-transfer"，但两阶段都绑定在（不同粒度的）预测任务上，而非纯 reconstruction/无监督指标**：
  - **Stage 1（Chronicle Alignment 预训练）**：把每个 horizon 分支的压缩 prefix 接一个轻量 target-attention 头，去预测被 mask 掉的近期行为标签 $\hat y_j$，用加权 BCE 联合训练。这本质上是一个**代理/pretext 任务**（预测的是"被 mask 的历史标签"而不是"真正线上下游要预测的候选点击率"），可以看作一种弱化版的 intermediate signal，但论文**没有单独把这一阶段的效果作为一个基准数字报出来**，只是作为 Stage 2 之前必需的预训练步骤（消融实验里"去掉 Chronicle Alignment"对应的是"不做这个预训练阶段"，而不是"报告 Stage-1-only 的指标"）。
  - **Stage 2（端到端下游训练）**：把（frozen 或 finetune 的）压缩器接入和所有 baseline 共享的同一个 ranking head，在真实点击/转化标签上端到端训练。论文里所有主表格（主结果、消融、位置消融）的 GAUC 数字都来自这一步，即**最终效果必须过下游头才能衡量**。
  - 所以严格来说，ChronicleRec **属于"端到端训练 + 最终任务指标评估"**，不依赖单独的 intermediate benchmark 数字来验证压缩表示的好坏。
- **但论文额外提供了一组"表示诊断"式的分析实验**（Effective-Rank/SVD 分析、Token 余弦相似度热力图、target-attention 权重可视化、Chronicle Token 前缀与标签之间的**累积互信息（debiased MI）曲线**），本质上是在不重跑完整下游训练的情况下，直接检验压缩 token 本身的信息量/冗余度/时间结构。其中累积 MI 分析（$\widetilde I(y;\mathbf c_{1:p})$ 随 $p$ 递增的曲线）是最接近"intermediate/intrinsic evaluation"的部分——它直接衡量"压缩 token 前缀"与"预测目标"之间的信息量，而不需要接完整 ranking head 重新训练。可以理解为：ChronicleRec 没有报告独立的 intermediate **基准分数**，但用了 MI/相似度/注意力可视化等**诊断性分析**去补充说明压缩表示为什么有效，作用类似但不等同于文档中其它方法的 Intermediate Performance Validation 列。
