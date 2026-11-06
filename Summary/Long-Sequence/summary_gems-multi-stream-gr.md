# GEMs: Breaking the Long-Sequence Barrier in Generative Recommendation with a Multi-Stream Decoder

> 论文来源：arXiv: 2602.13631
> 作者：Yu Zhou, Chengcheng Guo, Kuo Cai, Ji Liu, Qiang Luo, Ruiming Tang, Han Li, Kun Gai, Guorui Zhou（快手科技）
> 代码/数据：工业数据，暂未开源

---

用户给出的综合理解：
> 本文做召回任务。主要工作是把用户序列分为三种：实时，中期，历史；实时序列采用正常的 full-attention, 中期采用 indexer 方式提取核心事件；历史采用编码压缩成固定长度。训练和推理时采用 encoder-decoder 结构，encoder 接收用户序列进行处理；推理时，decoder 输入 [bos], kv 来自 encoder，产出召回 item.

**整体非常准确**，以下是逐条确认和小修正：

| 理解点 | 确认/修正 |
|--------|-----------|
| ✅ 做召回任务 | 正确 |
| ✅ 用户序列分三种：实时、中期、历史 | 正确 |
| ✅ 实时序列采用 full-attention | 正确（多层自注意力 + 位置编码） |
| ✅ 中期采用 indexer 方式提取核心事件 | 正确（蒸馏训练 + Top-K 稀疏注意力） |
| ⚠️ 历史采用编码压缩成**固定长度** | 小修正：QLA 压缩后的长度取决于压缩比率，不是严格固定长度；压缩表示存在 KV 系统中在线检索 |
| ✅ Encoder-Decoder 结构 | 正确 |
| ✅ Encoder 接收用户序列进行处理 | 正确 |
| ✅ 推理时 decoder 输入 [BOS]，KV 来自 encoder | 正确 |
| ✅ 产出召回 item | 正确（Beam Search 生成 SID 三元组映射回物品） |

**补充**：三个流在 decoder 侧是**独立的流特定交叉注意力 + 无参数求和融合**，不是简单拼接。

## 一、研究背景与问题

生成式推荐（Generative Recommendation, GR）凭借自回归预测语义 ID 的范式，展现出强大的序列推理能力。然而在工业级超长用户行为序列（可达 10 万+ 交互）场景下，现有 GR 面临三大核心瓶颈：

1. **序列长度壁垒**：GR 的 Transformer 架构计算成本高，工业部署的 GR 序列长度普遍限制在 1 万以内，仅覆盖用户近几个月行为，无法捕捉终身兴趣。
2. **序列扩展效率低（Sub-linear Gain）**：实验表明 GR 从增加历史序列长度中获得的收益是次线性的，在严格延迟要求下，长序列场景 ROI 极低。
3. **注意力机制的"近因偏差"（Recency Bias）**：分析发现 GR 的注意力主要集中在序列近期片段，长期历史信号被削弱。

---

## 二、GEMs 核心思想

将超长序列建模视为**多时间尺度记忆读取问题**——不同时间范围的行为需要不同的计算策略。GEMs 把用户终身序列 $X_u = [x_1, \dots, x_T]$（$T$ 可达 10 万+）按两个阈值 $T_{\text{lifecycle}} < T_{\text{recent}}$ 划分为三个时序流：

| 时序流 | 范围 | 特点 | 建模策略 |
|--------|------|------|----------|
| **Recent**（近期） | $X_u[T_{\text{recent}}:]$ | 即时动态、快速演化 | 单阶段实时提取器 |
| **Mid-term**（中期） | $X_u[(T_{\text{lifecycle}}+1):T_{\text{recent}}]$ | 数千 token，含兴趣漂移 | 轻量级索引器 + 稀疏交叉注意力 |
| **Lifecycle**（终身） | $X_u[:T_{\text{lifecycle}}]$ | 10 万+ token，超长 | 两阶段离线-在线压缩模块 |

---

## 三、关键技术细节

### 3.1 Recent Stream（近期流）

提取多模态特征（视频 ID、作者 ID、标签、时间戳、播放时长、视频时长、交互标签），拼接后经 MLP 投影，再用带位置编码的多层自注意力捕捉即时兴趣。

### 3.2 Mid-term Stream：轻量级索引器交叉注意力（Lightweight Indexer for Cross Attention）

这是论文的核心创新之一，旨在对数千 token 的中期序列在**保证精度的同时控制在线计算成本**：

- **索引器设计**：用远少于标准注意力头数的头数 $H_{\text{indexer}} \ll H$（实验中 $H=8, H_{\text{indexer}}=2$）生成稀疏注意力分数。
- **知识蒸馏训练**：训练时同时计算全注意力分数 $S^{(j)}_{\text{mid-term}}$（教师）和索引器分数 $S^{(j)}_{\text{indexer}}$（学生），用 KL 散度对齐：
  $$\mathcal{L}_{indexer} = \sum_{j=1}^L D_{KL}(S^{(j)}_{\text{indexer}} \parallel \text{Softmax}(S^{(j)}_{\text{mid-term}}))$$
- **推理时稀疏注意力**：根据索引器分数为每个 Query 选 Top-K 的 Key/Value 做稀疏注意力，将复杂度从 $O(n_q n_m d)$ 降到 $O(n_q K d)$，把部分计算成本转移到训练阶段。
- **解耦设计**：索引器在训练时 detach，仅由 $\mathcal{L}_{\text{indexer}}$ 优化，避免"选择"与"注意力"耦合导致的训练-推理不一致。

### 3.3 Lifecycle Stream（终身流）

借鉴 VISTA 的两阶段机制处理 10 万+ 交互：

- **离线阶段**：用拟线性注意力（Quasi-Linear Attention, QLA）+ QLU 优化压缩终身历史，通过生成式重建损失监督压缩质量：
  $$\mathcal{L}_{\text{recon}} = \sum_{i=1}^{M-1} \|v_i - u_{i+1}\|_2^2$$
  其中 $v_i$ 为压缩表示，$u_{i+1}$ 为重建表示。
- **在线阶段**：从低延迟 KV 系统检索压缩表示，经轻量自注意力处理。

### 3.4 无参数融合策略（Parameter-Free Fusion）

解码端对三个流分别做独立的因果自注意力 + 流特定交叉注意力 + FFN，得到三个隐藏状态 $D^{(j)*}_{\text{recent}}, D^{(j)*}_{\text{mid-term}}, D^{(j)*}_{\text{lifecycle}}$，然后**直接求和**融合：
$$D^{(j)} = D^{(j)*}_{\text{recent}} + D^{(j)*}_{\text{mid-term}} + D^{(j)*}_{\text{lifecycle}}$$

**为什么不用可学习融合（如门控、MLP）？**
在超长序列场景下，中期/终身流与目标 next-item 的分布偏移比近期流大（暴露偏差 + 时间漂移）。可学习融合会**退化为过度加权近期流**，抑制长期信号。实验验证（见 RQ3）表明无参数求和最稳定、效果最好。

### 3.5 三阶段训练策略

1. **Stage 1**：禁用索引器选择，冻结 $\mathcal{L}_{\text{indexer}}$，用全注意力收敛主任务，避免索引器监督信号不稳定。
2. **Stage 2**：启用 $\mathcal{L}_{\text{indexer}}$，对齐索引器分数与全注意力分数。
3. **Stage 3**：启用索引器 Top-K 选择，使训练目标与在线推理一致。

### 3.6 推理策略

Beam Search：从 [BOS] 开始自回归生成语义 ID，生成步数等于码本深度，Beam 宽度由返回物品数决定。

---

## 四、实验与结果

### 4.1 数据集

快手真实工业数据：4 亿日活用户、1 亿物品、500 亿日交互、平均序列长度 1.45 万、最大 10 万。

### 4.2 总体性能（RQ1）

相比 9 个基线（MPFormer、TDM、MISS、GPRP、Kuaiformer、MIND、TIGER、GRank 等），GEMs 在所有 Recall@K / NDCG@K 上全面领先：

| 指标 | GEMs | 最佳基线 | 相对提升 |
|------|------|----------|----------|
| Recall@100 | 0.1553 | 0.1282 (TIGER) | **+21.14%** |
| NDCG@100 | 0.0064 | 0.0053 (TIGER) | **+20.75%** |
| Recall@500 | 0.2911 | 0.2396 (GRank) | +21.49% |
| Recall@1000 | 0.3511 | 0.3178 (GRank) | +10.47% |

**关键发现**：在较小 K 值下提升更显著，说明终身序列对精准检索贡献更大。

### 4.3 消融实验（RQ2）

移除任意流都会导致性能下降，三个流互补：
- 移终身流：Level 1 影响小，但 Level 2/3 大幅下降（终身流主要贡献推理后期）
- 移中期流：Level 1 显著下降（中期流主要贡献首层推理）
- 仅近期流：所有层级全面下降

### 4.4 融合策略对比（RQ3）

四种融合方案：(a) 全拼接共享编解码器；(b) 分离编码器 + 拼接输入共享解码器；(c) 分离编码器 + 门控加权；(d) 本文无参数融合。

结果：(d) 最优。注意力热图可视化证实可学习融合（a/b/c）把大部分注意力分配给近期流，长期信号被压制；无参数融合让三流均衡贡献。

### 4.5 索引器效率（RQ4）

| 方案 | 延迟(ms) | Recall@1000 |
|------|----------|-------------|
| Brute-force (T=2000) | 60.18 | 0.3312 |
| Brute-force (T=5000) | 103.68 | 0.3464 |
| **Indexer (T=5000)** | **32.2** | **0.3511** |

索引器不仅把延迟降低约 70%，还通过去噪提升了效果。

### 4.6 模型规模（RQ5）

随隐藏维度和注意力头数增加，分层召回稳步提升，验证框架具备良好的可扩展性。

### 4.7 在线 A/B 测试（RQ6）

在快手主站和极速版各 5% 用户流量运行 7 天：
- 快手极速版：总使用时长 **+0.17%**，视频观看时长 **+0.35%**
- 快手主站：总使用时长 +0.13%，单视频观看时长 +0.42%

所有指标在 0.05 显著性水平下统计显著。

---

## 五、工业部署优化

### 训练侧
- BF16/FP16 混合精度：batch size +42%，训练时间 -37%
- Flash Attention：单 GPU batch size +10%

### 推理侧
- 轻量索引器：延迟 -70%
- 混合精度推理：graph runtime -21%
- TensorRT 优化：显存释放 ~70%
- 用户批处理（User Batching）：延迟再 -12%

GEMs 是**首个成功部署在高并发工业环境的终身 GR 框架**，可处理 10 万+ 交互序列并保持高效推理。

---

## 六、核心贡献总结

1. 提出首个统一的终身生成式推荐框架，将超长序列建模转化为多时间尺度记忆读取问题，用三流分治 + 无参数融合高效整合。
2. 轻量级索引器通过"知识蒸馏训练 + Top-K 稀疏推理"在精度与成本间取得优异平衡。
3. 无参数融合策略系统性解决了可学习融合在超长序列下的"近期流坍缩"问题。
4. 首次实现终身 GR 的工业级部署，延迟和效果均达标，A/B 测试显著提升用户留存与观看时长。

---

## 七、个人思考与启发

1. **分治是处理超长序列的有效范式**：不同时间尺度的行为本质不同，强行用同一注意力机制既不经济也不有效。GEMs 的 Recent/Mid/Lifecycle 切分可推广到其他序列建模任务。
2. **训练-推理成本迁移**：索引器把昂贵的全注意力计算从推理转移到训练，这是工业部署的关键洞察——训练可以贵，但推理必须快。
3. **融合并非越复杂越好**：分布偏移下可学习融合容易坍缩，简单求和反而更鲁棒。这一发现对多模态、多兴趣融合也有参考价值。
4. **与现有长序列工作的关系**：TWINS/OneTrans 等主要在传统排序/检索模型中做长序列，GEMs 则在 GR 范式下首次做到 10 万+ 终身序列，填补了该领域空白。

---

## 八、讨论与问答

### Q1：流特定交叉注意力是什么？

**流特定交叉注意力（Stream-specific Cross-Attention）** 是 GEMs 解码器中的关键设计。在每个 decoder block $j$ 中，流程是：

1. 解码器先做**因果自注意力**（Causal Self-Attn），让层级化的 SID token 互相收集信息，得到 $D^{(j)\prime}$
2. 然后 $D^{(j)\prime}$ 作为 **Query**，分别与三个流各自的编码记忆做交叉注意力：
   $$D^{(j)}_{s} = D^{(j)\prime} + \mathrm{CrossAttn}(D^{(j)\prime},\ H^{(L_s)}_{s},\ H^{(L_s)}_{s}), \quad s \in \{\text{recent},\ \text{mid-term},\ \text{lifecycle}\}$$

**"流特定"的含义**：三个流的交叉注意力拥有**完全独立的参数**（独立的 Q/K/V 投影矩阵和 FFN），而不是共享一套交叉注意力再输入不同的 K/V。这与论文消融实验中的策略 (a)/(b)（共享编解码器）形成对比——独立参数让每个流能以不同方式与解码器交互，最后通过无参数求和融合。

### Q2：三阶段训练都是为了 Mid-term Stream 中的索引器？

**不完全是。** 三阶段训练的核心确实是围绕索引器展开，但 Stage 1 另有作用：

| 阶段 | 索引器状态 | $\mathcal{L}_{\text{indexer}}$ | 主要目的 |
|------|-----------|-------------------------------|----------|
| **Stage 1** | 禁用选择 | 冻结（$\lambda=0$） | 用全注意力收敛**主任务 NTP**，同时收敛近期流、终身流、解码器等所有组件，为索引器训练提供稳定基础 |
| **Stage 2** | 禁用选择 | 启用（$\lambda=1$） | **专门训练索引器**，用 KL 散度对齐索引器分数与全注意力分数 |
| **Stage 3** | 启用 Top-K 选择 | 启用 | 让训练目标与推理行为一致，联合微调 |

所以更准确地说：Stage 1 是打基础（主任务收敛），Stage 2 和 Stage 3 才是专门为索引器服务的。如果主任务没先收敛，索引器的蒸馏信号会不稳定。

### Q3：Lifecycle Stream 的表达应该是提前训练好的？具体长度是多少？

这里需要**厘清论文中的符号不一致**。论文有两种描述方式：

**方式一（形式化公式，从最旧端数索引）**：
$$1<T_{\text{lifecycle}}<T_{\text{recent}}<T$$
- lifecycle = $X_u[:T_{\text{lifecycle}}]$（最旧的一段）
- mid-term = $X_u[(T_{\text{lifecycle}}+1):T_{\text{recent}}]$（中间）
- recent = $X_u[T_{\text{recent}}:]$（最新）

**方式二（实现细节，从最新端数数量）**：
- recent = 最新 256 个交互
- mid-term = 第 257~5000 个交互（从最新端往前数）
- lifecycle = 第 5001 个及更早的所有交互

两种方式方向相反，但语义一致。按实现细节，**Lifecycle Stream 的长度 = 总序列长度 - 5000**，最大可达约 **95,000 token**（总序列上限 10 万）。

关于"是否提前训练好"：
- **压缩表示是离线预计算的**：Lifecycle Stream 用拟线性注意力（QLA）+ QLU 压缩，压缩后的表示存储在低延迟 KV 系统中，在线推理时直接检索，不需要实时处理 9.5 万 token。
- **但压缩模块本身是否与主模型联合训练，论文未完全明确**。从 $\mathcal{L}_{\text{recon}}$ 重建损失和端到端 GR 范式推断，更可能是联合训练的，只是推理时把压缩结果缓存了。论文借鉴的 VISTA 是离线-在线两阶段架构，但具体到 GEMs 的训练方式需要进一步确认。

### Q4：4.3 消融实验中 Level 1, 2, 3 分别表示什么？

Level 1/2/3 对应**三层 Semantic ID 码本的 beam search 中间步骤**。

GEMs 用 Residual K-means 把每个物品量化为三级码本 $[8192, 8192, 8192]$，即每个物品对应 3 个层级化的 SID：$(s_1, s_2, s_3)$，码本深度 $d=3$。

推理时 beam search 逐步生成 SID：
- **Level 1**：生成第 1 个 SID（最粗粒度，定位到大类）时的召回
- **Level 2**：生成第 2 个 SID（中等粒度）时的召回
- **Level 3**：生成第 3 个 SID（最细粒度，精确定位到物品）时的召回

`Hrecall@Level l@1000` 衡量 beam search 第 $l$ 步时，top-1000 候选中正确物品所在分支的召回率。Level 1 最容易（粗粒度匹配），Level 3 最难（精确定位）。消融结果也印证了这一点——逐级递减，且移除不同流对不同层级的影响不同（终身流主要影响 Level 2/3，中期流主要影响 Level 1）。

### Q5：训练时采用正负 label 还是 next-token prediction？

**用的是 Next-Token Prediction（NTP），不是正负 label 对比学习。**

论文明确给出主任务损失：
$$\mathcal{L}_{\mathrm{NTP}} = -\log p_\theta(s_{x_t}^d \mid \ldots)$$

具体流程：
1. 每个物品被量化为 3 个层级 SID $(s_1, s_2, s_3)$
2. 输入用户三流历史序列，解码器从 [BOS] 开始**teacher forcing** 自回归预测目标物品的 3 个 SID
3. 解码器最终隐状态 $D^{(j)}$ 与码本向量 $\mathbf{C} \in \mathbb{R}^{M \times d_h}$ 做点积，计算正确 SID 的生成概率
4. 用负对数似然（NLL）最大化正确 SID 序列的概率

这与 TIGER、OneRec 等 GR 方法的训练范式完全一致——本质是把推荐转化为序列生成问题，用标准的语言模型式 next-token prediction 来训练。正负 label 对比学习是传统推荐模型（如双塔）的做法，GEMs 作为生成式推荐走的是完全不同的路线。

### Q6：Stream-specific Cross-Attention 中，解码器的输入是什么？是三个流吗？

**不是。三个流不是解码器的输入，而是解码器 cross-attention 中的 Key/Value。** 这个理解很关键。

**解码器的输入 $D^{(0)}$ 是 Semantic ID token 序列：**
$$D^{(0)} = [E_{\text{BOS}};\ E_{s^1_{x_t}};\ \ldots;\ E_{s^d_{x_t}}]$$
- **训练时（teacher forcing）**：$D^{(0)}$ = [BOS] + 目标物品 $x_t$ 的 3 个 SID 的 embedding，共 4 个 token
- **推理时（自回归生成）**：从 [BOS] 开始，每生成一个 SID 就拼接到输入后面，逐步生成 3 层 SID

**三流在解码器中扮演的角色**：三个流是 Encoder 侧的产物，作为 Key/Value 记忆被 cross-attention 读取。完整数据流如下：

```
┌─────────────────────────────────────────────────────────┐
│  Decoder (4层, 输入是 SID token 序列)                    │
│                                                         │
│  D^(0) = [BOS, s1, s2, s3]  ← 这才是 decoder 输入        │
│       │                                                 │
│       ▼                                                 │
│  ┌──────────────────┐                                   │
│  │ Causal Self-Attn │  ← SID token 之间互相看           │
│  └────────┬─────────┘                                   │
│           │ D' 作为 Query                                │
│     ┌─────┼───────────────┬────────────────┐            │
│     ▼     ▼               ▼                ▼            │
│  Cross  Cross           Cross            ...            │
│  Attn   Attn            Attn                           │
│  (recent) (mid-term)   (lifecycle)  ← 三流是 K/V 记忆   │
│     │     │               │                            │
│     ▼     ▼               ▼                            │
│  D_recent D_mid        D_lifecycle                     │
│     │     │               │                            │
│     └─────┴───────────────┘                            │
│           │  无参数求和                                  │
│           ▼                                             │
│       D^(j) → 下一层                                    │
└─────────────────────────────────────────────────────────┘
           ▲
┌──────────┴──────────┐
│  Encoder 侧:        │
│  Recent Stream → H_recent  (K/V)
│  Mid-term  Stream → H_mid   (K/V)
│  Lifecycle Stream → H_lc    (K/V)
└─────────────────────┘
```

**类比理解**：可以类比为阅读理解——
- **Decoder 输入（SID token）** = 你要生成的"答案"（Query）
- **三个流的编码记忆** = 三篇不同时间尺度的"阅读材料"（K/V），decoder 通过 cross-attention 从这三篇材料中检索相关信息来生成答案

所以更准确的说法是：**解码器的输入是 SID token 序列，三个流作为外部记忆通过流特定交叉注意力被解码器读取。**

### Q7：训练时每个样本就是 4 个 Token 吗？推理时一个 [BOS] 就能产生召回？

**方向正确，但有两个重要细节需要澄清。**

#### 训练样本不只是 4 个 Token

4 个 token 只是**解码器侧的输入**，完整的训练样本还包括**编码器侧的三个长序列**：

```
┌──────────────────────────────────────────────────────────┐
│  一个完整训练样本                                          │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │ Encoder 侧 (输入很长)                               │  │
│  │   Recent Stream:    256 token                       │  │
│  │   Mid-term Stream:  ~4,700 token (第257~5000)       │  │
│  │   Lifecycle Stream: ~95,000 token (第5001及更早)    │  │
│  │   → 编码为三路 KV 记忆                              │  │
│  └────────────────────────────────────────────────────┘  │
│                         │                                │
│                         ▼ 作为 KV                        │
│  ┌────────────────────────────────────────────────────┐  │
│  │ Decoder 侧 (输入很短, 4 token)                      │  │
│  │   [BOS, s1_target, s2_target, s3_target]            │  │
│  │   → teacher forcing, 并行预测每个位置的下一个SID     │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

- 解码器输入确实是 4 个 token（[BOS] + 目标物品的 3 个 SID），用 teacher forcing 并行训练
- 三个 stream 作为 KV **指导**解码器生成正确的 SID 序列
- 但编码器侧的输入其实很长（256 + 4700 + 95000 ≈ 10 万 token），只是它们不直接进入 decoder，而是编码成 KV 记忆

#### 推理不是一步生成，而是多步 Beam Search

推理时解码器输入会随生成**逐步增长**，三个 stream 在每一步都作为 KV 被 cross-attention 读取：

```
Step 1:  输入 [BOS]
         → 生成 Level-1 SID 的 top-B 候选 (B=返回物品数, 如 1000)
         → 得到 [BOS, s1⁽¹⁾], [BOS, s1⁽²⁾], ..., [BOS, s1⁽ᴮ⁾]

Step 2:  每个分支输入 [BOS, s1⁽ⁱ⁾]
         → 生成 Level-2 SID 的 top-B 候选
         → 保留全局 top-B 个 [BOS, s1⁽ⁱ⁾, s2⁽ⁱʲ⁾]

Step 3:  每个分支输入 [BOS, s1⁽ⁱ⁾, s2⁽ⁱʲ⁾]
         → 生成 Level-3 SID 的 top-B 候选
         → 保留全局 top-B 个 (s1, s2, s3) 三元组

最终:    (s1, s2, s3) 三元组 → 映射回具体物品 → 召回列表
```

- Beam 宽度 B = 每个请求返回的物品数，所以是**同时产生多个召回物品**，不是一个
- 生成的 SID 三元组通过码本映射回具体物品 ID，构成最终召回列表

#### 训练 vs 推理对比

| 维度 | 训练 | 推理 |
|------|------|------|
| Decoder 输入 | 固定 4 token `[BOS, s1, s2, s3]`（teacher forcing） | 逐步增长：`[BOS]` → `[BOS, s1]` → `[BOS, s1, s2]` |
| 生成方式 | 并行预测所有位置的下一个 token | 自回归 + Beam Search 逐步生成 |
| 三流角色 | KV 记忆（每一层 cross-attn 都用） | KV 记忆（每一步每一层都用） |
| 输出 | NLL 损失（与目标 SID 对比） | top-B 个 SID 三元组 → 召回物品 |

### Q8：推理时 Encoder 侧能否提前计算？Decoder 输出能否缓存？

#### Encoder 侧缓存：部分可以，三流策略不同

| 流 | 能否预计算 | 原因 |
|----|-----------|------|
| **Recent (256 token)** | ❌ 不能 | 用户每次新交互都会改变近期序列，必须实时编码 |
| **Mid-term (~4700 token)** | ⚠️ 部分可以 | 序列变化较慢，indexer 分数可周期性更新，但仍需准实时处理 |
| **Lifecycle (~95000 token)** | ✅ 已经做了 | 论文明确说离线用 QLA 压缩，压缩表示存低延迟 KV 系统，在线直接检索——这本身就是预计算缓存！ |

Lifecycle Stream 的两阶段设计**本质上就是 encoder 侧缓存**：把最昂贵的 9.5 万 token 压缩计算离线完成，在线只做轻量检索。

#### Decoder 输出缓存：不能！关键误解

虽然 decoder 的输入 `[BOS]` 对所有用户相同，但 decoder 输出**高度用户相关**：

```
所有用户的 decoder 输入都是 [BOS]  ← 输入相同
         │
         ▼
  Causal Self-Attn (所有用户相同)
         │
         ▼
  Cross-Attn (recent KV)  ← 用户 A ≠ 用户 B
  Cross-Attn (mid KV)     ← 用户 A ≠ 用户 B
  Cross-Attn (lifecycle KV) ← 用户 A ≠ 用户 B
         │
         ▼
  用户 A 的输出 ≠ 用户 B 的输出  ← 不可缓存！
```

**原因**：decoder 每一层的 cross-attention 都会读取**用户特定的 KV 记忆**，这些记忆来自 encoder 对该用户历史序列的编码。即使输入 token 相同，读取的"阅读材料"不同，生成的答案自然不同。

类比：就像所有考生拿到相同的题目 `[BOS]`，但每个人的参考书（三流 KV）不同，最终答案不可能一样，所以无法把答案缓存下来复用。

### Q10：三个 stream 无参数求和是什么意思？维度怎么控制相等？

#### 无参数求和的确切含义

**就是逐元素加法（element-wise addition），不加任何可学习参数。**

$$D^{(j)} = D^{(j)*}_{\text{recent}} + D^{(j)*}_{\text{mid-term}} + D^{(j)*}_{\text{lifecycle}}$$

- ❌ 不是拼接（concatenation）
- ❌ 不是加权求和（如 $\alpha_1 D_{\text{recent}} + \alpha_2 D_{\text{mid}} + \alpha_3 D_{\text{lc}}$，其中 $\alpha$ 可学习）
- ❌ 不是门控融合（如 sigmoid gate）
- ❌ 不是 MLP 投影后融合
- ✅ 就是三个张量对应位置直接相加，零参数

#### 维度怎么控制相等？

这是 **Transformer 架构设计的自然结果**，不是额外控制的。来看每个流的计算过程：

```
共享输入 D^(j)' ∈ R^(n_q × d_h)   ← 三个流用同一个 Query 输入
    │
    ├──────────────────────┬──────────────────────┐
    ▼                      ▼                      ▼
  Stream recent         Stream mid            Stream lifecycle
  (独立参数)            (独立参数)             (独立参数)
    │                      │                      │
    ▼                      ▼                      ▼
  CrossAttn             CrossAttn              CrossAttn
  (Q=D^(j)', K=H_recent, (Q=D^(j)', K=H_mid,  (Q=D^(j)', K=H_lc,
   V=H_recent)            V=H_mid)             V=H_lc)
    │                      │                      │
    │  残差: D' + CrossAttn(D', H, H)             │
    │  → 维度不变: R^(n_q × d_h)                   │
    ▼                      ▼                      ▼
   RMSN + FFN            RMSN + FFN            RMSN + FFN
    │                      │                      │
    │  残差: D + FFN(RMSN(D))                      │
    │  → 维度不变: R^(n_q × d_h)                   │
    ▼                      ▼                      ▼
  D_recent* ∈ R^(n_q×d_h) D_mid* ∈ R^(n_q×d_h) D_lc* ∈ R^(n_q×d_h)
    │                      │                      │
    └──────────────────────┴──────────────────────┘
                           │
                           ▼  逐元素相加 (维度天然匹配)
                    D^(j) ∈ R^(n_q × d_h)
```

**关键点**：
1. **隐藏维度统一为 $d_h = 1024$**（论文 Implementation Details 明确设置）
2. 三个流的 encoder 输出 $H_{\text{recent}}, H_{\text{mid}}, H_{\text{lifecycle}}$ 都投影到 $d_h$（通过各自的 behavior encoder + MLP）
3. Cross-Attention 的输出维度由 Query 维度决定（$d_h$），与 K/V 长度无关
4. 残差连接（Residual）强制输入输出维度一致
5. FFN 也是标准的 $d_h \to d_{\text{ff}} \to d_h$ 结构，输出维度仍是 $d_h$

所以三个流的输出**天然都是 $[n_q, d_h]$**（$n_q$ = decoder token 数，训练时 4，推理时 beam 数），可以直接逐元素相加。

#### 维度对比例子（训练时）

```
D^(j)'           : [4, 1024]  ← decoder 输入: 4个SID token, hidden=1024
H_recent         : [256, 1024] ← recent 流 KV
H_mid            : [4700, 1024] ← mid-term 流 KV
H_lifecycle      : [95000, 1024] ← lifecycle 流 KV (压缩后长度可能更短)

CrossAttn(D', H_recent, H_recent)  → [4, 1024]  ← 输出只和 Query 长度有关
CrossAttn(D', H_mid, H_mid)        → [4, 1024]
CrossAttn(D', H_lc, H_lc)          → [4, 1024]

+ 残差 + FFN: 保持 [4, 1024]

三个 [4, 1024] 张量逐元素相加 → [4, 1024]  ← 完美匹配
```

注意：K/V 的长度可以不同（256 vs 4700 vs 95000），但 Cross-Attention 的输出维度**只由 Query 决定**，所以三个流输出维度完全一致。

#### 为什么不用可学习权重？

论文 RQ3 消融实验验证：可学习融合（门控、MLP）在超长序列下会**坍缩为过度加权 recent stream**，因为 mid/lifecycle 与目标分布偏移更大，模型倾向于依赖更近的信号。无参数求和让三个流**等权贡献**，反而最鲁棒。

