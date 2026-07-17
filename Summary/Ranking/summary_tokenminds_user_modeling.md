# TokenMinds: 面向大型推荐系统用户理解的预训练用户 Token 与嵌入

## 基本信息

- **论文标题**: TokenMinds: Pretrained User Tokens and Embeddings for User Understanding in Large Recommender Systems
- **arXiv ID**: 2606.25147
- **作者**: Qingyun Liu 等（Google DeepMind / YouTube）
- **领域**: 推荐系统、用户建模、生成式推荐

---

## 核心思想（用户理解）

本文训练了一个 LLM，训练语料是用户历史交互视频的序列（并 SID 化），训练目标是预测未来的交互视频。LLM 为 Encoder-Decoder 结构：
- **Encoder**：产生 dense embedding（用户密集表示）
- **Decoder**：产生视频 SID 序列（用户兴趣的离散表示）

这些用户表示作为缓存保留下来。当用户进来后：
1. 检索缓存获取 dense embedding 和 SID sequence
2. 或重新 serving 一次获取最新的用户表示
3. 用于下游的召回、排序任务

---

## 核心问题

工业推荐系统中的用户建模通常采用密集嵌入（dense embeddings），但这种方式存在以下局限性：

1. **表示约束**：固定维度的向量难以捕捉用户兴趣的完整谱
2. **泛化能力**：传统 LEM（Large Embedding Models）在复杂网络上的泛化能力有限
3. **文本用户画像的不足**：使用 LLM 生成文本用户画像往往只捕获主题共现，而非深度序列行为动态，且难以与商品属性对齐

现有的 SID（Semantic ID）范式已在商品表示上取得成功（如 PLUM 框架），但基于 SID 的**用户**离散表示仍未被探索。

---

## 核心方法：TokenMinds 框架

### 1. 双输出架构（Dual-Output Architecture）

TokenMinds 采用 **Encoder-Decoder 架构**，同时生成两种用户表示：

| 输出类型 | 来源 | 特点 |
|---------|------|------|
| **密集用户嵌入** | Encoder | 通过池化（last-token 或 mean pooling）获得，与现有下游模型兼容 |
| **离散 SID 用户 Token** | Decoder | 通过 Beam Search 生成多个 SID 序列，每个序列截断为粗粒度前缀作为单个用户 Token |

### 2. 训练目标

采用 **Look-ahead Sampling** 策略：
- 从未来 24 小时窗口中随机采样最多 N=15 个目标视频
- 避免过拟合到紧邻的下一个观看视频，提升泛化能力
- 损失函数考虑参与度奖励（engagement reward），鼓励多样化和高价值消费

$$\mathcal{L} = - \sum_{i=1}^{N} r(W_i) \cdot \sum_{j=1}^{L} \log P(SID_{i,j} \mid W_1, \dots, W_t, \, W_{<i}, \, SID_{i,<j})$$

### 3. SID 表示

- 使用 RQ-VAE 将视频内容特征编码为层级化的离散码本（Semantic ID）
- 保留前缀 L=4 的 SID 码，利用层级结构在较粗粒度上表示视频
- 优势：更好的泛化能力、更高的时间稳定性（缓解词汇表更新问题）

### 4. 跨场景建模（Cross-Scenario Modeling）

**统一训练**：
- 在每个观看行为前添加场景条件 Token（`&lt;LFV&gt;` / `&lt;SFV&gt;`）
- 支持搜索查询的自然集成（添加 `&lt;Search&gt;` Token）
- 时序交错的跨场景序列训练

**多上下文解码（Multi-Context Decoding）**：
- 单次 Encoder 前向传递生成共享用户表示
- Decoder 通过不同场景前缀条件化，并行生成场景特定的用户 Token
- 消除冗余计算，实现场景感知输出

### 5. 下游适配策略

将离散 SID 用户 Token 投影到连续空间的三种方法：

| 方法 | 原理 | 特点 |
|------|------|------|
| **Prefix Embedding Mapping** | 将预测的 SID 前缀映射回原始内容嵌入，mean-pool 共享相同前缀的视频 | 静态映射，无需额外训练 |
| **N-gram Embedding** | 将 SID 序列切分为固定长度 N-gram，每个子词映射到学习嵌入 | 可学习，端到端训练 |
| **SPM Embedding** | 使用 SentencePiece 学习变长子词 | 可学习，端到端训练 |

**实验结论**：Learnable Embedding（LE）优于静态映射（EM）。

### 6. 服务架构

基于 **UBS（User Behavior Service）** 的异步服务框架：
- 用户表示异步生成并缓存到 KV 存储
- 实时评分直接读取缓存表示
- 缓存过期或缺失时，后台 Refresh Service 重新生成

---

## 讨论与问答

### Q1: 使用一个例子说明训练全过程

**假设场景**：用户 Alice，科技爱好者，喜欢观看 AI 相关视频

#### Step 1: 数据准备与时间划分

Alice 的观看序列：

| 时间 | 视频内容 | 观看时长 | 设备 |
|------|---------|---------|------|
| Day 1 10:00 | "GPT-5 最新进展" | 15分钟 | PC |
| Day 1 14:30 | "LLM 推理优化" | 8分钟 | Mobile |
| Day 2 09:00 | "Diffusion 模型原理" | 20分钟 | PC |
| Day 2 20:00 | "Mamba 架构解析" | 12分钟 | Mobile |
| Day 3 11:00 | "Gemini 1.5 评测" | 18分钟 | PC |
| Day 3 16:00 | "RAG 实战教程" | 25分钟 | PC |

**时间划分**：
- Cutoff T = Day 3 12:00
- History = 前5个视频（Day 1-3 11:00）
- Future Window = "RAG 实战教程" 及未来24小时内的观看

#### Step 2: SID 编码（视频语义化）

每个视频通过 RQ-VAE 编码为层级化 SID（完整长度 L_full=8），保留前缀 L=4：

```
"GPT-5 最新进展" → SID完整: [A12, B278, C23, D77, E15, F99, G42, H101]
                    → SID前缀: [A12, B278, C23, D77]

"LLM 推理优化" → SID前缀: [A12, B278, C45, D123]

"Diffusion 模型原理" → SID前缀: [A8, B156, C78, D34]

"Mamba 架构解析" → SID前缀: [A5, B99, C67, D201]

"Gemini 1.5 评测" → SID前缀: [A12, B278, C23, D77]  ← 与GPT-5视频共享前缀
```

SID 前缀层级含义：A12=科技>AI领域，B278=大语言模型子类别，C23=模型架构主题，D77=具体技术方向

#### Step 3: 输入 Token 构建

每个观看记录 = 场景条件 Token + SID 硬 Token + 非 SID 软 Token

```
观看1: [<LFV>, A12, B278, C23, D77, SOFT_1]
       ↑        ↑                          ↑
   场景标记   4个SID硬Token          软Token(观看时长15min, 设备PC, 点赞)

观看2: [<LFV>, A12, B278, C45, D123, SOFT_2]

...（共5个观看记录）
```

软 Token 生成过程：原始特征 → 独立嵌入 → 拼接 → MLP 投影 → 单个 SOFT 向量

#### Step 4: Look-ahead 目标采样

从未来窗口随机采样 N=3 个目标视频：

```
采样目标:
  1. "RAG 实战教程" → SID前缀: [A12, B278, C89, D156]
  2. "PyTorch 2.0 新特性" → SID前缀: [A8, B200, C45, D78]
  3. "视觉Transformer 入门" → SID前缀: [A15, B56, C34, D90]
```

#### Step 5: 训练目标计算

模型预测这3个目标视频的 SID 前缀序列：

```
输入序列 (Encoder):
[<LFV>, A12, B278, C23, D77, SOFT_1, <LFV>, A12, B278, C45, D123, SOFT_2, ...]

目标序列 (Decoder):
目标1: [<LFV>, A12, B278, C89, D156]
目标2: [<LFV>, A8, B200, C45, D78]  
目标3: [<LFV>, A15, B56, C34, D90]
```

损失函数：

$$\mathcal{L} = - \sum_{i=1}^{3} r(W_i) \cdot \sum_{j=1}^{4} \log P(SID_{i,j} \mid History, W_{<i}, SID_{i,<j})$$

其中奖励 $r(W_i)$ 根据参与度设定：$r(W_1)=1.5$（高）、$r(W_2)=1.0$（中）、$r(W_3)=0.8$（低）

梯度流向：Decoder 损失 → Cross-Attention → Encoder（隐式学习用户表示）

#### Step 6: 跨场景扩展（可选）

如果 Alice 同时观看短视频和搜索：

```
完整输入序列:
[<LFV>, A12, B278, C23, D77, SOFT_1,
 <SFV>, X3, Y15, Z78, W23, SOFT_6,    ← 短视频观看
 <Search>, "AI 最新论文",              ← 搜索查询
 <LFV>, A8, B156, C78, D34, SOFT_3]
```

### Q2: TokenMinds 线上如何进行推理？以一个例子说明

**整体架构**：基于 UBS（User Behavior Service）的异步缓存服务，将用户表示生成与实时评分解耦。

#### 推理流程示例

**场景**：用户 Alice 打开 YouTube 首页，系统需要为她推荐视频。

##### Step 1: 请求到达，查询缓存

当 Alice 打开 YouTube 时，实时评分服务首先查询缓存：

```
用户ID: alice_123
缓存查询 Key: "tokenminds_user_rep:alice_123"

缓存命中情况检查:
├── 情况A: 缓存有效 → 直接返回 (96.4% 命中率)
└── 情况B: 缓存过期/缺失 → 触发后台刷新
```

##### Step 2: 缓存命中，直接使用

如果缓存有效（最近24小时内刷新过），直接读取缓存的用户表示：

```
缓存内容 (KV Store):
{
  "user_id": "alice_123",
  "dense_embedding": [0.12, -0.34, 0.56, ...],  // 1152维密集嵌入
  "sid_tokens": {
    "lfv": [                                       // 20个LFV SID序列
      [A12, B278, C23, D77],
      [A12, B278, C45, D123],
      [A8, B156, C78, D34],
      ...
    ],
    "sfv": [                                       // 20个SFV SID序列
      [X3, Y15, Z78, W23],
      [X5, Y99, Z67, W201],
      ...
    ]
  },
  "refresh_time": "2026-07-17 08:30:00",
  "expire_time": "2026-07-18 08:30:00"
}
```

##### Step 3: 下游模型使用用户表示

**密集嵌入**直接作为特征输入到排序模型。

**SID Token 转换为嵌入**（通过 Learnable Embedding 方法）：每个 SID 序列转换为嵌入，然后通过注意力池化聚合为单个用户向量。

##### Step 4: 缓存未命中，触发后台刷新

如果缓存过期或缺失，执行以下流程：

```
Step 4.1: 获取用户最新观看历史（最近1200条）
Step 4.2: TokenMinds模型推理
  - Encoder前向传递 → 1152维密集嵌入
  - Decoder Beam Search (B=40) → 40个SID序列
Step 4.3: 写入缓存，设置24小时过期时间
```

##### Step 5: 跨场景多上下文解码

一次 Encoder 前向传递生成共享用户表示，并行生成 LFV 和 SFV 的 SID 序列：

```
共享Encoder传递 → 共享上下文表示
                    ↓
        ┌──────────┴──────────┐
        ↓                     ↓
    LFV分支              SFV分支
条件=<LFV>            条件=<SFV>
→ 20个LFV SID       → 20个SFV SID
```

#### 性能指标

| 指标 | 值 |
|------|-----|
| **单用户推理时间** | ~339ms |
| **缓存命中率** | 96.4% |
| **在线QPS** | 1.44M 读取请求/秒 |
| **缓存刷新周期** | 24小时 |
| **SID序列数量** | 40个/用户 (20 LFV + 20 SFV) |
| **密集嵌入维度** | 1152维 |
| **Token存储大小** | 1,280字节 (比嵌入少72%) |

#### 与 ShopX 推理的对比

| 维度 | TokenMinds | ShopX |
|------|-----------|-------|
| **推理模式** | 异步缓存，离线生成 | 在线实时推理 |
| **延迟** | 缓存命中: 极低；未命中: 后台异步 | 每次请求都需模型推理 |
| **适用场景** | 大规模推荐，全量用户 | Agentic购物对话 |
| **输出** | 用户表示（嵌入+Token） | 商品SID + 自然语言响应 |

### Q3: 下游模型如何使用用户表示？是否与 TokenMinds 共同训练？

### Q4: TokenMinds 是否使用 Encoder-Decoder 架构？输入输出流程是怎样的？

**是的，TokenMinds 使用的是 Encoder-Decoder 架构**。

#### 整体架构流程

```
用户观看序列 (SID + 软Token) → Encoder → 上下文表示 → Pooling → dense embedding
                                    ↓
                              Cross-Attention
                                    ↓
                              Decoder → Beam Search → SID序列 (LFV/SFV)
```

#### 设计动机

选择 Encoder-Decoder 的两个原因：

1. **Encoder 更擅长捕获完整序列模式**：能够从完整用户历史中提取更好的上下文表示，适合生成密集嵌入
2. **部署灵活性**：Encoder 和 Decoder 可以解耦部署，低频重 Encoder 用于历史压缩，高频轻 Decoder 用于捕捉近期行为

#### 输入序列结构

每个观看记录包含：

```
W_k = [条件Token, SID前缀(L=4), 软Token]

例如: [<LFV>, A12, B278, C23, D77, SOFT_1]
```

- **条件Token**：`<LFV>` / `<SFV>` / `<Search>`
- **SID前缀**：4个硬Token，表示视频的语义类别
- **软Token**：非SID特征通过 MLP 投影得到的单个向量

#### 训练阶段

```
输入 (Encoder):
[<LFV>, A12, B278, C23, D77, SOFT_1,
 <LFV>, A8, B156, C78, D34, SOFT_2, ...]

目标 (Decoder):
目标1: [<LFV>, A12, B278, C89, D156]
目标2: [<LFV>, A8, B200, C45, D78]  
目标3: [<LFV>, A15, B56, C34, D90]
```

**训练过程**：
1. Encoder 处理完整输入序列，产生上下文表示
2. Decoder 通过 Cross-Attention 关注 Encoder 的输出
3. Decoder 自回归生成目标视频的 SID 序列
4. 损失函数：多目标 Look-ahead 采样的交叉熵损失

#### Serving 阶段

```
输入 → Encoder → 上下文表示 → Pooling → 1152维 dense embedding
                   ↓
           Cross-Attention
                   ↓
           Decoder Beam Search (B=40)
                   ↓
           ┌──────┴──────┐
           ↓             ↓
       LFV分支        SFV分支
   20个LFV SID      20个SFV SID
```

#### 与 Decoder-only 的对比

| 维度 | Encoder-Decoder (TokenMinds) | Decoder-only (PLUM/OneRec) |
|------|-----------------------------|---------------------------|
| **输入处理** | Encoder 专门处理历史 | 历史和目标都在同一个 Decoder 中 |
| **嵌入提取** | 自然从 Encoder pooling 获得 | 需要额外的 pooling 层 |
| **部署** | 可解耦（低频 Encoder + 高频 Decoder） | 整体更新 |

---

### Q3: 下游模型如何使用用户表示？是否与 TokenMinds 共同训练？

#### 用户表示转换流程

无论缓存命中与否，KV Store 中的内容都包含两个部分，处理方式不同：

```
KV Store 内容:
├── dense_embedding: 1152维 (直接来自 Encoder pooling)
└── sid_tokens: 40个序列 (20 LFV + 20 SFV)
        ↓
        ↓ Token-to-Embedding 转换 (LE 方法)
        ↓
    40个 SID 嵌入向量
        ↓
        ↓ Attention/Mean/Max Pooling
        ↓
    aggregated_token_embedding: 1152维 (单个用户向量)
```

#### Token-to-Embedding 的三种方法

| 方法 | 原理 | 是否可学习 | 是否与下游模型共同训练 |
|------|------|-----------|----------------------|
| **EM** | 将 SID 前缀映射回原始内容嵌入，mean-pool | 否 | 否（静态映射） |
| **LE-N-gram** | 切分为固定长度 N-gram，映射到学习嵌入表 | 是 | **是**（端到端训练） |
| **LE-SPM** | 使用 SentencePiece 学习变长子词 | 是 | **是**（端到端训练） |

**关键结论**：LE 优于 EM，因为它允许下游模型学习专门的嵌入空间。

#### 聚合为单个用户向量

Beam Search 产生 B=40 个 SID 序列，通过以下方式聚合：

```python
aggregated_token_embedding = attention_pooling(sid_embeddings)
# 或 mean_pooling / max_pooling，效果相当
```

论文指出：聚合方式不是关键，关键在于 Token 本身携带的信息。

#### 下游模型的使用方式

最终有两个用户向量输入下游模型：

```
最终输入特征:
├── dense_embedding: 1152维 (来自 Encoder)
└── aggregated_token_embedding: 1152维 (来自 SID Token + LE + Pooling)
```

**使用方式**：
1. **直接作为输入特征**：拼接到其他用户特征中，输入到下游排序/检索模型
2. **作为 Cross-Attention 的 Key-Value**：候选商品通过 Cross-Attention 关注用户表示

#### 训练关系

```
┌─────────────────────────────────────────────────────────────┐
│                    训练阶段                                  │
├─────────────────────────────────────────────────────────────┤
│  TokenMinds 模型  ← 独立训练（预测未来观看的 SID）           │
│      ↓                                                      │
│  导出用户表示（dense embedding + sid tokens）               │
│      ↓                                                      │
│  下游模型（排序/检索/LLM）  ← 独立训练，但 LE 嵌入表一起训练 │
│      ↓                                                      │
│  LE 嵌入表的梯度会更新，但 TokenMinds 模型参数保持冻结       │
└─────────────────────────────────────────────────────────────┘
```

**关键点**：
- **TokenMinds 模型是独立训练的**，预测未来观看的 SID 序列
- **下游模型是独立训练的**，但 **LE 嵌入表与下游模型端到端共同训练**
- TokenMinds 的参数在下游训练时是 **冻结的**（frozen），只有 LE 嵌入表会被更新

#### 下游模型类型

- **排名模型（Ranking Models）**：论文重点讨论
- **检索模型（Retrieval Models）**：用于召回阶段
- **LLM 系统**：生成式推荐

#### 训练成本对比

| 集成方式 | 训练成本增加 | 训练速度变化 | 服务吞吐量变化 |
|---------|------------|------------|--------------|
| Token-only | +2.85% | -0.7% | -1.3% |
| Embed + Token | +3.05% | -4.2% | -7.4% |

---

## 实验结果

### 1. 离线实验

**训练目标消融**（Recall@10）：

| 模型变体 | Session Recall | Cold-Start Recall |
|---------|---------------|------------------|
| TokenMinds (Ours) | 0.291 | 0.210 |
| w/o Multiple Targets | 0.265 (-8.9%) | 0.203 (-3.3%) |
| w/o Look-ahead Window | 0.278 (-4.5%) | 0.189 (-10.0%) |
| w/o SID Truncation | 0.247 (-15.1%) | 0.174 (-17.1%) |

**初始化策略**：CPT（Continued Pre-Training）> Pre-Trained Gemini > Random

**搜索查询的影响**：添加 10 个搜索查询可提升 Recall@10 达 +23.5%（Session）和 +31.5%（Cold-Start）

### 2. 在线实验

**下游质量对比**（SFV/LFV 表面）：

| 表示类型 | Engaged Users | Satisfied Engagement |
|---------|---------------|---------------------|
| **SFV - Embed-only** | 0.00% | +0.05% |
| **SFV - Token-only** | +0.04% | +0.40% |
| **SFV - Embed+Token** | **+0.11%** | **+0.62%** |
| **LFV - Embed-only** | **+0.04%** | +0.03% |
| **LFV - Token-only** | +0.01% | +0.04% |
| **LFV - Embed+Token** | **+0.02%** | **+0.08%** |

**关键发现**：
- SID 用户 Token 在生产系统中提供增量价值
- 嵌入 + Token 的组合产生放大效应，验证了双输出设计的互补性

**跨场景建模效率**：
- 训练计算减少 50%（单模型替代两个独立模型）
- 上游服务计算减少 31%（多上下文解码共享 Encoder）

### 3. 缩放研究

- **历史长度**：1K 观看历史后性能开始饱和
- **批次大小**：16K batch 比 4K 提升 Recall@10 达 +7.6%/+13.7%（SFV/LFV）
- **架构变体**：MoE Decoder 在相同 FLOPS 下优于 Dense Decoder

---

## 核心贡献

1. **双输出架构**：统一生成密集嵌入和离散 SID 用户 Token
2. **基于 SID 的用户表示**：验证了离散 Token 在工业规模用户建模中的可行性
3. **跨场景建模**：通过多上下文解码实现 LFV/SFV 统一建模，显著降低计算成本
4. **工业部署**：在 YouTube 多个主要表面上线，服务数十亿用户

---

## 与现有工作的关系

| 对比维度 | TokenMinds | PLUM | LIGER/COBRA |
|---------|-----------|------|-------------|
| 关注点 | 用户建模 | 商品检索 | 商品表示 |
| 输出类型 | 嵌入 + Token | Token | 稀疏 + 稠密 |
| 场景 | 跨场景（LFV/SFV） | 单一场景 | 单一场景 |

---

## 关键洞察

1. **离散与稠密的互补性**：Token 捕获细粒度兴趣信号，嵌入提供全局表示，两者组合效果最优
2. **CPT 的重要性**：SID 语义对齐的预训练显著提升下游任务性能
3. **搜索信号的价值**：文本搜索查询与观看历史互补，增强用户意图理解
4. **异步服务的必要性**：LLM 推理成本高，需通过缓存机制实现实时服务
5. **粗粒度 SID 的优势**：前缀截断提升多样性，缓解过拟合问题

---

## 与 ShopX 的对比分析

### 核心定位对比

| 维度 | TokenMinds (2606.25147) | ShopX (2606.31693) |
|------|------------------------|---------------------|
| **作者** | Google DeepMind / YouTube | ShopX Team (阿里巴巴) |
| **核心问题** | 密集嵌入的表示约束与泛化能力 | 工具中介式架构的接口信息损失 |
| **关注点** | **用户建模**：如何更好地表示用户 | **商品履约**：如何从意图到商品 |
| **领域** | 视频推荐（LFV/SFV） | 电商购物（淘宝） |
| **核心范式** | 生成式用户表示 | 模型原生履约 |

### 技术架构对比

#### 模型架构

**TokenMinds**: Encoder-Decoder 架构
- Encoder 捕获完整序列模式生成密集嵌入
- Decoder 自回归生成离散 SID Token
- 可解耦部署（低频重 Encoder + 高频轻 Decoder）

**ShopX**: Decoder-only LLM + Serving Harness
- 四槽位动作协议：Plan → Execute → Fulfill → Update
- 统一意图理解、执行规划和商品空间操作

#### SID 设计

**TokenMinds**: 前缀截断的层级化 SID
- 完整长度 L_full=8，保留前缀 L=4
- RQ-VAE 编码视频内容特征
- 粗粒度，鼓励多样性

**ShopX**: G₂+L₄ 混合全局-局部 SID
- 2 个全局前缀（RQ）+ 4 个局部后缀（VQ）
- 全局前缀保证生成稳定性，局部后缀保留细粒度属性

### 核心功能对比

| 设计理念 | TokenMinds | ShopX |
|---------|-----------|-------|
| **SID 的角色** | 用户兴趣的离散表示 | 商品空间的操作语言 |
| **表示方式** | 双输出（嵌入 + Token） | 单一输出（SID + 文本） |
| **训练目标** | 预测未来观看（自监督） | 商品检索+排序+响应（多任务） |
| **场景处理** | 跨场景统一建模 | 单一场景深度优化 |
| **服务策略** | 异步缓存，降低在线成本 | 在线推理，实时交互 |
| **与现有系统兼容** | 完全兼容（输出嵌入） | 模型原生，需全新架构 |

### 互补关系与整合可能性

**互补关系**：
- TokenMinds 擅长：用户兴趣建模、跨场景表示、大规模实时服务
- ShopX 擅长：意图理解、商品空间操作、多轮对话履约

**整合方案**：
```
用户行为历史 → TokenMinds 生成用户表示 → ShopX 接收用户表示 → 意图理解 → 商品检索 → 响应生成
                        ↑                                                        ↑
                  密集嵌入 + SID Token                                  利用用户表示增强个性化
```

具体整合点：
1. **用户表示注入**：TokenMinds 生成的用户嵌入和 SID Token 可以作为 ShopX 的 Context 输入
2. **跨场景扩展**：TokenMinds 的多上下文解码可以为 ShopX 提供跨场景用户兴趣
3. **服务优化**：TokenMinds 的异步缓存架构可以降低 ShopX 的在线推理成本

---

## 参考文献

- Rajput et al., "TIGER: Token-based Item Graph Representation for Recommendation" (2023)
- He et al., "PLUM: Pre-trained Large Unified Models for Generative Recommendation" (2025)
- Liu et al., "OneRec-Think: Large Language Model with Explicit Reasoning for Recommendation" (2025)