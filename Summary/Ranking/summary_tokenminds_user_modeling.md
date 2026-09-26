# TokenMinds: 面向大型推荐系统用户理解的预训练用户 Token 与嵌入

## 基本信息

- **论文标题**: TokenMinds: Pretrained User Tokens and Embeddings for User Understanding in Large Recommender Systems
- **arXiv ID**: 2606.25147
- **作者**: Qingyun Liu 等（Google DeepMind / YouTube）
- **领域**: 推荐系统、用户建模、生成式推荐

---

## 核心思想（用户理解）

本文训练了一个模型（Encoder-Decoder 结构），训练语料是用户历史交互视频的序列（并 SID 化），训练目标是预测未来的交互视频：
- **Encoder**：产生 dense embedding（用户密集表示）
- **Decoder**：产生视频 SID 序列（用户兴趣的离散表示）

Encoder 输入用户截止时间之前的历史序列（SID + common features），其 hidden states 通过 cross-attention 提供给 Decoder。训练时，Decoder 输入不是另一份“近期历史”，而是未来目标 SID 序列的右移版本（teacher forcing），并自回归预测下一个 SID；推理时则从场景条件 Token 开始生成未来兴趣 SID。

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

### 3. SID 表示与 RQ-VAE 生成过程

#### 3.1 Full SID 与 TokenMinds SID prefix

TokenMinds 使用独立训练的 RQ-VAE/SID model 将视频内容特征编码为层级化 Semantic ID。论文明确给出的设置是：完整 item SID 包含 `L_full=8` 个 codewords，而用户建模只保留前 `L=4` 个 codewords。

| Representation | Length | Purpose | Mapping to videos |
|----------------|-------:|---------|-------------------|
| Full item SID | 8 codewords | 尽可能标识具体视频 | 接近一对一，但仍可能发生 collision |
| TokenMinds SID prefix/user token | First 4 codewords | 表示粗粒度未来兴趣区域 | Many-to-one，多个视频可共享同一 prefix |

TokenMinds 继承的 PLUM SIDv2 报告 full-SID uniqueness 为 96.7%，即约 96.7% 的视频拥有唯一完整 SID；其余约 3.3% 与其他视频发生 collision。论文没有披露 collision bucket 的平均值、最大值或分布。

对 TokenMinds 实际使用的 4-level prefix，论文也没有报告平均 `videos/prefix`。Prefix collision 是设计目标的一部分：同一粗粒度语义兴趣下的视频应共享前缀，Decoder 需要预测 interest region，而不是唯一恢复一个视频。静态 Prefix Embedding Mapping 会对共享同一 prefix 的所有视频 content embeddings 做 mean pooling，也说明该映射是 many-to-one。

因此，SID-prefix Recall@10 不能直接解释为 item Recall@10：命中目标视频的 prefix 不等于召回了目标视频本身。

#### 3.2 RQ-VAE/SIDv2 架构

TokenMinds 正文只重新说明了 `L_full=8`、`L=4` 和视频 content embeddings 输入；更完整的 SID model 设置来自它引用和延续的 PLUM SIDv2。TokenMinds 没有逐项确认所有 PLUM hyperparameters 均原样复用，因此下面需要理解为其 SID 构建方法的上游来源，而不是 TokenMinds 单独披露的新配置。

PLUM SIDv2 先融合多个预计算的多模态视频 embeddings，再执行 Residual Quantization：

```text
Embedding source x_1 → Encoder E_1 → z_1
Embedding source x_2 → Encoder E_2 → z_2
...
[z_1, z_2, ...] → Concatenate + Projection → z
                                              ↓
                                       Residual Quantizer
                                              ↓
                                   [sid_1, ..., sid_8]
```

每层递归量化 residual：

```text
r_0 = z
e_l* = nearest_code(r_{l-1})
r_l = r_{l-1} - e_l*
z_hat = sum(m_l * e_l*)
```

PLUM SIDv2 使用 multi-resolution codebooks，前层分辨率高、后层只编码低熵 residual：

| Quantization level | Codebook cardinality |
|-------------------:|---------------------:|
| 1 | 2048 |
| 2 | 1024 |
| 3 | 512 |
| 4 | 256 |
| 5 | 128 |
| 6 | 64 |
| 7 | 32 |
| 8 | 16 |

公式为 `K_l = 2048 / 2^(l-1)`。理论上，4-level prefix space 为 `2^38`，完整 8-level space 为 `2^60`；但真实 Encoder 只占用其中很小一部分层级路径，而且相似视频会被主动聚类，所以不能用理论组合数除以视频数来估计实际 bucket size。

PLUM SIDv2 还加入两个关键设计：

- **Progressive Masking**：训练时随机选择截断层级，只保留前若干量化层，使较短 prefix 本身也具有稳定的粗粒度语义。
- **Co-occurrence Contrastive Regularization**：将用户行为中共同出现的视频拉近、非共现视频推远，把 collaborative signal 注入内容量化空间。

整体训练目标为：

```text
L_SID = L_recon + L_rq + L_con
```

其中 `L_recon` 重建各模态输入 embeddings，`L_rq` 是 residual quantization 的 codebook/commitment loss，`L_con` 是视频共现对比损失。因此 SID 同时编码 multimodal content semantics 和 user co-occurrence semantics，而不是纯内容聚类。

#### 3.3 是否使用预训练 RQ-VAE？

有一个在 TokenMinds 之前独立训练的 RQ-VAE/SID model，但它与 Gemini Continued Pre-Training 是两个不同阶段：

```text
Stage 1: Train the SID tokenizer/indexer
Multimodal video embeddings + behavior co-occurrence
    → RQ-VAE/SIDv2
    → assign an 8-level SID to each corpus video

Stage 2: Ground SID tokens in the foundation model
SID vocabulary + video metadata + behavior sequences
    → Gemini Continued Pre-Training

Stage 3: Train user representations
Historical watches/searches
    → TokenMinds SFT for future SID prediction
```

论文没有说使用公开的预训练 RQ-VAE checkpoint，也没有披露 RQ-VAE 初始化方式；准确表述是：先用内部内容和行为数据独立训练 SID tokenizer/indexer，再用它给视频 corpus 生成固定 SID。TokenMinds SFT 没有描述向 RQ-VAE 反向传播，因此应理解为消费预先生成的 SID，而不是与 RQ-VAE 联合训练。

RQ-VAE 训练完成后还需要 Gemini CPT 来“理解”这些新离散符号。PLUM 的 CPT 将 user behavior data 与 SID-video metadata corpus 各占 50%，报告训练 100 万步、batch size 16、约 260B tokens；TokenMinds 明确从 PLUM-style SID-aligned CPT checkpoints 初始化 Encoder 和 Decoder，但没有确认完全复用这些 CPT 超参数。

总结来说：RQ-VAE 负责定义稳定的 item semantic vocabulary，Gemini CPT 负责把该 vocabulary 与语言及行为语义对齐，TokenMinds SFT 再把历史行为映射为未来兴趣 SID prefixes。

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

SID 前缀层级含义（仅作直观示意）：A12≈科技/AI 领域，B278≈大语言模型子类别，C23≈模型架构主题，D77≈具体技术方向。真实 RQ-VAE codewords 是数据驱动学习的离散码，论文没有为每个 codeword 公开人工可读标签，不能把上述名称理解为固定 taxonomy。

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

模型只对历史执行一次 Encoder forward，并将 3 个未来目标串联成一条 Decoder 监督序列：

```text
Encoder input（cutoff 前的历史）:
[<LFV>, A12, B278, C23, D77, SOFT_1,
 <LFV>, A12, B278, C45, D123, SOFT_2, ...]

Decoder labels（3 个目标串联）:
[<LFV>, A12, B278, C89, D156,
 <LFV>, A8,  B200, C45, D78,
 <LFV>, A15, B56,  C34, D90, <EOS>]

Decoder input（teacher forcing，labels 右移一位）:
[<BOS>, <LFV>, A12, B278, C89, D156,
        <LFV>, A8,  B200, C45, D78,
        <LFV>, A15, B56,  C34, D90]
```

这里是 **concatenation/packing**，不是 padding：不同样本的目标序列组成 batch 时，才会在末尾补 `<PAD>` 并 mask 掉这些位置。训练目标按自回归方式分解，后一个目标可以条件化于前面的目标，即公式中的 `W_{<i}`；但 teacher forcing 已提供完整右移输入，因此 Decoder 借助 causal mask，在一次 forward 中并行计算所有位置的 logits，并非真的循环调用模型逐 token 生成。真正逐 token 的 autoregressive generation 发生在推理阶段。场景 condition-token positions 不计算 loss；`<BOS>` 是标准右移关系的示意名称，论文未披露其实际 token 名称。

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

**SID Token 转换为嵌入**（通过 Learnable Embedding 方法）：每个 SID sequence 转成 task-specific interest embedding；下游既可以通过 Pooling 聚合为单个用户向量，也可以保留多个 interests 供 candidate-aware Cross-Attention 使用，详见 Q5。

##### Step 4: 缓存未命中，触发后台刷新

如果缓存过期或缺失，执行以下流程：

```
Step 4.1: 获取用户最新观看历史（最近1200条）
Step 4.2: TokenMinds模型推理
  - Encoder前向传递 → 1152维密集嵌入
  - 两个场景 context 分别进行 Beam Search (B=20)
  - 输出20个LFV SID序列 + 20个SFV SID序列
Step 4.3: 写入缓存，设置24小时过期时间
```

##### Step 5: 跨场景多上下文解码

一次 Encoder 前向传递生成共享用户表示，并行运行 LFV 和 SFV 两个解码分支。每个分支分别使用 `beam width B=20`，即各自维护 20 条候选路径；每条 Beam 从对应的场景 condition token 开始，逐个生成 4 个 SID codewords：

```text
共享Encoder传递 → 共享上下文表示
                    ↓
        ┌──────────┴──────────┐
        ↓                     ↓
    LFV分支              SFV分支
条件=<LFV>            条件=<SFV>
Beam width=20         Beam width=20
→ 20个LFV SID       → 20个SFV SID
```

因此，`B=20` 不是两个场景合计 20 条，而是每个场景 20 条，最终缓存共包含 40 条长度为 4 的 SID sequences。推理时没有真实 future targets 或 teacher forcing 输入，Decoder 只能使用场景 condition、Encoder states 和自己此前生成的 SID codewords。

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

#### 训练阶段：Encoder 与 Decoder 输入示例

假设 cutoff 之前有两次观看，未来 24 小时采样到两个目标视频：

```text
历史 W1（LFV）: SID = [A12, B278, C23, D77], common features → SOFT_1
历史 W2（SFV）: SID = [X3, Y15, Z78, W23],   common features → SOFT_2

未来目标 F1（LFV）: SID = [A12, B278, C89, D156]
未来目标 F2（SFV）: SID = [X3, Y42, Z11, W88]
```

**Encoder 输入**只包含 cutoff 之前的历史。每次观看由场景条件、4 个 SID codewords 和 1 个 soft token 组成：

```text
Encoder input =
[<LFV>, A12, B278, C23, D77, SOFT_1,
 <SFV>, X3,  Y15,  Z78, W23, SOFT_2]
```

Encoder 将上述序列编码成 hidden states；Decoder 在每一步都通过 cross-attention 读取这些 states。训练时，两个未来目标串成一条序列并采用 teacher forcing。为直观看清“右移”，可以写成：

```text
期望输出序列 =
[<LFV>, A12, B278, C89, D156,
 <SFV>, X3,  Y42,  Z11, W88, <EOS>]

Decoder input（右移一位）=
[<BOS>, <LFV>, A12, B278, C89, D156,
        <SFV>, X3,  Y42,  Z11, W88]

Next-token labels =
[<LFV>, A12, B278, C89, D156,
 <SFV>, X3,  Y42,  Z11, W88, <EOS>]
```

例如，当 Decoder 已读入 `[<BOS>, <LFV>, A12, B278]` 时，它结合 Encoder 历史表示预测 `C89`；完成 F1 后，再以 `<SFV>` 为条件生成 F2。论文的损失明确计算在 SID positions 上，因此 `<LFV>` / `<SFV>` 的 label positions 会被 mask；EOS 是否参与 loss 没有明确说明。`<BOS>` 仅用于展示标准右移关系，论文没有披露它的实际 token 名称。

**训练过程**：
1. Encoder 处理 cutoff 前的完整输入序列，产生上下文表示
2. Decoder 读取右移后的未来目标序列，并通过 Cross-Attention 关注 Encoder 输出
3. Decoder 自回归预测下一个 SID codeword；condition-token positions 不计算损失
4. 损失函数：多目标 Look-ahead 采样的交叉熵损失

#### Serving 阶段

```
输入 → Encoder → 上下文表示 → Pooling → 1152维 dense embedding
                   ↓
           Cross-Attention
                   ↓
       ┌──────────┴──────────┐
       ↓                     ↓
LFV context, Beam=20   SFV context, Beam=20
       ↓                     ↓
  20个LFV SID             20个SFV SID
```

#### 与 Decoder-only 的对比

| 维度 | Encoder-Decoder (TokenMinds) | Decoder-only (PLUM/OneRec) |
|------|-----------------------------|---------------------------|
| **输入处理** | Encoder 专门处理历史 | 历史和目标都在同一个 Decoder 中 |
| **嵌入提取** | 自然从 Encoder pooling 获得 | 需要额外的 pooling 层 |
| **部署** | 可解耦（低频 Encoder + 高频 Decoder） | 整体更新 |

---

### Q5: 下游模型如何使用用户表示？是否与 TokenMinds 共同训练？

#### 用户表示转换流程

KV Store 为每个用户缓存一个 1152-D dense embedding 和 40 条 SID sequences，其中 LFV、SFV 各 20 条。40 条是用户级缓存总量；具体下游客户端通常读取与当前 surface 对应的 20 条场景条件 SID，而不是无差别消费全部 40 条：

```text
KV Store
├── dense_embedding: 1 × 1152-D
└── sid_tokens
    ├── LFV: 20 × [4 SID codewords]
    └── SFV: 20 × [4 SID codewords]
             ↓ select current surface
        B=20 SID sequences
             ↓ Token Adaptation
        B continuous interest embeddings
             ↓ Pooling or Cross-Attention
        downstream ranking/retrieval model
```

具体客户端可以根据任务配置选择 SID 集合；论文报告的存储量仍按 40 条总输出计算。

#### Token-to-Embedding 的三种方法

| 方法 | 原理 | 是否可学习 | 是否与下游模型共同训练 |
|------|------|-----------|----------------------|
| **EM** | 将 SID prefix 静态映射到对应内容 embedding，再进行聚合 | 否 | 否（静态映射） |
| **LE-N-gram** | 将 SID codewords 切成 Unigram/N-gram，查询下游任务自己的 embedding table | 是 | **是**（端到端训练） |
| **LE-SPM** | 在序列化 SID 上用 SentencePiece 学习变长 pieces，再查询可学习 embedding table | 是 | **是**（端到端训练） |

对于下游任务 `t` 和第 `b` 条 SID sequence，可统一写成：

```text
SID_b = [A_i, B_j, C_k, D_l]
pieces_b = Tokenize_t(SID_b)
h_b = Pool(E_t[piece] for piece in pieces_b)
```

其中 `E_t` 由当前 ranking/retrieval loss 学习，维度也由下游模型决定。它与 TokenMinds Decoder 中用于 future-SID generation 的 vocabulary embedding 只共享 SID 符号，不共享参数和训练目标。

**关键结论**：论文的 pivot study 证明 LE 比静态 EM 更适合下游 ranking，但没有提供 Unigram、不同 N-gram 和 SPM 的统一同口径对比，因此不能进一步断言某一种 LE tokenization 始终最优。正式 SFV 实验主要采用 Unigram LE，LFV 辅助实验采用 SPM-based LE，报告的方向一致。

#### 两种下游融合方式

**方式一：先聚合为单个用户向量。** 每个 surface 的 `B=20` 条 SID sequences 经 LE 得到 `B` 个 interest embeddings，再通过 Attention、Mean 或 Max Pooling 形成一个普通连续特征：

```text
20 SID sequences
      ↓ LE
20 SID-interest embeddings
      ↓ Attention / Mean / Max Pooling
aggregated_token_embedding: d_downstream
      ↓ concatenate with dense embedding and other features
Downstream ranker
```

```text
最终输入特征
├── traditional user/context features
├── dense_embedding: 1152-D (Encoder pooling，可选)
└── aggregated_token_embedding: d_downstream (SID + LE + Pooling，可选)
```

**方式二：保留多兴趣做 candidate-aware Cross-Attention。** 不先把 `B` 个 SID interests 压成一个向量，而是让候选 embedding 作为 Query、SID embeddings 作为 Key/Value：

```text
Candidate embedding → Query
SID interests       → Key / Value
                     ↓ Cross-Attention
Candidate-specific user-interest feature
```

这种方式更直接地利用多个 Beam 的多兴趣结构，但计算成本高于一次 pooling。论文说明这些接口都可以使用 SID representation，却没有报告 pooling 与 Cross-Attention 在相同 ranker、参数量和服务预算下的严格消融；因此现有证据不能证明 Cross-Attention 一定优于 pooling。

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

### Q6: 为什么还要生成 SID 用户 Token？它是否只是 Decoder 的训练目标？

“SID Token 用于训练 Decoder”这个理解是正确的，但只覆盖了它的第一层作用。TokenMinds 选择 SID 作为目标，同时服务于**模型训练**和**下游用户表示**。

#### 作用一：为 Encoder-Decoder 提供可扩展的生成式监督

TokenMinds 没有为 dense user embedding 单独设计监督目标。训练损失只计算在 Decoder 预测的未来视频 SID 上，Encoder 通过 Decoder Cross-Attention 间接获得梯度：

```text
History → Encoder hidden states
             ↓ Cross-Attention
Future SID ← Decoder
             ↓
Autoregressive SID loss
```

因此，预测 future SID 确实是训练整个用户模型、让 Encoder 学到未来兴趣信息的关键。如果 SID 只承担这个作用，推理时可以丢弃 Decoder，只保留 Encoder dense embedding。

#### 作用二：突破单个 dense embedding 的容量瓶颈

论文没有丢弃 Decoder 输出，而是将 Beam Search 得到的多个 SID 序列作为第二种用户表示，因为它与 dense embedding 携带不同信息：

- **Dense embedding**：把完整历史 Pooling 成一个全局连续向量，接口简单稳定，但细粒度兴趣可能在固定维度瓶颈中被平均或覆盖。
- **SID user tokens**：40 个 Beam 分别表示不同的预测性未来兴趣，可以保留多兴趣结构，而不是把所有兴趣压进一个向量。
- **层级语义**：SID 前缀对应 item semantic space 中的粗粒度兴趣区域，比任意离散 user ID 更容易泛化到相似内容和新内容。
- **场景条件化**：同一次 Encoder 计算可以分别生成 LFV 和 SFV SID tokens，使用户表示带有明确的场景语义。

#### 作用三：提供多种下游消费接口

SID Token 不只用于产生一个 pooled continuous embedding，主要有三种消费方式：

1. **传统 Ranking/Retrieval 模型**：通过 EM、LE-N-gram 或 LE-SPM 将 SID 转成 continuous embeddings，聚合后作为普通特征输入。
2. **Cross-Attention 模型**：保留当前 surface 的多个 SID interest embeddings，作为 Key/Value 供候选 item 查询，不必先聚合为单一向量。
3. **原生支持 SID vocabulary 的生成式模型或 LLM**：理论上可以直接消费离散 SID tokens，无需额外 Token-to-Embedding adapter；但论文的正式线上证据主要来自前两类传统 production rankers。

对于传统连续模型，论文后续实验统一采用 **Learnable Embedding（LE）** 做 SID adaptation，因为线上 pivot study 显示 LE 优于静态 Prefix Embedding Mapping。LE 表由各下游模型独立训练，因此 token-derived continuous embedding 是 task-specific 的；相比之下，Encoder dense embedding 是冻结的通用上游表示。

#### 实际线上使用结论

论文比较了三种配置：

- **Embed-only**：只使用 Encoder dense embedding。
- **Token-only**：只使用 SID tokens 经 LE 得到的 continuous embeddings。
- **Embed+Token**：同时使用两种表示。

SID Token 并非只为训练而存在，因为 `Token-only` 本身能带来线上增益。SFV 上 `Embed+Token` 达到 `+0.11% Engaged Users / +0.62% Satisfied Engagement`，明显优于任何单一表示，说明两者具有互补性；但 LFV 上 `Embed-only` 的 Engaged Users 数值高于组合方案，所以不能概括为“组合在所有 surface 和所有指标上都最好”。

因此可以总结为：

> **SID prediction 是训练任务，SID user tokens 是训练任务产生的、可继续服务下游的显式多兴趣表示。** 如果只保留 dense embedding，Decoder 仍能充当训练器；但继续服务 SID Token 能保留 dense pooling 中容易损失的细粒度、场景化和 item-grounded 兴趣信号。

另外，论文只明确 Encoder dense embedding 为 1152 维；SID Token 经 LE 和 pooling 后的维度由下游 embedding table 与模型接口决定，不要求也是 1152 维。

---

## 实验结果与解读

### 1. 实验问题与配置

论文围绕三个研究问题组织实验：

1. **RQ1 - Token Adaptation and Viability**：SID 用户 Token 应如何接入传统连续模型，单独使用时能否产生线上增益？
2. **RQ2 - Complementary Values**：Encoder dense embedding 与 Decoder SID Token 是否携带互补信息？
3. **RQ3 - Cross-Scenario Modeling Impact**：联合建模 LFV 和 SFV 能否在不损失质量的前提下降低训练和服务成本？

正式模型基于 Gemini V1.5 的 Encoder-Decoder 架构：

| Component | Main configuration | Function |
|-----------|--------------------|----------|
| Encoder | 370M-parameter MoE Encoder | 读取完整历史，输出 contextual hidden states，并通过 pooling 产生 1152-D dense user embedding |
| Decoder | 370M-parameter Dense Decoder | Cross-Attend Encoder states，自回归生成未来兴趣 SID sequences |

Encoder 和 Decoder 都从 SID-aligned Continued Pre-Training（CPT）checkpoint 初始化，并不是“Gemini Decoder + 普通推荐 Encoder”。Dense embedding 也不是外部 Gemini embedding API 的输出，而是 Gemini-based Encoder hidden states 的 pooling 结果。

训练和服务配置如下：

- 每个用户样本读取最近最多 1200 次观看和最近 10 条搜索 Query，Token 化后的最大输入长度为 1024。
- 每次观看由一个场景条件 Token、4 个 SID prefix tokens 和一个 non-SID soft token 表示。
- 把 non-SID features 压成一个 soft token 会损失约 5% offline Recall，但显著缩短输入。
- 从未来 24 小时窗口中最多采样 15 次观看作为 Decoder targets。
- 持续训练每天使用最新数据；constant learning rate 比 same-day 最优的 warmup/cosine schedule 更能抵抗跨天分布漂移。
- 服务端每 24 小时刷新一次表示，每个用户输出一个 1152-D dense embedding，以及 20 个 LFV、20 个 SFV SID sequences。

### 2. 离线表示质量实验

#### 2.1 为什么使用 Beam Search Recall，而不是双塔内积？

离线实验首先验证 TokenMinds 自身的生成目标是否学好，还没有引入任何下游 adapter、item tower 或 ranker。评估定义为：

```text
History
  → TokenMinds Decoder
  → Beam Search Top-10 SID prefixes
  → target video's SID prefix 是否出现在 Top-10 中
```

因此，这里的 Recall@10 是 **SID-prefix prediction Recall@10**，而不是标准的 item-level retrieval Recall@10。一个长度为 4 的 SID prefix 可以对应多个具体视频，命中 prefix 不代表准确召回了某个 item。

采用 Beam Search 的原因是：

- Decoder 的训练目标就是 autoregressive future-SID generation，Beam Search 与训练目标直接对应。
- Encoder dense embedding 没有配套的 item tower，也没有通过 user-item dot-product/contrastive loss 直接训练，不能天然拿来做内积召回。
- 如果额外训练双塔或 ranker，指标会同时受到 adapter、item tower、negative sampling 和下游 loss 的影响，无法单独判断上游 SID 表示质量。

论文实际采用两阶段证据链：

```text
Stage 1: Upstream intrinsic evaluation
SID Beam Search → SID-prefix Recall@10

Stage 2: Downstream extrinsic evaluation
Cached dense/SID representations → production ranker → online A/B metrics
```

论文**没有报告** SID generative retrieval、dense two-tower retrieval 和精排模型在统一 item candidate set 上的 Recall/NDCG/AUC 对比，所以不能从现有结果回答三者差异有多大。公平对比需要把 SID prefixes 进一步展开为 items，并统一候选库、target 粒度、召回数和计算预算：

```text
SID path: Beam SID prefixes → expand items in SID buckets → item ranking
Dense path: user embedding × item embeddings → ANN Top-K
Ranker path: rerank the same retrieved candidates
```

#### 2.2 两种离线协议

| Protocol | Input | Target | Purpose |
|----------|-------|--------|---------|
| Session Recall | 近完整历史 `[W1, ..., Wn-1]` | 最后一次观看 `Wn` | 测试当前 Session 的即时兴趣预测 |
| Cold-Start Recall | 截断历史 `[W1, ..., Wt]` | 未来窗口中随机采样的观看 | 测试较少上下文下对未来兴趣的泛化 |

两种协议都生成 Top-10 SID prefix sequences，并计算 SID-prefix Recall@10。

#### 2.3 训练目标消融

| Model | Session Recall | Cold-Start Recall |
|-------|---------------:|------------------:|
| TokenMinds | 0.291 | 0.210 |
| Without multiple targets | 0.265 (-8.9%) | 0.203 (-3.3%) |
| Without look-ahead window | 0.278 (-4.5%) | 0.189 (-10.0%) |
| Without SID truncation | 0.247 (-15.1%) | 0.174 (-17.1%) |

三个设计分别解决不同问题：

- **Multiple targets** 提高单个样本的监督密度，并迫使 Decoder 覆盖多个未来兴趣，而不是只拟合一个 next item。
- **Look-ahead sampling** 对 Cold-Start 更重要，说明预测未来 24 小时中的兴趣比只预测紧邻行为更能学习可泛化表示。
- **SID prefix truncation** 的影响最大。Full SID 更接近具体 item identity，容易记忆视频；较短 prefix 强制模型预测粗粒度语义兴趣区域，对 Cold-Start 更友好。

#### 2.4 初始化与搜索信号

下表是相对 `Random initialization without search` 的 Recall@10 提升：

| Initialization | Search Queries | Session | Cold-Start |
|----------------|----------------|--------:|-----------:|
| Pre-Trained Gemini | No | +3.3% | +5.7% |
| SID-aligned CPT | No | +5.3% | +8.7% |
| Random | Yes | +12.5% | +16.9% |
| Pre-Trained Gemini | Yes | +18.5% | +25.1% |
| SID-aligned CPT | Yes | +23.5% | +31.5% |

结论是：搜索 Query 提供了观看历史之外的显式意图，收益大于单独更换初始化；CPT 又让模型预先建立 SID 与内容语义的对应关系，因此最能利用文本 Query 与行为 SID 的组合。

#### 2.5 Token diversity 与 dense embedding stability

- **Token diversity**：在 5K 用户上计算 SID Token Collision Rate 和 SID Prefix Duplication Rate。生成 Beam 的多样性与真实未来观看分布相近，说明多个 Beam 没有严重坍缩为同一个兴趣。
- **Embedding stability**：在 2K 用户上随机删除部分历史，原始与扰动 dense embeddings 的 cosine similarity 为 0.993，不同用户之间为 0.761。结果说明 embedding 对少量历史扰动稳定，但不同用户相似度仍较高，可能存在一定 embedding anisotropy；该实验不能替代 retrieval/ranking quality evaluation。

### 3. Token Adaptation Pivot Study

在投入正式双输出模型的大规模计算之前，论文先在 SFV surface 上使用一个 “lightweight 110M model” 比较两种 SID adaptation：

| Adaptation | Engaged Users | Satisfied Engagement |
|------------|--------------:|---------------------:|
| Static Prefix Embedding Mapping (EM) | +0.07% | -0.02% |
| Learnable Embedding (LE) | +0.08% | +0.22% |

论文没有进一步说明 110M 指整个 Encoder-Decoder、仅 Decoder，还是其他 pilot configuration，也没有说明它是否使用与正式模型完全相同的 Gemini CPT 初始化。因此不能把它确定为“110M Encoder-Decoder”，也不能直接等同于后续 scaling study 中的 `420M Encoder + 110M Dense Decoder`。

#### Decoder SID embedding 与下游 LE 并不相同

两套 embedding 共享 SID 符号，但参数、空间和训练目标都不同：

```text
TokenMinds Decoder:
SID ID → Decoder vocabulary embedding/output head
Objective: future SID generation

Downstream model:
SID ID or SID N-gram → task-specific LE table
Objective: CTR / engagement / ranking loss
```

对于下游任务 `t`，LE 可以表示为：

```text
z_sid = sum(E_t[ngram] for ngram in SID)
score = Ranker_t(candidate, z_sid, other_features)
```

Ranking loss 会更新 `Ranker_t` 和 `E_t`，但 TokenMinds 已冻结，不会更新 Decoder embedding。因此同一个 SID 在不同客户端可以有不同向量：

```text
E_SFV[SID] != E_LFV[SID] != E_Retrieval[SID]
```

Decoder embedding 学习“如何理解和生成 SID”，下游 LE 学习“该 SID 对当前业务目标意味着什么”。这就是 LE 能让每个下游模型学习 task-specific SID embedding space 的原因。后续正式实验统一采用 LE；SFV 使用 Unigram，LFV 的辅助实验使用 SPM-based LE，趋势一致。

### 4. 正式线上实验

线上 A/B 实验持续 7 天，覆盖多个 YouTube production ranking models。论文分别在 SFV 和 LFV surfaces 上报告 Engaged Users 与 Satisfied Engagement；粗体结果表示达到 95% 统计显著性。

| Surface / Representation | Engaged Users | Satisfied Engagement |
|--------------------------|--------------:|---------------------:|
| SFV - Embed-only | 0.00% | +0.05% |
| SFV - Token-only | +0.04% | **+0.40%** |
| SFV - Embed+Token | **+0.11%** | **+0.62%** |
| LFV - Embed-only | **+0.04%** | +0.03% |
| LFV - Token-only | +0.01% | +0.04% |
| LFV - Embed+Token | **+0.02%** | **+0.08%** |

结果说明：

- `Token-only` 自身已有线上收益，所以 SID Token 不是训练完即可丢弃的辅助标签。
- SFV 上 `Embed+Token` 在两个指标上都最好，SID 多兴趣信号带来的增益明显。
- LFV 上 `Embed-only` 的 Engaged Users 数值更高，而 `Embed+Token` 的 Satisfied Engagement 更高；因此“组合在每一个指标都最好”并不准确。
- 另外两个 LFV surfaces 上线 `Token-only` 后，也获得显著的 Engaged Users（+0.04%/+0.16%）和 Satisfied Engagement（+0.07%/+0.11%）增益。

#### 4.1 下游使用方式的消融证据链

这些实验不是一个单独的大表，而是通过 pilot study 和正式线上 A/B 逐层隔离变量：

| 要回答的问题 | 关键比较 | 主要控制变量 | 论文支持的结论 |
|-------------|---------|-------------|----------------|
| SID 如何转换更好？ | `LE` vs. `EM` | 同一 SFV pilot，改变 SID-to-embedding adapter | 可学习、task-specific LE 优于静态 prefix mapping，尤其改善 Satisfied Engagement |
| SID 本身是否可用？ | `Token-only` vs. production baseline | 不输入 dense embedding，只加入 SID-derived feature | SID Token 具有独立线上价值，不只是 Decoder 的训练标签 |
| Dense 与 SID 谁更好？ | `Embed-only` vs. `Token-only` | 同一 surface 分别只输入一种表示 | 结果依 surface/metric 而异；SFV 更受益于 SID，LFV 的 Engaged Users 更受益于 dense embedding |
| 两种表示是否互补？ | `Embed+Token` vs. 两个 single-branch variants | 同时暴露全局 dense signal 和多兴趣 SID signal | SFV 上组合严格最好；LFV 上组合的满意度最好，但 Engaged Users 不是最好 |
| SID 的额外代价多大？ | `Token-only` vs. `Embed+Token` 成本表 | 同一类 production clients | 组合质量更强但 serving-throughput 代价更大，需要看 quality-cost Pareto trade-off |

相应的 pairwise interpretation 是：

```text
LE - EM
  → 隔离“静态还是可学习 SID adapter”

Token-only - baseline
  → 隔离“SID representation 是否有独立增量”

Embed+Token - max(Embed-only, Token-only)
  → 检验两类表示是否包含互补信息
```

论文现有证据可以支持“LE 优于 EM”“SID 有独立价值”“Dense 与 SID 具有互补性”，但不能支持以下更强结论：

- **不能证明某种 LE tokenization 最优**：没有在同一模型、数据和预算下完整比较 Unigram、不同 N-gram 与 SPM。
- **不能证明 Cross-Attention 优于 Pooling**：论文没有报告 candidate-to-SID Cross-Attention 与 Mean/Max/Attention Pooling 的严格对照。
- **不能证明 SID 普遍优于 dense embedding**：LFV Engaged Users 的结果就是反例。
- **不能比较 SID generative retrieval 与 dense two-tower retrieval**：二者没有在统一 item candidate set、Top-K 和计算预算下评估。
- **不能证明 20 beams/场景是最优数量**：论文验证了生成多样性，但没有系统报告 beam-count quality-cost curve。

如果要完整回答“哪种下游使用方式最好”，合理的补充实验矩阵应固定相同 ranker、训练数据、candidate set 和尽量接近的参数/FLOPs，依次比较：

```text
No TokenMinds
Dense-only
SID + EM
SID + LE-Unigram
SID + LE-N-gram
SID + LE-SPM
Dense + best SID adapter
Pooled SID vs. candidate-to-SID Cross-Attention
Beam count: 1 / 5 / 10 / 20
```

评估不能只看离线 AUC/NE 或线上业务指标，还应同时报告训练成本、P99 latency、serving throughput、缓存大小和统计显著性，从而选择 quality-cost Pareto frontier，而不是只选择绝对收益最高的组合。

#### 下游成本与异步服务

| Metric | Token-only | Embed+Token |
|--------|-----------:|------------:|
| Training cost | +2.85% | +3.05% |
| Training speed | -0.7% | -4.2% |
| Serving throughput | -1.3% | -7.4% |

- 联合生成 dense embedding 和 SID tokens 每用户约需 339 ms，但全部由后台异步服务承担。
- 缓存命中率为 96.4%，支撑约 1.44M reads/s。
- 单个 SID representation 占 1280 bytes，dense embedding 占 4608 bytes，Token-only 存储减少 72%。这个数字不能理解为 `Embed+Token` 的总存储也减少 72%。
- 预计算 dense embedding 作为普通特征几乎没有额外在线计算；SID Token 的 LE lookup、聚合或 Cross-Attention 会产生额外下游成本。

### 5. SFV/LFV 跨场景实验

正式实验同时覆盖 SFV 和 LFV，但需要区分**统一上游用户模型**和**独立下游推荐客户端**：

```text
Chronologically interleaved LFV + SFV + Search history
                         ↓
                 Shared Encoder pass
                         ↓
        ┌────────────────┴────────────────┐
   <LFV> context                     <SFV> context
        ↓                                  ↓
20 LFV SID interests              20 SFV SID interests
        ↓                                  ↓
LFV production ranker             SFV production ranker
```

Unified TokenMinds 并不是把两类训练数据完全分开：

- LFV/SFV 行为按时间顺序交错进入同一个历史。
- 每个行为前添加 `<LFV>` 或 `<SFV>` condition token。
- Future targets 从 LFV 和 SFV 中均匀采样，防止高频场景主导训练。
- 推理时两个 context branches 共享一次 Encoder pass，但分别生成场景特定 SID tokens。

仍保留场景区分，是因为 SFV 属于连续滑动消费、通常没有显式 click initiation，反馈频率和 feedback loop 均不同于 LFV。两类内容的 item space、消费模式和下游指标也不同；没有 condition token 时，Decoder 无法判断应生成哪种场景的未来兴趣。

论文使用两套不同 baseline：质量与 LFV-only model 比较，效率与分别训练的 LFV/SFV 两个模型比较。

| Metric | SFV | LFV |
|--------|----:|----:|
| Engaged Users | +0.02% | +0.00% |
| Satisfied Engagement | +0.03% | +0.03% |
| Fresh Engagement | **+0.33%** | **+0.19%** |

| Resource phase | Unified model impact |
|----------------|---------------------:|
| Upstream training | -50% compute |
| Upstream serving | -31% compute |

服务成本从两个独立模型共 698 chips 降到统一模型 481 chips。统一模型在固定上下文中加入 SFV 后，可容纳的 LFV 历史几乎减半，但核心指标仍保持稳定，说明跨场景信号能够补偿部分同场景历史，并显著改善 fresh-content engagement。

### 6. Scaling Studies

Scaling study 使用加速离线协议：随机打乱的 7 天用于训练，按时间顺序的第 8 天用于 Recall@10 评估。

#### 架构与容量分配

比较的 warm-start variants 包括：

- **Balanced MoE**：370M Encoder / 370M MoE Decoder。
- **Balanced Dense**：370M Encoder / 370M Dense Decoder。
- **Unbalanced Dense**：420M Encoder / 110M Dense Decoder。

Unbalanced Dense 的 Training Recall 较低，但 8th-Day Recall 与 Balanced Dense 接近，表明表示质量更依赖 Encoder，而较轻 Decoder 仍可能满足未来兴趣解码。这为“低频刷新重 Encoder + 高频刷新轻 Decoder”的解耦服务方式提供了依据。相同 FLOPs 下，MoE Decoder 的 8th-Day Recall 高于 Dense Decoder。

#### 历史长度

LFV 和 SFV 的 8th-Day Recall@10 在约 1K watches 后开始饱和；扩展到 2K 对 SFV 持平或略有下降。这不能证明更久历史没有价值，也可能来自固定 context budget、远期噪声，以及 24 小时预测目标与长期行为之间的时间错配。

#### Batch size

| Batch size | SFV Recall improvement vs. 4K | LFV Recall improvement vs. 4K |
|------------|------------------------------:|------------------------------:|
| 8K | +2.5% | +5.5% |
| 16K | +7.6% | +13.7% |

更大的 Batch 在相同训练窗口内提高吞吐、降低梯度噪声并加速收敛，但最终选择仍受硬件容量约束。

### 7. Training Token Volume and Inference Storage

#### 7.1 用户窗口与训练样本组织

TokenMinds 的一个训练样本不是单次点击，而是一个 user-cutoff window：

```text
User sequence W1, ..., Wn
          ↓ choose cutoff T
History: watches/searches before T
Future window: behaviors in [T, T+24h]
          ↓
Sample up to 15 future watches as Decoder targets
```

历史侧使用最近最多 1200 次观看作为原始候选行为，未来侧固定为 24 小时窗口。论文没有披露每个用户每天产生几个 cutoff、cutoff 的采样方式、同一用户是否产生多个样本，以及 history 的实际时间跨度。因此，"millions of examples per day"表示百万级 user-cutoff windows，不等同于百万次观看，也不一定等同于百万个独立用户。

一个窗口最多聚合 15 个正样本，而不是为每个 future watch 重新构造一份相同 history：

```text
Single-target construction:
Same history × 15 targets → up to 15 repeated Encoder passes

TokenMinds multi-target construction:
Same history → one Encoder pass → up to 15 autoregressive targets
```

因此，多目标训练既增加监督密度，也把重 Encoder 计算分摊到最多 15 个目标上。真实分摊倍数取决于 future window 的平均有效 target 数，论文未披露该分布。

#### 7.2 Encoder 输入 Token 数

统一模型中，每次完整观看理论上占 6 个 positions：

```text
1 scenario condition token
+ 4 SID prefix tokens
+ 1 non-SID soft token
= 6 positions/watch
```

搜索 Query 还会占用一个 `<Search>` Token 和数量不定的文本 subword tokens。论文明确规定最大 Encoder sequence length 为 1024，因此：

```text
L_encoder <= 1024
```

忽略 Search 和其他特殊 Token 时，1024 positions 最多只能完整容纳约 `floor(1024/6)=170` 次观看；加入最多 10 条 Search Queries 后还会更少。这与“最近最多 1200 watches”的描述存在明显口径差异。最合理但未被论文确认的解释是：1200 watches 是预处理读取的原始候选历史，随后经过采样、截断或 packing，最终输入不超过 1024 positions。

因此，正式配置处理的是**有界近期历史**，不是用户完整生命周期行为。Scaling study 虽然出现 `HL=128/256/512/1024/2048 watches`，也没有解释这些 watch counts 如何映射到 1024-token 上限。

#### 7.3 Decoder 输入与监督 Token 数

多个 future targets 在训练时被连接为一个 autoregressive sequence。单场景下，每个目标包含 4 个 SID codewords，末尾再添加 EOS：

```text
L_decoder_single <= 15 targets × 4 SID tokens + 1 EOS
                 = 61 positions
```

统一 LFV/SFV 训练图中，每个 target 还带一个 `<LFV>` 或 `<SFV>` condition token。如果每个 target 都显式携带 condition，则最大长度约为：

```text
L_decoder_unified <= 15 × (1 condition + 4 SID) + 1 EOS
                  = 76 positions
```

Condition tokens 不计算 loss；公式明确只在 SID positions 上计算损失，因此每个样本最多有 `15×4=60` 个 SID-token supervised positions。EOS 是否计入 loss 未明确说明。Teacher forcing 时 Decoder input 是 target sequence 的 right-shifted 版本，所以输入长度与 61/76 基本相同。

| Per-sample component | Maximum positions | Supervised SID positions |
|----------------------|------------------:|-------------------------:|
| Encoder | 1024 | 0 direct token loss |
| Decoder, single-scenario | About 61 | Up to 60 |
| Decoder, unified | About 76 | Up to 60 |
| Encoder + unified Decoder | About 1100 | Up to 60 |

训练与 serving 的输出长度不同：训练时一个 Decoder sequence 最多串联 15 个 targets；serving 时每个 Beam 只生成一个长度为 4 的 SID prefix，40 个 Beams 是并行候选路径，不是一条长度为 160 的 Decoder sequence。

#### 7.4 每日训练 Token 量估算

论文只披露每天处理 “millions of examples”，没有给精确值。设每天有 `E million` 个训练样本，则最大 Token positions 约为：

```text
Encoder positions/day <= 1.024E billion
Decoder positions/day <= 0.076E billion
SID loss positions/day <= 0.060E billion
Total positions/day <= 1.10E billion
```

仅为展示数量级，如果 `E` 在 1 到 10 之间：

| Daily examples | Encoder positions | Decoder positions | SID loss positions |
|---------------:|------------------:|------------------:|-------------------:|
| 1M | <=1.024B | <=76M | <=60M |
| 5M | <=5.12B | <=380M | <=300M |
| 10M | <=10.24B | <=760M | <=600M |

实际均值应低于上限，因为用户历史不一定填满 1024 positions，未来窗口也不一定有 15 个目标。若实际是数千万样本，则上述结果按比例线性增加。

Scaling study 的 4K/8K/16K batch 对应的最大 global token counts 约为：

| Global batch | Encoder positions/batch | Unified Decoder positions/batch | SID loss positions/batch |
|-------------:|------------------------:|--------------------------------:|-------------------------:|
| 4K | 4.19M | 0.31M | 0.25M |
| 8K | 8.39M | 0.62M | 0.49M |
| 16K | 16.78M | 1.25M | 0.98M |

这些是 scaling experiments 的 global batch 估算，不代表 production run 的实际 batch。论文没有披露 micro-batch、gradient accumulation、并行策略、训练 GPU/TPU 数量或 wall-clock time。

#### 7.5 Inference 输出与存储

每个用户生成 20 个 LFV 和 20 个 SFV SID sequences。每个 sequence 是长度为 4 的 prefix，因此存在两种 Token 数口径：

| Counting convention | Count/user |
|---------------------|-----------:|
| Semantic user-interest tokens | 40 |
| Atomic SID codeword IDs | 160 = 40 × 4 |

如果假设每个 atomic SID code 强制按 FP32 存储：

```text
160 × 4 bytes = 640 bytes/user
```

但 SID 本质是整数 ID，并不需要使用 FP32。论文实际报告 SID representation 为 1280 bytes/user，恰好等于 `160×8 bytes`，与每个 code 使用 8-byte integer 的存储口径一致，但论文没有明确声明具体整数类型。

每个用户还缓存一个 1152-D dense embedding；其 4608 bytes 大小恰好对应 FP32：

```text
1152 × 4 bytes = 4608 bytes/user
```

| Cached representation | Raw storage/user |
|-----------------------|-----------------:|
| 160 SID codes, hypothetical FP32 | 640 bytes |
| SID representation, paper-reported | 1280 bytes |
| 1152-D dense embedding, FP32 | 4608 bytes |
| Dense + hypothetical FP32 SID codes | 5248 bytes = 5.13 KiB |
| Dense + paper-reported SID format | 5888 bytes = 5.75 KiB |

按论文实际双输出格式，每 10 亿用户的单副本 raw payload 约为 5.888 TB（5.355 TiB）。这不包括 user keys、TTL、版本信息、序列化、KV index、replication 和容灾；如果简单采用三副本，仅 payload 就约 17.7 TB。

在论文披露的 1.44M reads/s 下，如果每次都读取完整双输出，理论 raw payload bandwidth 约为 8.48 GB/s，其中 dense 部分约 6.64 GB/s，SID 部分约 1.84 GB/s。实际网络开销还会包含协议、key 和系统元数据。

SID 经 LE 转换后的 continuous embeddings 通常由下游客户端根据缓存 SID IDs 做 lookup，不必存入 TokenMinds KV。若选择持久化 40 个维度为 `d` 的 FP32 SID embeddings，额外存储将是 `40×d×4 bytes/user`；论文没有披露该 downstream dimension。

### 8. 实验证据的边界

实验较有力地证明了 SID Token 的独立线上价值、dense/token 双输出的互补性，以及统一跨场景上游模型的成本优势。但仍有以下边界：

- 没有提供 SID generative retrieval、dense two-tower retrieval 与精排模型的同口径 item-level 对比。
- 离线 Recall@10 衡量的是粗粒度 SID prefix，不是具体 item recall。
- 线上结果是把表示加入现有 production ranker 后的相对增益，无法单独归因到某一种召回机制。
- 历史实验主要覆盖约 1K-2K watches，不能证明已经捕获 lifecycle-level 全历史。
- 用户表示每 24 小时刷新一次，实验没有单独披露 freshness/staleness 的系统性消融。

---

## 核心贡献

1. **双输出架构**：统一生成密集嵌入和离散 SID 用户 Token
2. **基于 SID 的用户表示**：验证了离散 Token 在工业规模用户建模中的可行性
3. **跨场景建模**：通过多上下文解码实现 LFV/SFV 统一建模，显著降低计算成本
4. **工业部署**：在 YouTube 多个主要表面上线，服务数十亿用户

---

### 3. Request-level efficient history access

| Work | Method | Input | Architecture | Training Method | Output / Reuse |
|------|--------|-------|--------------|-----------------|----------------|
| **LONGER** ([2505.04421](https://arxiv.org/abs/2505.04421)) | Merges adjacent history and extracts it through a small set of global and recent queries | Long behavior sequence, non-sequential features, and a target item | Adjacent Token Merge + InnerTrans; Cross-Causal Attention; later self-attention only over `m+k` query outputs | End-to-end ranking; causal masking prevents target leakage into reusable history KV | Request-local compressed query states, target representation, and score; history KV can be reused across candidates but is not normally a persistent profile |
| **TM20K** ([2608.07055](https://arxiv.org/abs/2608.07055)) | Keeps full tokens in the teacher and merges tokens in the student | Up to about 20K behavior tokens plus the target item | LITM merges repeated IDs; PATM uses larger merge steps for older segments; LPTM forms a layer-wise token pyramid | Knowledge distillation from a full-token teacher to a merged-token student | Student ranking representation and score; average sequence length drops from about 8.8K to 1.8K, but no reusable user profile is produced |
| **SlimPer** ([2607.12281](https://arxiv.org/abs/2607.12281)) | Builds a fixed-size latent knowledge base for each `<user,item>` pair while repeatedly reading raw user tokens | User sequence, sparse/dense features, and the candidate item | `K=64, d=256` knowledge tokens refined through multi-layer Select-Match-Refine | End-to-end multi-task ranking objectives train latent queries and prediction heads | Candidate-conditioned knowledge base and logits; fixed hidden-state size but not candidate-independent or persistently cacheable |
| **SIM/TWIN** | Retrieves target-relevant Top-K events from lifelong history before exact sequence modeling | Very long user behavior history plus the current target item | General Search/target-aware retrieval followed by Exact Search/target attention; TWIN aligns features and parameters across stages | CTR/CVR ranking objectives with joint or staged training of retrieval and exact modeling | Candidate-conditioned Top-K history representation and score; efficient but discards non-retrieved events for that candidate |
| **STCA + RLB** ([2511.06077](https://arxiv.org/abs/2511.06077)) | Lets one target query access the full history linearly and shares the user path across candidates in the same request | Up to about 10K historical events plus a target item | Each layer uses the target state as Q and full history as K/V; Request-Level Batch shares user computation | Stochastic-length training with about 2K average training length and 10K inference length; request-level candidate/gradient aggregation | Candidate-conditioned representation and score; O(L) access and request-level reuse, but no token reduction or persistent user token |
| **HSTU** ([2402.17152](https://arxiv.org/abs/2402.17152)) | Sequentializes heterogeneous user actions and scales direct long-history modeling through efficient operators | Unified sequence of user actions, items, and context | Pointwise aggregated attention, U-gating, ragged kernels, and M-FALCON history-KV sharing | Generative next-action/next-item training with stochastic history length | Contextual sequence/user states and generative predictions; primarily architecture/kernel efficiency rather than precomputed compression |

这类方法解决的是“当前模型如何便宜地读取长历史”，并不等价于 TokenMinds 的“如何产出跨任务用户表示”：

- **LONGER/TM20K** 真正减少进入后续层的物理 Token 数，但表示通常只在当前模型或请求内有效。
- **SIM/TWIN** 通过 target-aware Top-K 将计算集中到相关行为，效率高，但不同候选会得到不同历史子集，难以预先生成统一用户表示。
- **SlimPer** 把中间激活限制在固定 latent knowledge base，却在每层读取原始用户 Token，最终表示仍由候选条件化。
- **STCA/HSTU** 尽量保留完整序列，依靠线性访问、ragged kernels、KV 复用和 stochastic length 提效；它们属于高效计算，而不是历史信息压缩。

------

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

- Liu et al., "TokenMinds: Pretrained User Tokens and Embeddings for User Understanding in Large Recommender Systems" (2026), arXiv:2606.25147.
- Doddapaneni et al., "User Embedding Model for Personalized Language Prompting" (2024), [arXiv:2401.04858](https://arxiv.org/abs/2401.04858).
- Xiong et al., "LLaTTE: Scaling Laws for Multi-Stage Sequence Modeling in Large-Scale Ads Recommendation" (2026), [arXiv:2601.20083](https://arxiv.org/abs/2601.20083).
- RecGPT Team, "RecGPT-V3 Technical Report" (2026), [arXiv:2607.15591](https://arxiv.org/abs/2607.15591).
- Yi et al., "RecGPT-V2 Technical Report" (2025), [arXiv:2512.14503](https://arxiv.org/abs/2512.14503).
- Li et al., "IAT: Instance-As-Token Compression for Historical User Sequence Modeling in Industrial Recommender Systems" (2026), [arXiv:2604.08933](https://arxiv.org/abs/2604.08933).
- Zhou et al., "GEMs: Breaking the Long-Sequence Barrier in Generative Recommendation with a Multi-Stream Decoder" (2026), [arXiv:2602.13631](https://arxiv.org/abs/2602.13631).
- "LONGER: Scaling Up Long Sequence Modeling in Industrial Recommenders" (2025), [arXiv:2505.04421](https://arxiv.org/abs/2505.04421).
- "TM20K: Teacher Retains Full Tokens, Student Merges Efficiently" (2026), [arXiv:2608.07055](https://arxiv.org/abs/2608.07055).
- Wang et al., "SlimPer: Make Personalization Model Slim and Smart" (2026), [arXiv:2607.12281](https://arxiv.org/abs/2607.12281).
- "Make It Long, Keep It Fast: End-to-End 10k-Sequence Modeling at Billion Scale on Douyin" (2025), [arXiv:2511.06077](https://arxiv.org/abs/2511.06077).
- Zhai et al., "Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations" (2024), [arXiv:2402.17152](https://arxiv.org/abs/2402.17152).
- Pi et al., "Search-based User Interest Modeling with Lifelong Sequential Behavior Data for Click-Through Rate Prediction" (SIM, 2020).
- Chang et al., "TWIN: TWo-stage Interest Network for Lifelong User Behavior Modeling in CTR Prediction at Kuaishou" (2023).
- Rajput et al., "Recommender Systems with Generative Retrieval" (TIGER, 2023).
- He et al., "PLUM: Adapting Pre-trained Language Models for Industrial-Scale Generative Recommendations" (2025), [arXiv:2510.07784](https://arxiv.org/abs/2510.07784).
- Liu et al., "OneRec-Think: Large Language Model with Explicit Reasoning for Recommendation" (2025).
