# UniSGR: 语义 ID 生成与排序的统一框架

> **论文信息**: UniSGR: Unified Framework for Semantic ID Generation and Ranking  
> **arXiv**: 2607.04068  
> **机构**: 阿里巴巴国际数字商业集团（Alibaba International Digital Commerce Group）  
> **关键词**: 生成式推荐、语义 Token 化、自回归生成、Scaling Law

---

## 1. 研究背景与动机

推荐系统在现代电商平台中至关重要，工业级系统通常采用**级联流水线架构**：召回 → 粗排 → 精排 → 重排。这种架构的根本问题在于各模块只优化局部子任务，而非端到端推荐目标，导致上游召回过滤掉的相关商品无法在下游被恢复。

**生成式召回**（Generative Retrieval）作为一种有前景的替代范式，通过直接生成离散语义 ID 来统一候选生成，但现有方法在**细粒度多目标排序**能力上仍然受限：
- 生成器只能选择语义相关的候选，缺乏工业排序所需的精细区分度
- 在生成式召回后再加一个判别式排序器会重新引入**目标不匹配**问题：生成器优化生成合理候选，而排序器在截断候选集上优化最终效用
- 多业务场景下，仅在目标场景训练会限制知识迁移能力，简单混合所有数据则会削弱场景特定适配

为此，作者提出 **UniSGR**，将生成式召回与判别式排序耦合在单个学习框架中，将排序导向的监督信号注入语义 ID 生成，使生成候选与下游业务目标更好对齐。

---

## 2. 核心方法

### 2.0 工作流程速览（读者理解 + 点评）

**读者理解**：本文的工作流程是：Qwen3-VL based decoder, 产生多组候选语义 ID (c1,c2,c3). 然后每个拼接上 decoder 中产生的 hidden states 作为特征，作为 target 参与 Cross-Attention 的打分，KV 是用户的序列特征。得到 Target 表达后再通过 PLE（MOE 模型）得到候选的 click, atc, pay 概率。

**点评**（逐条核对）：

| 读者表述 | 评判 | 说明 |
|---|---|---|
| Qwen3-VL based decoder | ❌ 关键误解 | Qwen3-VL 用在 **Tokenizer 阶段（离线）**，不是 decoder 的一部分。Decoder 是 UniSGR 自研的 Sparse MoE Decoder（Transformer + GQA + SwiGLU-MoE）。Qwen3-VL 负责"建字典"（商品图文 → embedding → RQ-VAE 量化成 (c1,c2,c3)），decoder 负责"用字典生成" |
| 产生多组候选语义 ID (c1,c2,c3) | ✅ 正确 | Beam Search 生成约 1024 组 (c1,c2,c3) 三元组 |
| 每个拼接上 decoder hidden states 作为特征 | ⚠️ 部分正确 | DRS 机制确实复用 hidden states；但"拼接"细节论文未明示（可能是 h_c3 单用或 [h_c1;h_c2;h_c3] 多层拼接）。排序模块还额外复用了语义 ID 查表表示 + 用户信息 |
| 作为 target 参与 Cross-Attention，KV 是用户序列特征 | ✅ 正确 | 这就是 Target Attention（DIN 风格）：Query = 候选语义 ID 表示，Key/Value = MemoryNet 输出 M |
| 通过 PLE（MOE 模型）得到 click/atc/pay 概率 | ⚠️ 部分正确 | PLE 确实输出三个概率，但需注意 **PLE 的"MoE" ≠ decoder 里的 SwiGLU-MoE**。PLE 核心是 CGC（Customized Gate Control）结构，每任务有专属专家+共享专家；SwiGLU-MoE 是 LLM 风格的稀疏 token 路由。两者是独立设计，不要混淆 |

**修正后的准确流程**：
MemoryNet 编码用户行为 → **Sparse MoE Decoder**（非 Qwen3-VL）通过 Beam Search + STARK 生成多组 (c1,c2,c3) 候选语义 ID → 排序模块以候选语义 ID 为 Query、M 为 KV 做 Target Attention → 拼接 [用户信息, 语义 ID 表示, Target Attention 输出, decoder hidden states] 输入 PLE（CGC 结构的多任务模型，非 LLM MoE）→ 输出 click/atc/pay 三概率，加权得最终分数返回 Top-K。

### 2.1 整体框架（非对称 Encoder-Decoder）

- **轻量特征编码器（MemoryNet）**：线性复杂度 O(L)
  - 用户画像经 MLP 映射后作为前缀 Token 拼接到行为序列
  - 历史商品用语义 ID 嵌入 + 行为类型 + 位置编码表示
  - 不使用 Self-Attention，只通过线性投影生成静态记忆表示 M，避免长用户历史上的二次方注意力开销

- **稀疏 MoE 解码器（Sparse MoE Decoder）**：
  - 自回归生成语义 ID Token，以可学习的 BOS Token 初始化，通过 Cross-Attention 条件化于 M
  - **Token Type Embedding (TTE)**：为每层语义 ID 添加层级结构嵌入，区分上层粗粒度聚类与下层细粒度区分
  - **Grouped Query Attention (GQA)**：降低 KV Cache 开销
  - **SwiGLU-MoE 层**：共享专家捕捉通用推荐模式，路由专家专门负责不同语义子空间和用户兴趣模式，每个 Token 仅激活约 7% 的总参数
  - 采用 RMSNorm 和 QK-Norm 稳定训练

### 2.2 两阶段训练范式

#### 阶段一：多场景预训练（Multi-Scenario Pre-training）

- 混合多个业务场景的用户行为日志，按时间序列构造下一商品生成样本
- 采用标准 **Next Token Prediction (NTP)** 损失：

$$\mathcal{L}_{\text{NTP}} = -\sum_{j=1}^{d} \log P_\theta(c_{v_{L+1}, j} \mid \mathbf{x}_u, \mathcal{S}_u, c_{v_{L+1}, <j})$$

- 该阶段为稀疏目标场景提供更好的覆盖度，减轻过拟合风险

#### 阶段二：场景特定对齐（Scenario-Specific Alignment）

##### (1) Value-Aware 并行多 Token 预测（VA-PMTP）

传统 NTP 只预测一个目标商品的语义 ID，但同一用户会话中存在多个并发兴趣（点击、加购、购买）。VA-PMTP 通过**自定义并行掩码**同时预测多个自回归语义 ID 目标：

- **同一商品内**：各层语义 ID 之间应用因果掩码（从上到下可见）
- **不同商品间**：互相不可见（互不干扰）
- **共享 BOS Token**：所有语义 ID 共享 BOS，统一多兴趣表示

并按业务价值对不同行为加权（购买 > 加购 > 点击 > 曝光）：

$$\mathcal{L}_{\text{VA-PMTP}} = -\sum_{\tau} w_{\tau}\sum_{i=1}^{\mathcal{S}_{\tau}} \sum_{j=1}^{d} \log P_\theta(c_{v_{L+i}, j} \mid \mathbf{x}_u, \mathcal{S}_u, c_{v_{L+i}, <j})$$

##### (2) 统一多目标排序模块

在生成式解码器上附加语义 ID 基的多目标排序模块，**三路共享表示**：
- **语义 ID 表示共享**：拼接多层语义 ID 作为排序模块的候选输入
- **编码器表示共享**：共享用户行为序列的编码器表示
- **解码器表示共享（DRS）**：解码器各层自回归过程中的隐藏状态被排序模块复用

排序模块架构：
- **Target Attention**：用候选语义 ID 对行为序列编码器表示做 Cross-Attention，精细建模用户兴趣
- **PLE 多目标排序**：将用户信息、语义 ID 表示、Target Attention 输出、解码器表示输入 PLE 模块，同时预测点击/加购/购买

##### (3) Task-Aware Tokens（TAT）

纯生成目标不显式编码不同业务目标所需的偏好信号，因此在解码器输入前**预置三个可学习的任务 Token**（$\mathbf{e}_{\text{click}}, \mathbf{e}_{\text{atc}}, \mathbf{e}_{\text{pay}}$）：

$$\mathbf{X}_{\text{dec}} = [\underbrace{\mathbf{e}_{\text{click}},\ \mathbf{e}_{\text{atc}},\ \mathbf{e}_{\text{pay}}}_{\text{task tokens}},\ \underbrace{\langle \text{bos} \rangle,\ s_1,\ s_2, \ldots}_{\text{sequence tokens}}]$$

**因果掩码遵循转化漏斗顺序**：click → atc → pay，所有语义 ID Token 可见前面的任务 Token。由于任务 Token 占固定前缀位置，**推理时不增加额外自回归步数**。

##### (4) Funnel-Aware Contrastive Learning (FACL)

任务 Token 主要通过注意力影响下游表示，容易出现弱监督和语义漂移，因此添加辅助对比学习：

| 任务 Token | 正样本 | 负样本（难度递增）|
|---|---|---|
| **Click** | 被点击的商品 | 随机商品 |
| **Add-to-cart** | 被加购的商品 | 被点击但未加购 |
| **Purchase** | 被购买的商品 | 被点击但未购买 |

$$\mathcal{L}_{\tau}^{\text{aux}} = -\log \frac{\exp(\text{sim}(\mathbf{h}_{\tau},\ \mathbf{v}^+) / t)}{\exp(\text{sim}(\mathbf{h}_{\tau},\ \mathbf{v}^+) / t) + \sum_{j=1}^{K} \exp(\text{sim}(\mathbf{h}_{\tau},\ \mathbf{v}_j^-) / t)}$$

##### (5) 联合训练目标

$$\mathcal{L} = \underbrace{\mathcal{L}_{\text{gen}} + \alpha \cdot \mathcal{L}_{\text{rank}}}_{\text{主任务}} + \underbrace{\mathcal{L}_{\text{aux}}}_{\text{辅助任务}}$$

排序损失梯度通过共享解码器参数反向传播，强化序列 Token 位置的多目标感知表示。

### 2.3 STARK：语义树注意力 + 重组 KV Cache

传统 Beam Search 在 batch 维度扩展 Top-K 候选，存在三大效率瓶颈：
1. **KV Cache 重复复制与重排**导致巨大内存带宽开销
2. **共享前缀的冗余注意力计算**
3. 语义 ID 极短，主流长序列优化的注意力核因 padding 过多**利用率低下**

STARK 的核心思想：**用序列维度扩展替代 batch 维度扩展**：

- 每步解码时，将候选 Token 沿序列维度拼接，而非复制为 K 个独立 batch 样本
- 通过预计算的**树注意力掩码**控制可见性：每个 Token 只能关注解码树中的祖先节点和自身
- KV Cache 按树拓扑组织：共享前缀的 K/V 只存一份，后代节点通过预计算索引映射引用

形式化：第 l 层候选路径 $P^{(l)}=\{p_1^{(l)},\ldots,p_k^{(l)}\}$，树掩码 $\mathbf{M}^{(l)}$：

$$\mathbf{M}^{(l)}_{i,j} = \begin{cases} 1, & j \in \mathrm{Anc}(p_i^{(l)}) \text{ 或 } j=i \\ 0, & \text{otherwise} \end{cases}$$

STARK 在工业场景下实现 **200% 吞吐提升**：

| 方法 | Batch Size | QPS | AVG Lat(ms) | P99 Lat(ms) |
|---|---|---|---|---|
| w/o STARK | 1 | 119 | 30.9 | 33.5 |
| STARK | 1 | 219 | 14.1 | 15.6 |
| STARK | 8 | 596 | 26.1 | 26.8 |

### 2.4 语义 ID Tokenizer

1. **协作感知多模态表示**：用 Qwen3-VL 微调，输入文本元数据 T_i + 视觉特征 V_i，用 InfoNCE 损失训练（正样本：交互目标商品；难负样本：曝光未点击；易负样本：同 batch）
2. **量化**：RQ-VAE + Sinkhorn-Knopp 均衡分配，生成 3 层语义码本（codebook size K=8192）

---

## 3. 实验结果

### 3.1 离线实验（Lazada 首页 "Guess You Like" 场景）

#### 与 SOTA 对比（多场景预训练阶段）

| 模型 | HR@50 | HR@100 | HR@200 | HR@500 |
|---|---|---|---|---|
| TIGER | 0.1445 | 0.2026 | 0.2698 | 0.3675 |
| OneRec-V2 | 0.1529 | 0.2126 | 0.2816 | 0.3812 |
| OneRec | 0.1538 | 0.2151 | 0.2855 | 0.3866 |
| **UniSGR-M (Ours)** | **0.1579** | **0.2195** | **0.2896** | **0.3913** |

#### 两阶段训练范式验证

| 模型 | Clk-HR@100 | Atc-HR@100 | Pay-HR@100 | Clk-HR@500 | Atc-HR@500 | Pay-HR@500 |
|---|---|---|---|---|---|---|
| 仅预训练 | 0.1886 | 0.2276 | 0.2783 | 0.3893 | 0.4288 | 0.4768 |
| 仅场景对齐 | 0.2297 | 0.2397 | 0.2573 | 0.4334 | 0.4374 | 0.4386 |
| **两阶段联合** | **0.2842** | **0.3073** | **0.3408** | **0.5166** | **0.5369** | **0.5632** |

#### 排序模块消融（对齐阶段 HR）

| 模型 | Clk-HR@100 | Atc-HR@100 | Pay-HR@100 |
|---|---|---|---|
| NTP | 0.2752 | 0.2841 | 0.3117 |
| VA-PMTP | 0.2777 | 0.3006 | 0.3382 |
| VA-PMTP + Ranking | 0.2903 | 0.3076 | 0.3621 |
| VA-PMTP + Ranking + TAT | 0.2924 | 0.3141 | 0.3614 |
| **Full UniSGR** | **0.2932** | **0.3176** | **0.3636** |

#### 排序模块消融（对齐阶段 AUC/GAUC）

| 模型 | Click GAUC | Click AUC | ATC GAUC | ATC AUC | Pay GAUC | Pay AUC |
|---|---|---|---|---|---|---|
| VA-PMTP | 0.5513 | 0.5684 | 0.5245 | 0.5360 | 0.5340 | 0.5603 |
| + Ranking | 0.5624 | 0.6218 | 0.5710 | 0.6780 | 0.5767 | 0.7662 |
| + Ranking + TAT | 0.5626 | 0.6219 | 0.5697 | 0.6855 | 0.5787 | 0.7656 |
| **Full UniSGR** | **0.5744** | **0.6334** | **0.5901** | **0.6978** | **0.6153** | **0.7870** |

### 3.2 码本大小消融

| 配置 | HR@100 | 冲突率 |
|---|---|---|
| 3L2048 | 0.2917 | 45.00% |
| 3L4096 | 0.3126 | 31.00% |
| **3L8192** | **0.3266** | 24.03% |
| 3L10240 | 0.3264 | 23.58% |

K 从 2048 → 8192 大幅提升（冲突率显著下降），继续增大到 10240 收益递减，因此采用 **3L8192**。

### 3.3 模型组件消融

| 模型 | HR@100 | 说明 |
|---|---|---|
| UniSGR-M | **0.2195** | 完整模型 |
| w/o SwiGLU | 0.2119 | 非线性表达能力下降 |
| w/o MoE | 0.1725 | 性能大幅下降（最关键组件）|
| w/o Shared Expert | 0.2180 | 共享专家捕捉通用模式，重要 |

**MoE 是对性能影响最大的模块**，验证了稀疏大模型在捕捉多样化用户兴趣中的核心作用。

### 3.4 Scaling Law

| 模型 | 参数 | HR@100 | HR@500 |
|---|---|---|---|
| UniSGR-XS | 0.2B | 0.1911 | 0.3489 |
| UniSGR-S | 0.4B | 0.2058 | 0.3705 |
| UniSGR-M | 0.8B | 0.2195 | 0.3913 |
| UniSGR-L | 1.2B | 0.2266 | 0.4018 |
| UniSGR-XL | 1.6B | 0.2298 | 0.4062 |
| UniSGR-XXL | 2.0B | 0.2311 | 0.4097 |

随模型规模扩大，HR 稳定提升，呈现清晰的 Scaling 行为；但边际收益逐步递减（小-中等规模区间收益最大）。

### 3.5 线上 A/B 测试（Lazada 首页推荐）

对比工业级生产级联推荐系统：

| 指标 | 提升 |
|---|---|
| **IPV（商品页浏览）** | **+3.36%** |
| **交易笔数** | **+2.17%** |
| **GMV** | **+5.68%** |

---

## 4. 贡献总结

1. **统一框架**：首次无缝集成语义 ID 生成与多目标排序，缓解传统级联架构固有的目标不匹配 gap
2. **两阶段训练**：多场景预训练 + 场景特定对齐，其中 VA-PMTP 提升高价值行为候选质量，TAT 注入任务感知信号
3. **FACL 辅助监督**：漏斗感知对比学习指导任务 Token，防止语义漂移
4. **STARK 高效推理**：树注意力 + 重组 KV Cache，消除冗余前缀计算，吞吐提升 200%
5. **大规模验证**：离线实验 + 线上 A/B 测试均取得显著收益，证明工业可扩展性

---

## 5. 与现有工作的关系定位

| 方向 | 代表性工作 | UniSGR 的差异化 |
|---|---|---|
| 语义 ID 生成召回 | TIGER, OneRec, OneRec-V2 | 额外耦合多目标排序模块，共享表示 |
| 统一点排 | OneRanker, GRank, GPR | 基于语义 ID 的生成解码器顶部直接挂排序头，三路表示共享 |
| 生成式推荐 Scaling | MTGR | 额外提出 STARK 推理加速，适配语义 ID 短序列场景 |
| 多任务排序 | PLE 等 | 与生成式解码器共享表示，排序梯度直接反哺生成质量 |

---

## 6. 讨论记录

### Q1：用一个例子从推理角度说明全流程，召回如何进行？排序如何与召回交互？

**以 Alice 在 Lazada 首页推荐为例**：行为序列 [手机壳(view), 蓝牙耳机(click), 充电线(atc), 手机支架(click)]，目标返回 Top-K。

#### Step 1：输入构造
- 用户画像 x_u（年龄/性别/地域/购买力）+ 行为序列 S_u（每个商品用 `语义ID嵌入 + 行为类型embedding + 位置编码`）+ 场景上下文

#### Step 2：编码器前向（MemoryNet）
- 用户画像经 MLP → 前缀 Token，拼到行为序列前
- MemoryNet 不用 Self-Attention，只做线性投影 → 输出静态记忆表示 M（d=128, O(L) 复杂度）
- 例如 Alice 的 M 编码了"近期对数码配件感兴趣、有 atc 行为表示购买意向"

#### Step 3：解码器自回归生成（Beam Search + STARK）

**解码器输入**：`[e_click, e_atc, e_pay, <BOS>, ...]`
- 前 3 个是预置 Task-Aware Tokens（可学习 embedding，不参与自回归），通过统一因果掩码让后续所有语义 ID 都能 attend → 从第一步注入 click/atc/pay 任务信号
- `<BOS>` 后开始逐层生成

| 层级 | Beam Width | 操作 | 示例 |
|---|---|---|---|
| 第 1 层（粗粒度聚类）| 512 | 在 `<BOS>` 输出 8192 维分布，取 top-512 个 c1 | c1=1234 → "数码配件"类簇 |
| 第 2 层（中粒度）| 512 | 对每个 c1 条件化生成 c2，保留 top-512 路径 | (1234, 0042) → "数码配件 → 无线耳机" |
| 第 3 层（细粒度，最宽）| 1024 | 对每条 (c1,c2) 生成 c3，保留 top-1024 完整序列 | (1234, 0042, 0901) → "某品牌真无线耳机 SKU" |

**STARK 关键作用**：传统 Beam Search 在第 2 层要把 512 个 c1 复制成 512 个 batch 样本（KV Cache 翻倍）；STARK 沿**序列维度**拼接所有候选 c2，用树掩码让每个 c2 只能 attend 自己的祖先 c1 → **共享前缀 KV Cache 只存一份**，吞吐提升 200%。

最终产出约 1024 个候选商品（受 24% 冲突率影响可能略少）。

#### Step 4：排序模块打分（**排序与召回的核心交互点**）

排序模块**不重新编码候选**，直接复用生成阶段产物：

**(a) 三路表示共享**：
- 语义 ID 表示共享：候选 (c1, c2, c3) 拼接作为排序模块输入
- 解码器表示共享（DRS）：复用生成过程中每层 decoder 的 hidden states（已包含 TAT 注入的任务信号）
- 编码器表示共享：复用 M

**(b) Target Attention**：用候选语义 ID 作 query，对 M 做交叉注意力。例如候选"真无线耳机" attend 到 Alice 的 [蓝牙耳机(click), 手机壳(view)]，捕获精细兴趣匹配。

**(c) PLE 多目标预测**：输入 = 用户信息 + 语义 ID 表示 + Target Attention 输出 + 解码器表示，同时输出 P(click), P(atc), P(pay)。例如对"真无线耳机"：P(click)=0.82, P(atc)=0.45, P(pay)=0.18。

**(d) 多目标加权排序**：
`score = w_click·P(click) + w_atc·P(atc) + w_pay·P(pay)`，权重按业务价值 pay > atc > click
例如 score = 0.2×0.82 + 0.3×0.45 + 0.5×0.18 = 0.454

#### Step 5：返回 Top-K
- 对 1024 个候选按加权分数排序，返回 Top-50/100，**无需独立 Item ID 排序模型**

#### 排序与召回的三个交互点（核心结论）

1. **生成阶段就注入任务信号（TAT）**：传统是"先召回再排序"，UniSGR 在生成阶段就通过 Task-Aware Tokens 把 click/atc/pay 信号注入，使生成的候选本身已偏向高价值行为
2. **表示共享（DRS）**：排序模块直接复用解码器隐藏状态，不重新编码；训练时排序梯度通过共享参数反哺生成器，强化表示的多目标感知
3. **联合打分无需独立排序模型**：复用生成器表示 + 轻量 PLE 头完成多目标打分，真正实现"生成-排序统一框架"

---

### Q2：候选语义 ID 与 (c1,c2,c3) 的关系？排序模块中是作为 Query 吗？

#### 概念澄清：「候选语义 ID」=「(c1, c2, c3) 三元组」

关系链：

```
商品 SKU
  → Qwen3-VL 提取协作感知 embedding
  → RQ-VAE + Sinkhorn-Knopp 量化
  → 固定的 3 层语义 ID (c1, c2, c3)   ← 离线一次性建好
```

反过来，Beam Search 生成的 (c1, c2, c3) 查表映射回商品 SKU（受 24% 冲突率影响可能多对一）。三层语义层级：

| 层 | 粒度 | 示例 |
|---|---|---|
| c1（上层）| 粗粒度聚类 | "数码配件" 大类 |
| c2（中层）| 中粒度子类 | "无线耳机" 子类 |
| c3（下层）| 细粒度唯一性 | 具体某品牌 SKU |

#### 「作为 Query」只说对了一半，分两个阶段看

**(a) Target Attention 阶段：候选语义 ID 作为 Query ✅**

沿用 DIN（Deep Interest Network）的 Target Attention 思路：
- **Query** = 候选语义 ID 表示（c1, c2, c3 各自从 codebook embedding table 查表 + Token Type Embedding 区分层级，拼接而成）
- **Key / Value** = MemoryNet 输出的用户行为序列表示 M
- **输出** = 每个候选关于用户兴趣的加权表示（捕获"候选 vs 用户历史"的精细匹配）
- 例如候选"真无线耳机" attend 到 Alice 的 [蓝牙耳机(click), 手机壳(view)]，蓝牙耳机权重最高

**(b) PLE 多目标预测阶段：候选语义 ID 是输入特征，而非 Query ❌**

- 这一步**不是 Attention**，而是特征拼接 + 专家路由
- 输入特征 = `[用户信息, 语义 ID 表示, Target Attention 输出, 解码器表示]`
- PLE 模块（多门控混合专家）对拼接特征做非线性变换，输出 P(click), P(atc), P(pay)

**结论**：「作为 Query」只适用于 Target Attention 这一步；到 PLE 阶段，候选表示退化为输入特征向量参与专家路由。

---

### Q3：「生成过程中每层 decoder 的 hidden states」是怎么被复用的？

#### decoder 在生成时输出了什么

对每个候选 (c1, c2, c3)，decoder 自回归生成过程中会输出一系列 hidden states：

| 位置 | Token | 输出 hidden state | 包含的信息 |
|---|---|---|---|
| 0 | e_click | h_click | click 任务偏好（FACL 直接监督过）|
| 1 | e_atc | h_atc | atc 任务偏好 |
| 2 | e_pay | h_pay | pay 任务偏好 |
| 3 | `<BOS>` | h_bos | 三任务信号融合后的初始状态 |
| 4 | c1 | h_c1 | 第一层语义聚类 + 用户上下文 |
| 5 | c2 | h_c2 | 第二层语义 + 前序累积 |
| 6 | c3 | h_c3 | 第三层语义 + 完整上下文（最丰富）|

这些 hidden states 是**逐步累积上下文**的——每一层都通过 self-attention 融合前序、通过 cross-attention 融合 M（用户行为）。

#### 为什么比单纯查 codebook embedding 强

- **静态查表**只能拿到 (c1, c2, c3) 各自的固定 embedding（商品本身的语义）
- **decoder 的 hidden state** 还融合了：
  - 用户画像和行为序列（通过 cross-attention 到 M）
  - 任务信号（通过 TAT 注入）
  - 语义层级间关系（c1 → c2 → c3 的累积影响）
  - 候选间相对位置（通过 STARK 树掩码）

#### 复用方式（论文原文："decoder outputs at <bos>, s1, s2, ... serve as task-aware features"）

把这些 hidden states 作为**任务感知特征**输入 PLE，分三类用途：

**1. 候选表示特征**
- 取 `h_c3`（最后位置）作为候选的"最终上下文感知表示"——包含最完整的累积信息
- 或拼接 `[h_c1; h_c2; h_c3]` 保留层级语义（更丰富但维度更高）

**2. 任务特定特征**
- `h_click`, `h_atc`, `h_pay` 这三个任务 Token 的输出，分别作为对应目标的偏好表示
- 这些是 FACL 对比学习直接监督过的表示，质量较高
- 可分别作为 PLE 各任务专家的专属输入

**3. 门控输入**
- `h_bos` 综合了三个任务 Token 的信息，可作为 PLE 门控网络的输入，决定路由到哪些专家
