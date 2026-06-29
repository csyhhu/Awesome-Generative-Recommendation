# UniFormer: Efficient and Unified Model-Centric Scaling for Industrial Recommendation

- **论文链接**: https://arxiv.org/abs/2606.27058
- **发表会议**: KDD 2025
- **作者机构**: 快手科技（Kuaishou Technology）
- **作者**: Bo Chen*, Jinlong Jiao*, Tijian Hu*, Ruihao Zhang, Yanzhi Liu, Chenghou Jin, Qinglin Jia, Baixuan He, Hechang Pan, Yiwu Liu, Jian Liang, Chaoyi Ma†, Ruiming Tang†, Han Li, Kun Gai

---

## 1. 研究动机

工业推荐系统通常由多个独立设计的模块组成：**行为建模模块（Behavior Modeling）**、**特征交互模块（Feature Interaction）** 和 **多任务建模模块（Task Modeling）**。近期，受到 LLM 成功的启发，推荐社区开始尝试通过参数扩展（Scaling）来提升模型能力，但这些工作大多停留在**组件级扩展（Component-Centric Scaling）**：

- **行为序列建模**：LONGER、STCA 等通过 Transformer 扩展长序列建模
- **特征交互**：RankMixer、HHFT 等通过 Token-Mixing/Self-Attention 扩展特征交互
- **多任务建模**：SMES 等通过 MoE 扩展多任务学习

这种碎片化的扩展方式导致以下问题：
1. 各模块独立演进，无法实现跨模块的联合收益
2. OneTrans、HyFormer 等虽探索了特征空间内的跨模块联合扩展，但仍忽略了任务空间
3. 缺乏一个统一的高效模型级扩展框架

## 2. 核心贡献

1. **范式转变**：提出从组件级扩展（Component-Centric Scaling）向**模型级扩展（Model-Centric Scaling）**的范式转变，并系统分析了构建统一扩展框架的关键挑战
2. **提出 UniFormer 框架**：一个高效的统一模型级扩展框架，将整体建模空间分解为特征空间和任务空间，分别通过 FIM 和 TIM 进行建模
3. **大规模实验验证**：在快手两大生产场景（快手主端和快手极速版）上进行了离线实验和在线 A/B 测试，取得一致且显著的提升

## 3. 方法

### 3.1 整体架构

UniFormer 由两大核心组件组成：

- **Tokenization 模块**：将序列特征（Sequential Features）、非序列特征（Non-Sequential Features）和任务特征（Task Features）转化为紧凑的 Token 表示
- **统一交互模块（Unified Interaction Block）**：由 $M$ 层 Feature-space Interaction Modules (FIMs) 和 $N$ 层 Task-space Interaction Modules (TIMs) 堆叠而成，每个模块均采用标准的 Transformer 式架构（Attention + FFN）

整体数据流：
```
输入特征 → Tokenization → FIM(×M层) → TIM(×N层) → 多任务输出
```
关键设计动机：为避免类似 OneTrans 在全空间做 Full Attention 的巨大计算开销，UniFormer 将整体建模空间**解耦为特征空间和任务空间**，分别由 FIM 和 TIM 建模。

---

### 3.2 Tokenization 设计

Tokenization 的核心设计原则是**语义分组（Semantic-based Tokenization）**。与 HyFormer 采用的全局分组（Global Tokenization）将所有特征打平为统一的 Token 序列不同，语义分组按特征语义和与候选 Item 的依赖关系进行分组，带来两个关键收益：

1. **推理加速**：将计算分解为 User 侧（请求级可复用）和 Item 侧（候选级独立计算），支持 User-Item 解耦
2. **多视图信息提取**：语义分组的 Query 提供多视角的注意力模式，从行为序列中提取多样化用户偏好

下面分别介绍三类特征的具体 Tokenization 方式。

---

#### 3.2.1 序列特征（Sequential Features）的 Tokenization

序列特征按**是否依赖目标 Item** 分为两类，采用不同的 Tokenization 策略：

**A. Item-Independent 行为序列**（不依赖目标 Item）

包括：**短期交互序列**（short-term interaction sequences）和 **长期压缩兴趣表示**（long-term compressed interest representations，如 C-Former）。

- 处理方式：**保留原始序列结构**，逐元素（per-element）进行 Token 化
- **短期序列 Token 构建**：第 $i$ 个元素的 Token 通过拼接 Item 的各类附属信息（side information）形成：

$$\mathbf{e}^{\mathrm{short}}_{i} = [\mathbf{e}^{\mathrm{pid}}_{i} \| \mathbf{e}^{\mathrm{tag}}_{i} \| \mathbf{e}^{\mathrm{paid}}_{i} \| \mathbf{e}^{\mathrm{duration}}_{i} \| \dots]$$

其中各分量含义：
  - $\mathbf{e}^{\mathrm{pid}}_{i}$：视频 ID Embedding
  - $\mathbf{e}^{\mathrm{tag}}_{i}$：内容类型 Embedding
  - $\mathbf{e}^{\mathrm{paid}}_{i}$：创作者信息 Embedding
  - $\mathbf{e}^{\mathrm{duration}}_{i}$：观看时长 Embedding

- **长期序列 Token 构建**：直接保留预训练的压缩 Embedding，不做进一步修改：

$$\mathbf{e}^{\mathrm{long}}_{i} = \mathbf{e}_{i}^{\mathrm{pretrained}}$$

**Token 投影与 KV 变换**（参考 OneRec-v2 设计）：

每个 Token 表示 $\mathbf{e}_i$（$\mathbf{e}^{\mathrm{short}}_{i}$ 或 $\mathbf{e}^{\mathrm{long}}_{i}$）先通过 **SwiGLU 层**投影到统一隐空间，得到 $\mathbf{z}_{i} \in \mathbb{R}^{d_{\mathrm{seq}}}$，其中维度定义为：

$$d_{\mathrm{seq}} = S_{\mathrm{kv}} \cdot L_{\mathrm{kv}} \cdot d_{\mathrm{model}}$$

参数解释：
- $S_{\mathrm{kv}}$：KV 分裂系数（key-value split coefficient），默认为 1
- $L_{\mathrm{kv}}$：KV 层数，默认 1
- $d_{\mathrm{model}} = G_{\mathrm{kv}} \cdot d_{\mathrm{head}}$，其中 $G_{\mathrm{kv}}$ 为 KV 头数，$d_{\mathrm{head}}$ 为注意力头维度

随后 $\mathbf{z}_i$ 被进一步变换为**层特定的 KV 对**：$[\mathbf{C}^{0}_i,\mathbf{C}^{1}_i,\dots,\mathbf{C}_i^{S_{\mathrm{kv}} \cdot L_{\mathrm{kv}}-1}]$，其中 $\mathbf{C}_i^{j} \in \mathbb{R}^{d_{\mathrm{model}}}$。

对于第 $l$ 层，经过 RMSNorm 归一化得到 KV 对：

$$\mathbf{k}_i^{(l)} = \mathrm{RMSNorm}_{k,l}(\mathbf{C}_i^{S_{\mathrm{kv}} \cdot l})$$

$$\mathbf{v}_i^{(l)} = \begin{cases} \mathbf{k}_i^{(l)}, & \text{if } S_{\mathrm{kv}} = 1 \text{ (shared key-value)} \\[4pt] \mathrm{RMSNorm}_{v,l}\big( \mathbf{C}_i^{S_{\mathrm{kv}} \cdot l +1} \big), & \text{if } S_{\mathrm{kv}} = 2 \text{ (separated key-value)} \end{cases}$$

最终短序列和长序列的输出分别为 $(\mathbf{K}_\mathrm{short}, \mathbf{V}_\mathrm{short})$ 和 $(\mathbf{K}_\mathrm{long}, \mathbf{V}_\mathrm{long})$。

> **Tokenization 的轻量性**：序列特征 Tokenization 的完整流程为 **Embedding 拼接 → SwiGLU（单层）→ KV 变换 → RMSNorm**，整个阶段**只有 SwiGLU 一层是可训练的核心变换**，没有多层深度网络。Tokenization 保持轻量，将复杂的交互建模留给后续的 FIM/TIM 模块。

**B. Item-Dependent 行为序列**（依赖目标 Item）

典型代表：**搜索式序列**（如 SIM），其序列中的 Item 根据与目标 Item 的相似度检索得到，因此与目标 Item 强相关。

- 处理方式：**聚合而非保留原始序列结构**
- 关键动机：使交叉注意力中的 KV 表示与目标 Item **解耦**，从而实现请求级推理加速（详见 3.5 节推理优化）
- 具体操作：使用 **Target Attention Tokenizer（TA）** 将整条搜索序列聚合为一个紧凑 Token：

$$\mathbf{e}_ {\mathrm{search}} = \text{TA}(\mathbf{e}_{\mathrm{target}}, \mathbf{K}_{\mathrm{search}}, \mathbf{V}_{\mathrm{search}})$$

其中 $(\mathbf{K}_{\mathrm{search}}, \mathbf{V}_{\mathrm{search}})$ 为搜索序列的 KV 对，计算方式与前述 Item-Independent 序列相同（式 2）。

> **关键设计意图**：聚合后 $\mathbf{e}_{\mathrm{search}}$ 只包含"用户对目标 Item 附近行为的聚合偏好"，不再携带原始序列中各 Item 的细粒度信息，从而使得该 Token 对所有候选 Item 一致，可复用。在后续处理中，该聚合 Token 被归类为 **Item-Independent 特征组**的一部分。

---

#### 3.2.2 非序列特征（Non-Sequential Features）的 Tokenization

非序列特征也按**依赖关系**分为两类，并在每类内部进一步按**语义含义**分组：

| 类别 | 典型特征 | 分组数 |
|------|---------|:-----:|
| **Item-Independent** | 用户 ID、用户画像特征、上下文请求特征（如时间、网络环境）| $m$ 组 |
| **Item-Dependent** | Item ID、Item 统计特征、User-Item 交叉特征 | $n$ 组 |

> 注意：经 Target Attention 聚合后的 Item-Dependent 行为序列（如 SIM）的 Token $\mathbf{e}_{\mathrm{search}}$ 也被归入 **Item-Independent** 特征组。

对于每个特征组，处理流程为：

1. **拼接**：将组内所有特征 Embedding 拼接
2. **SwiGLU 投影**：通过 SwiGLU 层投影到统一 $d_{\mathrm{model}}$ 维隐空间

最终得到非序列特征 Token 矩阵：

$$\mathbf{N_{\mathrm{NS}}} \in \mathbb{R}^{q \times d_{\mathrm{model}}}, \quad q = m + n$$

其中 $m$ 个 Item-Independent Token 每个请求只需计算一次，$n$ 个 Item-Dependent Token 需对每个候选 Item 分别计算（这是 User-Item 解耦推理的基础）。

**语义分组 vs 全局分组的优势**：

| 对比维度 | 全局分组（HyFormer） | 语义分组（UniFormer） |
|---------|---------------------|---------------------|
| 推理效率 | 所有 Token 需逐候选重算 | User 侧 Token 跨候选复用 |
| 注意力多样性 | Token 间缺乏语义区分 | 语义 Query 提供多视图信息 |
| 跨层行为 | 注意力趋于同质化 | 不同层关注不同行为模式 |

---

#### 3.2.3 任务特征（Task Features）的 Tokenization

对于每个任务（如 Effective-view、Long-view、Like、Follow），将其任务特定特征分组为一个独立 Token：

- **组成**：Task ID Embedding、任务相关偏置特征（task-related bias features）
- **处理**：拼接组内所有特征 Embedding → SwiGLU 投影到 $d_{\mathrm{model}}$ 维

得到任务特征 Token 矩阵：

$$\mathbf{N_{\mathrm{T}}} \in \mathbb{R}^{t \times d_{\mathrm{model}}}$$

其中 $t$ 为任务数量（快手场景下 $t=21$）。

> 每个任务的 Token 承载了该任务的"身份信息"和"任务特定的先验偏置"，将在 TIM 中作为 Query 从 FIM 的高阶特征表示中检索与自身优化目标相关的信息。

---

### 3.3 Feature-space Interaction Module (FIM) — 特征空间交互

FIM 由 $M$ 层堆叠而成（默认 $M=3$），每层包含两个互补的交互子组件：**序列导向交互（Sequential-oriented Interaction）** 和 **非序列导向交互（Non-Sequential-oriented Interaction）**。

---

#### 3.3.1 序列导向交互（Sequential-oriented Interaction）

**设计动机**：传统方法将异构行为序列（短/长/跨域）拼接后做共享交叉注意力，容易导致**偏好坍塌（Preference Collapse）**——某一种行为模式主导用户表示，丢失其他行为视角的信息。

**方案：多序列交叉注意力（Multi-Sequence Cross-Attention）**

短序列和长序列**分别独立**进行交叉注意力计算，以短序列为例，第 $l$ 层的操作为：

$$\mathbf{H}_{\mathrm{short}}^{\mathrm{feat,}(l)}
= \mathrm{CA}\Big(
\mathrm{RMSNorm}(\mathbf{Q}^{\mathrm{feat,}(l-1)}_{\mathrm{cross}}),\;
\mathbf{K}_{\mathrm{short}}^{(l)},\;
\mathbf{V}_{\mathrm{short}}^{(l)}
\Big) + \mathbf{Q}^{\mathrm{feat,}(l-1)}_{\mathrm{cross}}$$

其中：
- **Query** $\mathbf{Q}^{\mathrm{feat,}(l-1)}_{\mathrm{cross}} \in \mathbb{R}^{q \times d_{\mathrm{model}}}$：来自上一层的输出（首层使用非序列 Token $\mathbf{N_{\mathrm{NS}}}$ 作为 Query），共 $q$ 个语义 Query 同时从行为序列中提取多视图信息
- **KV** $(\mathbf{K}_{\mathrm{short}}^{(l)}, \mathbf{V}_{\mathrm{short}}^{(l)})$：短序列的层特定 KV 表示
- **Pre-Norm RMSNorm**：用于稳定训练
- **输出维度**：$\mathbf{H}_{\mathrm{short}}^{\mathrm{feat,}(l)} \in \mathbb{R}^{q \times d_{\mathrm{model}}}$，**输出 Token 数恒为 $q$（非序列 Token 数）**，与序列长度无关。序列 Token 仅作为 KV 在 softmax 中被加权求和，不产生独立输出 Token

长序列处理方式完全对称，得到 $\widetilde{\mathbf{H}}_{\mathrm{long}}^{\mathrm{feat,}(l)}$。

**Lazy KV 设计**（参考 OneRec-v2）：

默认 $S_{\mathrm{kv}}=1, L_{\mathrm{kv}}=1$，即所有层的交叉注意力**共享同一套 KV 表示**，无需逐层重新计算 KV，大幅降低内存占用和计算开销。具体来说：在 Tokenization 阶段，$S_{\mathrm{kv}}=1, L_{\mathrm{kv}}=1$ 时 $d_{\mathrm{seq}}=d_{\mathrm{model}}$，每个序列元素只产生一个 $\mathbf{C}_i^0$，所有 FIM 层共享这同一底层表示，仅通过不同参数的 RMSNorm 产生层特定的 K 和 V（如 $\mathbf{k}_i^{(l)} = \mathrm{RMSNorm}_{k,l}(\mathbf{C}_i^0)$）。这一设计的核心收益是：KV 只计算一次，不随层数线性增长，显著节省显存和计算。

**序列特定 FFN（S-FFNs）**：

交叉注意力输出再经过序列特定的 SwiGLU FFN 进行深度变换：

$$\widetilde{\mathbf{H}}_{\mathrm{short}}^{\mathrm{feat,}(l)}
= \mathrm{FFN}^{\mathrm{feat,}(l)}_{\mathrm{short}}\Big( \mathbf{H}_{\mathrm{short}}^{\mathrm{feat,}(l)} \Big) + \mathbf{H}_{\mathrm{short}}^{\mathrm{feat,}(l)}$$

长序列同理。使用独立的 S-FFNs 而非共享 FFN 的原因是：短序列、长序列、跨域序列本质上捕捉的是不同类型的用户行为信号，使用**序列特定的建模网络**可在专属隐空间中更好地刻画异构行为模式，实现更细粒度的用户兴趣建模。

**自适应融合策略**（两种方案）：

1. **全局自适应融合**（Global Adaptive Fusion）：
   学习一个全局系数 $\alpha \in [0,1]$，线性加权融合：

   $$\mathbf{H}^{\mathrm{feat,}(l)}_{\mathrm{cross}} = \alpha \widetilde{\mathbf{H}}_{\mathrm{short}}^{\mathrm{feat,}(l)} + (1-\alpha) \widetilde{\mathbf{H}}_{\mathrm{long}}^{\mathrm{feat,}(l)}$$

2. **个性化自适应融合**（Personalized Adaptive Fusion）：
   基于用户特征（如活跃度、交互行为等）动态预测用户特定的融合系数 $\alpha_i$：

   $$\mathbf{H}^{\mathrm{feat,}(l)}_{\mathrm{cross}} = \alpha_i \widetilde{\mathbf{H}}_{\mathrm{short}}^{\mathrm{feat,}(l)} + (1-\alpha_i) \widetilde{\mathbf{H}}_{\mathrm{long}}^{\mathrm{feat,}(l)}$$

   高活跃用户与冷启动用户的长短期行为模式差异巨大，个性化融合能更好适应这种差异。

---

#### 3.3.2 非序列导向交互（Non-Sequential-oriented Interaction）

**设计动机**：序列导向交互的输出 $\mathbf{H}^{\mathrm{feat,}(l)}_{\mathrm{cross}}$ 主要被序列信息主导，缺少非序列特征的高阶交互建模。因此需要引入交互增强（Interaction Enhancement）。

**步骤①：拼接增强**

将序列交互输出与上一层 Query 拼接，注入非序列信息：

$$\mathbf{X}^{\mathrm{feat,}(l)} = [\mathbf{H}^{\mathrm{feat,}(l)}_{\mathrm{cross}} \;\|\; \mathbf{Q}^{\mathrm{feat,}(l-1)}_{\mathrm{self}}]$$

其中首层 $\mathbf{Q}^{\mathrm{feat,}(0)}_{\mathrm{self}} = \mathbf{Q}^{\mathrm{feat,}(0)}_{\mathrm{cross}} = \mathbf{N_{\mathrm{NS}}}$，维度从 $q$ 扩展到 $2q$。

**步骤②：Self-Attention 交互**

$$\mathbf{H}^{\mathrm{feat,}(l)}_{\mathrm{self}} = \mathrm{SA}\Big( \mathrm{RMSNorm}(\mathbf{X}^{\mathrm{feat,}(l)}) \Big) + \mathbf{X}^{\mathrm{feat,}(l)}$$

其中 $\mathbf{H}^{\mathrm{feat,}(l)}_{\mathrm{self}} \in \mathbb{R}^{2q \times d_{\mathrm{model}}}$。

**步骤③：特征特定 FFN（NS-FFNs）**

Self-Attention 输出的 $2q$ 个 Token 具有不同的语义含义（$q$ 个来自序列交互，$q$ 个来自上层非序列），因此对每个切片 $\mathbf{h}^{\mathrm{feat,}(l)}_i$ 使用**独立的特征特定 FFN**：

$$\mathbf{f}^{\mathrm{feat,}(l)}_i = \mathrm{FFN}^{\mathrm{feat,}(l)}_{i}\Big( \mathbf{h}^{\mathrm{feat,}(l)}_i \Big) + \mathbf{h}^{\mathrm{feat,}(l)}_i, \quad i \in \{0, \dots, 2q-1\}$$

得到第 $l$ 层 FIM 的输出 $\mathbf{F}^{\mathrm{feat,}(l)} \in \mathbb{R}^{2q \times d_{\mathrm{model}}}$。

**步骤④：层间分裂（Layer Split）**

FIM 的输出按**层自适应分裂比** $\beta^{(l)}$ 沿特征维度拆分为两部分：

$$\mathbf{F}^{\mathrm{feat},(l)} = [\mathbf{Q}^{\mathrm{feat},(l+1)}_{\mathrm{cross}} : \mathbf{Q}^{\mathrm{feat},(l+1)}_{\mathrm{self}}]$$

其中：
- $\mathbf{Q}^{\mathrm{feat},(l+1)}_{\mathrm{cross}} \in \mathbb{R}^{2q\beta^{(l)} \times d_{\mathrm{model}}}$ → 流入下一层 Cross-Attention
- $\mathbf{Q}^{\mathrm{feat},(l+1)}_{\mathrm{self}} \in \mathbb{R}^{2q(1-\beta^{(l)}) \times d_{\mathrm{model}}}$ → 流入下一层 Self-Attention

默认 $\beta^{(l)} = 0.5$（均分），同时支持**金字塔式设计**：$\beta^{(l)}$ 随层数递减，使深层 Cross-Attention 逐步轻量化，进一步提升效率。

**FIM 输出总览**：

$M$ 层 FIM 的最终输出为 $\mathbf{F}^{\mathrm{feat},(M)} \in \mathbb{R}^{2q \times d_{\mathrm{model}}}$。这 $2q$ 个 Token 可理解为两类表示的融合：

| 来源路径 | 数量（$\beta=0.5$） | 物理含义 |
|---------|:---:|------|
| **序列增强路径**（$\mathbf{Q}_{\mathrm{cross}}$ 侧） | $\approx q$ | 非序列特征反复通过 Cross-Attention 从行为序列中**汲取用户兴趣信号**后的表示 |
| **非序列交互路径**（$\mathbf{Q}_{\mathrm{self}}$ 侧） | $\approx q$ | 特征之间经过多轮 Self-Attention + NS-FFN 的**高阶交叉特征表示**（用户 × 物品 × 上下文的组合模式） |

两类 Token 每层拼接 → Self-Attention 混合 → 再分裂，信息在两路径间持续交换，共同描述**特征空间的完整建模结果**。

> **Self-Attention 路径的双重角色**：$\mathbf{Q}_{\mathrm{self}}$ 路径不接触序列，只专注于非序列特征间的高阶交互。在下一层中，它通过拼接 $\mathbf{X}^{(l+1)} = [\mathbf{H}_{\mathrm{cross}}^{(l+1)}\;\|\;\mathbf{Q}_{\mathrm{self}}^{(l)}]$ 注入非序列信息，实现**交互增强（Interaction Enhancement）**——防止序列信号主导整个表示、保持特征理解的多样性，同时作为一条逐层进化的"记忆通道"持续积累高阶交互模式。

---

### 3.4 Task-space Interaction Module (TIM) — 任务空间交互

FIM 完成特征空间内的充分交互后，引入 TIM（$N$ 层）建模特征与任务之间的关系。TIM 同样包含两个互补组件。

---

#### 3.4.1 特征导向交互（Feature-oriented Interaction）

**目标**：让每个任务从 FIM 产生的高阶特征表示中**主动选择**对自己有用的信息，本质是任务感知的特征加权聚合。

以 Task Tokens 为 Query，对 FIM 的最终输出做 Cross-Attention：

$$\mathbf{H}_{\mathrm{cross}}^{\mathrm{task},(l)}
= \mathrm{CA}\Big(
\mathrm{RMSNorm}(\mathbf{Q}^{\mathrm{task},(l-1)}_{\mathrm{cross}}),\;
\mathbf{K}_{\mathrm{feat}}^{(l)},\;
\mathbf{V}_{\mathrm{feat}}^{(l)}
\Big) + \mathbf{Q}^{\mathrm{task},(l-1)}_{\mathrm{cross}}$$

其中：
- **Query** $\mathbf{Q}^{\mathrm{task,}(l-1)}_{\mathrm{cross}} \in \mathbb{R}^{t \times d_{\mathrm{model}}}$：$t$ 个任务 Token
- 首层 $\mathbf{Q}^{\mathrm{task,}(0)}_{\mathrm{cross}} = \mathbf{N_{\mathrm{T}}}$（来自 Tokenization 的任务特征 Token）
- **KV**：首层 $\mathbf{K}_{\mathrm{feat}}^{(1)} = \mathbf{V}_{\mathrm{feat}}^{(1)} = \mathbf{F}^{\mathrm{feat,}(M)}$（FIM 最终层输出）
- 同样采用 **Lazy KV 设计**：各层共享 KV，提升效率

这一设计在功能上**等价于 MMoE 框架**——每个任务通过注意力权重自适应地从全局特征池中聚合信息，形成任务特定的表示。

---

#### 3.4.2 任务导向交互（Task-oriented Interaction）

**目标**：在任务感知特征聚合之后，进一步捕获**任务间交互**（如 Click 和 Like 的相关性）。

先通过 Self-Attention 建模任务间关系，随后使用**任务特定 FFN（T-FFNs）** 增强每个任务的建模：

$$\mathbf{H}^{\mathrm{task,}(l)}_{\mathrm{self}} = \mathrm{SA}\Big( \mathrm{RMSNorm}(\mathbf{H}_\mathrm{cross}^{\mathrm{task,}(l)}) \Big) + \mathbf{H}_\mathrm{cross}^{\mathrm{task,}(l)}$$

$$\mathbf{f}^{\mathrm{task,}(l)}_i = \mathrm{FFN}^{\mathrm{task,}(l)}_{i}\Big( \mathbf{h}^{\mathrm{task,}(l)}_i \Big) + \mathbf{h}^{\mathrm{task,}(l)}_i, \quad i \in \{0, \dots, t-1\}$$

其中 $\mathbf{h}^{\mathrm{task,}(l)}_i$ 是 $\mathbf{H}^{\mathrm{task,}(l)}_{\mathrm{self}}$ 的第 $i$ 个切片（对应第 $i$ 个任务）。

第 $l$ 层 TIM 的输出 $\mathbf{F}^{\mathrm{task,}(l)} \in \mathbb{R}^{t \times d_{\mathrm{model}}}$ 直接作为下一层的 Query：$\mathbf{Q}^{\mathrm{task,}(l+1)}_{\mathrm{cross}} = \mathbf{F}^{\mathrm{task,}(l)}$。

**TIM Token 数量与物理含义**：

TIM 全程 Token 数恒定为 $t$（任务数），不增不减。每层 $t$ 个 Token 的语义演变如下：

| 阶段 | 每个 Token 的物理含义 |
|------|------|
| **输入** $\mathbf{N_T}$ | 任务的"身份信息"：Task ID Embedding + 先验偏置特征 |
| **Cross-Attn 后** $\mathbf{H}_{\mathrm{cross}}$ | **任务感知的特征聚合**：每个任务从 $2q$ 个特征 Token 中自适应地加权提取对其优化有用的信息（等价于 MMoE Gate） |
| **SA 后** $\mathbf{H}_{\mathrm{self}}$ | **任务间知识共享**：任务间互相通信，捕捉"点击→点赞""分享→关注"等任务相关性 |
| **T-FFN 后** $\mathbf{F}^{\mathrm{task,}(l)}$ | **任务专属深度表示**：经独立 FFN 非线性变换，每个任务的最终精炼表示 |

---

### 3.5 输出模块与损失函数

TIM 第 $N$ 层的输出 $\mathbf{F}^{\mathrm{task,}(N)}$ 即为**任务特定的最终表示**，送入任务特定 FFN Head + Sigmoid 激活，产生最终预测：

$$\hat{y}_i = \sigma\big( \mathrm{FFN}_{i}( \mathbf{f}^{\mathrm{task,}(N)}_i ) \big)$$

采用加权多任务损失：

$$\mathcal{L} = \sum_{i=1}^{t} \lambda_i \mathcal{L}_i(y_i, \hat{y}_i)$$

其中 $\mathcal{L}_i$ 为任务特定损失（如 BCE），$\lambda_i$ 为任务权重。

---

### 3.6 三类特征在多视图 FFN 中的参数扩展

UniFormer 的核心创新之一是针对不同建模组件设计了**三类独立的多视图 FFN**，支持灵活可扩展的参数分配：

| FFN 类型 | 应用位置 | 作用 | 参数量扩展方式 |
|----------|---------|------|:------------:|
| **S-FFNs** | FIM 序列导向交互后 | 短/长/跨域序列各自的专属建模 | 每个序列类型独立扩展 |
| **NS-FFNs** | FIM 非序列导向交互后 | 每个非序列 Token 的特征特定建模 | $2q$ 个独立 FFN，按 Token 数扩展 |
| **T-FFNs** | TIM 任务导向交互后 | 每个任务的专属建模 | $t$ 个独立 FFN，按任务数扩展 |

这种设计使 UniFormer 在 Scale Up 时能够**均衡地扩展各组件容量**，避免参数过度集中于某一模块（如传统组件级扩展中常见的特征交互模块参数臃肿），从而有效缓解收益递减现象。

### 3.7 优化与部署

#### 3.7.1 训练优化

**User-Level Common Compression（用户级公共压缩）**：

同一请求中多个候选 Item 共享完全相同的用户侧行为序列。由于行为特征的多值特性和可观长度，冗余存储和重复处理会引入大量内存和计算开销。UniFormer 对每个用户的序列特征**仅存储和处理一次**，通过索引映射将共享表示关联到对应样本。

对于包含 $B$ 个样本、来自 $U$ 个独特用户的 batch，设 $\bar{k}=B/U$ 为每用户平均样本数，$C_{\mathrm{com}}$ 为用户侧公共特征处理代价，$C_{\mathrm{sample}}$ 为剩余特征代价：

- 压缩前：$B(C_{\mathrm{com}}+C_{\mathrm{sample}})$
- 压缩后：$U C_{\mathrm{com}} + B C_{\mathrm{sample}}$
- 理论加速比：$\displaystyle \frac{C_{\mathrm{com}}+C_{\mathrm{sample}}}{C_{\mathrm{sample}}+C_{\mathrm{com}}/\bar{k}}$

当 $\bar{k}$ 较大（即每用户样本数多）时加速效果尤为显著。

**Variable-Length FlashAttention**：

语义分组的 Query Token 需要 attend 到不同长度的行为序列。给定 batch 内序列长度 $\{L_i\}_{i=1}^{B}$ 和 Query 长度 $q$：
- Padding Cross-Attention 复杂度：$O(B q L_{\max} d)$
- 可变长度 Cross-Attention 复杂度：$O(q \sum_i L_i d)$

配合 FlashAttention 的 IO-aware tiling（避免显式构建完整注意力矩阵），同时减少冗余计算和内存带宽。

**BF16 混合精度训练**：相比 FP32，BF16 张量存储减半，可利用低精度 Tensor Core 加速矩阵密集算子，同时保持与 FP32 相同的指数位宽，提供比 FP16 更好的数值稳定性。

---

#### 3.7.2 推理加速：User-Item 解耦

这是 UniFormer 在工业部署中的关键优化，直接受益于语义 Tokenization 的精心设计。

**问题背景**：在线 serving 时，1 个用户请求通常伴随 $I$ 个候选 Item（通常 $I=512$ 或 $1024$）进行排序。如果所有计算逐候选重复，推理成本将随 $I$ 线性增长。

**解耦原理**：

得益于语义 Tokenization 将非序列特征分解为 $m$ 个 Item-Independent Token（User 侧）和 $n$ 个 Item-Dependent Token（Item 侧），同时将 Item-Dependent 行为序列（如 SIM）聚合为紧凑 Token（使 Cross-Attention 的 KV 与目标 Item 解耦），可以实现：

- **User 侧**：Item-Independent Token 的**所有计算**（包括 Cross-Attention 和 FFN）每个请求**仅执行一次**，跨 $I$ 个候选复用
- **Item 侧**：仅对 $n$ 个 Item-Dependent Token 逐候选独立计算

**Self-Attention 处理**：

Cross-Attention 和 FFN 天然支持按 Token 解耦。但 Full Self-Attention 涉及所有 Token 之间的交互，会使 User 侧表示依赖于 Item 侧。解决方式：使用 **Attention Mask 切断 User 侧 Query 到 Item 侧 Key 的注意力**，使 User 侧的 Self-Attention 计算也可复用。

复杂度对比：
- 耦合推理：$O(I \cdot (m+n))$ 计算量
- 解耦推理：$O(m + I \cdot n)$ 计算量
- 当 $I$ 很大时，加速比趋近于 $1 + m/n$

**实际效果**：在生产环境中（每请求 512 候选），User-Item 解耦使推理 QPS 提升 **48%**，且离线 GAUC 退化可忽略。

## 4. 实验

### 4.1 实验设置

- **数据集**：快手单列短视频推荐场景（4亿+日活用户，500亿+日交互日志）
- **评估任务**：Effective-view、Long-view、Like、Follow
- **评估指标**：GAUC（快手最重要的离线指标）
- **基线**：SIM+DCN、SIM+HoME、SIM+RankMixer、HyFormer、MixFormer

### 4.2 离线性能对比

| 模型 | Effective-view | Long-view | Like | Follow | 参数量 |
|------|:------------:|:---------:|:----:|:------:|:------:|
| SIM+DCN | 0.7418 | 0.7734 | 0.8486 | 0.8361 | 115.8M |
| SIM+HoME | +0.08% | +0.08% | +0.09% | +0.16% | 114.8M |
| SIM+RankMixer | +0.34% | +0.30% | +0.25% | +0.16% | 492.0M |
| HyFormer | +0.39% | +0.36% | +0.31% | +0.24% | 496.5M |
| MixFormer | +0.43% | +0.39% | +0.33% | +0.32% | 489.4M |
| **UniFormer** | **+0.53%** | **+0.48%** | **+0.53%** | **+0.89%** | 516.0M |
| **UniFormer-Large** | **+0.63%** | **+0.58%** | **+0.61%** | **+1.04%** | 995.3M |

> 注：在亿级样本规模下，GAUC 提升 **0.05%** 即被视为具有显著商业价值的提升

### 4.3 消融实验

| 消融变体 | 结论 |
|---------|------|
| **w/o Multi-Seq CA** | 交互类指标（如 Like）明显下降，验证了防止偏好坍塌的有效性 |
| **w/o IE（交互增强）** | 去除 Self-Attention 增强导致性能下降 |
| **w/o TIM** | 缺少任务空间建模导致全面退化 |
| **w/o S-FFNs / NS-FFNs** | **性能下降最严重**，说明灵活参数分配对均衡扩展至关重要 |

### 4.4 Scaling Law 分析

- UniFormer 展现出明显的 **Scaling Law** 趋势：模型性能随参数增加持续提升
- 得益于多视图 FFN 设计，UniFormer 保持更强的性能增长曲线
- 有效缓解了传统组件级扩展中较早出现的**收益递减（Law of Diminishing Returns）**现象

### 4.5 可视化分析

- **语义 Tokenization vs 全局 Tokenization**：语义 Token 在不同层展现出更多样化的注意力模式，避免注意力同质化（Attention Homogenization）
- **特征-任务关系**：不同任务自适应地关注不同特征源；相关任务（如评论时长和评论率）展现出高度一致的注意力分布，验证了模型捕获任务相关性的能力

### 4.6 在线 A/B 测试

| 指标类别 | 指标 | 快手极速版 | 快手主端 |
|---------|------|:--------:|:-------:|
| 用户时长 | App Stay Time | **+0.260%** | **+0.101%** |
| | Watch Time | **+1.113%** | **+0.729%** |
| | Video View | +0.252% | +0.249% |
| 互动 | Like | **+1.089%** | +0.155% |
| | Comment | **+1.818%** | **+1.488%** |
| | Collect | +0.930% | +0.647% |
| | Forward | +1.274% | +0.157% |

## 5. 总结与启示

1. **模型级扩展 > 组件级扩展**：将行为建模、特征交互和任务建模统一在标准化 Transformer 框架下联合扩展，远比独立扩展各组件更有效
2. **多视图 FFN 设计**是灵活参数分配的 key design——允许不同组件（序列、非序列、任务）按需独立扩展容量
3. **语义 Tokenization + User-Item 解耦**使统一模型架构在工业级高并发场景下仍可高效部署（QPS 提升 48%）
4. **多序列交叉注意力**有效防止偏好坍塌，是处理异构行为序列的有效方案
5. 该工作已在快手主端和极速版**全量上线**，服务超过 4 亿日活用户
