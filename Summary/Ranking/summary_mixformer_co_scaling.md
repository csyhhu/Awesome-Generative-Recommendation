# MixFormer: Co-Scaling Up Dense and Sequence in Industrial Recommenders

- 论文：[*MixFormer: Co-Scaling Up Dense and Sequence in Industrial Recommenders*](https://arxiv.org/abs/2602.14110)
- 会议：KDD 2026
- 作者单位：ByteDance
- 关键词：Large Recommender Model、Feature Interaction、Long Sequence Modeling、Scaling Law、Request-Level Batching

## 1. 核心结论

MixFormer 的核心不是简单地把序列模型和特征交互模型拼在一起，而是让二者在每一层中反复进行：

```text
非序列特征
    ↓
Query Mixer：构造高阶 user/item/context query
    ↓
Cross Attention：用这些 query 聚合原始行为序列
    ↓
Output Fusion：逐 head 融合非序列与序列信息
    ↓
下一层继续联合建模
```

传统工业架构通常是 `Sequence Model -> Dense Model` 或 `Sequence Model ⊕ Dense Model`，两部分参数彼此独立。当计算预算固定时，加长序列会挤压 Dense Backbone 的容量，扩大 Dense Backbone 又会限制序列长度。MixFormer 将二者放入统一的 Transformer-style Backbone，使同一组 per-head FFN 同时参与特征交互和序列融合，从结构上缓解参数与 FLOPs 的分配冲突。

在抖音万亿级样本上，MixFormer-medium 相比最强的 `STCA -> RankMixer` 基线：

- Finish AUC / UAUC 从 `+1.12% / +1.40%` 提升到 `+1.28% / +1.60%`；
- Skip AUC / UAUC 从 `+1.43% / +2.14%` 提升到 `+1.60% / +2.46%`；
- Dense 参数量相近（1.226B vs 1.255B），FLOPs 从 6,736 降到 3,503 GFLOPs/Batch；
- UI-MixFormer 在精度不变的情况下进一步降到 2,242 GFLOPs/Batch，相比原始 MixFormer 减少约 36%；
- 配合 Request-Level Batching 后，线上 Serving 加速约 30%~34%。

## 2. 问题定义：为什么需要 Co-Scaling

现代推荐排序模型有两个主要扩展轴：

1. **Dense Scaling**：增加特征交互网络的宽度、深度和参数量，学习更高阶的 user-item-context 关系。
2. **Sequence Scaling**：增加用户历史长度，捕捉更长期、更多样的兴趣。

现有方案通常采用两类组合方式：

| 范式 | 典型结构 | 主要问题 |
|---|---|---|
| Stacked | `STCA -> RankMixer` | 序列先被压缩成固定表示，再进入 Dense 模型，后续高阶特征无法反向参与细粒度序列聚合 |
| Parallel | `STCA ⊕ RankMixer` | 两路输出最后拼接，交互更浅，序列与 Dense 参数仍然各自扩展 |
| Heterogeneous Transformer | OneTrans | 将异构特征放入 Transformer，但 full self-attention 带来很高的二次复杂度 |

固定序列长度时，论文发现扩 Dense 模块的 FLOPs ROI 更高；但固定 Dense 参数、增加序列长度时，序列模型的收益斜率更大。因此，独立参数化的模型必须在两种互相竞争的收益曲线之间手工分配预算，很难同时获得好的 Dense Scaling 和 Sequence Scaling。

MixFormer 的设计目标就是：**让一份 Backbone Capacity 同时服务于高阶特征交互和序列建模。**

## 3. 输入表示

### 3.1 序列特征

用户行为序列长度为 $T$。每个行为包含 item ID、action type、timestamp 和 side information，各字段独立 Embedding 后拼接为：

$$
\mathbf{S}=[\mathbf{s}_1,\mathbf{s}_2,\ldots,\mathbf{s}_T].
$$

MixFormer 不在序列 token 之间做 full self-attention，而是让非序列 query 对整段序列做 Cross Attention，因此复杂度随 $T$ 线性增长。

### 3.2 非序列特征

用户、候选物品和上下文特征的 Embedding 先拼接：

$$
\mathbf{e}_{\mathrm{ns}}=[\mathbf{e}_1;\mathbf{e}_2;\ldots;\mathbf{e}_M]
\in \mathbb{R}^{D_{\mathrm{ns}}}.
$$

再将其均匀切成 $N$ 段，每段独立投影为一个 $D$ 维 head：

$$
\mathbf{x}_j=\mathbf{W}_j\mathbf{e}_{\mathrm{ns}}[d(j-1):dj],
\qquad d=D_{\mathrm{ns}}/N.
$$

最终输入为：

$$
\mathbf{X}=[\mathbf{x}_1,\ldots,\mathbf{x}_N]\in\mathbb{R}^{N\times D}.
$$

与“所有特征压成一个向量”相比，多 head 保留了更多异构语义；与“一字段一 token”相比，它又避免 token 数过大。这里的 head 不要求对应明确字段，而是对拼接空间进行固定切片和独立投影。

## 4. MixFormer Block

每个 Block 对应 Decoder-only Transformer 的三个部件：

| 标准 Transformer Decoder | MixFormer |
|---|---|
| Self-Attention | Query Mixer |
| Cross-Attention | Cross Attention |
| FFN | Output Fusion |

论文使用 $L$ 个 Block，最后连接多个任务网络。

### 4.1 Query Mixer：低成本高阶特征交互

推荐特征来自用户、物品、场景和超大稀疏 ID 空间，不像语言 token 那样天然处于统一语义空间。因此，作者认为用内积相似度做非序列特征 Self-Attention 并不可靠，且成本较高。

MixFormer 采用 RankMixer 风格的无参数 HeadMixing：

```text
X: [N, D]
 -> reshape:   [N, N, D/N]
 -> transpose 前两个维度
 -> flatten:   [N, D]
```

它本质上是一次确定性的 reshape + transpose，让每个输出 head 获得所有输入 head 的一部分信息，不引入额外矩阵乘。随后每个 head 使用独立的 SwiGLU FFN：

$$
\mathbf{P}=\operatorname{HeadMixing}(\operatorname{RMSNorm}(\mathbf{X}))+\mathbf{X},
$$

$$
\mathbf{q}_i=\operatorname{SwiGLUFFN}_i(\operatorname{RMSNorm}(\mathbf{p}_i))+\mathbf{p}_i.
$$

这里的 $\mathbf{q}_i$ 已经不是原始特征切片，而是融合了跨 head 信息的高阶 query。Per-head FFN 为不同异构子空间保留独立参数。

### 4.2 Cross Attention：高阶 Query 直接聚合行为序列

第 $l$ 层先使用该层独立的 SwiGLU FFN 变换每个行为：

$$
\mathbf{h}_t^{(l)}=
\operatorname{SwiGLUFFN}^{(l)}(\operatorname{RMSNorm}(\mathbf{s}_t))+\mathbf{s}_t,
$$

再将 $\mathbf{h}_t^{(l)}$ 切成 $N$ 个 head，分别投影为 key/value。第 $i$ 个 Query Mixer 输出直接作为第 $i$ 个 Cross-Attention query：

$$
\mathbf{z}_i=
\sum_{t=1}^{T}
\operatorname{softmax}\left(
\frac{\mathbf{q}_i^\top\mathbf{k}_t^i}{\sqrt{D}}
\right)\mathbf{v}_t^i+\mathbf{q}_i.
$$

这一设计有三个关键含义：

1. 序列聚合不再只由 target embedding 驱动，而是由融合了 user、item、context 的高阶 query 驱动。
2. $N$ 个 query head 可以从不同语义子空间关注不同历史行为，不需要再做标准 Attention 的 query splitting 投影。
3. 每层重新变换序列表示，并用上一层融合后的非序列表示再次检索序列，形成“特征交互 ↔ 序列聚合”的逐层迭代。

### 4.3 Output Fusion：逐 Head 深度融合

Cross Attention 的输出已经同时包含 query 残差和序列聚合信息。MixFormer 再用独立的 per-head SwiGLU 做非线性融合：

$$
\mathbf{o}_i=
\operatorname{SwiGLUFFN}_i(\operatorname{RMSNorm}(\mathbf{z}_i))+\mathbf{z}_i.
$$

$\{\mathbf{o}_i\}$ 作为下一层 MixFormer Block 的输入。Query Mixer 与 Output Fusion 中的 per-head FFN 都会处理已经融合的序列和非序列信号，这是论文所谓“统一参数化”的核心。

## 5. UI-MixFormer：面向多候选请求的计算复用

原始 MixFormer 会在 Query Mixer 中混合 user-side 和 item-side 信号，因此同一请求中的不同候选无法直接共享用户计算。UI-MixFormer 为 Request-Level Batching（RLB）做了两项改造。

### 5.1 Feature Decoupling

将非序列特征拆为：

- User-side：用户画像、用户上下文等，可在同一请求内复用；
- Item-side：候选 item 及 item-dependent context，每个候选独立计算。

二者分别生成 $N_U$ 和 $N_G$ 个 head，并保持 $N_U+N_G=N$。论文按两侧 Embedding 维度分配 head，线上实践取 $N_U:N_G=1:1$。

### 5.2 Masked HeadMixing

若直接做 HeadMixing，item 信息会流入 user head，使 user head 变成候选相关、无法共享。作者构造 mask：

$$
\mathcal{M}[i,j]=
\begin{cases}
0,&i<N_U\ \text{且}\ j\ge N_U\frac{D}{N},\\
1,&\text{其他情况},
\end{cases}
$$

$$
\operatorname{HeadMixing}_{\mathrm{decouple}}(\cdot)
=\mathcal{M}\odot\operatorname{HeadMixing}(\cdot).
$$

该 mask 实现单向信息流：

- user head 不接收 item 信息，因此可跨候选复用；
- item head 仍可接收 user 信息，因此没有退化成缺少交互的双塔模型；
- 用户行为序列及 user-side Cross Attention 也可按 request 共享。

这是一种“保留 user-to-item 交互，同时阻断 item-to-user 污染”的工程折中。

## 6. 实验设置

- 数据：抖音连续两周日志，包含万亿级 user-item 交互；
- 特征：300+ 个非序列与序列特征；
- 任务：Finish 和 Skip 两个 CTR-style 二分类任务；
- 指标：AUC、UAUC、Dense 参数量、GFLOPs/Batch；
- 训练：数百张 GPU，Sparse 参数异步更新、Dense 参数同步更新；
- 优化器：Dense 使用 RMSProp，学习率 0.01；Sparse 使用 Adagrad；
- Batch Size：1,500；
- 默认序列长度：512；
- MixFormer-small：$N=16,L=4,D=386$，282M Dense 参数；
- MixFormer-medium：$N=16,L=4,D=768$，1.226B Dense 参数。

## 7. 离线结果

表中的增益均以 `TA -> DLRM` 为参照；该参照的绝对指标为 Finish AUC/UAUC `0.8554/0.8270`，Skip AUC/UAUC `0.8124/0.7294`。

| 模型 | Finish AUC | Finish UAUC | Skip AUC | Skip UAUC | Dense 参数 | GFLOPs/Batch |
|---|---:|---:|---:|---:|---:|---:|
| STCA -> RankMixer | +1.12% | +1.40% | +1.43% | +2.14% | 1,255M | 6,736 |
| OneTrans | +1.05% | +1.31% | +1.30% | +1.95% | 316M | 23,371 |
| STCA ⊕ RankMixer | +1.11% | +1.38% | +1.42% | +2.11% | 1,255M | 6,736 |
| MixFormer-small | +1.01% | - | +1.18% | - | 282M | 733 |
| **MixFormer-medium** | **+1.28%** | **+1.60%** | **+1.60%** | **+2.46%** | 1,226M | 3,503 |
| **UI-MixFormer-medium** | **+1.28%** | **+1.60%** | **+1.60%** | **+2.46%** | 1,226M | **2,242** |

主要观察：

1. `STCA -> RankMixer` 与 `STCA ⊕ RankMixer` 几乎没有差异，说明只改变独立模块的连接方式无法产生深层协同。
2. OneTrans 的精度较强，但 full heterogeneous attention 导致 FLOPs 极高。
3. MixFormer-medium 以略少于强基线的参数获得更高精度，同时 FLOPs 约为其 52%。
4. UI-MixFormer 的 mask 和计算复用没有造成可见精度损失，却进一步减少 36% FLOPs。

## 8. 消融实验

以 MixFormer-small 为基准，图中给出的 AUC 变化为：

| 改动 | AUC Gain 变化 | 结论 |
|---|---:|---|
| Query Mixer 去掉 HeadMixing | -0.03 | 跨 head 信息交换有效 |
| HeadMixing 换成 Self-Attention | +0.00 | 更贵的相似度 Attention 没有带来收益 |
| Query Mixer 去掉 per-head FFN | -0.04 | 高阶 query 构造很重要 |
| Cross Attention 的 per-layer FFN 改为共享 FFN | -0.03 | 各层独立序列表征有价值 |
| Output Fusion 的 per-head FFN 改为共享 FFN | **-0.06** | 异构 head 的独立融合最重要 |
| Pre-RMSNorm 改为 Post-LayerNorm | -0.01 | Pre-RMSNorm 略优且更符合深层稳定训练需求 |

值得注意的是，HeadMixing 替换为 Self-Attention 后没有精度提升。这支持论文的核心判断：推荐中的异构 feature head 不一定适合用 token 相似度决定交互权重，确定性的全局 mixing 加独立 FFN 反而更划算。

## 9. Co-Scaling 分析

### 9.1 Dense Scaling

固定序列长度为 512，以 FLOPs 而非参数量为横轴：

- 扩大 `TA + RankMixer` 的 Dense 模块比扩大 `STCA + DCNv2` 的序列模块有更高边际收益；
- 独立组合必须在两类模块间分配 FLOPs，出现明显 trade-off；
- MixFormer 的起点更高，Scaling Slope 也保持竞争力，在所有 FLOPs 区间领先。

这说明在序列长度固定时，target/non-sequential 的高阶交互仍是非常高效的容量投入方向。

### 9.2 Sequence Scaling

固定 Dense 模型规模，将序列长度从 512 扩展到 2,048、8,192、10,000：

- STCA 这类序列侧投入更高的模型，其长度 Scaling Slope 高于 `TA + RankMixer`；
- MixFormer 的长度 Scaling Slope 与 STCA 接近，但整体 AUC Gain 始终更高；
- 在 10K 长度附近，图中 MixFormer AUC Gain 约为 `1.63%`，STCA + DCNv2 约为 `1.22%`，TA + RankMixer 约为 `1.11%`。

因此，MixFormer 的优势并非只来自一个更高的固定截距：它同时保留了 Dense 扩容效率和长序列扩展能力。

需要谨慎理解的是，论文展示的是经验 Scaling Curve，没有拟合显式幂律、报告指数或外推误差；“Co-Scaling”更多是架构与实验现象，而非严格的 Scaling Law 定量模型。

## 10. Serving 与在线 A/B

### 10.1 Serving Latency

随着候选数增加，原始 MixFormer 延迟从约 35.3 ms 增长到 74.2 ms；UI-MixFormer 从约 24.7 ms 增长到 49.0 ms。对应加速比从 30.0% 上升到 34.0%。

候选越多，普通模型越容易达到 GPU 饱和；UI-MixFormer 的用户侧计算只执行一次，因此延迟增长更缓。这说明其收益不是单纯 Kernel 优化，而是改变了计算随候选数增长的方式。

### 10.2 两周在线实验

线上基线为超过 1B 参数的 `STCA -> RankMixer`，实验覆盖抖音与抖音极速版 Feed 推荐。论文称所有结果均统计显著，且实验结束时仍未收敛。

| App | 人群 | Active Day | Duration | Like | Finish | Comment |
|---|---|---:|---:|---:|---:|---:|
| Douyin | Overall | +0.0415% | +0.2799% | +0.1766% | +0.3897% | +0.7035% |
| Douyin | Low-active | +0.2263% | +0.2468% | +0.0771% | +0.4123% | +1.2483% |
| Douyin | Middle-active | +0.0998% | +0.2719% | +0.2445% | +0.2796% | +0.6718% |
| Douyin | High-active | +0.0203% | +0.2938% | +0.3810% | +0.3335% | +0.8356% |
| Douyin Lite | Overall | +0.0252% | +0.4105% | +0.2125% | +0.2924% | +1.9097% |
| Douyin Lite | Low-active | +0.2543% | +0.6044% | +3.0565% | +0.6157% | +2.6452% |
| Douyin Lite | Middle-active | +0.1218% | +0.4184% | +0.2329% | +0.2951% | +1.3286% |
| Douyin Lite | High-active | +0.0237% | +0.4042% | +0.4871% | +0.2097% | +2.1170% |

低活用户的 Active Day 增益明显更高，说明更充分的长期兴趣与高阶上下文联合建模，可能对历史信号稀疏或活跃度较低的用户更有帮助。不过论文没有给出分桶样本量和置信区间，不能仅凭该表确认具体因果机制。

## 11. 与 Summary 中相关工作的关系

MixFormer 最适合归入 **Unified Sequence-Feature Interaction Backbone（统一序列-特征交互骨干）**，而不只是普通长序列模型或 TokenMixer 模型。它的技术位置可以概括为：

> **MixFormer = RankMixer/TokenMixer 风格的 Dense Mixing + STCA 风格的线性序列 Cross Attention/RLB + HyFormer/UniFormer/WHALE 风格的逐层统一融合。**

### 11.1 最相似的统一架构

| 相似度 | 工作 | 共同点 | 与 MixFormer 的关键区别 |
|---|---|---|---|
| 最高 | **HyFormer** | 都将高阶非序列 Query 构造与行为序列聚合交替堆叠；HyFormer 的 `Query Decoding <-> Query Boosting` 与 MixFormer 的 `Query Mixer -> Cross Attention -> Output Fusion` 高度同构 | HyFormer 有显式 Query Generation、原始非序列 token 直通和多序列 query；MixFormer 采用固定切片 head，并更强调 Co-Scaling 与 UI-MixFormer/RLB |
| 很高 | **WHALE** | 都让非序列交互结果作为 Query、长行为序列作为 KV，并在多层中渐进融合 | WHALE 保留 Wukong 与 HSTU 两条持续演化的分支；MixFormer 将三类操作收进单一 Decoder-style Block，主状态是融合后的非序列 heads |
| 很高 | **UniFormer** | 都反对独立模块拼装，主张模型级统一扩展；都让多个非序列视角查询序列，并设计 user-item 解耦 serving | UniFormer 还通过 TIM 统一多任务空间，覆盖范围更广；其 Dense 交互使用 Self-Attention，MixFormer 使用无参数 HeadMixing |
| 较高 | **STCA/RLB** | 都以 Query-to-History Cross Attention 实现对序列长度线性的建模，并重视请求级多候选复用 | STCA 是单 target query 的序列模块；MixFormer 的 Query 是 user/item/context 交互后的多个高阶 head，并将序列聚合嵌入每一层 |
| 较高 | **TokenMixer-Large** | 都使用 reshape/transpose 式无参数 Mixing、per-head/per-token SwiGLU 和 RMSNorm 处理异构推荐特征 | TokenMixer-Large 是 Dense Backbone，序列通常由外部模块预压缩；MixFormer 在每层直接 Cross-Attend 未压平的行为序列 |
| 相关 | **LONGER** | 都用少量 Query 对长历史做 Cross Attention，并考虑多候选 Serving 复用 | LONGER 重点是将长序列压缩后继续 Self-Attention；MixFormer 不以物理压缩为核心，而关注 Dense Capacity 与 Sequence Length 的协同扩展 |
| 理论相关 | **UniMixer** | 都属于异构推荐 token mixing 与 Scaling 路线；UniMixer 可用于解释 HeadMixing 在特征交互算子谱系中的位置 | UniMixer 统一的是 Attention、TokenMixer 与 FM 类 mixing 算子，本身不负责行为序列融合 |

从完整架构看，**HyFormer 是最接近 MixFormer 的工作**：两者都不满足于把 Sequence Encoder 的最终输出作为一个普通 Dense Feature，而是让当前层产生的高阶 query 再次读取原始行为序列，由此形成双向、逐层的协同建模。

WHALE 与它们也属于同一核心族，但更接近“双 Backbone + Cross Fusion”：Wukong 负责非序列特征交互，HSTU 负责维护完整序列状态。MixFormer 则把 Query Mixing、序列读取和融合压进同一种 Block，参数边界更弱。

UniFormer 的统一范围比 MixFormer 更广。它除了统一序列与非序列 Feature Space，还用 Task Token 和 TIM 统一 Task Space；MixFormer 没有专门解决多任务之间的交互，主要聚焦 Dense/Sequence 两个 Scaling 轴。

### 11.2 与 Dense Backbone 的关系

MixFormer 的 Query Mixer 明显继承 RankMixer/TokenMixer 路线：

```text
RankMixer / TokenMixer-Large
    reshape + transpose HeadMixing
    per-token/per-head FFN
                |
                v
MixFormer Query Mixer
    HeadMixing + per-head SwiGLU
    生成用于读取行为序列的高阶 Query
```

二者的关键分界在于序列进入模型的位置：

- **TokenMixer-Large**：DIN、LONGER 等外部序列模块先把行为压缩成固定 Embedding，Backbone 将它当作普通特征 token；
- **MixFormer**：行为序列保持为逐行为 KV，高阶非序列 Query 在每个 Block 中直接读取序列；
- **UniMixer**：更偏 mixing 算子的理论统一，可作为 Dense 侧替代模块，但没有直接处理 Sequence-Dense Co-Scaling。

因此，TokenMixer-Large 和 UniMixer 可以看作 MixFormer 的 **Dense Backbone 父系**，而不是完整架构同类。

### 11.3 与长序列模型的关系

STCA、LONGER、HSTU 和 IAT 代表不同的长序列扩展方式：

| 工作 | 长序列策略 | 与 MixFormer 的关系 |
|---|---|---|
| **STCA + RLB + Ext** | 单 target query 多层扫描历史，复杂度对长度线性；训练与推理按 request 复用 | 是 MixFormer Cross Attention 和 UI/RLB 设计最直接的序列侧参照 |
| **LONGER** | Token Merge + sampled queries 压缩长序列，再在短序列上 Self-Attention | 可提供更强的序列预处理，但其重点不是 Dense-Sequence 的逐层联合参数化 |
| **HSTU/GR** | 用 Sequential Transduction Unit 统一行为序列建模，并走生成式推荐路线 | 同属推荐大模型 Scaling，但 HSTU 主状态是行为序列，MixFormer 主状态是用于排序的融合 feature heads |
| **IAT** | 将历史训练实例压缩为高信息密度 InsEmb token | 属于输入侧增强，可以作为 MixFormer 的序列 token 来源，但不解决 Backbone 融合问题 |

MixFormer 与 `STCA -> RankMixer` 的核心区别是：后者只在两个模块的边界交接一次；前者在每层都使用最新的高阶非序列 Query 重新聚合行为序列。换言之，STCA 是 MixFormer 的 **序列模块来源**，MixFormer 则将它提升为统一 Backbone 内部的反复交互过程。

### 11.4 技术谱系

```text
Dense Feature Interaction
RankMixer / TokenMixer-Large / UniMixer
        |
        | HeadMixing、per-head FFN、Dense Scaling
        v
    +----------------+
    |   MixFormer    |
    +----------------+
        ^            ^
        |            |
        |            | 逐层 Sequence <-> Dense 融合
        |            |
STCA / LONGER     HyFormer / WHALE / UniFormer
Long Sequence     Unified Architecture
RLB / KV Cache    Model-level Co-Scaling
```

可以据此将 Summary 中的工作分成四组：

1. **统一序列-特征交互 Backbone**：MixFormer、HyFormer、WHALE、UniFormer。这是 MixFormer 所属的核心类别。
2. **Dense Feature Interaction Backbone**：RankMixer、TokenMixer-Large、UniMixer、Wukong。它们提供高效特征 Mixing 和 Dense Scaling 能力。
3. **Long Sequence Encoder / Sequence Feature Source**：STCA、LONGER、HSTU、IAT。它们分别解决线性 Attention、序列压缩、生成式序列转导和高信息密度 token。
4. **Generative Retrieval / Recommendation**：OneRec、OneReason、TSGR、TBGRecall 等。它们也追求“大推荐模型统一化”，但统一的是召回、排序或生成接口，不是 MixFormer 所处理的判别式 Ranking Backbone，因此只属于更外围的相关谱系。

### 11.5 最小关联阅读集合

若希望用最少论文覆盖 MixFormer 的来源与最近邻，可以按以下顺序阅读：

1. **MixFormer**：主线，理解统一 Block、Co-Scaling 和 UI-MixFormer。
2. **HyFormer**：与 MixFormer 最接近的逐层 Query Decoding/Boosting 架构。
3. **UniFormer**：理解 Feature Space 与 Task Space 的更全面统一。
4. **STCA + RLB + Ext**：理解线性长序列 Cross Attention、10K 外推和请求级复用。
5. **TokenMixer-Large**：理解 HeadMixing、per-head FFN 及 Dense Backbone Scaling。
6. **WHALE**：对比单一统一 Backbone 与 Wukong + HSTU 双分支融合路线。

这组工作共同回答一个更大的问题：**工业推荐 Scaling 不应继续独立扩大序列模块、特征交互模块和任务塔，而应让它们在可复用的计算图中逐层交换信息，并围绕请求级 Serving 约束共同设计。**

## 14. 一句话总结

**MixFormer 用“每层先做高阶特征交互，再以高阶 query 聚合行为序列，最后逐 head 融合”的统一 Backbone，取代独立 Sequence + Dense 模块的拼装方式，并通过单向 user-to-item mask 支持请求级复用，从而同时获得更好的 Dense Scaling、Sequence Scaling、离线精度和线上效率。**
