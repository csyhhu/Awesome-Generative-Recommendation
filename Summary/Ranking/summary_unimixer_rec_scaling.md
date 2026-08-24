# UniMixer: A Unified Architecture for Scaling Laws in Recommendation Systems

论文链接：<https://arxiv.org/abs/2604.00590>
发表会议：NeurIPS 2024
作者单位：快手科技（Kuaishou Technology）

***

## 一、研究动机与问题定义

大语言模型（LLM）的 Scaling Laws 揭示：随着模型规模、数据量和计算资源的增加，性能会持续可预测地提升，这激励了推荐系统社区探索适合推荐任务的扩展框架。

推荐系统与 NLP 的核心差异：NLP 所有 token 共享统一的 embedding 空间，而推荐系统的特征空间天然**异构**（用户画像、物品特征、行为序列、Query 特征等来自不同语义域），因此必须专门设计**异构特征交互**模块。

当前三类主流推荐扩展架构：

- **Attention-Based**（HiFormer、FAT、HHFT）：为每个 token 构建 token-specific Q/K/V 投影
- **TokenMixer-Based**（RankMixer、TokenMixer-Large）：使用基于规则、无参数的 token mixing 操作
- **FM-Based**（Wukong、Kunlun）：引入 FM Block 对输入 embedding 之间进行交互建模

**核心问题**：能否构建一个统一的推荐扩展模块，兼具三类方法的优势？

***

## 二、核心方法与贡献

### 2.1 TokenMixer 的等价参数化

论文发现 TokenMixer 操作等价于置换矩阵 W\_perm 与展平输入的乘积，该置换矩阵具有以下关键性质：

- 可压缩性：可分解为 Kronecker 积 G x I，参数量从 O(T^2 D^2) 大幅压缩
- 双随机性：每行每列之和均为 1
- 稀疏性：每行/列恰好只有一个非零元素
- 对称性：当 T=H 时为对称矩阵

### 2.2 UniMixing 模块

将规则化 TokenMixer 替换为可学习的参数化结构，分为全局交互矩阵 W\_G（控制 block-to-block 的交互模式）和局部交互矩阵 W\_Bi（为每个 block 分配独立的特征交互参数）。

通过优化计算流程，将复杂度从 O(L^2) 降低到 O(L^2/B + LB)，避免产生大中间变量。

参数约束：Sinkhorn-Knopp 迭代满足双随机性；温度系数 tau 控制稀疏性；对称化操作保证对称性。

### 2.3 统一理论框架

在 UniMixing 的框架下，三类主流方法可被统一：

| 方法                      | 局部交互模式  | 全局交互模式                   |
| ----------------------- | ------- | ------------------------ |
| Self-Attention          | XW\_V   | softmax(QK^T/sqrt(d))    |
| Heterogeneous Attention | 异构 V 投影 | softmax(异构 QK^T/sqrt(d)) |
| TokenMixer              | X（无参数）  | 固定置换矩阵 G                 |
| FM (Wukong)             | 固定矩阵 Y  | XI(XI)^T                 |

UniMixing 相当于同时具有可学习的局部交互（类 Attention）和低参数化的全局交互（类 TokenMixer）。

### 2.4 UniMixing-Lite（轻量化版本）

- 局部交互：引入 basis 分解，用 b 个基矩阵的线性组合动态生成每个 block 的局部权重
- 全局交互：使用低秩分解替代完整矩阵，进一步压缩参数和计算量

### 2.5 SiameseNorm 与训练策略

引入 SiameseNorm（双流耦合归一化）解决深层网络训练稳定性问题，缓解 Pre-Norm 和 Post-Norm 之间的矛盾。

温度退火策略：高温（tau=1.0）初始化 → 线性退火至低温（tau=0.05） → 支持 warm-up 预训练再 fine-tune。

***

## 三、实验结果

### 3.1 实验设置

- 数据集：快手广告投放场景真实日志，超过 7 亿用户样本，跨越一年
- 任务：预测用户次日留存（User Retention）
- 指标：AUC、UAUC、参数量、FLOPs
- 硬件：40 GPU 混合分布式训练

### 3.2 主要性能对比（约 100M 参数规模）

| 模型                           | AUC    | ΔAUC    | 参数量    |
| ---------------------------- | ------ | ------- | ------ |
| Heterogeneous Attention（基线）  | 0.7446 | -       | 132.7M |
| RankMixer                    | 0.7493 | +0.475% | 135.5M |
| TokenMixer-Large             | 0.7484 | +0.383% | 103.3M |
| UniMixer-Lite-4-Blocks 38.2M | 0.7523 | +0.775% | 38.2M  |
| UniMixer-Lite-4-Blocks 84.5M | 0.7527 | +0.814% | 84.5M  |

UniMixer-Lite 以更少的参数（38.2M vs 135.5M）取得了明显更高的 AUC。

### 3.3 Scaling Law 拟合

| 模型            | 参数 scaling 指数 | FLOPs scaling 指数 |
| ------------- | ------------- | ---------------- |
| RankMixer     | 0.116         | 0.117            |
| UniMixer      | 0.132         | 0.126            |
| UniMixer-Lite | 0.142（最优）     | 0.135（最优）        |

UniMixer-Lite 的 scaling 指数最大，每新增单位参数量能带来最大的性能收益。

### 3.4 消融实验（6.57M 参数模型）

| 设置                      | ΔAUC           |
| ----------------------- | -------------- |
| 完整 UniMixer             | -              |
| 去掉温度系数                  | -0.1645%（影响最大） |
| 去掉 Warm-Up              | -0.0856%       |
| 去掉对称约束                  | -0.0573%       |
| 去掉 block-specific 局部权重  | -0.0436%       |
| SiameseNorm → Post Norm | -0.0273%       |

### 3.5 深度 Scaling 对比

| 模型                     | AUC    | 参数量          |
| ---------------------- | ------ | ------------ |
| RankMixer-2-Blocks     | 0.7478 | 4.44M        |
| RankMixer-4-Blocks     | 0.7467 | 8.66M（性能下降！） |
| UniMixer-Lite-2-Blocks | 0.7492 | 4.97M        |
| UniMixer-Lite-4-Blocks | 0.7508 | 9.72M（稳定提升）  |

RankMixer 在深层堆叠时出现性能下降，而 UniMixer 借助 SiameseNorm 可持续受益于深度扩展。

### 3.6 在线 A/B 测试

在快手多个广告投放场景部署，30 天累计活跃天数（CAD D1-D30）多场景平均提升超过 15%。

***

## 四、主要结论

1. **理论统一**：首次建立统一框架，将三类主流推荐扩展方法（Attention/TokenMixer/FM）纳入同一理论体系
2. **性能领先**：UniMixer-Lite 在参数效率和计算效率上均超越现有 SOTA，scaling 指数最优
3. **工业验证**：在快手真实广告场景取得显著业务指标提升（CAD 平均 +15%）
4. **深度扩展**：通过 SiameseNorm 解决深层堆叠训练稳定性问题

***

## 五、相关性与应用价值

- **直接相关**：推荐系统 Scaling Laws、特征交互建模、异构特征学习
- **技术迁移**：UniMixer 模块可扩展到用户行为序列建模和生成式推荐任务
- **工业指导**：提供了推荐系统从 Attention/TokenMixer/FM 三条路线走向统一的实践路径
- **参数效率**：UniMixing-Lite 的 basis 分解和低秩近似为大规模推荐模型的参数压缩提供了新思路

***

## 六、深度讨论笔记

### 6.1 UniMixer 是否改变了 TokenMixer 的设计初衷？

TokenMixer 的设计初衷是用固定 shuffle（置换）替代复杂矩阵计算，完全规避异构 token 之间内积无语义意义的问题，代价是零参数、近似 O(1) 计算（纯 reshape）。

UniMixer 确实改变了这一极简主义，将固定置换矩阵替换为可学习的软置换矩阵（双随机矩阵），通过 Kronecker 分解 + 计算流程优化将计算复杂度控制在 O(L^2/B + LB)。当 B ≈ sqrt(L) 时约为 O(L^1.5)，相比原始 TokenMixer 的 O(1) 有实质性开销增加。

这是一个明确的设计取舍：**用有限的计算开销换取可学习性和更强的 scaling 能力**。实验也证实 UniMixer 的实际 FLOPs 普遍高于 RankMixer（2.07T vs 1.68T），UniMixer 放弃了计算效率，换来了 scaling 指数的提升（0.132 vs 0.116）。

### 6.2 Sinkhorn-Knopp 算法

Sinkhorn-Knopp 是一种将任意正矩阵归一化为双随机矩阵（行列和均为 1）的迭代算法，核心操作极为简单：交替对行和列做归一化，反复迭代直到收敛。

UniMixer 中的使用流程：

1. 对称化：(W + W^T) / 2，满足对称性约束
2. 温度缩放：W / tau，tau 越小矩阵越稀疏
3. exp(·) 保证所有元素为正（Sinkhorn 的前提）
4. 交替行列归一化迭代

温度系数 tau 是关键：高温（tau=1.0）矩阵均匀，低温（tau=0.05）矩阵尖锐稀疏，趋近真正的置换矩阵。消融实验显示去掉温度系数导致 AUC 下降 0.1645%，是影响最大的组件。

### 6.3 统一框架（Eq. 7）的本质

论文将所有方法统一为 全局混合 x 局部混合 的两因子结构：

- 局部混合：决定每个 token/block 如何提炼自身内容（类比 Attention 的 V 投影）
- 全局混合 G(X, W\_G)：决定 token 之间的交互强度

三类方法的核心差异在于全局混合是否依赖输入 X：

- TokenMixer：固定置换矩阵，完全不依赖 X，彻底规避异构内积问题
- Attention/FM：动态计算，依赖 X，存在异构内积的语义隐患
- UniMixer-Lite：参数化但不依赖 X，介于两者之间，兼顾可学习性和异构安全性

FM 是 Attention 的特殊退化：令 W\_Q=I, W\_K=I, V=Y（固定），Attention 退化为 FM。

### 6.4 UniMixing-Lite 的参数设计

局部混合的参数化：

- Z（基矩阵集合）：shape (b, B, B)，b 个共享基矩阵，所有 block 公用，是局部交互模式的"词典"
- omega（组合系数）：shape (N, b)，每个 block 独有，决定如何从词典中组合出专属的 W\_B^{\*i}

定义位于 Eq. 8 正文（公式前的文字），W\_B^{\*i} 的计算公式位于 Eq. 8 之后的 where 从句。

全局混合的参数化：A\_G (N, r) 和 B\_G (r, N) 的低秩分解，Sinkhorn 约束后得到 W\_r (N, N)。

### 6.5 局部混合与 Heterogeneous Attention 的关系

两者在数学结构上同族（论文 Eq. 7 中已证等价性），但 UniMixing-Lite 是轻量受限版本：

- 视野更窄：只在 block 内部（维度 B）做投影，而 Attention 的 V 投影作用在完整 token（维度 D），跨维度信息流动更丰富
- 自由度更低：W\_B^{\*i} 被约束为双随机矩阵（行列和为 1，接近置换），不能做缩放和降维；Attention 的 W\_V 是完全自由的实数矩阵

这是论文用来换取参数效率的代价。

### 6.6 TokenMixer 在代码中的体现

TokenMixer 思想体现在 Step 4 的全局混合环节（W\_r @ H）。原始 TokenMixer 是 UniMixing-Lite 全局混合的特殊退化情形：

- 令 W\_B^{\*i} = I（局部矩阵退化为单位矩阵，H = X\_blocks）
- 令 W\_r 为固定置换矩阵（不学习）

满足这两个条件，Step 4 就退化为纯粹的 TokenMixer shuffle 操作。

### 6.7 Feature Tokenization 中 $N$ 的含义

$N$ 是**特征域（feature domains）的数量**。在 Tokenization 前，$F$ 个原始特征先按语义归属分组为 $N$ 个不相交的特征域（User Profile / Item Features / Behavior Sequence / Query Features 等），每个域独立 Embedding 后拼接为长向量 $\mathbf{E} = \[\mathbf{e}\_1, ..., \mathbf{e}\_N]$。

$N$ 远小于 $F$（原始特征数），也远小于 $T$（最终 Token 数），仅作为中间分组，目的是让同语义域的多个特征共享 Embedding 空间，避免异构特征被强行挤进同一个 Embedding 维度。

### 6.8 Step 1 输出维度 $T \times D$ 的物理含义

- **$T$（Token 数量）**：长向量 $\mathbf{E}$（总长度 $L$）按固定步长 $d$ 均匀切分后的 block 数 $T = L/D$。每个 block 独立投影为一个 Token。与 UniFormer 的关键区别：UniMixer 的 Token 是**匿名**的——不携带显式语义身份，可能包含多个特征域的 Embedding 碎片。
- **$D$（每个 Token 的隐向量维度）**：每个 block 经过专属线性投影 $W\_i^{\text{proj}} \in \mathbb{R}^{D \times d}$ 映射到统一隐空间的维度。$D$ 是架构超参，不直接等于任何原始特征的 Embedding 维度。

### 6.9 同一特征是否跨 Token？（匿名 Token 的设计考量）

**是的，同一个特征（如用户性别）的 Embedding 完全可能出现在两个不同的 Token 中。** 原因在于 Step 1 的均匀切分不考虑语义边界：

```
E = [e_user_gender(64维) | e_user_age(32维) | e_item_id(128维) | ...]
     └── Block 0 ──┘── Block 1 ──┘─── Block 2 ───┘── Block 3 ──┘ ...
         (d=96)          (d=96)         (d=96)          (d=96)
```

这是 TokenMixer 系列方法的**刻意设计**，不是 Bug。UniMixer 的 Token 没有"我是性别特征"这种身份意识，靠后续 UniMixing 的可学习软置换矩阵自动重组这些碎片化的信息。这与 UniFormer 按语义域显式分组的 Tokenization 策略形成鲜明对比：**UniMixer 把信息组织的责任推给可学习模块，而非依赖语义先验。**

### 6.10 UniMixing 中的 $H$ 与两套分块体系

**$H$ 的物理含义**：

$H = \[\boldsymbol{x}\_1 W\_B^1 ;|; \boldsymbol{x}_2 W\_B^2 ;|; ... ;|; \boldsymbol{x}_{L/B} W\_B^{L/B}]$ 是**局部混合的中间结果**——$\text{flatten}(X)$ 被切块后每个块用专属 $W\_B^i$ 做完块内变换，但尚未进行全局混合。

之后全局混合为 $W\_G \cdot \text{reshape}(H, L/B, B)$。

**Step 1 与 Step 2 的两套独立分块体系**：

| <br /> | Step 1 (Tokenization)          | Step 2 (UniMixing)                  |
| ------ | ------------------------------ | ----------------------------------- |
| 切分对象   | 拼接后 Embedding 长向量 $\mathbf{E}$ | $\text{flatten}(X)$（$T$ 个 Token 展平） |
| 总长度    | $\sum d\_{\text{domain}}$      | $L = T \times D$                    |
| 块大小    | $d$（原始 block 维度）               | $B$（局部混合块大小）                        |
| 块数量    | $T = L/D$（Token 数量）            | $L/B$（局部混合粒度）                       |
| 作用     | 线性投影 → Token                   | 块内特征混合 $W\_B^i$                     |
| 超参     | $T$（Token 粒度）                  | $B$（局部交互粒度）                         |

举例：$T=16$, $D=64$, $L=1024$, $B=16$ → Step 2 把 1024 维切成 64 个 block。两者通过 $L = T \times D$ 隐式关联，但 $T$ 和 $B$ 独立可调。

### 6.11 SiameseNorm 的输入输出维度

**全程保持 $\mathbb{R}^{T \times D}$，不增不减。** 双流（$\bar{X}$ 与 $\bar{Y}$）每层独立更新后均保持 $T \times D$，最终融合 $X\_{\text{output}} = \bar{X}\_M + \text{RMSNorm}(\bar{Y}\_M)$ 仍为 $T \times D$。

SiameseNorm 仅改变残差流的组织方式（单流变双流），不改变 Token 数量或维度。$L/B$ 是 UniMixing 内部 $\text{flatten}(X)$ 的计算粒度，与 SiameseNorm 层面的 $T$ 无关。

### 6.12 预测 Head：从 $X\_{\text{output}}$ 到概率

UniMixer 面向标准 CTR/CVR 二分类任务（论文为次日留存预测），输出流程为：

$$X\_{\text{output}} \in \mathbb{R}^{T \times D} \xrightarrow{\text{Flatten}} \mathbb{R}^{TD} \xrightarrow{\text{MLP}} \mathbb{R} \xrightarrow{\text{Sigmoid}} \hat{y} \in \[0,1]$$

$T$ 个匿名 Token 全部展平送入 MLP，由预测 Head 自行从这些"计算碎片"中提取预测信号，不做任何语义汇聚或池化。

### 6.13 UniMixer vs UniFormer 异同对比

两者同出快手，分别解决推荐 Scaling 的不同层次问题，是互补关系而非竞争。

#### 核心定位差异

| <br />   | UniMixer                                 | UniFormer                      |
| -------- | ---------------------------------------- | ------------------------------ |
| **发表**   | NeurIPS 2024                             | KDD 2025                       |
| **定位**   | 统一**特征交互模块**                             | 统一**全模型架构**                    |
| **出发点**  | Attention/TokenMixer/FM 三类方法各自为政 → 模块级统一 | 行为建模/特征交互/多任务三组件独立演进 → 架构级联合扩展 |
| **理论野心** | 数学证明三类方法均为 `全局混合 × 局部混合` 的特殊情形           | 提出"组件级 → 模型级"扩展的范式转变           |
| **问题范围** | 窄：深耕单一模块内部                               | 宽：覆盖推荐完整架构                     |

#### 技术实现对比

| 维度               | UniMixer                               | UniFormer                                        |
| ---------------- | -------------------------------------- | ------------------------------------------------ |
| **Tokenization** | 单层线性投影，无 SwiGLU                        | 语义分组 + SwiGLU（单层）+ RMSNorm                       |
| **Token 语义**     | 匿名（均匀切分，不保留语义身份）                       | 显式语义标签（按特征域分组）                                   |
| **序列/行为建模**      | ❌ 不涉及                                  | ✅ 核心组件（Cross-Attn + S-FFNs + Lazy KV）            |
| **特征交互机制**       | 全局双随机矩阵 × 局部 Block 权重（矩阵乘法，无 QK 内积）    | Cross-Attn + Self-Attn + NS-FFNs（标准 Transformer） |
| **多任务**          | ❌ 单任务                                  | ✅ TIM（Cross-Attn + SA + T-FFNs）                  |
| **FFN**          | Pertoken SwiGLU（每个 token 独立参数）         | NS-FFNs / T-FFNs（per-group / per-task）           |
| **归一化**          | SiameseNorm（双流耦合，Pre-Norm + Post-Norm） | Pre-Norm RMSNorm（单流）                             |
| **深度扩展策略**       | SiameseNorm 解决深层训练不稳定                  | 多视图 FFN 保证各组件均衡扩展                                |

#### 交互机制的根本差异

- **UniMixer**：用可学习双随机矩阵替代 QK 内积，彻底规避异构 token 间内积缺乏语义意义的问题。输出 = $W\_G \cdot (W\_B \cdot X)$，$W\_G$ 不依赖输入 $X$。
- **UniFormer**：底层仍是标准 Attention（QK 内积），通过**语义 Tokenization** 让 Query 有明确物理含义来缓解异构内积问题，但没有根除。

#### 两者的层级关系

```
┌──────────────────────────────────────────────┐
│                  UniFormer                    │
│  ┌──────────┐  ┌──────────────┐  ┌────────┐ │
│  │ 行为建模  │  │  特征交互(FIM) │  │ 多任务  │ │
│  │ 序列CA   │  │  ★UniMixer   │  │ TIM    │ │
│  │ S-FFNs   │  │  可嵌入此处★  │  │ T-FFNs │ │
│  └──────────┘  └──────────────┘  └────────┘ │
└──────────────────────────────────────────────┘
```

UniMixer 的 UniMixing-Lite 模块理论上可替换 UniFormer FIM 中的 Self-Attention + NS-FFN，进一步降低特征交互的计算开销。

#### 架构流程对比

| 阶段          | UniMixer                            | UniFormer                            |
| ----------- | ----------------------------------- | ------------------------------------ |
| Token 数全程   | $T$（恒定）                             | $q \to 2q$（FIM 内膨胀分裂）$\to t$（TIM 恒定） |
| 输出 Token 含义 | $T$ 个匿名计算单元                         | FIM: $2q$ 个特征胶囊；TIM: $t$ 个任务表示       |
| 最终预测        | Flatten $T$ 个 Token → MLP → Sigmoid | $t$ 个任务各自 Head → Sigmoid             |
| 信息组织方式      | 靠学习模块自动重组                           | 靠语义先验显式分组                            |

#### 一句话总结

> **UniMixer** 是推荐 Scaling 的"显微镜"——用数学框架统一特征交互模块内部的三种范式；**UniFormer** 是推荐 Scaling 的"望远镜"——用工程架构统一行为建模、特征交互、多任务三大组件的联合扩展。一个是组件级统一，一个是架构级统一，两者互补。

### 6.14 UniMixer 如何处理序列特征？

**核心结论：UniMixer 本身不直接处理序列特征（行为序列建模），它的定位是一个统一的特征交互模块，而非完整的序列建模架构。**

#### 多角度证据

1. **研究动机中的特征描述**：论文提到推荐系统的异构特征包括"用户画像、物品特征、行为序列、Query特征等"，但 UniMixer 关注的是这些特征被 Embedding 之后的**交互环节**，而非行为序列本身如何建模（如时间衰减、自注意力序列建模等）。
2. **与 UniFormer 的显式对比（6.13节）**：

| 维度          | UniMixer | UniFormer                             |
| ----------- | -------- | ------------------------------------- |
| **序列/行为建模** | ❌ 不涉及    | ✅ 核心组件（Cross-Attn + S-FFNs + Lazy KV） |

这是最直接的证据：**UniMixer 的设计范围不包含序列建模组件**，而行为序列建模是 UniFormer（同团队后续工作，KDD 2025）的三大核心组件之一。

1. **匿名 Tokenization 策略（6.9节）**：UniMixer 对拼接后的 Embedding 长向量做均匀切分，不保留语义身份。行为序列（如果已有独立的序列 Encoder 输出 embedding）会被切成碎片混入匿名 Token 中，**时间顺序信息在 Tokenization 阶段就被碎片化了**，UniMixer 无法利用序列的先后顺序关系。
2. **技术迁移表述（第五节）**：论文提到"UniMixer 模块可扩展到用户行为序列建模和生成式推荐任务"，关键词是"**可扩展到**"——说明当前工作尚未覆盖该场景，只是指出了未来方向。

#### 设计取舍的深层原因

这是一个**刻意的设计取舍**，与 UniMixer 的理论定位有关：

1. **Scope 聚焦**：UniMixer 的核心野心是用数学框架统一 Attention/TokenMixer/FM 三种特征交互范式（Eq. 7 统一公式）。引入序列建模组件会模糊理论焦点。
2. **匿名 Token 哲学**：UniMixer 的设计理念是"不依赖语义先验，把信息组织责任推给可学习模块"。序列的时间顺序是强语义先验，与匿名 Token 哲学矛盾。
3. **层级分工**：推荐架构的合理分工应为：
   - **上游**：独立序列 Encoder（如 UniFormer 的 Cross-Attn + S-FFNs）处理行为序列，输出序列表示
   - **中游**：UniMixer（可嵌入 FIM）处理异构特征（含已编码序列表示）之间的交互
   - **下游**：多任务 Head 或预测 Head

#### 一句话总结

> **UniMixer 不做序列建模——它假设行为序列已被上游 Encoder 编码为静态 Embedding，然后将该 Embedding 与其他异构特征一起碎片化送入匿名 Token 管道，靠可学习软置换矩阵完成特征交互。真正的序列建模留给 UniFormer 等上层架构解决。**

### 6.15 Heterogeneous Attention vs Self-Attention 的本质区别

两者核心差异在于**投影矩阵是否 token-specific**：

| 维度            | Self-Attention                                                                      | Heterogeneous Attention                                                                                                         |
| ------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Local Mixing  | $XW\_V$（所有 token **共享**一组 $W\_V$）                                                   | $X\tilde{W}\_V$（**每个 token 专属** $W\_V^{ih}$）                                                                                    |
| Global Mixing | $\text{softmax}!\left(\frac{(XW\_Q)(XW\_K)^\top}{\sqrt{d}}\right)$（共享 $W\_Q, W\_K$） | $\text{softmax}!\left(\frac{(X\tilde{W}\_Q)(X\tilde{W}\_K)^\top}{\sqrt{d}}\right)$（token-specific $\tilde{W}\_Q, \tilde{W}\_K$） |

**本质差异**：

- Self-Attention 假设**所有 token 来自同质 embedding 空间**（NLP 场景成立），一组 $W\_Q/W\_K/W\_V$ 就能统一处理。
- Heterogeneous Attention 针对**推荐系统的异构特征空间**（用户画像 / 物品 / 行为序列 / Query 来自不同语义域），每个 token 有自己专属的 $W\_Q^{ih}, W\_K^{ih}, W\_V^{ih} \in \mathbb{R}^{D \times d}$，通过 token-specific 投影把异构 token 拉到统一可计算空间。
- 但 Heterogeneous Attention **没有根除**异构内积的隐患——它的 global mixing 仍然依赖 $X$ 计算 $QK^\top$，会导致 attention 权重尖锐稀疏（[UniMixer.tex:34-46](file:///d:\WorkSpace\Awesome-Generative-Recommendation.cache\unimixer_src\sections\UniMixer.tex) Fig. perm\_weight(a) 第 10/15 行）、梯度回传困难。

这也是 UniMixer 出场的动机：把 global mixing 也参数化为**不依赖 X** 的 $W\_G$，彻底规避异构内积。

### 6.16 TokenMixer 等价参数化的正确理解

**原始 TokenMixer**：

- 把每个 token $x\_t \in \mathbb{R}^D$ 切成 $H$ 个 head（每个 $D/H$ 维）
- 第 $h$ 个输出 token $s^h = \text{concat}(x\_1^{(h)}, \dots, x\_T^{(h)})$
- 强制约束 $H = T$（否则输入输出维度不一致）

**关键澄清**：原始 TokenMixer 的 "local mixing" 实际上就是 **identity**（每个 head 直接 concat，不做任何变换）。所以"按列取作为新的特征"更准确的说法是 **"每个 head 直接成为新 token，不做局部投影"**。

**UniMixing 的等价参数化**：

$$W^{\text{perm}}_{TD \times TD} = G_{T^2 \times T^2} \otimes I\_{(D/T) \times (D/T)}$$

- $G$ 是 global permutation（决定 block 间如何重组）
- $I$ 是 identity（block 内不变换）

**UniMixing 推广后的形式**：

- 把 $\text{flatten}(X) \in \mathbb{R}^L$ 切成 $L/B$ 个 block，每个 block 维度 $B$
- 每个 block 有自己专属的 $W\_B^i \in \mathbb{R}^{B \times B}$
- 同一个 block 内的 $B$ 个维度共享同一个 $W\_B^i$，不同 block 用不同 $W\_B^i$
- "维度更小的置换矩阵" = Kronecker 分解：$W^{\text{perm}}$（$L \times L$）分解为 $W\_G$（$L/B \times L/B$）与 ${W\_B^i}\_{i=1}^{L/B}$（每个 $B \times B$）
- 参数量从 $O(L^2)$ 降到 $O((L/B)^2 + (L/B) \cdot B^2)$

**与原始 TokenMixer 的对应**：

- 原始 TokenMixer 是 UniMixing 的特殊退化：$B = D/T$（强制）、$W\_G = G$（固定）、$W\_B^i = I$（identity）
- UniMixing 把 $B$ 解耦为**自由超参**，并让 $W\_G$ 和 $W\_B^i$ 都可学习

### 6.17 UniMixing 为什么用 $L$ 而不用 $D$？

$$
L = D x T
$$
**$D$ 是 token 级的二级结构维度，$L$ 是 flatten 后的一级维度**。UniMixing 操作的是置换矩阵，置换矩阵的维度是 $L$，所以必须用 $L$。

### 6.18 Eq.14 vs Eq.17 — 谁是 General Formulation？

#### (1) 谁是 General Formulation？

严格来说，**Eq.17 才是论文里"General Formulation"**，Eq.14 是 Eq.17 的**特殊情形**：

- Eq.17：$G(X, W\_G)$ 允许依赖 $X$（比如 attention 的 $\text{softmax}(QK^\top)$）
- Eq.14：$G(X, W\_G) = W\_G$（**不依赖 X**，是参数化但固定的）

UniMixer 的设计选择是 Eq.14 这一支（global mixing 不依赖 X），但理论框架是 Eq.17——所以 Eq.17 才能统一 Attention（G 依赖 X）、TokenMixer（G 固定）、FM（G 依赖 X）三种方法。

#### (2) 改写方向

**Eq.17 → Eq.14**（不是反过来）：

- Eq.17 是更一般的形式（$G$ 可以是 $X$ 的函数）
- 把 $G(X, W\_G)$ 替换为 $W\_G$（不依赖 X），就得到 Eq.14
- 反过来 Eq.14 推广到 Eq.17 才是"改写为更一般形式"

但整体思路对：unit 内部先做 local projection（$x\_i W\_B^i$），再乘 global mixing（$W\_G$ 或 $G(X, W\_G)$），且 $G$ 可以是带输入的模型（如 attention 的 softmax(QK^T)）。

#### (3) 训练时 $W\_B$ 不是"可选添加"，而是默认可学习

在 Eq.14（UniMixing）中，**$W\_G$ 和 $W\_B^i$ 都是默认可学习的参数**，两者都通过 Sinkhorn-Knopp 约束：

$$\bar{W}\_G = \text{Sinkhorn-Knopp}!\left(\frac{(W\_G + W\_G^\top)/2}{\tau}\right), \quad \bar{W}\_B^i = \text{Sinkhorn-Knopp}!\left(\frac{(W\_B^i + W\_B^{i\top})/2}{\tau}\right)$$

训练时联合优化 $W\_G$ 和 ${W\_B^i}$，**不存在"local mixing 可选加入"的设定**。代码实现中 `W_B_star` 和 `W_r` 都是从 Parameter 计算出来的，都参与反向传播。

消融实验（[Experiments.tex:99](file:///d:\WorkSpace\Awesome-Generative-Recommendation.cache\unimixer_src\sections\Experiments.tex)）显示去掉 block-specific local mixing 权重会掉 0.0436% AUC，证明它确实是有贡献的可学习组件。

### 6.19 UniMixing-Lite 的 Low-rank 与 Basis 设计

#### 结构

- **$W\_G$ 用 low-rank**：$W\_r = \text{Sinkhorn-Knopp}(A\_G \cdot B\_G)$，$A\_G \in \mathbb{R}^{(L/B) \times r}$，$B\_G \in \mathbb{R}^{r \times (L/B)}$
- **$W\_B$ 用共享 basis + 各项线性组合**：$W\_B^{\*i} = \text{Sinkhorn-Knopp}!\left(\sum\_{\ell=1}^b \omega\_\ell^i Z\_\ell\right)$，$Z\_\ell$ 是 $b$ 个共享基矩阵，$\omega^i$ 是每个 block 专属的组合系数

#### 为什么 $W\_B$ 也要 Sinkhorn-Knopp？

3 个理由：

1. **保持 $W^{\text{perm}}$ 的整体性质**：UniMixing 的 $W^{\text{perm}} = W\_G \otimes {W\_B^i}$ 是广义 Kronecker 积。要让这个重构出来的大矩阵满足双随机性、稀疏性、对称性，**每个 $W\_B^i$ 本身也必须满足这些性质**（原始 TokenMixer 里 local 部分是 identity，天然满足，所以不需要约束；换成可学习的 $W\_B^i$ 后必须显式约束）。
2. **保持"接近置换矩阵"的设计哲学**：UniMixer 的核心思想是"把 rule-based 的置换矩阵参数化"，不是改成自由矩阵。$W\_B^i$ 如果不约束，就退化成普通的 per-block 线性层，失去了"软置换"的归纳偏置。消融实验显示温度系数（控制稀疏性）影响最大（-0.1645%），证明稀疏性约束至关重要。
3. **与 Heterogeneous Attention 的"等价"需要**： 论证 UniMixer local mixing 等价于 attention 的 $V$ 投影（在 $W\_V^i = W\_B^i$ 且维度一致时）。但 $W\_B^i$ 多了双随机约束，这是论文用来换取参数效率的代价。

#### Low-rank vs Basis+线性组合 的优劣

**Low-rank（用于 $W\_G$）**：

- **针对的冗余类型**：**单个大矩阵内部**的低秩结构
- **参数**：从 $N^2$ 降到 $2Nr$（$N = L/B$，比如 $N=128$ 时 $N^2=16384$ → $2 \cdot 128 \cdot 16 = 4096$）
- **优点**：全局交互模式（哪些 block 之间互动强）通常确实是低秩的（类似 attention 矩阵的 low-rank 现象）
- **缺点**：表达力被 rank $r$ 限制，无法表达秩 $> r$ 的 global pattern

**Basis+线性组合（用于 $W\_B$）**：

- **针对的冗余类型**：**多个小矩阵之间**的相似性（cross-block redundancy）
- **参数**：从 $N \cdot B^2$ 降到 $b \cdot B^2 + N \cdot b$（$N$ 个 $B \times B$ 矩阵 → $b$ 个共享 basis + $N$ 个 $b$ 维系数）
- **优点**：不同 block 共享"交互原语词典"（$Z\_\ell$），只组合系数不同；可解释性强
- **缺点**：表达力被 basis 张成空间限制

**为什么不交叉使用（$W\_G$ 用 basis、$W\_B$ 用 low-rank）？**

- $W\_G$ 只有**一个**矩阵，没有"多个相关矩阵"可共享 basis
- $W\_B$ 有 $N$ 个 $B \times B$ 小矩阵，cross-block 冗余是主要矛盾（小矩阵本身已经小，再做 low-rank 收益有限）

### 6.20 Kimi Delta Attention (KDA) 与 UniMixer 框架的整合分析

> 参考论文：Kimi Linear（arxiv 2510.26692）。
> KDA 是 Gated Delta Rule 的线性注意力变体，引入 channel-wise forget gate $\operatorname{Diag}(\boldsymbol{\alpha}\_t)$ 和 DPLR（Diagonal-Plus-Low-Rank）状态转移矩阵。论文将其定位为「可学习位置编码」（替代 RoPE）。

#### 6.20.1 KDA 的两种等价数学形式

**Recurrent 形式**：

$$\mathbf{S}_t = \bigl(\mathbf{I}-\beta\_t\mathbf{k}_{t}\mathbf{k}_{t}^{\top}\bigr),\operatorname{Diag}!\bigl(\boldsymbol{\alpha}t\bigr),\mathbf{S}_{t-1} + \beta\_t\mathbf{k}{t}\mathbf{v}\_{t}^{\top}\in\mathbb{R}^{d\_k\times d\_v},\qquad \mathbf{o}\_t = \mathbf{S}\_t^\top \mathbf{q}\_t$$

其中 $\operatorname{Diag}(\boldsymbol{\alpha}\_t)$ 是 channel-wise（逐维度）遗忘门，$\beta\_t$ 是标量学习率，$(\mathbf{I}-\beta\_t\mathbf{k}\_t\mathbf{k}\_t^\top)$ 是 Householder 式反射（DeltaNet 风格的状态更新）。

**展开后的 parallel 形式**：

$$\mathbf{o}_t = \sum_{i=1}^t \Bigl(\mathbf{q}_t^\top \Bigl(\prod_{j=i+1}^t \operatorname{Diag}(\boldsymbol{\alpha}\_j)\bigl(\mathbf{I}-\beta\_j\mathbf{k}\_j\mathbf{k}\_j^\top\bigr)\Bigr)\mathbf{k}\_j\Bigr)\mathbf{v}\_j$$

**DPLR 等价改写**：KDA 可重写为 $\mathbf{S}\_t = (\mathbf{D} - \mathbf{a}\_t\mathbf{b}_t^\top)\mathbf{S}_{t-1} + \beta\_t\mathbf{k}\_t\mathbf{v}\_t^\top$，其中 $\mathbf{D} = \operatorname{Diag}(\boldsymbol{\alpha}\_t)$、$\mathbf{a}\_t = \beta\_t\mathbf{k}\_t$、$\mathbf{b}\_t = \mathbf{k}\_t \odot \boldsymbol{\alpha}\_t$。即「对角阵 + rank-1 修正」的 DPLR 结构，但通过 $\mathbf{a}=\mathbf{b}=\mathbf{k}$ 的绑定大幅降低 chunkwise 计算开销（相比通用 DPLR 加速约 $2\times$）。

 ### 6.21 KDA 在基本形式下(非DPLR)能直接融入 Eq.17

#### 6.21.1 Eq.17 的 $G(X, W_G)$ 接口天然容纳 KDA

Eq.17 的 $G(X, W_G)$ 接口是「输入 $X$ 与参数 $W_G$，输出 $T\times T$ 矩阵」，**不限制 $G$ 的内部计算方式**——既可以是 $\text{softmax}(\mathbf{Q}\mathbf{K}^\top)$ 这样的 pairwise 内积，也可以是其他任何把 $X$ 映射成矩阵的函数。

把 KDA 的 recurrent 形式展开为 parallel form. 明确给出 recurrent 与 parallel 两种 mathematically equivalent 形式）：

$$\mathbf{o}_t = \sum_{i=1}^t \beta_i\,\Bigl(\mathbf{q}_t^\top \Bigl(\prod_{j=i+1}^t \mathbf{M}_j\Bigr)\mathbf{k}_i\Bigr)\mathbf{v}_i,\qquad \mathbf{M}_j := (\mathbf{I}-\beta_j\mathbf{k}_j\mathbf{k}_j^\top)\operatorname{Diag}(\boldsymbol{\alpha}_j)$$

写成矩阵乘：定义 $\mathbf{G}^{\text{KDA}}(X)\in\mathbb{R}^{T\times T}$（下三角，因果）

$$\mathbf{G}^{\text{KDA}}(X)_{t,i} = \begin{cases}\beta_i\,\mathbf{q}_t^\top\bigl(\prod_{j=i+1}^t \mathbf{M}_j\bigr)\mathbf{k}_i & i \le t \\ 0 & i > t\end{cases}$$

则 $\mathbf{O} = \mathbf{G}^{\text{KDA}}(X)\cdot\mathbf{V}$，**完全符合 Eq.17 的 $G(X, W_G)\cdot\text{Local}$ 接口**。

对齐：
- **Local Mixing** = $XW_V$（per-token V 投影，与 Self-Attention 同族）
- **Global Mixing** = $G^{\text{KDA}}(X, W_G)$，$W_G$ 涵盖 $W_Q, W_K$ 及生成 $\boldsymbol{\alpha}, \beta$ 的参数

**结论：KDA 在基本形式下能直接融入 UniMixer Eq.17，无需扩展 General Formulation。**

#### 6.21.2 之前分析的错误所在

- **6.20.3 障碍 1**（"KDA 是递归状态、不是矩阵，无法纳入 Eq.17"）**错误**——把 KDA 的 recurrent **实现形式**与数学本质混淆了。任何 recurrent 形式都有等价的 parallel 矩阵形式，KDA 论文 Table 1 即明示二者 mathematically equivalent。
- **6.20.4**（"必须把 Eq.17 扩展为状态递归形式才能容纳 KDA"）基于上述错误前提，**亦不成立**。
- **6.20.8 结论**中"(1) 计算图形式不兼容（矩阵 vs 递归状态）"**错误**。正确差异只有两点：(2) 顺序假设（无序 vs 因果）、(3) 设计目标——这两点是**应用层面**的障碍，而非形式层面的不兼容。

#### 6.21.3 KDA 的真正新维度：$G(X)$ 的依赖结构

既然形式上 KDA 能融入 Eq.17，那 KDA 给 UniMixer 设计空间带来的真正新维度是 **$G(X)$ 内部元素 $G_{t,i}$ 的依赖结构**：

| 方法 | $G_{t,i}$ 依赖的输入 |
|---|---|
| Self-Attention / FM | 仅 $(\mathbf{q}_t, \mathbf{k}_i)$ — **pairwise 瞬时** |
| TokenMixer | 不依赖任何 token — **固定** |
| **KDA** | $(\mathbf{q}_t, \mathbf{k}_i)$ + **中间所有 token** $\{\mathbf{k}_j, \boldsymbol{\alpha}_j, \beta_j\}_{j=i+1}^t$ — **cumulative 路径依赖** |

Attention 的 $G=\text{softmax}(\mathbf{Q}\mathbf{K}^\top)$ 可分解为 $\mathbf{Q}$ 与 $\mathbf{K}$ 的外积再逐元素变换；KDA 的 $G^{\text{KDA}}$ **不可如此分解**——$\prod_{j=i+1}^t \mathbf{M}_j$ 把不同位置的 $\mathbf{k}_j$ 通过矩阵乘耦合，$G_{t,i}$ 无法写成 $\mathbf{q}_t$ 与 $\mathbf{k}_i$ 的简单函数。这才是 KDA 在 Eq.17 框架中开拓的真正新象限：**$G(X)$ 从 pairwise 瞬时扩展到 cumulative 路径依赖**。原 Tab_0301 只覆盖"固定 / 瞬时数据依赖"两类，KDA 引入第三类"累积数据依赖"。

#### 6.21.4 状态递归是高效实现，非形式上的必需扩展

- **数学形式上**：KDA 在 parallel form 下就是 $G^{\text{KDA}}(X)\cdot\mathbf{V}$，**无需扩展 Eq.17**。
- **实现效率上**：KDA 论文的核心贡献是证明这个 $T\times T$ 矩阵 $G^{\text{KDA}}(X)$ **无需显式构造**——可通过 bounded recurrent state $\mathbf{S}_t\in\mathbb{R}^{d_k\times d_v}$ 以 $O(T d_k d_v)$ 而非 $O(T^2 d_k)$ 的成本计算。

也就是说，"bounded state"是 KDA **实现** $G^{\text{KDA}}$ 的高效算法，而不是它在数学形式上跳出 Eq.17 的证据。论文 Table 1 把 recurrent 与 parallel 形式列为 mathematically equivalent 正说明这一点。状态递归视角可视为按"如何实现 $G(X)$"对 Eq.17 方法做的**计算复杂度分类**（显式构造 $T\times T$ / 固定矩阵 / bounded state 递归），而非数学形式的扩展。

#### 6.21.5 扩展的 Tab_0301（按 $G$ 依赖结构 + 实现方式分类）

| 方法 | Local Mixing | Global Mixing $G(X,W_G)$ | $G_{t,i}$ 依赖结构 | $G$ 的实现方式 |
|---|---|---|---|---|
| Self-Attention | $XW_V$ | $\text{softmax}((XW_Q)(XW_K)^\top/\sqrt{d})$ | pairwise 瞬时 | 显式 $T\times T$ 矩阵 |
| Heterogeneous Attention | $X\tilde{W}_V$ | $\text{softmax}((X\tilde{W}_Q)(X\tilde{W}_K)^\top/\sqrt{d})$ | pairwise 瞬时 | 显式 $T\times T$ 矩阵 |
| TokenMixer | $X$ | $G$（固定置换）| 与 X 无关 | 固定矩阵 |
| FM | $Y$ | $XI(XI)^\top$ | pairwise 瞬时 | 显式 $T\times T$ 矩阵 |
| **KDA** | $XW_V$（per-token KV）| $\beta_i\,\mathbf{q}_t^\top(\prod_{j>i}\mathbf{M}_j)\mathbf{k}_i$（下三角，累积）| **cumulative 路径依赖** | bounded state 递归 |

读法：KDA 在 Local Mixing 维度与 Self-Attention 同族；在 Global Mixing 维度开拓了「累积路径依赖」新象限；在 $G$ 的实现方式上以 bounded state 递归替代显式 $T\times T$ 矩阵构造，复杂度从 $O(T^2)$ 降到 $O(T)$。

#### 6.21.6 仍成立的限制（应用层面，非形式层面）

形式上能融入 ≠ 应用上能直接用。唯一仍成立的根本限制是**因果性失配**：KDA 的 $G^{\text{KDA}}$ 严格下三角（causal），而 UniMixer 特征无序——这是**应用层面**障碍，不影响 Eq.17 **形式上**的容纳能力。落地需"无序化改造"（双向扫描或随机置换）。（6.20.7 限制表中"框架不兼容"一行应删除，"数值稳定性"是次要工程问题，非根本障碍。）

#### 6.21.7 实现要点（unimixer_lite.py，reduce_method="kda"）

已在 [unimixer_lite.py](file:///d:\WorkSpace\Awesome-Generative-Recommendation\WorkSpace\unimixer_lite.py) 中实现 KDA 作为第 5 种 `reduce_method`（与 self_attention / tokenmixer / wukong 并列），**不识别 DPLR**——用 naive parallel 双层循环显式构造 $G^{\text{KDA}}$（$O(T^2 D^2)$，可读性优先，不实现论文 `chunk_kda` 的 chunkwise 优化）。Local Mixing = $W_V$，Global Mixing = $G^{\text{KDA}}(X)$，forward 自动走 `W_G @ (X W_V)` 即 $\mathbf{G}^{\text{KDA}}\cdot\mathbf{V}$，完全对齐 Eq.17。

实现中发现论文 Table 1 未显式标注、但展开 recurrent 时必需的三个细节：

1. **$\beta_i$ 因子**：$G^{\text{KDA}}[t,i] = \beta_i\,\mathbf{q}_t^\top(\prod_{j=i+1}^t \mathbf{M}_j)\mathbf{k}_i$。论文 Table 1 caption 注明 omitted $\beta_t$ for brevity，但 recurrent 的 write term $\beta_t\mathbf{k}_t\mathbf{v}_t^\top$ 带 $\beta_t$，展开后 $\mathbf{v}_i$ 项必然带 $\beta_i$。漏掉会导致 parallel 与 recurrent 数值不一致（验证脚本测得相对误差 100%，FAIL）。

2. **$\mathbf{M}_j$ 因子顺序**：$\mathbf{M}_j = (\mathbf{I}-\beta_j\mathbf{k}_j\mathbf{k}_j^\top)\operatorname{Diag}(\boldsymbol{\alpha}_j)$——先 $\operatorname{Diag}(\boldsymbol{\alpha})$（channel-wise 衰减）后 Householder 反射（沿 $\mathbf{k}$ 写入/擦除），与 Eq.1 字面顺序一致。顺序反了同样不一致。

3. **key L2 normalize**：保证 $(\mathbf{I}-\beta\mathbf{k}\mathbf{k}^\top)$ 是良性反射（谱半径 $\le 1$），结合 $\boldsymbol{\alpha}\in(0,1)$ 保证状态稳定。

**数值验证**（[verify_kda.py](file:///d:\WorkSpace\Awesome-Generative-Recommendation\WorkSpace\verify_kda.py)）：parallel $\mathbf{G}^{\text{KDA}}\cdot\mathbf{V}$ 与 recurrent Eq.1 直接计算 $\mathbf{o}_t=\mathbf{S}_t^\top\mathbf{q}_t$ 的 max diff = $1.16\times10^{-10}$（机器精度），**PASS**。验证了 6.21.1 的论断：KDA parallel form 完全符合 Eq.17 的 $G(X, W_G)\cdot\text{Local}$ 接口。

#### 6.21.8 blocking 视角与 attention / FM / TokenMixer 的等价性

UniMixer 的 blocking（block_size B）是一个"分辨率/视角"选择，同一 L=T·D 维输入在不同 B 下对应不同方法。这是 Tab_0301 未显式列出但隐含的**第三设计维度——blocking 视角**。

**B = D（block = 一个 token）**：
- n_blocks = T，每 block 是 D 维 token
- Local W_B = W_V（D×D，所有 token 共享同一投影）→ 与标准 attention 的 per-token QKV 一致
- Global W_G = softmax(QK^T)（T×T，inter-token）→ 标准 self-attention
- **等价于单头 self-attention**（所有 token 共享 QKV 投影矩阵）
- FM 同理：Local = Y（D×D 固定），Global = XX^T（T×T）→ 标准 FM

**B = T（block = 一个 feature 跨所有 token）**：
- n_blocks = D，每 block 是 T 维（一个 feature 在 T 个 token 上的取值）
- Local = identity（T×T，feature block 内不变）
- Global = 固定置换（D×D，feature 间）
- 这是 TokenMixer 的自然视角

**TokenMixer 的"转置"本质**：核心操作是 (T,D)↔(D,T) 转置，即一个 L×L 置换矩阵 $W_{\text{transpose}}$。在 UniMixer 框架下，这等价于"切换 blocking 视角"——从 B=D（token 视角，attention 自然）切到 B=T（feature 视角，TokenMixer 自然）。attention 看 T×T 的 inter-token 交互，TokenMixer 转 90° 看 D×D 的 inter-feature 交互。所以 attention 与 TokenMixer 的区别不仅是 Local/Global 形式（$W_V$/identity, softmax/固定置换），还有 **blocking 视角**（B=D vs B=T）。

**框架细节澄清**：UniMixer 的 forward **固定一个 block_size B**。attention 用 B=D，TokenMixer 用 B=T，二者 B 不同。为在同一 B 下统一比较，[unimixer_lite.py](file:///d:\WorkSpace\Awesome-Generative-Recommendation\WorkSpace\unimixer_lite.py) 固定 B，所有 reduce_method 共用——此时 TokenMixer 简化为"在 B=D 下做 T 个 block 间 swap 置换"（[compute_W_G](file:///d:\WorkSpace\Awesome-Generative-Recommendation\WorkSpace\unimixer_lite.py) 的 tokenmixer 分支），而非完整 L×L 转置。真正的 TokenMixer 转置是跨 blocking 视角的 L×L 操作；代码里是教学简化版。

**KDA 的 blocking 选择**：KDA 在 B=D 下最自然——每 block = 一个 token，$G^{\text{KDA}}$ 是 T×T inter-token 累积混合，与 KDA 论文（token 序列）一致。若 B≠D，KDA 的"token"变成 block，因果性作用于 block 间，语义偏离原论文。

**核心洞察**：UniMixer 的统一性来自把"token 级操作"抽象为"block 级操作"，通过 B 切换分辨率。B=D 时 block=token（attention/FM/KDA 自然），B=T 时 block=feature（TokenMixer 自然），B=其他时是混合粒度（UniMixing-Lite 的 basis 组合）。

#### 6.21.9 TokenMixer 两次 reshape 与 Eq.14 的对照

TokenMixer-Large 用两次对称 reshape 维持维度循环，与 UniMixer Eq.14 的 reshape 形成机制对照（详见 [TokenMixer summary 五]

**TokenMixer-Large 的两次 reshape**：
- Mixing: (T,D) → split → (T,H,D/H) → concat → (H, T·D/H)  # 转置 token↔feature
- Reverting: (H, T·D/H) → split → (T,H,D/H) → 重组 → (T,D)  # 反转置还原

**UniMixer Eq.14 的 reshape**：
- H = [x_1 W_B^1; ...; x_{L/B} W_B^{L/B}]  # 局部混合结果（L 维）
- 全局混合 = W_G · reshape(H, L/B, B)  # reshape 成 (L/B, B) 让 W_G 作用
- → reshape 回 L 维  # 还原，残差成立

**结构对照**（两者都靠 reshape 维持维度循环）：

| 步骤 | TokenMixer-Large | UniMixer Eq.14 |
|---|---|---|
| 组织形状以便 mixing | Mixing reshape (T,D)→(H,T·D/H) | reshape H→(L/B, B) |
| mixing 操作 | **reshape 即 mixing**（转置=信息交换）| W_G 矩阵乘（reshape 只是辅助）|
| 还原到输入维度 | Reverting reshape →(T,D) | reshape 回 L 维 |
| 目的 | 残差 X+output 维度一致 | 残差 X+output 维度一致 |

**本质区别**（核心对照点）：

- **TokenMixer**：reshape **即 mixing**——转置让信息跨 token 流动，**零参数、零矩阵乘**，纯形状重组完成信息交换。这是 TokenMixer-Large "O(1) 计算"的根源。
- **UniMixer**：reshape 只是**形状辅助**，真正的 mixing 是可学习矩阵 W_G·W_B。reshape 让矩阵乘能正确作用在 block 维度。

对照结论：TokenMixer 用 reshape 做 mixing（极简、零参数），UniMixer 用 reshape 配合可学习矩阵做 mixing（更强但更贵）。这正是 [6.1 节](file:///d:\WorkSpace\Awesome-Generative-Recommendation\Summary\Ranking\summary_unimixer_rec_scaling.md)说的"UniMixer 放弃 TokenMixer 的 O(1) 计算效率，换来可学习性与 scaling"。

**TokenMixer-Large 在 token 交互上的创新定位**（除 Global Token 外，不含 Pertoken SwiGLU 单 token 计算）：

诚实地说，TokenMixer-Large 在 token 交互上的创新主要是**结构性**的，而非**机制性**的——mixing 底层操作仍是 reshape 转置（与 RankMixer 相同），没有引入新的 mixing 算子。创新集中在"如何组织 reshape 使残差稳定"：

| 创新 | 内容 | 性质 |
|---|---|---|
| Mixing & Reverting 对称两步 | 单次 Mixing 改为对称 Mixing+Reverting，保证 T→H→T 维度循环 + 残差语义对齐（OTR/TSA）| 核心，结构性 |
| Global Token | BERT [CLS] 式全局聚合 token，提供全局交互锚点 | 辅助 |
| Inter-Residual 间隔残差 | 每 2~3 层跨层残差，增强低层→高层 token 信号传输 | 辅助，跨层 |
| Auxiliary Loss | 低层 token 输出 logits 参与高层联合训练 | 辅助，训练策略 |

**关键观察**：TokenMixer-Large 明确指出"只要每个新 Token 包含所有原始 Token 的信息，具体切分方式（垂直/对角/随机）不影响效果"——这说明 mixing 机制本身被设计成无关紧要，重要的是信息完整性 + 残差结构。TokenMixer-Large 的重心在工程结构（残差、归一化、MoE、并行），而非 mixing 算法创新。

**与 UniMixer 的方向对比**：

- TokenMixer-Large：保持 reshape 极简 mixing，创新在如何组织 reshape + 残差结构使深层可训（工程极致）
- UniMixer：把 reshape 升级为可学习软置换矩阵 W_G·W_B，创新在 mixing 机制本身（数学统一 attention/FM/TokenMixer）

两者方向相反：TokenMixer-Large 工程极致优化（保 reshape），UniMixer 数学理论统一（替 reshape）。

#### 6.21.10 修正后的结论

**KDA 在基本形式下能直接融入 UniMixer Eq.17 的 $G(X, W_G)$ 接口**，无需扩展 General Formulation。KDA 给 UniMixer 设计空间带来的真正扩展是 **$G(X)$ 的依赖结构从 pairwise 瞬时扩展到 cumulative 路径依赖**，以及在**实现层面**提供 bounded state 递归这一 $O(T)$ 复杂度的 $G$ 计算路径。6.20.6 的三个落地启发（DPLR 式 $W_G$、channel-wise local gating、bounded state 全局混合）仍然有效，但其前提不再是"必须扩展 Eq.17"，而是"在 Eq.17 框架内借鉴 KDA 的 $G$ 依赖结构与实现技巧"。