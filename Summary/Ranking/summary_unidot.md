# UniDot：大规模推荐中统一序列建模与特征交互的网络

> **论文**: *UniDot: A Unified Network for Sequence Modeling and Feature Interaction in Large-scale Recommendation*
> **来源**: arXiv:2608.16797, KDD Cup 2026 Tencent UniRec Challenge Workshop (Jeju, Korea, 2026.08.12)
> **作者**: Rongcheng Lin, Yan Sun, Jamey Zhang, Guanglei Xiong, Ivan Ji, Xianjie Chen, Shujian Bu（Meta）
> **成绩**: TAAC × KDD Cup 2026 工业赛道 **亚军**，AUC = **0.83217**

---

## 1. 背景与动机

工业推荐系统的预测模型长期沿着两条**独立演进**的路线发展：

1. **特征交互模型**（Feature Interaction）：针对多域用户/物品静态特征，学习显式与隐式的特征交叉。代表：Wide&Deep、DeepFM、xDeepFM、DCN-v2、FiBiNet、**Wukong**、DHEN。
2. **序列用户兴趣模型**（Sequential Modeling）：建模用户行为的动态演化，通常使用 target-aware attention 对单条行为历史进行聚合。代表：DIN、DIEN、SIM、ETA、TWIN、LONGER、TIN。

现有生产系统只是松散地把两者串联（先序列模块 → 后交互模块），没有真正**统一**。TAAC × KDD Cup 2026（Tencent Uni-Rec Challenge）正是瞄准这一鸿沟，要求：

- 一个**统一的 Token 化方案**
- 一个**同构、可堆叠的 Backbone**
- 在推理延迟预算下按 AUC 排名

### 核心洞察：FM 内积 = Attention Q·K 点积

UniDot 的名字来自 **Uni**fied modeling via **D**ot-products **O**f **T**okens——作者从**因子分解机（FM）视角**出发：

FM 的用户-物品嵌入内积 $\langle v_u, v_i \rangle$ 是协同过滤的引擎（让模型泛化到未见的 u-i 对），而 Attention 的 Q·K 打分**本质上也是同一个点积运算**。因此，**一个统一的 Token 点积**可以同时支撑特征交互与序列建模。

与现有统一方法（把内积变成深度交互栈的隐式产物）不同，UniDot **显式保留**跨用户-物品、候选-历史边界的点积信号，用更深的机制（Token Mixing、门控 MLP、Attention）去**优化点积的操作数**，而非替换点积本身。

---

## 2. 整体架构

UniDot 由 4 个关键模块组成：

```
[输入]                               [计算]                         [输出]
┌──────────────────────┐     ┌───────────────────────────────┐
│ user/item profile    │────▶│ Token 化 → U, I, Ih           │
│ 多域行为序列 S1-S4   │────▶│ Seq Pipeline (embed once) → H │
└──────────────────────┘     └───────────────┬───────────────┘
                                             │
                          ┌──────────────────▼──────────────────┐
                          │         Macro-Block × L 层          │
                          │  ┌───────────────────────────────┐  │
                          │  │  Token-Mixing Bus (Wukong)    │  │
                          │  │        ║ FuseFFN (MLP-Mixer)  │  │
                          │  │  Sequence-Retrieval Bus       │  │
                          │  │   (Ih cross-attends H)        │  │
                          │  └──────┬───────────────┬────────┘  │
                          │         │ FM Highway    │           │
                          │         └───────┬───────┘           │
                          └─────────────────┼───────────────────┘
                                            │
                          ┌─────────────────▼───────────────────┐
                          │  Classifier: ρ + [φ¹..φᴸ] + e_skip  │
                          │    MLP → ŷ + aux delay head         │
                          └─────────────────────────────────────┘
```

**四项核心贡献**：

| # | 贡献 | 说明 |
|---|------|------|
| (1) | **双路径、每层并行的 Macro-Block** | Token-Mixing Bus 与 Sequence-Retrieval Bus **并行**运行，每层通过 MLP-Mixer 交换状态，使 profile 与序列信号**共进化** |
| (2) | **共享、候选感知、多域序列流水线** | 所有行为域 **只嵌入一次**并通过位置局部 fid-轴压缩到统一宽度；DIN 式条件门控 SwiGLU 注入候选上下文；时间戳交错的合并流建模跨域时序 |
| (3) | **FM Highway（显式特征交互）** | 每层的 per-域 Q-K 点积、聚合 Gram 矩阵、跨 Bus User-Item 点积**直接拼接**送入分类器，绕过融合残差路径，保留 FM 式二阶信号 |
| (4) | **共享嵌入的多路径互学习（DML）** | 两条 UniDot 路径共用稀疏嵌入表，互相蒸馏预测，收敛到更平坦的泛化最优点；推理时平均 logits，或仅用单路径（1× 成本） |

---

## 3. 符号与问题定义

任务：**点击后转化率预测（Post-click CVR）**，输入 $x = (u, i, \mathcal{S})$，二分类标签 $y \in \{0,1\}$。

**(1) Token 化**：
$$
\begin{aligned}
\mathbf{U} &= \mathrm{Tok}_u(u) \in \mathbb{R}^{T_u \times d} \\
\mathbf{I} &= \mathrm{Tok}_i(i) \in \mathbb{R}^{T_i \times d} \\
\mathbf{I}_h &= \mathrm{Tok}_i^{h}(i) \in \mathbb{R}^{T_{ih} \times d} \quad \text{(序列检索 Bus 的查询 Token)} \\
\mathbf{H}^{(s)} &= \mathrm{Seq}(S_s) \in \mathbb{R}^{L_s \times d} \quad \text{(第 s 个行为域的嵌入视图)}
\end{aligned}
$$

**(2) 可堆叠 Block**：初始化 $Z^{0}_{\text{mix}} = [\mathbf{U};\mathbf{I}]$，$Z^{0}_{\text{seq}} = \mathbf{I}_h$，对 $\ell = 1..L$：
$$
(Z^{\ell}_{\text{mix}}, Z^{\ell}_{\text{seq}}, \phi^{\ell}) = \mathrm{Block}_\ell(Z^{\ell-1}_{\text{mix}}, Z^{\ell-1}_{\text{seq}}, \{\mathbf{H}^{(s)}\})
$$
其中 $\phi^\ell$ 是该层的 FM Highway 信号。

**(3) 读出**：
$$
\hat{y} = \sigma\Big(\mathrm{MLP}\big[\, \rho(Z^{L}_{\text{mix}},Z^{L}_{\text{seq}});\; \phi^{1};\dots;\phi^{L};\; e_{\text{skip}} \,\big]\Big)
$$
注意：$\phi^\ell$ 是**拼接而非求和**，保证每层显式交互不被稀释。

**(4) 目标函数**：
$$
\mathcal{L} = \underbrace{-y\log\hat{y}-(1-y)\log(1-\hat{y})}_{\text{BCE}} + \lambda\,\underbrace{\mathcal{L}_{\text{delay}}}_{\text{aux 延迟头 MSE}}
$$

---

## 4. 方法详解

### 4.1 Token 化（§3.2）

把异构输入映射到**统一的 $d$-维 Token 空间**。两个压缩原语贯穿全文：

| 原语 | 全称 | 结构 | 作用 |
|------|------|------|------|
| **LCB** | Linear Compression Block | 单层 token-轴 Linear + LayerNorm | 压缩 Token 维度 |
| **NCB** | Nonlinear Compression Block | 两层 Linear + GELU + LayerNorm | 非线性压缩 Token 维度 |

**(a) 类别型（fid）特征**：每个 fid 按位置嵌入，位置束沿 Token 轴用一个学习到的 NCB 压缩到 Token 预算。因为 NCB 是**可学习**的，由数据决定哪些特征组合形成每个 Token（而不是固定的平均池化）。
- 产出 User Tokens $\mathbf{U}$ ($T_u$)
- 两个独立 Item 表：紧凑视图 $\mathbf{I}$ ($T_i$, Mix Bus) 与更丰富的 $\mathbf{I}_h$ ($T_{ih}$, Seq Bus, 驱动 Cross-Attention 查询)

**(b) 多值 fid：FAFE（Field-Aware Feature Embedding）**：少数高价值多值 fid（行为 ID 列表，**位置不变**）使用 DIN 式**候选感知注意力池化**而非静态 NCB——该字段的 Token 对每个候选物品是不同的组合。

**(c) 预训练嵌入特征**：稠密预训练向量（SUM、LMF4Ads 等）先归一化，再用小型 MLP 投影为额外 Token，追加到匹配的 Token 集中。
- LMF4Ads (320-d) 的特殊处理：结构化为 10 个独立 L2-normalized 的 32-d 子向量，分别用共享 Linear 提升为 Token → 掩码零填充槽 → LCB 压缩 → LayerNorm。

### 4.2 序列编码器（§3.3）

序列流水线**每个 forward 只运行一次**，产出被所有消费者共享的视图 $\mathbf{H}^{(s)}$，约束推理延迟。四个阶段：

```
行为事件 → [fid-轴 NCB] → [Depthwise Conv1d] → [CondGatedSwiGLU] → [Causal Transformer (RoPE)] → View Projectors → H(s)
                        ↘ (合并流) 交错时间戳 ↗
```

1. **跨域合并流（Merged Cross-domain Stream）**：子集域按时间戳交错成一条额外的跨域流，下游更容易学习跨域交互。最终 $S = 4 + 1 = 5$ 个序列。
2. **Depthwise Conv1d（局部 N-gram）**：轻量深度卷积，在短窗口上按通道混合，廉价捕获**爆发、相邻事件模式**等 N-gram 结构，不必让 self-attention 重新学习。
3. **DIN 式条件 SwiGLU（CondGatedSwiGLU）**：复合 cond 向量（user + item LCB tokens + emb-cond → NCB 融合）**只进入 SwiGLU 的 Gate 支路**，Value 支路保持纯序列内容。Gate 按候选相关性选择位置（DIN 风格），但不改变位置的内容本身。
4. **Causal Transformer + RoPE**：浅层 Transformer（1 层，4 head），带**因果窗口注意力**（window $w=128$），按时间顺序建模 Token 依赖。Per-consumer **View Projector**（Linear→LN→GELU）输出 $D$-dim 视图 $\mathbf{H}^{(s)}$，每 view-stride 层刷新一次。

### 4.3 每层计算（§3.4）

每个 Macro-Layer 并行运行两条 Bus，然后融合。

#### 4.3.1 Token-Mixing Bus

Bus 状态（User + Item + Pooled Tokens 拼接）通过 $W$ 个跨 Token Block。Block 是**可交换插槽**，默认用 **Wukong**（并行 LCB + FMB，FMB 提供显式点积交互）。作者也在该插槽测试了 TokenMixer 和 UniMixer，但在本数据规模下均未超过 Wukong（可能需要更多数据收敛）。

Pooled Tokens 每层之间会被**切掉**（不写回），保持持久状态为 $(T_u + T_i, D)$。

#### 4.3.2 MultiChannelSeqPool（多通道序列池化）

灵感来自 NetVLAD / NeXtVLAD 的多聚类软分配。对每个序列视图：

- $C$ 个通道，每个通道用**独立的 sigmoid 门**（不是 softmax）对所有位置打分
- 门控权重由当前 Mixing-Bus 状态的摘要**条件化** → 各通道独立激发，不竞争固定注意力质量
- 每个池化 Token = 门控加权求和位置 → **ℓ₂ 归一化**（sigmoid 无界）→ 拼接 $S \cdot C$ 个 Token → LayerNorm（与 NCB-tailed profile Token 尺度匹配）

```
位置 h_ℓ ──→ 每通道 sigmoid 门 w_{ℓ,c} = σ(MLP[h_ℓ; summary(Z_mix)]) ──→ Σ w·h ──→ ℓ₂ norm ──→ p_c
                                                                                                                                                                                                                                                 ──→ Concat(S·C) ──→ LN
```

#### 4.3.3 Sequence-Retrieval Bus

单层单历史等价于标准 target-attention CTR 模块；这里扩展到 $S$ 域、每层查询、FM-Highway 读出。

步骤：
1. **Per-序列 Cross-Attention**：Item 状态 $T_{ih}$ 个 Token 作为 Query，分别对 $S$ 个序列视图做 Cross-Attention → $A_0..A_{S-1} \in \mathbb{R}^{B\times T_{ih} \times D}$
2. **LCB 聚合**：$\text{attn\_agg} = \text{LCB}(\text{stack}(A_0..))$，线性聚合（融合 FFN 已有非线性）
3. **Per-Token 融合 FFN**：小型 FFN 将每个 Item Token 与从 $S$ 个序列检索到的向量混合 → 对 Item 状态做 $D$-维残差增量
4. **FM Highway 信号**（绕过融合，跨层拼接）：
   - **Per-序列点积** $d_i[t] = \langle \text{item}[t], A_i[t] \rangle$：候选与每个行为域的 per-Token 亲和力
   - **融合域点积**：NCB 跨 $S$ 输出，再与 Item 状态点积
   - **聚合 Gram** $G = \text{item} \cdot \text{attn\_agg}^\top$：完整的成对点积矩阵
   - **User-Item 交叉点积**：Mix Bus 的 User Tokens 与 Ret Bus 的 Item Tokens 内积
   - 四者拼接形成 $\phi^\ell$

#### 4.3.4 FuseFFN（Bus 级融合：规范 MLP-Mixer）

三组分 Token 拼接：$w_{\text{out}}$（Mix Bus 输出，$T_u{+}T_i$）| pooled（池化 Token，$S{\cdot}C$）| $h_{\text{out}}$（Seq Bus 输出，$T_{ih}$）

```
Concat → NCB Token-Mix (跨 Token 轴 2 层 MLP + LN) → SwiGLU Channel-Mix (按 D，跨 Token 共享权重)
       → 切片 → 每侧 zero-init 投影 × 可学习门控 → 残差加到两侧 Bus
```

关键：**侧投影零初始化**，融合从恒等映射开始逐步学习；Pooled slice 只读不回写。

### 4.4 分类器读出（§3.5）

不全部展平最终 Bus 状态，压缩读出 $\rho$ 包含两部分：
1. **NCB 压缩**：将 $(T_u{+}T_i{+}T_{ih})$ 拼接 Token 压缩到少量读出 Token → 展平
2. **Cross-Dot Gram**：在**未压缩**状态上计算 Mix Bus 与 Ret Bus Token 间的**所有成对内积**，保证压缩不抹掉显式二阶信号

分类器输入**拼接**：
- $\rho$（压缩读出 + Cross-Dot Gram）
- $[\phi^{1};\dots;\phi^{L}]$（每层 FM Highway，LayerNorm'd）
- $e_{\text{skip}}$（skip-embedding 信号，LayerNorm'd）

→ 2 层 MLP + Linear Head，logits clamp 到 $[-20, 20]$；另加 aux 延迟头。

### 4.5 多路径互学习（DML，§3.6）

训练时**两条 UniDot 路径**在同一 batch 上联合训练，**共享一套（占主导的）稀疏嵌入表**，推理时平均两个 logits。

$$
\mathcal{L} = \sum_{n=1}^{N}\Big[\,\mathcal{L}_{\text{task}}(y, p_n) + \frac{\lambda}{N-1}\sum_{i\neq n} D\big(\,\mathrm{sg}[p_i],\, p_n\big)\Big]
$$
- $\mathrm{sg}[\cdot]$：stop-gradient
- $D(a,b) = (a-b)^2$：概率上的 MSE 作为蒸馏距离
- $N=2$，$\lambda=20$，从第 1 epoch 后开启互学习项

**部署灵活性**：单路径推理（1× 成本）AUC 达 0.83184，仅比双路径 0.83217 低 0.033%——多路径训练已把单路径拉到了更好的最优点。

---

## 5. 实现与训练细节（附录 C）

### 5.1 数据集（工业赛道）

| 项 | 值 |
|----|----|
| Train / Test | 35M / 12M（round1 为 2M） |
| 列数（类别数） | 142（7） |
| User Int / Dense fids | 54 / 17 |
| Item Int / Dense fids | 17 / 4 |
| 预训练 emb fids（u/i） | 7 / 4 |
| 对齐位置 fids | 10 |
| 行为域（各域字段数） | 4（9/14/12/10） |
| 高基数 fid 116 | ≈ 9.4M |
| 标签 / 指标 | 转化率 / AUC |

### 5.2 提交配置

| 项 | 值 |
|----|----|
| Macro-layers $L$ / 每层 Mix-Block $W$ | 6 / 2 |
| $d_{\text{model}}$ | 128 |
| Token-Mix Block | Wukong（并行 LCB + FMB，rank 32） |
| User Token 预算 | 8 NS + 7×2 emb = 22 |
| Item Token（Mix / Ret Bus） | 4+8=12 / 16+8=24 |
| 各序列最近窗口 | a:256, b:256, c:512, d:512 + merged:512 → S=5 |
| Seq fid-压缩 Token | 4（→ 512-d per position） |
| Trunk Encoder | Transformer，1层，4 head，RoPE，窗口 w=128 |
| Pre-trunk Conv | Depthwise Conv1d，kernel 21 |
| 总参数量 | ≈ **2.1B**（嵌入主导，稠密路径仅数千万） |

### 5.3 数据特定处理

**(a) Per-Position 权重（user_dense 配对）**：
10 个 user_dense 数组不是特征，而是与对应 multi-value user_int fid **位置对齐的乘数**（如停留时间、交互计数）。
- 计数类 → 固定 $\log(1+x)/10$ 变换后相乘
- 相似度类 → clamp 到 $[-1,1]$ 后直接作为**有符号乘数**（负相似可减贡献）

**(b) FAFE 候选感知池化**：高价值多值 fids（fid 15, 63-66, 115-118, 121, 122）用 DIN attention 对 ranking candidate 打分，产出候选依赖的字段 Token。

**(c) 预训练嵌入投影与归一化**：
- 计数维度用**训练集固定统计量**标准化（非 per-batch）
- 2 层 MLP（Linear-GELU-Linear）投影到 $d$-dim Token
- 共享 Per-Token LayerNorm 使尺度与 NCB Token 匹配

**(d) 高基数 Skip Embedding**：超过 2M 的 fid（包括 fid 116 的 9.4M）使用共享**哈希表**（hashing-trick），其池化信号直接送入分类器（$e_{\text{skip}}$），冷重启时重新初始化。Item-id fid 额外专用 2M slot 乘法哈希表。

**(e) 辅助转化延迟头**：回归 $\log(1 + (t_{\text{label}} - t_{\text{event}}))$，MSE 损失，掩码为存在下一动作的行（覆盖点击和转化行，约 **8× 于正例的 aux 信号**）。

### 5.4 优化与训练

**双优化器**：
- 嵌入参数 → **Adagrad**（稀疏梯度）
- 稠密参数拆分：
  - ≥2D 矩阵权重 → **Muon（Moonshot 变体）**：解耦权重衰减，AdamW 迁移好；wd=1e-3
  - 1D 参数（LN 缩放、bias）→ **AdamW aux**，wd=0

**冷重启（Cold Restart）**：每个 epoch 边界（首个之后），**所有嵌入表 + Adagrad 状态重新初始化**，稠密参数跨 epoch 保持。这样 Backbone 跨 epoch 训练而嵌入每epoch重学，是对** train/test 时间间隙一步领先**（one-step-ahead gap）的正则化。每次重初始化后，稠密 LR **重新 warmup**。

**其他技巧**：
- 时间桶：64 个嵌入槽，边界覆盖 1s ~ 1.5 年，容量集中在 1h-18 月
- Logit clamp：$[-20, 20]$
- **EMA 权重**：稠密参数指数移动平均（decay 0.999），有验证集时选 live/EMA 更高者，全数据最终提交用 EMA。EMA 在后重初始化 warmup 窗口**重置**为 live 权重。

---

## 6. 实验结果

### 6.1 主结果

**最终排行榜（工业赛道 Top 10）**：

| Rank | AUC | 与 #1 差距 |
|------|-----|------------|
| 1 | 0.83254 | — |
| **2（UniDot，本文）** | **0.83217** | 0.037% |
| （单路径 UniDot） | 0.83184 | 0.070% |
| 3 | 0.83145 | 0.109% |

训练曲线（d=128, N=2 DML）：最佳 epoch 5，Test AUC 0.83217。

### 6.2 增量改进（从基线到最终）

| # | 改动 | Test AUC | Δ |
|---|------|----------|---|
| 0 | 比赛基线 | 0.81398 | — |
| 1 | **+ UniDot 架构** | 0.82500 | **+1.102%**（最大单次提升） |
| 2 | + item-id 哈希嵌入 | 0.82565 | +0.064% |
| 5 | + 更多 FM-Highway 点积 | 0.82722 | +0.018% |
| 6 | + Depthwise Conv pre-trunk | 0.82736 | +0.014% |
| 7 | + Aux 延迟损失 | 0.82812 | +0.075% |
| 8 | + FAFE | 0.82837 | +0.026% |
| 10 | + EMA 权重 | 0.82993 | +0.099% |
| 13 | + 多路径 DML (d=64) | 0.83128 | +0.085% |
| 14 | DML (d=128) | 0.83196 | +0.068% |
| 15 | + 全数据重训 | **0.83217** | +0.021% |
| **合计** | | | **+1.818%** |

### 6.3 组件消融（tiny 模式，单路径 d=64）

| 变体 | AUC | Δ AUC |
|------|-----|-------|
| Full UniDot（6 layers） | 0.83657 | — |
| − **FM Highway（全部）** | 0.83530 | **−0.127%**（最大损失） |
| − Cross-Bus Dots | 0.83570 | −0.087% |
| − Seq Cross-Attention | 0.83604 | −0.053%（MultiChannel Pool 已捕获大部分信号） |
| − FuseFFN | 0.83679 | +0.022%（但 LogLoss 变差，融合器可能设计不足） |
| − Merged 跨域流 | 0.83619 | −0.038% |

深度扫描：2/4/6/8/10 层中 6 层最优，更多层在 4M 数据上过拟合。

### 6.4 扩展研究

**稠密扩展优于稀疏扩展**：加倍嵌入宽度无收益；加宽稠密路径 d=64→96→128，AUC 单调增（累计 +0.050%）；**DML 双路径扩展更有效**：d=64 双路径 +0.135%，超过整个宽度扩展的收益。两者组合：d=128 + 2-path = +0.203%。

| 配置 | 稠密参数量 | GFLOP | Test AUC | Δ |
|------|-----------|-------|----------|---|
| d=64, sparse×2 | 18.8M | 5.9 | 无收益 | 无 win |
| d=64 基线 | 18.8M | 5.9 | 0.82993 | — |
| d=96 | 36.7M | 12.4 | 0.83024 | +0.031% |
| d=128 | 60.3M | 21.2 | 0.83043 | +0.050% |
| d=64, 2-path | 37.6M | 11.7 | 0.83128 | +0.135% |
| d=128, 2-path | **120.6M** | **42.5** | **0.83196** | **+0.203%** |

### 6.5 数据扩展

Held-out AUC 随训练数据呈**对数线性增长**：4M(0.83657) → 8M(0.83899) → 16M(0.84187) → 32M(0.84396)，每翻倍约 +0.0025 AUC——说明 Unified Block **数据饥渴，尚未饱和**，生产规模扩展空间大。

---

## 7. 与相关工作的关系

| 方法 | 统一方式 | 内积形式 | UniDot 的区别 |
|------|---------|----------|---------------|
| InterFormer / Kunlun | 交错序列与交互学习 | 隐式 | UniDot 保持跨边界信号为**显式点积** |
| HSTU / OneTrans / TokenFormer | 所有特征放入一个 Transformer 流 | 隐式 | UniDot 的双 Bus 分离天然避免了「序列坍缩」 |
| UniMixer / TokenMixer | 单栈 Token-Mixing Backbone | 隐式 | UniDot 增加专用 Seq-Retrieval Bus + FM Highway |
| Wukong / DHEN | 纯粹特征交互 | 部分显式 | UniDot 扩展到多域序列场景 |
| SlimPer | 每一层重读原始 Token 精炼知识 | 隐式 | UniDot 是 SlimPer 框架在公开数据集上的初步测试 |

---

## 8. 总结与未来方向

UniDot 以「**FM 内积 = Attention 点积**」这个统一的点积原语为核心，构建了一个可堆叠的双 Bus 并行架构：

1. **Token-Mixing Bus** 处理非序列特征（Wukong 作默认 Mixer）
2. **Sequence-Retrieval Bus** 用 Item Token 对多域行为历史做 Cross-Attention
3. **FuseFFN** 每层交换两者状态（MLP-Mixer 风格）
4. **FM Highway** 把每层显式点积**直达**分类器，保护二阶信号不被残差稀释
5. **多路径 DML** 共享嵌入互相蒸馏，收敛到更泛化的平坦最小点

最终在工业赛道获得亚军，**架构与规模驱动**（无手工跨特征）。

**未来工作**：
1. 在真实生产系统部署 UniDot，系统研究统一 Block 的 Scaling Law
2. 改进 FuseFFN：当前静态路由是薄弱环节（消融中去掉反而 AUC 微升），研究**输入条件化的二阶融合器**

---

## 7. 讨论记录

### 7.1 Q1：用一个具体例子详细描述整个前向过程

**问题**：用一个例子，比如 user profile 有 id, gender, age 三个特征；item profile 有 id, brand id, price 三个特征；Behavior sequence 有 L 个 events，每个 event 有 id, user_x_id_ctr 等 side info 特征。详细描述整个前向过程，要求每个子过程表示成 `output = Process(input)`，需要指明 input/output dimension，描述 Process 中的具体操作，以及每一关键步骤的 input/output 维度变化。

#### 7.1.1 示例数据与全局超参

**示例输入数据**：

| 类别 | 字段 |
|------|------|
| User profile | `user_id`（高基数多值 fid），`gender`（低基数），`age`（低基数），`SUM`（预训练 256-d emb） |
| Item profile | `item_id`（高基数），`brand_id`，`price`，`item_emb`（预训练 64-d emb） |
| Behavior sequence | L=200 events，每个 event 含 `event_id` + `user_x_id_ctr`（side info 标量，per-position 权重） |

**全局超参（按论文提交配置简化）**：

| 符号 | 含义 | 取值 |
|------|------|------|
| $D$ | d_model | 128 |
| $T_u$ | User Token 数（含 NS+emb） | 10（8 NS + 2 emb） |
| $T_i$ | Item Mix-Bus Token 数 | 6（4 NS + 2 emb） |
| $T_{ih}$ | Item Retrieval-Bus Token 数 | 18（16 NS + 2 emb） |
| $L$ | 序列长度 | 200 |
| $S$ | 序列视图数 | 5（4 行为域 + 1 merged） |
| $C$ | MultiChannel 池化通道数 | 4 |
| $L_{\text{block}}$ | Macro-block 层数 | 6 |
| $W$ | 每层 Mix-Block 数 | 2 |

---

#### 7.1.2 阶段 1：Tokenization（§3.2）

##### (1a) User NS fid 嵌入 + NCB 压缩

**Input**: 3 个 user fid（user_id, gender, age）的稀疏 ID 查表嵌入
- `user_id` → emb (1, 128)
- `gender` → emb (1, 128)
- `age` → emb (1, 128)
- 拼接（按位置） → `user_fid_embs` ∈ $\mathbb{R}^{3 \times 128}$

**Process**: `U_NS = NCB(user_fid_embs)` 
- NCB（Nonlinear Compression Block）：沿 **token 轴**两层 MLP
  - `Linear(3→8)` + GELU + `Linear(8→8)` + LayerNorm
  - 论文里说"NCB 是可学习的，由数据决定哪些特征组合成每个 token"

**Output**: `U_NS` ∈ $\mathbb{R}^{8 \times 128}$  （8 个 user NS tokens）

##### (1b) User 多值 fid 的 FAFE 池化（论文 §3.2 FAFE）

**假设 `user_id` 实际是高基数多值列表**（如用户历史交互过的多个 id），用 DIN 式候选感知池化：

**Input**: 
- `user_id_list_embs` ∈ $\mathbb{R}^{K \times 128}$（K 个 id 嵌入）
- `item_id_emb` ∈ $\mathbb{R}^{1 \times 128}$（ranking candidate 作 query）

**Process**: `pooled_user_id = FAFE(user_id_list_embs, item_id_emb)`
- DIN attention：$a_k = \mathrm{softmax}_k\, g(\text{item\_id\_emb}, \text{user\_id\_list\_embs}_k)$，g 为小型 MLP
- 加权求和：$\mathbf{p} = \sum_k a_k \cdot \mathbf{e}_k$
- 该字段对每个候选物品产出不同的字段 token

**Output**: 候选感知 user_id token ∈ $\mathbb{R}^{1 \times 128}$

##### (1c) User 预训练 emb 投影

**Input**: `SUM` 向量 ∈ $\mathbb{R}^{256}$

**Process**: `U_emb = MLP_proj(Normalize(SUM))`
- 用**训练集固定统计量**标准化（非 per-batch）
- 2 层 MLP：`Linear(256→128)` → GELU → `Linear(128→128)` → reshape 成 2 个 token
- 共享 per-token LayerNorm → 尺度与 NCB token 匹配

**Output**: `U_emb` ∈ $\mathbb{R}^{2 \times 128}$

##### (1d) 拼接得到最终 User Tokens

**Output**: `U = concat(U_NS, U_emb)` ∈ $\mathbb{R}^{10 \times 128}$  （$T_u=10$）

##### (1e) Item Mix-Bus Tokens（紧凑视图）

**Input**: 3 个 item fid 嵌入 `item_fid_embs` ∈ $\mathbb{R}^{3 \times 128}$ + `item_emb` ∈ $\mathbb{R}^{64}$

**Process**: 
- `I_NS = NCB(item_fid_embs)` → $\mathbb{R}^{4 \times 128}$ （NCB 压缩到 $T_i=4$）
- `I_emb = MLP_proj(item_emb)`：`Linear(64→128)` → GELU → `Linear(128→128)` → reshape 2 token → LN
- `I = concat(I_NS, I_emb)`

**Output**: `I` ∈ $\mathbb{R}^{6 \times 128}$  （$T_i=6$）

##### (1f) Item Retrieval-Bus Tokens（更丰富视图）

**Input**: 同样 3 个 item fid 但查**另一个 item 嵌入表** + item_emb

**Process**: 
- `I_h_NS = NCB(richer_item_fid_embs)` → $\mathbb{R}^{16 \times 128}$ （压缩到 $T_{ih}=16$）
- `I_h_emb = MLP_proj(item_emb)` → $\mathbb{R}^{2 \times 128}$
- `I_h = concat(I_h_NS, I_h_emb)`

**Output**: `I_h` ∈ $\mathbb{R}^{18 \times 128}$  （$T_{ih}=18$，作 Cross-Attention 查询）

##### (1g) Behavior Sequence Tokenization（per 域，假设域 a）

**Input**: L=200 个 events，每个 event 2 个 fid：`event_id`, `user_x_id_ctr`

**Process**:
- 每个 fid 按位置嵌入：
  - `event_id_embs` ∈ $\mathbb{R}^{200 \times 128}$
  - `user_x_id_ctr` 作为 per-position 权重：标量 $\in \mathbb{R}$ → 与 event_id 嵌入逐元素相乘
  - `event_token_uncompr` = `event_id_embs ⊙ ctr_weights` ∈ $\mathbb{R}^{200 \times 128}$
- 多个 fid 沿 fid 轴拼接（若有 n 个 fid，输出 (L, n·D)）
- **Position-local fid-axis NCB**：在每个 position 上独立做 `Linear(n·D → 4D)` + GELU + LN，把每个 event 压到统一宽度 4D=512
  - 即对每个 $\ell \in [1, L]$：$\text{NCB}(e_\ell) : \mathbb{R}^{n \cdot 128} \to \mathbb{R}^{512}$

**Output**: `H_raw^(a)` ∈ $\mathbb{R}^{200 \times 512}$  （per-position 512-d）

> 注：论文实际有 4 个域 a/b/c/d，每域独立 token 化。简化下我们只跟踪域 a 和 merged 流。

---

#### 7.1.3 阶段 2：Sequence Encoder（§3.3）

四个域 + 1 个 merged 流共 $S=5$ 个 trunk。这里以域 a 为例。

##### (2a) Cross-domain Merge（merged 流生成）

**Input**: 4 个域的 raw tokens $\{H_{\text{raw}}^{(a)}, H_{\text{raw}}^{(b)}, H_{\text{raw}}^{(c)}, H_{\text{raw}}^{(d)}\}$，每个 ∈ $\mathbb{R}^{200 \times 512}$，含各自时间戳

**Process**: 按时间戳交错成单条流
- 4×200 = 800 events 重新按 timestamp 排序
- 截断/补齐到长度 $L_{\text{merged}}=512$（论文配置）

**Output**: `H_raw^(merged)` ∈ $\mathbb{R}^{512 \times 512}$

##### (2b) Depthwise Conv1d（局部 N-gram）

**Input**: `H_raw^(a)` ∈ $\mathbb{R}^{200 \times 512}$

**Process**: `H_conv = DepthwiseConv1d(H_raw^(a), kernel=21)`
- 每通道独立 1D 卷积（depthwise = 通道间不混合），窗口 21
- 廉价捕获局部 N-gram 模式（爆发、相邻事件 motif）

**Output**: `H_conv` ∈ $\mathbb{R}^{200 \times 512}$  （shape 不变）

##### (2c) CondGatedSwiGLU（候选感知过滤）

**Input**:
- `H_conv` ∈ $\mathbb{R}^{200 \times 512}$（value path 的来源）
- `cond_vec`：候选感知的复合 cond 向量
  - 构造：`[U(item LCB tokens); I(item LCB tokens); emb_cond]` → NCB 融合 → $\mathbb{R}^{1 \times 128}$

**Process**: `H_gate = CondGatedSwiGLU(H_conv, cond_vec)`
- Value path（纯序列）: $\text{val} = \text{Linear}_{512→128}(H_{\text{conv}}) \in \mathbb{R}^{200 \times 128}$
- Gate（候选注入）: $\text{gate} = \sigma(\text{Linear}([\text{val}; \text{cond}; \text{val} \odot \text{cond}])) \in \mathbb{R}^{200 \times 128}$
  - cond 通过广播到每个 position
- Output = $\text{val} \odot \text{gate}$

> 关键设计：cond 只进入 gate，**不污染 value 内容**——DIN 式地"按候选相关性选择位置"，但保留位置内容纯净

**Output**: `H_gated` ∈ $\mathbb{R}^{200 \times 128}$

##### (2d) Causal Transformer（长程依赖）

**Input**: `H_gated` ∈ $\mathbb{R}^{200 \times 128}$

**Process**: `H_tf = CausalTransformer(H_gated)`
- 1 层 Transformer，4 head，head_dim=32
- **窗口注意力** window $w=128$（每个 token 只 attend 前 128 个）
- RoPE 旋转位置编码
- **因果 mask**（按时间顺序）
- SDPA/FlashAttention 融合核

**Output**: `H_tf` ∈ $\mathbb{R}^{200 \times 128}$

##### (2e) View Projector

**Input**: `H_tf` ∈ $\mathbb{R}^{200 \times 128}$

**Process**: `H^(a) = ViewProj(H_tf)` = Linear(128→128) → LN → GELU

**Output**: `H^(a)` ∈ $\mathbb{R}^{200 \times 128}$

对每个域 + merged 流重复 (2b)-(2e)，得到 $S=5$ 个视图：$\{\mathbf{H}^{(s)}\}_{s=1}^{5}$

---

#### 7.1.4 阶段 3：Macro-Block × 6 层（§3.4）

**初始化**：
- $Z_{\text{mix}}^{(0)} = [\mathbf{U}; \mathbf{I}] = \mathrm{concat}((10,128), (6,128))$ → $\mathbb{R}^{16 \times 128}$
- $Z_{\text{seq}}^{(0)} = \mathbf{I}_h \in \mathbb{R}^{18 \times 128}$
- 输入视图 $\{\mathbf{H}^{(s)}\}_{s=1}^{5}$

下面对单层 $\ell=1$ 展开（其余 5 层结构相同、参数共享或独立）。

##### (3.1) Token-Mixing Bus（W=2 个 Wukong Block）

##### (3.1.1) Wukong Block #1

**Input**: `Z_mix_in` ∈ $\mathbb{R}^{16 \times 128}$  （U; I 拼接）

**Process**: `Z_mix_mid = Wukong(Z_mix_in)`
- **LCB 分支**（Linear Compression Block）:
  - `Linear(16→16)` + LayerNorm → $\mathbb{R}^{16 \times 128}$  （沿 token 轴压缩/混合）
- **FMB 分支**（Factorization-Machine Block，rank=32）:
  - 低秩分解的 pairwise dot products：$Z \cdot Z^\top \in \mathbb{R}^{16 \times 16}$
  - 经低秩线性层 $\text{Linear}(16 \to 16 \cdot 32 \to 16)$ → 显式二阶交互
  - 输出 ∈ $\mathbb{R}^{16 \times 128}$
- **并行残差**：`out = LCB(x) + FMB(x)`（两个分支并行，结果相加）

**Output**: `Z_mix_mid` ∈ $\mathbb{R}^{16 \times 128}$

##### (3.1.2) Wukong Block #2

**Input**: `Z_mix_mid` ∈ $\mathbb{R}^{16 \times 128}$

**Process**: 同 (3.1.1)

**Output**: `w_out` ∈ $\mathbb{R}^{16 \times 128}$

##### (3.2) MultiChannelSeqPool（多通道序列池化）

**Input**:
- 5 个序列视图 $\{\mathbf{H}^{(s)}\}$，每个 ∈ $\mathbb{R}^{200 \times 128}$
- Mixing-Bus 状态摘要：`summary(Z_mix)` ∈ $\mathbb{R}^{1 \times 128}$（如 mean-pool over tokens）

**Process**: 对每个序列 $s$：
- 对每个通道 $c \in [1, C=4]$：
  - 计算 per-position 门控：$w_{\ell, c} = \sigma(\text{MLP}([\mathbf{H}^{(s)}_\ell; \text{summary}]))$
    - MLP: `Linear(128+128→64)` → GELU → `Linear(64→1)` → $\mathbb{R}^{200 \times 1}$
  - 池化：$\mathbf{p}_c = \sum_\ell w_{\ell, c} \cdot \mathbf{H}^{(s)}_\ell \in \mathbb{R}^{128}$
  - **ℓ₂-normalize over D**：$\mathbf{p}_c \leftarrow \mathbf{p}_c / \|\mathbf{p}_c\|_2$（restore unit norm，因 sigmoid 无界）
- 4 个通道 → $\mathbb{R}^{4 \times 128}$
- 5 个序列 → 拼接 $\mathbb{R}^{20 \times 128}$
- LayerNorm 尾

**Output**: `pooled` ∈ $\mathbb{R}^{20 \times 128}$  （$S \cdot C = 5 \times 4 = 20$）

##### (3.3) Sequence-Retrieval Bus

##### (3.3.1) Per-sequence Cross-Attention

**Input**:
- Query: `I_h` ∈ $\mathbb{R}^{18 \times 128}$ （Item Retrieval tokens，由 $Z_{\text{seq}}^{(\ell-1)}$ 提供）
- Key/Value: 每个 $\mathbf{H}^{(s)}$ ∈ $\mathbb{R}^{200 \times 128}$

**Process**: 对 $s=1..5$：
- $Q = \text{Linear}_Q(I_h) \in \mathbb{R}^{18 \times 128}$
- $K = \text{Linear}_K(\mathbf{H}^{(s)}) \in \mathbb{R}^{200 \times 128}$
- $V = \text{Linear}_V(\mathbf{H}^{(s)}) \in \mathbb{R}^{200 \times 128}$
- $A_s = \mathrm{softmax}(Q K^\top / \sqrt{d_k}) \cdot V$  （FM 视角：$Q \cdot K$ 就是 token 点积打分）
- Output $A_s$ ∈ $\mathbb{R}^{18 \times 128}$  （Item Token 对该序列的检索结果）

> 论文强调："cross-attention 被读作 FM scoring between query and key tokens"——这就是 UniDot 名字的来源

**Output**: $\{A_0, A_1, A_2, A_3, A_4\}$，每个 ∈ $\mathbb{R}^{18 \times 128}$

##### (3.3.2) LCB Aggregation

**Input**: `stack([A_0..A_4])` ∈ $\mathbb{R}^{5 \times 18 \times 128}$ → reshape $\mathbb{R}^{18 \times 640}$  （5×128=640）

**Process**: `attn_agg = LCB(stacked)` = `Linear(640→128)` + LN
- 线性聚合（因下游 FuseFFN 已提供非线性）

**Output**: `attn_agg` ∈ $\mathbb{R}^{18 \times 128}$

##### (3.3.3) Per-token Fusion FFN

**Input**:
- `I_h` ∈ $\mathbb{R}^{18 \times 128}$
- `attn_agg` ∈ $\mathbb{R}^{18 \times 128}$

**Process**: 对每个 item token $t \in [1, 18]$：
- 拼接：$[\mathbf{I}_h[t]; \text{attn\_agg}[t]] \in \mathbb{R}^{256}$
- FFN: `Linear(256→256)` → GELU → `Linear(256→128)`
- 残差增量 $\Delta_h[t]$ ∈ $\mathbb{R}^{128}$

**Output**: `h_out` ∈ $\mathbb{R}^{18 \times 128}$  （对 $Z_{\text{seq}}$ 的残差 delta，**还未加回**）

##### (3.3.4) FM Highway 信号（该层 $\phi^\ell$）

**Input**: `I_h`, `attn_agg`, 5 个 $A_s$, mixing bus 的 `U` 部分

**Process**: 计算 4 类显式点积：
1. **Per-sequence dots**: $d_s[t] = \langle \mathbf{I}_h[t], A_s[t] \rangle$
   - 每个 (s, t) 一个标量 → $5 \times 18 = 90$ 维
2. **Fused-domain dot**: `NCB(stack(A_0..A_4))` → $\mathbb{R}^{18 \times 128}$，再与 $\mathbf{I}_h$ 点积
   - 每个 t 一个标量 → 18 维
3. **Aggregated Gram**: $G = \mathbf{I}_h \cdot \text{attn\_agg}^\top$ 
   - $\mathbb{R}^{18 \times 18} = 324$ 维
4. **Cross-bus User-Item dots**: $U \cdot \mathbf{I}_h^\top$
   - $\mathbb{R}^{10 \times 18} = 180$ 维

**Output**: `φ^ℓ` ∈ $\mathbb{R}^{612}$  （拼接，绕过 FuseFFN，直达分类器）

##### (3.4) FuseFFN（MLP-Mixer 跨 Bus 融合）

**Input**:
- `w_out` ∈ $\mathbb{R}^{16 \times 128}$  （Mix Bus 输出）
- `pooled` ∈ $\mathbb{R}^{20 \times 128}$  （池化 Token）
- `h_out` ∈ $\mathbb{R}^{18 \times 128}$  （Seq Bus 输出）

**Process**: 
1. **Token 轴拼接**：`concat([w_out; pooled; h_out], dim=0)` → $\mathbb{R}^{54 \times 128}$
2. **NCB Token-Mix**（跨 token 轴 2 层 MLP）:
   - `Linear(54→54)` → GELU → `Linear(54→54)` + LayerNorm → $\mathbb{R}^{54 \times 128}$
   - 每个 token 与所有其他 token 混合（**跨 Bus 交换**）
3. **SwiGLU Channel-Mix**（按 D，权重跨 token 共享）:
   - `Linear(128→256)` → 拆为 (gate, value) 各 128 → `gate ⊙ GELU(value)` → $\mathbb{R}^{54 \times 128}$
4. **切片**:
   - `w_slice` ∈ $\mathbb{R}^{16 \times 128}$
   - `pooled_slice` ∈ $\mathbb{R}^{20 \times 128}$  （**只读，丢弃，不写回**）
   - `h_slice` ∈ $\mathbb{R}^{18 \times 128}$
5. **Zero-init 侧投影 + 可学习门控**:
   - $\Delta_w = \text{proj}_w(w_{\text{slice}}) \times s_w$  // proj_w 零初始化，s_w 学习标量
   - $\Delta_h = \text{proj}_h(h_{\text{slice}}) \times s_h$
6. **残差加**:
   - $Z_{\text{mix}}^{(\ell)} = Z_{\text{mix}}^{(\ell-1)} + \Delta_w$
   - $Z_{\text{seq}}^{(\ell)} = Z_{\text{seq}}^{(\ell-1)} + \Delta_h$

**Output**:
- $Z_{\text{mix}}^{(\ell)}$ ∈ $\mathbb{R}^{16 \times 128}$
- $Z_{\text{seq}}^{(\ell)}$ ∈ $\mathbb{R}^{18 \times 128}$

> 由于 proj_w/proj_h 零初始化，融合**从恒等映射开始逐步学习**，不会破坏初始模型

经过 6 层 Macro-Block 后得到：
- $Z_{\text{mix}}^{(6)}$ ∈ $\mathbb{R}^{16 \times 128}$
- $Z_{\text{seq}}^{(6)}$ ∈ $\mathbb{R}^{18 \times 128}$
- $\Phi = [\phi^1; \phi^2; \dots; \phi^6]$ ∈ $\mathbb{R}^{3672}$  （612 × 6，每层拼接而非求和）

---

#### 7.1.5 阶段 4：Classifier Readout（§3.5）

##### (4.1) Skip-Embedding 信号

**Input**: 高基数 fid（如 `user_id` >2M cardinality 部分）经共享 hash 表池化

**Process**: `e_skip = LayerNorm(HashPool(high_card_fids))`

**Output**: `e_skip` ∈ $\mathbb{R}^{64}$  （假设 64 维，具体未明示）

##### (4.2) 压缩读出 ρ

**Input**: `concat([Z_mix^(6); Z_seq^(6)])` ∈ $\mathbb{R}^{34 \times 128}$

**Process**:
1. **NCB 压缩**到 4 readout tokens:
   - `Linear(34→4)` + GELU + `Linear(4→4)` + LN → $\mathbb{R}^{4 \times 128}$
   - 展平 → $\mathbb{R}^{512}$
2. **Cross-Dot Gram**（在**未压缩**状态上计算）:
   - $Z_{\text{mix}}^{(6)} \cdot {Z_{\text{seq}}^{(6)}}^\top$ → $\mathbb{R}^{16 \times 18} = 288$ 维
   - 保证压缩不抹掉显式二阶信号

**Output**: `ρ` = concat([flatten(NCB_out); cross_dot_gram]) ∈ $\mathbb{R}^{800}$

##### (4.3) 最终分类器

**Input**: `concat([ρ; Φ; e_skip])` ∈ $\mathbb{R}^{4536}$  （800 + 3672 + 64）

**Process**:
- `Linear(4536→256)` → GELU → `Linear(256→1)`
- Logit clamp 到 $[-20, 20]$
- $\hat{y} = \sigma(\text{logit})$

**Output**: `ŷ` ∈ $\mathbb{R}^{1}$  （转化概率）

---

#### 7.1.6 阶段 5：辅助转化延迟头（§3.6 + 附录 C.5）

**Input**: 同样骨干末端特征（具体 trunk 论文未详述）

**Process**: 
- $\hat{d} = \text{MLP}_{\text{delay}}(\text{trunk features})$ → 回归 $\log(1 + (t_{\text{label}} - t_{\text{event}}))$
- 掩码：仅 $t_{\text{label}} > t_{\text{event}}$ 的行（覆盖点击+转化行，约 8× 正例信号）

**Output**: `delay_pred` ∈ $\mathbb{R}^{1}$

---

#### 7.1.7 阶段 6：Loss 计算

**Input**: `ŷ`, `delay_pred`, 真实标签 `y`, `delay_true`

**Process**:
$$
\mathcal{L} = \underbrace{-y\log\hat{y} - (1-y)\log(1-\hat{y})}_{\text{BCE}} + \lambda \cdot \underbrace{(\hat{d} - d_{\text{true}})^2}_{\text{MSE delay}}, \quad \lambda=0.01
$$

---

#### 7.1.8 关键维度流转总览

```
┌─────────────────────────────────────────────────────────────────────┐
│ User profile                  Item profile                          │
│  3 fids → (3,128)              3 fids → (3,128)                    │
│   NCB → U_NS (8,128)            NCB → I_NS (4,128)   [Mix Bus]     │
│                                  NCB → I_h_NS (16,128) [Ret Bus]   │
│  SUM(256) → MLP → U_emb(2,128)  item_emb(64) → MLP → I_emb(2,128)  │
│                                                                       │
│  U (10,128)                     I (6,128)    I_h (18,128)           │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ Behavior sequence (per domain a)                                     │
│  L=200 events × n fid embs → (200, n·128)                            │
│  Position-local NCB → (200, 512)     [raw per-position width]        │
│  DepthwiseConv1d(k=21) → (200, 512)                                 │
│  CondGatedSwiGLU(cond 1×128) → (200, 128)                            │
│  CausalTransformer(window=128) → (200, 128)                          │
│  ViewProj → H^(a) (200, 128)                                        │
│                                                                       │
│  + Merged stream (512, 512) → same trunk → H^(merged) (512, 128)    │
│  → 5 sequence views {H^(s)}, each (L_s, 128)                         │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ Macro-Block ℓ = 1..6  (init: Z_mix=(16,128), Z_seq=(18,128))        │
│                                                                       │
│ Token-Mixing Bus:    Z_mix → Wukong×2 → w_out (16,128)               │
│ MultiChannelSeqPool: {H^(s)} + summary(Z_mix) → pooled (20,128)      │
│ Seq-Retrieval Bus:   I_h × {H^(s)} → 5×A_s (18,128 each)            │
│                       → LCB stack → attn_agg (18,128)                │
│                       → FFN per-token → h_out (18,128)               │
│ FM Highway φ^ℓ:      612-dim (per-seq dots + fused + Gram + cross)   │
│ FuseFFN:             concat(54,128)→NCB→SwiGLU→slice                 │
│                       → Δ_w (16,128), Δ_h (18,128)                   │
│ Residual:            Z_mix += Δ_w,   Z_seq += Δ_h                     │
└─────────────────────────────────────────────────────────────────────┘

After 6 layers:
  Z_mix^(6) (16,128)   Z_seq^(6) (18,128)   Φ = [φ¹..φ⁶] (3672,)

┌─────────────────────────────────────────────────────────────────────┐
│ Classifier                                                            │
│  NCB compress (34,128)→(4,128)→flatten → (512,)                      │
│  + Cross-Dot Gram (16×18=288)                                         │
│  → ρ (800,)                                                           │
│  + Φ (3672,) + e_skip (64,)                                           │
│  → MLP (4536→256→1) → clamp[-20,20] → σ → ŷ                           │
└─────────────────────────────────────────────────────────────────────┘
```

---

#### 7.1.9 几个值得强调的设计细节

1. **序列只编码一次**：5 个视图 $\{\mathbf{H}^{(s)}\}$ 在阶段 2 计算后，被所有 6 层 Macro-Block 共享，**不在每层重新 forward**——这是约束推理延迟的关键。
2. **FM Highway 绕过融合**：$\phi^\ell$ 跨层**拼接**而非求和，保证每层显式二阶信号不被残差路径稀释。这是消融中代价最大的组件（−0.127% AUC）。
3. **Zero-init FuseFFN**：融合器的侧投影零初始化，模型初始时 $\Delta_w = \Delta_h = 0$，融合从恒等映射开始逐步学习，避免破坏初始 backbone。
4. **Pooled Token 只读不写回**：MultiChannelSeqPool 的池化结果在 FuseFFN 中参与跨 Bus 交换但**不写回任何 Bus 的状态**，每层重新生成。
5. **NCB 压缩可学习**：User/Item 的 fid 压缩是 NCB（而非固定 pooling），由数据决定哪些特征组合成每个 token——避免平均池化抹平位置级 salience。
6. **候选感知只进 SwiGLU 的 Gate**：CondGatedSwiGLU 中候选 cond 向量只影响 gate 支路，**value path 保持纯序列内容**，符合 DIN 设计哲学。
