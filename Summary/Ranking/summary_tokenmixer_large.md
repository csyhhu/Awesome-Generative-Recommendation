# TokenMixer-Large: 工业级大规模排序模型扩展架构

论文链接：https://arxiv.org/abs/2602.06563
发表会议：CIKM 2025
作者单位：字节跳动（ByteDance）

---

## 一、研究动机与问题定义

TokenMixer（RankMixer 中提出的架构）将 Self-Attention 替换为轻量级的 token mixing 操作（纯 reshape），大幅降低了计算开销，同时保持 Scaling 效率。然而 RankMixer/TokenMixer 存在五大核心局限：

1. **次优残差设计**：Mixing 后 Token 数 $H$ 与原始 Token 数 $T$ 不一致，残差连接无法稳定跨层传播，且直接相加导致语义不对齐
2. **不纯净的模型架构**：遗留大量碎片化算子（DCN、LHUC 等），这些 I/O 密集型算子降低整体 MFU
3. **深层梯度更新不足**：RankMixer 通常仅堆叠 2~4 层，深层训练不稳定
4. **MoE 稀疏化不充分**：采用"Dense Train, Sparse Infer"范式，无法降低训练成本；ReLU-MoE 激活动态不可预测
5. **扩展规模受限**：仅推到约 1B 参数

TokenMixer-Large 系统性地解决上述问题，将参数规模推到 **7B（在线）/ 15B（离线）**。

---

## 二、核心架构

### 整体流程

```
原始特征 → Tokenization → 多TokenMixer-Large Block 堆叠
                              │
                    ┌─────────┼─────────┐
                    │  Mixing & Reverting │
                    │  Pertoken SwiGLU    │  (Pre-Norm RMSNorm)
                    │  Sparse-Pertoken MoE│
                    │  Inter-Residual     │
                    └─────────┼─────────┘
                              ↓
                          Mean Pooling → 多任务预测
```

### 2.1 Tokenization

**语义分组 Tokenizer**：

与 RankMixer/UniMixer 的均匀切分不同，TokenMixer-Large 采用**语义分组**策略：
- 将不同维度的 Embedding 按语义分组 $\{G_1, ..., G_{T-1}\}$
- 每组内 Embedding 拼接后通过**不同 DNN（MLP）** 投影到统一维度 $D$：
  $$\mathbf{X}_i = \text{MLP}_i(\text{concat}[e_l, ..., e_m]), \quad e_l, ..., e_m \in G_i$$
- 每个组的 MLP 参数独立，保留异构性

**Global Token**：

引入类似 BERT `[CLS]` 的全局 Token，聚合所有特征的全局信息：
$$\mathbf{X}_G = \text{MLP}_g(\text{concat}[G_1, ..., G_{T-1}])$$

最终输入：$\mathbf{X} = \text{concat}[\mathbf{X}_G, \mathbf{X}_0, ..., \mathbf{X}_{T-1}] \in \mathbb{R}^{T \times D}$

### 2.2 Mixing & Reverting（核心创新）

**问题**：RankMixer 中 Mixing 后 Token 数变为 $H$，而残差加法要求输入输出 Token 数一致。若各层 $H$ 不同，残差无法稳定传播。

**解决方案 — 对称的两层结构**：

- **第 1 层（Mixing）**：将 $T$ 个原始 Token split → concat 为 $H$ 个混合 Token → Pertoken SwiGLU：
  ```
  X ∈ R^{T×D} → split → R^{T×H×(D/H)} → concat → R^{H×(T·D/H)} → pSwiGLU
  ```

- **第 2 层（Reverting）**：将 $H$ 个混合 Token 还原为 $T$ 个 Token → Pertoken SwiGLU：
  ```
  R^{H×(T·D/H)} → split → R^{T×H×(D/H)} → 重组每个原始位置的碎片 → R^{T×D} → pSwiGLU
  ```

$$
\begin{aligned}
[\dots,[\mathbf{x}_t^{(1)}, \dots, \mathbf{x}_t^{(H)}],\dots] &= \text{split}(\mathbf{X}) \\
\mathbf{H}_h &= \text{concat}[\mathbf{x}_1^{(h)}, \dots, \mathbf{x}_T^{(h)}] \\
\mathbf{H}^{\text{next}} &= \text{Norm}(\text{pSwiGLU}(\mathbf{H}) + \mathbf{H}) \\
[\dots,[\mathbf{x'}_1^{(h)}, \dots, \mathbf{x'}_T^{(h)}],\dots] &= \text{split}(\mathbf{H}^{\text{next}}) \\
\mathbf{X}_t^{\text{revert}} &= \text{concat}[\mathbf{x'}_t^{(1)}, \dots, \mathbf{x'}_t^{(H)}] \\
\mathbf{X}^{\text{next}} &= \text{Norm}(\text{pSwiGLU}(\mathbf{X}^{\text{revert}}) + \mathbf{X})
\end{aligned}
$$

**核心收益**：输入输出始终为 $T \times D$，残差连接可以稳定跨层，同时保留了 Mixing 的信息交换能力。

### 2.3 Pertoken SwiGLU

将 RankMixer 的 Pertoken FFN 升级为 Pertoken SwiGLU：

$$\text{pSwiGLU}(\mathbf{x}_t) = W_{\text{down}}^t \left(\text{Swish}(W_{\text{gate}}^t \mathbf{x}_t + \mathbf{b}_{\text{gate}}^t) \odot (W_{\text{up}}^t \mathbf{x}_t + \mathbf{b}_{\text{up}}^t)\right) + \mathbf{b}_{\text{down}}^t$$

- $W_{\text{up}}^t, W_{\text{gate}}^t \in \mathbb{R}^{D \times nD}$，$W_{\text{down}}^t \in \mathbb{R}^{nD \times D}$
- 每个 Token 有独立的参数（与 UniMixer/UniFormer 的 Pertoken/多视图 FFN 理念一致）
- $n$ 为隐藏层扩展比例

### 2.4 Residual & Normalization

- 将 RankMixer 的 Post-Norm 改为 **Pre-Norm RMSNorm**：Post-Norm 效果略好但易梯度爆炸，Pre-Norm 保证训练稳定性
- RMSNorm 相比 LayerNorm 去除了均值中心化步骤，吞吐量**提升 8.4%**

### 2.5 Inter-Residual & Auxiliary Loss

**Interval Residual**（间隔残差）：

- 每隔 2~3 层添加跨层残差连接，增强低层特征到高层的信号传输
- 加速浅层参数收敛，缓解深层梯度衰减
- **最后一层不加**（避免原始低层信息干扰高层抽象特征的精确性）

**Auxiliary Loss**（辅助损失）：

- 将低层输出的 logits 与高层 logits 合并计算联合损失
- 使低层网络额外学习"预测高层特征偏差"的能力
- 防止网络加深导致低层参数训练不足

### 2.6 Sparse-Pertoken MoE

**"First Enlarge, Then Sparse"策略**：

不是直接在 MoE 中增加专家数，而是先在 Dense 模型上找到最优性能，再通过稀疏化获得效率收益：

1. 将 Pertoken SwiGLU 中的 FC 拆分为细粒度 Experts
2. 通过 Router + Softmax 稀疏激活 top-k 个 Expert
3. 每个 Token 有自己专属的 Expert 集合（与标准 MoE 的核心区别）

$$\text{S-P MoE}(\mathbf{x}_t) = \alpha \cdot \sum_{j=1}^{k-1} g_j(\mathbf{x}_t) \cdot \text{Expert}_j(\mathbf{x}_t) + \text{SharedExpert}(\mathbf{x}_t)$$

**关键设计**：

| 设计 | 作用 |
|------|------|
| **Shared Expert** | Per-token 专用共享专家，提升训练稳定性 |
| **Gate Value Scaling（$\alpha$）** | Router 输出乘以缩放常数，增强被选中 Expert 的梯度更新。$\alpha$ 与稀疏比成反比（1:2 时 $\alpha=2$，1:4 时 $\alpha=4$） |
| **Down-Matrix Small Init** | $W_{\text{down}}$ 初始化标准差降到 0.01，使 SwiGLU 输出接近恒等映射，改善训练初期稳定性 |

**统一"Sparse Train, Sparse Infer"范式**：训练和推理均为稀疏激活，相比 RankMixer 的"Dense Train, Sparse Infer"显著降低训练成本。目前 **1:2 稀疏比零精度损失部署**。

### 2.7 Pure Model Design（纯净模型设计）

随着 TokenMixer-Large 参数规模增大，DCN、LHUC 等碎片化算子的收益逐渐被 TokenMixer-Large 自身吸收：

| 参数规模 | DCN 额外增益 |
|---------|:---:|
| 150M | +0.09% |
| 500M | +0.04% |
| 700M | +0.00% |

因此大模型中去掉所有碎片化算子，只保留无参数的 Mixing/Reverting 和高效的 GroupedGemm，MFU 达到 **60%**。

### 2.8 训练/推理优化

**高性能定制算子**：

- **MoEPermute**：batch-first → expert-first 转换，使每个 Expert 输入连续
- **MoEGroupedFFN**：单个 kernel 计算所有 Expert FFN，减少调度开销
- **MoEUnpermute**：计算多个激活 Expert 输出的加权和

**FP8 量化**：推理时 FP8 E4M3 后训练量化，1.7× 加速，无精度损失。

**Token Parallel**：

- 将 Pertoken 操作的权重按 Token 维度切分到多设备
- Mixing 和 Reverting 两步的 `all2all` 通信实现数据交换
- 相比朴素的模型并行，通信次数从 $4L$ 降至 $2L+1$
- 4 路 Token Parallel（总 batch 320）：**推理吞吐 +29.2%**，通信-计算重叠后达 **+96.6%**

---

## 三、实验结果

### 3.1 实验设置

- 场景：抖音电商（主场景）、抖音 Feed 广告、抖音直播
- 日均样本量：电商 4 亿条，广告 3 亿条，直播 170 亿条
- 训练框架：64 GPU（电商）/ 256 GPU（广告/直播）混合分布式训练
- 指标：AUC / UAUC / 参数量 / FLOPs / MFU

### 3.2 SOTA 对比（电商，约 500M 参数）

| 模型 | CTCVR ΔAUC | 参数量 | FLOPs/Batch |
|------|:---:|:---:|:---:|
| DLRM-MLP | — | 499M | 125.1T |
| HiFormer | +0.44% | 570M | 28.8T |
| DCNv2 | +0.49% | 502M | 125.8T |
| DHEN | +0.63% | 415M | 103.4T |
| AutoInt | +0.75% | 549M | 138.6T |
| Wukong | +0.76% | 513M | 4.6T |
| Group Transformer | +0.81% | 550M | 4.5T |
| FAT | +0.82% | 551M | 4.59T |
| RankMixer | +0.84% | 567M | 4.6T |
| **TokenMixer-Large 500M** | **+0.94%** | 501M | 4.2T |
| **TokenMixer-Large 4B** | **+1.14%** | 4.6B | 29.8T |
| **TokenMixer-Large 7B** | **+1.20%** | 7.6B | 49.0T |
| **TokenMixer-Large 4B SP-MoE** | **+1.14%** | 2.3B/4.6B | 15.1T |

### 3.3 Scaling Law

- 在 Feed 广告 / 电商 / 直播三个场景分别推到 **15B / 7B / 4B**（离线）和 **7B / 4B / 2B**（在线）
- **均衡扩展各维度**（宽度 D + 深度 L + SwiGLU 扩展比例 n）在 1B 以上比单维度扩展 ROI 更高
- **更大模型需要更多数据**：30M→90M 收敛需 14 天样本，500M→2B 需 60 天

### 3.4 消融实验（4B 模型）

**Block 组件消融**：

| 移除组件 | ΔAUC |
|------|:---:|
| 完整 TokenMixer-Large | — |
| w/o Global Token | -0.02% |
| w/o Mixing & Reverting | **-0.27%**（影响最大） |
| w/o Residual | -0.15% |
| w/o Internal Residual & AuxLoss | -0.04% |
| Pertoken SwiGLU → 共享 SwiGLU | -0.21% |
| Pertoken SwiGLU → Pertoken FFN | -0.10% |

**Sparse-Pertoken MoE 消融**：

| 移除组件 | ΔAUC |
|------|:---:|
| w/o Shared Expert | -0.02% |
| w/o Gate Value Scaling | -0.03% |
| w/o Down-Matrix Small Init | -0.03% |
| Sparse-Pertoken MoE → 标准 MoE | **-0.10%** |

### 3.5 在线 A/B 测试

| 场景 | 核心指标 | 提升 |
|------|------|:---:|
| Feed 广告 | ADSS（广告主满意度） | **+2.0%** |
| 电商 | GMV | **+2.98%** |
| 电商 | 订单数 | **+1.66%** |
| 直播 | 付费金额 | **+1.4%** |

基线上线模型：广告场景 RankMixer-1B，电商 RankMixer-150M，直播 RankMixer-500M。已服务数亿用户。

### 3.6 与 RankMixer 的详细对比

TokenMixer-Large 修复了 RankMixer 的三个残差设计缺陷：

| 属性 | 含义 | Group Transformer | RankMixer | TokenMixer-Large |
|------|------|:---:|:---:|:---:|
| SR | 标准残差连接 | ✅ | ✅ | ✅ |
| OTR | 原始 Token 语义保留到最终输出 | ✅ | ❌ | ✅ |
| TSA | 残差前后 Token 语义对齐 | ✅ | ❌ | ✅ |

**Mixing 策略**：只要每个新 Token 包含所有原始 Token 的信息，具体切分方式（垂直/对角/随机）不影响效果。

---

## 四、主要结论

1. **系统性升级**：Mixing & Reverting 解决了 TokenMixer 残差设计的根本问题，保证 TSA/OTR/SR 三重属性全部满足
2. **极限扩展**：成功推到 15B（离线）/ 7B（在线）参数，Scalng Law 持续收益
3. **高效稀疏化**：Sparse-Pertoken MoE 实现"Sparse Train, Sparse Infer"，1:2 稀疏比零精度损失
4. **工程成熟**：Pure Model 设计 + 高性能算子 + FP8 + Token Parallel，MFU 达 60%
5. **工业验证**：字节跳动多业务线全量部署，GMV +2.98%，ADSS +2.0%

---

## 五、与 UniMixer / UniFormer 的关系

### 与 UniMixer 的关系

两者的共同源头均为 RankMixer/TokenMixer：

| | TokenMixer-Large | UniMixer |
|------|------|------|
| 出发点 | 修复 TokenMixer 工程缺陷（残差、梯度、稀疏化） | 数学统一 TokenMixer/Attention/FM |
| Tokenization | 语义分组 + Global Token | 均匀切分（匿名 Token） |
| 核心操作 | Mixing & Reverting（对称两层面） | 可学习双随机矩阵 $W_G \times W_B$ |
| 残差设计 | Mixing-Reverting 保证 TSA + Pre-Norm RMSNorm | SiameseNorm 双流耦合 |
| MoE | Sparse-Pertoken MoE（Sparse Train + Sparse Infer） | Pertoken SwiGLU（无 MoE） |

### 与 UniFormer 的关系

| | TokenMixer-Large | UniFormer |
|------|------|------|
| 序列建模 | 序列模块（DIN/LONGER）输出作为 Token | FIM 内 Cross-Attention 与序列 KV 交互 |
| 多任务 | 共享 Backbone + 多 Head | TIM 专门任务交互模块 |
| Token 语义 | 语义分组（有明确语义身份） | 语义分组（FIM） |

三者均出自字节/快手，代表了推荐 Scaling 的不同方向：**TokenMixer-Large 是工程极致优化**，**UniMixer 是数学理论统一**，**UniFormer 是架构全面统一**。

---

## 六、TokenMixer-Large vs RankMixer：远不止 Mixing & Reverting

Mixing & Reverting 是最核心的结构性修复（解决残差维度 $T \neq H$ 不匹配问题），但 TokenMixer-Large 对 RankMixer 的升级是**全系统、多维度**的：

| # | 模块 | RankMixer | TokenMixer-Large |
|---|------|-----------|------------------|
| 1 | **Mixing 机制** | 单次 Mixing，Token 数 $T \to H$，残差维度不一致 | **Mixing & Reverting** 对称两层面，保证 $T \to H \to T$ 循环 |
| 2 | **FFN** | Pertoken FFN（GELU） | **Pertoken SwiGLU**（"we upgrade the pertoken FFN to a pertoken SwiGLU"） |
| 3 | **归一化** | Post-Norm LayerNorm | **Pre-Norm RMSNorm**（吞吐 +8.4%，Post-Norm 在深层训练中易 NaN） |
| 4 | **间隔残差** | ❌ 无 | **Inter-Residual**（每 2~3 层跨层连接，缓解深层梯度衰减） |
| 5 | **辅助损失** | ❌ 无 | **Auxiliary Loss**（低层 logits 参与联合训练，防止低层参数训练不足） |
| 6 | **MoE** | Dense Train, Sparse Infer（ReLU-MoE） | **Sparse-Pertoken MoE**（Sparse Train + Sparse Infer，1:2 零精度损失） |
| 7 | **碎片算子** | 保留 DCN、LHUC 等 | **Pure Model**（全部移除，大模型自身吸收其增益，MFU 达 60%） |
| 8 | **Tokenization** | 语义分组 + 均匀切分 | 语义分组 MLP + **Global Token**（BERT `[CLS]` 式全局聚合） |
| 9 | **分布式策略** | 无 | **Token Parallel**（Pertoken 操作按 Token 维度切分，推理吞吐 +97%） |
| 10 | **量化** | 无 | **FP8 E4M3 后训练量化**（1.7× 推理加速，无精度损失） |
| 11 | **初始化** | 标准初始化 | **Down-Matrix Small Init**（$W_{\text{down}}$ stddev 降至 0.01，改善深层收敛） |
| 12 | **规模** | ~1B | 7B（在线）/ 15B（离线） |

### 6.1 Mixing & Reverting 为什么是核心？

RankMixer 单层 Mixing 后 Token 数从 $T$ 变为 $H$，而残差连接要求输入输出维度一致。这导致：

| 残差属性 | 含义 | RankMixer | TokenMixer-Large |
|------|------|:---:|:---:|
| SR（Standard Residual） | 标准残差连接 | ✅ | ✅ |
| OTR（Original Token Retention） | 原始 Token 语义保留到最终输出 | ❌ | ✅ |
| TSA（Token Semantic Alignment） | 残差前后 Token 语义对齐 | ❌ | ✅ |

Mixing & Reverting 修复了 OTR 和 TSA 两个根本缺陷，使得残差信号可以稳定跨层传播，这是**其他所有升级（深网络、大模型、MoE 稀疏化）得以生效的前提条件**。

## 七、TokenMixer 对序列特征的处理

### 7.1 输入覆盖范围

TokenMixer 处理的是**全量特征 Token**，不仅限于 Target 侧：

```
输入 Token 组成：
├── User Features（用户画像）
├── Item Features（候选物品）
├── Cross Features（交叉特征）
└── Sequence Features（序列特征）← 经外部序列模块预压缩
```

Architecture 图标题明确指出：

> "Raw tokens include all original features as well as features from **sequence aggregation and extraction** (such as DIN/LONGER)."

### 7.2 序列特征的两阶段处理

TokenMixer **内部不包含序列建模能力**，序列特征需要外部模块预压缩：

```
用户行为序列（变长：点击/购买/曝光历史）
    │
    ▼
┌──────────────────┐
│  外部序列模块      │  DIN（短期）/ SIM（长期）/ LONGER（超长期）
│  Sequence Module  │  变长序列 → 固定长度 Embedding e_s
└──────────────────┘
    │
    ▼  固定长度 Embedding（序列信息已被"压平"）
┌──────────────────┐
│  Tokenization    │  与 User/Item/Cross Embedding 混合
│  语义分组 → Token │  作为普通 Token 送入 Backbone
└──────────────────┘
    │
    ▼
TokenMixer Block（纯特征交互，不感知时序结构）
```

**关键问题**：TokenMixer 收到序列 Embedding 后，通过 reshape shuffle 的 Mixing & Reverting 与其它 Token 混合。它对 TokenMixer 来说是**一个普通 Token**——与"用户性别 Token"在交互方式上没有区别，不感知该 Token 携带着行为序列的时序信息。

### 7.3 与 UniFormer 的序列处理对比

| | TokenMixer-Large | UniFormer |
|---|---|---|
| **序列建模位置** | **模块外**：DIN/LONGER 独立处理 | **模块内**：FIM 的 Cross-Attention 直接与序列 KV 交互 |
| **序列信息粒度** | 粗：整个序列被压缩为一个固定 Embedding | 细：每个序列 item 保留为 KV 对，可细粒度 Attend |
| **与 Backbone 耦合度** | 松散：序列模块输出只是 Backbone 的一个 Token 输入 | 紧密：Cross-Attn 是 FIM 内部组件，与特征交互深度融合 |
| **扩展方式** | 单独升级序列模块（DIN → LONGER） | 随 Backbone 统一扩展（Lazy KV、动态 Key 裁剪） |

### 7.4 为什么选择外部序列模块？

这是 TokenMixer 系列的**刻意设计选择**——保持 Backbone 的"纯净性"（Pure Model），不引入需要处理变长序列的复杂算子：

- 序列建模是 I/O 密集型操作（变长、稀疏 lookup），与 Backbone 的 Compute-Bound GEMM 特征不匹配
- 外部序列模块独立迭代（DIN → SIM → LONGER），不影响 Backbone 架构
- Backbone 只关注**固定长度 Token 间的特征交互**，更容易做到高 MFU（60%）

**代价**是序列信息的细粒度交互能力弱于 UniFormer——UniFormer 的 Cross-Attention 可以直接在序列 item 级别做注意力，而 TokenMixer 只能在一个压缩过的 Embedding 上做交互。
