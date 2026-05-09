# UniMixer: 推荐系统统一扩展架构

论文链接：https://arxiv.org/abs/2604.00590
发表会议：NeurIPS 2024
作者单位：快手科技（Kuaishou Technology）

---

## 一、研究动机与问题定义

大语言模型（LLM）的 Scaling Laws 揭示：随着模型规模、数据量和计算资源的增加，性能会持续可预测地提升，这激励了推荐系统社区探索适合推荐任务的扩展框架。

推荐系统与 NLP 的核心差异：NLP 所有 token 共享统一的 embedding 空间，而推荐系统的特征空间天然**异构**（用户画像、物品特征、行为序列、Query 特征等来自不同语义域），因此必须专门设计**异构特征交互**模块。

当前三类主流推荐扩展架构：
- **Attention-Based**（HiFormer、FAT、HHFT）：为每个 token 构建 token-specific Q/K/V 投影
- **TokenMixer-Based**（RankMixer、TokenMixer-Large）：使用基于规则、无参数的 token mixing 操作
- **FM-Based**（Wukong、Kunlun）：引入 FM Block 对输入 embedding 之间进行交互建模

**核心问题**：能否构建一个统一的推荐扩展模块，兼具三类方法的优势？

---

## 二、核心方法与贡献

### 2.1 TokenMixer 的等价参数化

论文发现 TokenMixer 操作等价于置换矩阵 W_perm 与展平输入的乘积，该置换矩阵具有以下关键性质：
- 可压缩性：可分解为 Kronecker 积 G x I，参数量从 O(T^2 D^2) 大幅压缩
- 双随机性：每行每列之和均为 1
- 稀疏性：每行/列恰好只有一个非零元素
- 对称性：当 T=H 时为对称矩阵

### 2.2 UniMixing 模块

将规则化 TokenMixer 替换为可学习的参数化结构，分为全局交互矩阵 W_G（控制 block-to-block 的交互模式）和局部交互矩阵 W_Bi（为每个 block 分配独立的特征交互参数）。

通过优化计算流程，将复杂度从 O(L^2) 降低到 O(L^2/B + LB)，避免产生大中间变量。

参数约束：Sinkhorn-Knopp 迭代满足双随机性；温度系数 tau 控制稀疏性；对称化操作保证对称性。

### 2.3 统一理论框架

在 UniMixing 的框架下，三类主流方法可被统一：

方法 | 局部交互模式 | 全局交互模式
--- | --- | ---
Self-Attention | XW_V | softmax(QK^T/sqrt(d))
Heterogeneous Attention | 异构 V 投影 | softmax(异构 QK^T/sqrt(d))
TokenMixer | X（无参数）| 固定置换矩阵 G
FM (Wukong) | 固定矩阵 Y | XI(XI)^T

UniMixing 相当于同时具有可学习的局部交互（类 Attention）和低参数化的全局交互（类 TokenMixer）。

### 2.4 UniMixing-Lite（轻量化版本）

- 局部交互：引入 basis 分解，用 b 个基矩阵的线性组合动态生成每个 block 的局部权重
- 全局交互：使用低秩分解替代完整矩阵，进一步压缩参数和计算量

### 2.5 SiameseNorm 与训练策略

引入 SiameseNorm（双流耦合归一化）解决深层网络训练稳定性问题，缓解 Pre-Norm 和 Post-Norm 之间的矛盾。

温度退火策略：高温（tau=1.0）初始化 → 线性退火至低温（tau=0.05） → 支持 warm-up 预训练再 fine-tune。

---

## 三、实验结果

### 3.1 实验设置

- 数据集：快手广告投放场景真实日志，超过 7 亿用户样本，跨越一年
- 任务：预测用户次日留存（User Retention）
- 指标：AUC、UAUC、参数量、FLOPs
- 硬件：40 GPU 混合分布式训练

### 3.2 主要性能对比（约 100M 参数规模）

模型 | AUC | ΔAUC | 参数量
--- | --- | --- | ---
Heterogeneous Attention（基线）| 0.7446 | - | 132.7M
RankMixer | 0.7493 | +0.475% | 135.5M
TokenMixer-Large | 0.7484 | +0.383% | 103.3M
UniMixer-Lite-4-Blocks 38.2M | 0.7523 | +0.775% | 38.2M
UniMixer-Lite-4-Blocks 84.5M | 0.7527 | +0.814% | 84.5M

UniMixer-Lite 以更少的参数（38.2M vs 135.5M）取得了明显更高的 AUC。

### 3.3 Scaling Law 拟合

模型 | 参数 scaling 指数 | FLOPs scaling 指数
--- | --- | ---
RankMixer | 0.116 | 0.117
UniMixer | 0.132 | 0.126
UniMixer-Lite | 0.142（最优） | 0.135（最优）

UniMixer-Lite 的 scaling 指数最大，每新增单位参数量能带来最大的性能收益。

### 3.4 消融实验（6.57M 参数模型）

设置 | ΔAUC
--- | ---
完整 UniMixer | -
去掉温度系数 | -0.1645%（影响最大）
去掉 Warm-Up | -0.0856%
去掉对称约束 | -0.0573%
去掉 block-specific 局部权重 | -0.0436%
SiameseNorm → Post Norm | -0.0273%

### 3.5 深度 Scaling 对比

模型 | AUC | 参数量
--- | --- | ---
RankMixer-2-Blocks | 0.7478 | 4.44M
RankMixer-4-Blocks | 0.7467 | 8.66M（性能下降！）
UniMixer-Lite-2-Blocks | 0.7492 | 4.97M
UniMixer-Lite-4-Blocks | 0.7508 | 9.72M（稳定提升）

RankMixer 在深层堆叠时出现性能下降，而 UniMixer 借助 SiameseNorm 可持续受益于深度扩展。

### 3.6 在线 A/B 测试

在快手多个广告投放场景部署，30 天累计活跃天数（CAD D1-D30）多场景平均提升超过 15%。

---

## 四、主要结论

1. **理论统一**：首次建立统一框架，将三类主流推荐扩展方法（Attention/TokenMixer/FM）纳入同一理论体系
2. **性能领先**：UniMixer-Lite 在参数效率和计算效率上均超越现有 SOTA，scaling 指数最优
3. **工业验证**：在快手真实广告场景取得显著业务指标提升（CAD 平均 +15%）
4. **深度扩展**：通过 SiameseNorm 解决深层堆叠训练稳定性问题

---

## 五、相关性与应用价值

- **直接相关**：推荐系统 Scaling Laws、特征交互建模、异构特征学习
- **技术迁移**：UniMixer 模块可扩展到用户行为序列建模和生成式推荐任务
- **工业指导**：提供了推荐系统从 Attention/TokenMixer/FM 三条路线走向统一的实践路径
- **参数效率**：UniMixing-Lite 的 basis 分解和低秩近似为大规模推荐模型的参数压缩提供了新思路

---

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
- 全局混合 G(X, W_G)：决定 token 之间的交互强度

三类方法的核心差异在于全局混合是否依赖输入 X：
- TokenMixer：固定置换矩阵，完全不依赖 X，彻底规避异构内积问题
- Attention/FM：动态计算，依赖 X，存在异构内积的语义隐患
- UniMixer-Lite：参数化但不依赖 X，介于两者之间，兼顾可学习性和异构安全性

FM 是 Attention 的特殊退化：令 W_Q=I, W_K=I, V=Y（固定），Attention 退化为 FM。

### 6.4 UniMixing-Lite 的参数设计

局部混合的参数化：
- Z（基矩阵集合）：shape (b, B, B)，b 个共享基矩阵，所有 block 公用，是局部交互模式的"词典"
- omega（组合系数）：shape (N, b)，每个 block 独有，决定如何从词典中组合出专属的 W_B^{*i}

定义位于 Eq. 8 正文（公式前的文字），W_B^{*i} 的计算公式位于 Eq. 8 之后的 where 从句。

全局混合的参数化：A_G (N, r) 和 B_G (r, N) 的低秩分解，Sinkhorn 约束后得到 W_r (N, N)。

### 6.5 局部混合与 Heterogeneous Attention 的关系

两者在数学结构上同族（论文 Eq. 7 中已证等价性），但 UniMixing-Lite 是轻量受限版本：

- 视野更窄：只在 block 内部（维度 B）做投影，而 Attention 的 V 投影作用在完整 token（维度 D），跨维度信息流动更丰富
- 自由度更低：W_B^{*i} 被约束为双随机矩阵（行列和为 1，接近置换），不能做缩放和降维；Attention 的 W_V 是完全自由的实数矩阵

这是论文用来换取参数效率的代价。

### 6.6 TokenMixer 在代码中的体现

TokenMixer 思想体现在 Step 4 的全局混合环节（W_r @ H）。原始 TokenMixer 是 UniMixing-Lite 全局混合的特殊退化情形：
- 令 W_B^{*i} = I（局部矩阵退化为单位矩阵，H = X_blocks）
- 令 W_r 为固定置换矩阵（不学习）

满足这两个条件，Step 4 就退化为纯粹的 TokenMixer shuffle 操作。

## Appendix
[AI Implementation](WorkSpace/unimixer_lite.py)