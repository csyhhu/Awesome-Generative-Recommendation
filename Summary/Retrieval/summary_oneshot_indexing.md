# OneShot：大规模检索中的排序内嵌索引与神经评分

> **论文**: OneShot: Index-in-Ranking with Neural Scoring for Large-Scale Retrieval
> **arXiv**: <https://arxiv.org/abs/2607.27475>
> **作者**: Ziwei Li, Shuyao Li, Xufeng Cai, Xue Zou, Yiming Ma, Huiting Lu, Wujie Yan, Zhichen Zhao, Yang Lu, Zhe Wang, Rui Luo, Zhengyu Su, Dan Zhang, Yimin Tan, Ji Liu (Meta Platforms, Inc.)

***

## 一、研究背景与核心问题

### 1.1 现代推荐系统架构

现代推荐系统通常采用级联架构，分为两个核心阶段：

- **召回阶段（Retrieval）**：将数十亿候选物品过滤到数千个，关注召回率（recall）
- **排序阶段（Ranking）**：对召回的候选集进行精排，关注精确率（precision）

### 1.2 核心矛盾

传统检索系统存在一个根本性的\*\*目标错位（Misalignment）\*\*问题：

| 目标                                 | 优化方向      | 描述                             |
| ---------------------------------- | --------- | ------------------------------ |
| 排序目标 $\mathcal{L}\_{\text{rank}}$  | 用户-物品交互对齐 | 使模型预测与用户行为一致                   |
| 索引目标 $\mathcal{L}\_{\text{index}}$ | 物品嵌入空间邻近性 | 基于 $k$-means/HNSW 等方法对物品进行聚类分组 |

这两个目标本质上是**解耦的**：排序目标优化用户行为预测，而索引目标优化物品嵌入的空间分组。这种错位导致：

1. **索引只能用简单的点积相似度**：由于索引是基于物品嵌入的空间邻近性构建的，限制了交互建模只能使用简单的点积
2. **交互表达能力受限**：无法在召回阶段引入更复杂的神经网络交互（如交叉注意力）

### 1.3 现有方法的局限

| 方法               | 排序-索引一致性 | 交互规模扩展 |
| ---------------- | :------: | :----: |
| $k$-means ANN    |     ✗    |    ✗   |
| HNSW / NANN      |     ✗    |    ✓   |
| Streaming VQ     |     ✓    |    ✗   |
| **OneShot (本文)** |   **✓**  |  **✓** |

***

## 二、OneShot 框架核心设计

### 2.1 模型内嵌层级索引（In-Model Hierarchical Index）

OneShot 提出了一种**端到端可训练的层级索引框架**，将索引学习与排序目标统一：

#### 核心思路

- 将物品表示分解为**连续嵌入**和**一热编码嵌入**两部分
- 一热编码由\*\*可训练的码本（Codebook）\*\*生成，作为索引的基础
- 整个过程在排序损失上端到端联合训练

#### 前向传播

1. 物品塔生成中间嵌入 $\mathbf{v}\_{\text{inter}}$
2. 投影层生成连续嵌入 $\mathbf{v}^{(\ell)}$ 和稠密嵌入 $\mathbf{v}\_d$
3. 计算码本分配分数 $\mathbf{z}^{(\ell)} = (\mathbf{C}^{(\ell)})^\top \mathbf{v}^{(\ell)}$
4. 通过 Softmax 得到软分配概率 $\mathbf{p}^{(\ell)}$
5. 通过 $\arg\max$ 得到硬一热编码 $\mathbf{e}^{(\ell)}$

#### 反向传播

使用**直通估计器（Straight-Through Estimator, STE）**：

- 前向：使用硬分配 $\mathbf{e}^{(\ell)}$
- 反向：使用软分配 $\mathbf{p}^{(\ell)}$ 传递梯度

#### Boosting 机制

多层索引采用**残差 Boosting** 策略：

- 每层在 logit 维度累积前面层的预测
- 层数越深，预测精度越高
- 最终公式：$\mathcal{L}_{\text{OneShot}} = \sum_\ell \mathcal{L}_{\text{rank}}(\sum_{\ell' \leq \ell} s\_{\ell'}(\mathbf{u}, \mathbf{v}_c^{(\ell')})) + \mathcal{L}_{\text{rank}}(s\_d(\mathbf{u}, \mathbf{v}_d)) + \mathcal{L}_{\text{bal}}$

### 2.2 神经评分交互扩展（Interaction Scale-Up）

#### 核心创新

通过将一热编码从重建损失中解耦，首次在召回阶段引入**神经网络评分函数**：

$$s\_\ell(\mathbf{u}, \mathbf{v}_c^{(\ell)}) = \text{NN}_\ell(\mathbf{u}, \mathbf{v}\_c^{(\ell)}), \quad s\_d(\mathbf{u}, \mathbf{v}\_d) = \text{NN}\_d(\mathbf{u}, \mathbf{v}\_d)$$

#### 高效计算

为解决 $O(B^2)$ 的计算瓶颈，引入**分段多头矩阵乘法**：

- 将嵌入 reshape 为矩阵 $\mathbf{U}$ $\[d', m]$ 和 $\mathbf{V}$ $\[d', n]$
- 计算 $\text{NN}(\text{flatten}(\mathbf{U}^\top \mathbf{V}))$
- 实验中使用 $d'=256, m=12, n=1$

#### 可扩展性

- 宽度 $m$ 和深度 $K$ 均可调
- 使排序模型的扩展技术（如交叉注意力、目标感知机制）可直接应用于召回

### 2.3 全局索引平衡（Global Index Balancing）

#### 问题

模型内嵌一热编码容易出现**索引崩塌**：大量物品集中到少数几个簇，导致：

- 码本表达能力浪费
- 服务时少数过载簇主导检索结果

#### 理论基础

基于\*\*随机组合优化（Stochastic Compositional Optimization, SCGD）\*\*理论：

优化目标：
$$\text{KL}\left(\mathbb{E}_{j \sim \mathcal{D}_{\text{item}}}\[\mathbf{e}\_j] | \text{Unif}(N)\right)$$

#### 代理损失推导

推导出一个**可直接嵌入现有优化器**的标量代理损失：
$$\mathcal{L}_{\text{bal}} = \frac{1}{|\mathcal{B}_{\text{item}}|} \sum\_{\ell=1}^L \sum\_{i \in \mathcal{B}\_{\text{item}}} \langle \mathbf{p}\_i^{(\ell)}, \log(\text{stopgrad}(\hat{\mathbf{q}}^{(\ell)})) \rangle$$

#### 多层联合平衡

- **边际平衡**：每层独立平衡
- **联合路径平衡**：跨层路径的联合分布平衡

***

## 三、实验结果

### 3.1 离线性能

#### 召回率对比（1% ranking volume 操作点）

| 方法                       | Recall                        |
| ------------------------ | ----------------------------- |
| $k$-means ANN (baseline) | 0.3414                        |
| OneShot 1-layer          | **0.4128** (+20%)             |
| OneShot 2-layer (效率提升)   | 同等召回率下 ranking volume 降低 10 倍 |

#### 交互扩展效果

| 宽度 $m$ | 深度 $K$ |   Recall   |
| :----: | :----: | :--------: |
|    3   |    0   |   0.4016   |
|    6   |    1   |   0.4374   |
|   12   |    1   |   0.4519   |
|   24   |    2   | **0.4613** |

#### 索引平衡对比

| 方法                   | Recall     | Max/Mean  | Std/Mean |
| -------------------- | ---------- | --------- | -------- |
| Baseline (无平衡)       | 0.2420     | 1281.26   | 16.91    |
| Streaming VQ         | 0.4279     | 36.20     | 1.04     |
| **OneShot KL-based** | **0.4679** | **10.20** | **0.60** |

### 3.2 线上 A/B 测试结果

OneShot 单层设置已在 Instagram 全球生产环境部署：

| 指标            | 提升          |
| ------------- | ----------- |
| 用户会话数（每日上限15） | +0.035%     |
| 观看时长          | +0.136%     |
| 曝光后互动         | +0.278%     |
| 点赞数           | +0.251%     |
| 1 日新鲜度        | +5.929%     |
| **检索来源占比**    | **+61.58%** |

### 3.3 Engagement ID (EID)

将一热编码集合 ${\mathbf{e}^{(\ell)}}$ 作为物品 ID，称为 **Engagement ID (EID)**：

- 与推荐目标联合训练，比语义 ID（SID）更适合检索

***

***

## 五、对生成式推荐的启示

- EID 显著优于离线生成的 SID，说明**与交互目标对齐的 ID 学习**是关键
- 混合语义和交互信息的表示可能是生成式推荐器的最佳方向
- OneShot 框架可直接用于生成式推荐系统中的 item token 表示学习

***

## 六、Q\&A 讨论记录

### Q1: 连续嵌入 $\mathbf{v}^{(\ell)}$ 和一热编码嵌入 $\mathbf{v}\_c^{(\ell)}$ 有什么区别？一热编码嵌入也会通过索引形成连续嵌入

**答**：你的观察非常准确——$\mathbf{v}\_c^{(\ell)} = \mathbf{C}^{(\ell)} \mathbf{e}^{(\ell)}$ 确实是一个连续向量（即码本的某一列）。两者的核心区别在于**值域约束**：

| <br />   | 连续嵌入 $\mathbf{v}^{(\ell)}$ | 一热编码嵌入 $\mathbf{v}\_c^{(\ell)}$ |
| -------- | -------------------------- | ------------------------------- |
| **值域**   | $\mathbb{R}^d$ 中任意连续值      | 只能取码本中 $N\_\ell$ 个离散向量之一        |
| **生成方式** | 投影层 $h\_\ell$ 自由生成         | $\arg\max$ 选择码本中最近的列            |
| **梯度传递** | 直接来自排序损失                   | 通过 STE（直通估计器）间接获得梯度             |
| **类比**   | 编码器输出                      | VQ 解码器的量化输出                     |

简单说：$\mathbf{v}^{(\ell)}$ 是"自由的"连续表示，而 $\mathbf{v}\_c^{(\ell)}$ 是"量化后的"连续表示——被约束在一个离散流形上。这与 VQ-VAE 中编码器输出和量化嵌入的关系完全一致。

### Q2: 中间嵌入 $\mathbf{v}\_{\text{inter}}$ 的输入是什么？

**答**：$\mathbf{v}\_{\text{inter}}$ 来自标准的 **Item Tower（物品塔）**。*注意：论文未显式列出 Item Tower 的具体输入特征，以下基于行业通用做法和论文上下文推断：*

- **物品 ID 嵌入** (item\_id embedding)
- **内容特征**（如视频的视觉/音频特征、文本描述的 embedding）
- **上下文特征**（如发布时间、作者信息、品类）
- **统计特征**（如历史曝光、点击计数等）

这些特征通过 Item Tower（标准的多塔神经网络编码器）融合成统一的 $\mathbf{v}\_{\text{inter}}$，然后再通过三个投影层分别生成：

- $\mathbf{v}^{(1)}, \mathbf{v}^{(2)}, \dots$：量化层输入（带 stopgrad，防止平衡损失干扰）
- $\mathbf{v}\_d$：稠密嵌入（无梯度截断，用于最终评分）

### Q3: 训练和推理全流程示例

**场景**：用户 u 喜欢猫咪视频，物品 i 是一个 30 秒的橘猫玩耍视频，2 层码本（$N\_1=4, N\_2=4$）

#### 训练阶段（以单个用户-物品对为例）

**Step 1 — 物品塔编码**

```
物品特征（ID + 视频内容 + 作者 + 时间等）
    → Item Tower（多层 MLP / Transformer）
    → v_inter  (例如 256 维连续向量)
```

**Step 2 — 投影分解**

```python
v1 = h1(stopgrad(v_inter))   # 量化层1的连续输入
v2 = h2(stopgrad(v_inter))   # 量化层2的连续输入
vd = hd(v_inter)              # 稠密嵌入（无stopgrad）
```

**Step 3 — 第 1 层量化**

```python
z1 = C1.T @ v1                # 计算与4个码字的内积: [3.2, 1.5, 0.8, -0.3]
p1 = softmax(z1 / T)          # 软分配概率: [0.87, 0.11, 0.02, 0.00]
e1 = argmax(p1)               # 硬分配: [1, 0, 0, 0]（选了码字0）
vc1 = C1[:, 0]                # 量化嵌入 = 码本第0列
```

**Step 4 — 第 2 层量化**

```python
z2 = C2.T @ v2                # 计算与4个码字的内积
e2 = argmax(softmax(z2 / T))  # 硬分配: [0, 0, 1, 0]（选了码字2）
vc2 = C2[:, 2]                # 量化嵌入 = 码本第2列
```

**Step 5 — 用户塔编码**

```
用户特征（ID + 历史观看序列 + 上下文等）
    → User Tower
    → u  (256 维用户嵌入)
```

**Step 6 — 神经评分与损失**

```python
s1 = NN1(u, vc1)              # 第1层评分
s2 = NN2(u, vc2)              # 第2层评分
sd = NN_d(u, vd)              # 稠密评分

# Boosting: 逐层累积预测
loss = L_rank(s1)              # 层1损失
     + L_rank(s1 + s2)         # 层2损失（包含层1+层2）
     + L_rank(sd)              # 稠密损失
     + L_bal                   # 平衡正则项
```

**Step 7 — 反向传播（STE）**

```python
# 稠密路径（正常反传）
loss → sd → NN_d → vd → hd → v_inter → Item Tower

# 量化路径（STE反传）
loss → s1 → NN1 → vc1
                  ↓ STE: 用 p1 替代 e1 的梯度
                 C1 和 v1 通过 p1 获得梯度

# 平衡路径（SCGD）
L_bal → 更新码本使用统计 q_hat → 影响码本分配
```

#### 推理（服务）阶段

**Step 1 — 用户嵌入计算（每次请求）**

```
用户特征 → User Tower → u
```

**Step 2 — Beam Search 索引选择**

```python
# 计算用户与所有码本的交互分数
for k in range(4):
    score1[k] = s1(u, C1[:, k])

# 假设 score1 = [3.2, 1.5, 0.8, -0.3]
# Beam width = 2，选 top-2 第1层码字: k=0, k=1

# 对每个第1层码字，计算第2层
for k1 in [0, 1]:
    for k2 in range(4):
        score_total = score1[k1] + s2(u, C2[:, k2])
    # 每个第1层选 top-2 第2层

# 最终 4 条路径: (0,1), (0,3), (1,0), (1,2)
# 合并这些路径关联的所有物品为候选集
```

**Step 3 — 稠密评分与排序**

```python
# 对候选集每个物品
for item in candidates:
    score = NN_d(u, vd[item])

# 排序输出 Top-K 给下游精排阶段
```

#### 训练 vs 推理对比总结

| 阶段     | 量化方式                    | 评分方式                 | 计算量                       |
| ------ | ----------------------- | -------------------- | ------------------------- |
| **训练** | $\arg\max$ 硬分配 + STE 反传 | NN 评分（所有物品 in-batch） | $O(B^2)$ + STE 开销         |
| **推理** | 预计算的码本索引 + Beam Search  | NN 评分（仅候选集）          | $O(N_1 + N_2)$ 码本查找 + $O(K)$ 候选评分 |

### Q4: 两层量化选出候选集的优势是什么？相比于直接用 dense embedding？

**答**：这个问题的核心在于——**正是因为 OneShot 引入了 NN 评分（突破了点积瓶颈），传统 ANN 索引失效了，所以才必须用量化码本作为可学习索引来初筛。**

#### 关键洞察：NN 评分打破了传统 ANN 的可行性

传统召回流程能高效运作，是因为评分函数是**点积**：

```
传统：dense embedding + 点积 → 可以用 k-means/HNSW 离线建索引 → O(log N) 检索
```

但 OneShot 引入了**神经网络评分** $\mathrm{NN}_d(\mathbf{u}, \mathbf{v}_d)$，这是一个**非线性函数**：

```
OneShot：dense embedding + NN_d 评分 → 无法用 k-means/HNSW 建索引！
```

因为 $k$-means/HNSW 的索引结构依赖于**嵌入空间的邻近性**（假设相似物品在空间中靠近），而 $\mathrm{NN}_d$ 是非线性的——**空间上靠近的物品，NN_d 评分可能差异巨大**。传统 ANN 索引在这种场景下完全失效。

#### 那为什么不用 dense embedding 逐个评分？

如果有 $N = 10$ 亿物品，每次用户请求都要计算 $10$ 亿次 $\mathrm{NN}_d$，这在工程上完全不可行。

#### 两层量化的解决方案

两层量化本质上构建了一个**可学习的倒排索引**：

```
全量物品 → 每个物品被分配到一个路径 (k1, k2)
         → 形成 N1 × N2 个桶（倒排列表）

用户请求 → 只评分 N1 + N2 个码本（几千次 NN_ℓ 计算）
         → Beam Search 选出几条路径
         → 只对这几条路径的物品做 NN_d 评分
```

#### 效率对比

| 方法 | 候选集大小 | 计算量 |
|------|-----------|--------|
| 直接用 dense + NN_d | $N = 10$ 亿 | $O(N)$ 次 NN 计算 ❌ 不可行 |
| 传统 ANN + 点积 | ~$N \times 1\%$ | $O(\log N)$ 检索 + 点积，但**不支持 NN 评分** |
| **OneShot 两层量化** | $N \times 0.06\%$ | $O(N_1 + N_2)$ 码本评分 + $O(K)$ NN_d 评分 ✓ |

论文数据显示：2-layer 设置在同等召回率下，ranking volume 仅为 ANN 的 1/10（0.061% vs 0.976%）。

#### 三层优势总结

1. **效率**：将 $O(N)$ 的全量 NN 评分降为 $O(\text{beam\_width})$ 的候选集评分
2. **兼容非线性评分**：传统 ANN 依赖点积，无法支持 $\mathrm{NN}_d$；量化索引不依赖空间邻近性，天然兼容
3. **目标对齐**：量化索引是端到端训练的，与排序目标一致，而传统 ANN 的 $k$-means 聚类与排序目标解耦

**一句话**：量化索引是"交互扩展"和"高效检索"能够共存的必要条件——NN 评分使得传统 ANN 失效，而可学习的量化码本填补了这一空缺。

