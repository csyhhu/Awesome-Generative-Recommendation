# Retrieval-GRPO: 面向淘宝搜索稠密检索的多目标强化学习框架

> 论文链接：https://arxiv.org/abs/2511.13885  
> 作者：Xingxian Liu, Dongshuai Li, Jiahui Wan et al.（阿里巴巴淘宝天猫事业群）  
> 关键词：Dense Retrieval, Large Language Model, Reinforcement Learning, E-commerce Search

---

## 1. 研究背景与问题

### 稠密检索在电商搜索中的地位
稠密检索（Dense Retrieval）是电商搜索引擎的核心模块，通过预训练嵌入模型将用户查询与商品映射到统一语义向量空间，再借助近似最近邻（ANN）算法实现大规模实时语义召回。

### 现有方法的两大痛点

1. **离线难负样本构建代价高**：现有方法（包括基于 LLM 的检索模型）仍沿用 BERT 时代的 SFT + 硬负样本挖掘训练范式。离线难负样本的构建是一套复杂的系统工程（需要静态旧模型采样 → 离线打分 → 过滤），严重制约了模型迭代速度和能力上限。

2. **多任务训练的 Seesaw 效应**：电商搜索不仅要优化语义相关性，还要同时优化品质分、互斥性等非相关性目标。多任务学习框架（多 loss 联合训练）存在"跷跷板效应"，各目标相互制约，限制了模型整体性能天花板。

---

## 2. 核心方法：Retrieval-GRPO

### 整体框架
Retrieval-GRPO 是一个基于强化学习的多目标稠密检索训练范式，分为两阶段：
- **阶段一：SFT（监督微调）**：在简单正负样本上赋予模型基础判别能力
- **阶段二：Retrieval-GRPO**：通过 RL 对长尾难样本进行实时纠错与多目标对齐

### Retrieval-GRPO 三步流程

#### Step 1：候选选取（Candidate Selection）
训练时，使用当前梯度更新中的稠密检索模型，通过 ANN 动态检索每条 query 的 Top-K 候选商品。**候选来自最新模型**（而非离线旧模型），天然包含当前模型的错误预测，形成"在线硬负样本"。

为降低对全量商品池做 ANN 的开销，实际在跨设备汇聚的大 Batch $\hat{B}$ 中取 Top-K（$k \ll \hat{B}$）。

#### Step 2：多目标奖励计算（Multi-Objective Reward）
$$r = f_{\text{relevance}}(q, d) + g_{\text{quality}}(d) + h_{\text{exclusivity}}(q, d)$$

| 奖励维度 | 来源 | 含义 |
|--------|------|------|
| **相关性奖励** $f$ | TaoSR1（42B MoE 相关性 LLM，由 RL 训练） | query-item 语义相关性实时打分 |
| **品质奖励** $g$ | 历史成交数据 + 用户满意度指标 | 商品客观质量分（离散值） |
| **互斥性奖励** $h$ | 与倒排索引召回通路的词汇重叠率 | 若商品已被倒排通路覆盖（高词汇重合），则给予较低奖励，鼓励稠密检索提供增量价值 |

#### Step 3：GRPO Loss 优化
采用 DeepSeek 提出的 GRPO 算法（无需 value model），对同一 query 下的 Top-K 候选商品计算**组间优势**：

$$\mathcal{J}_{GRPO}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\left\{\min\left[\frac{\pi_\theta}{\pi_{\theta_{old}}}\hat{A}_{i,t},\ \text{clip}\left(\frac{\pi_\theta}{\pi_{\theta_{old}}}, 1-\epsilon, 1+\epsilon\right)\hat{A}_{i,t}\right] - \beta\mathbb{D}_{KL}[\pi_\theta || \pi_{ref}]\right\}\right]$$

奖励低的商品被"软惩罚"（连续值而非硬标签），等效于自动发现并利用了难负样本。

### SFT 阶段的改进
在传统 InfoNCE in-batch 负采样基础上，引入**全局负采样**（从全量商品池随机抽取负样本），确保商品池中所有商品都能参与训练：

$$\mathcal{L} = -\log \frac{\exp(s(q_i, d_j^+)/\tau)}{\sum_{d_j \in \mathcal{B}} \exp(s(q_i, d_j)/\tau) + \sum_{d_k^- \in \mathcal{G}} \exp(s(q_i, d_k^-)/\tau)}$$

---

## 3. 实验结果

### 离线评估指标
- **Hitrate@6k**：稠密检索 Top-6000 结果中包含真实相关商品的比例
- **Goodrate@100**：Top-100 结果中被相关性模型判为 Good（L3/L4）的比例

### 主要结果（Hitrate@6000）

| 模型 | General | Long-Tail |
|------|---------|-----------|
| BERT-base | 32.28 | 19.10 |
| Qwen-3B | 45.08 | 33.67 |
| Tbstars SFT Only | 48.00 | 37.52 |
| Tbstars SFT + DPO(Hard Neg) | 48.20 | 39.35 |
| **Tbstars SFT + R-GRPO** | **49.35** | **41.07** |

### 主要结果（Goodrate@100，长尾体验集）

| 模型 | Q&A | Alternative | Negative | Knowledge | Overall |
|------|-----|-------------|----------|-----------|---------|
| Tbstars SFT Only | 84.01 | 38.70 | 62.60 | 68.45 | 63.44 |
| Tbstars SFT + DPO(Hard Neg) | 84.96 | 40.21 | 63.60 | 68.76 | 64.38 |
| **Tbstars SFT + R-GRPO** | **85.05** | **43.32** | **65.80** | **69.50** | **65.91** |

**结论**：R-GRPO 在 Overall Goodrate 上超越最佳 Baseline 约 **+2.47pt**，在 Alternative 类别（替代品查询）提升最显著（+3.11pt）。

### 消融实验关键结论
1. **多目标奖励有效**：仅使用相关性奖励会导致商品质量分下降（-0.15%），加入品质+互斥性奖励后，召回率提升的同时商品质量提升 **+11.64%**
2. **互斥性奖励降低重叠**：加入互斥性奖励后，与倒排通路的重叠率从 25.7% 降至 **17.6%**，稠密检索提供了更多增量覆盖
3. **Reward Model 质量至关重要**：用 4 层 BERT 相关性模型作为奖励模型几乎无增益，替换为 TaoSR1（42B MoE）后显著提升，说明**高质量奖励信号是 RL 范式的核心**

### 线上 A/B 测试结果

| 查询类型 | GSB | Query Goodrate | Item Goodrate |
|--------|-----|----------------|---------------|
| Q&A | +24.30% | +10.64pt | +8.2pt |
| Alternative | +28.57% | +10.87pt | +9.1pt |
| Negative（否定语义） | **+41.16%** | +17.07pt | +11.65pt |
| Knowledge | +25.5% | +12.65pt | +7.60pt |
| Overall | +29.88% | +12.81pt | +9.14pt |

---

## 4. 主要贡献与创新点

1. **将 GRPO 引入稠密检索训练**：首次将生成式 LLM 的强化学习训练范式（GRPO）适配到稠密检索的 Embedding 优化场景
2. **消除离线难负样本依赖**：动态在线采样取代复杂的静态难负挖掘流水线，加速模型迭代
3. **多目标融合奖励缓解 Seesaw**：通过统一奖励函数替代多 loss 联合训练，更灵活地平衡相关性、品质、互斥性
4. **工业级落地验证**：已在淘宝搜索（中国最大电商平台）全量部署，线上指标显著提升

---

## 5. 与本项目的关联

### 对生成式推荐研究的启发

- **RL 在检索/召回层的应用**：本文将 GRPO 这一原本用于 LLM 对齐的技术迁移到向量检索任务，为生成式推荐中的召回优化提供了新范式
- **多目标奖励设计**：如何在生成式推荐中同时优化相关性、多样性、商业化指标，本文的多目标奖励融合方案（直接加和 vs 多 loss）值得借鉴
- **LLM 作为 Reward Model**：用大规模 LLM 作为奖励信号来源（而非人工标注），是连接生成式 AI 与推荐系统的关键桥梁，可探索在生成式推荐的训练中类似使用 LLM 打分

### 潜在研究方向
- 将 Retrieval-GRPO 的思路扩展到生成式推荐的 Item Generation（直接生成 Item ID / 描述）场景
- 探索互斥性奖励在推荐系统中的类比：鼓励模型召回与协同过滤等其他通路形成互补
- 研究 Reward Model 能力与检索模型性能上限之间的关系（本文结论表明这是强约束）

---

## 6. 深入讨论笔记

### 6.1 整体框架的正确理解

本文的核心变化可以概括为：**模型架构不变（双塔+内积），训练分两阶段（SFT → GRPO），负样本获取方式从离线静态变为在线动态**。

具体而言：
- **模型架构**：仍是标准双塔 Dual-Encoder，共享编码器 $f_\theta$，相似度用余弦/内积计算，架构完全不变
- **训练阶段**：并非"SFT 替换为 GRPO"，而是两阶段串联。消融实验证明跳过 SFT 直接做 GRPO 效果反而更差（Hitrate 46.20 vs SFT Only 48.00），SFT 是建立基础判别能力的必要"热身"
- **核心变化**：负样本的产生机制（离线旧模型采样 → 在线当前模型 Top-K）和监督信号（二值 label → 连续奖励值）

### 6.2 GRPO 为何能解决难负样本采集问题

传统难负样本的核心矛盾是**分布偏移（Distribution Shift）**：

```
用旧模型 M_t 挖掘难负样本 → 训练得到 M_{t+1}
→ M_{t+1} 已进化，但负样本仍来自 M_t
→ 需要重新跑挖掘流程，工程代价极高
```

GRPO 通过 **on-policy 采样**天然规避了这个问题：每个 training step 的候选均由当前参数 $\theta$ 的模型实时检索，样本始终与当前模型能力分布对齐，无需离线维护。

此外，LLM 奖励分是**连续值**而非二值 label，使得同为"负样本"的商品也有优先级区分（如相关度 0.44 vs 0.29），GRPO 的组间优势 $\hat{A}_i$ 会自动体现这种差异，比传统硬标签信息量更丰富。

**需要注意的是**：这个"解决"有前提——你需要一个足够强的奖励模型。消融实验已证明用 4 层 BERT 相关性模型作为奖励几乎无效，必须达到 TaoSR1（42B MoE）级别的模型质量才能有效。

### 6.3 GRPO 详解：以本文样本为例

#### 样本组织形式

GRPO 以 **query 为单位组织"组（Group）"**，每组包含该 query 的 Top-K 候选商品及其奖励分。

以论文 Case Study 为例，query = `"方形不锈钢水箱"`，Top-K=4 候选：

```
候选组 G = {
  d1: 不锈钢方形水槽（正方形款）  → r1 = 0.99
  d2: 不锈钢方形水槽（长方形款）  → r2 = 0.99
  d3: 塑料水箱                   → r3 = 0.44
  d4: 水槽置物架                  → r4 = 0.29
}
```

#### 计算组间优势 $\hat{A}_i$

$$\hat{A}_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$$

代入数值（mean=0.6775，std≈0.316）：

| 候选 | 奖励 $r_i$ | 优势 $\hat{A}_i$ | 梯度方向 |
|-----|-----------|----------------|---------|
| d1 不锈钢方形水槽 | 0.99 | **+0.99** | 拉近（强化） |
| d2 不锈钢方形水槽 | 0.99 | **+0.99** | 拉近（强化） |
| d3 塑料水箱 | 0.44 | **-0.75** | 推远（抑制） |
| d4 置物架 | 0.29 | **-1.23** | 推远（强力抑制） |

#### GRPO Loss 与 clip 机制

$$\mathcal{J}_{GRPO}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G} \left\{ \min\left[ \rho_i \hat{A}_i,\ \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon)\hat{A}_i \right] - \beta \mathbb{D}_{KL}[\pi_\theta || \pi_{ref}] \right\} \right]$$

其中重要性权重 $\rho_i = \pi_\theta(s(q,d_i)) / \pi_{\theta_{old}}(s(q,d_i))$，策略分布定义为 Top-K 候选上的 Softmax：

$$\pi_\theta(s(q, d_i)) = \frac{\exp(s(q, d_i) / \tau)}{\sum_{j=1}^{K} \exp(s(q, d_j) / \tau)}$$

**clip 的本质**：当模型已在正确方向走得足够远时（如 $\hat{A}_i<0$ 且 $\rho_i$ 已很小），限制梯度继续大幅更新，防止过度优化导致策略崩塌。

#### 梯度直觉

最终梯度近似为：

$$\Delta\theta \propto \underbrace{(+0.99)}_{\text{拉近 d1}} \nabla s(q,d_1) + \underbrace{(+0.99)}_{\text{拉近 d2}} \nabla s(q,d_2) + \underbrace{(-0.75)}_{\text{推远 d3}} \nabla s(q,d_3) + \underbrace{(-1.23)}_{\text{强推远 d4}} \nabla s(q,d_4)$$

一次梯度步同时完成拉近高质量商品、推远低质量商品，且力度与奖励差距成正比。

#### 与传统 InfoNCE 对比

| | InfoNCE (SFT) | GRPO |
|--|-------------|------|
| 负样本来源 | 随机 in-batch | Top-K 在线难负（当前模型检索） |
| 监督信号 | 二值 label（0/1） | 连续优势值 $\hat{A}_i$ |
| 梯度策略 | 等力推远所有负样本 | 按奖励差距**差异化**拉近/推远 |
| 稳定性机制 | 无 | clip + KL 约束 |
| 本质 | 对比学习 | **带自适应权重的对比学习** |
