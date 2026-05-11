# Rank-GRPO：用强化学习训练基于 LLM 的对话式推荐系统（ConvRec-R1）

- 论文：["Rank-GRPO: Training LLM-based Conversational Recommender Systems with Reinforcement Learning"](https://arxiv.org/abs/2510.20150)
- 代码与数据：<https://github.com/yaochenzhu/Rank-GRPO>

## 1. 论文想解决什么问题

大语言模型（LLM）使「对话中表达偏好并直接生成推荐列表」成为可能，但与真实推荐任务对齐仍困难，作者归纳三类痛点：

1. **目录外（OOC）与幻觉**：预训练 LLM 不熟悉平台专属片单/目录，易生成不存在或不在库内的条目。
2. **格式不合规**：难以稳定输出下游匹配所需的结构化文本（例如电影需「片名 + 上映年份」）。
3. **列表尾部质量崩塌**：自回归生成时，排名越靠后的条目质量往往急剧下降（高质量「排序式」监督在预训练中稀缺）；小模型在工业延迟/成本约束下更突出。

此外，将 **RLVR（可验证奖励的强化学习）** 直接套到对话推荐上还有两类障碍：难以大规模获得「目录内、排序好」的人工示范做行为克隆 warm-start；以及 **GRPO 等算法与「排序式输出」结构不匹配**——序列级奖励（如整表 NDCG/DCG）过粗，却均匀分摊到每个 token；token 级更新又过细（一个条目多 token），造成 **非因果的信用分配**、重要性权重与优势估计错位、策略更新不稳定。

## 2. 方法总览：两阶段框架 ConvRec-R1

### 阶段 1：监督微调（SFT）—— Remap–Reflect–Adjust

用强教师黑盒 LLM（如 GPT-4o）在训练对话上生成零样本推荐，再通过三阶段管线得到 **锚定在目标目录 \(\mathcal{C}\)** 上的高质量排序示范，用于行为克隆。三步本质上是在 **全目录** \(\mathcal{C}\) 上维护并迭代一条分数向量 \(\bm{s}\in\mathbb{R}^{|\mathcal{C}|}\)，最后按 \(\bm{s}_{\text{final}}\) 取 Top-\(N\)，格式化成与自回归训练一致的文本（如「片名 (年份)」、条目间换行等）。

SFT 目标为对示范序列的负对数似然最小化，为后续 RL 提供 **目录感知、格式正确、初步排序能力**。

#### 输入与输出

| | 内容 |
|---|------|
| **输入** | 训练对话 \(x_i^{\text{SFT}}\)；教师按固定格式生成的初步列表 \(y_{i,\text{raw}}^{\text{SFT}}\)（条目落在教师推荐空间 \(\mathcal{C}_\Theta\)，可能含 OOC、格式不一、排序带教师偏见）。 |
| **输出** | 目录内 Top-\(N\) 排序示范 \(y_i^{\text{SFT}}\)，供行为克隆。 |

下文按 **Remap → Reflect → Adjust** 顺序展开（与论文正文及附录「Details for the Remap-Reflect-Adjust Pipeline」一致；实现针对电影任务，其它品类可类比）。

#### Step 1：Remap（映射进目录 + 粗排序）

**目的**：把「教师原始列表里出现了什么、排在第几位」与「目录里每一项长什么样、和对话多相关」融合成 \(\bm{s}_{\text{remap}}\)。

1. **元数据生成**（便于后续语义相似度）  
   对 \(\mathcal{C}_\Theta\) 与 \(\mathcal{C}\) 中每条目生成统一模板：**片名 (年份)、Keywords、Plots**。分三类：  
   - 教师**已知**的目录内条目：零样本 prompt 直接摘要；  
   - **OOC 或幻觉**条目：仍由教师生成「看似合理」的元数据，用于再软匹配回目录；  
   - 目录内有、教师**不熟**的条目（冷门/新片）：**RAG**（如 Google 检索 Wikipedia / IMDb / Rotten Tomatoes）+ **ICL** few-shot，再生成元数据。

2. **相似度**（论文附录使用句向量模型 **NovaSearch/stella_en_400M_v5**）  
   - **条目–条目**：\(\mathcal{C}_\Theta\) 与 \(\mathcal{C}\) 元数据嵌入的余弦相似度 → 矩阵 \(\bm{S}_{\text{item-item}}\in\mathbb{R}^{|\mathcal{C}_\Theta|\times|\mathcal{C}|}\)。  
   - **对话–条目**：整段对话嵌入 vs 每个目录条目元数据嵌入 → \(\bm{s}_{\text{context}}\in\mathbb{R}^{|\mathcal{C}|}\)（正文里也写作对话–条目信号 \(\bm{s}_{\text{conv-item}}\)，含义相同）。

3. **分数聚合**  
   - \(\bm{p}\in\mathbb{R}^{|\mathcal{C}_\Theta|}\)：仅在教师**原始列表**出现过的条目上非零，第 \(k\) 位为 **\(1/\sqrt{k}\)**，其余为 0。  
   - \(\bm{I}_{\text{ic}}\)：若教师第 \(u\) 项与目录第 \(v\) 项**完全一致**（精确匹配），则 \(I_{\text{ic}}[u,v]=1\)，把教师位次分直接打到对应目录项。  
   - **公式**（文中 \(\lambda=1\)）：  
     \[
     \bm{s}_{\text{remap}} = \bm{p}^\top (\bm{S}_{\text{item-item}} + \bm{I}_{\text{ic}}) + \lambda \cdot \bm{s}_{\text{context}}.
     \]

#### Step 2：Reflect（LLM-as-a-judge，补上下文）

**目的**：Remap 主要靠嵌入相似度，对多面向、弦外之音的偏好较弱；用教师对一批高分候选再做**对话条件下的相关性判别**。

1. 取 \(\bm{s}_{\text{remap}}\) 的 **Top-\(N_r\)**，其中 **\(N_r>N\)**（附录示例为 Top-100；\(N\) 为最终列表长度，如 20）。  
2. 将对话与候选片名交给教师，在 **五档** \(\{-2,-1,0,1,2\}\)（坏 → 极好）上逐条打分；解析为目录上的稀疏向量 \(\bm{r}_{\text{reflect}}\)。  
3. **公式**（文中 \(\gamma=0.5\)，使反射项大致落在 \([-1,1]\)，与 \(\bm{s}_{\text{remap}}\) 量级可比）：  
   \[
   \bm{s}_{\text{reflect}} = \bm{s}_{\text{remap}} + \gamma \cdot \bm{r}_{\text{reflect}}.
   \]

#### Step 3：Adjust（用训练集真实正例纠偏）

**目的**：缓解教师与前面步骤带来的**流行度 / 趋势**等残差，使分数场与 **训练对话中用户点正的真实片单**经验分布更一致。

1. 在**全目录**上学习共享的条目级 **乘性偏置** \(\bm{w}\) 与 **加性偏置** \(\bm{b}\)（维度均为 \(|\mathcal{C}|\)）。  
2. \(\bm{s}_{\text{final}} = \bm{w} \odot \bm{s}_{\text{reflect}} + \bm{b}\)（\(\odot\) 为逐元素乘）。  
3. \(\bm{w},\bm{b}\) 通过最大化训练集上 ground-truth 正例在 **softmax 全目录分布**下的多项对数似然来估计，并对 \(\|\bm{w}-\bm{1}\|_2^2\)、\(\|\bm{b}\|_2^2\) 做正则（文中系数 **0.01**）。每条对话的 \(\bm{s}_{\text{reflect}}\) 不同，\(\bm{w},\bm{b}\) 在全体样本上**共享**（全局条目纠偏）。  
4. 按 \(\bm{s}_{\text{final}}\) 取 **Top-\(N\)**，得到 \(y_i^{\text{SFT}}\)。

#### 三步对照（便于记忆）

| 步骤 | 在做什么 |
|------|----------|
| **Remap** | 教师说了什么、排在第几位 → 通过语义相似 + 精确匹配 + 对话相关，落到**目录上的分数场** \(\bm{s}_{\text{remap}}\)。 |
| **Reflect** | 对高分候选让教师**结合对话再判一遍**，修正 \(\bm{s}_{\text{reflect}}\)。 |
| **Adjust** | 用数据里用户**真喜欢的片**学 \(\bm{w},\bm{b}\)，得到 \(\bm{s}_{\text{final}}\)，拉齐流行度等分布偏差。 |

附录含完整 prompt 与定性示例（例如教师零样本中多部电影为 OOC，Remap 后变为目录内可匹配片名，再经后续步骤改善排序）。

### 阶段 2：RL 对齐—— Rank-GRPO

在 RL 数据 \(\mathcal{D}_{\text{RL}}\) 上，每条对话 \(x_i^{\text{RL}}\) 配有用户给出正反馈的目录内集合 \(y_i^{\text{gt}}\) 作为监督信号。作者将 **「每个排名位置 \(k\)」** 视为自然动作单元（介于 token 与整序列之间），提出 **Rank-GRPO**：

- **Rank 级优势**：在同一对话下从旧策略采样 \(G\) 条完整列表，对 **第 \(k\) 位** 用组内相对标准化定义 \(\hat{A}_{i,k}\)（基于 rank 级回报 \(r(x, y_i^{(k)})\) 在 \(i'\) 上减均值除标准差），而非整表单一 \(\hat{A}_i\) 复制到每个 token。
- **Rank 级重要性比**：  
  \(\bar{\pi}_\theta(y_i^{(k)}|x) = \big(\prod_t \pi_\theta(y_{i,k,t}\mid x, y_{i,k,<t})\big)^{1/|y_i^{(k)}|}\)，即该位条目 token 概率的 **几何平均**，再与 \(\theta_{\text{old}}\) 比值得到 \(w_{i,k}(\theta)\)，缓解不同条目 token 长度差异带来的不稳定；目标形式类似 PPO/GRPO 的 clip。
- **回报设计（奖励塑形）**：指出整表 DCG@\(N\) 可分解为「对第 \(k\) 位非因果」的前缀部分与「因果」的后缀部分。Rank-GRPO 采用 **DCG@\(k\!:\!N\)**（从 \(k\) 到 \(N\) 的折扣相关性和）作为第 \(k\) 位的 rank 回报，使信用分配与自回归生成方向一致。另给出指数衰减变体 **Rank-GRPO (\(\exp_\Gamma\))**；\(\Gamma=\infty\) 时仅强调当前位相关性，文中记 **Rank-GRPO (\(\exp_\infty\))**，实践里简单稳定。另对 **列表过短**（提前 `<eos>`）与 **超长溢出** 施加负奖励以约束条数。

与 vanilla GRPO 的梯度对比：Rank-GRPO 在 **rank 级** 对齐重要性比、优势与 \(\nabla \log\) 聚合（对第 \(k\) 个条目的 token 梯度取平均），从机制上缓解序列级奖励 + token 级更新带来的错配。

## 3. 实验设置与主要结论

- **主数据集**：公开 **Reddit-v2**（约 40 万对话级 CRS 基准），划分 train/val/test 为 383,013 / 9,421 / 10,972；与 prior 不同之处在于要求输出 **片名+年份**，利于消歧与目录匹配。SFT 用训练集 **25%** 及全验证集走 Remap–Reflect–Adjust，并发布 SFT 数据与 checkpoint。
- **骨干**：Qwen2.5-0.5B-Instruct、Llama-3.2-1B-Instruct、Llama-3.2-3B-Instruct（侧重可部署延迟）。
- **现象**：SFT 后目录内比例快速超过 99%，相对零样本 NDCG@20 可提升数倍到数十倍；RL 阶段 Rank-GRPO 在训练/验证上 **各 rank 回报更单调改善**，尤其 **大 \(k\)（列表后部）** 相对 GRPO/GSPO 优势更明显；离策略（如 \(\mu=2\)）下方差更大但 Rank-GRPO 仍优于 GRPO/GSPO。
- **测试集**：相对传统 CRS、零样本 GPT-4o-mini / GPT-4o、以及 CRAG 等强 prompt 管线，**SFT + Rank-GRPO** 在 Recall@\(k\) / NDCG@\(k\) 上整体更优；例如 **0.5B** 可超过 GPT-4o-mini，**1B** 可与 GPT-4o 比肩，**3B** 在 @20 上可超过 GPT-4o 与 CRAG，且推理成本低于多次调用 GPT-4o 的 RAG 式管线。
- **消融**：去掉 Remap 或 Reflect 或整段 SFT 均显著损伤；完整 Remap–Reflect–Adjust + SFT + Rank-GRPO 最佳。

## 4. 小结

论文贡献可概括为：（1）**ConvRec-R1** 两阶段管线，用教师蒸馏得到目录锚定的 SFT 数据，再 RL 精排；（2）**Rank-GRPO**，以 rank 为单位的相对优势与几何平均 rank 概率比，配合 **因果化** 的 DCG@\(k\!:\!N\)（及 \(\exp_\infty\) 变体）缓解 GRPO 在排序生成上的结构性错配。对「生成式检索 / 有序列表输出」类任务，rank 级 RL 设计具有可推广性。
