# OneRanker：工业广告推荐中单模型统一生成与排序（中文摘要）

- **论文**：OneRanker: Unified Generation and Ranking with One Model in Industrial Advertising Recommendation  
- **arXiv**：<https://arxiv.org/abs/2603.02999>（cs.IR）  
- **作者**：Dekai Sun, Yiming Liu, Jiafan Zhou, Xun Liu, Chenchen Yu, Yi Li, Jun Zhang, Huan Yu, Jie Jiang（腾讯）

---

## 1. 背景与动机

端到端生成式范式正在推动广告推荐从传统级联架构走向统一建模（与 OneRec、GPR 等工业路线一致）。但在广告场景的「排序阶段」深度采用生成式方法时，作者归纳出三类核心矛盾：

1. **兴趣目标与商业价值不对齐**：多 token 预测（MTP）常按点击/转化等行为优化，而广告还需对齐 eCPM 等商业价值；若在单阶段强行融合（如直接把 eCPM 注入 MTP 头），兴趣覆盖与价值优化会在共享表示空间中相互掣肘；若采用「先生成再排序」的解耦，生成阶段感知不到排序目标，易在生成阶段系统性过滤高价值候选。
2. **生成过程的「目标无关」（target-agnostic）**：用户表示在生成过程中相对静态，难以随候选物品特性动态调整，限制细粒度用户–物品交互。
3. **生成与排序的三重割裂**：表示空间不一致、用户表示重复编码带来的计算冗余、生成语义漂移难以在排序阶段纠正，阻碍全局最优。

本文提出 **OneRanker**：在架构层面深度耦合生成与排序，并以输入–输出双侧一致性保证端到端协同优化。

---

## 2. 方法总览（三阶段）

整体框架对应图中三步，逻辑递进：

| 阶段 | 名称 | 作用简述 |
|------|------|----------|
| **Step 1：生成** | Generation | 与 GPR 类似：用户行为序列 token 化为异构 token 流（用户/上下文/内容/物品），基于 **HSTU Decoder-only** 做多兴趣路径的并行 MTP；单次前向可并行生成多条完整语义 ID 路径。 |
| **Step 2：多任务 / 目标感知增强** | Multi-Task / Target-Aware | 用 **任务 token** 解耦兴趣与价值目标；用 **Fake Item Token** 做粗粒度目标感知；输出供排序使用的高质量、目标敏感表示。 |
| **Step 3：统一排序** | Ranking | **排序解码器（R-Decoder）** 通过候选与任务 token 的细粒度交叉注意力完成商业价值对齐；与 Step 1/2 通过 **K/V 贯通** 耦合。 |

设计理念可概括为：**价值引导的生成、粗细粒度协同感知、双侧一致性保障**。

---

## 3. Step 2 关键技术

### 3.1 价值感知的多任务解耦（Value-Aware Multi-Task Decoupling）

- 构造可学习的任务 token 序列 \(\mathbf{T} = [\mathbf{t}_{i_1},\ldots,\mathbf{t}_{i_m}, \mathbf{t}_v]\)：前 \(m\) 个为 **兴趣任务 token**，\(\mathbf{t}_v\) 为 **价值感知任务 token**（学习最终商业价值）。
- 共享底层用户表示，但通过 **独立预测头** 解耦输出空间，缓解兴趣覆盖与价值优化的张力。
- 结合 **任务顺序先验**（如曝光 \(\rightarrow\) 点击 \(\rightarrow\) 转化 \(\rightarrow\) 价值）与 **因果 mask**：后续任务可读前置任务表示，实现由兴趣到价值的渐进细化。

### 3.2 Fake Item Token（粗粒度目标感知）

- 在全库物品嵌入上做 **K-means**，得到 \(k\) 个簇中心向量 \(\mathbf{F}=[\mathbf{f}_1,\ldots,\mathbf{f}_k]\)，作为物品语义空间的锚点。
- 将 Fake Item Token 接在任务 token 后组成 Query \(\mathbf{Q}=[\mathbf{T};\mathbf{F}]\)，Step 1 输出作为 Cross-Attention 的 **Key/Value**，使生成过程能感知物品语义分布，缓解 target-agnostic。

### 3.3 异构注意力解码器

相对标准 Decoder 的两点改动：

1. **Cross-Attention 优先**：先做 Cross-Attention（对齐「先充分聚合用户多兴趣」），再做 Self-Attention（任务与 Fake Item 内部整合）。
2. **异构 Mask**：任务 token 间因果；任务 token 与 Fake Item 间双向全连接式访问；**Fake Item 彼此之间不可见**，避免簇中心相互干扰。

### 3.4 双通道表示（检索 / MTP）

- **任务语义通道**：任务 token 的语义向量。
- **目标感知通道**：每个 Fake Item 与当前任务 token 拼接后经 MLP 得到 \(k\) 维偏好分数，再 sum pooling 聚合为 \(\mathbf{s}^{(i)}_{\text{target}}\)。
- 用户侧：\(\mathbf{e}_{\text{user}}^{(i)} = \text{Concat}(\mathbf{e}_{\text{task}}^{(i)}, \mathbf{s}_{\text{target}}^{(i)})\)。
- 物品侧对称增强：物品嵌入与各簇中心余弦相似度拼接到 \(\mathbf{e}_{\text{item}}^{\text{enhanced}}\)。
- 检索得分由内积自然分解为「语义匹配 + 目标感知」两项之和（文中式对应 \(\text{score}(user,item)\)）。

价值相关 head 还采用 **价值加权采样**，使生成分布更偏向高价值物品。

---

## 4. Step 3：统一排序与双侧一致性

### 4.1 R-Decoder（细粒度目标感知）

- **Query**：排序任务 token \(\mathbf{T}_r\) + Step 2 MTP 产生的 \(n\) 个候选物品 token。
- **Key/Value**：融合 **Step 1 原始多兴趣表示** 与 **Step 2 精炼表示**，使排序同时利用「兴趣广度」与「生成阶段的决策依据」。
- **Mask**：候选 token **彼此不可见**（对角式隔离），保证各候选独立打分；候选可读 \(\mathbf{T}_r\)。
- 结构与 Step 2 类似（Cross 优先 + Self），但 **仅 1 层**；每个候选位置经轻量 MLP 输出标量分数 \(s_i\)。

### 4.2 输入侧一致性：K/V 贯通

- Step 3 直接复用 Step 1、Step 2 输出作为 K/V，避免「外部排序」面对黑盒生成结果，减轻表示碎片化。

### 4.3 输出侧一致性：联合损失与 DC

总损失：

\[
\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{MTP}} + \beta \mathcal{L}_{\text{rank}} + \gamma \mathcal{L}_{\text{DC}}
\]

- **\(\mathcal{L}_{\text{MTP}}\)**：多兴趣路径 NLL（softmax 在全语义空间）；兴趣头对齐点击/转化等，价值头配合价值倾向。
- **\(\mathcal{L}_{\text{rank}}\)**：候选对上的 **BPR**，标签为真实商业价值（如 eCPM）。
- **\(\mathcal{L}_{\text{DC}}\)（Distribution Consistency）**：将 Step 3 分数经 softmax（温度 \(\tau\)）得到 \(p^{\text{target}}\)，对生成策略 \(\pi_\theta\) 做交叉熵式约束，使生成阶段分布与排序器的「校准后打分空间」对齐；文中说明其源自避免 RL 采样不稳定的监督替身，形成 **「真实倾向监督 + 模型分布一致性」** 的双重监督，并为排序到生成提供梯度回路。

---

## 5. 实验要点

### 5.1 离线（与 GPR 设定一致的大规模真实数据）

- **基线**：HSTU、GPR（MTP 版本）。
- **指标**：HR@K、NDCG@K。
- **实现摘要**：各类 token 嵌入维度 128；Step 1 序列最长 2048，G-Decoder 4 层；Step 2 兴趣任务 token 数 6，价值感知 2 个 token（bagging 聚合），**32** 个 Fake Item Token，异构解码器 **2** 层；Step 3 R-Decoder **1** 层。

**主表结果（节选）**：OneRanker HR@1 达 **0.2639**，相对 GPR（0.1824）约 **+44.7%**；NDCG@5 等排序指标亦大幅提升（如 NDCG@5：0.6818 \(\rightarrow\) 0.7904）。

### 5.2 消融

- 去掉 DC、去掉 Step 2 注入 K/V、仅用 Step 1 做 S3 等均劣于完整模型；去掉 Fake Item（Target）或再去掉 MDA 显著损伤 Step 2 性能。
- Step 2 内去掉 **Cross-Attention 优先（CA-Pri）** 或 **异构 Mask（H-Mask）** 均明显下降。

### 5.3 \(\mathcal{L}_{\text{DC}}\) 分析

- 秩偏差箱线图与 Top-K 重叠曲线显示：加入 DC 后 Step 2 与 Step 3 排序更一致、Top-K 重叠更高。

### 5.4 微信视频号在线 A/B

- 相对「生成 + 独立排序」基线，多流量阶段下 **GMV-Normal**、**Costs** 等有提升；文中摘要强调全量上线及 **GMV-Normal +1.34%** 等量级的业务收益（具体置信区间见原文 Table 在线实验）。

---

## 6. 结论（归纳）

OneRanker 针对生成式广告推荐中的 **兴趣–价值冲突**、**目标无关生成**、**生成–排序割裂** 三类问题，通过 **价值感知多任务解耦**、**Fake Item + R-Decoder 的粗细目标感知**、以及 **K/V 贯通 + DC 损失的输入输出双侧一致性**，实现单模型内生成与排序的深度集成与端到端协同；并在腾讯微信视频号广告场景完成规模化落地验证。

---

## 7. 讨论与辨析（问答整理）

本节把阅读过程中围绕动机、结构与实现的问答整理进摘要，便于和正文对照；表述尽量 **客观、少修辞**。

### 7.1 「兴趣–价值错配」为什么会出现？

**问题**：第一点动机里，兴趣–价值错配，原因是什么？

**答**：根因是 **优化 proxy 与商业 KPI 不恒等**：MTP/行为监督常对齐点击、转化等 **engagement**，而广告排序要对齐 **eCPM、GMV** 等；二者相关但 **非单调等价**（高 CTR 未必高 eCPM）。在 **单阶段、共享表示** 里同时拉「兴趣覆盖」与「价值尖锐化」，易出现 **optimization tension**；自回归 **长链反传** 还会放大多任务梯度冲突。若业务目标 **严格** 就是单一 CTR 或单一 CVR，且训练标签与线上一致，则论文里那种 **「行为 vs 竞价价值」** 的错配会 **明显减弱**；但若仍有 **CTR vs CVR**、**代理指标 vs 真实 KPI**、**位置/选择偏差** 等，错配会以 **其他形式** 存在。论文第二、三点（target-agnostic、阶段割裂）与是否广告 **相对独立**。

### 7.2 「生成–排序割裂」是以哪篇工作为基础讨论的？

**问题**：生成–排序割裂是以哪个工作为基础展开讨论的？

**答**：原文 **没有** 把该现象归因于单篇奠基论文，而是先描述工业/学术里常见的 **两阶段范式**：行为序列作前缀 **自回归生成候选 token**，再交给 **独立排序模型** 打分，并指出 **异构表示、语义漂移、难全局最优** 等问题。文献对接主要在 Related Work **「Joint Optimization of Generation and Ranking」**：用排序信号融入生成的工作，如 **LTRGR**（AAAI 2024）、**FLAME**、**RankGR**、**SynerGen** 等；作者认为其中不少仍偏 **软协同 / 后处理**，未从根上消掉表示空间异质性。OneRanker 的 **K/V 贯通 + 联合损失** 是在这条线上往 **硬共享、硬对齐** 方向走。

### 7.3 三个 motivation，本文分别怎么接？

（与第 1–4 节一致，此处用「对接表」压缩。）

| Motivation | 机制要点 |
|------------|----------|
| 兴趣 vs 价值 | 任务 token 序列 + 独立头 + 顺序先验与因果 mask；价值 token \(\mathbf{t}_v\) 与价值加权采样等 |
| Target-agnostic | Step 2：Fake Item（粗）+ 双通道内积；Step 3：真实候选 token 上 R-Decoder（细） |
| 生成–排序割裂 | 输入：Step1+Step2 → Step3 的 **K/V 复用**；输出：**\(\mathcal{L}_{\text{MTP}}+\mathcal{L}_{\text{rank}}+\mathcal{L}_{\text{DC}}\)**，DC 把排序 softmax 分布压回生成 |

### 7.4 Task token 序列：每个 token 就是一个 CTR/CVR 任务吗？6 个头与最终结果如何构成？

**问题**：task token sequence 里每个 token 都是一个 task 吗（例如 CTR、CVR）？实验里 6 个多兴趣预测头，是否每个都走单独预测头？最终结果怎么来？

**答**：

- **不是**「第 1 个 token = CTR、第 2 个 = CVR」这种固定命名；而是 **每个任务 token 位置绑定一条独立预测支路**，语义上可以是 **多兴趣维度**，也可以是 **漏斗链**（文中举例曝光 \(\rightarrow\) 点击 \(\rightarrow\) 转化 \(\rightarrow\) 价值）。CTR/CVR **可**对应链上环节，由 **监督与顺序先验** 定义，而非文内硬编码词表。
- 实验配置：**6 个 interest task token** 与 **6 个 MTP 头**对齐；**价值侧 2 个 token**，输出 **bagging 聚合**。
- **「最终结果」**：列表上的 **最终序** 由 **Step 3 R-Decoder** 对每个候选输出标量分 \(s_i\) 得到；候选 **\(n\) 个 item token** 来自 **Step 2 多路径 MTP 生成**。多路径如何截断/合并成 \(n\) 个，正文未写到伪代码级；训练上 **\(\mathcal{L}_{\text{MTP}}\)** 对 **多头 \(j\)** 与 **语义层级** 求和。

### 7.5 Fake Item Token 与真实 item 各在哪个阶段进入？

**问题**：3.2.2 用 K-means 得到 fake item；**真实待打分的 target / item** 在哪个阶段进入？

**答**：

- **Fake**：全库聚类中心，只在 Step 2 作 **全局语义锚**，**不是**某次请求里的真实候选。
- **真实 item**：① Step 2 的 **\(\mathcal{L}_{\text{MTP}}\)** 里，ground-truth 语义 ID 嵌入参与 **全库 softmax**；② **双通道 score** 里与真实 \(\mathbf{e}_{\text{item}}\)（及增强版）做内积；③ Step 3 的 Query 里 **\(n\) 个候选 item token** 为 **真实候选**（语义 ID 化），由 Step 2 生成路径得到，经 R-Decoder 打分。数据集描述里 **每个 target 配一批 sampled candidates** 支撑商业价值学习与排序。

### 7.6 Dual-channel 里「current task token」是不是候选 item token？

**问题**：target-aware channel 写「current task token 与 fake token 拼接」——这里的 current task token 仍是前面的 task token，不是候选 item token？

**答**：**是** Step 2 任务序列里的 **\(\mathbf{T}_i\)**，**不是** Step 3 的候选 item token。候选 item token 只在 **Step 3** 作为 \(\mathbf{I}\) 出现。

### 7.7 Dual-channel 表示在做什么？是否主要加强 user？

**问题**：dual channel 是否 task+fake 搭配产生更复杂的 user 表达？3.2 是否主要在解决召回？item 侧是否对称？

**答**：

- **Dual-channel**：对每个任务头 \(i\)，\(\mathbf{e}_{\text{user}}^{(i)}=\text{Concat}(\mathbf{e}_{\text{task}}^{(i)},\mathbf{s}_{\text{target}}^{(i)})\)，即在 **任务语义** 上拼接 **相对 \(k\) 个簇锚的偏好聚合**，使 user 侧显式带 **物品语义空间结构**；检索内积可拆成语义项与目标感知项。
- **Step 2**：既有 **全库内积 + softmax 的 MTP**（形态上像 **生成式检索/召回**），又向 Step 3 **提供 K/V**，故 **不宜** 说成「只做召回一节」；**叙事与参数重心在 user 侧** 属实，item 侧为 **与各簇中心余弦 + 拼接**，比 user 侧 **更轻**。

### 7.8 Unified Ranking（3.3）：结构主义归纳

**讨论**：可用少修饰的说法概括排序子模块——**Q 侧以候选为主，K/V 为前两步的 user/生成侧表示**；并区分 **3.3.2 输入侧一致性** 与 **3.3.3 输出侧一致性**（编号以 PDF 中 `\subsubsection` 为准，对应原文 **Input-Side Consistency** / **Output-Side Consistency**）。

**答**：

- **R-Decoder**：**Query = \([\mathbf{T}_r;\mathbf{i}_1,\ldots,\mathbf{i}_n]\)**——除 **\(n\) 个候选 item token** 外，还有 **排序专用 query token \(\mathbf{T}_r\)**，不是「纯候选矩阵当 Q」。**Key/Value** 为 **Step1 与 Step2 输出融合**（多段「用户/生成上下文」，不宜简化成单一静态 \(\mathbf{u}\)）。**顺序**：与 Step 2 一致，**先 Cross-Attention（对 K/V）再 Self-Attention（在 Q 子空间）**；候选间 **互不可见 mask**，保证各候选分数独立。每层后 **每候选 MLP → 标量分**。
- **3.3.2 输入侧**：规定 **K/V 必须从 Step1、Step2 直通接入**，等价 **复用上游状态、少重复编码、固定信息流**。
- **3.3.3 输出侧**：**BPR** 用业务价值标签调成对序；**DC** 将排序分数 **softmax 成候选集分布**，再约束 **生成分布 \(\pi_\theta\)**，形成 **排序 → 生成** 的梯度回路。

### 7.9 读者综合理解与修正（统一召回–排序？内积 vs 生成？）

**读者归纳**：本文对召回–排序统一建模；召回侧 user 构造复杂并传到排序；召回仍用内积、**没有真实生成 target item id**；排序为 self-attention + query attention，**query 由召回得到的 target item token 构成**。

**修正与对齐原文**：

- **统一建模**：大体成立——**多路径生成候选 + 同一套中间表示做 list 打分 + 联合损失**；更精确是 **「生成式 MTP 路径 + 内积全库训练信号 + list 上 R-Decoder」** 串在同一前向里。
- **「没有真实生成 target item id」**：易误解。Step1/2 仍在 **语义 ID 空间** 上做 **MTP/路径生成**；训练中有 **对 ground-truth 语义的 NLL**；Step3 的候选来自 **生成路径上的 item token**。应区分：**大量训练信号是「user–item 内积 + 全库 softmax」**，与 **「完全不生成 ID」** 不是一回事。
- **Query 组成**：应为 **\(\mathbf{T}_r\) + 多候选 item token**；**不是**仅「召回来的单一 target」；**\(\mathbf{T}_r\)** 非召回侧产出。
- **注意力顺序**：若口语说「self + query attention」，需与原文 **先 cross 再 self** 对齐。

**较贴论文的一句压缩**：**前面用 task/fake/双通道把 user（及 item 轻量增强）做实，并用内积支撑全库 MTP；后面用 \(\mathbf{T}_r\) 与多路径给出的候选 item token 作 Q，对 Step1/2 的 K/V cross→self，每候选一头 MLP 出分；BPR+DC 把排序与生成分布绑在一起。**

---

*摘要依据 arXiv TeX 源 `sample-sigplan.tex` 整理；第 7 节融入阅读讨论与澄清。图表与公式编号以原稿为准。*
