# RecGPT Technical Report (V1)

## 基本信息

- **论文标题**：RecGPT Technical Report
- **arXiv**：[2507.22879](https://arxiv.org/abs/2507.22879)（2025年7月）
- **作者**：RecGPT Team（淘宝）
- **领域**：LLM 驱动的生成式推荐、意图挖掘、Human-LLM 协同评估、工业部署
- **基础模型**：Qwen3-14B（训练底座与解释生成模型）、TBStars-MoE-42B-A3.5B（淘宝内部稀疏 MoE 模型，线上推理激活 3.5B 参数）、DeepSeek-R1（教师模型）、QwQ-32B（增量学习自动判官）
- **部署场景**：淘宝首页"猜你喜欢"（Guess What You Like）

---

## 一句话概括

RecGPT 用三个专职 LLM（$\mathcal{LLM}_{UI}$ 兴趣挖掘、$\mathcal{LLM}_{IT}$ 商品标签预测、$\mathcal{LLM}_{RE}$ 推荐解释生成）串成"用户兴趣挖掘 → 商品标签预测 → 商品召回 → 解释生成"的闭环，把传统"从点击学点击"的协同过滤范式改造为显式的意图中心式推荐；三个 LLM 都通过统一的多阶段训练范式（课程学习多任务微调 → 推理增强预对齐 → 自训练进化）完成领域对齐，并配合 Human-LLM 协同评判系统（LLM-as-a-Judge + 人类里程碑校准）替代人工标注实现规模化质量控制。这是已知首个在服务十亿级用户和商品的工业系统中完整部署"推理增强型百亿规模推荐基础模型"的公开报告。

---

## 1. 问题背景与核心思路

传统工业推荐系统高度依赖历史共现模式与 log-fitting 目标（优化历史交互而非显式建模用户意图），容易造成对狭窄历史偏好的过拟合，强化 Filter Bubble 与长尾商品的马太效应。RecGPT 的核心思路是：利用 LLM 的世界知识、语义理解与链式推理能力，显式挖掘用户潜在兴趣，并将其转化为可检索的商品标签，从而把"协同过滤"升级为"兴趣增强"的候选生成过程，同时不改变下游排序/重排基础设施。

整体工作流（图 `Figures/overview.pdf`）由四个模块组成：
1. **User Interest Mining**（$\mathcal{LLM}_{UI}$）：对用户终身多行为序列做显式兴趣挖掘；
2. **Item Tag Prediction**（$\mathcal{LLM}_{IT}$）：基于兴趣推理生成细粒度商品标签，代表用户潜在偏好分布；
3. **Item Retrieval**（Tag-Aware Semantic Relevance Retrieval, TAR）：将标签映射到具体商品，融合语义相关性与协同过滤信号；
4. **Personalized Explanation Generation**（$\mathcal{LLM}_{RE}$）：基于用户兴趣与召回结果生成用户友好的推荐解释。

相较传统依赖隐特征和最终用户反馈端到端优化的推荐算法，RecGPT 采用逐阶段的显式文本建模，优点：(1) 可对中间过程和各阶段模型表现做可解释监控；(2) 可通过过程级监督引入专家知识，针对单一组件做定向优化。

---

## 2. User Interest Mining（§3）

### 2.1 两大挑战
- **上下文窗口限制**：淘宝用户平均拥有 3.7 万+条历史行为记录，远超当前 LLM 128K token 的上下文窗口。
- **领域知识缺口**：通用 LLM 缺乏对淘宝等平台领域特征的专门理解，难以专家级抽取/抽象用户兴趣。

### 2.2 Reliable Behavioral Sequence Compression
- **Reliable Behavior Extraction**：只保留能真实反映用户兴趣的行为——(1) **Intentional Feedback Behaviors**（收藏、购买、加购、详情页浏览、评论阅读等高参与度行为）；(2) **Search Behaviors**（搜索查询）。普通点击行为因噪声较大被排除。
- **Hierarchical Behavior Compression**（Item 级 + Sequence 级两层压缩）：
  - *Item-level*：先用 LLM 把单个商品的详细信息压缩为核心属性（名称、类目、品牌等），降低单条信息密度损耗；
  - *Sequence-level*：按时间粒度分区（月内按天、月间按月、年外按年），执行两步聚合——**Step 1 Temporal-Behavioral Aggregation**（以"时间-行为类型"为 key 聚合同期同行为的商品）、**Step 2 Item-based Reverse Aggregation**（再以商品为 key 反向聚合其对应的时间-行为组合），最终形成 `时间1(行为1,行为2,...), 时间2(...) | 商品1,商品2,...` 的紧凑格式。
- **效果**：该压缩方案使 **98%** 的用户行为可容纳进 128K token 窗口（未压缩仅 88%），并将兴趣推理效率提升 **29%**。

### 2.3 Task Alignment for $\mathcal{LLM}_{UI}$（三阶段）
1. **Curriculum Learning-based Multi-task Fine-tuning (CL-MFT)**：设计 16 个前置子任务（共 16.3k 样本），按难度/依赖关系拓扑排序，分 Foundation（如 Query 类目预测、Query-Item 相关性判断、商品标题关键信息抽取）→ Intermediate（电商"买什么"任务、Query 纠错/改写等，占样本量最大的 13.3k）→ Advanced（因果推理、归纳推理、情感分析、文本分类等）三级课程，循序建立领域基础能力。
2. **Reasoning-Enhanced Pre-alignment**：用 DeepSeek-R1 生成 9.0 万条初始样本，经人工精炼蒸馏为 1.9 万条高质量数据，作为知识蒸馏基础对 $\mathcal{LLM}_{UI}$ 做预对齐微调，使其性能逼近教师模型。
3. **Self-Training Evolution**：模型自生成训练数据并迭代优化，收集 2.11 万条高质量自训练样本，通过 Human-LLM 协同的 LLM-as-a-Judge 低成本过滤自生成输出（详见 §5）。

### 2.4 Prompt 设计与数据质量控制
Prompt 输入包含用户属性、压缩后可靠行为序列、预设兴趣候选池，并用 CoT 引导模型逐步推理而非直接输出兴趣。数据质量控制采用两维度拒绝采样：
- **Willingness**（意愿性）：\correct Spontaneity（自发兴趣）vs \wrong Necessity（生活必需，非真实爱好）；
- **Reasonableness**（合理性）：\correct Strong Correlation vs \wrong Weak Correlation / No Correlation / Hallucination（无行为依据的臆造兴趣）。
只有同时满足 Willingness 和 Reasonableness 才判定为正确样本。

### 2.5 人工评估结果
| 模型 | DeepSeek-R1 | Qwen3-Base | Qwen3-SFT | TBStars-SFT |
|---|---:|---:|---:|---:|
| 兴趣挖掘通过率 | 70.00% | 59.74% | **77.28%** | 74.39% |

推理增强的 DeepSeek-R1 明显优于未对齐的 Qwen3-Base，验证长序列理解对兴趣挖掘的重要性；多阶段对齐后的 Qwen3-SFT 达到最优；线上实际部署选用轻量稀疏架构 TBStars-SFT（激活参数仅 3.5B），兼顾精度与推理效率。

### 2.6 在线部署
离线用 $\mathcal{LLM}_{UI}$ 预测用户兴趣偏好，平均每用户产出 **16.1** 条兴趣；每两周迭代刷新一次用户兴趣，兼顾时效性与个性化动态变化。

---

## 3. Item Tag Prediction（§3.2）

### 3.1 Task Alignment（两阶段：Reasoning-Enhanced Pre-Alignment + Self-Training Evolution）
- **标签格式**："Modifier + Core-Word"（如"户外防水防滑 登山靴"），用 CoT 充分利用推理能力。
- **Prompt 约束**：Interest Consistency（与已挖掘兴趣一致）、Diversity Enhancement（至少 50 个标签，缓解信息茧房）、Semantic Precision（避免宽泛类目）、Temporal Freshness（避免重复推荐近期已交互商品）、Seasonal Relevance（结合时间戳做季节适配）。
- 输出为 **(Tag, Associated Interest Preference, Rationale)** 三元组列表。

### 3.2 数据质量控制（4 项，全部满足才算合格）
Relevance（标签与兴趣的直接关联）、Consistency（推理是否基于真实用户画像/历史行为而非臆造）、Specificity（避免"时尚运动装备"这类过泛标签）、Validity（标签对应真实存在的商品）。

### 3.3 人工评估结果
| 模型 | DeepSeek-R1 | Qwen3-Base | Qwen3-SFT | TBStars-SFT |
|---|---:|---:|---:|---:|
| 标签预测通过率 | 80.00% | 33.70% | 84.80% | **88.80%** |

Qwen3-Base 直接应用效果很差（33.70%），凸显领域对齐的必要性；对齐后模型（Qwen3-SFT、TBStars-SFT）均超过教师模型 DeepSeek-R1，TBStars-SFT 因低延迟优势最适合工业部署。

### 3.4 Incremental Learning（双周增量更新）
面对季节性变化等在线分布漂移，设计三步流程处理最近 14 天的在线交互数据：
1. **Data Purification**：用 QwQ-32B 作自动判官，从 Relevance（行为与兴趣一致性）和 Timeliness（商品是否符合当季/临近季节）两方面过滤低质量噪声行为（如误触点击）；
2. **Interest Completion**：用 QwQ-32B 基于用户画像、历史行为推理出 (Tag, Interest, Rationale) 三元组，真实交互场景下直接用商品标题作为标签；
3. **Data Balancing**（两阶段重采样）：先为每用户随机采样对应 80 个商品标签的行为记录保证多样性与训练效率；再用预训练 Tag-to-Cate 模型 $\phi(\cdot)$ 把标签映射到类目做二次采样（每类目最多采 2 条），缓解类目分布不均衡带来的多样性损失与马太效应。

**效果评估**：设计 HR@30 指标（预测 30 个标签→30 个类目，命中用户真实下一次交互类目的比例），TBStars-SFT 引入增量学习后 HR@30 从 0.3671 提升到 **0.3776（+1.05%）**，验证了增量学习对适应动态偏好和新品趋势的有效性。

---

## 4. Item Retrieval：User-Item-Tag Retrieval (TAR)（§3.3）

### 4.1 整体架构：三塔并行
- **Item Tower**：稀疏特征（商品ID、类目、品牌等）+ 稠密特征（价格、销量等离散化）经 embedding 层后接 DNN 得到商品表示 $\mathbf{h}_v$；
- **User Tower**：用户ID + 多行为序列（点击/购买等）各自 mean pooling 后拼接过 DNN 得到 $\mathbf{h}_u$；
- **Tag Tower**：LLM 生成的标签分词后 mean pooling 过 DNN 得到 $\mathbf{h}_t$。

生成两类打分：协同分数 $\hat{y}_{col}=\mathbf{h}_u^T\mathbf{h}_v$，语义分数 $\hat{y}_{sem}=\mathbf{h}_t^T\mathbf{h}_v$。

### 4.2 优化目标
- **Collaborative Optimization**：user-item 对比学习损失 $\mathcal{L}_{col}$（点击为正例，负采样未点击商品为负例）；
- **Semantic Optimization**：tag-item 对比学习损失 $\mathcal{L}_{tag}$（点击商品的标签为正例）+ 类目对比损失 $\mathcal{L}_{cate}$（同类目商品为正、跨类目为负，防止模型过拟合描述性标签词面而丢失细粒度类目区分）；
- **总损失**：$\mathcal{L}_{TAR}=\mathcal{L}_{col}+\alpha\mathcal{L}_{tag}+(1-\alpha)\mathcal{L}_{cate}$，实验中 $\alpha=0.5$。

### 4.3 在线推理：动态融合
推理时对 user tower 与 tag tower 输出做加权融合 $\mathbf{h}_{fuse}=\beta\mathbf{h}_u+(1-\beta)\mathbf{h}_t$，等价于对协同分数与语义分数加权求和 $\hat{y}_{final}=\beta\hat{y}_{col}+(1-\beta)\hat{y}_{sem}$，$\beta$ 可灵活调节协同信号与语义理解的平衡，兼顾探索（用户潜在多样偏好）与利用（既有行为模式）。

---

## 5. Personalized Explanation Generation（§4）

为满足线上低延迟要求，避免为每个 user-item 对实时生成解释：

### 5.1 Task Alignment for $\mathcal{LLM}_{RE}$
沿用两阶段范式（DeepSeek-R1 推理增强预对齐 → 自训练进化）。Prompt 要求模型两步推理：**Context Understanding**（理解用户兴趣与商品特征）→ **Explanation Generation**（若存在合理关联则生成融合两者的口语化短语，否则基于商品自身卖点生成）。

**数据质量控制（4 项全满足才合格）**：Relevance（解释与兴趣/商品的对齐）、Factuality（不夸大不虚构功能材质等）、Clarity（语言流畅、语法正确）、Safety（不含敏感/隐私信息）。

**人工评估结果**：
| 模型 | DeepSeek-R1 | Qwen3-Base | Qwen3-SFT |
|---|---:|---:|---:|
| 解释生成通过率 | 92.7% | 30.0% | **95.8%** |

### 5.2 Offline Production：Interest-Item-Explanation 查找表
用预训练 Tag-to-Cate 模型 $\phi(\cdot)$ 把标签 $T$ 映射到类目 $C$，通过共享类目建立"兴趣-商品"配对关系（而非穷举"用户-商品"组合，大幅缩小生成目标规模），离线批量生成所有匹配"兴趣-商品"对的解释，构建 **Interest-Item-Explanation 查找表**；线上按当前推荐商品与用户兴趣集合直接查表返回解释，实现实时低成本的解释交付。

---

## 6. Human-LLM Cooperative Judge System（§5）

### 6.1 动机与挑战
人工众包评估在工业场景下成本高、周期长，难以规模化；直接用 LLM-as-a-Judge 面临两大挑战：
- **Cognitive Bias**：推荐任务需要理解复杂用户行为、商品特征和运营策略，原生 LLM 常因知识局限和预训练偏见产生认知偏差；
- **Temporal Misalignment**：用户行为模式演化、商品特征更新、评估标准（业务策略）变化，三者叠加使静态 Judge 逐渐失准。

### 6.2 LLM-as-a-Judge（§5.1）
- **数据构造**：评估任务分为 **Binary Classification**（如标签"Relevance"的 Yes/No 判断）与 **Multi-level Evaluation**（如解释"Truthfulness"的 Excellent/Good/Bad）；训练数据来自 Pre-alignment（DeepSeek-R1 推理增强数据）与 Self-training（模型自生成多轮迭代样本），经人工标注汇总进 **Judge Data Buffer**。
- **Data Rebalancing Strategy**：针对严重类别不均衡（ERM 原则下模型偏向多数类），设计 (1) **Minority Class Augmentation**：累计历史多轮标注样本补充少数类；(2) **Recency-Prioritized Downsampling**：对多数类做时间衰减降采样，优先保留近期样本、逐步纳入早期样本。

### 6.3 Human-in-the-Loop（§5.2）
提出 **Milestone-Based Human Supervision**：在重大版本更新时，(1) 收集专家对近期生成样本的标注；(2) 系统性对比 LLM Judge 与人类评估结果；(3) 若检测到显著性能退化，用新标注数据对 LLM-Judge 做针对性微调持续训练。该机制在保持运营效率的同时维持与演化数据分布的对齐。

### 6.4 Judge 与人类一致性评估结果
基础 Judge 模型为 Qwen3（Qwen3-Judge-Base），经人类判断数据 SFT 后为 Qwen3-Judge-SFT：

| 任务 | 模型 | ACC | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| 兴趣挖掘 | Base | 0.6777 | 0.6742 | **0.9777** | 0.7968 |
| 兴趣挖掘 | SFT | **0.7689** | **0.7996** | 0.8575 | **0.8275** |
| 标签预测 | Base | 0.8741 | 0.9310 | 0.9196 | 0.9253 |
| 标签预测 | SFT | **0.9308** | **0.9714** | **0.9463** | **0.9587** |
| 解释生成 | Base | 0.5677 | 0.8753 | 0.5677 | 0.6657 |
| 解释生成 | SFT | **0.8976** | **0.9067** | **0.8976** | **0.9016** |

三个任务上 SFT 后的 Judge 均显著优于 Base（准确率提升 5.7~32.9pp），尤以解释生成任务提升最大（56.77%→89.76%），验证了少量人类判断数据 SFT 即可让 LLM-as-a-Judge 达到可替代人工评估的可靠水平，大幅加速迭代周期。

---

## 7. 在线实验（§6）

### 7.1 实验设置
- **用户分组**：在淘宝首页"猜你喜欢"场景做为期一个月的 A/B 测试，面向活跃用户 Top 1/3，实验组/对照组各占 1% 流量；
- **基础设施**：FP8 量化 + KV Cache、Megatron 分布式训练框架处理超长行为序列，推理速度提升 **57%**；
- **实现细节**：初期用 Qwen3-14B 训练 & 积累数据；线上兴趣挖掘/标签预测采用轻量 TBStars-MoE-42B-A3.5B（激活 3.5B 参数）；解释生成保留 Qwen3-14B 以保证质量。

### 7.2 A/B 测试结果（2025年6月17-20日）

| 场景 | DT | EICD | CICD | IPV | CTR | DCAU | ATC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Guess | +4.82% | +0.11% | +6.96% | +9.47% | +6.33% | +3.72% | +3.91% |

- **用户视角**：Dwell Time +4.82%、CICD +6.96%（曝光类目多样性 EICD +0.11%），表明系统成功挖掘用户潜在多样兴趣，缓解 Filter Bubble；
- **平台视角**：IPV +9.47%（用户探索商品深度增加）、CTR +6.33%（推荐精准度提升）、DCAU +3.72%（用户活跃留存提升）；
- **商户视角**：不同人气水平商品组的 CTR 分布更均衡，Page View Rate 长尾分布被"拉平"，缓解马太效应，让长尾商户获得有效曝光，形成用户-商户-平台三方共赢的可持续生态。

### 7.3 Case Study
以一位杭州 30 岁女性用户三年行为历史为例（涵盖旗袍、水墨风女装、婴儿防晒帽、儿童学习桌等分散行为），$\mathcal{LLM}_{UI}$ 识别出"服饰穿搭风格"与"育儿母婴"两大主题兴趣；$\mathcal{LLM}_{IT}$ 转化为"亚麻拼接阔腿套装""婴儿洗澡水温感应器"等具体标签；TAR 召回匹配商品；$\mathcal{LLM}_{RE}$ 生成结合地域/季节/育儿安全语境的解释（如"杭州夏日气息新品""温度掌控 妈妈安心"）。展示了从行为洞察到语义解释的完整闭环，能挖掘表面无关行为背后的隐含关联（如传统服饰兴趣关联文化认同、育儿关注关联安全意识）。

### 7.4 用户体验调研（500 名活跃用户问卷）
- 采用三评审员一致性机制（仅全票一致才算有效响应），流程为历史回顾 → 推荐分析 → 冗余度评估；
- **结果**：整体重复感知率从 37.1% 降至 36.2%；Top 4 展位相似商品聚类比例从 27.7% 降至 25.3%（用户最关注的位置改善最明显）；剔除广告卡片后重复度改善幅度从 0.88% 提升到 1.57%（近乎翻倍），说明核心推荐能力（排除广告干扰）在缓解内容同质化上效果更显著，尤其在 Top 8 展位。

---

## 8. 结论、局限与未来方向（§7）

**核心贡献**：(1) 首次在服务十亿级用户/商品的工业系统中完整部署推理增强的百亿规模推荐基础模型；(2) 提出系统性多阶段训练框架（课程学习 MFT → 推理增强预对齐 → 自训练进化），配合 Human-LLM 协同判官系统实现从人工专家评审到自动化协同评审的渐进过渡；(3) 全链路（兴趣挖掘→标签预测→检索→解释）在线 A/B 验证了用户、商户、平台三方共赢。

**局限（论文自述）**：
1. **超长用户序列建模**：约 2% 序列仍超出 128K token 限制，计算开销大且长序列中噪声干扰兴趣判断；未来计划探索面向 LLM 的 **Context Engineering**（动态长短期记忆管理、上下文选择与信息压缩）。
2. **多目标联合学习与强化学习**：当前依赖监督学习+定期更新，静态训练难以适应动态偏好演化，且兴趣挖掘/标签预测/解释生成三任务分开训练、未联合优化；计划用 **ROLL**（大规模 RL 优化库）构建 RL 驱动的多目标联合优化（同时优化参与度、转化率、长期平台健康度）。
3. **端到端 LLM-as-a-Judge**：现有评估框架按任务/维度分别训练，缺乏整体上下文理解；计划引入 **RLHF** 训练端到端多任务评判系统，并探索推理时可扩展的生成式奖励模型（inference-scaling generative reward models）动态分配评估计算资源。

**与后续版本的关系**：RecGPT-V2（arXiv:2512.14503）针对 V1 的"多路并行计算冗余、解释模板固化、纯监督学习泛化不足、Judge 结果导向"四大局限，引入 Hybrid Representation Inference（原子实体压缩）、Hierarchical Multi-Agent System（Planner-Experts-Arbiter）、Meta-Prompting 动态解释生成、Agent-as-a-Judge + Judge-as-a-Reward 自我迭代飞轮等改进；RecGPT-V3（arXiv:2607.15591）在 V2 基础上进一步引入 Memory Hub、Hybrid-modal Foundation Model 与 Latent Intent Reasoning。
