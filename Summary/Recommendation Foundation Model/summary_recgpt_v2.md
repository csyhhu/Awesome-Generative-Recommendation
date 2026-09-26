# RecGPT-V2 Technical Report

## 基本信息

- **论文标题**：RecGPT-V2 Technical Report
- **arXiv**：[2512.14503](https://arxiv.org/abs/2512.14503)
- **作者**：RecGPT Team（淘宝）
- **领域**：生成式推荐、多智能体系统、强化学习、LLM-as-a-Judge、工业部署
- **基础模型**：Qwen-14B（Expert 模型）、Qwen3-32B-Instruct（Judge 模型）
- **部署场景**：淘宝首页“猜你喜欢”，Item 与 Feed 两个场景

---

## 一句话概括

RecGPT-V2 把 RecGPT-V1 中并行、重复编码全量行为序列的多路 LLM cognitive channel，改造成 **Planner → Experts → Arbiter** 三层协同的 Hierarchical Multi-Agent System（HMAS），配合 **Hybrid Representation Inference**（把商品/查询文本压缩成单一 atomic token）大幅降低计算量；同时引入 **Meta-Prompting** 做动态解释生成、**Constrained Reward Shaping (CRS)** 解决多奖励强化学习冲突、**Agent-as-a-Judge + Judge-as-a-Reward** 构建可自我迭代的评估-奖励飞轮。

---

## 1. 问题背景：RecGPT-V1 的四个局限

1. **多路架构计算冗余**：V1 为覆盖天气、热点、季节等不同上下文信息，扩展出多条独立 LLM 推理路线，每条路线都要重新编码约 **32K tokens** 的完整用户行为序列（表示层浪费），且各路线相互独立推理导致候选重复率高达 **13.46%**（认知层浪费）。
2. **解释生成模板固化**：固定 prompt 模板生成的推荐解释同质化严重，无法根据实时上下文（天气、季节、热点）自适应调整。
3. **监督学习泛化不足**：仅靠静态语料 SFT 学习多目标（相关性、多样性、新颖性等）生成任务，难以适应用户需求的动态演化。
4. **LLM-as-a-Judge 评估过于结果导向**：一次性直接预测质量分数，忽略人类评估中多维度、多步骤的推理过程，导致与人类标准对齐度有限。

---

## 2. Hybrid Representation Inference：把 32K 压缩到 11K

### 2.1 Atomized Entity Compression（两阶段）

**Stage 1：原子表示编码。** 用预训练 embedding 模型（BGE、Qwen3-Embedding、TBstars-Embedding）把商品标题、查询文本等实体 \(e\) 编码为 \(\mathbf{h}=f_{\text{embed}}(\mathbf{x})\)，再用一个两层 MLP adaptor 投影进 LLM 输入空间：

\[
\mathbf{z} = f_{\text{adapt}}(\mathbf{h}) = \mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \mathbf{h} + \mathbf{b}_1) + \mathbf{b}_2.
\]

这个原子单元 \(\mathbf{z}\)（记作 `[entity]`）替换掉原本多 token 的文本描述。论文给出的例子：一个 12-token 的中文商品标题被压缩为 1 个原子 token，压缩比 12:1；一段完整用户行为序列从 21,349 tokens 压缩到 5,158 tokens（压缩 76%），压缩过程中用户属性、时间戳等自然语言信息保持不变，只替换商品/查询文本部分。

相较于直接向词表插入新 token 的方法（OneRec-Think、LC-Rec、CoLLM），adaptor 方案的优势：只训练 adaptor、冻结 LLM backbone，因而**参数高效**、**保留原始语言理解能力（泛化性更好）**、**模块化程度高**（可自由更换 embedding 模型或 LLM）。

### 2.2 Hybrid Representation Adaptation（对齐训练）

训练时始终冻结 LLM，只训练 adaptor 参数，包含两类任务：

1. **Self-Perception Tasks（"what-is-it"）**：用 GPT-4 基于 in-context-learning 自动生成围绕商品属性的多样化 QA 对（例如材质、适用季节、防滑性能、适用场景），作为监督信号训练 adaptor 保留实体语义完整性。
2. **Production-Oriented Alignment**：把压缩后的原子单元代入 V1 的两个核心任务——**User Interest Mining** 和 **Item Tag Prediction**，用全文本 prompt 下冻结 LLM 产生的真实回答作为参考标签。

两类任务共享统一优化目标：构造 hybrid prompt \(\mathcal{P}_{\text{hybrid}}=\phi(\mathcal{P}_{\text{full}})\)（把全文本 prompt 中所有实体替换为 adaptor 投影后的原子表示），最小化 hybrid prompt 下模型输出与全文本 prompt 参考答案之间的交叉熵：

\[
\mathcal{L}(\theta_{\text{adapt}}) = -\sum_{t=1}^{|\mathbf{y}^*|} \log p\left(y_t^* \mid \mathcal{P}_{\text{hybrid}}, \mathbf{y}_{<t}^*\right).
\]

联合优化后 adaptor 最终达到 **7× 压缩比**，同时保持任务性能与全文本方案功能等价。

### 2.3 基础设施工程优化

- **Disaggregated Prefill-Decode Architecture**：推荐生成任务输入/输出长度极不对称（输入约 10K tokens、输出仅几百 token）。Prefill 阶段计算密集（\(\mathcal{O}(L_{in}^2)\)），Decode 阶段访存密集（\(\mathcal{O}(L_{in}\times L_{out})\)）。V2 把两阶段拆分到不同 GPU 池：更多资源给 prefill 以最大化长上下文吞吐，更少资源给 decode 以匹配其访存特性，两阶段通过 KV cache 传输通信。
- **XQA Kernel Integration**：用 XQA kernel 替代 FlashInfer，在 H20 GPU 上利用 FP8 精度加速 attention 计算（FlashInfer 主要面向 BF16）。
- **效果**：这两项基础设施优化把 MFU 从 V1 的 **11.56%** 提升到 **17.04%**；叠加 Atomized Entity Compression 与 HMAS（下节）后，整体 MFU 相对 V1 提升 **53.11%**，prefill 阶段 QPS 提升 **69.30×**，decode 阶段 TPS 提升 **7.35×**。

---

## 3. Hierarchical Multi-Agent System（HMAS）：Planner → Experts → Arbiter

### 3.1 整体设计动机

V1 的多条并行路线各自独立处理相同的用户上下文，既有计算层面的重复编码，又有认知层面的重复候选（13.46% 路线间重复率）。HMAS 用三层协同架构替代孤立并行路线：Global Planner 统一做一次意图分解，产出多个互补的 persona；每个 persona 驱动一个专属 Expert 做角色化的商品 tag 预测；Decision Arbiter 对所有 expert 的候选做联合评审、去重、精选。

### 3.2 Global Planner

**输入上下文** \(\mathcal{C} = \{\mathcal{B}, \mathcal{U}, \mathcal{E}\}\) 由三部分构成：

- \(\mathcal{B}=\{(a_i,e_i,t_i)\}\)：时序排列的用户行为（动作类型、实体、时间戳），其中实体 \(e_i\) 采用 Atomized Entity Compression 表示；
- \(\mathcal{U}=\{\mathcal{U}_{attr}, \mathcal{U}_{int}\}\)：静态人口属性 + 动态兴趣（如"骑行爱好者""二次元""数码控"），保持自然语言编码；
- \(\mathcal{E}\)：天气、季节、热点事件等实时多源环境信号，同样保持自然语言编码。

Global Planner 对 \(\mathcal{C}\) 做深度推理，分解出 \(K\) 个互补 persona：\(\{p_1,\ldots,p_K\}=f_{\text{planner}}(\mathcal{C})\)。这一步同时达成两个目标：**只做一次意图分解**（消除重复编码），以及**显式协调各 persona 覆盖不同语义方向**（避免专家之间语义重叠）。

### 3.3 Distributed Experts：两阶段训练

每个 Expert 在被分配的 persona 下独立生成一组商品 tag：\(\mathcal{T}_k = f_{\text{expert}}(p_k)\)。

**Stage 1：SFT。** 用 GPT-4 判断用户后续交互中的商品类目是否与某个 persona 语义相关，构造固定大小（15 个）的目标标签集合 \(\mathcal{C}_k^{\text{target}}\)（不足则用 GPT-4 生成的合成标签补齐，超出则随机采样 15 个），再做标准的 next-token-prediction 交叉熵训练。训练数据混合了纯行为模式（32.17%）、热点事件（6.97%）、天气相关上下文（1.19%）、其他情境信号（7.36%）以及通用指令数据（52.31%），以兼顾领域能力与通用语言/推理能力。

**Stage 2：Constrained Reinforcement Optimization（GRPO + CRS）。**

- **策略优化**：采用 GRPO，对每个输入采样一组 \(G\) 个输出，用组内相对优势 \(\hat{A}(x,y)=R(x,y)-\frac{1}{G}\sum_i R(x,y_i)\) 做裁剪式策略梯度更新，并用 KL 惩罚约束新策略不过度偏离 SFT 参考模型。
- **多奖励建模**（四项）：
  - **Accuracy Reward** \(R_{\text{acc}}\)：预测 tag 映射到类目后，对用户真实交互类目的召回率；
  - **Alignment Reward** \(R_{\text{align}}\)：用一个基于偏好对训练的奖励模型评估每个 tag 与 persona 意图及人类质量标准的对齐度；
  - **Diversity Reward** \(R_{\text{div}}\)：用 BGE embedding 计算预测 tag 集合内部的平均余弦距离，鼓励语义多样性；
  - **Length Reward** \(R_{\text{len}}\)：按词数分段给分（6–11 词满分 1.0，4–6 或 11–13 词半分 0.5，其余为 0），避免 tag 过短缺乏信息量或过长影响检索。
- **Constrained Reward Shaping (CRS)**：不同于直接加权求和（SUM）容易造成不同奖励维度梯度冲突（简单目标如多样性主导训练、准确率被牺牲），CRS 把次要奖励作为**硬约束**、只有全部达标才让主奖励生效：

\[
R_{\text{total}} = R_{\text{acc}} \cdot \mathbb{I}[R_{\text{align}} \geq \tau_{\text{align}}] \cdot \mathbb{I}[R_{\text{div}} \geq \tau_{\text{div}}] \cdot \mathbb{I}[R_{\text{len}} \geq \tau_{\text{len}}].
\]

任一约束不满足则总奖励归零，从而把"满足次要约束"和"优化主目标"解耦为两阶段过程，避免梯度互相干扰。实验（Figure 5-6, Table 4）显示 CRS 相比 SUM 梯度范数与 KL 散度更低、训练更稳定，且能同时保持准确率与多样性双双正向优化；HR@30 上：V1 26.29% → V2 Base 23.08%（未领域适配，反而更低）→ SFT 29.20% → GRPO(SUM) 27.38%（不如 SFT，验证 SUM 梯度冲突）→ **GRPO(CRS) 32.60%**（相对 SFT +3.40pp，相对 V1 +6.31pp）。

### 3.4 Decision Arbiter 与下游召回

Arbiter 对所有 expert 候选 tag 池 \(\mathcal{T}_{\text{all}}=\bigcup_k \mathcal{T}_k\) 结合完整上下文 \(\mathcal{C}\) 做联合评审（而非逐条打分），综合考虑行为相关性、画像一致性、内容具体性和有效性，选出 top-N 精炼 tag 集合 \(\mathcal{T}_{\text{final}}\)。

下游进一步做：

- **Multi-Interest User Encoding**：沿用 V1 的 user-item-tag 三塔架构，引入 Poly-Encoder 风格的 \(K\) 个可学习 context code，把用户行为编码为多个兴趣向量 \(\{\mathbf{u}_1,\ldots,\mathbf{u}_K\}\)，分别与商品塔做点积匹配；
- **Traffic Allocation via Quadratic Programming**：把认知探索（cognitive channel）与既有效果渠道（utility channel）之间的流量分配建模为带约束二次规划问题（详见附录），在保证转化目标 \(\mathcal{C}\)（下限）与曝光条目数区间 \([\mathcal{Q},\mathcal{P}]\) 的前提下最大化点击收益并通过正则项 \(\frac{\lambda}{2}\|\mathbf{x}\|^2\) 避免曝光过度集中。KKT 求解给出解析解，生产环境进一步简化为硬阈值策略 \(x_i^*=\mathbb{1}[h_i>\lambda]\)（\(h_i=s_i+\alpha(o_i-\bar o)+r\) 综合了点击收益、转化溢价与预算压力），省去分数曝光带来的推理开销。

---

## 4. Dynamic Explanation Generation：Meta-Prompting

### 4.1 问题

V1 用固定模板拼接用户兴趣和商品属性生成解释，长期上线后暴露三个问题：信息密度低（泛泛而谈）、时效适配弱（无法响应季节/热点）、表达同质化。

### 4.2 评估维度扩展

从 V1 的 4 个维度（Relevance、Factuality、Clarity、Safety）扩展到 7 个维度，新增：**Timeliness**（是否贴合当下趋势/季节/事件）、**Informativeness**（是否提供超出泛泛描述的实质信息）、**Attractiveness**（情感吸引力/说服力）。

### 4.3 两阶段生成

- **Stage 1：Style Synthesis**，模型先根据用户兴趣 \(\mathcal{U}\)、商品属性 \(\mathcal{I}\)、情境信号 \(\mathcal{S}\) 生成风格指南 \(g=f_{\text{meta}}(\mathcal{U},\mathcal{I},\mathcal{S})\)（例如"为家长群体写一段俏皮、视觉化、带情感共鸣的圣诞儿童玩具短文案"）；
- **Stage 2：Style-Conditioned Explanation Generation**，在风格指南 \(g\) 约束下生成最终解释 \(e=f_{\text{exp}}(g,\mathcal{U},\mathcal{I},\mathcal{S})\)（例如"像蓝色蝴蝶在空中旋转"）。

这种两阶段解耦让模型可以"扮演"不同风格人设，从而产出更多样、更贴合场景的解释，而不是被单一固定模板锁死。

### 4.4 Preference-Aware RL

沿用 §3.3 的 GRPO + CRS 框架，但奖励换成解释任务专属的混合奖励：

- **Rule-Based Diversity Reward**：维护大小为 160 的 FIFO 历史解释缓冲区 \(\mathcal{M}\)，用类 IDF 的方式给生成解释中的每个 token 打分（\(\log \frac{|\mathcal{M}|}{|\{e'\in\mathcal{M}: w_i \in e'\}|+1}\)），罕见 token 获得更高奖励，从而鼓励用词多样、抑制重复套话；
- **Model-Based Alignment Reward**：训练一个基于 listwise 偏好数据的奖励模型评估解释的主观质量（如 informativeness）。

CRS 把 alignment 设为主奖励、diversity 设为门控约束：\(R_{\text{total}}=R_{\text{align}}\cdot\mathbb{I}[R_{\text{div}}\geq\tau_{\text{div}}]\)。

### 4.5 效果

Diversity（基于同一商品多条解释间 1 − 平均 ROUGE-L 相似度）从 V1 的 0.631 提升到 V2 的 0.677（**+7.30%**）；人工评估的 Quality（七维度全部达标才算高质量）从 36.03% 提升到 40.73%（**+13.04%**）。

---

## 5. Agentic Judge Framework：从结果打分到过程化多维评估

### 5.1 Agent-as-a-Judge

V1 的 LLM-as-a-Judge 直接端到端预测一个质量分数，跳过了人类评估者实际经历的多维度、多步骤推理过程。V2 引入分层多智能体评审：

- **Multi-Dimension Sub-Evaluators**：为每个评估维度（详见 Appendix，item tag 4 维：Relevance/Consistency/Specificity/Validity；解释 7 维见上）实例化一个专属子评估器 \(\mathcal{E}_i\)，各自独立给出维度分 \(s_i=\mathcal{E}_i(y,d_i)\)，把复杂多目标评估拆成多个可控的单目标子任务；
- **Senior Reviewer Agent**：聚合所有维度分数 \(\{s_1,\ldots,s_D\}\)，按两阶段流程给出三档最终判断——**Superior (S)** / **Average (A)** / **Bad (B)**：(a) **缺陷检测**：任一维度出现负面/不合格信号即判为 B；(b) **优秀分级**：无致命缺陷时，再根据正面反馈的比例/模式并结合阈值 \(\tau\) 区分 S 与 A。
- **模型训练**：用模型自产样本 + 强模型（DeepSeek-R1、Qwen3-235B）输出构建训练语料；对 relevance 等维度用 batch 内随机打乱配对自动构造负例，对需要细腻判断的维度采用人工标注（覆盖维度分和整体 S-A-B 判断），最终 SFT 一个轻量的 Qwen3-32B-Instruct 作为评审模型。

**效果**（Table，人类标注为 ground truth，衡量对 Superior 识别的一致性）：item tag prediction 上三种打分模型（GPT5-mini/Qwen3-Base/Qwen3-SFT）Accuracy 均有小幅提升（+0.10pp/+0.20pp/+0.38pp），F1 提升更明显（+0.36pp/+0.60pp/+1.33pp）；解释生成任务上 GPT5-mini 和 Qwen3-SFT 同样提升（Qwen3-SFT: +1.21pp Accuracy、+5.20pp F1），但 Qwen3-Base 上 Accuracy 反而下降（0.3423→0.2764），说明 Agent-as-a-Judge 的收益依赖评审底座模型本身的能力，并非在所有配置下都一致提升。

**重要边界**：Agent-as-a-Judge 评的是**单次生成内容**（某个 tag 或某段解释），维度也局限于 Appendix 定义的任务级标准（tag 4 维、解释 7 维），并不评估 Global Planner 的 persona 分解是否合理、Arbiter 的最终选择是否恰当，更不涉及 IPV/CTR/GMV 等线上业务指标——那是第 6 节线上 A/B 测试的职责。它只是"生成质量"评审器，不是"整个推荐系统"评审器。

**示例（延续 §6.4/§7 天津用户案例）**：假设母婴专家 Expert 针对该用户的 persona 采样出三条候选 tag —— (1) "儿童保湿乳"、(2) "婴儿护肤用品"、(3) "母婴用品"。
Multi-Dimension Sub-Evaluators 分别打分：
- 候选(1)：Relevance（命中用户近期母婴+秋燥语境）✔️、Consistency（有历史母婴购买行为支撑）✔️、Specificity（具体到"保湿乳"这一品类）✔️、Validity（对应真实商品）✔️ → 四维全部通过，Senior Reviewer 先做缺陷检测（无负面信号）、再做优秀分级 → 判定 **Superior (S)**。
- 候选(2)："婴儿护肤用品"范围略宽，Specificity 打分中等（不算致命缺陷，但不够突出）→ **Average (A)**。
- 候选(3)："母婴用品"过于宽泛，Specificity 直接判负 → 缺陷检测阶段即被判 **Bad (B)**，不进入优秀分级。

这一步的产出只是这三条候选各自的 S/A/B 标签，尚不能直接当作连续可微的 RL 奖励，也没有对"这次推荐请求整体好不好"下结论——这正是 §5.2 要解决的问题。

### 5.2 Judge-as-a-Reward：把离散判断蒸馏成稠密奖励

直接把 Agent-as-a-Judge 用于 RL 训练面临两个问题：离散 S/A/B 标签粒度太粗，不利于策略梯度估计；多步评估在线 RL 训练中开销过大。因此设计蒸馏框架：

- **架构**：从 Agent Judge checkpoint 初始化，把语言建模头替换为标量 value head（sigmoid 输出到 \([0,1]\)）：\(r=f_{\text{RM}}(y,\mathcal{U},\mathcal{I},\mathcal{S})\)；
- **Listwise Learning-to-Rank 训练**：按 Senior Reviewer 给出的三档标签分组，对每个质量档 \(g\)，同档样本互为正例、所有更低档样本作为负例，用统一的对比损失同时捕获 S vs AB、A vs B 等全部两两偏好关系：

\[
\mathcal{L}_{\text{RM}} = -\sum_{g \in \{\text{S}, \text{A}\}} \sum_{y_g \in \mathcal{Y}_g} \log \frac{\exp(f_{\text{RM}}(y_g))}{\exp(f_{\text{RM}}(y_g)) + \sum_{g' < g} \sum_{y_{g'} \in \mathcal{Y}_{g'}} \exp(f_{\text{RM}}(y_{g'}))}.
\]

- **工程加速**：同一对比组内样本共享相同上下文 prompt，只是生成内容不同，因此可以只计算一次共享前缀表示并在候选间复用，减少冗余计算。

**接上例**：候选(1)(2)(3)被 Senior Reviewer 标注为 S/A/B 后，构成一个对比组：训练 Judge-as-a-Reward 时，(1) 作为正例、(2)(3) 作为负例（S vs AB）；(2) 作为正例、(3) 作为负例（A vs B），联合优化让打分模型满足 \(f_{\text{RM}}(1) > f_{\text{RM}}(2) > f_{\text{RM}}(3)\)。训练好之后，这个奖励模型可以直接对**任意新采样的 (tag, persona) 组合**给出一个 \([0,1]\) 的连续分数，而不必再跑一遍完整的"多维子评估器 + Senior Reviewer"多步推理——§3.3 中 Expert 做 GRPO 训练时，Alignment Reward \(R_{\text{align}}\) 正是直接调用这个训好的 \(f_{\text{RM}}\) 对新一轮采样输出打分，从而把 §5.1 这次离散评审的经验，转化成了可以驱动下一轮策略更新的连续奖励信号。

**Self-Improving Flywheel**：策略生成多样输出 → Agent-as-a-Judge 给出多维 S-A-B 判断 → Judge-as-a-Reward 把判断蒸馏为稠密可微奖励 → GRPO 用该奖励驱动策略优化；该循环在初始人工标注之后可自主运转，持续积累训练信号。

**效果**：Listwise RM 相对 V1 提升明显（HR@30 +24.1%、Quality +13.0%），相对 Point-wise RM 也有提升（HR@30 +4.4%、Quality +8.2%），验证了保留完整层级偏好顺序（S ≻ A ≻ B）比逐点独立训练更有利于提供判别性优化信号。

---

## 6. 实验：线上 A/B 与案例研究

### 6.1 实验设置

在淘宝首页"猜你喜欢"进行为期两周的在线 A/B 测试：实验组与对照组（RecGPT-V1）各占总流量 1%；分别在 **Item 场景**（网格布局的直接商品推荐）和 **Feed 场景**（混合商品、广告、直播等内容的信息流）评估。

### 6.2 指标

- **短期指标**：IPV（商品详情页访问量）、CTR（点击率）、TV（交易额）、GMV（含退款的总交易额）、ATC（加购数）；
- **长期指标**：NER（新颖曝光率，未交互过的商品占比）、LT-14/LT-30（14/30 日留存）。

### 6.3 结果

| 场景 | IPV | CTR | TV | GMV | ATC | NER | LT-14 | LT-30 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Item | +3.64% | +3.01% | +2.11% | +3.39% | +3.47% | +11.46% | -- | -- |
| Feed | +1.29% | +1.50% | +0.34% | +1.53% | +0.99% | +4.49% | +0.04% | +0.05% |

Item 场景各项短期指标均有明显提升；Feed 场景提升幅度较小但方向一致。长期指标上 NER 提升尤其显著（Item +11.46%、Feed +4.49%），论文将其归因于多智能体协同 + 环境信号融合有效缓解了信息茧房效应；LT-14/LT-30 绝对提升虽小（+0.04%/+0.05%），但对平台长期健康度而言仍具有意义。

摘要与引言部分给出的整体数字与实验章节 Item 场景细项数字略有出入（如摘要称 +2.98% CTR / +3.71% IPV / +2.19% TV，引言称 +3.01%/+3.64%/+2.11%，结论称 +4.68%/+3.40%/+4.05%），推测对应不同统计口径（如 Item+Feed 加权、不同实验周期等），论文正文未明确解释三处数字的具体差异来源，读者引用时应以第 6 节 Table（Item/Feed 分场景明细）为准。

### 6.4 案例研究

针对一位 35 岁天津女性用户，系统结合压缩后的行为历史与实时环境信号（转凉天气、临近中秋节、临近万圣节），Global Planner 分解出三个互补 persona：**女装专家**、**母婴专家**、**健康专家**。三个 Expert 并行独立生成对应标签：女装专家响应降温天气推荐"羊毛混纺开衫"；母婴专家同时给出"儿童保湿乳"（应对秋燥）和"儿童万圣节服装"（预判节日），体现时效适应能力；健康专家结合历史健身兴趣与天气推荐"可调节哑铃套装"。Decision Arbiter 综合选出三件最终商品，并配上 Meta-Prompting 生成的场景化解释（如"裹住整个秋日暖阳""润泽宝贝肌肤""哑铃就够了"）。该案例直观展示了 HMAS 在多样意图覆盖与情境化适配上的协同效果。

---

## 7. 端到端示例：训练与推理视角下的完整流程

沿用 §6.4 天津用户（35 岁女性）的真实案例，把它拆成"离线训练各模块"和"线上一次推理请求"两条线，串联本文前述所有组件。

### 7.1 训练视角：各模块分别在哪个阶段、用什么数据训练

各模块训练相互独立、产出的模型/权重在推理时被组合调用，而不是一次性端到端联合训练（这也是论文在 §7 中指出的局限）：

1. **Adaptor（§2）**：离线一次性训练，输入是海量商品标题/用户历史查询文本（如该用户过去点击过的"高筒靴女秋冬新款""复古蓝牙小音箱"等）。用 BGE/Qwen3-Embedding 编码后接 adaptor 投影，用 GPT-4 生成的 self-perception QA 对 + User Interest Mining/Item Tag Prediction 的全文本参考答案做监督，训练目标是让 hybrid prompt（把商品文本替换成 `[entity]`）复现全文本 prompt 下的模型输出。LLM backbone 全程冻结，只更新 adaptor 参数。训练完成后，该用户后续所有行为记录都可以离线/在线地被压缩成原子表示，供下游 Planner/Expert 复用，不需要为每个用户单独训练。
2. **Expert（§3.3）**：先用 SFT 让每个 persona（如"女装专家"）学会从该 persona 视角输出商品 tag——训练样本由 GPT-4 判断用户历史上"后续交互类目"是否语义符合该 persona 而自动构造（例如该用户之后购买/浏览的女装类目会被标为"女装专家" persona 的正向监督）；再用 GRPO + CRS 做强化学习，其中 Accuracy Reward 看预测 tag 是否命中用户真实交互类目，Alignment Reward 直接调用 §7.2 中训练好的 Judge-as-a-Reward 模型（下一条）打分，Diversity/Length Reward 按规则计算。三个 persona（女装/母婴/健康专家）对应三份独立训练好的 Expert（或同一 Expert 权重在不同 persona prompt 下推理，论文未明确是否共享权重）。
3. **Meta-Prompting 解释模型（§4）**：先 SFT 学会"给定用户兴趣+商品属性+情境信号 → 风格指南 → 风格化解释"两段式生成（例如学会针对"母婴+秋燥+儿童保湿乳"这类组合产出"润泽宝贝肌肤"式的解释），再用 GRPO + CRS 做 RL，Alignment Reward 同样来自 Judge-as-a-Reward，Diversity Reward 由 160 条历史解释组成的 FIFO 缓冲区规则计算。
4. **Agent-as-a-Judge（§5.1）**：用真实/合成的 tag 预测与解释生成样本，混合模型自产输出、强模型（DeepSeek-R1、Qwen3-235B）输出和人工标注，训练出 Qwen3-32B-Instruct 的多维子评估器 + Senior Reviewer，使其能像人类一样先逐维度打分（相关性、时效性…）再汇总成 S/A/B。
5. **Judge-as-a-Reward（§5.2）**：从上一步的 Agent Judge checkpoint 出发，替换成标量 value head，用同一批样本按 S≻A≻B 的层级偏好做 listwise 对比学习，得到可以给任意 (tag, persona) 或 (解释, 用户/商品/情境) 组合直接打分的稠密奖励模型——它就是第 2、3 步 RL 训练中 Alignment Reward 的来源，从而闭合"评审模型训练好之后反过来指导 Expert/解释模型训练"的飞轮。

也就是说，围绕这一位用户的推荐效果，实际由五组分别训练、互不实时耦合的模型共同决定：**adaptor（表示压缩）→ Expert × 3（persona 化 tag 生成）→ 解释模型（风格化文案）**，以及在幕后支撑训练的 **Agent-as-a-Judge → Judge-as-a-Reward**（不出现在线上单次请求的关键路径中，只在训练/评估阶段被调用）。

### 7.2 推理视角：一次线上请求的完整链路

当这位用户打开淘宝首页时（背景：转凉天气、临近中秋节、临近万圣节）：

1. **表示压缩（§2）**：她的完整行为历史（可能数万 token）先经过训练好的 adaptor，把其中的商品/查询文本替换成 `[entity]` 原子表示，只保留用户属性、时间戳等自然语言信息，形成压缩后的 hybrid context（对应 §2.2 案例中 21,349→5,158 tokens 的压缩效果）。
2. **Global Planner 分解意图（§3.2）**：Planner 读取 \(\mathcal{C}=\{\mathcal{B},\mathcal{U},\mathcal{E}\}\)——压缩后的行为 \(\mathcal{B}\)、用户画像 \(\mathcal{U}\)、以及本次请求的实时环境信号 \(\mathcal{E}\)（转凉天气、中秋、万圣节）——一次性分解出三个互补 persona：女装专家、母婴专家、健康专家，分发给对应 Expert。
3. **Distributed Experts 并行生成（§3.3）**：三个 Expert 各自基于分配到的 persona 独立推理，互不重复编码完整历史：女装专家输出"羊毛混纺开衫"；母婴专家输出"儿童保湿乳"和"儿童万圣节服装"；健康专家输出"可调节哑铃套装"。
4. **Decision Arbiter 联合评审（§3.4）**：把三路候选 tag 池 \(\mathcal{T}_{\text{all}}\) 连同完整上下文 \(\mathcal{C}\) 一起交给 Arbiter，按行为相关性、画像一致性、内容具体性和有效性联合打分排序，选出最终 3 个 tag（对应 §6.4 中三件被选中商品）。
5. **召回与流量分配**：最终 tag 经 tag 塔编码后与商品塔做多兴趣匹配召回具体商品；这批"认知探索"候选与既有"效果渠道"候选一起，按 §3.4 末尾的二次规划流量分配策略（阈值 \(h_i>\lambda\) 则曝光）决定最终是否进入展示位。
6. **Meta-Prompting 生成解释（§4.3）**：对每件被选中商品，解释模型先做 Stage 1 风格合成（如"面向家长、俏皮温暖的语气"），再做 Stage 2 风格条件生成，产出"裹住整个秋日暖阳""润泽宝贝肌肤""哑铃就够了"这类场景化文案，与商品一起展示给用户。
7. **（不在关键路径上）Agent-as-a-Judge 离线评估**：本次曝光及用户后续点击/购买行为会被回收，用于离线评估（Agent-as-a-Judge 打分）和后续 RL 迭代（Judge-as-a-Reward 提供奖励），但不会在这次请求的实时链路中对展示结果做二次评审——评审/奖励回路作用于"下一版" Expert 和解释模型的训练，而不是当前这次请求。

这条链路体现了论文强调的关键区别：**推理时只跑一次 Planner + 一轮 Expert 并行推理 + 一次 Arbiter 评审**，避免了 V1 多路线各自重复编码 32K 行为序列的浪费；而 Judge/Reward 相关的多智能体评审只发生在离线训练侧，不增加线上请求的延迟。

---

## 8. 小结与局限
**与 RecGPT-V3 的关系**：后续的 RecGPT-V3（arXiv:2607.15591）延续了本文的 Global Planner 概念，并在此基础上引入 Memory Hub（免除每次请求都要重新处理长历史）、Hybrid-modal Foundation Model（原生联合建模 Text Tag 与 Semantic ID）以及 Latent Intent Reasoning（把显式 CoT 压缩为至多 10 个 latent token），RecGPT-V3 的技术报告将 V2 的 HitRate 奖励机制列为其 RLRF 改进的直接对比基线。
