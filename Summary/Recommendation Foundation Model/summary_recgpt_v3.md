# RecGPT-V3 Technical Report

## 基本信息

- **论文标题**：RecGPT-V3 Technical Report
- **arXiv**：[2607.15591](https://arxiv.org/abs/2607.15591)
- **作者**：RecGPT Team（淘宝）
- **领域**：生成式推荐、用户建模、Semantic ID、LLM 推理、工业部署
- **基础模型**：Qwen3-14B
- **部署场景**：淘宝首页“猜你喜欢”，覆盖数亿日活用户

---

## 综合理解与点评

**读者理解**：本文在 Qwen3-14B 基础上训练了一个特异化的 LLM，输入用户文本表达的历史行为，输出用户感兴趣的 Tag 和对应的 SID prefix，之后再用这些 Tag 做下游任务。

这个理解抓住了论文的主线：RecGPT-V3 确实从 Qwen3-14B 出发，通过推荐领域的 continual pre-training、instruction tuning、显式到隐式 CoT 对齐和 RLRF，得到能够理解用户意图并操作商品 SID 的特异化模型。不过，若严格对齐论文，需要做四点修正：

1. **它不是单一 LLM 直接完成全部流程**。完整系统由 Memory Hub、Global Planner、多 Expert、Hybrid-modal Foundation Model、Hybrid Retriever 和 Production Ranker 组成；Qwen3-14B 特异化模型是推理核心，但不是整个推荐 pipeline。需要注意，论文中经过 continual pre-training + instruction tuning + 显式/隐式 CoT 对齐 + RLRF 训练的，是 **Expert / Hybrid-modal Foundation Model**，而不是 Global Planner。Global Planner 延续自 RecGPT-V2（引用为独立技术报告 arXiv:2512.14503），V3 正文只描述了它的功能（读取用户状态、拆解出 intent persona、把 persona 连同相关 memory 和近期点击分发给对应 Expert），并未说明它具体基于哪个基础模型、是否与 Expert 共享权重、或是否在 V3 中重新训练。V3 对 Planner 的改动只是**输入来源**：用 Memory Hub 产出的压缩记忆 + 近期行为 delta，替换掉原本要重新编码的完整行为序列，因此表 2 中"每次推理 33.43%"和"周期性 Memory Curation 10.77%"衡量的是 Planner 侧用户建模计算量的变化，而不是对 Planner 模型本身的重新训练。
2. **输入不是纯文本历史**。Global Planner 主要读取自然语言形式的结构化 Memory Units 和 recent behavior delta；Expert 的近期点击输入中，每个商品由 **完整 SID + 短标题** 表示。Foundation Model 的 `sid2sid` 训练甚至完全在 SID 行为序列上进行。
3. **输出的是完整两级 SID，不宜称为 SID prefix**。每个商品 SID 由两个 codebook token 构成：第一个是 coarse semantic code，第二个负责细粒度区分。相似商品会共享第一段 prefix，但模型用于检索的是完整两-token SID；这不同于 TokenMinds 主动截断 full SID、只输出粗粒度 prefix 的设计。
4. **下游不是只使用 Tag**。Expert 联合生成 Text Tag 与 SID；Hybrid Retriever 分别嵌入两种信号并融合。Tag 提供开放世界覆盖，SID 提供更具体、用户特异的协同信号，之后候选再进入 Production Ranker。仅在 RLRF 的奖励定义中，论文主要以生成的 Tag 集合检索商品并读取 Top-100 CTRScore。

更准确的一句话表述是：

> **RecGPT-V3 以 Qwen3-14B 为基础训练推荐特异化的混合模态 LLM；在线读取结构化用户 Memory、近期行为的 SID 与短标题，通过 latent reasoning 生成 Text Tags 和完整两级 SIDs，再由 Hybrid Retriever 联合两种意图信号召回商品，并交给生产 Ranker 排序。**

---

## 0. V1 → V2 → V3 全流程对照表

一个常见但需要修正的理解是："V1/V2/V3 的核心都是产生 Tag + SID，供后续检索/排序使用"。按论文原文核对：**只有 V3 引入了 SID**；V1 和 V2 的输出始终只有自然语言 Tag，没有 SID 这个概念——V1 的 TAR 三塔和 V2 的 Multi-Interest 检索都只嵌入 Tag，SID 是 V3 才新增的第二种意图表达（两级离散 code，配套 Hybrid-modal Foundation Model 和 Hybrid Retriever 才存在）。因此三者的区别不止"训练流程改进 + V3 加 Memory Hub + V3 加 latent reasoning"，而是在**输出的中间表达、整体推理架构、上下文压缩方式、是否引入 RL、检索侧结构**等多个维度都发生了变化。下表按阶段逐一对照（各行分别对应 `summary_recgpt_v1.md`、`summary_recgpt_v2.md` 与本文件相应章节）：

| 阶段 | RecGPT-V1 | RecGPT-V2 | RecGPT-V3 |
|---|---|---|---|
| **输出的中间表达** | 仅 **Text Tag**（"Modifier + Core-Word"，如"户外防水防滑登山靴"） | 仅 **Text Tag**（persona-conditioned，无 SID） | **Text Tag + 两级 Semantic ID**（65,536 SID token 词表，与 Tag 同时生成） |
| **整体推理架构** | 3 个独立专职 LLM **串行**两步：$\mathcal{LLM}_{UI}$ 兴趣挖掘 → $\mathcal{LLM}_{IT}$ 标签预测（各自独立训练、独立调用） | **Hierarchical Multi-Agent System**：Global Planner 一次性分解出多个互补 persona → 多个 persona-conditioned Expert **并行**直接生成 tag（隐式合并了"兴趣挖掘"与"标签预测"两步）→ Decision Arbiter 联合评审精选 | 沿用 V2 的 **Global Planner**（结构不变，只换输入来源）→ Expert 升级为单一 **Hybrid-modal Foundation Model**，原生同时输出 Tag 与 SID，中间推理用至多 10 个 **latent token** 完成 |
| **用户历史/上下文压缩方式** | **规则式文本压缩**：Reliable Behavior Extraction（过滤低质行为）+ Item-level/Sequence-level 两层聚合，把行为序列压成可读文本；98% 用户可纳入 128K 窗口，但每次请求仍需重新读取压缩后全文本 | **可学习的原子表示压缩（Atomized Entity Compression）**：adaptor 把商品/查询文本投影为单一 embedding token `[entity]`，7× 压缩比（32K→11K tokens）；每次请求仍需从头压缩并编码完整历史 | **持久化状态记忆（Memory Hub）**：把长期历史一次性压缩成结构化 Memory Units，之后仅**增量维护**（约两月一次 curation）；线上请求只读"记忆 + 近期行为 delta"，不再重新处理完整历史，是三代里唯一做到"有状态"的方案 |
| **训练数据构造 & 训练范式** | 纯 **SFT/自训练**，无策略梯度 RL：CL-MFT（16 个课程任务）→ DeepSeek-R1 蒸馏的 Reasoning-Enhanced Pre-alignment → Self-Training Evolution（自产数据 + LLM-as-a-Judge 过滤）；标签任务另有双周 Incremental Learning（QwQ-32B 做数据净化/补全/再平衡） | **SFT + 强化学习**：Expert 先 SFT（GPT-4 标注 persona-target 类目），再用 **GRPO + Constrained Reward Shaping (CRS)** 做 RL，四项奖励 Accuracy/Alignment/Diversity/Length；Alignment Reward 来自本版本新增的 Judge-as-a-Reward 模型 | Foundation Model 走 **Continual Pre-Training**（SID-grounding + ~10% 通用数据防遗忘）→ **Instruction Tuning**（sid2title/title2sid/sid2tag/tag2sid/sid2cmd/sid2sid 等多任务）；latent reasoning 再走 **Explicit-to-Implicit CoT Alignment**（DeepSeek-V3.2 teacher）+ **RLRF**（沿用 V2 的 CRS 门控结构，但主奖励从 V2 的 Accuracy/HitRate 换成**生产 Ranker 真实 CTRScore**） |
| **Intermediate Evaluation**（用什么评测 tag 好坏） | **LLM-as-a-Judge**：专门训练的 LLM 分类器逐维度打分/分类，辅以定期的人工校准（Milestone-Based Human Supervision），主要用于离线过滤自训练数据、监控质量，不接入 RL | **Agent-as-a-Judge**：多个 LLM 子评估器分维度打分 → Senior Reviewer 汇总成 S/A/B → 蒸馏成连续的 **Judge-as-a-Reward** 模型，首次把"评测结果"直接接入 RL 训练 | 沿用 V2 的评审-奖励机制做门控约束，但主奖励换成 **RLRF**——直接读取生产排序模型对候选商品的真实 CTRScore 来判定 tag 好坏，比 LLM 主观判断更贴近线上真实效果 |
| **Tag(&SID) 的下游使用方式** | **TAR 三塔**（Item/User/Tag），协同分数与语义分数按可调权重 $\beta$ 线性融合 | 三塔扩展为 **Multi-Interest User Encoding**（Poly-Encoder 多兴趣向量）+ tag 塔匹配；新增基于**二次规划**的认知探索 / 效果渠道流量分配 | **Hybrid Retriever**：Tag embedding 与 SID embedding 拼接投影成统一 intent vector 做 Target-Attention 打分，用 utility + relevance 联合损失约束偏好顺序（Tag 负责开放世界覆盖，SID 负责用户特异协同信号） |

**小结**：把 Memory Hub 和 latent reasoning 看作 V3 的两个亮点没有错，但它们只是"V2→V3"这一跳的变化；"V1→V2"这一跳的核心变化其实是**架构从串行独立 LLM 变成分层多智能体**、**压缩方式从规则文本聚合变成可学习原子 embedding**、以及**训练范式从纯 SFT 引入 GRPO+CRS 强化学习和评审-奖励飞轮**——这些改动早于 V3、且并不涉及 SID。SID 本身连同 Hybrid-modal Foundation Model、Hybrid Retriever，是 V3 独有的新增内容，不应归到 V1/V2。

---

## 1. 问题背景

RecGPT 系列希望让 LLM 不只是拟合行为共现，而是理解用户行为背后的意图。但 V1/V2 的工业部署暴露出三个瓶颈：

1. **无状态用户建模**：每次请求都重新处理完整历史。高活跃用户的序列约为 55K tokens；Global Planner 占 agentic intent analysis 约 95% 的计算量，而且反复分析已经识别过的复购周期、品牌忠诚和季节性偏好。
2. **Tag-to-Item 信息瓶颈**：LLM 先生成自然语言商品标签，再由下游模型映射到具体商品。标签适合表达开放世界意图，却难以携带细粒度 item-level 与协同过滤信号。
3. **显式推理成本过高**：显式 CoT 通常长达数千 tokens，串行自回归解码难以满足十亿用户、高 QPS 场景的延迟和成本约束。

RecGPT-V3 对应提出三个核心模块：

| 瓶颈 | RecGPT-V3 模块 | 核心变化 |
|---|---|---|
| 每次重读全历史 | Memory Hub | 将长期行为压缩成可增量维护的结构化记忆 |
| 文本标签难以精确落到商品 | Hybrid-modal Foundation Model | 在文本词表之外原生加入 Semantic ID token |
| 长 CoT 解码昂贵 | Latent Intent Reasoning | 用至多 10 个 latent token 内化推理，并支持按需重建文本解释 |

整体数据流可以概括为：

```text
完整长期行为
  -> 初次 Structured Behavior Compression
  -> 持久化 Memory Units

Memory Units + 新增行为 delta
  -> Evolving Memory Curation
  -> 更新后的用户记忆
  -> Global Planner / Multi-Expert
  -> 文本 Tag + Semantic ID + Latent CoT
  -> Hybrid Retriever
  -> 候选商品
  -> Production Ranker
```

---

## 2. Memory Hub：从无状态长历史到持续演化的用户记忆

### 2.1 记忆单元

Memory Hub 将原始行为序列

\[
\mathcal{B}=\{b_1,\ldots,b_N\}
\]

压缩为数量远小于行为数的记忆集合

\[
\mathcal{M}=\mathcal{F}_{\phi}(\mathcal{B})=\{m_1,\ldots,m_K\},\quad K\ll N.
\]

每个 memory unit 包含：

- **Behavior Pattern**：从数百种电商行为模式 taxonomy 中选择的模式标签；高置信度但未被覆盖的模式允许扩展 taxonomy。
- **Preference Summary**：自然语言形式的偏好、趋势和消费特征总结。
- **Representative Indices**：指回原始行为的索引，用于 provenance 和解释。
- **Preferred Brands**：偏好品牌。
- **Temporal Activity**：首次出现、活跃高峰等时间模式。
- **Timestamp**：记忆创建或更新时间。

该设计强调三种性质：

- **Condensed**：只保留连贯、高置信度模式，显著缩短 Planner 输入。
- **Traceable**：记忆保留到原始行为的索引，避免摘要完全脱离证据。
- **Evolvable**：新行为到达后只更新相关单元，并为无法归入已有单元的行为创建新记忆。

### 2.2 初始压缩与增量维护

Memory Hub 分成两个过程：

1. **Structured Behavior Compression**：首次读取完整历史并创建结构化记忆。
2. **Evolving Memory Curation**：后续只读取旧记忆与新增行为，不再访问完整历史。

设时刻 \(t\) 的记忆为 \(\mathcal{M}^{(t)}\)，新增行为为 \(\Delta\mathcal{B}^{(t,t+\delta)}\)，增量维护执行：

\[
\mathcal{G}(\mathcal{M}^{(t)},\Delta\mathcal{B}^{(t,t+\delta)})
\rightarrow
(\mathcal{M}^{\mathrm{update}},\mathcal{M}^{\mathrm{new}}).
\]

其中：

- 若新增行为与已有模式相关，则重新综合该单元的当前状态，而不是把新文本机械追加到旧摘要。
- 若没有相关新证据，则保留原单元不变，避免无依据漂移。
- 未匹配行为若形成连贯的新模式，则创建新单元。
- 更新已有单元与抽取新单元在一次模型 forward 中联合完成。

论文给出的生产配置是 **每两个月执行一次增量 curation**。在线推理时，Planner 使用“压缩记忆 + 最近行为 delta”，而不是完整历史。

### 2.3 质量与计算收益

人工评估：

| 评估对象 | 标注数 | 准确率 |
|---|---:|---:|
| Behavior Pattern | 2,514 | 82.89% |
| Behavior Index | 21,268 | 95.27% |

计算成本以 RecGPT-V2 的完整历史建模为 100%：

| RecGPT-V3 成本项 | 相对成本 |
|---|---:|
| 每次推理 | 33.43% |
| 周期性 Memory Curation | 10.77% |
| 合计 | 44.20% |

因此 Memory Hub 将 Planner 侧总计算降低 **55.8%**。其本质是把每次请求都支付的长历史读取成本，改成低频、可摊销的记忆维护成本。

需要注意，论文不同位置对压缩率有两种表述：Memory Hub 开头写“80% token reduction”，图注和引言写“94.5%”。正文没有解释两者是否对应不同用户集合、阶段或统计口径，不能直接视为同一个严格复现实验结果。

### 2.4 生成 Memory Hub 时的输入数据格式

Memory Hub 有两个生成阶段，输入格式相同、覆盖范围不同：

- **Structured Behavior Compression 的输入**：完整原始行为序列 \(\mathcal{B}=\{b_1,\ldots,b_N\}\)。论文正文没有给出 \(b_i\) 的严格字段 schema（不是形如 `{item_id, action, category, timestamp}` 的结构化 JSON），只把它称为“raw behavioral sequence”。但从论文案例描述看，每个 \(b_i\) 对应一次电商行为（搜索、点击、浏览或购买），并绑定具体的查询词或商品，例如案例研究中提到的“customized-keyboard interactions”“Astrox-related searches and clicks”。
- **Evolving Memory Curation 的输入**：不再读取 \(\mathcal{B}\)，而是「上一版记忆 \(\mathcal{M}^{(t)}\) + 新增行为 delta \(\Delta\mathcal{B}^{(t,t+\delta)}\)」。\(\mathcal{M}^{(t)}\) 中每个单元遵循固定 schema：Behavior Pattern、Preference Summary、Representative Indices、Preferred Brands、Temporal Activity、Timestamp；\(\Delta\mathcal{B}\) 的形式与初次压缩的输入相同，只是仅包含区间 \([t,t+\delta)\) 内新增的行为。

论文 Table 1 给出的真实记忆单元示例（用作后续 curation 的输入之一）：

| 字段 | 示例 |
|---|---|
| Behavior Pattern | K-pop Fandom |
| Preference Summary | Dedicated fan of a K-pop girl group. Full-lifecycle engagement from album pre-orders to merchandise collection and live concert participation. Strong focus on collection completeness and item preservation. |
| Representative Indices | 549, 553, 558, 563, 564, 565, ... |
| Preferred Brands | SingBA, 时代良品, 珍琢, ... |
| Temporal Activity | First appeared March 2023; peaked June and Aug–Oct 2023. |
| Timestamp | Created: 2023-10-31 |

下一次 curation 时，模型同时读取这个单元和新一批行为 delta（例如新增的点击/购买记录），联合产出「更新后的单元」或「保持不变」的判断，而不重新读取该用户全部历史行为。

### 2.5 生成的 Memory Hub 如何作用于下游任务

Memory Hub 的产出（记忆单元集合 \(\mathcal{M}\)）替换的是 Global Planner 原本要重新编码的完整行为序列：Planner 现在读取“压缩记忆 + 最近行为 delta”来判断用户当前最强的意图簇、分配 persona，再把 persona、相关记忆和近期点击交给对应 Expert；Expert 才生成 Text Tag 与 SID。也就是说，Memory Hub 不直接产出推荐结果，而是把“用户是谁、要什么”的结构化理解沿 Planner → Expert 这条链路逐层传递下去。

论文附录的 latent token 重建案例可以印证记忆内容如何具体落入 Expert 的 prompt。一个 Expert 输入的用户轮包含：

```text
You are a bags and accessories expert. User profile: ..., who needs to balance
fashion and practicality in daily work and life...
Interest pattern analysis:
- Repeatedly explores and purchases backpacks, soft-leather bags, and bags with
  rhinestone/diamond details; favors novel designs such as neo-Chinese vintage
  backpacks.
- Values both function and appearance: bags emphasize capacity ...
```

这里的“Interest pattern analysis”正是 Memory Hub 产出的结构化偏好摘要（对应 Preference Summary 字段）在下游 prompt 中的呈现形式；Expert 基于它和近期点击做 latent reasoning，再输出 Text Tag / SID。

论文正文的端到端案例研究（对应 Figure "case study"）更完整地展示了这条链路：一位淘宝用户的原始行为被 Memory Hub 压缩为“数码、羽毛球、母婴”等结构化记忆单元；随后 Astrox 相关的搜索和点击到达时，只更新羽毛球单元（母婴单元随子女成长自然演化，家装类单元因无新证据被保留，新的跑步兴趣被建为新单元）。基于更新后的记忆，RecGPT-V3 仅用 10 个 latent token 完成推理，可按需重建出“该用户是进攻型打法的资深羽毛球爱好者，需要全碳球拍、球线、维护工具”等可读 rationale，并最终生成由 SID 落地的羽毛球装备推荐。第 6 节中 Alice 的示例即参考了这一真实案例研究构造，用于展开更完整的训练与推理细节。

---

## 3. Hybrid-modal Foundation Model：文本与 SID 联合建模

### 3.1 为什么文本标签不够

自然语言标签有较强泛化能力，可以表达“适合马拉松训练的轻量跑鞋”这类开放意图；但一个标签通常对应大量异质商品，不能精确表达商品协同关系和 item-level 证据。

RecGPT-V3 因此让 Qwen3-14B 原生支持两种输出模态：

- **Text Tag**：覆盖广、利用开放世界知识，负责表达“用户想要什么”。
- **Semantic ID**：范围窄、用户特异性强，负责表达“哪些商品可能满足该需求”。

### 3.2 商品多模态表示

每个商品包含文本、图像和 side information。系统使用 CN-CLIP 分别编码三种输入，再由 Q-Former 融合：

\[
\mathcal{H}^i = \mathcal{F}(\mathcal{E}(I^i_{\text{text}}),
\mathcal{E}(I^i_{\text{image}}),
\mathcal{E}(I^i_{\text{side}})).
\]

正样本来自行为日志中的高频共现商品，但会过滤多模态相似度过低的共现对，以减少偶然共现噪声。训练使用 fused-level、text-level 和 image-level 三项 InfoNCE，使表示同时吸收内容语义与协同行为信号，并为后续 RQ-VAE 提供较稳定、可分离的空间。

### 3.3 两级 RQ-VAE 与 SID 词表

多模态商品向量经 RQ-VAE 量化为两级离散代码：

- 每级 codebook 大小为 32,768。
- 共向 LLM 词表加入 65,536 个 token，即 `<C_0>` 到 `<C_65535>`。
- 每个商品 SID 由两个 token 构成。
- 第一个 token 表示粗粒度语义簇，第二个 token 在簇内进一步区分商品。

相似商品可共享 coarse prefix，使模型能在 SID 空间中利用层级语义进行泛化。

### 3.4 两阶段训练

#### 阶段一：Continual Pre-Training

目标是让随机初始化的 SID token embedding 与文本语义对齐，同时避免基础模型灾难性遗忘。

- SID-grounding 样本把商品 SID 与 title、category 等文本属性配对。
- 训练数据中混入约 **10% 通用领域文本**，覆盖数学、代码、科学、医疗和 instruction-following。

#### 阶段二：Instruction Tuning

训练任务包括：

| 任务 | 映射 | 数据占比 |
|---|---|---:|
| `sid2title` | SID -> title | 13.8% |
| `title2sid` | title -> SID | 14.7% |
| `sid2tag` | SID -> tag | 11.5% |
| `tag2sid` | tag -> SID | 6.2% |
| `sid2cmd` | SID -> commodity category | 13.8% |
| `sid2sid` | 历史 SID -> 下一次点击 SID | 20.0% |
| 通用领域任务 | 推理、实体抽取、电商指令等 | 约 20% |

`sid2sid` 完全在 SID 空间中完成序列推荐：输入按 3 天、2 天、1 天、12 小时、1 小时等多粒度时间窗组织的点击 SID，输出下一次点击 SID。这一任务用于把 collaborative signal 注入模型，而不依赖商品文本。

### 3.5 通用能力是否保留

加入通用数据后，相对原始 Qwen3-14B：

- GSM8K：94.31% -> 92.65%。
- MMLU：下降 2.65%。
- CMMLU：下降 4.49%。
- IFEval：81.52% -> 75.60%。

不加入通用数据时出现严重能力坍塌：GSM8K 4.70%、MMLU 0.12%、CMMLU 0.01%、IFEval 23.29%。这说明在推荐领域继续预训练时，通用数据混合不是可选装饰，而是保持 LLM 推理和指令遵循能力的关键正则化手段。

同时，加入通用数据没有明显破坏 SID-text 双向对齐，并把下游 category-level HR@30 从 0.2250 提升到 0.3050，平均 tag 数从 23.88 增至 28.77，平均覆盖 category 数从 32.49 增至 41.73。

---

## 4. Hybrid Retrieval：联合消费 Tag 与 SID

附录给出了下游检索器的具体实现。每个意图同时有文本标签 \(t_i\) 和 SID \(d_i\)，分别映射为 embedding 后拼接投影：

\[
\mathbf{q}_i = W_{\text{proj}}[\mathbf{e}^{(i)}_t \| \mathbf{e}^{(i)}_d] + \mathbf{b}.
\]

这些融合后的 intent vectors 作为 Target-Attention 的 query，与候选商品表示交互并计算检索分数。

**Tag/SID 生成之后如何具体用于召回**：Expert 一次推理通常会输出多个 \((t_i, d_i)\) 意图对（例如 6.6 节示例中的 Tag1/SID1、Tag2/SID2、Tag3/SID3）。每个意图对独立走一遍上述流程：查文本 embedding 表和 SID embedding 表得到 \(\mathbf{e}^{(i)}_t\)、\(\mathbf{e}^{(i)}_d\)，拼接投影得到该意图专属的 query 向量 \(\mathbf{q}_i\)；这个 \(\mathbf{q}_i\) 再对候选商品库中每个商品的表示做 Target-Attention 打分，得到该意图下所有候选商品的检索分数。服务时按分数取每个意图各自最相关的一批商品，再把同一次请求里全部意图（本例中 3 组 Tag/SID）的结果合并、去重，构成这次请求的 Hybrid Retrieval 候选集，随后一并送入 Production Ranker 做统一排序（对应 6.6 节 Step 4 -> Step 5）。论文没有给出服务时每个意图具体截断的候选数量，只给出了离线评估口径下 HR@500 / HR@1000（即候选池规模在几百到上千量级）。

训练不仅优化点击，还显式约束语义相关性。论文定义的偏好次序为：

```text
Clicked & Relevant
  > Unclicked & Relevant
  > Unclicked & Irrelevant
```

总损失为：

\[
\mathcal{L}=\alpha\mathcal{L}_{\text{util}}+\beta\mathcal{L}_{\text{rel}},
\quad \alpha=1,\ \beta=0.5.
\]

- `utility loss` 负责点击效用。
- `relevance loss` 在相同或相近交互优先级内保持与上游意图的语义一致性。
- 为避免 relevance objective 与点击信号冲突，负样本按 clicked > exposed > non-exposed 的交互深度限制。

离线 item-level retrieval 结果：

| 输入模态 | HR@500 | HR@1000 |
|---|---:|---:|
| Text Tag | 0.1503 | 0.2044 |
| SID | 0.1539 | 0.2144 |
| Hybrid | **0.1571** | **0.2168** |

两种模态确实互补，但绝对提升不大：Hybrid 相对 SID 在 HR@500 上增加 0.0032，在 HR@1000 上增加 0.0024。

进一步分析显示：

| 指标 | Text Tag | SID |
|---|---:|---:|
| 平均 Category Breadth | 1.40 | 0.84 |
| 跨用户 Category Overlap | 11.36% | 4.61% |

这支持论文的解释：Tag 更广、更依赖共享世界知识；SID 更集中、更具有个体协同特征。

---

## 5. Latent Intent Reasoning：用 10 个 token 内化数千 token 的 CoT

### 5.1 目标

显式模型生成推理链 \(R\) 和最终输出 \(y\)：

\[
p_\theta(R,y\mid x)=p_\theta(R\mid x)p_\theta(y\mid x,R).
\]

Latent Intent Reasoning 用短 latent sequence \(z=(z_1,\ldots,z_K)\) 替换文本推理链：

\[
p_\theta(z,y\mid x)=p_\theta(z\mid x)p_\theta(y\mid x,z).
\]

显式 trace 按行分成连续、互不重叠的片段，每个 latent token 对应一个片段：

\[
K=\min\left(\left\lceil \frac{N}{C}\right\rceil,K_{\max}\right).
\]

生产配置使用 \(C=20\)、\(K_{\max}=10\)，因此最多只生成 10 个 latent CoT tokens。

### 5.2 Latent token warm-up

新 token 的 embedding 随机初始化，不在预训练语言 embedding 分布内。Warm-up 随机用 latent token 替换一个或多个连续 CoT span，并只对保留的文本 token 做标准 autoregressive loss。目标是先把 latent embedding 拉入与语言 token 兼容的表示空间，获得粗粒度上下文语义。

### 5.3 多粒度重建对齐

为让第 \(j\) 个 latent token 真正编码第 \(j\) 个 reasoning segment，模型执行三类 masked reconstruction：

1. **Single-Segment Reconstruction**：只隐藏一个片段，利用周围显式文本恢复它。
2. **Multi-Segment Reconstruction**：同时隐藏多个片段，要求 latent token 在混合文本上下文中仍包含足够信息。
3. **Full-Trace Reconstruction**：隐藏全部推理片段，只根据 \((x,z,y)\) 恢复完整 CoT。

统一目标为：

\[
\mathcal{L}(\mathcal{J})=-\log p_\theta(R_{\mathcal{J}}\mid c(\mathcal{J}),\mathcal{P}),
\]

其中 \(\mathcal{J}\) 是被 latent token 替换的片段集合，\(\mathcal{P}\) 是重建指令。

该设计的关键不是让 latent token 完全不可解释地承担中间计算，而是让它们能在需要审计时重新解码成自然语言 rationale。

### 5.4 两阶段 Post-training

#### Stage 1：Explicit-to-Implicit CoT Alignment

1. 使用 DeepSeek-V3.2 作为 teacher 产生显式推理链。
2. 先用显式 CoT 做 SFT。
3. 再通过三种 reconstruction task 把推理压入 latent tokens。

训练数据混合：

| 类型 | 占比 |
|---|---:|
| Reasoning Alignment | 21.43% |
| General Reasoning | 69.58% |
| Tag Prediction | 8.43% |

Reasoning Alignment 内部由 single-segment 6.45%、multi-segment 6.56%、full-trace 8.42% 构成。Tag Prediction 只监督最终输出 token，避免直接监督 latent 位置导致表示坍塌。

#### Stage 2：Reinforcement Learning from Ranking Feedback

RecGPT-V2 使用 HitRate 奖励，但存在两点问题：

- 同组 rollout 可能获得相同稀疏奖励，使 GRPO advantage 为零。
- 离线 HitRate 与线上真正决定曝光的生产 ranker 不完全一致。

RecGPT-V3 提出 RLRF，直接从生产排序模型读取候选商品的 CTRScore。对输出 tag 检索出的商品取 Top-100，并计算平均分：

\[
r_{\text{ctr}}(y)=\frac{1}{K}\sum_{k=1}^{K}s_k,\quad K=100.
\]

最终奖励仍保留 RecGPT-V2 的 Constrained Reward Shaping：只有 alignment、diversity 和 length 都超过阈值，CTRScore 才生效：

\[
\mathcal{R}(y)=r_{\text{ctr}}(y)
\mathbb{1}[\text{align}(y)\ge\tau_{\text{align}}]
\mathbb{1}[\text{div}(y)\ge\tau_{\text{div}}]
\mathbb{1}[\text{len}(y)\ge\tau_{\text{len}}].
\]

策略优化使用 GRPO。

### 5.5 质量与效率

| 模型阶段 | HR@30 (Category) | CTR |
|---|---:|---:|
| Qwen3-14B | 0.2276 | - |
| + Native Reasoning | 0.2347 | - |
| Hybrid-modal Foundation Model | 0.3050 | 0.0624 |
| + Explicit CoT SFT | 0.3508 | 0.0638 |
| + Latent Reasoning | 0.3462 | 0.0649 |
| + RL（完整 RecGPT-V3） | **0.3693** | **0.0679** |

显式 CoT 与 latent reasoning 的同硬件、1,000 样本对比：

| 模式 | 平均输出长度 | Input TPM | Output TPM | 总时间 |
|---|---:|---:|---:|---:|
| Explicit CoT | 2,840 | 166K | 531K | 1,020s |
| Latent Reasoning | 122 | 498K | 66.7K | 295s |

Latent reasoning 将总输出长度降低 95.7%，总时间降低 71.1%，即约 **3.46 倍加速**。论文所称“200x token reduction”对应的是约 2,300/2,700 个 reasoning tokens 压缩为至多 10 个 latent tokens；而表中的 2,840 -> 122 包括最终输出，因此是 95.7% reduction。两者统计对象不同。

---

## 6. 端到端示例：一位羽毛球爱好者的训练与推理流程

下面用一个贯穿全链路的简化例子说明各模块如何配合。假设用户 Alice 长期购买羽毛球装备，近期又连续点击进攻型球拍、球线和手胶。示例中的商品名、SID 数值和模型输出均为便于理解而构造，并非论文公开的真实线上样本。

### 6.1 训练前准备：先建立商品 SID 语言

假设商品库中有以下商品：

| 商品 | 文本/图像/属性信息 | 行为共现信号 |
|---|---|---|
| A：进攻型全碳球拍 | 头重、硬杆、适合扣杀 | 常与高磅球线、吸汗手胶一起购买 |
| B：耐打型羽毛球线 | 0.70 mm、耐用、适合高磅 | 常与进攻型球拍一起购买 |
| C：儿童入门球拍 | 轻量、低磅、适合初学者 | 常与儿童训练球共现 |

首先训练商品 tokenizer：

1. CN-CLIP 编码商品标题、图片与属性文本，Q-Former 将多模态信息融合成商品向量。
2. 行为日志中高频共现且多模态相似度合理的商品组成正样本，例如 A 与 B；其他 batch 内商品作为负样本，以 InfoNCE 拉近相关商品、分离无关商品。
3. 两级 RQ-VAE 将商品向量量化成两个 code。为便于说明，假设：

```text
商品 A -> <C_120><C_40120>
商品 B -> <C_120><C_40777>
商品 C -> <C_982><C_33910>
```

商品 A、B 共享第一个 coarse code `<C_120>`，表示它们处在相近的羽毛球装备语义区域；第二个 code 负责更细粒度区分。真实 code 完全由数据学习，不能把 `<C_120>` 理解为人工定义的“羽毛球”类别。

### 6.2 Foundation Model 训练：让 LLM 理解文本与 SID

在 Qwen3-14B 词表中加入 65,536 个 SID tokens 后，训练分两步进行。

#### Step 1：Continual Pre-Training

模型看到 SID 与商品文本的配对语料：

```text
输入/目标文本：
商品 <C_120><C_40120> 的标题是“进攻型全碳羽毛球拍”，
类别是“羽毛球拍”，属性包括“头重、硬杆、适合扣杀”。
```

经过语言建模训练，模型逐渐学会 `<C_120><C_40120>` 不是两个无意义符号，而是对应某类具体商品语义。训练语料同时混入约 10% 的数学、代码、科学和 instruction-following 数据，避免模型只会处理电商 SID 而丢失通用语言能力。

#### Step 2：Instruction Tuning

同一组商品和用户行为被改造成多种任务：

```text
sid2title:
<C_120><C_40120> -> “进攻型全碳羽毛球拍”

title2sid:
“进攻型全碳羽毛球拍” -> <C_120><C_40120>

sid2tag:
<C_120><C_40120> -> “进攻型羽毛球装备”

tag2sid:
“适合扣杀的专业羽毛球拍” -> 候选 SID

sid2sid:
[过去点击的球拍 SID, 球线 SID, 手胶 SID]
-> 下一次可能点击的商品 SID
```

其中 `sid2sid` 让模型直接从行为序列学习 collaborative signal；双向 SID-text 任务则让商品 ID 空间与自然语言空间对齐。

### 6.3 Memory Hub：把 Alice 的长期历史变成可复用状态

Alice 的原始长期行为可能包含数千条记录：数码产品、婴儿用品、家装商品和羽毛球装备混杂在一起。初次 Structured Behavior Compression 后，系统生成若干 memory units：

```text
m_1: Technology
  summary: 长期关注定制键盘和桌面设备
  evidence: [12, 47, 103, ...]

m_2: Badminton
  summary: 偏好全碳、进攻型球拍；经常购买高磅球线和吸汗手胶
  evidence: [549, 553, 558, ...]

m_3: Baby Care
  summary: 近期从婴儿喂养转向辅食与早教用品
  evidence: [721, 735, ...]
```

一段时间后，Alice 新增了以下行为 delta：

```text
搜索“Astrox 进攻拍”
点击“头重型全碳球拍”
点击“高磅耐打球线”
购买“吸汗手胶”
首次浏览“马拉松腰包”
```

到周期性 curation 时，模型只读取旧 memory 与这批 delta：

- 更新 `m_2`，把“偏好进攻型球拍”细化为“关注 Astrox 类头重球拍、高磅球线和维护配件”。
- 保留没有新证据的 `m_1`、`m_3`。
- 若跑步相关行为形成稳定模式，则新建 `m_4: Running`；证据不足时暂不创建。

这一步不是每次推荐请求都执行。论文的生产配置是约每两个月做一次 curation；请求侧读取已经维护好的 memory，并结合尚未被归档的近期行为 delta。

### 6.4 Latent Reasoning 训练：把长推理蒸馏为至多 10 个 token

对于羽毛球 expert，训练输入可以写成：

```text
x = {
  persona: “专业羽毛球装备推荐专家”,
  memory: m_2,
  recent_clicks: [球拍 SID + 短标题, 球线 SID + 短标题, 手胶 SID + 短标题]
}
```

#### Step 1：Teacher 产生显式推理

DeepSeek-V3.2 teacher 先生成长 CoT，例如：

```text
1. 用户持续关注羽毛球，而非一次性偶然点击。
2. 近期行为集中在头重、全碳、进攻型球拍。
3. 高磅球线和吸汗手胶说明用户可能已有较成熟打法。
4. 除球拍外，还应覆盖球线、手胶、护框贴和穿线工具。
5. 标签应具体，避免只输出“体育用品”。
...
最终输出：进攻型全碳球拍、高磅耐打球线、吸汗手胶、球拍维护工具等。
```

模型先通过 SFT 学习 teacher 的显式推理与最终输出。论文中的 teacher trace 平均约为 2,300 tokens；不同效率实验统计约为 2,700 或 2,840 个相关输出 tokens，口径略有不同。

#### Step 2：Warm-up 与多粒度重建

显式 trace 按每 20 个 reasoning steps 一段进行切分，最多映射到 10 个 `<cot>` tokens：

```text
长显式 CoT
-> [R_1, R_2, ..., R_K]
-> [<cot1>, <cot2>, ..., <cotK>], K <= 10
```

训练先随机用 `<cot>` 替换部分文本 span 做 warm-up，再执行：

- 单片段重建：根据上下文与 `<cot2>` 恢复 `R_2`。
- 多片段重建：同时根据 `<cot1>`、`<cot3>` 恢复多个片段。
- 全链重建：只给 `x + <cot1>...<cotK> + 最终输出`，恢复完整显式 CoT。

因此 `<cot>` tokens 被训练成推理链的压缩表示，而不只是任意占位符。服务时默认不解码长 CoT；只有在审计或展示解释时，才运行 reconstruction 将 latent tokens 转回自然语言。

#### Step 3：RLRF 对齐线上排序目标

latent model 对 Alice 采样多组推荐标签。每组标签经过线上同构的召回链路获取商品，再由生产 ranker 打分：

```text
输出 A：“进攻型全碳球拍、高磅球线、吸汗手胶”
  -> Top-100 商品平均 CTRScore = 0.071

输出 B：“体育用品、运动装备、球类用品”
  -> Top-100 商品平均 CTRScore = 0.048
```

若输出 A 同时通过 alignment、diversity 和 length 阈值，其 0.071 奖励进入 GRPO；宽泛的输出 B 即使能召回热门商品，也可能因 alignment 或 diversity 不合格而被门控。模型由此学习生成更符合生产排序器偏好的具体意图。

### 6.5 Hybrid Retriever 训练：兼顾点击与意图一致性

训练样本把 expert 生成的 tag、SID 和候选商品组合起来：

```text
Tag: “进攻型全碳羽毛球拍”
SID: <C_120><C_40120>

正例 1: 用户点击且与意图相关的头重型球拍
正例 2: 用户未点击但与意图相关的同类球拍
负例: 用户未点击且语义无关的儿童入门球拍
```

训练目标要求：

```text
Clicked & Relevant
  > Unclicked & Relevant
  > Unclicked & Irrelevant
```

`utility loss` 学点击倾向，`relevance loss` 防止模型只追逐热门商品而偏离上游意图。Tag embedding 与 SID embedding 拼接投影为统一 intent vector，再通过 Target-Attention 给候选商品打分。

### 6.6 一次线上推理请求

当 Alice 打开淘宝“猜你喜欢”时，完整推理流程如下。

#### Step 1：读取用户状态

系统读取已经持久化的 memory units，并拼接 curation 后新增的最近行为：

```text
Planner input = [m_1, m_2, m_3, ...] + recent behavior delta
```

它不再重新编码 Alice 的全部生命周期行为。

#### Step 2：Planner 分配 persona

Global Planner 判断当前最强信号之一是羽毛球装备需求，构造类似“专业羽毛球装备推荐专家”的 persona，并把该 persona、相关 memory 和近期点击交给 expert model。

#### Step 3：Expert 执行 latent reasoning

expert 不生成数千 token 的显式分析，而只生成至多 10 个 latent tokens：

```text
<cot1><cot2>...<cot10>
```

随后输出双模态意图。示意结果为：

```text
Tag 1: “进攻型全碳羽毛球拍”
SID 1: <C_120><C_40120>

Tag 2: “高磅耐打羽毛球线”
SID 2: <C_120><C_40777>

Tag 3: “吸汗防滑羽毛球手胶”
SID 3: <C_120><C_41502>
```

Tag 扩展开放世界语义和类别覆盖，SID 把意图锚定到包含协同信号的具体商品区域。

#### Step 4：Hybrid Retrieval 召回商品

Expert 一次给出了 3 组 Tag/SID 意图对，Retriever 对每一组分别执行第 4 节的公式 \(\mathbf{q}_i = W_{\text{proj}}[\mathbf{e}^{(i)}_t \| \mathbf{e}^{(i)}_d] + \mathbf{b}\)：

- 用 (Tag 1, SID 1) 查表得到 \(\mathbf{e}^{(1)}_t\)（“进攻型全碳羽毛球拍”的文本 embedding）和 \(\mathbf{e}^{(1)}_d\)（`<C_120><C_40120>` 的 SID embedding），拼接投影得到意图向量 \(\mathbf{q}_1\)；(Tag 2, SID 2)、(Tag 3, SID 3) 同理分别得到 \(\mathbf{q}_2\)、\(\mathbf{q}_3\)。
- 每个 \(\mathbf{q}_i\) 作为 Target-Attention 的 query，对候选商品库里每件商品的表示打分：\(\mathbf{q}_1\) 倾向于给头重型全碳进攻拍类商品打高分，\(\mathbf{q}_2\) 倾向于给高磅球线类商品打高分，\(\mathbf{q}_3\) 倾向于给手胶及维护配件打高分。
- 三路打分各自取分数最高的一批商品，再合并、去重成这次请求的 Hybrid Retrieval 候选集合——Tag 通道贡献更广的类目覆盖（例如连带召回维护工具），SID 通道贡献与 Alice 历史共现、语义更接近的具体商品簇。
- 这个候选集合本身不是最终展示顺序，只是交给 Step 5 的 Production Ranker 排序前的候选池。

#### Step 5：Production Ranker 排序并展示

候选进入生产 ranker，与其他广告、直播和内容一起完成最终排序。排在前面的可能是：

```text
1. 头重型全碳进攻拍
2. 适合高磅穿线的耐打球线
3. 吸汗手胶套装
4. 护框贴与球拍维护工具
```

用户的点击、加购和购买继续写入行为日志：短期内作为 recent delta 参与请求；到下次 curation 时再用于更新长期 memory。于是系统形成闭环：

```text
行为 -> 记忆 -> latent intent -> Tag + SID -> 召回 -> 排序 -> 新行为
  ^                                                        |
  +--------------------------------------------------------+
```

### 6.7 训练与推理的关键区别

| 环节 | 训练时 | 线上推理时 |
|---|---|---|
| 商品 SID | 训练多模态 encoder 与 RQ-VAE，并给商品分配 SID | 直接查询已分配的 SID；新商品需走 SID 生成流程 |
| Foundation Model | CPT + 多任务 instruction tuning | 读取文本与 SID，生成双模态意图 |
| 显式 CoT | Teacher 生成，用于 SFT 和 latent reconstruction | 默认不生成，只在需要解释时重建 |
| Latent token | 通过 warm-up 和三种 reconstruction 学习 | 只生成至多 10 个 token 完成低延迟推理 |
| Production Ranker | 提供 Top-100 CTRScore 作为 RLRF 奖励 | 对实际召回候选执行最终排序 |
| Memory Hub | 初次压缩历史；周期性增量 curation | 每次请求读取 memory + recent delta |
| Hybrid Retriever | 用 utility + relevance 联合目标训练 | 联合 Tag 与 SID 召回候选商品 |

---

## 7. 线上 A/B 与系统成本

实验组和 RecGPT-V2 对照组各使用淘宝总流量的 1%。论文分别报告 Item Scenario 与 Feed Scenario：

| 场景 | IPV | CTR | PV | DAU | TC | GMV |
|---|---:|---:|---:|---:|---:|---:|
| Item | +3.08% | +0.98% | +2.02% | - | +3.10% | +7.51% |
| Feed | +1.28% | +1.00% | +0.83% | +0.56% | +1.97% | +3.97% |

业务收益不是只体现在点击：Item 场景下 GMV +7.51%、TC +3.10%，明显高于 CTR +0.98%，说明改进可能更偏向高购买意图商品，而不是单纯增加点击。

成本方面：

- SID 输入/输出与 latent reasoning 使 expert model 相对 V2 expert 增加约 15% 计算开销。
- 但当前部署中 Global Planner 的成本约为单个 expert 的 20 倍。
- Memory Hub 将 Planner 侧成本降低 55.8%。
- 按系统组件加权后，RecGPT-V3 的端到端 serving resource consumption 相对 V2 降低 **52.4%**。
- 论文首页还称 V3 只使用 V1 的 19% compute，但正文没有给出 V1 -> V2 -> V3 的完整成本拆解。
