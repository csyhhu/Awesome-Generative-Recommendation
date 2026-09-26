# LLM-Based User Personas for Recommendations at Scale

- **论文**：[LLM-Based User Personas for Recommendations at Scale](https://arxiv.org/abs/2606.12198)
- **arXiv ID**：2606.12198
- **作者**：Haoting Wang、Haokai Lu、Zheyun Feng、Jenny Huang、Yifat Amir、Gregory Hinkson、Ben Most、Zelong Zhao、Kelly (Yixin) Cui、Rein Zhang、Fabio Soldo、Yu Xia、Nihar Bhupalam、Minmin Chen、Konstantina Christakopoulou、Lichan Hong、Ed Chi
- **机构**：Google DeepMind、Google、GNucleus AI
- **关键词**：用户画像、兴趣总结、兴趣探索、知识蒸馏、异步推理、工业推荐系统

---

## 1. 一句话总结

本文在大型视频推荐系统中，把用户近期观看历史先聚成若干主题簇，再让 LLM 为每个簇同时生成一个可解释的“已知兴趣”和三个相关但新颖的“探索兴趣”；系统用 Gemini 1.5 Pro 生成蒸馏数据、用量化 Gemini Nano 在线异步刷新画像，并通过受文本兴趣约束的近邻检索把自由文本重新落到视频候选空间。

它最重要的贡献不是提出新的推荐主干模型，而是证明自然语言用户画像可以成为生产召回源：在不阻塞当前请求的异步架构下，论文报告观看时长提升 `+0.04%`、活跃用户数提升 `+0.03%`，同时增加用户接触并持续参与的新主题。

---

## 2. 论文要解决什么问题

传统工业推荐系统通常用 item ID、cluster ID 或 dense embedding 表示用户。它们便于高效召回和排序，但存在三个限制：

1. **语义不透明**：ID 很难表达"为什么用户喜欢这些内容"，也不能直接用于可展示给用户的兴趣标签。
2. **容易延续反馈回路**：从已有点击和观看中归纳相似内容，往往只强化已经曝光过的主题，难以主动提出合理的新兴趣。
3. **LLM 难以进入在线链路**：同步调用大模型的延迟和成本无法支撑十亿级用户；纯离线画像又难以及时反映兴趣变化。

论文采用分层规划视角，将推荐拆为两层：

```text
高层语言策略：观看历史 → 自然语言兴趣画像
                         ↓
低层 item 策略：文本兴趣 → 相关视频候选 → 下游排序
```

高层策略负责语义理解与探索，低层策略复用成熟的工业检索和排序基础设施。这样既保留 LLM 的开放世界知识，也避免让 LLM 直接生成具体视频 ID。

---

## 3. 整体方法

完整链路如下：

```text
用户观看历史
    ↓ 过滤低质量、负反馈、不安全内容
合格观看序列
    ↓ 聚类并从每簇采样代表视频
结构化的视频标题 Prompt
    ↓ 蒸馏后的量化 Gemini Nano
每个簇：
  - 1 个 Summarized Interest
  - 3 个 Exploration Interests
  - 对应 reasoning
    ↓ 安全分类器
自然语言用户画像缓存
    ↓ 每次请求随机采样 1 个总结兴趣 + 1 个探索兴趣
文本兴趣约束的 ANN 召回
    ↓
下游排序模型
```

画像同时承担两个目标：

- **Exploitation / 总结兴趣**：把已有观看行为压缩为简洁、具体、可读的主题。
- **Exploration / 探索兴趣**：基于总结兴趣和 LLM 世界知识，提出语义相关但带有新视角的主题，用于打破推荐反馈回路。

统一 serving prompt 要求模型对每个输入簇输出一个总结兴趣、理由和三个探索兴趣。把两个任务放进一次推理，是为了减少在线模型调用成本。

| 维度 | 本文对应描述 |
|---|---|
| **Input** | 用户近期和长期观看历史。预处理先过滤低质量、负反馈、不安全或敏感视频，再对历史聚类，并从每个簇采样少量代表视频的**标题**组成结构化 Prompt。Teacher 数据使用基于 salient terms 的语义聚类；正式线上 serving 为满足规模与吞吐约束，改用视听 embedding 聚类。 |
| **Training Methods** | 采用 teacher-student knowledge distillation。Gemini 1.5 Pro 作为 teacher，通过多次 LLM 调用和 CoT 分别生成“总结兴趣 + reasoning”与“探索兴趣 + reasoning”；仅保留数量和格式正确的响应，并按 `80% / 20%` 划分训练与评估集。随后 fine-tune Gemini Flash 和 Gemini Nano，使 student 用一个统一 Prompt、一次生成同时完成总结与探索。线上 Nano 进一步量化。论文未披露具体 token-level loss、optimizer、batch size 或硬件配置。 |
| **Output** | 每个用户得到一份可缓存的、开放词表自然语言 persona。对每个观看簇输出 `1` 个 **Summarized Interest**、对应 reasoning，以及 `3` 个相关但新颖的 **Exploration Interests**。输出既不是 dense user embedding，也不是与 item codebook 对齐的 SID；它是可直接阅读的文本型用户表示。 |
| **How to use** | 在线请求先读取 persona cache。每次请求从画像中随机采样 `1` 个总结兴趣和 `1` 个探索兴趣，用它们约束 sequential Transformer 的 nearest-neighbor candidate retrieval，候选随后进入既有 ranker。画像缺失或过期时只异步触发重算，当前请求不等待；不安全的新画像回退到上一份安全画像。文本 persona 也具备用户可见解释的潜力，但本文的生产实验主要把它用作 retrieval source。 |
| **Intermediate Performance Validation** | 用户表示选择阶段以用户点击过的文本主题为 reference，使用 **BLEURT** 比较标题/描述/salient terms、顺序/语义/视听聚类、模型规模和 few-shot 设置。蒸馏阶段使用 **IFR** 验证格式与任务完成度、用 **BLEURT** 验证 teacher imitation、用 LLM side-by-side **Creativity Score** 验证探索新颖性。产品上线前还通过数千人调查验证标签准确性和继续观看意愿。这里没有像 TokenMinds 那样用 decoder SID Recall 直接验证 item-space prediction。 |
| **Resource Consumption** | **Training Data**：数万名合格用户的观看历史，精确样本数、平均簇数和 Prompt token 数未披露；teacher 为多调用生成，student 为单调用蒸馏。**Training Accelerators**：未披露。**Inference Compute**：量化 Gemini Nano 在后台异步运行，当前推荐请求只承担缓存读取；单用户生成延迟、QPS、刷新周期、模型参数量和加速器数量均未披露。**Inference Storage**：每个合格用户缓存一份文本 persona，但平均字节数、覆盖用户数和副本策略未披露，因此只能表示为 `N_eligible_users × average_persona_bytes`，不能像 TokenMinds/LLaTTE 一样计算总 TB。 |

### 3.2 放入该分类体系后的核心判断

这篇论文最接近 Roadmap 中的 **Discrete User Representation: Using LLMs to generate text-based user tokens**，但"discrete"需要谨慎理解：自然语言最终由离散 vocabulary token 组成，却没有稳定的固定长度、固定码本或 user-token ID。它的优势是开放世界推理、可解释性和可直接产生探索主题；代价是表示长度不固定、存储与解析协议不如 SID 稳定，而且必须额外通过文本检索将 persona grounding 到 item 空间。

从 Interest Cache 视角看，它缓存的是**高层语义策略**，而不是可以直接插入 ranker 的 dense feature 或 SID embedding。最自然的用法是作为独立 retrieval source，或与 TokenMinds/LLaTTE 的低层 user representation 组合：persona 提出“探索什么”，dense/SID 表示和 ranker 决定“具体推荐什么、是否值得曝光”。

---

## 4. 用户历史如何表示

### 4.1 为什么不能直接使用 item ID

Item ID 对推荐模型有意义，但对通用 LLM 没有天然语义。论文因此把用户行为转为文本，并系统比较了视频字段、输入组织方式、模型规模和 few-shot 数量。

### 4.2 视频字段

作者比较了标题、描述和 salient terms：

| 视频表示 | BLEURT |
|---|---:|
| Salient terms | 0.2376 |
| 视频描述 | 0.2552 |
| **视频标题** | **0.2699** |

标题效果最好。作者认为标题语义密度高；描述容易混入推广、版权和背景音乐等噪声；salient terms 又往往过于宽泛，无法提供细粒度主题。

### 4.3 输入结构

| 输入结构 | BLEURT |
|---|---:|
| 按时间顺序排列 | 0.2458 |
| **按语义相似度聚类** | **0.2706** |
| 按视听 embedding 相似度聚类 | 0.2483 |

简单拼接观看标题容易让 LLM 输出宽泛标签；先把历史分成语义一致的簇，可以把“孟加拉电视剧”细化到具体剧集主题。语义聚类相对顺序输入的 BLEURT 提升约 `10.1%`。

论文讨论了两种聚类：

- **视听 embedding 聚类**：根据音频和视觉 embedding 做在线层次聚类，稳定、可扩展，但相似画面和声音不一定代表相同抽象主题。
- **语义聚类**：用标题、描述、上传者等元数据训练出的 unigram/bigram salient terms 表示视频，再按 cosine similarity 在线构建个性化兴趣簇，解释性和离线质量更高。

值得注意的是，训练数据和离线最优方案采用语义聚类，但正式线上实验为了吞吐和可扩展性改用视听 embedding 聚类。这是论文中一个重要的质量与系统成本折中。

### 4.4 模型规模和 few-shot

| 模型 | BLEURT |
|---|---:|
| Gemini Flash | 0.2454 |
| **Gemini Pro** | **0.2613** |
| Gemini Ultra | 0.2606 |

Pro 明显优于 Flash，但 Ultra 没有继续提升，说明该任务在 Pro 规模附近出现能力饱和。这一结果促使作者用大模型生成监督数据，再蒸馏到小模型。

| Prompt 示例数 | BLEURT |
|---|---:|
| 0-shot | 0.2201 |
| 1-shot | 0.2556 |
| **2-shot** | **0.2679** |

因此，离线研究最终选择“语义聚类 + 视频标题 + two-shot prompt”作为高质量数据生成配置。

---

## 5. 知识蒸馏

### 5.1 Teacher 数据构造

Teacher 使用 Gemini 1.5 Pro。数据来自数万名同意训练数据使用、近期拥有足够高满意度观看行为的用户，并先过滤不安全或敏感视频。

Teacher 并不直接执行线上那种单次统一 Prompt，而是采用多步工作流：

```text
观看历史
    ↓ salient-term 语义聚类
每簇代表性标题
    ↓ 第一次 LLM 调用
总结兴趣 + reasoning
    ↓ 第二次 LLM 调用
探索兴趣 + reasoning
    ↓ 格式和数量校验
蒸馏样本
```

只保留兴趣数量正确且严格满足输出格式的响应，然后按 `80% / 20%` 划分训练集和评估集。这种做法让昂贵的 teacher 在离线阶段使用多次调用和 CoT 提高标签质量，而 student 学习在一次调用中完成相同任务。

### 5.2 Student 模型与指标

作者微调 Gemini Flash 和 Gemini Nano，并用三个指标选择 checkpoint：

- **Instruction Following Rate（IFR）**：是否同时完成总结与探索、格式正确、输出兴趣数量完全匹配。
- **BLEURT**：student 总结兴趣与 teacher 总结兴趣之间的语义相似度。
- **Creativity**：LLM autorater 比较 student 与 teacher 的“总结兴趣—探索兴趣”对，判断哪一方更新颖。

| Epoch | Nano IFR | Flash IFR | Nano BLEURT | Flash BLEURT | Nano Creativity | Flash Creativity |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.07% | 1.82% | N/A | 0.286 | N/A | -0.588 |
| 4.37 | - | 99.54% | - | 0.334 | - | 0.008 |
| 10.92 | 98.97% | 99.75% | 0.321 | 0.338 | -0.012 | 0.016 |
| 26.20 | **99.08%** | **99.68%** | **0.328** | **0.345** | -0.018 | **0.023** |

主要结论：

- 格式学习很快，未蒸馏的小模型几乎无法可靠完成统一任务。
- 总结质量随训练持续提升，Nano 最终接近较大模型的初始水平。
- Flash 能学到更好的探索能力；Nano 的 creativity 始终低于 teacher，说明新颖兴趣生成对容量更敏感。
- 线上最终仍选择 epoch 26.20 的 Nano，因为其总结质量足够接近 Flash，而成本和延迟更低。

### 5.3 Teacher、Student 与生产推理的准确分工

首先需要澄清术语：这里应写作 **text prompt** 和 **textual persona / natural-language persona**，不是 texture prompt 或 texture persona。

| 阶段 | 使用模型 | 作用 |
|---|---|---|
| 蒸馏数据生产 | Gemini 1.5 Pro teacher | 离线使用多次调用和 CoT，生成高质量的总结兴趣、探索兴趣与 reasoning，作为监督标签。 |
| Student 训练 | Gemini Flash / Gemini Nano | 在 teacher 数据上 fine-tune，学习通过一个统一 Prompt、一次调用完成总结与探索。 |
| 生产画像生成 | 量化的 Gemini Nano | 用户访问触发缺失或过期检查后，在后台异步生成新 persona 并写入缓存。正式线上实验使用 epoch 26.20 checkpoint。 |
| 当前推荐请求 | 不同步调用 LLM | 读取已有 persona；如果画像过期，当前请求仍不等待后台重算。 |

因此，“蒸馏小模型用于大规模生产文本”这一理解是对的，但论文描述的生产模式不是“离线批处理一次 + 请求内在线快速生成”两套路径，而是**线上访问触发、后台异步计算、结果缓存供后续请求使用**。Teacher 只负责离线造训练标签；实际生产文本由量化 Nano 生成。论文没有说明 Nano 是否还承担独立的周期性全量离线 batch inference。

### 5.4 蒸馏样本规模

论文明确披露的信息只有：

- 随机抽取**数万名**合格且同意数据用于训练的用户。
- 每个用户需要有足够多的近期高满意度观看事件，并先删除不安全、敏感及低质量内容。
- 历史经过 semantic clustering，少量观看的簇被删除，再用每簇若干视频标题构造输入。
- Teacher 响应只有在兴趣数量与格式完全正确时才保留。
- 通过质量检查的数据按 `80% / 20%` 划分为训练和评估集。

论文没有披露过滤后的精确样本数，也没有说明一个用户是否只产生一个时间截面样本。因此只能判断训练集是**万级到数万级 user-persona examples**，不能直接把“数万用户”等同于最终训练 example 数。

#### 5.4.1 一个蒸馏训练样本

下面是严格按照论文 Prompt 结构构造的**示意样本**，不是论文公开的真实用户日志。假设经过安全过滤、semantic clustering 和小簇删除后，某用户保留三个观看簇：

```text
[Group 0]
- How to Dial In Espresso at Home
- Beginner Latte Art Tutorial
- Choosing a Burr Grinder for Espresso

[Group 1]
- Three Days in Kyoto: Temple and Food Guide
- Osaka Street Food Tour
- How to Use Japan's Railway Pass

[Group 2]
- Python AsyncIO Explained
- Building a FastAPI Service
- Profiling Slow Python Applications
```

Teacher 数据生成不是直接调用一次统一 Prompt，而是使用 Gemini 1.5 Pro 做多步生成。概念上可以表示为：

```text
Call A: 对每个 group 生成 Summarized Interest + reasoning

Group 0:
  Home espresso brewing
  Reasoning: videos cover espresso dialing, grinders and latte preparation.

Group 1:
  Independent travel in Japan
  Reasoning: videos focus on itinerary planning, transport and local food.

Group 2:
  Practical Python backend engineering
  Reasoning: videos cover async programming, APIs and performance debugging.

Call B: 基于每个 summarized interest 生成、检查和完善探索兴趣

Home espresso brewing
  -> Coffee bean roasting basics
  -> Water chemistry for coffee
  -> Cafe workflow design

Independent travel in Japan
  -> Japanese regional festivals
  -> Rural rail journeys
  -> Traditional lodging etiquette

Practical Python backend engineering
  -> Distributed tracing
  -> Event-driven system design
  -> Rust extensions for Python
```

Teacher 响应通过数量和格式检查后，被整理为 student 的一条 `(input, target)` SFT 样本。Student input 是带有统一任务说明和 few-shot examples 的 grouped titles；target 使用附录中的 serving 格式：

```text
[Group 0]
Task 1:
**Home espresso brewing**: The videos focus on dialing espresso,
grinder selection, and preparing milk drinks at home.
Task 2:
&&Coffee bean roasting basics&&,
&&Water chemistry for coffee&&,
&&Cafe workflow design&&

[Group 1]
Task 1:
**Independent travel in Japan**: The videos cover transport,
itineraries, local food, and planning trips around Kyoto and Osaka.
Task 2:
&&Japanese regional festivals&&,
&&Rural rail journeys&&,
&&Traditional lodging etiquette&&

[Group 2]
Task 1:
**Practical Python backend engineering**: The videos cover async code,
API development, and application profiling.
Task 2:
&&Distributed tracing&&,
&&Event-driven system design&&,
&&Rust extensions for Python&&
```

训练时对 target tokens 做 teacher-forced next-token prediction，使 Flash/Nano 学会在一次前向生成中复现 Teacher 多步流程的最终结构。论文没有明确写出 token-level loss，但这种 SFT 通常使用 autoregressive cross-entropy；因此这里应视为基于常规 SFT 的合理解释，而不是论文披露的特殊 loss。

可以用以下变量估算 token 量：

\[
L_{\mathrm{input}}
\approx P+G\times M\times L_{\mathrm{title}},
\]

\[
L_{\mathrm{output}}
\approx G\times L_{\mathrm{persona/group}},
\qquad
T_{\mathrm{train}}
\approx 0.8N\times E\times
(L_{\mathrm{input}}+L_{\mathrm{output}}).
\]

其中 `N` 是质量过滤后的样本数，`G` 是每个用户保留的兴趣簇数，`M` 是每簇标题数，`P` 是指令与 few-shot 开销，`E` 是训练 epochs。论文没有给出这些变量的实际值。

为了理解数量级，可以构造一个**非论文数据、仅用于容量规划**的中位场景：

| 假设变量 | 示例值 |
|---|---:|
| 质量过滤后样本 `N` | 50K |
| 训练样本 | 40K |
| 每用户兴趣簇 `G` | 4 |
| 每簇标题 `M` | 5 |
| 每标题平均 token | 10 |
| 指令与 few-shot `P` | 300-600 tokens |
| 每簇输出 | 50-100 tokens |

在这个假设下，每个训练样本约有 `500-800` input tokens 和 `200-400` output tokens，即总计约 `700-1,200` tokens。40K 训练样本对应每 epoch 约 `28M-48M` token presentations；训练到论文报告的 epoch 26.20，大约是 `0.73B-1.26B` token presentations。这个估算不包含 teacher 多步生成成本，也不代表去重后的 unique tokens。

Teacher 数据生产成本更难估计。正文只说使用 multiple LLM calls 和 CoT 分别生成、完善总结与探索结果，没有披露每个用户或每个簇的调用次数。能够确定的是，teacher 阶段每份标签需要多次 Gemini 1.5 Pro inference，而 student serving 将其压成一次统一生成。

### 5.5 Teacher 与 Student 模型尺寸

| 模型 | 公开信息 | 本文能否确定参数量 |
|---|---|---|
| Gemini 1.5 Pro teacher | Gemini 1.5 官方报告称其为 sparse MoE Transformer，但没有公开 total parameters 或 active parameters。 | **不能** |
| Gemini Flash student | Gemini 1.5 官方报告称 Flash 是从更大的 1.5 Pro online-distilled 的低延迟 Transformer decoder，但原始 Flash 参数量未公开。 | **不能** |
| Gemini Flash-8B | 官方后续报告明确它是 8B、且小于原始 Flash 的独立衍生型号。 | 本文写的是 Flash，不是 Flash-8B，**不能套用 8B** |
| Gemini Nano student | 本文只写 Gemini Nano，没有版本或参数量。Gemini 1.0 官方报告公开 Nano-1 为 `1.8B`、Nano-2 为 `3.25B`，均面向端侧并使用 4-bit 部署。 | **不能确认本文使用哪一个，也不能确认是否就是公开 Nano-1/2** |

如果仅为了估算，并假设本文 Nano 与公开 Gemini 1.0 Nano 一致，则纯权重的理论下限为：

| 假设模型 | BF16 | INT8 | INT4 |
|---|---:|---:|---:|
| Nano-1，1.8B | 3.6 GB | 1.8 GB | 0.9 GB |
| Nano-2，3.25B | 6.5 GB | 3.25 GB | 1.63 GB |

真实 serving memory 还需要 embeddings、量化 scale、KV cache、activation、runtime buffer 和 batch 调度空间，明显高于纯权重。本文只说 student 做过 quantization，没有披露 bit-width，因此 `0.9/1.63 GB` 不能作为论文的实际显存数字。

从训练机制上看，还存在两层能力迁移：公开的 Gemini Flash/Nano 基座本身已经通过更大 Gemini 模型蒸馏；本文又使用 Gemini 1.5 Pro 生成 recommendation-specific labels，对这些小模型做任务级 fine-tuning。本文表格中的 26.20 epochs 指后一个任务级训练过程，不是从头训练 Gemini 基座。

参考公开模型报告：[Gemini 1.0](https://arxiv.org/abs/2312.11805)、[Gemini 1.5](https://arxiv.org/abs/2403.05530)。

---

## 6. 十亿用户规模的异步服务

### 6.1 缓存刷新

用户访问平台时，系统先读取画像数据库：

```text
画像存在且新鲜 → 直接供当前请求使用
画像缺失或过期 → 触发后台生成，但当前请求不等待
                 → 拉取历史、调用 student、做安全检查
                 → 将新画像写回数据库，供后续访问使用
```

这种架构的关键是把 LLM 推理从当前请求的 latency critical path 中移除，同时让生成频率不再与请求量一一绑定。更新频率和观看历史窗口可配置，用于平衡 freshness 与成本；student 还进一步做了量化。

严格来说，这里的“real-time”指**由线上访问触发并服务于后续请求的异步近实时刷新**，不是在同一次请求中基于最新行为同步生成画像。突发兴趣在画像刷新前仍会缺失。

### 6.2 输入预处理

只有拥有足够合格观看事件的用户才生成画像。系统使用较长历史覆盖长期和近期兴趣，排除用户明确报告为负面体验的内容，再把历史聚成少量簇，并从每簇采样少量视频标题控制 Prompt 长度。

### 6.3 安全回退

LLM 输出经过安全分类器。新画像如果包含不安全或敏感兴趣，系统不会直接暴露它，而是回退到上一份已经通过审核的画像。论文没有报告安全拒绝率、误杀率或旧画像回退比例。

### 6.4 文本兴趣如何召回视频

论文探索两种接口，两者不能混为一谈：

1. **Two-Tower**：persona 文本本身经过 query tower 得到向量，item tower 产生视频向量，再按两者的 cosine similarity 检索。这里文本 persona 是直接的 ANN query。
2. **Sequential Transformer + Restricted Nearest Neighbor Search**：正式线上实验采用该方案。Sequential Transformer 根据用户行为序列产生个性化 user embedding，persona 不替代这个 user embedding，而是限制允许参与近邻搜索的视频集合。

第二种方案可以形式化为：

\[
\mathbf{u}=f_{\mathrm{seq}}(H_u),
\qquad
\mathcal C(t)=\{i\in\mathcal I\mid i\text{ 与文本兴趣 }t\text{ 在语义上相关}\},
\]

\[
\operatorname{Retrieve}(u,t)
=
\operatorname{TopK}_{i\in\mathcal C(t)}
\operatorname{sim}(\mathbf u,\mathbf v_i).
\]

其中，`H_u` 是用户行为序列，`u` 是 sequential Transformer 输出的用户向量，`v_i` 是视频向量，`t` 是选中的自然语言兴趣。没有 persona 约束时，近邻搜索在全量视频集合 `I` 上找与 `u` 最相似的 item；加入 persona 后，只在语义相关子集 `C(t)` 中进行 Top-K 搜索。因而它结合了两种信号：

- **文本 persona 决定“允许从哪个主题区域召回”**，承担高层语义规划与探索。
- **Sequential Transformer 的 user-item score 决定“该主题区域内哪些视频最适合这个用户”**，保留生产模型的个性化能力。

线上每次请求随机选择一个 summarized interest `t_s` 和一个 exploration interest `t_e`，可等价理解为分别形成两个受限候选空间 `C(t_s)` 与 `C(t_e)`，执行个性化近邻检索，再把得到的候选送入下游 ranker。随机选择而不是固定取最高分兴趣，是为了在两次 persona 刷新之间均匀覆盖多个主题，并重新激活较长期的兴趣。论文没有说明两路候选的 quota、去重和 merge 规则。

### 6.5 “语义约束”具体做到哪一步

当前论文只明确说：搜索空间被限制为“与 LLM 文本兴趣语义相关的视频”。它**没有披露**以下关键实现：

- 如何把自由文本兴趣构造成 `C(t)`：文本 embedding 相似度、搜索索引、主题分类器还是 cluster 映射。
- 使用哪个 text/video encoder、相似度阈值以及每个兴趣对应多少视频。
- 约束是在 ANN 搜索前通过 restrict token / posting list 完成，还是先扩大召回再做 post-filter。
- reasoning 是否参与语义匹配。正文只说 summarized/exploration textual interests 用于 retrieval，没有证据表明 reasoning 被用作 query。

其引用的前序工作 *LLMs for User Interest Exploration in Large-scale Recommendation Systems* 使用的是更明确的固定词表方案：LLM 被 fine-tune 为生成一个预定义 cluster description，将文本精确映射回 cluster ID，再把 sequential recommender 的 item-level softmax 或 ANN 搜索限制到这些 cluster ID。本文强调输出是 free-form natural language，不再局限于几百个预定义 cluster，因此不能直接假定仍采用完全相同的 exact-match 映射。

更稳妥的理解是：本文公开了**两阶段约束逻辑**，即先由文本兴趣确定语义候选域，再由 sequence model 在域内个性化 Top-K；但自由文本到候选域 `C(t)` 的工程实现被当作已有 retrieval infrastructure，没有在论文中展开。

### 6.6 一个完整的生产推理样本

继续使用上面的示意用户。假设其 persona cache 已经过期，第一次访问发生以下流程：

```text
Request R1 arrives
  → Persona DB: stale
  → 异步触发 Refresh Job
  → R1 不等待 Gemini Nano
  → R1 是否继续使用 stale persona，论文未披露
```

后台任务从用户数据库读取新历史。正式线上实验使用可扩展的视听 embedding clustering，而不是 teacher 数据阶段的 semantic clustering；之后仍从每簇采样少量标题，构造与训练相同的统一 Prompt：

```text
Refresh Job
  → 拉取合格观看历史
  → 删除负反馈和不安全视频
  → embedding-based clustering
  → 每簇采样代表标题
  → 量化 Gemini Nano 单次生成
  → safety classifier
  → 写入 Persona DB
```

解析后的缓存可以概念化为：

```json
{
  "user_id": "example_user",
  "generated_at": "T1",
  "interests": [
    {
      "summarized": "Home espresso brewing",
      "reasoning": "The videos focus on dialing espresso, grinders, and milk drinks.",
      "exploration": [
        "Coffee bean roasting basics",
        "Water chemistry for coffee",
        "Cafe workflow design"
      ]
    },
    {
      "summarized": "Independent travel in Japan",
      "reasoning": "The videos cover transport, itineraries, and local food.",
      "exploration": [
        "Japanese regional festivals",
        "Rural rail journeys",
        "Traditional lodging etiquette"
      ]
    },
    {
      "summarized": "Practical Python backend engineering",
      "reasoning": "The videos cover async code, APIs, and profiling.",
      "exploration": [
        "Distributed tracing",
        "Event-driven system design",
        "Rust extensions for Python"
      ]
    }
  ],
  "safety_status": "passed"
}
```

在后续请求 `R2` 中，系统命中新画像，并随机选择一个 summarized interest 和一个 exploration interest：

```text
Selected summarized interest:
  t_s = "Home espresso brewing"

Selected exploration interest:
  t_e = "Water chemistry for coffee"
```

语义 grounding 层分别构造相关候选域。由于论文没有披露这一步的实现，下面只用集合表示，不假设它一定采用 persona embedding Top-M：

```text
C(t_s) = {espresso dialing video, grinder review, latte tutorial, ...}
C(t_e) = {coffee mineral composition, brewing water recipe, pH guide, ...}
```

与此同时，生产 Sequential Transformer 根据用户行为历史生成 user embedding：

\[
\mathbf u=f_{\mathrm{seq}}(H_u).
\]

受限近邻检索只在两个语义集合内部按 user-item similarity 选择候选：

\[
R_s=\operatorname{TopK}_{i\in\mathcal C(t_s)}
\operatorname{sim}(\mathbf u,\mathbf v_i),
\qquad
R_e=\operatorname{TopK}_{i\in\mathcal C(t_e)}
\operatorname{sim}(\mathbf u,\mathbf v_i).
\]

```text
R_s + R_e
  → 合并/去重（具体规则未披露）
  → 下游 Ranker
  → 与其他召回源候选一起排序和展示
```

该例子体现了三种不同职责：Nano 负责低频更新 persona，semantic grounding 负责确定主题候选域，Sequential Transformer 和下游 ranker 负责请求级个性化。论文没有说明两路 quota、候选合并规则或自由文本到 `C(t)` 的具体映射。

### 6.7 Persona 的缓存空间

论文没有披露每个用户实际保留多少个簇、每个 label/reasoning 的平均长度、序列化格式或是否只保存解析后的兴趣标签，因此无法给出类似 TokenMinds `5,888 bytes/user` 的精确结果。

若每个用户有 `G` 个兴趣簇，缓存完整模型响应的 raw payload 可以写成：

\[
B_{\mathrm{persona}}
\approx G\times
(B_{\mathrm{summary}}+B_{\mathrm{reasoning}}+3B_{\mathrm{exploration}})
+B_{\mathrm{metadata}}.
\]

这里还有一个原文口径差异：训练流程图称 exploration 输出包含 reasoning，但附录中的统一 serving prompt 只要求三个 exploration labels，没有要求 exploration reasoning。生产数据库究竟保存完整原始响应，还是只保存解析后的 summary/exploration labels，也没有说明。

一个用于容量规划的合理区间是：

- **只存解析后的 labels**：假设 `G=4`，共 `4` 个 summarized labels 和 `12` 个 exploration labels，加结构化字段后约 `0.5-1 KB/user`。
- **保存 labels + summarized reasoning + 生成元数据**：若完整响应约 `200-400` tokens，按英文文本和 JSON 开销估算约 `1-3 KB/user`。
- **保存 token IDs 或做压缩**：可能低于上述 UTF-8/JSON 估算，但论文没有说明。

对应的 raw payload 数量级为：

| 每用户 payload | 100M 合格用户 | 1B 合格用户 | 1B 用户、3 副本 |
|---:|---:|---:|---:|
| 0.5 KB | 50 GB | 0.5 TB | 1.5 TB |
| 1 KB | 100 GB | 1 TB | 3 TB |
| 3 KB | 300 GB | 3 TB | 9 TB |

这些是十进制容量且只计算 value payload，不包括 user key、TTL、版本号、索引、KV 元数据、复制协议和容灾开销。实际覆盖人数还是 `N_eligible_users`，不一定等于平台全部十亿用户。

与固定维度 dense embedding 或 SID 相比，文本 persona 的 raw payload 可能更小，但它是变长对象，长度分布、解析版本和多语言字符都会增加运营复杂度。更重要的是，persona cache 之外还需要“文本兴趣到语义候选域”的索引或 grounding 服务，其存储与计算成本没有计入上述估算。

---

## 7. Intermediate Performance Validation

这里的 Intermediate Performance Validation 指的是：在 persona 接入召回并进行最终线上 A/B 测试之前，验证这个中间用户表示是否正确、稳定、具有探索性且能被生产系统消费。本文的验证分为输入设计、student 蒸馏和用户感知三个层次。

### 7.1 输入表示是否能产生正确画像

作者使用几百名用户主动点击过的文本主题作为高置信度 reference。不同形式的观看历史输入 Gemini 后，生成 Summarized Interests，再与 reference 计算 BLEURT：

```text
某种观看历史表示
  → Gemini 生成 Summarized Interests
  → 与用户点击过的主题文本计算 BLEURT
```

BLEURT 衡量语义接近程度，不要求字面完全一致。该实验确定了 semantic clustering、视频标题、Gemini Pro 和 two-shot prompting 分别是各自维度的最佳配置。它验证的是“什么输入更容易产生正确文本画像”，不是 item retrieval 或业务效果。

### 7.2 Student 是否学会 Teacher 的任务

**Instruction Following Rate（IFR）**定义为完全满足任务和格式要求的响应比例：

\[
\mathrm{IFR}
=
\frac{\text{同时完成两项任务、数量正确且格式可解析的响应数}}
{\text{全部评估响应数}}.
\]

未蒸馏时，Nano IFR 只有 `0.07%`，Flash 只有 `1.82%`；蒸馏到 epoch 26.20 后分别达到 `99.08%` 和 `99.68%`。这主要说明模型已经学会输出协议，不等价于语义质量达到 teacher 水平。

蒸馏阶段的 **BLEURT** 以 teacher 的 Summarized Interests 为 reference，比较 student 输出。它衡量 teacher imitation，而不是直接衡量真实用户兴趣准确率；并且只在通过 IFR 的样本上计算。Epoch 26.20 时 Nano/Flash 分别为 `0.328/0.345`。

**Creativity Score** 用 LLM autorater 对 teacher 和 student 的 `{summarized interest, exploration interests}` 做 side-by-side 比较，判断哪组更有新颖性。Epoch 26.20 时 Nano 为 `-0.018`、Flash 为 `0.023`，说明探索能力对模型容量更加敏感。论文没有披露 judge 模型、Prompt、分数归一化和人工校准，因此该指标更适合选择 checkpoint，而不是解释为绝对创造力。

### 7.3 用户是否认可 Persona

论文对数千名近期活跃的美国用户开展调查。每位参与者看到一个观看簇中的三个代表视频，在确认看过后，对 LLM 标签的准确性和继续观看意愿进行五级评分。

- 超过 `80%` 的用户认为标签“非常准确”或“极其准确”地总结了这些视频。
- `71%` 希望继续观看该标签相关的视频。
- 与知识图谱实体标签相比，`57%` 偏好 LLM persona，`20%` 认为两者相当。

用户不满意主要来自遗漏主要兴趣、把偶发行为误判为稳定兴趣、保留过时兴趣和生成重复标签。真人调查弥补了 BLEURT 无法衡量覆盖率、时效性和错误归因的问题，但仍没有验证 persona 是否能正确映射到具体 item。

### 7.4 中间验证缺失的一环

本文没有独立验证 `Persona → Semantic Grounding → Restricted Candidate Set`：

- 没有报告 persona 语义候选域的 Precision/Recall。
- 没有报告 restricted retrieval 的 Recall@K、NDCG 或 Hit Rate。
- 没有比较 summarized 与 exploration interests 各自的 item relevance。
- 没有验证自由文本到 `C(t)` 的映射准确率。
- 没有验证 reasoning 是否参与或改善召回。
- 没有评估 persona freshness 对候选质量的影响。

因此本文的证据链是：

```text
BLEURT        → 文本总结是否接近 reference
IFR           → 输出是否能稳定解析
Creativity    → 探索兴趣是否具有新颖性
用户调查      → 用户是否认可兴趣标签
线上 A/B      → Persona + Grounding + Retrieval + Ranking 整体是否有效
```

与 TokenMinds 可以用 future SID Recall@10 直接验证 decoder 输出和 item space 的对齐不同，本文只验证了 persona 的格式、文本语义、探索性和用户认可度，最后依靠线上 A/B 间接证明整个 retrieval pipeline 有效。persona-to-item grounding 是其 Intermediate Performance Validation 中最明显的证据缺口。

---

## 8. 线上实验

### 8.1 设置

- 平台服务十亿级用户。
- Control 和 Treatment 使用等量且不重叠的流量。
- 实验持续超过 30 天。
- Control 使用原有生产推荐栈。
- Treatment 增加由 Gemini Nano 生成的自然语言画像召回源。
- 线上采用视频标题、视听 embedding 聚类、epoch 26.20 的 Nano。
- 每次请求随机采样一个总结兴趣和一个探索兴趣。

### 8.2 Viewer Value

相对高度优化的生产基线：

| 指标 | 相对提升 | 显著性 |
|---|---:|---:|
| 用户观看时长 | **+0.04%** | `p < 0.05` |
| 活跃用户数 | **+0.03%** | `p < 0.05` |

绝对百分比不大，但在十亿级平台上对应可观的总量。满意观看时长的主要提升来自 casual users。作者认为原因是 LLM 更能从稀疏行为推断偏好，而低活跃用户兴趣通常更集中，少量画像主题更容易覆盖其核心偏好。

### 8.3 探索效果

| 指标 | 相对提升 |
|---|---:|
| 用户参与的主题数 | **+0.04%** |
| 拥有多个持续参与主题的用户数 | **+0.03%** |

总结兴趣和探索兴趣还呈现出一个重要漏斗差异：

- 探索兴趣召回的视频获得的曝光比总结兴趣少 `40.91%`，说明下游 ranker 对新颖内容存在明显压制。
- 一旦真正曝光，探索兴趣视频被观看的概率反而高 `13.6%`。

这说明 LLM 确实找到了传统推荐系统遗漏的高潜主题，但现有 ranking policy 仍会因为历史反馈稀疏、预估不确定或分布偏移而降低其曝光。探索模块和 ranker 之间仍存在目标不一致。

### 8.4 长期影响

观看画像召回视频的用户之后产生了更多回访。论文据此认为语义兴趣不只带来一次点击，也可能形成持续价值；但没有公开具体提升幅度、观察窗口和归因方法，因此这一结论的证据弱于主线上指标。
