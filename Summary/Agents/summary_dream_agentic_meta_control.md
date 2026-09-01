# DREAM Technical Report: 用 Agentic 元控制范式重构工业推荐系统

> **论文链接**: https://arxiv.org/abs/2608.09408
> **团队**: DREAM Team（淘宝/天猫集团 Alibaba，约 90+ 位作者，含暑期实习生贡献）
> **领域**: cs.IR (Information Retrieval)
> **核心贡献**: 提出 DREAM（Developing Recommender Engine with Agentic Methods）——一个**不替换**现有召回-排序-重排流水线，而是在其之上叠加"感知-决策-执行"策略层的自主优化控制架构。在淘宝首页 Feed 的大规模 A/B 测试中，仅控制重排即可使 IPV +2.06%、Core IPV +2.39%、GMV +0.88%；扩展到精排后进一步提升至 +2.71%、+3.06%、+1.31%。

---

## 0. 读者综合理解与点评

### 读者的综合理解

本文在原有推荐链路的基础上，透露出多个控制参数，来改变整个链路的过程，如召回的侧重，重排各目标比重等。然后使用 LLM 判断用户意图，并产生具体的控制参数以调整链路。

### 点评

**整体评价：✅ 骨架准确，抓住了 DREAM 的三个核心定位，但遗漏了四个工程关键机制。**

这个一句话理解已经穿透了 DREAM 的主干——比"LLM 加权推荐"的直觉理解要准确得多。下面逐点评判。

#### ✅ 正确的部分

| 读者表述 | 点评 |
|----------|------|
| "在原有推荐链路的基础上" | **准确**。抓住了 DREAM 作为 **overlay（叠加层）** 而非 replacement（替换层）的根本特征——这是它能在不损害服务稳定性的前提下上淘宝首页 Feed 的工程根基。 |
| "透露出多个控制参数" | **准确且用词精妙**。"透露"一词暗示这些参数（召回配额、散列窗、过滤阈值、目标权重等）**本来就在生产链路里**，DREAM 只是暴露了一个可控接口让 LLM 来调——这恰好对应论文的 M3 参数翻译层和 Unified Outlet 局部覆写。参数不是 DREAM 新造的，而是既有链路被"暴露"出来的控制面。 |
| "改变整个链路的过程，如召回的侧重，重排各目标比重等" | **准确**。理解了影响范围覆盖召回/排序/重排多阶段，且举例"召回的侧重"（对应 `category_prefer`/`cardtype_prefer`）和"重排各目标比重"（对应 `ranking_weight_boost`）都正确。 |
| "使用 LLM 判断用户意图" | **方向正确**。LLM 确实在做意图判断，且这是整个流程的起点。 |
| "产生具体的控制参数以调整链路" | **方向正确**。最终产出确实是控制参数，且确实用于调整链路。 |

#### ⚠️ 遗漏的工程关键机制（四点）

虽然骨架准确，但这句话作为"一句话总结"省略了四个让 DREAM 能真正落地的工程机制。这些机制不是次要细节，而是 DREAM 区别于"传统 LLM 加权推荐"和"简单 LLM 控制器"的关键。

**(1) 是两个独立的 LLM，不是一个**

读者说"使用 LLM 判断用户意图，并产生控制参数"——听起来像一个 LLM 既判意图又出参数。实际上是两次独立 LLM 推理：
- **Intent Engine LLM**（0.8B Main + 4B expert + 4B Dreaming）：理解"用户要什么"→ 产出三层结构化意图
- **MetaModel LLM**（Qwen3）：决定"系统该做什么"→ 产出 6 维度策略 bundle

中间还隔着**频率控制**（Lagrangian 决策 $a_{u,t}=m_{u,t}\cdot\mathds{1}[S_{u,t}\geq\theta_{u,t}]$），决定是否值得调第二个 LLM。不是每次请求都跑完整流程。

**(2) LLM 不直接产生"控制参数"，而是产生语义级策略，再经确定性编译**

读者说"产生具体的控制参数"——暗示 LLM 直接输出数值参数。实际上 LLM（MetaModel）的输出是**语义级 M2 bundle**（如 `cvr:+2`，值取自 $\{-2..2\}$ 严格枚举），再经 **M3 确定性编译**（如 $\delta_{cvr}=w_{cvr}^0\times 2=0.08$）翻译为生产参数。

这是 DREAM 可审计性的关键：LLM 的不确定性被限制在 M2 语义层，M3 是无 LLM 推理的确定性映射。如果 LLM 直接产出数值参数，可审计性和可回滚性都会大打折扣。

**(3) 缺少安全护栏：四道校验门 + 默认回退**

读者说"调整链路"——但没有提到调整是**有界的局部覆写**。Unified Outlet 的执行公式是 $p_s^{\mathrm{exec}}=\mathrm{Guard}_s(p_s^0\oplus\mathcal{T}_s(a_u))$，其中 $\oplus$ 是局部覆写非全配置替换。每次覆写过四道校验门（JSON 解析→schema 校验→白名单→范围校验），失败即回退到默认配置 $p^0$。这是"不损害服务稳定性"的安全网。

**(4) 缺少闭环反馈：Reward Dual Loop**

读者的描述是开环的——"判意图→出参数→调链路"。但 DREAM 是闭环自进化的：在线环把执行后的 IPV/CVR/GMV 反馈对照策略，验证结论存入 Strategy Memory；离线环用 Evaluator 在 logged 上下文上 replay 候选 bundle 探索策略空间；M2 检索 Strategy Memory 历史经验（正向作参考、负向作约束）。这让系统能持续自进化而非一次性前向推理。

#### 修正后的"一句话理解"

如果要把读者的理解补全为一句话：

> 本文在原有推荐链路的基础上**暴露出多个控制参数**，来改变整个链路的过程（如召回的侧重、重排各目标比重等）。然后**使用两组 LLM**——Intent Engine LLM 判断用户意图产出**三层结构化意图**，MetaModel LLM 基于意图产出**6 维度语义级策略 bundle**——经**确定性 M3 翻译**为具体控制参数，通过**局部覆写 + 四道校验门**以调整链路；调整结果经**在线+离线双环反馈**持续自进化策略与意图理解。

核心保留读者的"overlay + 多参数 + LLM 意图驱动"骨架，补全"双 LLM 分工 + M3 确定性编译 + 安全护栏 + 闭环反馈"四个工程关键机制。

#### 读者理解演进轨迹

读者在 Q2-Q4 三次提问中逐步逼近 DREAM 的完整定位，本综合理解是这一演进的结果：

| 阶段 | 理解 | 层次 |
|------|------|------|
| Q2 | LLM 判意图 → 出目标权重 | 仅 `ranking_weight_boost` 一个维度 |
| Q3 | → 影响 6 个维度，覆盖召回/排序/重排 | 完整 M2 策略 bundle |
| Q4 | → 触发不同排序逻辑 | ❌ 仍差"参数覆写 vs 逻辑替换" |
| Q4 修正 | → 调整既有逻辑的参数（overlay） | ✅ 理解 overlay 定位 |
| **综合理解** | overlay + 多参数 + LLM 意图驱动 | ✅ 骨架准确，遗漏四工程机制 |

---

## 1. 动机与问题

工业推荐系统普遍采用"召回 → 排序 → 重排"的级联流水线，模块化设计稳定高效，但随着用户行为复杂化和业务目标多元化，暴露出四个结构性瓶颈：

1. **信息碎片化（Information Fragmentation）**：上游模块对下游结果不可见，反之亦然，跨模块信息无法贯通（时间穿越？）。
2. **目标分散（Objective Scattering）**：CTR、CVR、增长、体验各自独立优化，缺乏统一仲裁（混排的事情）。
3. **策略刚性（Strategy Rigidity）**：多数策略仍来自静态规则、人群切分和人工调参，无法跟随实时状态（还是混排的事情）。
4. **实时意图感知薄弱（Weak Real-time Intent Awareness）**：浏览、比价、购买等会话级状态切换得不到充分响应（用户意图识别，模型也能做，倒不如说是可解释性）。

### 1.1 现有 LLM Agentic 推荐工作的三大局限

论文梳理了四类已有 agentic 推荐方向——选品 agent、用户模拟 agent、对话 agent、系统编排 agent，指出它们各自只解决问题的一个侧面，普遍存在：

- **意图感知与策略生成脱节**：选品 agent 仅感知个体历史，错失端侧微信号、会话级意图和全局业务状态；编排 agent 自动化架构演进，但策略规划缺乏实时用户意图，"看得远却瞄不准"。
- **策略僵化、多目标协调弱**：规则无法跟踪实时状态切换；单体 agent 把所有目标揉进一次决策，无显式仲裁。AgenticRecTune 虽用 Pareto Memory 调融合权重，但仅在全局层面，非动态 per-user 策略。
- **缺乏从执行到上游优化的闭环反馈**：现有编排方案的 loop 服务于架构级迭代，而非用户级参数的实时校准；执行信号无法回流到感知和决策模块，优化循环被切断。

DREAM 的核心定位：**一个感知-aware、可编排、可审计的策略层**（那就是一个大混排？），作为 overlay 叠加在流水线之上而非替换任何模型。

---

## 2. DREAM 框架总览

DREAM 由三个核心模块构成一个协作闭环：

```
          ┌─────────────────────────────────────────────────────┐
          │                Reward Dual Loop                      │
          │   离线仿真探索策略空间  ⇄  在线真实反馈校准          │
          │   (Evaluator + Replay)        (IPV/CVR/GMV 结论入库) │
          └──────────┬───────────────────────────┬─────────────┘
                     │ Retrieve Experience        │ Closed-loop Feedback
                     ▼                            ▼
   行为信号 ──→ [Intent Engine] ──→ 结构化意图 ──→ [Meta Engine] ──→ 策略参数
   (端侧/云侧)    感知层                (L0/L1/L2)    决策层              │
                     ▲                                                    │
                     └────────────── 执行信号回流 ──────────────────────────┘
                                                            │
                                                            ▼
                                                  [Unified Outlet]
                                            默认回退 + 个性化覆写 + 安全护栏
                                                            │
                                                            ▼
                                          Homepage Feed / ND / Supply / TAB2
                                       (召回 / 排序 / 重排 流水线不变)
```

**三大模块职责**：

| 模块 | 角色 | 关键产出 |
|------|------|----------|
| **Intent Engine**（感知层） | 把异构行为信号转为结构化三层意图（L0/L1/L2），解耦意图"生产"与"消费" | 统一意图表示 + 实时状态 + 系统反馈 |
| **Meta Engine**（决策层） | MetaModel 协调子 agent，做 M1→M2→M3 分层推理，将意图翻译为有界可执行参数 | 策略 bundle（JSON，受 schema 约束） |
| **Unified Outlet**（执行层） | 通过"默认回退 + 个性化覆写"机制注入下游场景，配实时监控与熔断 | 受护栏保护的增量参数覆写 |

闭环数据流：行为信号从执行上行到感知 → 意图表示从感知流入决策 → 策略参数从决策回到执行。跨会话积累意图历史与策略结果，DREAM 持续精化意图理解与策略决策。

---

## 3. Intent Engine（意图引擎）详解

Intent Engine 是 DREAM 的感知基础，定位为**实时意图识别与分发中枢**——一个介于上游行为数据源与下游推荐应用之间的智能中间件。它遵循四条设计原则：① 生产与消费解耦；② 实时优先（高优信号亚秒级更新）；③ 可解释（每个意图带置信度和推理理由）；④ 渐进式 rollout。

架构遵循 **trigger-route-evolve-supply** 控制流，包含三个紧耦合子模块。

### 3.1 Multi-Source Data Perception（多源数据感知）——Traffic Funnel F1–F4

淘宝规模下原始信号空间庞大异构：跨多个业务域（详情、搜索、购物车、支付、订单、直播、内容等），数百种用户行为。核心挑战是**信号采集瓶颈**：原始行为流量远超云端推理引擎在延迟和预算约束内可吞咽的量，而端侧算力、带宽、功耗又严格受限。

解决方案是 **Traffic Funnel**——一个级联的端云链（F1–F4），随信号从客户端向云端传播，逐级过滤、压缩、富化：

| 阶段 | 位置 | 功能 | 决策输出 |
|------|------|------|----------|
| **F1** | 端侧 | 信号编码 | 原始追踪点编码为可学习特征 |
| **F2** | 端侧 | 触发判定 | 意图变化点是否触发上报 |
| **F3** | 端侧 | 最小上报 | ID 编码的行为包是否上传 |
| **F4** | 云侧 | 富化 + 准入 | 是否调用 Intent Reasoning Core |

**端侧分流（F1–F3）** 在 <100ms 预算内完成主要压缩：
- **F1 信号采集**：实时把原始追踪点编码为命名信号、行为 ID 和少量离散特征维度，将端侧可用行为词表从 6 种扩展到 60+ 种。
- **F2 触发判定**：单层 GRU 在 7 维离散特征流上做变点检测（隐藏态跨事件传递），仅疑似意图变化点才触发上报。短窗误触保护（2s 内返回原场景或曝光前返回）抑制误触；RuleTree 兜底保证模型不可用时门控仍可用。生产中此门仅放行约 **15%** 的行为。
- **F3 最小上报**：F2 触发时，把近期行为序列（至多 50 个事件，带 item/shop 实体 ID）打包为 ID 编码的最小行为包，端侧不做语义富化，最小化带宽同时保留完整行为轨迹。区别于传统单事件上报，这是会话聚合的跨域上下文载荷。

**云端富化与准入（F4）**：行为包到达后，F4 恢复端侧剥离的语义，做第二次准入决策。把包与最近的意图列表（L1–L2）和 L0 静态画像（$P_{L0}$）做 join，把 `item_id`、`scene_id` 等映射回文本（标题、店铺、类目），约 5ms/行为。最后准入检查判断富化更新是否足够实质性以触发模型推理，平凡或冗余的包在云边被吸收，不升级到推理核心。

**整体效果**：从约 8.7% 的行为最终触发云端推理——这是摘要中"reporting volume ≈ 8.7%"的来源（端侧 15% × 云端二次准入后进一步筛减）。

### 3.2 Intent Reasoning Core（意图推理核心）

认知中枢，结合行为包、L0 画像和先前意图状态，产出结构化 L1/L2 意图。需同时满足：在线延迟预算内的**及时推理** + 行为证据累积下的**一致维护**。由两个互补机制支撑。

#### 3.2.1 Multi-Scale Intent Model（多尺度意图模型）

五个要素：三层意图 schema、同步 0.8B Main Agent、双层路由、异步精化层、自进化循环。

**分层意图 Schema**：每个用户由共享 L0 画像 $P_{L0}$ + 一个活跃意图列表表示。每个活跃意图含 L1 需求表示和 L2 偏好表示。

| 层级 | 名称 | 内容 | 更新节奏 |
|------|------|------|----------|
| **L0** | Physical（物理层） | 人口属性、持久兴趣、长期行为记忆、身份标签；所有活跃意图共享 | 粗粒度 |
| **L1** | Demand（需求层） | 需求品类、需求场景、品类认知、目标受众、时间节点、意图置信度 | 在线实时 |
| **L2** | Preference（偏好层） | 子类目、品牌倾向、价格偏好、属性偏好、决策状态、实时心理 | 在线实时，异步精化的主要目标 |

每个活跃意图还带稳定 intent ID、1–5 优先级、三种意图类型之一（`goal-driven` / `inspiration exploration` / `aimless browsing`）。未知字段显式表示为 null 或空列表，而非臆测。

**Main Agent**：0.8B 语言模型，对 Traffic Funnel 准入的每个请求同步运行。用 $P_{L0}$、上一轮意图列表 $\mathcal{I}_{t-1}$、当前行为包 $\mathcal{B}_t$ 更新活跃意图列表，**立即**返回下游（不等异步精化）。形式化：

$$(\Delta\mathcal{I}_t,\, r_t) = f_\theta(P_{L0},\, \mathcal{I}_{t-1},\, \mathcal{B}_t)$$

其中意图更新 $\Delta\mathcal{I}_t$ 含 `insert`/`update` 操作（仅增量，非重写全表，保持解码量与实际变化成正比）；路由决策 $r_t$ 指明是否需异步精化及目标 agent。

**双层路由**：Main Agent 返回后立即运行，无额外模型推理、不增加延迟。
- **规则层**：检查观测行为数和 L1 需求品类数，超阈值即标记。
- **模型层**：用 Main Agent 输出的升级置信度 $c \in [0,1] \cup \{\text{null}\}$，超阈值 $\tau$ 提供额外升级信号；null 时交规则层独判。

任一层出阳性即升级。生产中约 **6.3%** 请求送入异步 4B 层。

**异步精化层**：两个专用 4B agent 分工——
- **context subagent**：L1 需求合理但子类目/意图结构不确定时，用更长行为历史、跨会话上下文和 L0 画像处理。
- **expert**：类目已确定但 item 级偏好仍不确定时，精化价格偏好、属性偏好、决策状态、意图类型等字段。

被选请求异步处理，Main Agent 输出仍立即返回。精化结果并入后续请求的意图状态。冲突用字段级确定性策略解决：近期可靠行为证据支持的字段优先；其余字段仅当异步估计的字段级置信度超阈值才更新。

**自进化**：两个时间尺度。
- **交互级**：异步输出触发对用户历史意图列表的异步更新，让后续推理无需立即更新参数即可受益（零成本、零延迟适配）。
- **模型级**：累积精化样本做 **on-policy distillation**。Main Agent 采样意图更新序列 $\hat{y}_t \sim p_\theta(\cdot|x_t)$，4B agent 在相同 student 生成前缀上提供 next-token 分布。目标为反向 KL（沿 Main Agent 生成的轨迹评估，对齐训练分布与推理状态）：

$$\mathcal{L}_{\mathrm{OPD}}(\theta) = \mathbb{E}_{x_t,\, \hat{y}_t \sim p_\theta(\cdot|x_t)} \left[ \frac{1}{|\hat{y}_t|} \sum_{j=1}^{|\hat{y}_t|} \mathrm{KL}\big( p_\theta(\cdot|x_t, \hat{y}_{t,<j}) \,\|\, p_\phi(\cdot|x_t, \hat{y}_{t,<j}) \big) \right]$$

仅监督意图更新序列 $\Delta\mathcal{I}_t$，不监督路由决策 $r_t$。

**评估**（两类基于后续行为的协议）：
- **LLM-as-a-Judge**：judge 模型收到预测意图 + 评估窗口内后续行为，按与后续曝光/搜索/点击的一致性逐字段评估。
- **Search-Behavior Recall**：用推理后首次搜索会话作为行为证据，测搜索词召回（预测意图与后续 query 语义一致性）和过滤器召回（预测细粒度偏好是否正确预判用户搜索时选的过滤条件）。

| 变体 | LLM-as-a-Judge 总分 | 搜索词召回 | 过滤器召回 |
|------|:---:|:---:|:---:|
| Baseline | 71.32% | 50.57% | 24.25% |
| + Routing | 78.20% (+6.88) | 51.68% | 26.79% |
| + Routing + Self-Evolution | **84.74%** (+6.54) | **56.05%** (+4.37) | 26.40% (-0.39) |

路由与自进化提供互补增益。

#### 3.2.2 Dreaming Mechanism（做梦机制）——离线全日整合

在线推理仅看触发时刻可得证据，窄视野导致三种失败模式：① **意图膨胀**（碎片证据造出语义等价的重复意图）；② **意图不准**（证据不足时就推断字段）；③ **意图遗漏**（单个弱但一致的信号永远不触发在线更新）。夜间用户请求大幅下降，GPU 闲置。Dreaming 利用低流量窗口整合意图状态，不影响峰值时段。

设 $\mathcal{I}_D$ 为第 $D$ 天在线最终意图列表，$\mathcal{B}_D$ 为当天完整行为轨迹。一次 Dreaming：

$$\mathcal{I}_{D+1}^{\mathrm{init}} = \textsc{Dream}(P_{L0},\, \mathcal{I}_D,\, \mathcal{B}_D)$$

它**完全替换**累积的在线意图状态，初始化第 $D+1$ 天首个请求的 $\mathcal{I}_{t-1}$。区别于在线增量 $\Delta\mathcal{I}_t$（仅 `insert`/`update`），Dreaming 在完整列表上用六种操作：

| 操作 | 定义 | 功能 |
|------|------|------|
| `keep` | 与扩展行为证据一致时保留 | 保存 |
| `correct` | 修正与扩展证据冲突的字段 | 纠错 |
| `enrich` | 用额外证据补全缺失/欠指定字段 | 信息补全 |
| `merge` | 合并指向同一底层需求的多个条目 | 去重 |
| `add` | 创建证据支持但在线列表缺失的意图 | 遗漏恢复 |
| `kill` | 移除已满足/被放弃/被取代/不再被证据支持的意图 | 状态退役 |

**触发与输入**：定时日 pass（部署路径）；亦可由用户长期不活跃、活跃意图数超阈值 $M$、未纳入行为证据量超阈值 $N$ 触发额外 pass（不清零日索引）。每次消费三输入：$P_{L0}$（稳定上下文）、$\mathcal{I}_D$（在线累积的完整意图状态）、$\mathcal{B}_D$（时序排列的全日行为轨迹，比单在线包提供更广的跨会话证据，含搜索/点击/购物车/购买及显式负反馈）。

**整合纪律**（五条证据处理规则）：
1. **结果优先推理**：先看最新相关行为、交易状态、显式放弃信号，再回看更早证据，支撑可靠的 `kill`/`correct`。
2. **行为聚类与意图对齐**：按品类/场景/时段聚簇后再对齐既有意图，为 `merge`/`add` 提供连贯证据。
3. **证据优先于画像**：仅行为证据不足时才用 L0 画像；画像不覆盖清晰近期的行为观察。
4. **意图优先级**：$R(i) = s(i)\,\kappa(i)\,\gamma(i)\,u(i)$（行为强度 × 收敛度 × 时近性 × 未满足度），归一化后按降序排列。
5. **属性级负证据**：显式负行为映射为属性级排除，不泛化为品类级排除（避免误杀下游召回）。

**评估**（683 用户配对设计，同一次日行为证据上比较在线列表 vs 整合后列表）：

| 层 | 字段 | 白天 | Dreaming 后 |
|---|------|:---:|:---:|
| Intent | Intent type | 0.583 | **0.680** |
| Intent | Priority | 0.524 | **0.571** |
| L1 | Demand category | 0.570 | **0.580** |
| L1 | Category cognition | 0.819 | **0.838** |
| L1 | Demand scenario | 0.797 | **0.817** |
| L1 | Temporal node | 0.893 | **0.897** |
| L1 | Intent confidence | 0.592 | **0.611** |
| L2 | Subcategory | 0.477 | **0.523** |
| L2 | Decision state | 0.561 | **0.592** |

所有字段均提升，Intent Type / Priority / Subcategory 提升最大（意图组织与品类解析增强）；本就可靠的字段（temporal node、Need Category）仅小幅提升。

### 3.3 Downstream Applications（下游应用）

三层结构化输出被下游消费，按使用方式分三类：

| 应用类别 | 消费的意图字段 | 实现方式 |
|----------|----------------|----------|
| **召回** - Cognitive Recommendation | L1 需求品类；L2 子类目 | 标签召回 |
| **召回** - Heuristic Recommendation | L1 需求品类；L2 子类目；L2 属性偏好-优先属性 | Inquiry Card（询问卡） |
| **显式文本交互** | L0 画像；L1 需求品类&场景；L2 决策状态 | Inquiry Card 文案 |
| **推荐策略自适应** | L1 需求品类&品类认知；L2 子类目&优先级 | 个性化参数 |

- **认知推荐**：L1 品类标签 + L2 子类目标签作为标签召回模型的统一输入，补充既有模型生成标签，提供显式分层需求语义。
- **启发式推荐**：通过 **Inquiry Card**（推荐流中与普通商品卡并排的交互卡，展示 articulating 用户需求的文案，点击进入落地页）浮现潜在需求。融合 L1/L2 多级信号，覆盖从开放探索到明确目标的全状态。
- **显式文本交互**：L0 画像提供个人/情境上下文；L1 品类+场景决定文案主题与角度；L2 决策状态决定沟通策略，结合消费心理学多种文案风格产出差异化文案。
- **推荐策略自适应**：按当前意图状态动态调引擎参数——需求集中的用户扩大相关通道规模与覆盖；意图分散的用户缩小部分通道以降噪。意图输出喂入轻量 LLM 推理模块，生成个性化参数（召回通道开关、配额参数、多样化参数）。

三层输出跨召回/交互/策略自适应被消费，推荐结果影响后续行为，驱动下一轮 Intent Engine 更新，形成闭环。其中 Intent Engine 负责感知，Meta Engine 决定何时触发推理、运行频率、如何编排策略。

---

## 4. Meta Engine（元引擎）详解

Meta Engine 是 DREAM 的策略决策层。Intent Engine 决定"此刻该如何理解用户"，Meta Engine 决定"系统该用这个理解做什么、如何落地"。它**不替换**召回/排序/重排，而是把每个下游模块当作可控工具，以结构化意图和实时状态为感知输入，MetaModel 作为主 agent 做全局策略规划。

MetaModel 基于 **Qwen3**，融合统一感知与历史经验，把"用户现在要什么"+"业务要优化什么"转为可执行策略指令，同时在转化 vs 体验、探索 vs 相关等竞争目标间仲裁。

### 4.1 Agent Loop 触发与频率控制

每个事件后都调 LLM 是浪费的（意图在相邻交互间常稳定）。频率控制被形式化为**选择性计算问题**：何时刷新意图的期望价值值得其成本？

**问题形式化**：设用户 $u$，近期历史窗 $W$，事件驱动决策 epoch $t$。决策上下文：

$$x_{u,t} = (\mathcal{H}_{u,t}^{W},\, p_u,\, c_{u,t},\, q_{u,t}^{\mathrm{cache}},\, h_{u,t}^{\mathrm{call}})$$

二元动作 $a_{u,t}\in\{0,1\}$ 决定调 LLM 还是复用缓存/启发式结果。反事实刷新价值：

$$\Delta_{u,t} = \mathbb{E}[R(q_{u,t}^{\mathrm{llm}};x_{u,t}) - R(q_{u,t}^{\mathrm{cache}};x_{u,t}) \mid x_{u,t}]$$

带预算的约束优化：

$$\max_\pi \mathbb{E}_\pi[a_{u,t}\Delta_{u,t} - \lambda_c a_{u,t} C_{u,t}^{\mathrm{llm}} - \lambda_l a_{u,t} L_{u,t}^{\mathrm{llm}}] \quad \text{s.t.} \quad \mathbb{E}_\pi[\sum_{(u,t)\in\mathcal{B}_{g,b}} a_{u,t}] \leq B_{g,b}$$

由于 $\Delta_{u,t}$ 在线不可得，用触发信号估计刷新紧迫度。单调校准 $\widehat{\Delta}_{u,t}=\phi_{g,b,s}(S_{u,t})$，点wise Lagrangian 决策等价于上下文相关的分数阈值：

$$a_{u,t}^\star = \mathds{1}[\widehat{\Delta}_{u,t} \geq \lambda_c C_{u,t}^{\mathrm{llm}} + \lambda_l L_{u,t}^{\mathrm{llm}} + \mu_{g,b}]$$

**四类触发信号**（用鲁棒归一化 $\mathcal{N}_{j,g,s}$ 映射到 $[0,1]$，防止高方差特征仅因尺度支配）：
- **行为分布漂移** $S^{\mathrm{drift}}$：用对称 Jensen-Shannon 散度测品类/品牌/动作类型/通道的当前 vs 历史分布漂移。
- **活跃度** $S^{\mathrm{act}}$：$\log(1+n^{\mathrm{pv}}+\alpha n^{\mathrm{uv}}+\beta n^{\mathrm{act}})$。
- **时间因子** $S^{\mathrm{time}}$：早高峰/下午/晚高峰/深夜 + 日级上下文。

融合：$S_{u,t} = w_1 S^{\mathrm{drift}} + w_2 S^{\mathrm{act}} + w_3 S^{\mathrm{time}}$。

**动态阈值**：固定阈值无法同时捕捉用户价值、流量、场景重要度与容量。阈值分解为全局基线 $\theta_0$ + 四个加性校正：

$$\theta_{u,t} = \mathrm{clip}(\theta_0 + \delta_{g(u)} + \delta_{b(t)} + \delta_{s(t)} + \delta_{d(t)},\, \theta_{\min},\, \theta_{\max})$$

- $\delta_{g(u)}$：用户分群偏移（按历史价值高/中/低分群，各群学一个偏移）
- $\delta_{b(t)}$：时段偏移（高峰 vs 非峰，重载时抬升 $\theta$ 抑制低价值调用）
- $\delta_{s(t)}$：场景偏移（首页刷新/搜索/详情页等重要性不同）
- $\delta_{d(t)}$：日级漂移校正

资格掩码 $m_{u,t}\in\{0,1\}$ 强制冷却、per-user 上限、场景白名单、过载保护。最终决策：

$$a_{u,t} = m_{u,t}\,\mathds{1}[S_{u,t} \geq \theta_{u,t}]$$

图中给出高/中/低价值用户阈值 0.45/0.60/0.75——价值越高，调用门槛越低。

### 4.2 策略编排与参数翻译（M1→M2→M3）

MetaModel 在**语义动作空间**推理，而生产服务消费数值参数/ID/过滤器/配额。DREAM 用共享策略契约 + 阶段专用 Tool Processor 桥接：契约跨流水线共享，参数解释各阶段本地。这让 LLM 协调耦合决策而不生成任意服务配置或替换既有实现。

**三层策略契约**：
- **M1**：总结当前用户状态，确立全局导向——决策理由、目标导向（IPV-oriented / GMV-oriented）、购买力分群（K1–K6）、活跃度分群（a1–a5）。
- **M2**：MetaModel 生成的 schema 约束 JSON bundle，含语义动作（objective boosts / business support / category & content-type preferences / experience constraints / position policies）。**值来自严格枚举**（如 $\{-2,-1,0,1,2\}$），压缩输出空间以保稳定且可审计。
- **M3**：**非自由 LLM 输出**，是把 M2 动作编译为生产服务消费参数的确定性结果。

| M2 字段 | 语义决策 | 当前消费者 |
|---------|----------|------------|
| `intent_summary` | 当前意图与可用供给总结 | MetaModel 规划上下文 |
| `ranking_weight_boost` | CTR/IPV/CVR/GMV 相对侧重 | 排序与重排 |
| `business_support` | 白名单保护的 PlanID | 排序截断与下游混合 |
| `category_preference` | 品类/品类标签偏好 | 召回、排序、重排 |
| `cardtype_preference` | 商品/视频/直播/内容供给偏好 | 召回与重排 |
| `experience_constraints` | 曝光/购买/密度约束 | 重排 |
| `top_ctr_strategy` | 分页 top-CTR 与倒序开关 | 重排 |

一个字段可被多阶段消费，但每个 processor 赋予阶段适用的操作含义。例如 `category_preference` 可在召回增大配额、排序放松品类散列窗、重排调品类构成——**语义偏好共享，底层参数不共享**。

**请求时 Tool 编排**：LLM 推理与延迟关键服务路径解耦。近线 MetaModel 把最新有效策略 bundle 写入在线缓存。请求到达时，Tool Process 读缓存 bundle，校验版本与 schema，分发相关字段到召回/排序/重排 processor：

$$p_s^{\mathrm{exec}} = \mathrm{Guard}_s(p_s^0 \oplus \mathcal{T}_s(a_u)), \quad s\in\{\mathrm{ret},\mathrm{rank},\mathrm{rerank}\}$$

$\oplus$ 是局部覆写而非全配置替换。processor 只读白名单字段，故缺失/过期/畸形 bundle 让对应默认参数不变。

**阶段专用参数翻译**：

- **召回**：`category_preference` → 提取品类标签值序列化为 `metamodel_boost_cate`，用 `metamodel-boost` 队列保护这些品类的召回量。`cardtype_preference` 当前支持硬视频供给抑制（视频值为 -2 时发 `metamodel_adjust_cardtype=video` 指令召回服务不召回视频）。
- **排序**：消费三个 M2 模块。
  1. `ranking_weight_boost` 修改原始 LTR 分（不替换）。目标 CTR/IPV/CVR/GMV，各目标读基线权重 $w_i^0$ 和有界语义级 $b_i\in\{-2,-1,0,1,2\}$，$\delta_i = w_i^0 b_i$。排序服务消费 `ltr_ctr_delta` 等作为乘性校正：$f_{\mathrm{rank}} = f_{\mathrm{ltr}} \prod_i (1+\delta_i \hat{v}_i)$。
  2. `business_support.plan_ids` → `guaranteed_plan_ids`，关联 item 可绕过正常截断配额（至上限 `business_support_max_num`）。
  3. `category_preference` 控截断前候选多样性：语义比 $b_c \in \{-2..2\}$ → 散列窗调整 $n_c^{\mathrm{new}} = \max(1, n_c^{\mathrm{default}} + b_c)$。负值强分散，正值允许意图明确时更集中。
- **重排**：应用剩余联合策略到最终列表构建——目标权重调整、品类/内容类型偏好、曝光与购买过滤、密度约束、分页 top 位置规则。配置既有生产 Generator 或白名单重排 recipe；**LLM 不输出 item ID 或最终排列**。

**校验/隔离/回退**：每次翻译过四道门——JSON 解析、schema 校验、阶段级白名单检查、参数范围校验。业务支持有上限、散列窗有正下界、枚举域外的语义级被拒。编译结果限定于指定请求与 processor，覆写不能修改无关阶段或全局配置。任一门失败，DREAM 记录失败并执行生产默认。这种"默认 + 局部覆写"设计给 MetaModel 有用控制面同时保留可审计性、可回滚和既有流水线的安全边界。

### 4.3 离线 RL 训练框架

DREAM 离线训练策略，同时通过生产推荐路径执行 rollout——避免在用户流量上做探索性决策，仅在策略经相同 Tool Processor 翻译后评估。学习问题是**单步上下文决策**：给定 logged 请求，MetaModel 产出一个策略 bundle，生产流水线执行，list-level Evaluator 给最终序列一个代理结果。**不模拟后续点击/交易/用户状态转移**。

**Replay 数据集与生产执行**：从真实在线请求输入日志构造。每条记录 $x$ 构造两个处理：

$$L_{x,k}^{0} = F(x; p^0, \xi_{x,k}^{0}), \quad L_{x,k}^{1} = F(x; p^0 \oplus \mathcal{T}(a_x), \xi_{x,k}^{1}), \quad a_x \sim \pi_\theta(\cdot|x)$$

$F$ 是既有"召回-排序-重排"流水线，$p^0$ 默认配置，$\mathcal{T}$ 阶段翻译器集合，$\xi$ 是 replay 调用中遇到的实时特征/模型/服务变异。两处理作为独立调用发到生产服务。即使同源 log 记录，输出也不必相同（在线特征与生产模型在 replay 时解析）。故每处理执行 $K$ 次，用生产 list-level Evaluator $E$ 打分：

$$u_{x,k}^{z} = E(L_{x,k}^{z}, x), \quad \bar{u}_x^{z} = \frac{1}{K}\sum_{k=1}^{K} u_{x,k}^{z}$$

均值聚合降低独立服务调用方差，但不把 replay 变成历史服务状态的精确重建，所有结论相对于 replay 窗口内的生产环境。

**二元 Evaluator 奖励**：刻意简单的决策对齐奖励——生成的策略仅当其均值 Evaluator 分超过同 logged 请求默认流水线均值分时获正奖励：

$$r(x, a_x) = \mathds{1}[\bar{u}_x^{1} > \bar{u}_x^{0}], \quad \max_\theta J(\theta) = \mathbb{E}_{x\sim\mathcal{D}, a_x\sim\pi_\theta(\cdot|x)}[r(x,a_x)]$$

该比较考虑了请求间 Evaluator 分数尺度差异，直接问"个性化全流水线覆写是否优于生产默认"。Evaluator 分差保留为诊断，但训练奖励本身是二元的——不结合 rank/lift/课程竞争/teacher distillation 项。

Evaluator 是即时列表质量的学习代理。延迟在线结果（点击、交易）不在此离线 loop 合成为 episode 反馈，它们属于 DREAM 的在线奖励 loop——观测结果用于监控策略、重校 Evaluator、把验证结论整合进 Strategy Memory。

**安全 Replay 与预部署验证**：replay 调用用隔离的压测流量，输出列表返训练环境不暴露给用户、不写曝光/归因/业务计数。训练与服务用相同 schema/翻译逻辑/范围检查/回退行为。策略无效、缓存 miss、翻译失败、参数越界都回退 $p^0$ 并记为无效策略动作。

上线前在留出请求日志上评估策略，主离线指标：对默认流水线的二元胜率、均值 Evaluator 分差、策略有效性、Tool Process 成功率、回退率。另按用户状态和意图分段检查确保聚合收益不掩盖系统性回退。只有通过的策略进入受护栏流量爬坡和随机在线 A/B。

---

## 5. Reward Dual Loop（奖励双环）

闭环优化机制，让系统无需人工干预持续自进化：

- **离线环（Offline Loop）**：用 Evaluator 对策略空间做模拟探索——在 logged 用户上下文上 replay 候选 bundle，产出打分后的策略结论。
- **在线环（Online Loop）**：用实时结果（用户反馈和 IPV/CVR/GMV 等指标信号）对照产生它们的策略，产出验证结论存入 Strategy Memory。

两条反馈路径把 loop 连回上游：
- **Retrieve Experience**：把结论喂入 Meta Engine 的策略规划阶段。
- **Closed-loop Feedback**：把执行信号传回 Intent Engine 做感知校准。

共同支撑"策略生成 → 执行 → 评估 → 经验积累 → 再生成"的持续循环。

**Strategy Memory**：记录某策略对某类用户状态是否有效。按 M1 分段索引，由 Reward Dual Loop 更新——在策略规划时把正向经验作为候选参考、负向经验作为约束，形成从生成到执行/评估/整合/再生成的闭环。

---

## 6. 实验结果

### 6.1 在线累计分阶段消融（淘宝首页"猜你喜欢"）

累计实验设计：同一 Intent Engine + MetaModel + 参数翻译 + 安全护栏，先加重排控制，再加精排控制，隔离扩展控制到下一阶段的条件增益。

| 配置 | PV↑ | IPV↑ | Core IPV↑ | PCTR↑ | Click UV↑ | UCTR↑ | GMV↑ | Ad Cost↑ |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| DREAM (@ rerank) | +1.03% | +2.06% | +2.39% | +0.76% | +0.54% | +0.53% | +0.88% | **+0.21%** |
| DREAM (@ rerank & rank) | **+1.04%** | **+2.71%** | **+3.06%** | **+1.25%** | **+0.68%** | **+0.81%** | **+1.31%** | +0.02% |

- 仅重排控制：所有指标全正，PV +1.03%、IPV +2.06%、Core IPV +2.39%、GMV +0.88%、PCTR +0.76%。
- 扩展到精排：IPV/Core IPV 升至 +2.71%/+3.06%，GMV 升至 +1.31%（额外 +0.65/+0.67/+0.43 pp）；PCTR 从 +0.76% 升至 +1.25%；Click UV/UCTR 从 +0.54%/+0.53% 升至 +0.68%/+0.81%。
- **PV 在两处理间几乎不变（+1.03% vs +1.04%）**——说明更深阶段增益主要来自既有曝光的参与度与转化提升，而非额外曝光量。这印证了"agentic 元控制收益随控制面扩大而复利累积"。

### 6.2 Intent Engine 下游应用评估

独立在线 A/B（各场景 treatment 集成对应意图字段，baseline 同场景无意图信号）。

**平台级效果**（跨所有推荐流）：

| 应用 | IPV↑ | Core IPV↑ | PCTR↑ | 交易量↑ |
|------|:---:|:---:|:---:|:---:|
| 召回（认知推荐） | +0.80% | +0.91% | +0.84% | +0.36% |
| 推荐策略自适应 | +0.52% | +0.70% | -0.02% | +0.33% |

策略自适应 PCTR 中性（-0.02%），说明策略调整提升转化深度而不牺牲点击率。

**启发式推荐场景内效果**：

| 应用 | PV↑ | Inquiry Card 点击↑ | CTR↑ |
|------|:---:|:---:|:---:|
| 召回（启发式推荐） | +5.06% | +7.17% | +2.00% |
| 显式文本交互 | +7.41% | +10.64% | +3.01% |

两类优化互补：召回集成决定哪些需求浮现为询问卡，文案优化决定如何向用户 verbalize 每个需求。

### 6.3 离线 MetaModel 策略评估（4B replay-RL vs Base）

| 策略 | pCTR↑ | pCVR↑ | pIPV↑ | pGMV↑ | Valid↑ |
|------|:---:|:---:|:---:|:---:|:---:|
| RL (4B) | **+2.42%** | -0.99% | **+1.38%** | **+0.37%** | **+22.25%** |

replay RL 在 pCTR/pIPV/pGMV 上正向，pCVR 略降。**有效性从 80.86% 升至 98.85%（+17.99 pp）**——replay RL 大幅改善策略可执行性。

### 6.4 Case Study（定性生产轨迹）

追踪一个匿名生产请求穿过 DREAM 模块边界：
- **Intent Engine** 输出 14 个并发意图卡（P5/P4/P3/P2 等），而非单一合并意图。展开的 P3 卡描述"以棒球帽和偏光/户外太阳镜为锚的灵感探索"，需求收敛中、中等置信度，保留新手品类认知、遮阳帽需求、夏季/夜驾场景、功能/版型/材质偏好。P2 卡捕获男休闲装收敛中新手需求（通勤/休闲/夏季/户外，复古/冰丝/扎染/透气/宽松工装/黑红印花偏好）。
- **MetaModel** 收到完整卡集后才开始：M1 输出 `intent_summary` 和 `category_preference`——识别运动/休闲鞋为主导意图（竞品选择/比价犹豫阶段），遮阳帽/男表/男休闲装/咖啡代理点单为次要意图，总结透气/防滑偏好和 400–520 预算。**主导/次要分配是 MetaModel 的判断，而非 Intent Engine 字段**。
- M2 结合用户分层（购买力 K1、活跃 a4，但聚合交易行为全零→用户级转化信号判无效）。因 P5 鞋意图已在竞品选择且市场级 item 效率高，MetaModel 选 IPV-oriented 目标。参数化为稀疏控制 bundle：`IPV=+2, CTR=+1, GMV=CVR=0`；视频 +2；选定 card type 和品类 +1/+2；体验约束有界调整（购买过滤 -1、密布局偏好 +1）。
- **重排/排序流水线** 应用有界覆写返回商品序列。蓝卡匹配鞋意图、绿卡匹配帽/眼镜意图、紫卡匹配 P2 男休闲装意图、灰卡无匹配——颜色编码表语义对齐，非对单控制参数的因果归因。

---

## 7. Prompt 与可控参数细节

### 7.1 Intent Engine Prompt

角色：电商意图平台核心意图理解引擎。输入模块全可选（User Data / On-device Behavior / Historical Intent），仅在提供模块上推理。关键推理规则：
- **信号优先级**：实时行为 > 短期 > 长期 > 静态属性。
- **历史意图融合**：新信号同意则增强置信+精化偏好；冲突则从新证据并降权历史；证据不足则保留历史但降置信；本轮无变化则不发 `update`。
- **意图类型**：`goal-driven`（明确品类目标+购买动机）/ `inspiration exploration`（松散定义域内广泛浏览）/ `aimless browsing`（随机非收敛、短停留、无搜索购物车收藏）。
- **决策状态**（10 态）：demand vague / converging / competitor comparison / price deliberation / decision bottleneck / about-to-buy / post-purchase exit / post-purchase verification / active abandonment / demand suspension。购买后某品类意图只能进入 post-purchase 两态。
- **路由自评**：先产同步 `actions`，再评是否需异步精化（不阻塞当前输出）。`route.tool` ∈ {none, context_subagent, expert}，`route_confidence` ∈ [0,1] ∪ {null}。
- **输出协议**：严格 JSON，`{"actions":[...], "route":{...}}`。`insert` 用于新意图（完整 intent 对象列表），`update` 按 `intent_id` 匹配（dot-path 键表示变更字段）。

### 7.2 Meta Engine Prompt

角色：`MetaBrain`，编排首页 Feed 推荐流水线的元模型。输入两段：
- **env_state**：事后 Feed 指标（1/3/7 日窗），6 种 card type（product/ad/universal_push/short_video/live/content）的市场级与用户级份额与效率；`a>b>c` 表升势，`a<b<c` 表降势；用户级稀疏时回退市场级。
- **user_intent**：三层结构化意图 JSON（L0 稳定画像 activity∈{a1..a5}/purchasing_power∈{K1..K6}；L1 per-intent 需求品类/置信度/意图类型；L2 细粒度 brand_tendency/price_preference/realtime_psychology/decision_state）。

输出策略 bundle：1 个 M1 模块 + 6 个 M2 模块（固定顺序），值均取自指定枚举。
- **M1**：`decision_rationale`（≤100 字符链式思维锚，须为首个键）、`strategy_orientation` ∈ {IPV-oriented, GMV-oriented}、`purchasing_power` ∈ {K1..K6}、`activity` ∈ {a1..a5}。
- **M2 六模块**（级别语义：+2 最强提升 / +1 弱提升 / 0 中性 / -1 弱抑制 / -2 最强抑制）：
  1. `intent_summary`（意图转发）
  2. `ranking_weight_boost`（目标偏好）：ctr/ipv/cvr/gmv_boost_ratio ∈ {-2..2}，调打分公式 $\alpha$ 权重。
  3. `cardtype_prefer`（属性偏好）：auction/ad/quanzhantui/video/live/pgc 各 ∈ {-2..2}，全 6 键必给、≥3 非零、≥1 负。
  4. `category_prefer`（属性偏好）：`"<真实品类名>": int`，≤6 键。
  5. `experience_constraints`（属性偏好，+ 收紧 / - 放松）：exposure_filter / purchase_filter / density_perception ∈ {-2..2}。
  6. `top_ctr_strategy`（策略控制，二元开关）：page0/pagen ∈ {0,1}，启用目标位纯 CTR 排序，**覆盖**模块 2 的 $\alpha$ 加权公式。

**关键决策原则**：每模块融合 intent-driven 信号（`user_intent`）与 post-hoc 信号（`env_state`），用户级优先、市场级回退。早期意图→IPV-oriented；晚期（about-to-buy、decision-stuck）→GMV-oriented。IPV-oriented 提升 {ctr,ipv} 抑制 {cvr,gmv}，GMV-oriented 反之。无信号→{0,0,0,0}。

### 7.3 可控参数空间（执行层）

执行层把 Meta Engine 决策送达在线流水线而不替换任何模块。基线配置保稳定，M3 参数翻译模块按请求选择性覆写一部分参数，全受安全护栏约束。

| 阶段 | 模块 | 控制机制 |
|------|------|----------|
| 召回 | BE Chain | 召回路径开关 |
| 召回 | Plan/Category Quota | 跨 plan 和品类重分配配额 |
| 召回 | Ranking Formula | 调 CTR/IPV/CVR/GMV 项权重 |
| 混合 | Natural Blending Alpha | 混合公式各目标权重 |
| 混合 | Global-Push Alpha Hat | 非全局推桶的 alpha 乘子 |
| 混合 | Extreme-CTR Strategy | 槽位公式 CTR 激进/均衡切换 |
| 混合 | PVR Control Weighting | 广告与 boost 控分合并系数 |
| Boosting | Boost Weighting | 指定 item/plan 加权 boost 分 |
| 体验 | Category Diversity | 同品类卡最小散列间隔 |
| 体验 | CSR Filtering | 低于 per-card-type 满意度阈值卡过滤 |
| 体验 | DTR Filtering | 高于 per-card-type 厌恶阈值卡过滤 |
| 体验 | Personalized Fatigue | 疲劳衰减系数 |
| 体验 | Vector Scatter | embedding 相似度间距阈值 |
| 体验 | Rule-based Scatter | 按 card type/brand/seller/category 间距 |
| 重排 | Generator | 注入额外 generator 模块 |

参数横跨相关性旋钮（召回路径、排序权重、混合 alpha）与体验旋钮（多样性、疲劳、过滤）。MetaModel 不同时操控全部——给定当前用户意图，M2 选对齐策略导向的稀疏子集，M3 翻译为具体值。每次覆写增量且有界，主流水线仍是安全网，DREAM 干预可审计可回滚。

---

## 8. 核心贡献与启示

1. **Agentic 元控制范式可行**：在工业推荐上验证了"在既有流水线之上叠加感知-决策-执行策略层"而非替换任何模型。不替换召回/排序/重排模型、不损害服务稳定性，取得 IPV +2.71%、Core IPV +3.06%、GMV +1.31% 的累计收益。
2. **意图感知与策略生成首次端到端贯通**：三层 Intent Engine 把端侧多源信号融合为结构化 L0/L1/L2 意图，端云触发链把上报量压到约 8.7%；Meta Engine 用 M1→M2→M3 分层推理把意图翻译为有界可执行参数，弥补了既有 agentic 推荐"感知与决策脱节"的缺口。
3. **闭环反馈从执行回到上游优化**：Reward Dual Loop 把离线仿真探索与在线真实反馈校准耦合，"策略生成→执行→评估→经验积累→再生成"持续循环，Strategy Memory 索引于 M1 分段，正向经验作参考、负向作约束，弥补了既有方案"无闭环反馈"的缺口。
4. **控制面扩大收益复利累积**：从仅重排到重排+精排，IPV/Core IPV/GMV 均进一步显著提升，PV 几乎不变（说明增益来自参与度与转化而非额外曝光），证明 agentic 元控制收益随控制面扩大而复利。
5. **安全可审计的工程范式**：默认回退 + 个性化覆写 + 四道校验门 + 字段级白名单 + 范围约束，每次覆写增量有界可回滚；LLM 不输出 item ID 或最终排列，只输出语义级策略 bundle 再确定性编译为参数。

### 与相关工作的差异定位

- 对比 **AgentX**（同为阿里系 agentic 推荐）：AgentX 是**研发流程**的 agent 自迭代（Brainstorm→Developing→Evaluation + SGPO harness 演进），优化的是"如何更快产出实验";DREAM 是**在线服务**的 agent 元控制，优化的是"运行时如何感知意图并编排策略"，二者层级不同。
- 对比传统 CTR/CVR 单目标排序：DREAM 用 MetaModel 在 IPV vs GMV、转化 vs 体验、探索 vs 相关间显式仲裁，而非把所有目标揉进一次打分。
- 对比纯端侧意图模型：DREAM 用端云 F1–F4 漏斗 + 多尺度模型（0.8B 同步 + 4B 异步精化 + 4B 夜间 Dreaming）+ on-policy distillation 自进化，在成本与质量间取得平衡。

---

## 9. 总结

DREAM 把推荐系统从"模块化但割裂"的级联流水线升级为"模块化 + 感知-决策-执行策略层"的自主优化控制架构。Intent Engine 解决"此刻如何理解用户"，Meta Engine 解决"用这个理解做什么、如何落地"，Reward Dual Loop 解决"如何持续自进化"。三者构成感知→决策→执行的闭环，行为信号上行、意图表示横流、策略参数下行。淘宝首页 Feed 大规模 A/B 验证了 agentic 元控制作为工业推荐优化实用范式的可行性，且收益随控制面扩大而复利累积。

---

## 10. 讨论问答

### Q1：用一个例子从推理角度说明本文的流程

**场景设定**：28 岁男性用户，购买力 K3，活跃度 a3，健身爱好者。会话行为：首页浏览运动周边 → 搜索"跑步鞋" → 点击 Nike Pegasus（停留 45s，深度滚动，放大看鞋底）→ 返回点击 Adidas Ultraboost（停留 30s）→ 看价格后返回首页 Feed。

#### 第 1 阶段：Traffic Funnel（端侧推理——"这个行为值得上报吗？"）

**F1 信号编码**：设备把原始追踪点编码为结构化特征，行为词汇从 6 种扩展到 60+ 种，保留细粒度动作轨迹。

**F2 触发判定（GRU 推理）**：单层 GRU 在 7 维离散特征流上做变点检测，隐藏态跨事件传递。核心问题——"用户意图是否变化？"行为序列从"泛浏览"切换到"定向搜索+深度比价"→ 变化点命中。误触保护：dwell>2s 且非曝光前返回，排除误触。决策：触发上报（此门仅放行约 15% 行为）。

**F3 最小上报**：打包近 50 事件为 ID 编码行为包（只带 item_id/shop_id/scene_id，不带语义），最小化带宽。

**F4 云端富化与准入（云端推理）**：恢复语义（item_id→"Nike Air Zoom Pegasus 40"）+ join 最近意图列表和 L0 画像 → 准入推理："跨品牌深度比价是实质性意图变化"→ 通过，调用 Intent Reasoning Core。

**整体效果**：约 8.7% 的行为最终触发云端 LLM 推理。

#### 第 2 阶段：Intent Reasoning Core（意图推理——"用户现在到底要什么？"）

**Main Agent（0.8B 同步推理）** 收到 $(P_{L0}, \mathcal{I}_{t-1}, \mathcal{B}_t)$，产出 $(\Delta\mathcal{I}_t, r_t)$。推理链：

| 推理步骤 | 推理内容 |
|----------|----------|
| 信号优先级判定 | 实时搜索+深度点击 > 长期兴趣 > 静态画像；实时证据权重最高 |
| 意图类型判定 | 有明确品类目标+购买动机 → `goal-driven` |
| 决策状态判定 | 跨品牌集中浏览但未加购 → `competitor comparison` |
| 实时心理推断 | Adidas 只停 30s 返回 + 行为集中在看价格 → `price deliberation` |
| 多意图处理 | 搜索+点击集中在 running_shoes 一个品类簇 → 合并为单一意图 |
| 置信度评估 | 多次搜索+集中点击 5 款 → high |
| 优先级赋值 | 强实时证据+晚决策阶段 → priority=4 |

输出 $\Delta\mathcal{I}_t$（增量 insert，非重写全表）：L1 需求品类=running shoes / confidence=high；L2 subcategory=Nike Pegasus/Adidas Ultraboost / brand_tendency=Nike>Adidas / decision_state=competitor comparison / realtime_psychology=price deliberation；**price_preference=null**（证据不足不臆测，触发异步精化信号）。

输出 $r_t$：`{tool: "expert", route_confidence: 0.82}`——推理："L1 品类已确定但 L2 item 级字段仍不确定 → 需专家 agent 用更长上下文精化。" 不阻塞当前输出，立即返回下游。

**双层路由判定**：规则层（行为数 5 未超阈、品类数 1 未超阈）不触发；模型层（0.82>τ=0.6）触发升级 → 送入异步 4B expert（约 6.3% 请求走此路）。

**异步精化（4B expert 推理）**：用更长历史+跨会话上下文+L0 画像精化 item 级字段。推理："过去三个月点击过的跑鞋价格集中在 400-600 元 → price_preference=mid-range"；"放大看鞋底 → 关注缓震 → attribute_preference 含 cushioning"。字段级合并：实时点击证据支持的字段优先保留；异步估计字段仅当置信度超阈值才更新。下次请求的 $\mathcal{I}_{t-1}$ 已含精化结果。

#### 第 3 阶段：频率控制（Meta Engine 触发推理——"现在值得调 MetaModel 吗？"）

核心问题：刷新策略的期望价值是否值得其成本？

**触发信号计算**：$S^{drift}=0.85$（品类分布漂移大）、$S^{act}=0.65$（中高活跃）、$S^{time}=0.70$（晚高峰）；融合 $S_{u,t}=0.74$。

**动态阈值**：$\theta_{u,t}=\mathrm{clip}(\theta_0+\delta_{g(u)}+\delta_{b(t)}+\delta_{s(t)}+\delta_{d(t)})=0.60+(-0.05)+0.10+0+0=0.65$（中等价值用户略降门槛，晚高峰略升门槛抑制低价值调用）。

**决策**：$a_{u,t}=m_{u,t}\cdot\mathds{1}[0.74\geq 0.65]=1$ → 调用 MetaModel。资格掩码检查（不在冷却、未超上限）通过。

#### 第 4 阶段：MetaModel 分层推理（M1→M2→M3——"系统该做什么？"）

MetaModel（基于 Qwen3）收到结构化意图 + env_state。

**M1 意图总结与全局导向**：推理——主导意图=running shoes（priority=4 最高）；目标导向判定：decision_state="competitor comparison"（晚期决策阶段，接近购买）→ 按 prompt 规则选 **GMV-oriented**（早期→IPV 多看，晚期→GMV 促转化）。输出：`{strategy_orientation: "GMV-oriented", purchasing_power: "K3", activity: "a3"}`。

**M2 策略规划**：检索 Strategy Memory（按 M1 分段索引）——"competitor-comparison + K3 + GMV-oriented"的正向经验：CVR/GMV boost 策略曾在 78% replay 中胜出。生成 6 模块（遵循"GMV-oriented 提升 {cvr,gmv} 抑制 {ctr,ipv}"）：

| 模块 | 推理 | 输出 |
|------|------|------|
| `intent_summary` | 主导意图概括 | "跑步鞋比选，Nike偏好，400-600元" |
| `ranking_weight_boost` | GMV-oriented | `{ctr:-1, ipv:-1, cvr:+2, gmv:+2}` |
| `cardtype_prefer` | 促转化→强推商品卡，抑视频/直播 | `{auction:+2, video:-2, live:-2, ...}` |
| `category_prefer` | 集中 running shoes | `{"running shoes":+2, "sports apparel":+1}` |
| `experience_constraints` | 比选阶段需多样性 | `{density_perception:-1}`（放松密度） |
| `top_ctr_strategy` | GMV+晚期→强制关闭 | `{page0:0, pagen:0}` |

**M3 参数翻译（确定性编译，非 LLM 推理）**：M3 是 DREAM 的关键设计——把 LLM 的不确定性限制在 M2 语义层，M3 是可审计的确定性编译。

| M2 语义 | M3 编译 | 生产参数 |
|---------|---------|----------|
| `ranking_weight_boost.cvr=+2` | $\delta_{cvr}=w_{cvr}^0\times 2$ | `ltr_cvr_delta=0.08` |
| `category_prefer["running shoes"]=+2` | 散列窗 $n_c^{new}=\max(1,n^{default}+2)$ | `metamodel_boost_cate=["running shoes"]` |
| `cardtype_prefer.video=-2` | 硬抑制 | `metamodel_adjust_cardtype=video` |

四道校验门（JSON 解析→schema 校验→白名单→范围校验）全通过。

#### 第 5 阶段：Unified Outlet 执行（"安全地注入流水线"）

请求到达时 Tool Process 读缓存 bundle：$p_s^{\mathrm{exec}}=\mathrm{Guard}_s(p_s^0\oplus\mathcal{T}_s(a_u))$。$\oplus$ 是**局部覆写**非全配置替换——bundle 缺失/过期/畸形则默认参数不变，主流水线始终是安全网。
- **召回**：扩大 running shoes 召回配额，抑制视频召回
- **排序**：$f_{rank}=f_{ltr}\times(1+\delta_{cvr}\hat{v}_{cvr})\times(1+\delta_{gmv}\hat{v}_{gmv})$ → 提升 CVR/GMV 预测分高的 item
- **重排**：放松品类密度约束，应用曝光过滤

用户最终看到：Nike Pegasus 和相关跑鞋更显著展示，视频/直播减少，跑鞋候选更集中。

#### 第 6 阶段：反馈闭环（"从结果中学习"）

**在线环**：用户点击 Nike Pegasus 并加购 → 观测 IPV/CVR/GMV 信号 → 与产生策略对照 → 验证结论存入 Strategy Memory："competitor-comparison + K3 + GMV-oriented + cvr/gmv boost → 正向"（下次被 M2 检索）。

**闭环反馈到 Intent Engine**：用户加购 → Intent Engine 更新 decision_state 从 "competitor comparison" → "about-to-buy"，置信度提升。

**夜间 Dreaming（离线整合）**：4B 模型处理全天行为轨迹，对意图列表做六操作——`enrich`（补全 price_preference）、`correct`（修正 brand_tendency 误判）、`keep`（意图仍活跃则保留）、`kill`（若已购买则移除）。第二天首请求的 $\mathcal{I}_{t-1}$ 已是整合后的干净状态。

#### 推理角度的核心洞察

从这个例子看出 DREAM 推理链的三个分层设计哲学：

1. **推理不确定性分层隔离**：端侧 GRU（F2）做"是否值得上报"的低成本推理 → 0.8B Main Agent 做"用户要什么"的同步推理 → 4B expert 做"细节偏好"的异步精化推理 → MetaModel 做"系统该做什么"的策略推理 → M3 做"参数是什么"的**确定性编译**（无推理，可审计）。每层推理的不确定性被限制在该层，不污染下游——这是 DREAM 工程稳定性的关键。

2. **推理成本与价值的动态匹配**：F2 GRU 把 85% 平凡行为挡在端侧；双层路由把 93.7% 请求留在 0.8B；频率控制把低价值 MetaModel 调用挡掉；Dreaming 把重推理挪到夜间闲置算力。推理资源始终流向"期望价值>成本"的时刻。

3. **推理结果始终增量且有界**：Intent Engine 只发 insert/update 增量；MetaModel 输出受 schema 枚举约束（{-2..2}）；Unified Outlet 只做局部覆写，默认配置是安全网；每次覆写过四道校验门，失败即回退。这让 DREAM 可以在不替换任何既有模型的前提下叠加智能层——这是它能上线的根本原因。

### Q2：是否可以认为本文是通过大模型判断用户意图，然后通过 MetaModel 产生出不同目标的权重？

**结论**：这个理解抓住了主干（Intent Engine 用 LLM 产出意图、MetaModel 含目标权重产出），但有三处偏差，恰好对应了 DREAM 与"传统 LLM 加权推荐"的关键区别。

#### 正确部分

- ✅ "通过大模型判断用户意图"——Intent Engine 的 Main Agent（0.8B）+ 异步 expert（4B）+ Dreaming（4B）都是 LLM 推理，产出结构化意图。
- ✅ "通过 MetaModel 产生不同目标的权重"——MetaModel 的 M2 模块中确实有 `ranking_weight_boost`，产出 CTR/IPV/CVR/GMV 权重调整（如 `{ctr:-1, ipv:-1, cvr:+2, gmv:+2}`）。

#### 三处需要修正的偏差

**偏差 1：意图不是"一个"，而是"三层结构化 + 多并发列表"**

"判断用户意图"听起来像产出一个单一标签。但 DREAM 输出的是：三层结构化（L0 静态画像 + L1 需求层 + L2 偏好层）+ 多并发意图列表（一个用户同时有多个活跃意图，Case Study 中用户同时有 14 个并发意图卡 P5/P4/P3/P2...，不是合并成一个）。更准确的说法是"把多源行为信号融合为三层结构化、多并发的意图状态"。

**偏差 2：MetaModel 不只产生"目标权重"，而是 6 个维度的策略 bundle**

`ranking_weight_boost`（目标权重）只是 M2 的 6 个模块之一。MetaModel 还产出 `intent_summary`（意图转发）、`cardtype_prefer`（卡片类型偏好）、`category_prefer`（品类偏好）、`experience_constraints`（体验约束）、`top_ctr_strategy`（排序策略开关）、`business_support`（业务白名单）。所以 MetaModel 是在多个语义维度上做策略规划，"目标权重"只是其中一个维度。

**偏差 3：遗漏了三件让 DREAM 能上线的关键机制**

描述"LLM 判意图 → MetaModel 出权重"像一个一次性的开环前向推理，但 DREAM 之所以能上淘宝首页 Feed，靠的是三个被省略的机制：

- **(a) 频率控制**：不是每次请求都调 MetaModel。频率控制用 Lagrangian 决策 $a_{u,t}=m_{u,t}\cdot\mathds{1}[S_{u,t}\geq\theta_{u,t}]$ 把低价值调用挡掉。没有这层 LLM 推理成本会爆炸。
- **(b) M3 确定性编译 + Unified Outlet 局部覆写**：LLM 不直接控制流水线。M2（LLM 输出）是语义级（如 `cvr:+2`）；M3 是确定性编译（如 $\delta_{cvr}=w_{cvr}^0\times 2=0.08$），不是 LLM 推理；Unified Outlet 用 $p_s^{\mathrm{exec}}=\mathrm{Guard}_s(p_s^0\oplus\mathcal{T}_s(a_u))$ 做局部覆写，默认配置是安全网；四道校验门失败即回退。LLM 的不确定性被限制在 M2 语义层，不直接接触生产参数——这是可审计可回滚的关键。
- **(c) Reward Dual Loop**：不是开环，是闭环。在线环把策略执行后的 IPV/CVR/GMV 反馈对照策略，验证结论存入 Strategy Memory；离线环用 Evaluator 在 logged 上下文上 replay 候选 bundle 探索策略空间；M2 检索 Strategy Memory 历史经验（正向作参考、负向作约束），不是从零推理。

#### 更准确的表述

> ~~通过大模型判断用户意图，然后通过 MetaModel 产生出不同目标的权重~~
>
> 通过**多尺度 LLM**（端侧 GRU 触发判定 + 0.8B Main Agent 同步 + 4B expert 异步精化 + 4B 夜间 Dreaming 整合）将端云多源行为信号融合为**三层结构化、多并发的意图状态**；在**频率控制**判定调用值得后，MetaModel **检索 Strategy Memory 历史经验**，基于意图状态 + 事后 Feed 指标，在**6 个语义维度**（目标权重只是其一）上做**有界、可审计的策略规划**；经**确定性 M3 翻译**为生产参数，通过**局部覆写 + 四道校验门**注入既有流水线；通过**在线+离线双环反馈**持续自进化策略与意图理解。

#### 与"传统 LLM 加权推荐"的本质区别

"LLM 判意图 → 出目标权重"恰好是传统 LLM 加权推荐的范式。DREAM 与之的根本区别：

| 维度 | 传统 LLM 加权 | DREAM |
|------|---------------|-------|
| 意图表示 | 单一意图标签 | 三层结构化 + 多并发意图列表 |
| 策略产出 | 仅目标权重 | 6 维度策略 bundle |
| 调用频率 | 每次请求都调 | 频率控制筛选（约 8.7% 行为触发） |
| LLM 与流水线关系 | 直接控制 | M2 语义层 + M3 确定性编译 + 局部覆写 |
| 反馈 | 开环 | 在线+离线双环自进化 |
| 经验积累 | 无 | Strategy Memory 按 M1 分段索引 |
| 安全性 | 依赖 LLM 稳定性 | 默认回退 + 四道校验门 |

**本质**：DREAM 不是"用 LLM 替代规则做加权"，而是"在既有流水线之上叠加一个感知-决策-执行策略层，让 LLM 在受约束的语义空间做策略规划，确定性编译后局部注入"。目标权重只是这个策略层诸多产出维度之一，把它等同于 DREAM 的全部会掩盖其作为"自主优化控制架构"的核心定位。

### Q3：本文工作除了目标权重，还会影响到什么？6 个维度都会影响吗？都举个具体例子

**结论**：6 个维度都会影响，且每个维度影响**不同的下游阶段和参数**。但不是每次都全部激活——MetaModel 根据当前意图状态选择对齐策略导向的**稀疏子集**。

#### 总览：6 个维度各自影响什么

| M2 维度 | 影响阶段 | 翻译成的生产参数 |
|---------|----------|-----------------|
| `ranking_weight_boost` | 排序 + 重排 | `ltr_ctr_delta` 等乘性校正系数 |
| `cardtype_prefer` | 召回 + 重排 | `metamodel_adjust_cardtype=video` 召回抑制指令 |
| `category_prefer` | 召回 + 排序 + 重排 | `metamodel_boost_cate` 召回配额 + 散列窗 $n_c$ + 品类构成 |
| `experience_constraints` | 重排 | 曝光过滤阈值 + 购买过滤 + 密度散列间隔 |
| `top_ctr_strategy` | 重排 | `page0/pagen` 分页纯 CTR 排序开关（**覆盖** ranking_weight_boost） |
| `intent_summary` | 策略自适应模块 | 轻量 LLM 生成的个性化召回通道开关/配额参数 |
| (补充) `business_support` | 排序截断 | `guaranteed_plan_ids` 绕过截断配额 |

#### 维度 1：`ranking_weight_boost`（目标权重）

**场景**：用户 about-to-buy 决策状态，MetaModel 判定 GMV-oriented。
**M2**：`{ctr:-1, ipv:-1, cvr:+2, gmv:+2}`
**M3**：$\delta_{cvr}=w_{cvr}^0\times 2=0.08$，$\delta_{gmv}=w_{gmv}^0\times 2=0.12$
**排序公式**：$f_{rank}=f_{ltr}\times(1+0.08\cdot\hat{v}_{cvr})\times(1+0.12\cdot\hat{v}_{gmv})$
**效果**：CVR/GMV 预测分高的 item 排名上升，CTR/IPV 高但转化弱的 item 排名下降。用户看到的列表更偏向"容易买"而非"容易点"。

#### 维度 2：`cardtype_prefer`（卡片类型偏好）

**场景**：用户在跑步鞋深度比价阶段（competitor comparison），注意力集中在商品详情，不需要视频/直播干扰。
**M2**：`{auction:+2, ad:0, video:-2, live:-2, pgc:0}`
**M3**：video=-2 → 硬抑制指令 `metamodel_adjust_cardtype=video` → **召回服务直接不召回视频**；live=-2 → 同样抑制直播召回；auction=+2 → 重排阶段商品卡权重提升。
**效果**：Feed 中视频和直播内容显著减少，商品卡占比上升。这是**召回源头**的调整，不是重排时过滤——拦截更靠前，节省下游算力。

#### 维度 3：`category_prefer`（品类偏好）——影响阶段最多

**唯一同时影响召回、排序、重排三个阶段**的维度，因为同一语义偏好在不同阶段有不同操作含义。

**场景**：用户同时有多个并发意图（跑步鞋 P4 + 运动袜 P3 + 运动水壶 P2），MetaModel 识别跑步鞋为主导。
**M2**：`{"running shoes":+2, "sports socks":+1, "water bottle":+1}`
**M3（三阶段不同操作）**：

| 阶段 | 操作 | 参数 |
|------|------|------|
| 召回 | 提取品类标签序列化为 `metamodel_boost_cate`，用 `metamodel-boost` 队列保护这些品类的召回量 | 召回配额增加 |
| 排序 | 语义比 $b_c=+2$ → 散列窗 $n_c^{new}=\max(1, n_c^{default}+2)$ → 放松品类散列窗，允许更多跑鞋进入截断前候选 | scatter=3 |
| 重排 | 调整品类构成，跑鞋占比提升 | 品类配额 |

**效果**：同一"跑步鞋偏好"语义，在召回端保护供给量、在排序端放松散列让更多候选通过截断、在重排端调整最终品类构成。**语义偏好共享，底层参数不共享**——这是 DREAM 设计的精妙之处。

#### 维度 4：`experience_constraints`（体验约束）

**场景**：用户 post-purchase verification 状态（刚买了跑鞋，在验证决策）。
**M2**：`{exposure_filter:+1, purchase_filter:+2, density_perception:0}`
**M3**：purchase_filter=+2 → 收紧购买过滤（已购买的跑鞋及相关 SKU 强过滤，避免重复推荐刚买过的商品）；exposure_filter=+1 → 收紧曝光过滤（已曝光多次的 item 阈值降低，减少重复曝光疲劳）。
**效果**：用户不会看到刚买的同款跑鞋，也不会看到已经曝光过多次的商品。这是**体验层**的调整，直接影响重排的过滤逻辑。

**反向例子**：如果是 inspiration exploration 状态（开放探索），MetaModel 可能输出 `density_perception:-1`（放松密度），让用户看到更多样化的候选，即使品类分散度较高。

#### 维度 5：`top_ctr_strategy`（排序策略开关）——能覆盖维度 1

这个维度最特殊，是一个**二元开关**，且**启用时会覆盖 `ranking_weight_boost` 的 $\alpha$ 加权公式**。

**场景**：用户在 aimless browsing 状态（漫无目的浏览），MetaModel 判定 IPV-oriented（多看多激发兴趣）。
**M2**：`{page0:1, pagen:0}`
**M3**：page0=1 → 第一页 top 位置启用纯 CTR 排序，覆盖 M2 模块 2 的 $\alpha$ 加权公式；pagen=0 → 后续页恢复正常加权。
**效果**：第一页头部位置全部展示 CTR 最高的 item（最吸引点击的），不管 CVR/GMV 如何。这是为了在用户漫无目的时先用高吸引力内容抓住注意力，后续页再用正常加权平衡转化。
**关键点**：如果 `top_ctr_strategy.page0=1`，那么维度 1 的 `ranking_weight_boost` 在第一页 top 位置**被覆盖失效**。这体现了 M2 模块间的优先级关系。

#### 维度 6：`intent_summary`（意图转发）——不直接变参数

这个维度最容易被忽略，它**不直接翻译成生产参数**，而是作为语义上下文喂给下游。

**场景**：用户意图 summary = "为下周城市马拉松训练采购，主导意图跑步鞋（Nike 偏好，400-600 元），次要意图运动袜和能量胶"。
**消费方式**：这个 summary 被喂给**推荐策略自适应模块**（轻量 LLM 推理），生成个性化参数：召回通道开关（开启"运动品类专属召回通道"）、召回配额（扩大运动品类召回配额）、多样化参数（调整多样性参数允许更多运动周边品类）。
**效果**：intent_summary 是一个"软影响"——它通过下游 LLM 推理间接变成参数，而非像其他 5 个维度那样确定性编译。这也是为什么 DREAM 的 M2 输出既有"硬"维度（直接编译）又有"软"维度（喂给下游推理）。

#### 补充维度：`business_support`（业务白名单）

**场景**：双 11 大促期间，某品牌 PlanID 需要业务保护流量。
**M2**：`{plan_ids: ["PLAN_12345"]}`
**M3**：→ `guaranteed_plan_ids=["PLAN_12345"]`
**效果**：关联该 PlanID 的 item 可绕过正常截断配额，至上限 `business_support_max_num`。即使排序分不够高，也能保住曝光位。这是业务策略与算法策略的协调通道。

#### 关键洞察：不是每次都全部激活

MetaModel **不同时操控全部**维度——给定当前用户意图，M2 选对齐策略导向的**稀疏子集**，M3 翻译为具体值。每个请求只激活对齐当前意图状态的稀疏子集，其余维度保持默认（0 或不输出）：

| 用户状态 | 激活的维度子集 |
|----------|----------------|
| about-to-buy + GMV-oriented | `ranking_weight_boost` + `business_support`（促转化） |
| competitor comparison | `category_prefer` + `cardtype_prefer`（集中候选） |
| aimless browsing + IPV-oriented | `top_ctr_strategy` + `experience_constraints:-1`（抓注意力+放松多样性） |
| post-purchase | `experience_constraints:+2`（强过滤已买） |

这是 DREAM 控制"策略干预量"的关键——不是每次都大改，而是精准地改需要改的部分。

#### 与"目标权重"的根本区别

如果 DREAM 只产出目标权重，它的影响范围仅限于**排序公式**。但实际上 6 个维度的影响范围覆盖了：

- `cardtype_prefer` 影响**召回源头**（拦截在最早阶段）
- `category_prefer` 影响**三个阶段**（召回配额 + 排序散列 + 重排构成）
- `experience_constraints` 影响**重排过滤逻辑**（用户体验）
- `top_ctr_strategy` 能**覆盖**目标权重（模块间优先级）
- `intent_summary` 影响**策略自适应**（间接生成通道开关）
- `business_support` 影响**排序截断**（业务保护）

这些维度的**组合**才构成了"元控制"——它不是在调一个权重旋钮，而是在**多个语义维度上协同调整整个流水线的行为**。目标权重只是其中一个旋钮，把它等同于 DREAM 的全部，就像把"方向盘"等同于"整辆车的控制系统"——忽略了油门、刹车、档位、转向灯的协同。

### Q4：相当于先用 LLM 分析用户意图，然后触发不同的排序逻辑，在召回、排序、重排阶段进行影响？

**结论**："意图 → 多阶段影响"的骨架正确，但有一个关键用词偏差需要修正——不是"触发不同的排序逻辑"，而是"调整既有排序逻辑的参数"。这恰好触及 DREAM 与"模块替换"的根本分界线。

#### 关键修正：不是"触发不同排序逻辑"，而是"调整既有逻辑的参数"

"触发不同的排序逻辑"暗示 LLM 在多套预定义排序算法间选择/切换。但 DREAM 明确**不替换任何模块的逻辑**，它做的是在既有逻辑内部调整参数：

| 表述 | 实际情况 |
|------|----------|
| 触发**不同的**排序逻辑 | 调整**既有**排序逻辑的**参数** |
| 在多套算法间选择 | 在同一套算法内做局部覆写 |
| 模块级切换 | 参数级覆写 |

对比：
- ❌ "LLM 判定 GMV-oriented → **切换到** GMV 排序算法"（这是替换模块）
- ✅ "LLM 判定 GMV-oriented → 在**既有** LTR 排序公式上**乘以** $(1+\delta_{cvr}\hat{v}_{cvr})$ 这个校正项"（这是参数覆写）

排序服务跑的还是原来那个 LTR 模型、原来那个重排 Generator——DREAM 只是往里注入增量参数。基线配置 $p^0$ 始终是安全网，覆写失败就回退到 $p^0$。

#### 微妙细节：参数覆写有连续和离散两种

虽然都是"调参数"而非"换逻辑"，但 6 个维度里混合了两种参数调整方式：

| 类型 | 维度 | 例子 |
|------|------|------|
| **连续参数调整** | `ranking_weight_boost`、`category_prefer`(散列窗)、`experience_constraints` | cvr 系数从 0 → 0.08，散列窗从 1 → 3 |
| **离散开关/硬抑制** | `top_ctr_strategy`、`cardtype_prefer`(video=-2) | page0=1 开关，video 召回开关 |

所以更精确的说法是：DREAM 产出**连续参数调整 + 离散开关**的混合 bundle，但**两者都是参数级覆写，不是逻辑级替换**。即使是 `top_ctr_strategy` 这个二元开关，也是作为 flag 传给**既有**重排 Generator（"启用你的纯 CTR 模式"），而非换成另一个重排算法。

#### 另一个需要补充的点：是两个 LLM，不是一个

"用 LLM 分析用户意图"实际上涉及两个独立的 LLM 推理步骤，职责不同：

| LLM | 模型 | 职责 | 产出 |
|-----|------|------|------|
| Intent Engine LLM | 0.8B Main + 4B expert + 4B Dreaming | 理解"用户要什么" | 三层结构化意图 |
| MetaModel LLM | Qwen3 | 决定"系统该做什么" | 6 维度策略 bundle |

意图分析（Intent Engine）和策略规划（MetaModel）是**两次独立的 LLM 推理**，中间还隔着**频率控制**（决定是否值得调第二个 LLM）。不是"一个 LLM 既分析意图又出策略"。

#### 修正后的准确表述

> ~~先用 LLM 分析用户意图，然后触发不同的排序逻辑，在召回、排序、重排阶段进行影响~~
>
> 先用 **Intent Engine LLM**（0.8B+4B）将行为信号融合为**三层结构化意图**；经**频率控制**筛选后，**MetaModel LLM**（Qwen3）检索 Strategy Memory，基于意图状态在 6 个语义维度上做策略规划；经**确定性 M3 翻译**为**连续参数调整 + 离散开关**的混合 bundle，通过**局部覆写**注入**既有**召回/排序/重排逻辑（**不替换任何模块**），默认配置始终是安全网。

#### 比喻：DREAM 是 overlay 不是 replacement

这是 DREAM 作为 **overlay（叠加层）** 而非 **replacement（替换层）** 的根本特征——也是它能不损害服务稳定性地上线的工程根基。如果把 DREAM 理解为"触发不同的排序逻辑"，就等于把它当作 replacement，会掩盖其"在不替换任何既有模型的前提下叠加智能层"的核心设计哲学。

用户三次提问的理解演进正好对应 DREAM 设计的三个层次：

| 提问 | 理解 | 对应的 DREAM 层次 |
|------|------|-------------------|
| Q2 | LLM 判意图 → 出目标权重 | 仅 `ranking_weight_boost` 一个维度 |
| Q3 | → 影响 6 个维度，覆盖召回/排序/重排 | 完整 M2 策略 bundle |
| Q4 | → 触发不同排序逻辑 | ❌ 仍差"参数覆写 vs 逻辑替换"这层 |
| Q4 修正后 | → 调整既有逻辑的参数（overlay） | ✅ 完整理解 DREAM 的 overlay 定位 |



### Q5：Reward Dual Loop 在线+离线双环自进化中，有哪些模型会被训练？

**结论**：涉及**三类模型**被训练（Main Agent / MetaModel 策略 / Evaluator + 频率控制策略），分布在不同 loop；Strategy Memory 是经验库不是模型，通过写入而非梯度进化。

#### 关键概念区分：loop ≠ 训练

论文明确："Delayed online outcomes...are not synthesized as episode feedback in this **offline loop**. They belong to DREAM's **online reward loop**, where observed outcomes can be used to monitor the policy, recalibrate the Evaluator, and consolidate validated conclusions into Strategy Memory."

| Loop | 性质 | 训练什么 |
|------|------|----------|
| **离线环（Offline Loop）** | 真正的"训练"——参数更新 | MetaModel 策略 + Evaluator 重校准 |
| **在线环（Online Loop）** | 非参数更新——经验积累 + 校准 | Strategy Memory（经验库）+ Evaluator 重校准 + 频率控制策略 |

#### 类别 1：Intent Engine 的 Main Agent（0.8B）——on-policy distillation

**证据**（core.tex:351-391）："At the model level, these outputs are used to **train the Main Agent** through **on-policy distillation**."

**训练机制**：
- **Teacher**：异步精化层产出的 4B context subagent / expert
- **Student**：0.8B Main Agent
- **数据来源**：被双层路由升级到 4B 的请求（约 6.3%），累积的精化样本
- **训练目标**：反向 KL 散度，沿 Main Agent 自己采样的轨迹评估

$$\mathcal{L}_{\mathrm{OPD}}(\theta) = \mathbb{E}_{x_t, \hat{y}_t \sim p_\theta(\cdot|x_t)} \left[ \frac{1}{|\hat{y}_t|} \sum_{j=1}^{|\hat{y}_t|} \mathrm{KL}\big( p_\theta(\cdot|x_t, \hat{y}_{t,<j}) \| p_\phi(\cdot|x_t, \hat{y}_{t,<j}) \big) \right]$$

其中 $p_\theta$ 是 Main Agent（student），$p_\phi$ 是 4B 专精子代理（teacher）。

**关键设计**：
- **on-policy**：轨迹 $\hat{y}_t$ 由 Main Agent 自己采样（不是 teacher 采样），保证训练分布对齐推理状态
- **仅监督意图更新序列 $\Delta\mathcal{I}_t$**：不监督路由决策 $r_t$（路由是独立逻辑）
- **缩周期**：不是每次推理都更新，而是"累积精化样本 → 周期性 distill"

**效果**：自进化使 LLM-as-a-Judge 总分从 78.20% 提升到 **84.74%**（+6.54）。

**注**：4B expert 和 4B context subagent 是 teacher，**不被此 loop 训练**——它们的能力来自预训练或独立更新。

#### 类别 2：MetaModel 策略（Qwen3）——离线 RL with replay

**证据**（offline_rl.tex:13-90）："DREAM **trains its strategy policy offline** while executing rollouts through the production recommendation path."

**训练机制**：
- **被训练对象**：MetaModel 的策略 $\pi_\theta(\cdot|x)$（Qwen3 的参数）
- **数据来源**：真实在线请求输入日志构造的 replay 数据集
- **训练目标**：最大化二元 Evaluator 奖励的期望

$$\max_\theta J(\theta) = \mathbb{E}_{x \sim \mathcal{D}, a_x \sim \pi_\theta(\cdot|x)} \left[ r(x, a_x) \right]$$

其中 $r(x, a_x) = \mathds{1}[\bar{u}_x^1 > \bar{u}_x^0]$（策略 bundle 的均值 Evaluator 分 > 默认流水线均值分）。

**关键设计**：
- **单步上下文决策**：不模拟后续点击/交易/用户状态转移，只评估"生成 bundle → 执行 → list-level Evaluator 打分"这一步
- **生产路径 replay**：rollout 通过真实生产推荐流水线执行（隔离的压测流量），不在用户流量上做探索
- **二元奖励**：刻意简单——胜出为 1，否则为 0；不结合 rank、lift、curriculum、teacher distillation
- **均值聚合降低方差**：每个处理执行 $K$ 次，$\bar{u}_x^z = \frac{1}{K}\sum_k u_{x,k}^z$

**效果**：replay RL 使 pCTR +2.42%、pIPV +1.38%、pGMV +0.37%，策略有效性从 80.86% 升至 **98.85%**（+17.99 pp）。

#### 类别 3：Evaluator + 频率控制策略——辅助训练

**(a) Evaluator（list-level 打分器）**

**证据**（offline_rl.tex:92-96）："The Evaluator is a **learned proxy** for immediate list quality... observed outcomes can be used to monitor the policy, **recalibrate the Evaluator**, and consolidate validated conclusions into Strategy Memory."

- Evaluator 是"learned proxy"——一个学习出来的列表质量代理模型
- 在线环用真实点击/交易结果**重校准**（recalibrate）Evaluator
- 重校准让 Evaluator 的打分更贴近真实业务结果
- **作用**：Evaluator 的打分是 MetaModel RL 训练的 reward 信号。如果 Evaluator 偏差大，MetaModel 会被误导。所以 Evaluator 需要持续重校准。

**(b) 频率控制策略**

**证据**（freq_control.tex:9, 104）："A policy controller produces group-, time-, and scenario-dependent thresholds; **Invocation outcomes are fed back to update the signal models and control policy**." / "$\delta_{g(u)}$: user-group offset... each group receives a **learned offset**."

- 动态阈值 $\theta_{u,t} = \mathrm{clip}(\theta_0 + \delta_{g(u)} + \delta_{b(t)} + \delta_{s(t)} + \delta_{d(t)}, \theta_{\min}, \theta_{\max})$ 中的 $\delta$ 偏移是 "learned offset"
- 调用结果（LLM 推理是否确实带来收益）反馈回来更新"signal models and control policy"
- **作用**：频率控制策略自己也在被训练——根据历史调用结果学习各用户群/时段/场景的最优阈值偏移。

#### 重要澄清：Strategy Memory 不是模型，不被训练

Strategy Memory 是**经验库**而非模型，它通过**写入**而非**梯度更新**进化：

**证据**（offline_rl.tex:94-96）："They belong to DREAM's online reward loop, where observed outcomes can be used to monitor the policy, recalibrate the Evaluator, and **consolidate validated conclusions into Strategy Memory**."

**机制**：在线环把"策略 → 真实结果"的对照验证为正向/负向经验，**写入** Strategy Memory。M2 推理时**检索**（非梯度更新）这些经验作为参考/约束。Strategy Memory 更像一个**检索增强的数据库**而非"被训练的模型"。

#### 完整训练图谱

| 被训练对象 | 模型/类型 | 所在 Loop | 训练方法 | Teacher/数据来源 |
|------------|-----------|-----------|----------|------------------|
| **Main Agent** | 0.8B LLM | Intent Engine 自进化 | on-policy distillation（反向 KL） | 4B expert/context subagent |
| **MetaModel 策略** | Qwen3 | 离线环 | 离线 RL（二元 Evaluator 奖励） | 生产路径 replay + Evaluator |
| **Evaluator** | list-level 打分器 | 在线环 | 持续重校准（recalibrate） | 真实点击/交易反馈 |
| **频率控制策略** | $\delta$ 偏移 | 在线环 | 调用结果反馈更新 | LLM 调用的实际收益 |

**不被训练的对象**（澄清）：
- 4B expert / 4B context subagent：作为 teacher，能力来自预训练或独立更新
- 4B Dreaming 模型：夜间整合用，不做参数训练
- 端侧 GRU（F2 触发判定）：论文未明确说是否训练，但提到"隐藏态跨事件传递"，推断是固定结构
- Strategy Memory：经验库，通过写入而非梯度进化
- 既有召回/排序/重排模型：完全不变（overlay 设计）

#### 三个值得注意的设计决策

**(1) Intent Engine 用 distillation，MetaModel 用 RL——为什么方法不同？**

因为两个 LLM 的目标不同：
- **Intent Engine Main Agent**：产出结构化意图，有明确的"正确答案"（4B expert 的精化结果），适合 distillation
- **MetaModel**：产出策略 bundle，没有明确的"正确答案"——只有"这个 bundle 是否比默认好"的相对判断，适合 RL 的奖励驱动

**(2) MetaModel RL 用二元奖励而非连续奖励——为什么？**

论文明确说"刻意简单"（deliberately simple）："The training reward itself is binary; it does not combine rank, lift, curriculum competition, or teacher distillation terms."

原因是：连续奖励会引入 Evaluator 分数尺度的偏差（不同请求的 Evaluator 分绝对值差异大），二元奖励通过"是否胜出默认"消除了这种尺度差异，直接问"个性化覆写是否优于生产默认"。

**(3) Evaluator 需要重校准——这是潜在的脆弱点**

Evaluator 是 MetaModel RL 的 reward 来源，但 Evaluator 自身又是在线环重校准的。这形成了一个**自我参照**：Evaluator 偏差会传导到 MetaModel。DREAM 的应对是"用真实点击/交易结果重校准 Evaluator"——但这依赖在线反馈的及时性和准确性，是系统的潜在脆弱点。