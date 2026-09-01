# Self-Evolving Recommendation System: End-To-End Autonomous Model Optimization With LLM Agents

> **论文**: [arXiv:2602.10226](https://arxiv.org/pdf/2602.10226)
> **会议**: RecSys '26 (20th ACM Conference on Recommender Systems)
> **机构**: Google Inc (YouTube)
> **作者**: Haochen Wang*, Yi Wu*, Daryl Chang*, Li Wei, Lukasz Heldt (*: Equal contribution)

---

## 1. 核心问题与动机

### 1.1 推荐系统优化的核心瓶颈

现代大规模推荐系统（如 YouTube 的视频排序）通常被建模为强化学习（RL）问题，目标是最大化长期用户满意度。然而，当前优化流程存在两大关键瓶颈：

1. **优化目标与实际指标的对齐鸿沟 (Alignment Gap)**：模型在训练时优化的是可微的代理损失（proxy loss），但真正关心的用户满意度是非可微、延迟、稀疏且语义复杂的。
2. **传统 AutoML 的能力边界**：标准 HPO（超参优化）、NAS（神经网络结构搜索）等方法仅能在**预定义的搜索空间**内进行数值调优，无法：
   - 创造全新的奖励逻辑（reward logic）
   - 从零设计新的神经网络交互层
   - 解读历史实验结果并推断特定用户切片的服务短板

### 1.2 从"自动调参"到"自主科学发现"

近期 AI Scientist 类工作（如 AlphaEvolve、The AI Scientist、MLE-STAR）已验证 LLM Agent 可编排完整的科学发现闭环：假设生成 → 代码编写 → 基于实证结果修正理论。但这些工作针对的是静态学术数据集（ImageNet、Kaggle 等），并未解决**工业级生产环境**的独特挑战：
- 含噪反馈循环
- 严格的安全护栏
- 复杂的用户-系统交互
- 必须遵循 A/B 测试协议

本文正是填补这一交叉空白：在 YouTube 生产级推荐系统上部署**自进化推荐系统**，由 LLM Agent 扮演专业机器学习工程师（MLE）的角色，端到端自主完成代码修改、训练、部署与 A/B 测试。

---

## 2. 问题形式化：双层优化问题

整个推荐系统被建模为一个**双层优化问题 (Bi-level Optimization)**：

### 2.1 下层：模型训练 (Lower Level)

排序模型参数 $\theta$ 被训练以最小化代理损失 $\mathcal{L}_{\text{proxy}}$：

$$\theta^*(\Phi) = \arg\min_{\theta} \mathcal{L}_{\text{proxy}}(\mathcal{D}; \theta, \Phi)$$

其中：
- $\mathcal{D}$ = 训练数据日志
- $\Phi$ = 系统元配置（meta-configuration），包含三大核心组件：
  - **优化器 ($\eta \in \Phi$)**：学习率、更新规则（如 AdaGrad、RMSprop）
  - **架构 ($\phi \in \Phi$)**：排序网络拓扑（如 DCN、GLU）
  - **奖励定义 ($r \in \Phi$)**：训练标签逻辑，平衡多种参与度信号

### 2.2 上层：北极星指标优化 (Upper Level)

寻找最优配置 $\Phi^*$ 使得部署后线上北极星业务指标 $\mathcal{M}$ 最大化：

$$\Phi^* = \arg\max_{\Phi} \mathcal{M}(\theta^*(\Phi)) \quad \text{s.t.} \quad \mathcal{G}(\Phi) \le C$$

其中 $\mathcal{G}(\Phi)$ 为系统级约束（如训练成本）。**传统上，这个上层优化完全由人类研究员手动完成**，而本文的目标正是用 MLE Agent 将其自动化。

---

## 3. 自进化系统架构 (Dual-Loop Framework)

### 3.1 整体架构：双环协同 + 实验日志中心

系统围绕一个共享的**实验日志 (Experiment Journal)** 运转，它是持久化的知识库，记录所有历史配置、对应的离线分数以及可用的线上指标。

在此之上构建两个异步运行的 Agent：

| 组件 | 频率 | 核心职责 | 类比 |
|---|---|---|---|
| **Offline Agent (Fast Loop)** | 每天运行，但每 5 分钟唤醒一次 | **提名 (Nominate)**：高频生成候选改进 | 初筛漏斗 |
| **Online Agent (Slow Loop)** | 每天运行一次 | **排序 (Rank)**：低频筛选高潜力候选并推进线上实验 | 终评裁判 |

### 3.2 共享 Prompt 模板

两个 Agent 共享同一 Prompt 模板结构（通过 `{AGENT_TASK}` 注入各自任务）：

```
# PERSONA
你是一位才华横溢、富有创新精神的机器学习科学家，精通编程和分析技能。
你希望改进模型，在 {AGENT_SPECIALIZATION} 领域拥有深厚专业知识。

{AGENT_TASK}  ← 注入 Offline/Online 特定任务

# CONTEXT
- 当前模型基线配置
- 可选：过去 SQL 数据分析的输出 {SQL_QUERY_OUTPUT}
- 历史实验日志（按离线分数排序）[EXPERIMENT JOURNAL]

# EXAMPLE PROPOSAL
{AGENT_EXAMPLE}  ← Few-shot 样例
```

---

## 4. Offline Agent (Fast Loop)：专业化 Persona 与工具链

### 4.1 为什么需要专业化 Persona？

单体 Persona 面对全量代码库（>400k tokens）会迅速出现**幻觉 (hallucination)**。因此，系统将推荐系统设计任务分解为 3 个专业化 Persona，每个只聚焦修改 $\Theta(10)$ 行代码（对应人类工程师典型改动规模）：

---

### 4.2 Persona A：优化器专家 (Optimizer Persona)

- **搜索空间**：优化器类型（Adagrad → RMSprop 等）及内部超参（动量、batch size、学习率等）
- **特点**：损失函数定义不变（proxy loss 可比），因此不同配置的 $\mathcal{L}_{\text{proxy}}$ 可以直接比较
- **评分工具**：`compute_loss`
- **排序准则**：$\Phi_A \succ \Phi_B \iff \mathcal{L}_{\text{proxy}}(\Phi_A) < \mathcal{L}_{\text{proxy}}(\Phi_B)$
- **流程**：批量生成配置 → 异步启动训练任务 → 计算每个配置的 Loss

---

### 4.3 Persona B：架构专家 (Architecture Persona)

- **搜索空间**：神经网络拓扑（层类型、连接方式、激活函数等）
- **与标准 NAS 的本质区别**：不是从固定菜单中选择，而是**直接写代码创造全新结构**
  - 例：将标准 embedding lookup 替换为自定义"Gated Path"门控路径机制
  - 例：在特定子塔引入 Layer Normalization
- **评分工具**：`compute_loss`（与优化器相同，因为损失定义不变）
- **核心能力**：既有探索（发明新结构）又有利用（微调已有优胜结构）

---

### 4.4 Persona C：奖励专家 (Reward Persona)

这是最复杂的 Persona，因为修改奖励定义会**从根本上改变优化景观**，不同奖励下的 $\mathcal{L}_{\text{proxy}}$ **不可比较**（例如"仅点击"奖励天然比"点击+满意度"奖励更容易学习，loss 更低并不代表更好）。

**多步推理流程**：

1. **开放式数据分析（Step 1）**：使用 `run_sql_query` 工具对 PB 级用户交互日志进行大规模 SQL 分析，挖掘假设。
   - 例：发现"被分享的视频平均观看时长更高"→ 推论分享信号与用户满意度正相关。

2. **基于发现生成配置（Step 2）**：将 SQL 分析结果注入 prompt（`{SQL_QUERY_OUTPUT}`），生成新的奖励函数配置。

3. **损失无关代理评估（Step 3）**：使用 `compute_eval` 工具计算**损失无关的代理指标**（surrogate proxy），例如：
   - **长观看相关性 (long-watch correlation)**：模型预测值与实际长观看行为的 Pearson/Spearman 相关系数，越高越好
   - 与留存率、重复消费等指标的相关性

- **目标**：识别与用户参与度高度预测相关的信号
- **工具**：`run_sql_query`（数据分析）+ `compute_eval`（奖励质量验证）

---

## 5. Online Agent (Slow Loop)：五阶段实验生命周期管理

Online Agent 以生产部署的**准确性与安全性**为第一优先级，从 Experiment Journal 中选择 Top-K 候选，在延迟的北极星指标 $\mathcal{M}$ 上进行真实验证。整个流程分 5 个阶段：

### 阶段 1：候选选择 (Selection)
- 输入：Experiment Journal 中所有候选（按离线分数排序，但需额外考虑线上历史表现综合排序）
- 输出：Top-K 候选分三路：
  - 新进入 Top-K → 进入阶段 2+3（训练 + 上线实验）
  - 已在活跃实验中且仍在 Top-K → 进入阶段 4（继续收集指标）
  - 跌出 Top-K → 进入阶段 5（资源回收）

### 阶段 2：模型训练 (Model Training)
- Agent 训练新配置的模型并监控收敛性，确保权重成功版本化并导出

### 阶段 3：线上实验 (Live Experimentation)
- Agent 为新训练的模型分配生产流量，开启线上 A/B 实验

### 阶段 4：指标综合 (Metric Synthesis)
- Agent 拉取线上北极星指标，写回 Experiment Journal，为下一轮推理提供关键数据

### 阶段 5：清理 (Cleanup)
- 对不再在 Top-K 中的候选，Agent 清理其训练器和实验资源，关闭无效探索方向

> **方法论总结**：双 Agent 而非单体 Agent，建立了严格的过滤漏斗。人类工程师只需要：① 向 Offline Agent 提出高层研究想法；② 审阅 Online Agent 收集的最终实验结果。

---

## 6. 生产部署与核心结果

### 6.1 总体影响力

自进化系统已在 YouTube 多个关键推荐面（surfaces）上线：
- Agent 生成的改进，平均超过了过去 6 个月中 **64%** 的传统手动方式上线项目（YouTube-level 指标）
- 在 surface-level 指标上，超过了 **73%** 的手动方式上线

### 6.2 优化器改进（Loss 优化）

| 发现 | YouTube 级指标 | Surface 级指标 |
|---|---|---|
| **切换至 RMSprop**（从 Adagrad → RMSprop(0.005, 0.95, ...)） | **+0.06%** ✓ | **+0.12%** ✓ |
| **训练效率 4× 提升**（batch size / epoch / 超参联合调优） | -0.01% (不显著) | +0.06% (不显著) |
| **训练效率 2× 提升**（累计 8× 总加速） | +0.01% (不显著) | **+0.09%** ✓ |

> 关键洞察：传统上优化器配置因调优成本高昂而长期静态，LLM 可以直接被要求"找出最好的 Keras 优化器"，无需枚举可用关键词。

### 6.3 架构改进（Loss 优化）

| 发现 | YouTube 级 | Surface 级 |
|---|---|---|
| **Gated Path (类 GLU)**：在子网络引入门控路径，替代原 MLP | **+0.06%** ✓ | **+0.14%** ✓ |
| **激活函数优化**：Sigmoid Gate → GELU + LayerNorm | -0.02% (不显著) | **+0.12%** ✓ |

> 架构具体演化对比：
> - **初始**：`layer_norm(relu(dense(relu(dense(inputs, 128), 128))))`
> - **进化后**：引入 value_path × gate_path 门控乘法，其中 gate_path 使用 sigmoid（后续又升级为 GELU + LN）

### 6.4 奖励函数改进（语义对齐）

| 发现 | YouTube 级 | Surface 级 |
|---|---|---|
| **多目标合成**：引入新信号（用户活跃参与度指示因子） | **+0.05%** ✓ | **+0.17%** ✓ |
| **奖励超参调优**（4 个超参联合调优，仅依赖 Slow Loop） | **+0.05%** ✓ | **+0.21%** ✓ |

> 奖励超参调优的突破性意义：人类研究员历时**数月**手动调优这些超参，始终无法同时改进 YouTube-level 与 surface-level 指标；Agent 仅用**两周**就找到了同时改进的配置。

### 6.5 跨面泛化验证 (Generalizability)

将完全相同的双 Agent 架构部署到另一个**特征 schema、训练数据集、模型配置完全不同**的 YouTube 推荐面，Agent 在头几轮迭代内就成功适应：

| 发现 | YouTube 级 | Surface 级 |
|---|---|---|
| 切换至 FTRL 优化器 | **+0.03%** ✓ | **+0.16%** ✓ |
| Wide & Deep 模型结构 | **+0.08%** ✓ | **+1.10%** ✓ |

证明框架优化的是**发现过程本身**，而非对特定数据集的记忆。

---

## 7. 消融研究：Agent 性能驱动因素分析

以优化器组件离线 loss 为任务，6 次独立运行 × 每次 70 个 idea，结果以 **normalized z-score（越低越好）** 呈现：

| Agent 配置 | Normalized Loss | 对比基线 |
|---|---|---|
| **opt_2p5** (Gemini 2.5 Pro + MLE Persona + 完整排序历史) | **-0.84** $[-1.70, -0.01]$ | 基线 |
| opt_top_5 (仅 Top-5 历史) | -0.72 | 略降 |
| opt_no_role (移除 MLE Persona 身份) | -0.52 | 明显下降 |
| opt_no_sort (按时间排序而非 loss 排序) | +0.06 | 反转 |
| opt_top_1 (仅 Top-1 历史) | +0.11 | 反转 |
| opt_flash (Gemini 2.5 Flash 轻量版) | +0.85 | 严重退化 |
| opt_no_context (无任何历史) | +1.05 | 最差 |

### 7.1 模型规模与推理能力的关键作用

Gemini 2.5 Pro 大幅优于 Gemini 2.5 Flash。算法发现所需的推理深度，直接受益于大模型的参数量和"深度思考"能力。

### 7.2 Persona 与上下文工程的作用

- **Persona 身份设定**（"你是专业 MLE"）显著影响提案的相关性与深度
- **上下文排序**：完整的、按离线 loss 排序的 Experiment Journal 至关重要；时间排序、仅 Top-K、或完全无上下文，效果逐级劣化

---

## 8. 效率与成本分析

### 8.1 实验速度红利 (Velocity Dividend)

| 指标 | 人工工作流 | Agent 工作流 |
|---|---|---|
| 实验吞吐量 | $\Theta(1) - \Theta(10)$ / 周 | **$\Theta(100)$ / 周**（数量级提升） |
| 每次实验工程师成本 | $\Theta(1) - \Theta(10)$ 小时 / 周 | **$0$ 小时 / 周** |

"想法 → 数据"的循环（Idea-to-Data cycle）被显著压缩，团队得以探索人类工程师带宽根本无法覆盖的配置空间长尾。

### 8.2 LLM Token 成本

6 个月期间 LLM Token 总花费约 **$20,000**，仅相当于一位全职 MLE 成本的极微小部分。

---

## 9. 经验教训 (Lessons Learned)

### L1：Delta 生成 vs. 全量配置生成
- **Delta 生成远优于全量生成**。要求 Agent 生成"相对于生产配置的增量修改"而非重写整个配置文件，原因：
  - 全量生成常出现幻觉（省略不变的必要参数、语法错误）
  - Delta 极度紧凑，因此可在每个 prompt 中附加 $\Theta(100)$ 条历史实验作为上下文；而 AlphaEvolve 等全量方法必须大幅裁剪上下文

### L2：Prompt 调优强制多样性
无显式指令时，Agent 强烈偏向安全的**增量微调**（如连续提出"学习率 0.1"→"0.11"）。必须显式要求 Agent **"平衡探索、利用与创新"**，强迫其做出产生显著增益所需的跨越性尝试。

### L3：Warm Start 的重要性
提名好配置的能力高度依赖 Experiment Journal 的密度。无历史时，Agent 往往提出教科书式的通用改进。做法：**每天 Offline Agent 的运行以昨日最佳 K 个发现作为 Warm Start**，使其推理建立在前一天的学习之上。

### L4：语义推理 vs. 数值调优
传统 AutoML 在标量调优上表现出色，但成熟系统中**最高杠杆的改进来自结构与语义层面的突变**。例如 Reward Persona 对奖励函数的重新定义，是纯数值调优永远无法实现的。

### L5：跨推荐面的可迁移性
见 §6.5，同一双 Agent 架构可无缝迁移到完全不同的推荐面，证明其优化的是**发现流程**而非特定数据集。

---

## 10. 结论与未来展望

本文提出了基于 LLM 的**自进化推荐系统**完整框架，并在 YouTube 规模化部署成功。核心范式转换：

1. **发现过程解耦**：将高频离线发现（Fast Loop，廉价代理信号驱动）与可靠但低频的线上验证（Slow Loop，延迟北极星指标驱动）解耦
2. **结构与语义创新**：在适当的上下文和工具支撑下，LLM 有能力在推荐系统中做出结构级（架构、优化器）和语义级（奖励函数）的创新
3. **实验吞吐量数量级提升**：从每周 $\Theta(10)$ 量级提升至 $\Theta(100)$ 量级，覆盖人类工程师无法触及的长尾配置

**未来方向**：MLE 的角色将发生范式转移——随着自进化系统执行模型改进，人类工程师的焦点转向定义**战略护栏、伦理约束以及系统的长期愿景**，将人类认知带宽从科学发现的日常瓶颈中解放出来。

---

## 附录：关键对比总结

| 维度 | 传统 AutoML / HPO | 本文自进化系统 |
|---|---|---|
| 搜索空间 | 预定义的数值/层菜单 | 开放语义空间（可写代码创造新结构） |
| 搜索对象 | 连续超参、离散层选择 | 优化器、架构、奖励函数（代码级） |
| 反馈信号 | 离线 loss/准确率 | 离线 loss + 损失无关代理 + 线上北极星 |
| 推理能力 | 贝叶斯/GP/进化算法（数值） | LLM 自然语言深度推理 + 代码生成 |
| 实验频率 | 低（人工介入每步） | $\Theta(100)$/周，完全自主 |
| 可发现创新 | 有限（在预设空间内） | 可发现奖励语义创新、新拓扑结构 |
