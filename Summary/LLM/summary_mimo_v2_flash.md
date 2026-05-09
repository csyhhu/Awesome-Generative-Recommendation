# MiMo-V2-Flash 技术报告总结

> 论文：MiMo-V2-Flash Technical Report  
> arXiv ID：2601.02780  
> 机构：小米 LLM-Core 团队  
> 提交日期：2026 年 1 月 6 日

---

## 一、论文动机与问题陈述

随着 AGI 研究的推进，强化学习（RL）驱动的推理链和自主 Agent 工作流成为两大核心前沿。然而，构建可扩展推理器和 Agent 面临一个关键瓶颈：**长上下文建模必须同时做到快速（Fast）和强大（Strong）**。

现有全局注意力（Global Attention, GA）模型在长序列下面临二次复杂度问题，KV Cache 存储和注意力计算开销巨大；而简单的滑动窗口注意力（SWA）在过小窗口或过高 SWA:GA 比例下往往导致性能明显下降。此外，大规模 RL 后训练中存在两个核心挑战：
1. **能力不均衡**（Capability Imbalance）：提升某一领域往往导致其他领域退化（"跷跷板效应"）；
2. **学习低效**（Learning Inefficiency）：多模型知识融合时训练信号利用不充分。

---

## 二、核心贡献与方法

### 2.1 模型架构：混合滑动窗口注意力 + MoE

**MiMo-V2-Flash** 是一个 **309B 总参数、15B 激活参数** 的混合专家（MoE）模型，核心架构特点：

- **混合注意力架构（Hybrid SWA + GA）**：
  - 48 层 Transformer，其中 39 层为滑动窗口注意力（SWA），9 层为全局注意力（GA）；
  - 滑动窗口大小 **W = 128**，SWA:GA 混合比例为 **5:1**；
  - 极小的窗口尺寸配合 5:1 的混合比，使长上下文的 KV Cache 存储和注意力计算量减少近 **6 倍**；
  - 第一层例外：使用全局注意力 + 稠密 FFN 以稳定早期表征学习。

- **可学习注意力 Sink Bias**：
  - 在 SWA 的 softmax 分母中引入可学习标量 $sink \in \mathbb{R}$，允许模型在需要时对某些 token 分配趋近于零的注意力权重；
  - 实验证明，W=128 + sink bias 的混合 SWA 不仅能追平全局注意力基线，在推理能力和长上下文任务中甚至 **超越全局注意力**。

- **MoE FFN 配置**：每层含 256 个专家，每 token 激活 8 个，无共享专家；稠密层（第一层及 MTP block）使用标准 FFN，中间维度 16384。

- **轻量级多 Token 预测（Multi-Token Prediction, MTP）**：
  - MTP block 使用 **稠密 FFN**（而非 MoE）和 **SWA**（而非 GA），每个 block 仅 0.33B 参数，极为轻量；
  - 预训练阶段仅使用 1 个 MTP head；后训练阶段扩展为 K 个 head（实验取 K=3）；
  - 推理时将 MTP 重用为 **投机解码（Speculative Decoding）草稿模型**，实现无需额外硬件的加速。

### 2.2 预训练

- **数据规模**：27 万亿 token，来源包括高质量公开网页、书籍、论文、代码、数学及 STEM 材料，特别强调长程依赖数据（仓库级代码、PR、issue、commit 历史等）；
- **三阶段训练调度**：
  1. **Stage 1（0–22T）**：通用语料，上下文长度 32K；
  2. **Stage 2（22–26T）**：上采样代码数据，引入约 5% 合成推理数据；
  3. **Stage 3（26–27T）**：上下文扩展至 256K，上采样长程依赖数据；
- **训练精度**：FP8 混合精度训练；
- **优化器**：AdamW，$\beta_1=0.9, \beta_2=0.95$，权重衰减 0.1。

### 2.3 后训练：多教师在策蒸馏（MOPD）

**MOPD（Multi-Teacher On-Policy Distillation）** 是本文最核心的创新后训练范式，分三个阶段：

#### Stage 1：监督微调（SFT）
- 训练数据涵盖通用对话、推理、代码、Agent 任务，包含 thinking 和 non-thinking 两种模式；
- 关键稳定性指标：零梯度参数数量（num-zeros），需监控专家路由均衡性。

#### Stage 2：领域专用强化学习
分别训练多个**领域专用教师模型**：
- **非 Agent 任务**：数学推理、通用推理、安全对齐；
- **Agent 任务**：代码调试（基于 GitHub issue，约 10 万+任务，多语言 Docker 环境）、终端操作（基于 Stack Overflow，约 3 万任务）、Web 开发（多模态验证器，视频级渲染评估）、通用 Agent（搜索 Agent + 函数调用 Agent）。

#### Stage 3：多教师在策蒸馏
将多教师知识整合转化为**在策强化学习过程**：

$$\mathcal{L}_{\text{MOPD}}(\theta) = -\mathbb{E}_{x \sim\mathcal{D}, y \sim \mu_{\theta}}\left[\frac{1}{|y|}\sum_{t=1}^{|y|} w_t \hat{A}_{\text{MOPD},t} \log\pi_{\theta}(y_t|x, y_{<t})\right]$$

其中优势函数为：

$$\hat{A}_{\text{MOPD},t} = \text{sg}\left[\log\frac{\pi_{\text{domain}_x}(y_t|x, y_{<t})}{\pi_\theta(y_t|x, y_{<t})}\right]$$

- 学生模型从**自身分布采样**（在策），避免离策方法的分布偏移；
- token 级别的 KL 散度奖励提供**密集监督信号**，加速收敛；
- 支持与 ORM（结果奖励模型）的奖励联合优化：$\hat{A}_t = \hat{A}_{\text{MOPD},t} + \alpha \hat{A}_{\text{ORM}}$；
- 支持**教师-学生协同进化**：蒸馏后的学生可重新进入专用 RL 阶段生成更强教师。

### 2.4 RL 基础设施

- **训练引擎**：Megatron-LM；**推理引擎**：SGLang；均采用 FP8 精度；
- **Rollout Routing Replay（R3）**：解决 MoE 模型在 rollout 和训练间路由不一致问题；
- **数据调度器**：基于历史通过率的细粒度序列调度，支持 partial rollout，减少 GPU 空闲；
- **Toolbox + Tool Manager**：通过 Ray 实现工具资源统一管理，支持预热、异步奖励计算和超时恢复。

---

## 三、主要结果

### 3.1 Base 模型性能
- 在推理任务（MMLU-Pro, GPQA-Diamond, AIME）上持续优于同类开源 Base 模型；
- SWE-Bench（few-shot）超过参数量约为 3 倍的 Kimi-K2-Base；
- 长上下文检索：32K–256K 成功率接近 **100%**；
- GSM-Infinite（16K→128K）性能退化极小，优于 DeepSeek-V3.2-Exp。

### 3.2 后训练模型性能

| 基准 | MiMo-V2-Flash | 备注 |
|------|--------------|------|
| SWE-Bench Verified | **73.4%** | 开源最优，接近 GPT-5-High |
| SWE-Bench Multilingual | **71.7%** | 开源最优 |
| AIME 2025 | 可与 Kimi-K2-Thinking / DeepSeek-V3.2-Thinking 相当 | — |
| BrowseComp | 45.4（+上下文管理后 58.3） | — |
| τ²-Bench（电信/零售/航空） | 95.3 / 79.5 / 66.0 | — |
| LongBench V2, MRCR | 超越更大的 Kimi-K2-Thinking | 混合 SWA 架构优势 |

### 3.3 MTP 推理加速
- 3 层 MTP，不同任务平均接受长度：代码任务（WebDev）约 **3.6 tokens**，高不确定性任务（MMLU Pro）约 **2.0 tokens**；
- 实际解码加速：在批大小 32–128、接受长度约 3.6 时，可实现 **约 2.4×** 的解码加速；
- 加速与接受长度呈线性关系，且无需额外硬件。

---

## 四、结论与局限性

**结论**：MiMo-V2-Flash 通过混合 SWA 架构、轻量 MTP 和 MOPD 后训练范式，实现了强推理 + 强 Agent 能力 + 快推理速度的三者兼顾，以 309B（15B 激活）规模比肩 DeepSeek-V3.2（600B+）和 Kimi-K2（约 1T 总参数）等更大模型。

**局限性**：
1. 与最强闭源模型仍有差距，计划通过扩大模型规模和训练算力进一步缩小；
2. 架构探索尚在初期，设计权衡分析有限；
3. MOPD 中教师-学生协同进化的潜力尚未充分释放。

---

## 五、潜在相关性与应用价值

1. **高效长上下文建模**：混合 SWA + 可学习 sink bias 的方案对需要处理长序列（如文档理解、代码仓库级任务）的研究具有直接参考价值；
2. **多能力融合训练**：MOPD 范式对多任务 LLM 训练、跨领域知识蒸馏提供了新的思路，可避免"跷跷板效应"；
3. **Agent 能力扩展**：代码 Agent、终端 Agent、Web Agent 的大规模 RL 训练策略和环境构建方案对 Agent 研究有直接应用价值；
4. **推理加速**：MTP 作为轻量草稿模型用于投机解码，无需额外模型即可实现 2–2.6 倍加速，对推理服务部署有实用价值；
5. **推荐系统角度**：MiMo-V2-Flash 在软件工程、Agent 工作流上的突破表明，超大规模推理型 LLM 有望成为下一代推荐系统（生成式推荐、智能 Agent 推荐）的底座模型。

---

## 六、MOPD 深度解析（讨论补充）

### 6.1 MOPD 的核心机制：教师不生成序列，只评分

MOPD 最容易误解的地方在于：**教师模型在整个训练过程中从不生成完整序列**，它只做一件事——对学生已经生成的 token 序列，在每个位置 \(t\) 上做一次 teacher-forcing 前向传播，查询教师在该 token 上的条件概率 $\pi_T(y_t \mid x, y_{<t})$。

具体流程：
1. **学生采样**：学生从自身分布 $\mu_\theta$ 自回归采样完整序列 $y = [y_1, y_2, ..., y_T]$
2. **两次前向传播**（并行）：
   - 学生模型对序列 $y$ 做 teacher-forcing 前向传播 → $\pi_S(y_t \mid x, y_{<t})$
   - 教师模型对**同一序列** $y$ 做 teacher-forcing 前向传播 → $\pi_T(y_t \mid x, y_{<t})$
3. **计算 token 级奖励**：$\hat{A}_t = \log \pi_T(y_t) - \log \pi_S(y_t)$
4. **Policy Gradient 更新**学生参数

关键语义：$\pi_T(y_t \mid x, y_{<t})$ 的含义是"**如果上下文走到了学生所走的这条路，教师在此岔路口选择 $y_t$ 的概率**"，与"教师自己会不会走这条路"无关。教师无法回头纠正已生成的 token，只能在当前位置往后修正。

---

### 6.2 Token 级 KL 奖励的数学本质

**Reverse KL 的 Monte Carlo 估计**：

完整的 Reverse KL 需要对词表求和：
$$D_{KL}(p_S \| p_T) = \sum_{v \in V} p_S(v) \log \frac{p_S(v)}{p_T(v)}$$

由于期望在**学生分布**下取，可用单样本 MC 估计代替全词表求和：
$$D_{KL}(p_S \| p_T) = \mathbb{E}_{y_t \sim p_S}\left[\log \frac{p_S(y_t)}{p_T(y_t)}\right] \approx \log \frac{p_S(y_t)}{p_T(y_t)} \quad \text{（单样本估计）}$$

取负号变为奖励：$\hat{A}_t = \log p_T(y_t) - \log p_S(y_t)$

这使计算量从 $O(\text{vocab\_size})$ 降至 $O(1)$，对 vocab\_size = 128K 的模型来说是工程上的关键优化。

**为什么用 Reverse KL 而非 Forward KL**：

| | Forward KL $D_{KL}(p_T \| p_S)$ | Reverse KL $D_{KL}(p_S \| p_T)$ |
|--|--------------------------------|--------------------------------|
| 期望在 | 教师分布下 | **学生分布下** |
| 必须 | 对全词表求和（教师无法采样） | 学生采样即可（在策天然契合） |
| 内存开销 | $(T, \text{vocab\_size})$ 完整 logits | 只需 $T$ 个标量 |
| 行为特性 | Mean-seeking，覆盖教师所有 mode | **Mode-seeking**，专注教师高概率区域 |

---

### 6.3 MOPD 之前的多教师融合方法及其缺陷

| 方法 | 核心缺陷 |
|------|---------|
| **参数合并（Model Soup / Task Arithmetic）** | 各专家的峰值能力被平均掉，合并后落在"山谷"里 |
| **顺序训练（Continual Training）** | 灾难性遗忘，新任务梯度覆盖旧任务知识 |
| **离策多教师 KD（Offline KD）** | 序列来自教师，学生推理时走自己的路径，严重分布偏移 |
| **混合 SFT** | 上限被数据质量限制，无法超越标注数据天花板 |
| **在策单教师 KD（MiniLLM）** | 解决了分布偏移，但只有一个教师，无法多能力融合 |

MOPD 的核心创新是**首次将三个已有思路正确组合**：
- 在策采样（MiniLLM/Agarwal 2023）→ 消除分布偏移
- Token 级 KL 奖励（MiniLLM）→ 稠密精准监督信号
- 多教师领域路由（本文新增）→ 解决能力不均衡

---

### 6.4 MiniLLM：MOPD 的直接前身

MiniLLM（Gu et al., ICLR 2024，微软）首次将 LLM 蒸馏从 Forward KL + 离策改为 Reverse KL + 在策，是 MOPD 的直接前身。其梯度分解为两项：

$$\nabla \mathcal{L} = \underbrace{-\sum_t \nabla \sum_{v \in V} \pi_S(v) \log \frac{\pi_T(v)}{\pi_S(v)}}_{\text{Single-step：当前位置全词表 KL}} + \underbrace{-\sum_t R_{t+1}^\text{Norm} \nabla \log \pi_S(y_t)}_{\text{Long-range：未来累积奖励}}$$

MOPD 相当于把 MiniLLM 的**累积奖励 $R_t$（从当前到序列末尾的和）简化为单步即时奖励 $\hat{A}_t$**，因为 ORM 结果奖励负责处理长程信用分配，token 级 KL 奖励只需要负责局部的教师偏好信号即可。

---

### 6.5 Distillation 与 RL 的统一视角

**Reverse KL + 在策采样本质上就是 RL**，两者在数学上共享同一个 Policy Gradient 框架：

$$\nabla \mathcal{L}_\text{reverse-KL} = -\mathbb{E}_{y \sim \pi_S}\left[\underbrace{\log \frac{\pi_T(y_t)}{\pi_S(y_t)}}_{\text{奖励（来自教师）}} \cdot \nabla \log \pi_\theta(y_t)\right]$$

**教师模型本质上是一个"软奖励模型"**，提供比稀疏 0/1 奖励信息量更丰富的 token 级指导。

三种奖励信号的分工：

| 奖励类型 | 信号粒度 | 告诉学生什么 | 不能解决什么 |
|---------|---------|------------|------------|
| 规则验证器（数学/代码） | 序列级（0/1） | 答案对不对 | 每步的信用分配 |
| 奖励模型 ORM（主观任务） | 序列级（连续值） | 输出质量好不好 | 推理路径的细节 |
| 教师 logits（KL 蒸馏） | **Token 级（连续值）** | 每步怎么走得更像专家 | 答案最终是否正确 |

MOPD 将三者统一在同一个 Policy Gradient 框架下叠加：

$$\hat{A}_{t}^\text{final} = \underbrace{\log\frac{\pi_T(y_t)}{\pi_S(y_t)}}_\text{稠密教师信号} + \alpha \cdot \underbrace{\frac{r_i - \bar{r}}{\sigma_r}}_\text{GRPO 组间归一化 ORM 信号}$$

---

### 6.6 GRPO + Distillation 的协同效应

**纯 GRPO 的局限**：同一个 prompt 采样 G 条序列，靠组内相对排名估计优势。对于"都答对了但表达方式差异大"的序列，GRPO 无法区分推理路径的质量优劣。

**加入 Distillation 后**：

```
序列 A：推理正确，表达流畅     → ORM: +1，教师 KL 高  → Â ≈ +2.0（强化）
序列 B：推理正确，表达啰嗦     → ORM: +1，教师 KL 低  → Â ≈ +0.1（几乎不强化）
序列 C：推理错误，走对一半     → ORM: -1，教师 KL 中  → Â ≈  0.0（中性）
序列 D：推理错误，完全跑偏     → ORM: -1，教师 KL 低  → Â ≈ -1.5（强烈抑制）
```

不只奖励"答案对不对"，还奖励"推理方式像不像专家"，这解释了为什么 MOPD 实验中学生能在多个任务上**超越单个教师**（AIME 2025: 教师 93.9 → 学生 94.1）。

工程实现上，两套稳定性机制可同时叠加使用：
- **IS 权重 $w_t$**（Distillation 的稳定机制）：过滤 $\pi_\theta$ 和采样策略 $\mu_\theta$ 差距过大的 token
- **PPO-clip**（GRPO 的稳定机制）：限制策略更新步长

$$\mathcal{L} = -\sum_t w_t \cdot \min\left(\rho_t \hat{A}_t^\text{final},\ \text{clip}(\rho_t, 1\pm\varepsilon) \hat{A}_t^\text{final}\right) \cdot \log \pi_\theta(y_t)$$

---

### 6.7 验证器（Verifier）与奖励函数（Reward）的关系

**验证器是实现奖励函数的一种具体方式**，两者是包含关系，不是并列关系：

```
Reward Function（奖励函数）—— 抽象概念：定义"什么样的输出是好的"
    │
    ├── 规则验证器（Verifier）      ← GRPO 常用，适合有标准答案的任务
    ├── 奖励模型（Reward Model）    ← RLHF 常用，适合主观偏好任务
    └── 教师 logits（KL 散度）      ← Distillation 的做法，token 级稠密奖励
```

#### 三种 Reward 实现方式详解

**方式一：规则验证器（Verifier）**

用程序或规则判断答案对错，给出确定性的 0/1 奖励：

```python
def math_verifier(response, ground_truth):
    extracted = extract_answer(response)     # 从回答中提取最终答案
    return 1.0 if extracted == ground_truth else -1.0

def code_verifier(response, test_cases):
    code = extract_code(response)
    passed = run_unit_tests(code, test_cases)
    return 1.0 if passed else -1.0
```

- **适用**：有标准答案的任务（数学、代码、逻辑推理）
- **优点**：客观、确定、可复现，无需训练
- **缺点**：只适用于可验证域；答案格式难提取时容易误判；奖励稀疏（序列级 0/1）

**方式二：奖励模型（Reward Model / ORM / PRM）**

用另一个神经网络打分，本质是用人类偏好数据训练的分类器：

```
训练：人类标注者对两个回答 A/B 做偏好选择
     → 训练 reward_model: (prompt, response) → score ∈ ℝ

推理：reward_model(prompt, student_response) → 0.73  ← 连续值
```

- **ORM（Outcome RM）**：只看最终答案，序列级打分
- **PRM（Process RM）**：对每个推理步骤打分，粒度更细
- **适用**：主观任务（写作质量、帮助性、安全对齐）
- **缺点**：需要人类标注数据；存在 reward hacking 风险

**方式三：教师 logits（Distillation Reward）**

用教师模型的输出概率直接作为 token 级奖励，无需标准答案：

```python
r_t = log(teacher_prob(y_t | x, y_{<t})) - log(student_prob(y_t | x, y_{<t}))
```

- **适用**：迁移专家"推理风格"，而非验证"正确性"
- **优点**：稠密（每个 token 都有反馈）；不需要标注数据
- **缺点**：教师自身也可能出错；无法判断最终答案是否正确

#### MiMo-V2-Flash 中的 Reward 设计

论文根据任务类型分别使用不同的 Reward 实现，再叠加 Distillation 奖励：

| 任务类型 | Reward 实现 | 备注 |
|---------|------------|------|
| 数学 / 代码 / 逻辑 | 规则验证器 + LLM judge 兜底 | 程序判题为主，边界情况用 LLM 辅助 |
| 开放对话 / 安全 | Rubric-based LLM judge | 给定评分细则，LLM 对照打分 |
| 所有任务（叠加） | 教师 logits KL 奖励 | 构成 $\hat{A}_\text{distill}$，与上述 ORM 奖励加权求和 |

三类奖励信号对比：

| 奖励类型 | 信号粒度 | 需要训练 | 可能出现的问题 |
|---------|---------|---------|-------------|
| 规则验证器 | 序列级（0/1） | ❌ | 答案格式难提取 |
| 奖励模型 ORM | 序列级（连续值） | ✅ 需人类标注 | Reward hacking |
| 教师 logits | **Token 级（连续值）** | ❌ | 教师也可能出错 |
