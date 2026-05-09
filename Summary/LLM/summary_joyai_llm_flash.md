# JoyAI-LLM Flash: Advancing Mid-Scale LLMs with Token Efficiency

**论文信息**
- 标题：JoyAI-LLM Flash: Advancing Mid-Scale LLMs with Token Efficiency
- 作者：来自 JD.com（京东）的团队（69 位作者）
- ArXiv：[2604.03044](https://arxiv.org/abs/2604.03044)
- 提交日期：2026 年 4 月 3 日

---

## 一、研究动机与问题陈述

大型语言模型（LLM）的发展面临两大相互交织的挑战：**低 Token 效率**和**高计算成本**。许多模型在推理时消耗大量 Token 才能输出准确结果，如何在性能强劲的同时保持高效成为关键问题。

JoyAI-LLM Flash 专注于 **sub-50B 参数**范围内的效率与性能权衡，旨在：
1. 在保持最高性能的同时，大幅降低 Token 消耗；
2. 通过架构稀疏化提升推理吞吐量；
3. 提出新的强化学习算法提升策略优化的稳定性与 Token 效率。

---

## 二、模型架构

### 2.1 整体配置

| 超参数 | JoyAI-LLM Flash 48B-A3B |
|--------|--------------------------|
| 总层数 | 40 层（1 层 Dense + 39 层 MoE） |
| 隐藏维度 | 2048 |
| 词表大小 | 129K |
| 最大上下文 | 128K |
| 注意力类型 | MLA（多头潜在注意力） |
| 总专家数 | 256（路由）+ 1（共享） |
| 激活专家数 | Top-8 路由 + 1 共享 |
| 总参数量 | **48B** |
| 每次前向激活参数 | **2.7B**（含 Embedding 为 3.28B） |

### 2.2 关键设计

- **MLA 注意力**（参照 DeepSeek-V2/V3）：RoPE + 非 RoPE 维度分离，SwiGLU 激活
- **细粒度 MoE**：256 个路由专家 + 1 个共享专家，门控用 Sigmoid（FP32），无辅助损失负载均衡
- **Muon 优化器**：矩阵正交化的牛顿式更新，比 Adam 收敛更快、训练更稳定（无 Loss Spike）
- **轻量级单层 Dense MTP 头**：既增强训练信号，又原生支持推测解码（无需独立 Draft 模型）

### 2.3 训练基础设施

基于 Megatron-Core 扩展，支持：
- 2-way Pipeline Parallelism、8-way Expert Parallelism（跨两节点）、ZeRO-1 Data Parallelism
- FlashAttention-3 + DeepEP（专家调度低延迟）
- 分布式异步 Checkpoint（恢复时间从 15 分钟降至 30 秒）
- **Packing 训练**（块对角 Attention Mask）：注意力前向/反向分别加速 50%/20%

---

## 三、预训练数据（20.7 万亿 Token）

### 3.1 数据来源

1. **网页数据**：Common Crawl（截至 2025.10），Trafilatura 文本提取 + Datatrove 规则清洗 + MinHash-LSH 去重（Jaccard=0.9，7-gram）+ BERT 安全分类器 + 基于 Qwen 的行级噪声过滤
2. **代码数据**：Stack v2 + GitHub，AST 可解析性、函数比例等语言特定质量过滤；Qwen2.5-3B 轻量评分器（标注由 Qwen2.5-Coder-32B 生成）；长上下文代码按依赖图拓扑排序构建
3. **PDF 文档**：MinerU + DeepSeek-OCR 解析，语义分块（约 4096 token/段）
4. **大规模合成数据**：
   - 知识重新表述（MAGA + Nemotron-CC QA 风格）
   - 长推理 QA（DeepSeek V3.2 生成 + 多数投票验证）
   - 智能体轨迹（GLM-4.6 扮演智能体，多轮任务执行 + LLM 评判过滤）

### 3.2 四阶段预训练课程

1. **基础阶段**（Stage 1）：通用语言能力，Warmup-Stable LR（峰值 4.2e-4），批大小 38M tokens
2. **代码-数学强化阶段**（Stage 2）：余弦 LR 衰减 4.2e-4 → 1.4e-4
3. **中训练阶段**（Stage 3）：超高质量数据（合成比例 >60%），开启 MTP（loss 权重 0.1），LR 衰减至 4.2e-5
4. **长上下文扩展阶段**（Stage 4）：64K → 128K，LR 继续衰减至 2.0e-5

---

## 四、后训练流程

### 4.1 监督微调（SFT）

三大类训练数据：

**通用 SFT**
- 覆盖数学、编码、工具调用、指令遵循、安全、科学、定理证明、创意写作、多语言等
- **Thinking + Non-thinking 混合训练**：混合训练显著提升非思考模式性能
- 128K 序列打包（填充率 0.01%），训练吞吐提升 1.5×

**环境与智能体学习**（SWE 任务）
- 基于 GitHub Issues/PR 构建可验证环境，12 种编程语言
- 三步流水线：候选任务挖掘 → 可验证环境构建（Agent 自动化 Docker 构建）→ 轨迹生成（OpenHands/SWE-agent）
- 原子能力任务：精确代码编辑、自动测试生成

**工具集成推理（TIR）**
- 代码解释器轨迹（复杂数学 → Python 迭代计算）
- 搜索中心轨迹（多跳 QA，平均 8.64 次搜索调用/轨迹）
- 混合工具轨迹（搜索 + 代码协同）
- 终端操作轨迹（Docker 环境，5 维 LLM 评判过滤）

### 4.2 直接偏好优化（DPO）

- 1000 步，LR 1e-6 余弦衰减至 1e-7
- 偏好对：高质量回答 vs 拒绝采样负例（主要针对幻觉和指令偏离）
- 关键作用：快速惩罚不良输出，为 RL 提供高质量初始点

### 4.3 强化学习（FiberPO）

#### 动机与理论背景

现有方法的问题：
- **PPO/GRPO**：仅提供 per-token 裁剪，无法约束轨迹级整体漂移；一旦轨迹漂移超限，大量 Token 同时饱和，破坏 Token 级区分能力
- **GSPO**：将每条轨迹压缩为单一聚合比率，抑制轨迹内 Token 区分能力，熵发散
- **TRPO**：在 LLM RL（undiscounted, γ=1）中信任域消失（TRPO Vanishing Theorem）

#### FiberPO 目标函数

$$\hat{J}^{\text{FiberPO}}(\theta|\theta_{\text{old}}) = \sum_{(s,a,\tau,t) \in \bar{\mathcal{X}}} \frac{1}{|\mathrm{Tj}^{\theta_{\text{old}}}|} \frac{1}{T_\tau} \cdot \mathcal{G}(r_\bullet)_{s,a,\tau,t} \cdot \hat{A}_{s,a}$$

门控映射的两尺度乘积分解：
$$\mathcal{G}(r_\bullet)_i = \underbrace{w^{\rm base}_\tau}_{\text{轨迹级基础权重}} \cdot \underbrace{\tilde{r}_i^{\rm fiber}}_{\text{Token 级门控残差}}$$

**基础权重（轨迹级）**：
- 对正/负 log-ratio 分开追踪聚合（$s_\tau^+$, $s_\tau^-$）
- 分段线性聚合门 $g^{\rm agg}$（三段：直通 → 回滚 → 归零）具备**恢复性梯度**：漂移过大时梯度反向，主动抵制漂移
- 5 种全局区间（G-I~G-III）覆盖各种漂移组合

**门控残差（Token 级）**：
- 仅对 Token 偏离同符号轨迹均值的程度进行 logclip
- 与轨迹整体漂移一致的 Token 不受影响，保留完整梯度信号
- 三种局部区间（L-I~L-III），推荐 $\epsilon \ll \delta$ 确保局部调控先于全局

#### 三条理论保证

1. **轨迹独立性**：Jacobian 块对角，各轨迹梯度完全解耦
2. **一阶一致性**：on-policy 点恢复真实 RL 目标一阶近似
3. **尺度分离**：局部梯度 O(1)，轨迹级修正 O(1/T_τ)

#### 实验结果

**单域 RLVR（DAPO-Math-17k，AIME 2024）：**

| 方法 | AIME 准确率 | 训练奖励 | 策略熵 | 响应长度 |
|------|-----------|---------|------|---------|
| GRPO | 0.220（崩溃） | −0.083 | 0.038（熵崩溃） | 2216 Token（退化） |
| GSPO | 0.658（停滞） | 0.075 | 1.99（发散） | 7380~8870 Token |
| **FiberPO** | **0.786** | **0.766** | **0.43（稳定）** | **4543 Token（效率提升）** |

- FiberPO 梯度范数仅增长 1.5×（GRPO 12×，GSPO 7.5×）
- 响应从 7902 → 4543 Token，同时准确率提升（Token 效率的核心体现）

**多域扩展**：高斯课程采样（$\mu_0=0.8$ 易 → $\mu_T=0.2$ 难）+ 两级领域均衡采样，FiberPO 在编码、数学、知识、指令遵循、语言等所有领域稳定提升，无灾难性遗忘。

---

## 五、推理优化

### 5.1 量化策略

- **QAT（量化感知训练）**：模拟 INT4 量化（QDQ + STE），同时稳定 RL 阶段的 Rollout
- **PTQ（训练后量化）**：FP8、W4AFP8（INT4 权重 × FP8 激活）
- 自研 **DoubleQuant**（适用于 GGUF）：超块内量化后再对比例因子做二次量化

与 Qwen3-30B-A3B 对比（MATH-500 + GPQA + MBPP 均值）：
- JoyAI FP8：吞吐 +17%，精度损失极小
- JoyAI W4AFP8：吞吐 +28%，精度损失 1.2%

### 5.2 MTP 推测解码加速

SpecBench MTP-3（并发 64）性能对比：

| 模型 | GPU 数 | MTP 加速比 | 接受长度 | Server TPS |
|------|------|-----------|---------|-----------|
| **JoyAI-LLM Flash** | **1** | **1.87×** | 2.20 | 4241 |
| GLM-5 | 8 | 1.82× | 3.03 | 1969 |
| DeepSeek-V3.2 | 8 | 1.79× | 2.55 | 1958 |
| Qwen3.5-35B-A3B | 1 | 1.61× | 3.18 | 5428 |
| Step-3.5-Flash | 4 | 1.39× | 2.21 | 3048 |

JoyAI 以单卡实现最高 MTP 加速比（1.87×），高于 8 卡的竞争对手。

QAT + MTP 联合优化（ISL=1K, OSL=2K, 并发 64）：
- BF16 MTP3：+1.57×；FP8 MTP3：+1.81×；W4AFP8 MTP2：**+1.96×**

### 5.3 服务与调度

- **短上下文**（交互对话）：推荐节点内 PD 共置部署，减少通信开销
- **长上下文**（RAG/多轮 Agent）：PD 分离 + Mooncake 集中式 KV 缓存管理
- 关键洞察：小模型对数据传输开销更敏感，KV 缓存传输延迟可能超过重计算成本，需谨慎评估

---

## 六、Instruct 模型评估亮点

与 Qwen3-Next-80B-A3B、Qwen3.5-35B-A3B、Qwen3-30B-A3B、GLM-4.7-Flash-Thinking 对比：

| 任务 | JoyAI 准确率 | JoyAI Token 消耗 | 对比最优 Token 消耗 |
|------|------------|----------------|----------------|
| MMLU-Pro | 81.6 | 900 | Qwen3.5: 4000 |
| MATH-500 | 98.2 | **1300** | 其余均>1400 |
| LiveCodeBench v6 | **65.6** | 7300 | GLM-Thinking: 53600 |
| SWE-bench Verified | **62.6** | 24400 | GLM-Thinking: 51400 |
| RULER 128K | 95.7 | **<100** | - |
| AIME'25 | 72.9 | **5400** | 其余均>6000 |

**核心成就**：
- LiveCodeBench 以 **85% 更少 Token** 超越 GLM-4.7-Flash-Thinking（+1.6%）
- SWE-bench Verified **62.6%** 达同规模最优
- 多数任务 Token 消耗为最低或接近最低，体现极高 Token 效率

---

## 七、主要结论

1. **架构创新**：48B 总参数仅激活 2.7B（稀疏比远高于同规模竞品），实现对 Qwen3-30B-A3B 吞吐 1.3~1.7× 的提升
2. **FiberPO**：基于纤维丛理论的双尺度 RL 算法，解决 PPO/GRPO 的熵崩溃问题和 GSPO 的 Token 级歧视能力缺失问题，同时降低响应长度并提升准确率
3. **Thinking/Non-thinking 混合 SFT**：混合训练显著提升非思考模式性能，实现自适应认知模式
4. **推理协同设计**：MTP + QAT 联合优化，单卡 MTP 加速比行业最高（1.87×），W4AFP8 吞吐可达 BF16 基线的 1.96×
5. **全量开源**：BF16/FP8/INT4/GGUF 全系列权重均在 HuggingFace 发布

---

## 八、潜在应用价值

- **推荐/搜索系统**：高 Token 效率模型适合在线 Query 理解和候选生成，降低推理成本
- **智能体（Agent）系统**：强软件工程能力（SWE-bench 62.6%）和 TIR 数据支撑复杂工具调用场景
- **边缘部署**：GGUF + DoubleQuant 支持消费级 GPU/CPU 推理
- **强化学习研究**：FiberPO 作为新型 RL 算法框架，可推广到其他 RLHF 任务（多智能体、异构奖励多域训练）
- **长上下文处理**：128K 上下文 + RULER 95.7% 支持 RAG 等长文档应用
