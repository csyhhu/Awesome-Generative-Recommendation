# EAGLE: 推测采样需要重新思考特征不确定性

> 论文：EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty  
> arXiv：[2401.15077](https://arxiv.org/abs/2401.15077)  
> ICML 2024 / PMLR：[li24bt](https://proceedings.mlr.press/v235/li24bt.html)  
> 作者：Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang（北大、微软研究院、滑铁卢大学、Vector Institute）  
> 代码：https://github.com/SafeAILab/EAGLE  

---

## 一、研究动机与问题定义

大语言模型（LLM）推理采用自回归逐 token 解码，延迟高、成本高。**推测采样（Speculative Sampling）** 将生成拆为两阶段：

1. **草稿（Draft）**：用小模型或轻量模块快速提出多个候选 token；
2. **验证（Verify）**：目标 LLM 一次前向并行校验多个 token，接受则一次推进多步。

验证阶段沿用 Leviathan 等人的接受–拒绝规则，可**理论上保证**输出分布与原始 LLM 完全一致（greedy 与非 greedy 均成立），这是与许多“有损加速”方法的关键区别。

传统推测采样的瓶颈在于**草稿模型**：

- 同系列小模型（如 LLaMA2 7B 作 70B 草稿）开销仍大，7B 加速 7B 几乎无收益；
- 跨系列草稿（如 TinyLLaMA）与 Instruct 模板不一致；
- 单独训练草稿模型成本极高（TinyLLaMA 需约 3000B token，而 EAGLE 仅 2–4B token 量级数据）。

已有轻量草稿路线（**Lookahead** 用 n-gram/Jacobi；**Medusa** 用 MLP 从倒数第二层特征预测 token）降低了草稿延迟，但接受率偏低（Medusa 约 0.6）。论文指出：应在**特征（倒数第二层 hidden state）** 层面做自回归，并正视**采样随机性带来的特征不确定性**。

---

## 二、两条核心观察

### 2.1 特征级自回归比 token 级更“好学”

“特征”指目标 LLM **LM Head 之前、倒数第二层** 的 hidden state。特征序列比离散 token 序列更平滑、更有规律；在 Vicuna 7B + MT-bench（T=0）上，仅用特征草稿的加速约 **1.9×**，纯 token 草稿约 **1.5×**。

### 2.2 采样不确定性限制“下一特征”预测

目标 LLM 对下一 token 采样具有随机性：同一上下文 $f_I$ 后，若采样为 “am” 或 “always”，后续特征轨迹会分叉（见图 2）。草稿模型若只知 $f_I$ 而不知**实际采样的 token**，无法唯一确定下一特征——这与 Medusa 对“间隔 token”的歧义类似。

**EAGLE 的解法**：向草稿模型输入**前移一个时间步的 token 序列**（已包含采样结果）。例如预测 $f_{\text{always}}$ 时用 $(f_I, t_{\text{always}})$，预测 $f_{\text{am}}$ 时用 $(f_I, t_{\text{am}})$。该设计几乎不增加结构复杂度，却将加速从约 1.9× 提升到约 **2.8×**（同设置消融）。

---

## 三、EAGLE 方法

**EAGLE**（Extrapolation Algorithm for Greater Language-model Efficiency）是完整的推测采样框架，包含草稿与验证两阶段。

### 3.1 草稿阶段

与 Medusa / Lookahead / 经典推测采样的对比如下（草稿 $t_4, t_5$ 示意）：

| 方法 | 草稿依据 |
|------|----------|
| Speculative Sampling / Lookahead | token → token |
| Medusa | 单一特征 $f_2$ 独立预测多个未来 token |
| **EAGLE** | 特征序列 $(f_1,f_2,\ldots)$ + **前移 token** $(t_2,t_3,\ldots)$ → 自回归预测下一特征 → LM Head 得分布 → 采样 token |

**草稿模型结构**（复用目标 LLM 的 Embedding 与 LM Head，仅训练 **Autoregression Head**）：

1. 输入：特征序列 `(bs, seq_len, hidden_dim)` + 前移 token 序列 `(bs, seq_len)`；
2. Token 经 Embedding 后与特征在最后一维拼接为 `(bs, seq_len, 2×hidden_dim)`；
3. FC 降维 + **单层 Transformer decoder** 预测下一特征 $\hat{f}_{i+1}$；
4. 目标 LLM 的 LM Head 得 $\hat{p}_{i+2}$，采样 $t_{i+2}$，再拼入输入继续自回归。

**树形草稿（Tree Attention）**：类似 SpecInfer/Medusa，用 $m$ 次草稿前向生成深度 $m$、节点数 $>m$ 的 token 树（例：3 次前向草稿 10 个 token）。验证阶段目标 LLM **一次前向** 计算树上各节点概率，并用 **Multi-round Speculative Sampling**（附录算法 1）递归接受/调整分布，保证与目标 LLM 分布一致。

可训练参数量（相对目标模型）：7B→0.24B，13B→0.37B，33B→0.56B，70B→0.99B，Mixtral 8×7B→0.28B。

### 3.2 训练

- **数据**：ShareGPT 约 68k 对话；**不**用目标 LLM 在线生成答案也可（固定数据集敏感性低）；
- **损失**：$L = L_{\text{reg}} + 0.1 \cdot L_{\text{cls}}$  
  - $L_{\text{reg}}$：Smooth L1，拟合下一真实特征；  
  - $L_{\text{cls}}$：对 LM Head 输出分布的交叉熵，直接优化最终 token 目标；
- **增强**：对目标 LLM 特征加 $\mathcal{U}(-0.1, 0.1)$ 噪声，缓解草稿阶段特征误差累积；
- **算力**：70B 对应草稿头在 4×A100 40G 上 1–2 天；7B/13B/33B 可在单卡 RTX 3090 节点 1–2 天完成。

**不微调目标 LLM**，仅外挂一层 decoder，部署简单。

### 3.3 验证阶段

树结构草稿经一次目标 LLM 前向得到各节点概率，按推测采样规则接受 token 并记录对应特征，供下一轮草稿使用。

---

## 四、实验设置

- **目标模型**：Vicuna（7B/13B/33B）、LLaMA2-Chat（7B/13B/70B）、Mixtral 8×7B Instruct；
- **任务 / 数据**：MT-bench（多轮对话）、HumanEval（代码）、GSM8K（数学）、Alpaca（指令跟随）；
- **对比**：Lookahead、Medusa（报告数字来自其技术报告）；同数据训练的 DistillSpec；同系列小模型作草稿的 Speculative Sampling；
- **主要指标**：墙钟 **加速比**、平均接受长度 $\tau$（每次目标 LLM 前向平均接受 token 数）、接受率 $\alpha$（链式草稿、无树 attention 时报告 $n\text{-}\alpha$，$n$ 为草稿自回归步数）；
- **设定**：多数实验 batch size = 1（与 DistillSpec、BiLD 等一致）；输出分布与目标 LLM 一致，故**不**单独评测生成质量。

---

## 五、主要结果

### 5.1 延迟加速（MT-bench）

| 设置 | 典型结论 |
|------|----------|
| T=0（greedy） | LLaMA2-Chat 70B：**2.7×–3.5×**；13B 最高约 **3.76×**（HumanEval）；相对 vanilla **2.1×–3.8×** |
| T=1 | 加速略低于 T=0，仍保持分布无损 |
| vs Lookahead | 约 **1.7×–2.1×** 更快 |
| vs Medusa | 约 **1.5×–1.6×** 更快；草稿接受率约 **0.8** vs Medusa ~0.6 |
| 经典推测采样 | 7B 无合适草稿；13B 用 7B 草稿无提升；33B/70B 仅约 1.12× / 1.88× |
| Mixtral MoE | 约 **1.5×**（MoE 验证阶段需访问更多 expert，加速受限） |

每次目标 LLM 前向平均接受约 **3.2–4.5** 个 token（$\tau$），相对 vanilla 单 token/步显著提升。HumanEval 因代码模板重复度高，加速往往最好（如 LLaMA2-Chat 13B **3.76×**，$\tau \approx 4.52$）。

### 5.2 与 gpt-fast 组合

EAGLE 可与量化、编译等正交叠加。LLaMA2-Chat 7B + EAGLE + gpt-fast 在单卡 RTX 3090、int4 下约 **160.4 tokens/s**（vanilla HuggingFace FP16 约 24.5，gpt-fast int4 约 106.9）。

### 5.3 吞吐量（batch > 1）

在显存允许的最大 batch 下，相对 vanilla 吞吐量约 **2×**（Vicuna 7B、LLaMA2-Chat 70B 均约 1.97×–1.99×）。batch 增大时加速比下降（GPU 算力更饱和）。

---

## 六、消融实验要点

| 因素 | 结论 |
|------|------|
| **树 attention** | $\tau$ 提升约 0.6–0.8，加速比再增约 0.3–0.5；无树仍有约 2.3×–2.7× |
| **输入类型** | feature > token；feature+token 略优；**feature+前移 token（EAGLE）** 对消解随机性收益最大 |
| **训练数据** | 固定 ShareGPT vs 目标 LLM 生成答案：7B 上约 2.78× vs 2.88×，差异小 |
| **特征误差** | $0\text{-}\alpha$ 明显高于 $1\text{-}\alpha$；$1\text{-}\alpha$ 至 $4\text{-}\alpha$ 变化平缓，说明对误差累积较鲁棒 |

---

## 七、与推荐 / 推理扩展的关联（阅读笔记）

本仓库中 [SOLARIS 总结](../Ranking/summary_solaris_inference_scaling.md) 将推荐场景的 FM→VM 嵌入预计算类比为 LLM **推测解码**。EAGLE 从另一角度说明工业界推测式加速的共性逻辑：

- **轻量草稿 + 并行验证**，在严格分布约束下换延迟；
- **在比 token 更底层的表示（特征 / 嵌入）上预测**，往往更易、接受更长；
- **必须把“随机分支”（采样结果 / 实际请求轨迹）”喂给草稿器**，否则表示级预测存在不可消解的歧义。

EAGLE 面向通用 LLM 对话/代码/数学/指令任务；训练一次草稿头即可服务多数据集（零样本权重），适合作为理解 **无损推测采样** 与 **特征级草稿** 的基准文献。

---

## 八、优势与局限

**优势**

- 分布无损（greedy / 非 greedy）；
- 不改动 backbone，训练数据少（2–4B token 量级）、训练时间短；
- 对 Vicuna / LLaMA2-Chat / Mixtral 等多尺寸、多任务泛化；
- 与量化、编译、其他加速栈可叠加。

**局限**

- 仍依赖额外草稿模块与树验证实现，显存略增（如 7B 最大 bs 8→7）；
- MoE、大 batch 场景加速比下降；
- 树结构按经验设计，未针对上下文动态优化（附录说明可进一步调优）。

---

## 九、结论

EAGLE 重新设计了推测采样的**草稿阶段**：在目标 LLM 的**倒数第二层特征**上自回归，并通过**前移一个时间步的 token 序列**消除采样带来的特征不确定性。在 LLaMA2-Chat 70B 上实现约 **2.7×–3.5×** 延迟加速与约 **2×** 吞吐量提升，同时保持生成文本分布不变；相对 Lookahead、Medusa 等同类无损推测方法有显著额外收益，且训练成本远低于训练独立小模型作草稿。

---

## 十、研读讨论记录（问答实录）

---

### Q1：为什么在倒数第二层特征上做自回归，比直接在 token 上更容易呢？

**答：**

论文里的说法可以概括成：**特征序列比 token 序列更「平滑、有结构」，对小草稿模型来说更像回归，而不是在超大词表上做分类。**

#### 1. 表示空间不同：连续 vs 离散

- **Token**：离散 ID，词表往往 3 万+；下一步是「从海量符号里选一个」，本质是**高维分类**，一步错就全错，对小模型很难。
- **倒数第二层特征**（LM Head 前的 hidden state）：高维**连续向量**，相邻位置的表示在训练好的空间里通常**平滑、相关**，更像「在已知语义轨迹上往前推一点」。

论文原话是：自然语言映射成 token 后不规则；而**特征序列更有规律性（more regularity）**，所以在同样参数量下，特征级自回归更容易学好。消融里 Vicuna 7B、T=0：纯特征草稿约 **1.9×**，纯 token 草稿约 **1.5×**。

#### 2. 任务难度：回归 vs 分类

EAGLE 训练草稿时对特征用 **Smooth L1（回归）**，对最终 token 分布再加一项分类损失。  
预测 $\hat{f}_{i+1} \approx f_{i+1}$ 是在**复现目标模型已经算好的内部状态**；预测 $\hat{t}_{i+1}$ 则要直接猜目标模型**采样前**会选哪个离散 token。

小草稿模块容量有限时：

- 回归「逼近一个向量」通常比「猜对词表里的某一个 ID」更稳；
- 难的部分（词表上的细粒度抉择）仍交给**目标 LLM 自己的 LM Head**，草稿只负责把状态推到「差不多对」的位置。

可以理解为：**特征级草稿在帮大模型做「预计算 hidden state」，而不是替大模型做完整的 next-token 决策。**

#### 3. 信息已经过深层编码

到倒数第二层时，上下文、语法、语义大多已编码进 $f_i$。  
下一步 token 的分布是 $p_{i+1} = \mathrm{LM\_Head}(f_i)$——**随机性主要留在 LM Head 这一刀**。在特征空间做 AR，是在**目标模型自己的表示流形**上延续轨迹，和主模型训练目标一致；在 token 上做 AR 还要重新从离散符号推断这些语义，对轻量草稿是绕远路。

Medusa 等也从特征猜 token，同一思路：hidden state 比原始 token 串更可预测。

#### 4. Token 层还有分词噪声

同一语义可能对应不同 BPE 切分（`always` / `am` 等），token 级 AR 要同时建模**语言 + 分词偶然性**。特征空间在很大程度上由主模型**抹平**了这类离散表面差异，序列更「规整」。

#### 不能忽视的另一面：特征级更难「唯一确定」

论文第二个观察：**特征级 AR 容易学，但单独预测「下一特征」有歧义**——因为下一 token 是采样出来的，$f_I$ 之后可能是 “am” 或 “always”，对应不同特征分支。  
所以 EAGLE 要加**前移 1 步的 token**，不是否定「特征更好学」，而是补上采样带来的不确定性。

#### 小结（Q1）

| 维度 | Token 级草稿 | 特征级草稿 |
|------|----------------|------------|
| 输出 | 离散、大词表分类 | 连续向量回归 |
| 序列性质 | 分词不规则 | 主模型表示空间更平滑 |
| 与目标模型分工 | 小包大模型全流程 | 小包只推 hidden，LM Head 仍用目标的 |

完整机制是「特征级 AR + 前移 token 消歧义」，两者一起才到 ~2.8× 那档加速。

---

### Q2：草稿模型的输入是特征（倒数第二层的内容）+ 上一个 token，输出是预测出来的 token 吗？之前的草稿模型一般是怎么做的呢？

**答：**

**不完全是**「特征 + 上一个 token → 直接输出 token」。

#### EAGLE 草稿模型：输入 / 输出

| 环节 | 内容 |
|------|------|
| **输入** | ① 目标 LLM 已经算好的**特征序列** $(f_1,\ldots,f_i)$（倒数第二层 hidden）<br>② **前移 1 步的 token 序列** $(t_2,\ldots,t_{i+1})$（包含「刚采样出来的那个 token」，用来消歧） |
| **草稿头输出** | 预测的**下一个特征** $\hat{f}_{i+1}$（回归，Smooth L1） |
| **再经目标 LM Head** | $\hat{p}_{i+1}=\mathrm{Softmax}(\mathrm{LM\_Head}(\hat{f}_{i+1}))$，再**采样**得到草稿 token $\hat{t}_{i+1}$ |
| **下一步** | 把 $\hat{f}_{i+1}$ 和 $\hat{t}_{i+1}$ 拼回输入，继续自回归；多步 + 树 attention 得到一棵候选 token 树 |
| **最终给用户看的 token** | 还要经过**验证阶段**：目标 LLM 一次前向算树上各位置概率，用推测采样规则接受/拒绝，保证分布与不用 EAGLE 时一致 |

要点：

- 草稿模块的**直接学习目标**是 **feature**，不是单独端到端输出 token；
- **token** 是 LM Head（用**目标模型权重、不训练**）在 $\hat{f}$ 上采样出来的；
- 「上一个 token」在实现里是整条 **shifted token 序列**，不是只喂一个 $t_i$。

论文例子：在 $f_I$ 之后若实际采样是 `always` 或 `am`，要分别用 $(f_I, t_{\text{always}})$ 或 $(f_I, t_{\text{am}})$ 去预测不同的下一特征。

#### 与「以前的草稿模型」对比

**1. 经典推测采样（Speculative Sampling / 小 LM 草稿）**

- **输入**：token（及位置编码等）
- **输出**：一串草稿 **token + 草稿分布** $\hat{T}, \hat{P}$
- **草稿体**：通常是**另一个完整小 LM**（如 7B 给 70B 打草稿）
- **特点**：和主模型同任务、同接口，但**开销大**；7B 给 7B 加速往往不划算

**2. Medusa**

- **输入**：目标 LLM 某一层的 **特征**（倒数第二层附近）
- **输出**：几个 **MLP 头并行**预测未来多个位置的 **token**（不是自回归地一步步预测 feature）
- **特点**：草稿极轻，但**不喂「已采样 token」**，接受率偏低（论文 ~0.6）

**3. Lookahead**

- **输入**：主要是 **token 序列**
- **输出**：n-gram + Jacobi 迭代猜一串 token
- **特点**：几乎不训练；非 greedy 时往往**不能保证**分布无损

**4. DistillSpec / 在线蒸馏类**

- **输入**：token
- **输出**：token + 分布（可能改接受率，有损或半有损）
- **草稿体**：专门蒸馏的小模型

**5. Self-Speculative / 跳层**

- **输入**：走**完整或部分**目标模型前向
- **输出**：中间层直接当下一步草稿

**6. EAGLE（本篇）**

- **输入**：**特征序列 + 前移 token 序列**（拼接后进 FC + 一层 decoder）
- **输出**：先 **$\hat{f}$**，再经 **目标 LM Head → token**

#### 逻辑对比（简化）

```text
经典小 LM:     tokens ──► [小 LM] ──► draft tokens + draft probs
Medusa:        feature ──► [MLP×K] ──► 多个未来 token（并行）
Lookahead:     tokens ──► [n-gram/Jacobi] ──► draft tokens
EAGLE:         features + shifted tokens ──► [小 decoder] ──► next feature
                                              ──► [目标 LM Head] ──► draft token
                                              ──► 树展开多步 ──► 验证接受
```

#### 直接回答（Q2）

1. **输入**：特征序列 + **前移一整段 token**（不是仅「上一个」）。  
2. **草稿头输出**：**下一个特征**；token 由**冻结的目标 LM Head** 采样；真正「算数」的是验证阶段的目标 LLM。

---

### Q3：本文的输入是特征序列 + 前移一整段 tokens，那不是有 2L 个序列长度？这样做的好处是什么呢？

**答：**

**不是时间维上拼成 2L 的两条序列**，而是 **长度仍为 L，每个位置把「特征」和「前移 token 的 embedding」在 hidden 维上拼在一起**。

#### 实际形状：仍是 L，不是 2L

- 特征：`(bs, seq_len, hidden_dim)` → $(f_1,\ldots,f_L)$
- 前移 token：`(bs, seq_len)` → 对齐 $(t_2,\ldots,t_{L+1})$
- 每个位置 $i$：$f_i$ 与 $\mathrm{Embed}(t_{i+1})$ 在最后一维 concat → `(bs, seq_len, 2×hidden_dim)`
- FC 压回 `(bs, seq_len, hidden_dim)`，再过一层 decoder

因此：

- **序列长度**仍是 `seq_len = L`，注意力仍是 $O(L^2)$ 量级，**没有**变成 2L；
- **变长的是通道**：每位置 `hidden_dim` → `2×hidden_dim`，多一次 FC。

论文例子：用 $(f_1,f_2)$ 和 $(t_2,t_3)$ 预测 $f_3$——两边长度都是 **2**，不是 4。

#### 为什么要「特征 + 前移 token」？

**1. 消歧**  
同一 $f_i$ 下采样可能是 `am` 或 `always`，下一特征依赖实际采到的 token。在位置 $i$ 用 $f_i$ + 已发生的 $t_{i+1}$ 预测 $f_{i+1}$；未前移的 $t_i$ 与「用 $f_i$ 预测下一特征」在时间上错位。

**2. 离散 token 几乎无误差**  
消融：feature+unshifted token 的 $0\text{-}\alpha$ 与纯 feature 接近，但 $1\text{-}\alpha$ 更好；前移版再解决**随机性**。

**3. 不 doubling 序列复杂度**  
相对只特征：不 doubling $L$，不换成第二个完整 LM，只是每位置多 1 个 token embedding（用目标 Embedding，不另训大表）。

#### 与「真·2L 输入」对比

| 做法 | 序列长度 | 含义 |
|------|----------|------|
| 特征 L + token L **按时间拼接** | 2L | 注意力、KV 按 2L，更贵 |
| EAGLE：**同 L 位置对齐融合** | L | 每步同时看连续状态 + 离散分支 |

#### 小结（Q3）

- **不是 2L 个位置**，而是 **L 个位置，每位置 2 路信息（$f_i$ + 前移 $t_{i+1}$）**。
- **好处**：几乎不拉长序列的前提下钉死采样歧义，用无噪声 token 稳住草稿自回归。

推理时：$L$ 来自上一轮验证后目标 LLM 已接受上下文上的特征与 token；每接受一批再滚到下一段。

---

### Q4：可以认为本文相对于其他 spec decoding 的方法，是 draft model's input 的信息变得更多吗？

**答：**

可以**部分这样理解**，但更精确的说法是：草稿输入不仅是「更多」，而是 **类型不同 + 对齐方式不同**。

#### 1. 相对「只有 token」的草稿（小 LM、Lookahead 等）

小 LM 也看到整段 **token 历史**，token 数量上未必比 EAGLE「少」。

EAGLE 多出来的是：

- **目标模型倒数第二层特征**（小 LM 草稿拿不到）；
- 特征与 **前移 token** 在**每个位置**融合，而不是让小模型从零推断 hidden state。

相对经典 spec decoding：与其说是「输入更长」，不如说是 **多了特权信息（target 的表示）+ 更贴主模型推理路径**；代价是草稿不再是一个完整 7B，而是一层 decoder。

#### 2. 相对「只有特征」的草稿（Medusa 等）

这一条更接近 **「输入信息更多」**：

| | Medusa 类 | EAGLE |
|---|-----------|--------|
| 特征 | 有（常是某一层的 $f$） | 有，且 **自回归** 用特征序列 |
| Token | 一般不喂已采样分支 | **有，且前移对齐** |
| 作用 | 并行猜多个未来 token | 逐步猜下一特征，再经 LM Head 出 token |

这里确实是 **在特征之外又加了 token 分支信息**，专门消歧；消融里收益最大。

#### 3. 不是「堆更多长度」

- 时间长度仍是 **L**，不是 2L；
- 不是盲目拉长上下文，而是 **每个时间步的条件更丰富**（$f_i$ +「这一步实际采到了哪个 token」）。

#### 归纳（Q4）

- **对 Medusa / 纯特征路线**：可以认为 **是的——草稿条件信息更多了**（特征 + 前移 token）。
- **对整个 spec decoding 家族**：更准确是 **「信息更对、更贴主模型」**——用 target 内部特征 + 采样结果，换掉「再跑一个小 LM」的重草稿；与「小 LM 看更多 token」不是同一维度的比较。

**若只记一个差异点：** EAGLE 的核心不是 input 变长，而是 draft 在预测下一特征时，同时知道「主模型此刻的状态」和「主模型刚走了哪条采样分支」——这两点一起带来更高的接受率 $\alpha$ 和加速比。
