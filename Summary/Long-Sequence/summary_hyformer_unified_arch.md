# HyFormer：统一长序列建模与特征交互的混合 Transformer（CTR 预测）

来源：arXiv: [2601.12681](https://arxiv.org/abs/2601.12681)（RecSys '25，ByteDance AML / Search）

## 综合理解

> **读者一句话理解**：本文提出一种 HyFormer module，motivation 是高效处理序列特征以及与非序列特征的融合。具体来说，先把非序列特征（包含 Target 侧特征）作为 Query，attention 序列特征，得到 Decoded Query，which is then RankMixer as Query Boosting，继续作为下一层的 Query 使用。序列侧采用 Full Attention，或用直接 SwiGLU 映射更新。

**点评：主干流程正确，但有 4 处需要精确化。**

✅ **正确的部分**：整体骨架抓得很准——「Query Gen → 对序列 K/V 做 Cross-Attention → MLP-Mixer 风格 Query Boosting → 输出回喂下一层 Query」确实是 HyFormer Module 的核心循环，且「序列侧可用 Full Attention 或 SwiGLU」也是事实。方向感对。

⚠️ **需精确化的 4 点**：

1. **Motivation 表述偏窄**。「高效处理 + 融合」只是**表象/手段**，不是真正的动机。论文真正动机是**打破两阶段解耦范式的三条瓶颈**（query 表征简化、late fusion、scaling 低效），把「先序列建模、再特征交互」的**单向流**升级为 **Decode ⇄ Boosting 的双向逐层共演化**。如果只讲"高效+融合"，会和 RankMixer/MTGR 等同样声称高效的方案无法区分，丢失了本文的范式创新点。

2. **Query 不是"直接把非序列特征当 Query"**。中间有一步 **Query Generation**：先把所有 NS 特征拼接 + 序列 MeanPool 得到**一个 Global Info 向量**，再用 **N 个独立 FFN** 把它"分头发射"成 N 个**差异化角色**的 Global Query Token（如长期兴趣/短期点击/作者维度等）。少了这一步，就解释不清为什么 N 个 Query 能各司其职、而非简单复制。

3. **Query Boosting 的输入不只 Decoded Query**。是 `[Decoded Query; 原始 NS tokens]` 一起送进 MLP-Mixer——NS tokens 有**直通通路**回到 Mixer，避免 Boosting 时丢失原始异构信号。这一条直通通路是它优于 MTGR 自注意退化的关键。

4. **序列侧漏了中间档 LONGER-style**。原文是**三档**权衡，不是两档：
   - (i) Full Transformer 全自注意力（最高容量）；
   - (ii) **LONGER-style：用紧凑短序列 $S_{\text{short}}$ 对全序列做 Cross-Attention**，复杂度从 $\mathcal{O}(L_S^2)$ 降到 $\mathcal{O}(L_H L_S)$——这是工业部署的**默认/推荐档**，被遗漏等于丢了效率核心；
   - (iii) SwiGLU 无注意力轻量映射。
   此外"SwiGLU 映射更新"更准确说法是：**序列表征 $H_l=\mathrm{SwiGLU}_l(S)$ 再投影成 K/V**，它是"序列编码策略"而非"更新机制"。

> 术语建议：「非聚合序列特征」表述有歧义（易被读成"未聚合的序列特征"），建议统一为「**非序列特征（含 Target 侧特征）**」。另「Full Attention」宜写为「Full Transformer 自注意力编码」以与 Boosting 侧的 Mixer 区分。

**一句话总评**：理解在**模块级**准确，在**范式级**略偏——补上「Query Generation 的 N 个独立 FFN 分发」「Boosting 拼回 NS tokens」「序列侧三档含 LONGER-style」这三处后即与论文一致；最关键的是把 motivation 从"高效+融合"修正为"打破单向两阶段、做双向交替优化"。

---

## 研究动机

工业级大规模推荐模型（Large Recommendation Models, LRMs）需在严格时延与算力约束下，**同时**建模长程用户行为序列与异构非序列特征（用户画像、上下文、交叉特征等）。当前主流架构普遍采用**解耦的两阶段范式**（"先长序列建模，再异构特征交互"）：

1. **长序列建模阶段**：用 LONGER、SIM、TWIN、ETA 等基于 query-token 的序列压缩器，把超长行为序列压成少量 token；
2. **特征交互阶段**：将压缩后的序列 token 与稠密特征通过 RankMixer / Wukong / Full Transformer 等 token-mixing 模块融合。

论文指出该范式存在三条**根本性局限**：

- **Query 表征过于简化**：用于聚合长序列的 query token 通常只来自候选相关或少量全局特征，能利用的上下文信息受限；而直接增加 query 数量会在 KV-Cache / M-Falcon serving 机制下显著拖慢推理。
- **延迟融合（late fusion）且单向**：序列 token 与异构非序列 token 的交互只发生在模型后段，跨特征推理被推迟到序列压缩之后，交互浅而隐式，早期层无法受益于跨域上下文。
- **Scaling 效率低**：堆深度/参数主要提升孤立组件而非联合表征，额外算力难以高效转化为端到端建模收益。

为此，作者提出 **HyFormer**：一个把长序列建模与特征交互**紧耦合到同一主干**的统一混合 Transformer，通过 **Global Tokens** 作为序列与异构特征之间的共享语义接口，把建模任务重构成一个**交替优化过程**。

## 核心方法：HyFormer = Global Tokens + Query Decoding ⇄ Query Boosting 交替堆叠

### 1) 问题定义

给定用户 $u$ 的行为历史 $S=\[i\_1^{(u)},\dots,i\_K^{(u)}]$、非序列描述符 $u$ 与候选 item $v$，估计交互概率 $P(y=1\mid S,u,v)\in\[0,1]$，以二元交叉熵训练：
$$\mathcal{L}=-\frac{1}{|\mathcal{D}|}\sum\_{(S,u,v,y)\in\mathcal{D}}\big\[y\log\hat y+(1-y)\log(1-\hat y)\big],\quad \hat y=f\_\theta(S,u,v).$$

### 2) 整体框架

每层 HyFormer 由两个互补模块组成，**交替执行**，构成跨层迭代优化：

- **Query Decoding（查询解码）**：用 MLP 把非序列特征扩张成多个语义 **Global Tokens**（即序列 query），对长序列的**逐层 K/V**做 cross-attention，让全局上下文直接塑造序列 token 的表征；
- **Query Boosting（查询增强）**：用 MLP-Mixer 风格的 token mixing 强化已解码 query 与非序列 token 之间的跨 query、跨序列异构交互。

二者紧耦合，使得"长序列建模"和"特征交互"不再是先后两段，而是**双向、逐层共演化**的信息流。

### 3) Query Generation（查询生成）

**Input Tokenization**：沿用 RankMixer 的分词策略，支持 semantic grouping（按语义分组：user/context/behavior）或 auto-split（展平后均匀切分）。HyFormer 选用 **semantic grouping** 以保留结构化归纳偏置与可解释性。

**Query 生成**：将所有非序列特征向量 $F\_1,\dots,F\_M\in\mathbb{R}^{1\times D}$ 拼接，并对行为序列做 pooling 得到序列级摘要作为额外共享输入，再经轻量投影生成 $N$ 个 query：
$$Q=\big\[\mathrm{FFN}\_1(\text{Global Info}),\dots,\mathrm{FFN}\_N(\text{Global Info})\big]\in\mathbb{R}^{N\times D},$$
$$\text{Global Info}=\mathrm{Concat}\big(F\_1,\dots,F\_M,;\mathrm{MeanPool}(Seq)\big).$$

为维持 serving 效率，模块支持特征选择与可选 query 压缩，使生成 query 数量稳定但仍有足够表征容量。

**关于深层 query 的再利用（FAQ 见附录）**：Query Generation（从 Global Info 经 MLP 扩展 query）**只在模型最开始做一次**。之后第 $l$ 层的输入 query $Q^{(l)}$ 来自：上一层完成 `CrossAttn → 拼 NS tokens → MLP-Mixer token mixing → 残差` 之后的输出，截取其中前 $N$ 个 global token 位置。Cross-Attention 只是这条链的中间一步。若深层重新运行 Query Generation，会把前面 Cross-Attention 吸收的序列信息与 Boosting 积累的交互结果全部重置，等价于退回 BaseArch "一次性序列压缩 + 单次特征交互"的效果。

### 4) Query Decoding：序列表征编码 + Query Cross-Attention

> **术语说明**：本节中的"跨注意力解码"就是标准的 **Query Cross-Attention**（以 Global Tokens 为 Q、序列编码结果为 K/V 的 Multi-Head Cross-Attention），名称沿用 NLP 中 decoder→encoder 的说法，后文统一简称为"Query Attention / 跨注意力"。

#### Sequence Representation Encoding（三档容量-效率权衡）

给定行为序列 $S$，每种策略产出逐层 K/V 表示 $(K^{(s)}\_l,V^{(s)}\_l)$：

- **(i) Full Transformer 编码**（最高容量）：$H\_l=\mathrm{TransformerEnc}\_l(S)$，全自注意力捕捉细粒度长程依赖。
- **(ii) LONGER 风格高效编码**：用紧凑短序列 $S\_{\text{short}}$（长度 $L\_H\ll L\_S$）作 query，对全序列 $S$ 做 cross-attention：$H\_l=\mathrm{CrossAttn}(S\_{\text{short}},S,S)$，复杂度从 $\mathcal{O}(L\_S^2)$ 降到 $\mathcal{O}(L\_H L\_S)$。
- **(iii) Decoder 风格轻量编码**（时延极敏感场景）：无注意力，仅 $H\_l=\mathrm{SwiGLU}\_l(S)$，用上下文容量换最低算力。

各策略下统一投影得到逐层 K/V：
$$K\_l=H\_l W\_l^K,\qquad V\_l=H\_l W\_l^V.$$
**K/V 每层重算**，让序列特征随解码深度协同演化，同时支持灵活部署配置。

#### Query Cross-Attention（跨注意力 / Query Attention）

对第 $l$ 层、行为序列 $S$：
$$\tilde Q_{(l)}=\mathrm{CrossAttn}\big(Q_{(l)},K_{(l)},V_{(l)}\big),\quad Q_{(l)}\in\mathbb{R}^{N\times D}.$$
这一步就是标准的 **Query Attention**。全局非序列特征以 Q 的角色直接 attend 长行为序列 K/V，把上下文信号注入序列感知的 query 表征；输出维度保持为 $N\times D$（Query 长度），原始长序列仍以 K/V 形式存在，**之后不再额外做序列级 attention**。解码后的 $\tilde Q_{(l)}$ 直接作为后续 Query Boosting 模块的语义接口。

### 5) Query Boosting：MLP-Mixer 风格 token 混合

解码后的 query 已含序列信息，但与静态异构非序列特征的交互仍不足。Boosting 模块显式跨 query 混合并注入额外非序列信号。

统一 query 表征：
$$Q=\[\tilde Q\_{(l)},F\_1,\dots,F\_M]\in\mathbb{R}^{T\times D},\quad T=N+M.$$

**MLP-Mixer 风格 token mixing**（受 RankMixer 启发）：每个 query token $q\_t$ 切成 $T$ 个通道子空间 $q\_t^{(h)}\in\mathbb{R}^{D/T}$；对每个子空间索引 $h$，跨所有 token 位置拼接聚合：
$$\tilde q\_h=\mathrm{Concat}\big(q\_1^{(h)},q\_2^{(h)},\dots,q\_T^{(h)}\big)\in\mathbb{R}^{D},$$
$$\hat Q=\[\tilde q\_1,\dots,\tilde q\_T]\in\mathbb{R}^{T\times D}.$$

随后逐 token 轻量 FFN 精炼：$\widetilde Q=\mathrm{PerToken\text{-}FFN}(\hat Q)$，对各 query token 独立做前向变换，子空间级精炼且保持线性复杂度。最后残差稳定优化：
$$Q\_{\mathrm{boost}}=Q+\widetilde Q.$$
增强后的 query 进入下一层 HyFormer，让深层用更丰富、更具表达力的表征继续"审问"长序列。

### 6) HyFormer Module：交替堆叠

每层 = Query Decoding 块 + Query Boosting 块：
$$\widehat Q^{(l)}=\mathrm{CrossAttn}\big(Q^{(l-1)},K^{(l)},V^{(l)}\big),$$
$$\widetilde Q^{(l)}=\mathrm{QueryBoost}\big(\mathrm{Concat}(\widehat Q^{(l)},\mathrm{NS\ Tokens})\big).$$
多层堆叠使语义 query 逐层精炼，深层以更高抽象层次概括长序列；顶层输出送入下游 MLP 出 CTR 预测。

### 7) Multi-Sequence Modeling（多序列独立建模）

工业场景常含多条异构序列（如视频观看序列、商品购买序列）。由于不同序列特征空间与语义差异大，作者实测 MTGR/OneTrans 的**简单序列合并**会导致显著性能退化（见消融 Table 3：HyFormer + Merge Seq 相对 HyFormer 损失 0.06% AUC）。HyFormer 对此采取**"每条序列独立 Query Decoding + Query 级 Token Mixing 完成跨序列交互"**的设计，具体如下：

1. **一组序列 = 一组独立的 Global Query Tokens**：本文实验 3 条序列（long-term / search / feed）→ 3 组 global tokens，每组 1 个，合计 3 个 global token + 13 个 NS token = 16 个 token，与 MLP-Mixer 输入对齐。
2. **Query 的来源与区分**：对第 $i$ 条序列 $S_i$，其专属 Query 由独立 FFN 从共用 NS 特征与各自序列的 MeanPool 合成：
   $$Q^{(i)}=\mathrm{FFN}^{(i)}\big(\mathrm{Concat}(F_{1..M},\mathrm{MeanPool}(S_i))\big).$$
   即每条序列的 Query 先"看到"**自己序列的整体池化信息**，避免共享 query 带来的语义混淆。此处 Query 不做 recent-k 等 item 采样（与 LONGER 不同），序列压缩工作交给序列编码模块（Full Transformer / LONGER-style / SwiGLU 三选一）。
3. **独立 Decoding**：每条序列独立算自己的逐层 K/V，用自己的 Query^{(i)} 做独立的 Cross-Attention，互不干扰，保留序列特异性。
4. **Query 级跨序列交互**：三条序列解码后的 $Q^{(1)},Q^{(2)},Q^{(3)}$（合计 $N$ 个位置）与 13 个原始 NS tokens 拼成 $16\times D$，**一起**送入 Query Boosting 的 MLP-Mixer。Token mixing 过程中自动完成跨 query（跨序列）、跨 NS 的高阶交互，无需再显式把序列拼起来。
5. **分配灵活性**：重要序列可分配 $N_i>1$ 个 Global Query Token，以更细粒度语义维度覆盖该序列的兴趣子空间，无需把其他序列的 Query 也一起扩容。

### 8) 训练与部署优化

- **GPU Pooling for Long-Sequence**：用户长序列特征体量极大，带来显著的 H2D 内存拷贝与 host 内存压力。但真正唯一的特征 ID 通常只占约 **25%**。利用该稀疏性去重：执行前以压缩 embedding table 存储，前向算子在 GPU 上重建原始序列特征，反向算子把序列梯度聚合回 embedding table，再上传更新稀疏参数，大幅降低传输成本与 host 内存占用。
- **Asynchronous AllReduce**：用异步 AllReduce 让第 $k$ 步梯度同步与第 $k{+}1$ 步前向/反向计算重叠，消除通信气泡、最大化 GPU 利用。代价是 dense 参数引入**一步延迟**：$W\_k=W\_{k-1}+g\_{k-1}$；而稀疏参数本地梯度算完即更新：$W\_k=W\_{k-1}+g\_k$（比 dense 提前一步）。实测该混合更新调度的轻微时间不一致不影响收敛与性能。

## 实验结果

### 实验设置

- **数据集**：Douyin Search System CTR 预测，70 天连续日志、**30 亿样本**；含 user / query / document / cross 特征与多条序列。
  - Long-term sequence：上限 **3000**；
  - Search sequence / Feed sequence：各 top-50，经 Query Search 模块过滤。
- **Baseline**：分两类。
  - **BaseArch（两阶段）**：序列建模 ∈ {LONGER, Full Transformer} × 特征交互 ∈ {RankMixer, Full Transformer, Wukong}；
  - **UniArch（统一块）**：MTGR / OneTrans（分 LONGER 版与 Full Transformer 版）。
- **指标**：Query-level AUC、Params（×10⁶）、训练 FLOPs（×10¹²，batch=2048）。
- **实现**：13 个非序列 token + 3 个 global token（每条序列一个）= 16 token；MLPMixer 输入 token 数对齐到 16；64-GPU 集群。

### 1) 总体性能（工业数据集）

| Sequence Modeling                   | Feature Interaction | AUC↑       | ΔAUC       | Params (×10⁶) | FLOPs (×10¹²) |
| ----------------------------------- | ------------------- | ---------- | ---------- | ------------- | ------------- |
| **BaseArch: Traditional Two-Stage** | <br />              | <br />     | <br />     | <br />        | <br />        |
| LONGER                              | RankMixer           | 0.6478     | —          | 386           | 3.5           |
| LONGER                              | Full Transformer    | 0.6472     | −0.09%     | 416           | 6.2           |
| LONGER                              | Wukong              | 0.6465     | −0.20%     | 385           | 5.2           |
| Full Transformer                    | RankMixer           | 0.6481     | +0.05%     | 388           | 6.6           |
| Full Transformer                    | Full Transformer    | 0.6474     | −0.06%     | 418           | 9.3           |
| Full Transformer                    | Wukong              | 0.6468     | −0.15%     | 387           | 8.3           |
| **UniArch: Unified-Block**          | <br />              | <br />     | <br />     | <br />        | <br />        |
| MTGR/OneTrans (w/ LONGER)           | —                   | 0.6480     | +0.03%     | 406           | 6.6           |
| MTGR/OneTrans (w/ Full Transformer) | —                   | 0.6483     | +0.08%     | 450           | 21.9          |
| **HyFormer (Ours)**                 | —                   | **0.6489** | **+0.17%** | 418           | **3.9**       |

要点：

- HyFormer 以 **0.6489 AUC** 取得最优，相对生产基线 LONGER+RankMixer **+0.17% AUC**，且 **FLOPs 仅 3.9×10¹²**，远低于多数竞品（含高表现的 MTGR Full Transformer 的 21.9×10¹²）。
- BaseArch 内：特征交互侧 RankMixer 一致优于 Self-Attention 与 Wukong；序列侧加全自注意力一般有小幅增益；最强组合 Full Transformer + RankMixer（0.6481）仍不及 HyFormer，印证单向信息流的固有局限。
- MTGR/OneTrans 用 Self-Attention 做特征交互，既拉低 AUC 又严重拖累效率；其把 Global+Seq token 合并做 key、只用 Global token 做 query 的设计，使 query 更易 attend 自身而非序列。HyFormer 强制**分离信息流**：先把具体序列 item 信息压缩吸收进 Global Tokens，再在不同抽象 Global Tokens 间做交互，两步跨层反复堆叠。此外 HyFormer 可独立调交互层/维度与序列层/维度，scaling 更灵活。

### 2) 消融实验

**Query Global Context**

| Config                                               | AUC    | ΔAUC   |
| ---------------------------------------------------- | ------ | ------ |
| HyFormer                                             | 0.6489 | —      |
| Query w/o Seq Pooling Tokens                         | 0.6486 | −0.05% |
| Query w/o Nonseq and Seq Pooling Tokens（仅 target 特征） | 0.6484 | −0.08% |

退回只含 target 特征的原始 query 严重限制后续深度交互（−0.08%）；去掉跨序列 pooling token 也损失 0.05%，证实在 HyFormer 内跨序列交互确有贡献。

**Query Boosting / 架构**

| Config                     | AUC    | ΔAUC   |
| -------------------------- | ------ | ------ |
| HyFormer                   | 0.6489 | —      |
| HyFormer w/o Global Tokens | 0.6484 | −0.08% |
| BaseArch w/ Global Tokens  | 0.6480 | −0.14% |
| BaseArch w/o Global Tokens | 0.6478 | −0.17% |

即便给 BaseArch 注入丰富 query 信息，缺乏深化交互也封顶收益（仅 +0.03%）；在 HyFormer 框架内扩展 query 信息带来 +0.08% 的更大增益。

**Multi-Sequence Modeling**

| Config               | AUC    | ΔAUC   |
| -------------------- | ------ | ------ |
| HyFormer             | 0.6489 | —      |
| HyFormer + Merge Seq | 0.6485 | −0.06% |

序列合并 + query 共享损失 0.06%，说明独立 token 建模能保留各序列特异性；这也部分解释 MTGR/OneTrans 为何不及 HyFormer。

### 3) Scaling 分析（200M → 1B+ 参数）

- **参数 scaling**：HyFormer 始终优于基线 LONGER+RankMixer，且保持更陡的 scaling 斜率——交替堆叠带来的双向信息流使其从加深深度中获得更大增益。
- **FLOPs scaling**：AUC 随 FLOPs 呈强幂律上升，算力扩张能让模型以更丰富的信息处理序列，受益于初始 query 扩张与 MLP-Mixer 对 query 的反复增强。
- **Sparse Dim scaling**（序列 token 输入维 / side info 丰富度）：

| Seq Length | Arch     | Sparse Dim | AUC    | ΔAUC   | ΔAUC Gap |
| ---------- | -------- | ---------- | ------ | ------ | -------- |
| 1k         | BaseArch | 64         | 0.6478 | —      | —        |
| 1k         | BaseArch | 224        | 0.6484 | +0.09% | —        |
| 1k         | HyFormer | 64         | 0.6489 | —      | —        |
| 1k         | HyFormer | 224        | 0.6497 | +0.12% | +0.03%   |
| 3k         | BaseArch | 64         | 0.6486 | —      | —        |
| 3k         | BaseArch | 224        | 0.6490 | +0.06% | —        |
| 3k         | HyFormer | 64         | 0.6499 | —      | —        |
| 3k         | HyFormer | 224        | 0.6507 | +0.12% | +0.06%   |

从 64 维（3 类 side info：item ID、search query textnet 分类、timestamp）扩到 224 维（7 类：再加 search query ID、author ID、event ID、playtime），HyFormer 增益（+0.12%）一致大于基线；且序列越长差距越大（1k 时 +0.03% gap，3k 时 +0.06% gap）。说明 HyFormer 把更丰富的全局信息纳入序列 query、加之 LONGER 与 Mixer 间的双向信息流，能把序列 K/V 扩展的价值放得更大。

### 4) 在线 A/B 测试（Douyin Search，高流量生产）

| Online Test Metric                 | Gain    |
| ---------------------------------- | ------- |
| Average Watch Time Per User ↑      | +0.293% |
| Video Finish Play Count Per User ↑ | +1.111% |
| Query Change Rate ↓                | −0.236% |

（Query Change Rate = $N\_{\mathrm{reform}}/N\_{\mathrm{total}}$，衡量用户手动把 query 改得更具体的概率，作为负向搜索体验指标。）相对 RankMixer 基线，三项关键指标均显著正向，验证 HyFormer 在十亿级用户生产环境的实用价值。目前已全量部署于 ByteDance，日均服务数十亿用户。

## 结论与启示

- HyFormer 用 **Global Tokens** 重新定义长序列建模与特征交互的角色：把"先序列建模后特征交互"的单向流，升级为 **Query Decoding ⇄ Query Boosting 的双向、共演化**范式，把 LRM 建模视作一个交替优化过程。
- 该设计既做更深、更早、双向的异构交互，又保持高效（3.9×10¹² FLOPs）；多序列独立建模避免合并退化；GPU Pooling 与异步 AllReduce 进一步支撑工业部署。
- 离线 + 在线实验共同验证：从单向信息流升级为双向共演化，能带来更优精度与更陡的 scaling 曲线，提升未来工业 LRM 的 scaling 上限。

**与相关工作的关系**：

- 序列侧直接继承 LONGER（同一作者团队）的 long-sequence 高效编码与 KV-Cache serving 思路，并把 LONGER 的 global token 思想扩展为跨层交替优化的全局语义接口；
- 交互侧沿用 RankMixer 的 MLP-Mixer token mixing 作为轻量 boosting；
- 相对 MTGR / OneTrans 等"统一块"方法，HyFormer 强调 query 数量受限以保 serving 效率，并通过分离式两步信息流（先吸收序列 item 进 Global Tokens、再做 Global Tokens 间交互）避免 MTGR 中 query 自注意自身的退化，从而在更少 FLOPs 下取得更高 AUC。

---

## 附：讨论问答（FAQ）

### Q1. HyFormer Module 是否可以理解为：把非序列特征（含 Target）用 MLP 生成 Global Tokens → 与序列 K/V 做 Cross-Attention → 解码结果用 RankMixer 加深表达？
基本正确，补两点：(1) Query Boosting 的输入**不只**是解码 Query，而是 `[decoded queries; 原始非序列 tokens]`，原始 NS tokens 有直通通路回到 Mixer；(2) 这一组合**跨层交替堆叠**：Boosting 输出的前 N 个 token 取下一层 Query，深层 Cross-Attention 继续"审问"全新一层的序列 K/V，而非只做一次序列压缩 + 一次交互。

### Q2. Query Generation 是否就是对非序列特征做 RankMixer？为什么写得复杂？深层不重新生成 Query 时，第一层结果还有用吗？
不是 RankMixer。区别：Query Generation 输入是**单个 Global Info 向量**（NS 拼接 + MeanPool(Seq)），用 N 个独立 FFN"扩展"出 N 个差异化角色的 Query Token（没有跨 token 交互）；RankMixer 输入是**已有的 T 个 tokens**，在它们之间做 token mixing 以交换信息。两者目的完全不同。

关于深层 Query：Query Generation **只运行一次**在最前面。每层 Query^{(l)} 来自上一层 `CrossAttn → 拼 NS → MLP-Mixer → 残差` 输出的前 N 个位置。如果深层重新运行 Query Generation，会把前面吸收的序列信息与交互结果全部重置，等价于 BaseArch 两阶段。原文表述"复用上一层 cross-attention 输出作为更新的 query"略简化，实际是上一层完整 Boosting 输出才是下一层 Query。

### Q3. LONGER 里的紧凑短 Query 与 Query Generation 的 Query 有何不同？解码后序列长度是否变成 Query 长度？之后还做 Query Attention 吗？
紧凑短 Query（LONGER 的 $S_{\text{short}}$）与 QueryGen Query 的差异：(a) 来源：前者从序列本身采样（recent-k 等），是真实 item embedding 子集；后者由 NS + MeanPool(Seq) 经 MLP 合成，是候选/上下文驱动的全局语义查询。(b) 角色：前者服务于"序列编码内部"的长→短压缩；后者直接"审问"序列编码后的 K/V。

Cross-Attention 输出尺寸是 $N\times D$（Query 长度），**长序列仍以 K/V 形式保持 $L_S$ 不变**。Cross-Attention 就是 Query Attention（只是 NLP 习惯叫"解码"），做完就接 Query Boosting，**不会再做序列级 Attention**。

### Q5. Multi-Sequence 中每条序列的 Query Token 如何设计？用哪些非序列特征？是否从序列中采样？
做法：(1) 每条序列配**独立的一组 Global Query Tokens**（本文实验每条序列 1 个，合计 3）；(2) 各自 Query 由 **共用的 NS 特征（User/Query/Candidate/Context/Cross 共 13 个）+ 各自序列的 MeanPool** 经独立 FFN 合成；(3) **不做序列采样**，序列侧压缩工作交给序列编码模块（Full Transformer / LONGER-style / SwiGLU）；(4) 解码后所有序列的 Query 与 NS tokens 拼成 16 个 token 一起走 MLP-Mixer，跨序列交互在此处完成；(5) 重要序列可配 $N_i>1$ 个 Query 以获得更充分的兴趣维度。

### Q6. HyFormer 与 LONGER/STCA 的差异是否只有 Query Boosting（Query Attention 后再跑一个 RankMixer）？
抓到了**最直观的增量**，但不止一层：确实，Query Boosting 是最显眼的模块差异，但它同时引入了 (a) **Query 来源扩充**：从少量 candidate 相关 → 全部 NS + 各序列 MeanPool + 独立 FFN 分头发射；(b) **信息流方向翻转**：从单向（序列压一次→交互一次）变成 双向交替堆叠（Decode⇄Boosting），且每层序列 K/V 重算，深层 Query 越"审问"越深；(c) **多序列独立 Query 架构**，原生支持跨序列交互。

简而言之：Query Boosting 是**触发范式切换的模块**，但真正的贡献是把"先序列建模、再特征交互"的单向两阶段，变成"交替优化"的统一主干——这一点不能用"只加一个 RankMixer 步骤"完全概括：就是只加一个RankMixer就可以概括。

### Q7. LONGER-style 编码把序列长度从 $L_S$ 降到 $L_H$，后面的层依旧沿用 $L_H$ 吗?
要分两层意思看，避免混淆：
-  **K/V 长度（Query Cross-Attention 看到的序列侧长度）**：每层都是 $L_H$。公式 $H_l=\mathrm{CrossAttn}(S_{\text{short}},S,S)$ 中 $S_{\text{short}}$（长 $L_H$）作 Q、$S$（长 $L_S$）作 K/V，输出 $H_l$ 长 $L_H$，故 $K_l,V_l$ 长 $L_H$，后续 Query Decoding 复杂度 $\mathcal{O}(N\cdot L_H)$。这一长度在所有层稳定为 $L_H$，既不回到 $L_S$ 也不继续缩小。
- **编码输入**：每层都从**原始全长 $S$（$L_S$）**重新压缩到 $L_H$，**不是**把上一层输出 $H_{l-1}$（$L_H$）当下一层输入再编码。**铁证**：TeX 第 741 行原文 "S is used as both keys and values"——K/V 永远是原始 $S$（长 $L_S$），而非 $H_{l-1}$；$S_{\text{short}}$ 跨层固定；层间差异完全来自**层特异性参数**（每层 CrossAttn 与 $W^K_l/W^V_l$ 不同），而非级联上一层 $H$。注意：级联只发生在 **Query 路径**（第 706-708 行 "each layer reuses the queries from the previous layer"），不发生在序列 K/V 路径。
 归纳："序列侧 K/V 稳定在 $L_H$"→**是**；"上一层 $L_H$ 输出作为下一层编码输入"→**否**。

 **输入/输出维度（LONGER-style 单层编码）**：
 | | 张量 | 维度 | 角色 |
 |---|---|---|---|
 | 输入 | $S$ | $\mathbb{R}^{L_S \times D}$ | K 与 V（原始全长序列） |
 | 输入 | $S_{\text{short}}$ | $\mathbb{R}^{L_H \times D}$ | Q（紧凑短序列） |
 | 输出 | $H_l$ | $\mathbb{R}^{L_H \times D}$ | Cross-Attn 输出，长=Q 长=$L_H$ |
 | 输出 | $K_l=H_l W^K_l,\;V_l=H_l W^V_l$ | $\mathbb{R}^{L_H \times D}$ | 喂给 Query Decoding |

 设计动机：避免级联式信息瓶颈（$L_H\to L_H'\to L_H''$ 逐层再压会持续丢信息），让每层对原始全序列 $S$ 保持"新鲜视图"、用不同层参数关注 $S$ 的不同子模式，同时把解码侧开销稳定锁在 $\mathcal{O}(N\cdot L_H)$。论文"K/V recomputed at each layer...evolve jointly with decoder depth"中的 evolve 指"靠层特异性参数逐层演化"，非"靠级联上一层 H 演化"。

  旁证（其他两档 K/V 长度）：(i) Full Transformer 自注意力保长 → 每层 K/V 长 $L_S$（最贵）；(iii) SwiGLU 位置式 FFN 保长 → 每层 K/V 长 $L_S$，但无 $\mathcal{O}(L_S^2)$ 自注意力开销。三档差异既在"生成 $H_l$ 开销"也在"K/V 长度"，LONGER-style 是唯一把 K/V 长度也降到 $L_H$ 的档。

Q8. LONGER-style 编码如何更新序列特征、序列如何变化？是否做了两次 cross-attn?
- **每层确有两次 cross-attn**：#1 编码（$Q=S_{\text{short}}\, L_H$，$K/V=S\, L_S$ → $H_l\, L_H$，**更新序列**）；#2 解码（$Q=$ 全局 Query $N$，$K/V=K_l,V_l$ → decoded query，**query 吸取序列**）。两个 Q 不同（$S_{\text{short}}$ 序列派生 vs 全局 Query 非序列生成）。
- **编码机制**：$H_l[i]=\sum_j \mathrm{attn}(S_{\text{short}}[i], S[j])\cdot S[j]$，即 $S_{\text{short}}[i]$ 引导下对 $S$ 做加权和；每个 $H_l[i]$ 是 $S$ 中相关子集的摘要。
- **序列单层内变化**：长度 $L_S \to L_H$（压缩）；内容从"$L_S$ 个原始 item embedding"变"$L_H$ 个 attention 加权摘要"。
- **序列跨层变化（修正 Q7 过度自信）**：Q7 按公式字面读"每层 K/V=原始 $S$"（层特异性参数 → 不同视图，参数化演化）。但论文"evolve jointly with decoder depth"+"increasingly expressive representations"更像**逐层级联精炼**（$H_{l-1}\to H_l$）：layer 0 把 $S$（$L_S$）压成 $H_0$（$L_H$），深层把 $H_{l-1}$（$L_H$）精炼成 $H_l$（$L_H$）。"S is used as both keys and values"只说明 cross-attn 里 K/V 同源，**不能排除级联**（级联下"该层序列输入"即 $H_{l-1}$，仍同源）。从措辞看级联读法更可能符合作者意图，但公式写 $S$、无显式 $H_{l-1}$，无法 100% 钉死，建议以图 hyformerv3 / 代码为准。两种读法下"序列侧更新机制"相同：$S_{\text{short}}$ 作 Q、序列作 K/V 的 cross-attn 把序列压成 $L_H$ 个摘要；区别仅在"深层 K/V 是原始 $S$ 还是 $H_{l-1}$"。

