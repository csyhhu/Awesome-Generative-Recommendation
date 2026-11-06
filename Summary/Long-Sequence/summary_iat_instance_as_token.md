# IAT: Instance-As-Token Compression for Historical User Sequence Modeling in Industrial Recommender Systems

- **arXiv**: [2604.08933](https://arxiv.org/abs/2604.08933)
- **作者**: Xinchun Li, Ning Zhang 等（ByteDance）
- **领域**: 工业推荐系统、用户历史序列建模、序列特征工程

---

## 1. 问题与动机

现代排序模型在 **序列建模范式**（DIN、LONGER、Transformer 等）上已很成熟，但下游仍依赖 **手工构造的行为序列特征**（item ID、类目、交互类型、时间戳等）。这带来两重瓶颈：

1. **信息容量受限**：存储/传输/算力约束下难以大规模、细粒度地展开特征，细粒度信号被丢弃，序列表示稀疏。
2. **工程成本高**：特征设计、开发、验证周期长，难以快速扩展。

与此同时，工业训练样本本身往往包含 **数千维特征**，能较完整地刻画一次历史交互；若只把少量手工序列特征喂给强大的 \(\mathcal{F}_{\text{seq}}\)，会形成 **信息瓶颈**，限制先进架构的 scaling。

**核心想法**：把每条历史 **训练实例（training instance）** 压缩成一个稠密 token（**Instance Embedding, InsEmb**），再在下游用标准序列模型建模「实例序列」，即 **Instance-As-Token (IAT)**。

---

## 2. 方法：两阶段框架

### 2.1 基础排序模型

特征分为序列 \(x_{\text{seq}}\) 与非序列 \(x_{\text{non-seq}}\)：

\[
h = \mathcal{F}_{\text{interaction}}\left(x_{\text{non-seq}}, \mathcal{F}_{\text{seq}}(x_{\text{seq}})\right), \quad \hat{y} = P(y=1 \mid u, v, h)
\]

IAT 不改变「上层怎么建模序列」的大方向，而是 **替换/增强 \(x_{\text{seq}}\) 的信息来源**。

### Stage I：IAT Compression（源模型）

目标：为每个训练实例生成低维 **InsEmb**（默认 \(D=64\)，原始表示约 6000 维），写入中心化存储。

#### （1）时序序源模型（Temporal-Order）

- 在 base 模型输出 \(h\) 后加 **压缩 + 解压** MLP：
  - \(h_{\text{compress}} = \sigma(W_1 h + b_1)\) → **存为 InsEmb**
  - \(h_{\text{decompress}}\) 再接预测头，用 BCE 端到端训练，保证压缩保留高价值信息。
- **按时间顺序逐条实例** 压缩，与 batch/stream 组织方式无关，工程简单。
- 缺点：单实例独立压缩，源模型可能略弱于 base（论文约 **-0.05% AUC**）；InsEmb 缺乏跨实例序列感知。

#### （2）用户序源模型（User-Order，推荐）

- 训练样本按 **用户聚合、时间排序** 组 batch（同用户历史实例在同一 batch）。
- 先对每条实例做同样压缩得到 \(H_{\text{com\_batch}} \in \mathbb{R}^{T \times D}\)。
- 引入 **Source Instance Transformer (SIT)**：因果 mask 的 Transformer，让每条实例的表示能看历史实例。
- **关键设计**：存储的是 **SIT 之前** 的 \(H_{\text{com\_batch}}\)（存 SIT 之后会损失基础信息、表示趋同，下游变差）。
- 优点：源模型 AUC 可提升约 **+0.6%**；InsEmb 隐式具备序列建模能力，**与下游更对齐**。
- **流式训练**：batch 内多用户时，从 PS **检索** 该用户过去 \(T-1\) 条 InsEmb，与当前条 concat 后过 SIT（`stop_gradient` 于历史），只更新并写入当前实例 InsEmb。

### Stage II：IAT Sequence Modeling（下游模型）

1. 按 **UID + 请求时间戳** 从 KV 取历史 **InsID 序列** 及可选 **核心侧信息**（多任务 label、归因相关时间戳等），截断防泄漏。
2. 从 **Embedding PS** 用 InsID 拉取 InsEmb。
3. **InsEmb Adaptation MLP** 映射到下游特征空间。
4. **InsToken** = Concat(适配后 InsEmb, 侧信息 embedding)。
5. **Query**：将下游当前请求的 seq + non-seq token **压缩** 为 query（优于简单 ID query）。
6. 用 DIN / LONGER / **Transformer** 等对 InsToken 序列建模；实践上把建模结果作为 token **输入上层 RankMixer 等复杂特征交互** 效果最好。

### 存储架构

- **InsID**：训练实例唯一键（如 hash 请求 ID + 创意 ID + 时间戳）。
- **Emb PS**：InsID → InsEmb。
- **KV**：UID → (InsID 列表, 侧信息)，按时间排序、截断长度 \(T\)（常用 256）。
- 存储量：\(N_{\text{daily}} \times T_{\text{retention}} \times d \times 4\) bytes（FP32）。

---

## 3. 实验设置（摘要）

- **数据**：真实广告 CVR 预测，4 个月、数百亿级样本；传统行为序列仅保留稀疏核心侧信息（ID、类目、行为类型等）。
- **Base 序列模块**：DIN / LONGER / Transformer；特征交互统一 **RankMixer**。
- **超参**：InsEmb \(d=64\)，SIT 2 层，下游 IAT 长度 \(T=256\)，传统行为序列长度 512。

---

## 4. 主要结果

### 4.1 离线（下游，+User IAT + Transformer 建模 IAT 序列）

| Base 架构 | ΔAUC | ΔLogLoss |
|-----------|------|----------|
| DIN       | +0.24% | -0.51% |
| LONGER    | +0.31% | -0.67% |
| Transformer | +0.29% | -0.63% |

- **Temporal IAT** 也有收益（最高约 +0.15% AUC），但弱于 User-Order。
- 工业场景 **0.1% AUC** 已属显著；手工 256 长度行为序列往往难达 +0.1% AUC。
- IAT 序列越长，增益越大（最长约 **+0.4% AUC**）；约 80% 用户历史实例数 < 256。
- **User-Order 源模型 + 下游 Transformer/LONGER** 对齐最好；DIN 建模 IAT 收益弱。

### 4.2 Scaling

- Base 扩到 **1B 参数**（dense / 序列长度 / 特征交互 scaling），User IAT 下游仍有约 **+0.33%** streaming AUC。
- 增大 SIT 层数/InsEmb 维度：源模型 AUC 增益可从约 0.52% 提到 0.80%，下游同步受益。
- 下游 Transformer 层数与 IAT 最大长度呈 **常规 scaling law**。

### 4.3 消融要点

- User-Order batch 更大 → 每用户覆盖更多历史，略利好 U-IAT。
- 去掉 **label 侧信息** 伤害明显（InsEmb 几乎不含 label）。
- Query 用压缩的 seq+non-seq 优于纯 ID query。
- IAT 建模应放在 **复杂特征交互之前**。

### 4.4 在线 A/B（已全量部署多场景）

**同域广告（相对强基线）**

| 方案 | ADSS | ADVV |
|------|------|------|
| Temporal IAT | +0.685% | +0.653% |
| User IAT     | +1.557% | +1.340% |

**跨域**（用广告场景训练的 IAT 迁移）：商城广告 CVR、信息流 CTR/CVR、非闭环广告、直播电商 CT-CVR 等均有显著提升（如信息流 CTR ADSS **+3.015%**，直播 GMV **+0.151%**）。

---

## 5. 与相关工作的关系

- **序列建模范式**（TWIN、LONGER、生成式推荐等）：多优化 \(\mathcal{F}_{\text{seq}}\)，未解决手工序列特征信息上限。
- **序列特征工程**（侧信息、多模态、语义 ID）：仍依赖手工特征或开销大、与下游对齐差。
- **两阶段排序**（LargeFM、HLLM、LLaTTE）：IAT 同属「上游压缩表示 + 下游消费」，但压缩对象是 **完整训练实例** 而非仅 item 语义或 user embedding。

---

## 6. 结论与展望

**IAT** 用「实例即 token」的两阶段设计，把用户历史交互从稀疏手工序列特征升级为 **高信息密度的 InsEmb 序列**，在离线 AUC/LogLoss 与多业务线在线指标上均显著优于 SOTA 实践，并具备 **域内与跨域可迁移性**（ByteDance 电商广告、商城营销、直播等）。

未来方向：**单阶段** 降低训练复杂度；更强压缩以降低 PS/KV 成本。

---

## 7. 个人笔记（Generative / 工业推荐视角）

- 与 **生成式推荐**（语义 ID、生成候选）正交：IAT 解决的是 **输入侧历史序列的信息瓶颈**，可与现有排序/生成框架叠加。
- **User-Order + SIT** 本质是「在特征生产阶段做一次因果序列编码」，与下游 Transformer 形成 **表示—消费** 分工，类似「离线实例 codec + 在线序列 reader」。
- 跨域 A/B 说明 InsEmb 捕获了 **可迁移的用户行为模式**，不仅限于单场景 ID 共现。

---

## 8. 讨论记录（Q&A，含全部问答原文）

---

### Q1：详细介绍 Stage I

**问：** 详细介绍下 stage 1。

**答：**

**Stage I = IAT Compression（压缩阶段）**，训练 **源模型（source model）**，不是直接上线的排序模型。

| 输入 | 输出 |
|------|------|
| 工业训练实例（一条样本 = 一次曝光/点击/转化等，含数千维特征） | 每条实例一个 **InsEmb**（默认 64 维） |
| 仍用原有排序任务标签（CTR/CVR 等） | 写入 **Embedding PS**，键为 **InsID** |

压缩接在 base 模型得到 \(h\) 之后（\(h\) 约 6000 维，已含 seq+non-seq+RankMixer 等交互）。

#### 方案 A：Temporal-Order（时序序）

- 在 \(h\) 后加 **压缩 + 解压** MLP：
  - \(h_{\text{compress}} = \sigma(W_1 h + b_1)\) → **存为 InsEmb**
  - \(h_{\text{decompress}}\) 再接预测头，BCE 端到端训练
- **按全局时间序逐条** 压缩；batch/stream **逻辑一致**
- 缺点：源模型 AUC 约 **-0.05%**；InsEmb **无跨实例** 信息

#### 方案 B：User-Order（用户序，推荐）

1. 按用户聚合、时间排序组 batch（同用户多条历史在同一 batch）
2. 逐条压缩得 \(H_{\text{com\_batch}} \in \mathbb{R}^{T \times D}\)
3. **SIT**（因果 Transformer）增强表示
4. **落盘：SIT 之前** 的压缩向量（存 SIT 之后约 -0.09% AUC）
5. 源模型 AUC 约 **+0.6%**；一般不直接上线 serving

**流式（User-Order）**：batch 混多用户时，从 PS 取该用户过去 \(T-1\) 条 InsEmb，`stop_gradient` 后 concat 过 SIT，只更新当前条 InsEmb。

**两遍训练（附录）**：正序/逆序各训一遍，缓解「先训用户 InsEmb 质量差」的不公平，源模型 AUC 可从约 0.6% 提到 0.8%。

**可选侧信息**：label、归因时间戳等可异步写入 KV，与 InsEmb 一起在 Stage II 拼 InsToken。

---

### Q2：方案 A 和方案 B 产生的内容维度一样吗？

**问：** 两种方案产生的内容维度是一样的吗？

**答：**

**向量维度相同；语义/信息内容不同。**

| 项目 | Temporal-Order | User-Order |
|------|----------------|------------|
| \(D_{\text{raw}}\) | ~6000 | ~6000 |
| InsEmb 维度 \(D\) | **64** | **64** |
| 写入 PS | \(h_{\text{compress}}\) | SIT **前** 的每行压缩向量 |
| 每条形状 | \(\mathbb{R}^{64}\) | \(\mathbb{R}^{64}\) |

原文：*「The compressed dimension is the same for the user-order source model」*。

**相同：** 都是 64 维；同一 InsID/PS/KV 接口；下游 adaptation 一致。

**不同：** User-Order 在因果 batch+SIT 下学压缩；Temporal 逐条独立。实验上 U-IAT 下游普遍强于 T-IAT。

**易混点：** 每条 InsEmb 是 1×64；用户序列长度 \(T\le256\) 是 Stage II 拉多少条，不是单条维度。

---

### Q3：方案 A 的 compress–decompress 是否多此一举？（相对融合后已有 MLP 压维算 logit）

**问：** 一般来说 seq 和 non-seq 融合后的 \(h\) 会经过多层 MLP 维度压缩再算 logits。方案 A 又引入压缩-解压，是不是多此一举？

**答：**

**若只关心单模型线上 AUC**：在已有 MLP 后再加 compress–decompress **多半多余**，论文 Temporal 源模型也约 **-0.05% AUC**。

**在 IAT 设定下仍需要独立瓶颈的原因：**

1. **两阶段 + 梯度截断**：Stage II 不能反传 InsEmb；不能指望预测 MLP 中间层自动成为可复用的 InsEmb。
2. **存储形态**：海量实例要固定 **64 维** 入库；\(h\)（~6000 维）或预测 MLP 中间层维度/语义随模型改版不稳定。
3. **目标不同**：预测 MLP 为 **当前 (u,v) logit**；InsEmb 为 **可检索、可跨域、给下游序列用的实例摘要**（instance codec）。
4. **方案 B 的存在**说明「只压 \(h\)」对下游序列对齐不够，还需 SIT。

**可替代思路**（论文未采用）：取现有 tower 某层 64 维 hidden 当 InsEmb——需固定维度、实例级、断梯度、A/B 验证。

**一句话**：不是「预测前多压一次」，而是为 **PS 存 InsEmb + 另一套下游** 服务的专用 codec；解压+BCE 保证 64 维有预测相关信息。

---

### Q4：64 维 InsEmb 是请求粒度还是用户粒度？

**问：** 64维的InsEmb是请求力度还是用户力度呢？

**答：**

**是「实例 / 请求样本」粒度，不是「一个用户一个 64 维向量」。**

| 粒度 | 是否 InsEmb |
|------|-------------|
| 每条历史训练实例 / 一次请求样本 | ✅ 一个 InsID → 一个 64 维 |
| 整个用户一个向量 | ❌ |

- 广告场景 InsID：hash(请求 ID + 创意 ID + 时间戳) 等。
- **一个用户** → **多个** InsEmb，排成序列（\(T\le256\)）。
- **一次线上请求**：从 KV 拉 **过去** InsID 列表；**当前候选** 走 Stage II 的 **Query**，不是把当前请求也压成 InsEmb 入库。

对比 **LLaTTE** 等：user-level 是一个 user 一个向量；IAT 是 user → **一串** InsEmb。

---

### Q5：请求粒度下如何「被另一个模型消费」？target 每次都不同

**问：** 如果是请求力度，那如何「可被另一个模型消费」呢？另一个模型消费时，target都不一样了。

**答：**

**InsEmb 不是「对当前 target 的预测向量」，而是「历史事件档案」；target 变靠 Query + 序列交互对齐。**

Stage II 每次请求：

```text
【用户侧】过去 InsEmb 序列（InsToken）     【候选侧】
KV+PS 拉 T 条历史                    Query = compress(当前 X_seq, X_nonseq)
              ↓                              ↓
         F_iat(Query, IAT序列)  →  H_iat  →  RankMixer → ŷ
```

- 历史 InsEmb 里嵌有 **当时** 的 item/context（像行为序列每一步）。
- **当前 item_j** 在 **Query** 里（论文消融：简单 ID query 会掉点）。
- 与 **DIN** 同范式：历史 token + 当前 target 做 attention。

**「另一模型消费」** = 当作 **新的用户历史序列特征** 读 PS，不是代替整个打分网络。

**两阶段**：Stage I 在各自标签下产 InsEmb；Stage II 用 adaptation + 自己的塔在 **当前 (u,v)** 上训 BCE，**不更新 InsEmb**。

**跨域 A/B** 说明 InsEmb 含可迁移行为模式，不是「一个向量通吃所有 target」。

---

### Q6：是否相当于训练样本变 token，形成历史序列，Stage II 再筛选一部分？

**问：** 其实相当于把训练样本变成一个token(InsEmb), 形成一个类似历史行为序列的样本序列？在Stage2时再从中筛选一部分出来计算。

**答：**

**理解正确，补两点：**

1. **筛选** = 按 **请求时间戳** 截断（防泄漏）+ 长度上限 **T≤256**，不是模型随意挑样本。
2. **Stage II 还有当前 Query**，不是只有历史序列。

```text
历史：[InsEmb_1, …, InsEmb_T]
当前：Query(当前 u, 候选 v, context)
```

论文 base 往往 **保留** 传统行为序列（512），**叠加** IAT，不是完全替换。

---

### Q7：Stage I 方案 B 相对 A 是否就是同用户 batch + Transformer？

**问：** Stage1中方案B相对于方案A的差异在于：每次只对同一个用户的样本操作，然后加上transformers?

**答：**

**大体对，但要补全：**

| | Temporal-Order | User-Order |
|---|----------------|------------|
| 组 batch | 全局时间，逐条独立 | **同用户** 多条历史成批 |
| 压缩 | ✅ | ✅ 先逐条 → \(H_{\text{com}}\) |
| Transformer | ❌ | ✅ **SIT**（在压缩**之后**，因果 mask） |
| 存盘 | 压缩 InsEmb | **SIT 前** 的压缩 InsEmb |
| 流式 | 与 batch 一致 | 多用户 batch + **PS 拉历史 InsEmb** |

**不是**「只对同用户」这么简单；流式阶段 batch 常混用户，靠 PS 检索拼序列。

**Transformer 不替代压缩**，而是加在压缩后；**存盘不用 SIT 输出**。

---

### Q8：Stage II 具体怎么做？

**问：** Stage2具体是怎么做的呢？

**答：（完整流程）**

**Step 0 检索**

1. KV：`UID` → `(InsID, 侧信息)`，按请求时间截断，≤256
2. PS：`InsID` → InsEmb[64]

**Step 1** Adaptation MLP：\(E_{\text{InsEmb\_adapt}} = \sigma(E_{\text{InsEmb}} W^T + b)\)

**Step 2** InsToken：\(\text{Concat}(E_{\text{InsEmb\_adapt}}, E_{\text{side}})\)，默认 64+64=128 维/步

**Step 3** Query：\(Q = \mathcal{F}_{\text{compress}}(\text{Concat}(X_{\text{seq}}, X_{\text{non-seq}}))\)，\(d_{\text{query}}=512\)

**Step 4** \(Q\) 与 IAT 序列投影到同维 → **DIN/LONGER/Transformer**（主实验 IAT 侧多用 Transformer）

**Step 5** 输出作为 **token** 进 **RankMixer**（在特征交互**之前**最好），与传统 \(F_{\text{seq}}(x_{\text{seq}})\)、\(x_{\text{non-seq}}\) 一起融合 → BCE

**训练**：InsEmb 不反传；源模型流式更新 InsEmb 同步 PS。

**默认超参**：IAT 长度 256；传统行为序列 512；RankMixer 统一上层。

---

### Q9：InsEmb、InsToken、IAT 序列的异同

**问：** InsEmb, InsToken, IAT序列有什么异同？

**答：**

```text
InsEmb（Stage I，64维，存 PS）
  → Adaptation + 侧信息
  → InsToken（单步完整 token）
  → 按时间排 T 步
  → IAT 序列（Stage II 序列模型输入）
```

| | InsEmb | InsToken | IAT 序列 |
|---|--------|----------|----------|
| 粒度 | 1 InsID 一条 | 1 时间步 | T 步 |
| 产生 | Stage I | Stage II 拼接 | Stage II 动态构建 |
| 存储 | PS | 一般不单独存 | 不存 |
| 含 label 等 | 基本不含 | 可含侧信息 | 每步可含 |
| 含当前候选 | 否 | 否 | 否 |

**记忆**：InsEmb=库里档案；InsToken=档案+备注的一步；IAT 序列=很多步；Query=当前候选。

---

### Q10：论文中 IAT 序列建模具体怎么做？

**问：** 论文中IAT 序列建模具体是怎么做的呢？

**答：**

论文写明：**投影到同维 → 任意主流序列网络（LONGER/Transformer 等）→ 结果作为 input token 进 RankMixer**；未展开每层结构。

**实验设定：**

- 传统分支：DIN/LONGER/Transformer 建模 **手工行为序列 512**
- IAT 分支：主表 IAT 侧 **Transformer**；`iat_other_arch` 中 IAT 可试 DIN/LONGER/Trans
- LONGER/Transformer 在 IAT 上约 +0.29%~+0.31% AUC；DIN 仅 +0.03%

**LONGER（论文描述）**：Perceiver + global token + token merge，降长序列复杂度。

**消融**：压缩 Query（非 ID query）；IAT 放在 RankMixer **前**（后放约 -0.04%~-0.06% AUC）。

**伪代码：**

```python
ins_token = concat(adapt_mlp(ps.mget(ins_ids)), embed(side))
Q = compress_mlp(concat(X_seq_cur, X_nonseq_cur))
H_iat = transformer_iat(proj_q(Q), proj_kv(ins_token))
h_all = rankmixer(x_nonseq, F_seq(x_seq_trad), H_iat_as_token, ...)
```

---

### Q11：心智模型——新序列 + Stage1 产序列 + Stage2 压成一维 + MLP 融合？

**问：** 可以认为加入一个新序列，stage1产生序列，stage2使用序列产生一个一维representation, 与原先的序列/非序列产生的representation一起处理（使用MLP）？

**答：**

**大体正确，三处修正：**

| 你的说法 | 更贴论文 |
|----------|----------|
| Stage1 产生序列 | Stage1 产 **每条 InsEmb**；**序列**在 Stage2 按 UID+时间 **拼出** |
| 一维 representation | ✅ \(H_{\text{iat}}\) 固定维向量/单个 token |
| 用 MLP 一起处理 | 上层是 **RankMixer** 等，非简单 MLP；\(H_{\text{iat}}\) 作为 **token** 在交互**之前**接入 |

**双分支：**

```text
x_seq(512) → H_trad
IAT序列    → H_iat
         ↘ RankMixer → ŷ
```

还需 **Query** 做 target-aware 建模，不是整段序列 mean-pool。

---

### Q12：\(H_{\text{trad}}\) 和 \(H_{\text{iat}}\) 都是序列吗？

**问：** H_trad和H_iat都是序列吗？

**答：**

**不都是。输入是序列；\(F_{\text{seq}}\)/\(F_{\text{iat}}\) 输出一般是聚合后的向量/少量 token，不是长度 T 的序列。**

```text
InsToken [T, d] + Query → F_iat → H_iat ≈ [d_model] 或 [1, d_model]
x_seq [512, ·]          → F_seq → H_trad ≈ 聚合表示
```

论文：IAT 建模结果作为 **an input token**（单数）进 RankMixer。

Eq.1：进 interaction 的是 \(\mathcal{F}_{\text{seq}}(x_{\text{seq}})\) 的输出，不是原始步序列。

---

### Q13：序列处理产生一个 token 还是多个？以 LONGER 为例

**问：** 现在对序列的处理方式，是产生一个token还是多个token呢？可以以LONGER为例

**答：**

**分三层：**

```text
① InsToken 序列 [T, d]           ← 很多 token（输入）
② LONGER 内部                    ← 通常多个 latent，再 merge
③ 进 RankMixer                   ← 论文写 "an input token"，多为 1 个或极少数 slot
```

| 模块 | 对上 RankMixer（典型） |
|------|------------------------|
| DIN | **1** 个兴趣向量 |
| Transformer | 多为 **1** 个（pool/query 位） |
| LONGER | 内部 **多个** Perceiver latent → 常再压成 **少量/1 个** 给上层 |

**LONGER 示意：**

```text
T 个 InsToken + Query
  → Perceiver latent (M≪T) + global token + token merge
  → 与 Query 融合
  → 少量 token / 1 个 H_iat → RankMixer
```

**IAT 实验：**

- 传统 512 行为序列、IAT 256 InsToken 可 **各用一路 LONGER**
- IAT 上 LONGER 明显好于 DIN（+0.31% vs +0.03% AUC）

**统一说法：** 序列进 → 聚合表示出 → RankMixer；LONGER 不是把 256 步原样上交。

论文未给出 LONGER 输出 token 个数的公式，需查 LONGER 原文 latent 数配置。

---

## 9. 讨论中的核心心智模型（汇总）

```text
Stage I:  每条历史训练实例 → InsEmb[64] → 写 PS（User-Order 更优）

Stage II:  UID+时间 → 拉 ≤256 条 → InsToken 序列
          当前请求 → Query
          F_iat(Query, IAT序列) → H_iat（一个/少量 token）
          F_seq(x_seq) → H_trad
          RankMixer(x_nonseq, H_trad, H_iat, …) → ŷ
```

**三个易混点：**

1. InsEmb = **实例级**，不是 user-level 单向量。  
2. target 变化靠 **Query**，不靠 InsEmb  alone。  
3. \(H_{\text{trad}}\)、\(H_{\text{iat}}\) 是 **序列模块的输出向量**，不是长度为 T 的序列本身。
