# Douyin Multimodal Embedding (DME) Technical Report

> 来源：字节跳动抖音搜索多模态团队 × 人大 GSAI，arXiv:2608.02148。骨干 Qwen3.5，2B / 9B 两个规格，基于 LoRA 微调。

## 一句话总览

**综合理解**：本文对 Query/Doc 数据，第一阶段采用常规的对比学习；第二阶段使用较强模型生成更复杂 Query/Doc 数据形式，然后通过设计精细目标学习该强化数据。最终使用 Query/Doc 各自前向获取 Embedding，使用双塔内积计算并上线。

**理解点评**：

| # | 原描述 | 评价 | 说明 |
|---|---|---|---|
| 1 | 第一阶段采用常规的对比学习 | ✓ 正确 | Stage 1 = query/doc 各自 forward + `<emb>` 单 token readout + InfoNCE；大规模（25M 对）弱监督对比；措辞准确 |
| 2 | 第二阶段使用较强模型生成更复杂 Query/Doc 数据形式 | ⚠️ 措辞需精化 | "较强模型"对（Seed-2.0-Pro）；但"**更复杂 Query/Doc 数据形式**"易误导——query/doc 本身不变，是为其三元组生成**更细粒度监督 label/target**（证据位置、local_summary、traj）；教师没生成"更复杂的 query/doc"，而是更复杂的监督标注（见 Q13、Q14） |
| 3 | 然后通过设计精细目标学习该强化数据 | ✓ 正确 | "设计精细目标"对（多损失 + label）；"强化数据"措辞别扭但意思对；可表述为"设计精细损失目标学习该强化监督" |
| 4 | 最终使用 Query/Doc 各自前向获取 Embedding，使用双塔内积计算并上线 | ✓ 正确，需补推理细节 | 双塔内积上线对；补充：query 侧 forward 带 anchor + typed latent（<1ms，在线算一次），document 侧带 anchor（无 typed latent，离线预算建 ANN）；2-B 解码通路推理时丢弃 |

**汇总**：2 句正确，1 句需精化，1 句正确需补细节。核心闭环理解（两阶段 → 各自前向 → 双塔内积上线）准确。最需精化点同 Stage 2 开头：**query/doc 数据形式不变，变的是监督 label 的复杂度**。

## 动机

多模态检索已成为 AI 时代基础设施：除搜索/推荐外，还要为 RAG 检索外部知识、作 agent 感知世界的工具。新一代调用方（AI search、agent）发的是多约束指令，期望能支撑后续推理的证据，而不只是"看似相关"的候选。抖音/小红书/YouTube 这类大规模图文视频平台，内容海量异构（文本、图像、视频帧、OCR、字幕、元数据），query 也可能是任意模态混合。实用检索模型须同时满足：

1. **十亿级索引下的效率**——bi-encoder + 离线预算 + ANN，是工业向量检索的标配；
2. **细粒度语义判别**——硬负常与正样本共享全局场景/主题，只差一小块视觉区域、关键帧、局部文本 span 或微妙语义条件。

现有范式各有短板：

| 范式 | 效率 | 细粒度判别 | 问题 |
|---|---|---|---|
| 对比式 MLLM embedder（VLM2Vec、GME、Qwen3-VL-Embedding） | 高（单次编码） | 粗（pair 级监督，只说"该近"不说"为何"） | 继承了 MLLM 的感知/推理潜力却只当编码器用，监督太粗 |
| CoT 式 embedder（Think-Then-Embed、TRACE、UME-R1、Embed-RL） | 低（显式生成推理链/重排） | 高 | 显式文本推理或重排式计算牺牲效率，难作主检索编码器上线 |

DME 要填的工业空白：**在不把在线检索变成显式生成/重排的前提下获得证据感知的推理能力**，且表征本身须保留足够对端细节区分硬负——两者都不能引入额外生成或多遍 forward，保持标准稠密向量接口。作者用 **semantic sufficiency（语义充分性）** 统一描述这个更强要求：embedding 不仅与相关实例 pair 对齐，还要"扎根于检索相关证据"且"能保留对端细粒度语义"。

## 方法

### 任务设定与双塔架构

- instruction-aware 通用多模态检索：document/query 都可纯文本/纯视觉/视频/混合模态；任务由自然语言 instruction $\iota$ 定义相关性准则；
- bi-encoder：query/document 各自独立编码 $\mathbf{z}_q=E_\theta^q(T_q(\iota,q))$、$\mathbf{z}_d=E_\theta^d(T_d(\iota,d))$，$\ell_2$ 归一化后内积打分 $s=\mathbf{z}_q^\top\mathbf{z}_d$；
- 输入序列 $T(\iota,x)=[\mathbf{u}_{\mathrm{inst}};\mathbf{u}_x;\mathbf{u}_{\mathrm{ret}}]$，retrieval token 放末尾，causal attention 下能 attend 全前文；
- 推理时 document 表征离线预算建 ANN 索引，query 在线编码一次即查 top-K，避免 query-candidate 逐对 cross-attention。

### 训练数据与监督源（按阶段递进质量/特异性）

- **Stage-1 对比数据**：~25M 对，大规模弱监督（文本/图像/视频）+ 自研+合成数据补充质量、场景覆盖、时序理解；图像 caption 强细粒度对齐，长视频 caption + 帧/clip 级监督强时序。
- **Stage-2 对比数据**：MMEB-v2 训练集 + 公开多模态检索/QA 数据，统一成 (instruction, query, positive) 格式，任务族间采样比例均衡防大集主导。
- **Stage-2A 结构化 CoT 监督**（教师 Seed-2.0-Pro 生成）：
  - 每个 query-positive-negative 三元组先归一化为 structured triplet，教师再生成 item 级 **structured anchor record**（不是模型 token，是离线标注）：含 query/pos/neg 侧证据条目（文本 span / 图像区域 / 视频帧引用）、item 级 local summary、typed trajectory；
  - typed trajectory 每步含检索角色 + 引用证据 id + 状态描述，角色 = `localize` / `align_pos` / `reject_neg` / `summarize`；query/pos 证据作 anchor-grounding 监督，neg 证据只构造 `reject_neg` 拒绝轨迹；
  - 预处理时对齐到序列化 Qwen3.5 输入得 token/patch/frame 级监督目标，local summary 与状态描述离线编码成缓存语义目标。
- **负样本构造**：Stage 1 主要 in-batch negatives + 大 batch；Stage 2 加 hard negatives（标注 distractor + retrieval-mined + 同任务族语义相近者），并用相似度过高过滤 pseudo negative 防 false-negative；任务均衡 in-batch 混合采样（同源按比例连续抽 + 跨任务批内混合）。

### Stage 1：大规模对比预训练

**综合理解**：文本化的 query, doc 各自在输入拼接上一个 `<emb>` token，然后进入模型，最后使用拼接的 `<emb>` 的 representation 作为 query, doc 的 representation。使用传统的对比学习训练。

**理解点评**：

| # | 原描述 | 评价 | 说明 |
|---|---|---|---|
| 1 | 文本化的 query, doc | ⚠️ 措辞易误导 | DME 是多模态，query/doc 可为文本/图像/视频/视觉文档/混合模态。"文本化"若指"把所有内容 token 化/序列化"则可接受（视觉→视觉 token），但图像/视频**不是变成文本**，而是变视觉 token。更准确：instruction + 序列化多模态内容（文本 token + 视觉 token） |
| 2 | 各自在输入拼接上一个 `<emb>` token | ✓ 正确，需补 | `<emb>` 拼在序列**末尾** $\mathbf{u}=[\mathbf{u}_{\mathrm{inst}};\mathbf{u}_x;\texttt{<emb>}]$；query 侧用模板 $T_q$、document 侧用 $T_d$，两侧模板可不同（side-specific），但共享同一 `<emb>` token 与同一骨干 |
| 3 | 然后进入模型 | ✓ 正确，需补 | 进入**同一个共享 MLLM 骨干**做两次**独立** forward（bi-encoder），query 与 doc 的 token **不交互**；不是一次 joint encoding（joint encoding 是 two_tower 里 CE 教师的做法） |
| 4 | 使用拼接的 `<emb>` 的 representation 作为 query, doc 的 representation | ⚠️ 措辞含糊 | "representation" 须明确：取 `<emb>` 位置经整个骨干后的**最终层 output hidden state** $\mathbf{h}_{\texttt{<emb>}}$ 作 readout $\mathbf{r}^{(1)}$，再过 embedding projection $W_{\mathrm{emb}}$ + $\ell_2$ 归一化得 $\mathbf{z}^{(1)}$。是它过骨干后的 **output 隐状态**，不是 `<emb>` 的 input embedding |
| 5 | 使用传统的对比学习训练 | ✓ 正确 | 标准 InfoNCE：拉正样本近、推负样本远；Stage 1 主要 in-batch negatives + 大 batch（128→8192）+ false-negative 过滤，pair 级监督（只说"该近"不说"为何"） |

**汇总**：3 处正确，2 处需精化。核心机制（双塔各自拼 `<emb>`、取 `<emb>` 隐状态作表征、对比学习）理解准确。最需精化的是"**文本化**"（多模态≠纯文本，视觉是视觉 token）与"**representation**"（是过骨干后的 output hidden state，非 input embedding）。

**详细展开**：

- 用 `<emb>` 单 token readout：$\mathbf{r}^{(1)}=\mathbf{h}_{\texttt{<emb>}}$，$\mathbf{z}^{(1)}=\mathrm{norm}(W_{\mathrm{emb}}\mathbf{r}^{(1)})$；
- 标准 InfoNCE：$\mathcal{L}_{\mathrm{S1}}^{\mathrm{CL}}=-\frac{1}{B}\sum_i\log\frac{\exp(s(q_i,d_i^+)/\tau)}{\exp(s(q_i,d_i^+)/\tau)+\sum_{d^-}\exp(s(q_i,d^-)/\tau)}$；
- 作用：对齐异构模态到统一空间、广任务覆盖、给 ANN 稳定几何；但监督仍 pair 级，不显式建模证据/对端语义——故有 Stage 2。

### Stage 2：语义充分性学习（按步骤流程总览）

**综合理解**：Stage2 是先使用较强模型产生更具体的数据样本，设计不同的目标以及对应的目标（如 pos_anchor 产生 doc 侧 anchor token 的注意力应该集中在哪些 token），进行多目标训练。

**理解点评**：

| # | 原描述 | 评价 | 说明 |
|---|---|---|---|
| 1 | 先使用较强模型产生更具体的数据样本 | ⚠️ 措辞需精化 | "较强模型"对（Seed-2.0-Pro）；"更具体"准（到 token/patch 级证据 + 角色轨迹，比 Stage 1 pair 级监督更具体）；但"**数据样本**"易误导——这些是**监督 label/target**，不是输入样本（见 Q13），不进输入序列 |
| 2 | 设计不同的目标以及对应的目标 | ⚠️ 措辞重复 | 意思应是"设计不同损失目标及对应 label"；Stage 2-A 有 $\mathcal{L}_{\mathrm{hit}}$/$\mathcal{L}_{\mathrm{sum}}$/$\mathcal{L}_{\mathrm{sem}}$/$\mathcal{L}_{\mathrm{align}}$/$\mathcal{L}_{\mathrm{reject}}$ 多损失，各有 label |
| 3 | 如 pos_anchor 产生 doc 侧 anchor token 的注意力应该集中在哪些 token | ✓ 例子正确 | pos_anchor → patch/frame 分布 $y_e(j)$，监督 document 侧 anchor 注意力 $p(j)$ 集中位置；抓对了 |
| 4 | 进行多目标训练 | ✓ 正确，需补范围 | $\mathcal{L}_{\mathrm{S2A}}$ 多目标加权对；完整 Stage 2 还含 $\mathcal{L}_{\mathrm{S2}}^{\mathrm{CL}}$（对比）+ $\mathcal{L}_{\mathrm{S2B}}$（重建），三项联合 $\mathcal{L}_{\mathrm{S2}}=\mathcal{L}_{\mathrm{S2}}^{\mathrm{CL}}+\mathcal{L}_{\mathrm{S2A}}+\mathcal{L}_{\mathrm{S2B}}$ |

**汇总**：1 处例子正确，2 处措辞需精化，1 处需补范围。核心理解（强模型产生监督 → 多损失多 label → 多目标训练）准确。最需精化的是"**数据样本**"（实为监督 label/target，非输入样本，详见 Q13）与"**目标以及对应的目标**"（措辞重复，应为"损失目标及对应 label"）。

Stage 2 在 Stage 1 的统一嵌入空间基础上，通过两个互补机制补"看对证据 + 保留对端语义"，联合优化三项 $\mathcal{L}_{\mathrm{S2}}=\mathcal{L}_{\mathrm{S2}}^{\mathrm{CL}}+\mathcal{L}_{\mathrm{S2A}}+\mathcal{L}_{\mathrm{S2B}}$。下面按执行顺序给出端到端流程（后续 2-A/2-B 小节为细节展开）。

**步骤 0：数据与监督准备（离线，训练前）**
- Stage 2 对比数据：MMEB-v2 训练集 + 公开多模态检索/QA，统一成 (instruction, query, positive) 格式，任务族间采样比例均衡；加 hard negatives（标注 distractor + retrieval-mined + 同任务族语义相近），并过滤 pseudo negative。
- 教师 **Seed-2.0-Pro** 对每个 query-positive-negative 三元组生成**结构化 CoT 监督**（structured anchor record，是离线标注、不是模型 token）：query/pos/neg 侧证据条目（文本 span / 图像区域 / 视频帧引用）、item 级 local summary、typed trajectory（每步：角色 `localize`/`align_pos`/`reject_neg`/`summarize` + 引用证据 id + 状态描述）。
- 预处理对齐到序列化 Qwen3.5 输入，得 token/patch/frame 级监督目标；local summary 与状态描述离线编码成缓存语义目标（嵌入）。

**步骤 1：输入序列构造（Stage 2 的 retrieval token 布局）**
- 输入仍是 $\mathbf{u}=[\mathbf{u}_{\mathrm{inst}};\mathbf{u}_x;\mathbf{u}_{\mathrm{ret}}]$，但 $\mathbf{u}_{\mathrm{ret}}$ 从 Stage 1 的单 `<emb>` 扩展为：**两侧**都有 modality-aware **anchor token**（用于证据锚定）；**仅 query 侧**有少量 **typed latent token**（带检索角色）；document 侧用标准 readout token。
- query/document 仍各自独立 forward（bi-encoder，token 不交互）。

**步骤 2：Stage 2-A 前向 — 证据锚定（Evidence grounding）**
- anchor 隐状态 $\mathbf{a}_{s,r}$ 经 probe 投影与**同模态**内容 token $\mathbf{x}_{s,j}$ 做缩放点积 softmax，得分布 $p_{s,r}(j)$（anchor 在该模态看哪）；加权求和得侧级证据池 $\mathbf{e}_{s,\mathrm{pool}}=\mathrm{Mean}_r(\sum_j p_{s,r}(j)\mathbf{x}_{s,j})$；
- 用教师证据目标 $y_e(j)$ 监督 anchor 分布：证据 hit loss $\mathcal{L}_{\mathrm{hit}}$（soft assignment 软分配 + 平衡正则 $\mathcal{L}_{\mathrm{bal}}$ 防全挤一个 anchor）；用缓存 local summary 嵌入 $\mathbf{s}$ 监督证据池语义 $\mathcal{L}_{\mathrm{sum}}=1-\cos(W_{\mathrm{sum}}\mathbf{e}_{s,\mathrm{pool}},\mathbf{s})$。

**步骤 3：Stage 2-A 前向 — 类型化隐式推理（仅 query 侧）**
- typed latent 隐状态 $\mathbf{h}_{q,k}^{\mathrm{traj}}$ 投影到检索空间 $\mathbf{q}_{q,k}^{\mathrm{traj}}$，按角色分别监督：`localize` 等语义态 $\mathcal{L}_{\mathrm{sem}}$ 对齐缓存教师状态嵌入 $\mathbf{g}_k$；`align_pos` $\mathcal{L}_{\mathrm{align}}$ InfoNCE 从 positive bank 检索出自己正 document；`reject_neg`（需 hard neg）$\mathcal{L}_{\mathrm{reject}}$ margin-ranking（与正样本余弦 > 与负样本余弦 + margin $\mu$）；$\mathcal{L}_{\mathrm{typed}}=\lambda_{\mathrm{sem}}\mathcal{L}_{\mathrm{sem}}+\lambda_{\mathrm{align}}\mathcal{L}_{\mathrm{align}}+\lambda_{\mathrm{reject}}\mathcal{L}_{\mathrm{reject}}$。

**步骤 4：Stage 2-A readout — 证据增强 readout**
- query：$\mathbf{r}_q=\mathbf{h}_{q,R}^{\mathrm{traj}}+\alpha\,\mathrm{sg}(W_e\mathbf{e}_{q,\mathrm{pool}})$（typed latent 终止隐状态 + stop-grad 证据池）；document：$\mathbf{r}_d=\mathbf{h}_{d,\mathrm{read}}+\alpha\,\mathrm{sg}(W_e\mathbf{e}_{d,\mathrm{pool}})$（标准 readout + stop-grad 证据池，**无 typed latent**）；embedding $\mathbf{z}_s=\mathrm{norm}(W_{\mathrm{emb}}\mathbf{r}_s)$；$\mathcal{L}_{\mathrm{S2A}}=\lambda_{\mathrm{hit}}\mathcal{L}_{\mathrm{hit}}+\lambda_{\mathrm{sum}}\mathcal{L}_{\mathrm{sum}}+\mathcal{L}_{\mathrm{typed}}$。

**步骤 5：Stage 2-B 前向 — 跨条件重建（NTP + MTP，训练 only）**
- 复用步骤 4 的 readout，取**归一化前**向量 $\tilde{\mathbf{z}}_q=W_{\mathrm{emb}}\mathbf{r}_q$ 作解码序列**首位唯一前缀 token**，喂回同一骨干；
- **NTP**：$\tilde{\mathbf{z}}_q$ 作前缀对 document 文本自回归解码，$\mathcal{L}_{\mathrm{NTP}}^{q\to d}$ 只在 document 文本 token 算交叉熵；对称 D→Q 方向 $\mathcal{L}_{\mathrm{NTP}}^{d\to q}$；
- **MTP**：挂 $D$ 个轻量 MTP 模块（sequential，共享 embedding/head），每深度预测未来第 $t+k$ token，$\mathcal{L}_{\mathrm{MTP}}^{q\to d}$，对称 D→Q；
- 关键：重建只把梯度回传到共享骨干与 readout，逼信息必经 embedding 这个瓶颈；$\mathcal{L}_{\mathrm{S2B}}=\lambda_{\mathrm{NTP}}(\mathcal{L}_{\mathrm{NTP}}^{q\to d}+\mathcal{L}_{\mathrm{NTP}}^{d\to q})+\lambda_{\mathrm{MTP}}(\mathcal{L}_{\mathrm{MTP}}^{q\to d}+\mathcal{L}_{\mathrm{MTP}}^{d\to q})$。

**步骤 6：联合损失与优化**
- $\mathcal{L}_{\mathrm{S2}}=\mathcal{L}_{\mathrm{S2}}^{\mathrm{CL}}+\mathcal{L}_{\mathrm{S2A}}+\mathcal{L}_{\mathrm{S2B}}$；对比项 $\mathcal{L}_{\mathrm{S2}}^{\mathrm{CL}}$ 维持全局检索几何（同 Stage 1 形式，但用 Stage 2 数据 + Stage 2 readout），2-A 给证据+隐状态监督，2-B 逼 readout 保留对端文本语义；顺序优化：先 Stage 1，再 Stage 2。

**步骤 7：推理（部署）**
- **丢弃**：2-B 的 NTP/MTP 解码通路 + $D$ 个 MTP 模块（训练 only）；**保留**：2-A 的 anchor + typed latent token（在 query 编码 forward 内，<1ms 延迟）；document embedding 离线预算建 ANN，query 在线编码一次（带 latent token）查 top-K；仍是标准 bi-encoder。

## 实验结果

### MMEB-v2 主表（78 任务，Image/Video/VisDoc 三域）

骨干 Qwen3.5，DME-2B / DME-9B 同一两阶段框架。整体分：

| 模型 | 规模 | Image Avg | Video Avg | VisDoc Avg | **All** |
|---|---|---|---|---|---|
| VLM2Vec-V2 | 2B | 64.9 | 34.6 | 69.2 | 59.2 |
| Qwen3-VL-Embedding | 2B | 75.0 | 61.9 | 79.2 | 73.2 |
| **DME** | **2B** | **75.9** | **65.6** | **79.9** | **74.8** |
| Qwen3-VL-Embedding | 8B | 80.1 | 67.1 | 82.4 | 77.8 |
| TTE-v2* | 7B | 79.2 | 60.7 | 82.3 | 75.7 |
| **DME** | **9B** | **79.8** | **70.8** | **82.0** | **78.4** |

三点观察：
1. **scale-wise 强**：2B/9B 各自在同档中领先；
2. **增益非单模态集中**：Video 上 DME-2B 达 65.6（远超同档 Qwen3-VL-2B 的 61.9、VLM2Vec-V2 的 34.6），VisDoc 也强；
3. **用轻量 latent-token 推理反超 reasoning-enhanced 基线**（IFM-TTE 74.1、TTE-v2 75.7、Embed-RL 68.1）——DME 不在评估时生成显式 CoT 或做 cross-encoder 重排，只靠编码器内少量软 token。

### 抖音工业部署

- **离线**：以 DME 初始化内部模型并迁移若干 DME 技术继续训练，相对上一生产模型整体 +2.92%，四方向一致（Text2Video +3.10%、Text2Image +3.03%、Image2Image +2.70%、Image2Video +2.83%）；
- **在线**：部署于抖音搜索（generative search、作 ranking 的检索特征），A/B 验证核心业务指标 **+0.1% Lifetime（LT）gain**。

### 消融（累积 recipe，MMEB-v2，2B）

| Stage1 | Stage2-A | Stage2-B | Image | Video | VisDoc | All |
|---|---|---|---|---|---|---|
| | | | 74.6 | 55.3 | 77.1 | 70.9 |
| ✓ | | | 74.8 | 59.3 | 79.0 | 72.5 |
| ✓ | ✓ | | 75.2 | 63.7 | 79.2 | 73.8 |
| ✓ | ✓ | ✓ | **75.9** | **65.6** | **79.9** | **74.8** |

- 全 recipe 把 70.9 抬到 74.8（+3.9）；
- **Stage 1** 主要拉 video（55.3→59.3）与 VisDoc（77.1→79.0），image 基本不变（74.6→74.8）——异构大规模对比预训练建统一空间；
- **Stage 2-A** 贡献 +1.3 整体，**video 单项最大**（59.3→63.7），与其"扎根于文本/视觉/时序局部证据"设计吻合；
- **Stage 2-B** 收尾到 74.8，**三域均衡**（image 75.2→75.9、video 63.7→65.6、VisDoc 79.2→79.9）——跨条件重建保留对端语义普适受益。

### 训练参数消融（可复用经验）

- **in-batch 负样本空间 scaling**：batch 128→8192 持续提升，再大收益递减甚至略降（更大 batch 增 false-negative 概率且 hard neg 梯度被大量 easy neg 稀释），须配合 false-negative 过滤 + hard neg 构造；
- **任务均衡 in-batch 混合采样**：混合比 0.25 最优（overall 0.6919）；太小缺跨任务/跨模态对比信号，太大批内过异构削弱任务一致性；
- **视觉预算**：image token 256→1280 全指标提升（overall 0.7049→0.7090）；video 训练/推理帧 32/32 是好 trade-off（8→32 提升明显，64 之后增益小）；最终用 1280 image token + 32 video 帧。

### 语义充分性的可量化度量：表征完备性（Representation Completeness）

把 NTP 重铸为有界可解释指标 $\mathrm{acc}@K$：以归一化前 embedding 作**唯一**前缀，teacher-forcing 过同一骨干，问每个位置 ground-truth token 是否在 Top-$K$ 预测内（只评前 $L_{\max}=10$ 个位置，micro-averaged）。四个方向：

| 方向 | 含义 | Image@1 | Video@1 | VisDoc@1 | **All@1** | All@10 |
|---|---|---|---|---|---|---|
| q2q | 自重建完备性 | 0.8758 | 0.8781 | 0.8909 | 0.8792 | 0.9262 |
| d2d | 自重建完备性 | 0.7737 | 0.6431 | 0.9455 | 0.7432 | 0.9355 |
| q2d | 对端完备性 | 0.6279 | 0.5920 | 0.9513 | 0.6585 | 0.8859 |
| d2q | 对端完备性 | 0.8736 | 0.8728 | 0.8978 | 0.8771 | 0.9318 |

- 自方向 Top-1 达 87.9%（q2q）/74.3%（d2d），Top-10 >92%——单向量保留了自身大部分 token 级内容，未塌成粗检索摘要；
- 对端方向更难（须重建只经匹配对共享的信息），但 d2q 仍 87.7% Top-1、q2d 65.9%，证实对端语义确被编码；
- 跨域：VisDoc 最易重建（文本重、规律，q2d 95.1% Top-1），Video 最难（文本 query 与长时序视觉目标信息差大，q2d 59.2%）。
- 这个 $\mathrm{acc}@K$ 可作工业里指导表征优化的可解释信号。

### 效率与 query 延迟

测 query 编码 forward 的 p50（排除候选编码/打分/ANN/数据加载），8 GPU、batch 4、20 warmup + 200 measured：

| query 类型 | 数据集 | w/o latent p50 | w/ latent p50 | Δ(ms/query) |
|---|---|---|---|---|
| Text | MSCOCO-T2I | 41.962 | 45.450 | +0.87 |
| Image+Text | OK-VQA | 111.084 | 111.399 | +0.08 |
| Video+Text | Video-MME | 1049.479 | 1052.694 | +0.80 |

latent 推理 token 每query 额外 <1ms（图像+文本近乎可忽略），因它只加几个软 token 到同一 forward，不做自回归/多轮；多模态输入（尤其 video）计算被原始视觉 token 主导，latent 相对成本极小。

## 讨论 Q&A

### Q10：DME 的输入是什么形式？是同时输入 user/item 还是各自输入算双塔内积？

**各自输入（双塔），不是同时输入。且要先纠正表述：DME 是检索场景的 query/document，不是推荐场景的 user/item——虽然概念可对应。**

**纠正表述**：DME 不是推荐场景的 user-item 匹配，是通用多模态检索的 query-document。document 指任意可检索项（文本/图像/视频/视觉文档/混合），query 也可任意模态。概念上对应：query（定义检索意图）↔ user，document（候选内容）↔ item，但术语和任务设定不同——relevance 由 instruction $\iota$ 定义，而非固定的 user-item 相关性。

**bi-encoder 设定（各自独立 forward、算内积）**：
- 给定 instruction $\iota$、query $q$、document $d$，模型**独立**编码两次 forward：
  - $\mathbf{z}_q = E_\theta^q(T_q(\iota,q))$
  - $\mathbf{z}_d = E_\theta^d(T_d(\iota,d))$
- $T_q, T_d$ 是 query 侧 / document 侧输入模板；$E_\theta^q, E_\theta^d$ 是对应提取过程。query 与 document 的 token 在编码时**不交互**（无 pairwise cross-attention）。
- 两侧 $\ell_2$ 归一化后内积打分 $s=\mathbf{z}_q^\top\mathbf{z}_d$（= 余弦相似度）。

**每次 forward 的输入序列形式**：$\mathbf{u}=T(\iota,x)=[\mathbf{u}_{\mathrm{inst}};\mathbf{u}_x;\mathbf{u}_{\mathrm{ret}}]$
- $\mathbf{u}_{\mathrm{inst}}$：instruction token（定义相关性准则，如"检索匹配文本描述的图像"）；
- $\mathbf{u}_x$：序列化多模态内容（文本 token + 视觉 token，视觉来自图像/视频帧/视觉文档）；
- $\mathbf{u}_{\mathrm{ret}}$：追加在内容后的 retrieval-specific token——**Stage 1 是单个 `<emb>`；Stage 2 query 侧是 anchor + typed latent，document 侧是 anchor + 标准 readout**。retrieval token 放末尾，causal attention 下能 attend 前面所有内容。

**两侧共享骨干但不对称**（关键点）：$E_\theta^q$ 和 $E_\theta^d$ 共用同一 MLLM 骨干（Qwen3.5）和 embedding projection $W_{\mathrm{emb}}$，但允许 side-specific 的 retrieval-token 布局和 readout 函数。即同一个模型两次 forward，但输入模板和读出方式不同——query 侧有 typed latent（localize/align_pos/reject_neg），document 侧只有 anchor + 标准 readout（详见 Q6 的不对称取舍）。

**推理时检索流程**：document embedding 全部离线预算、存 ANN 索引；在线 query 编码一次（Stage 2 带 latent token，<1ms），按向量相似度查 top-K。避免 query 与每个候选 document 逐对 cross-attention，是大规部署的关键。

**与 [two_tower 论文](../Retrieval/summary_llm_native_two_tower.md) 的关系**：同属 bi-encoder 双塔各自输入算点积。DME 的增量在：① instruction-aware（每个输入拼 instruction token 定义相关性，而非固定 user-item 相关）；② 两侧 readout 不对称且随训练阶段变（Stage 1 单 `<emb>` token，Stage 2 query 侧 typed latent+anchor、document 侧 anchor+标准 readout）；③ document 是任意多模态可检索项而非推荐 item。

> 易混点：DME 两次 forward 不是 Teacher/Student 关系（two_tower 的 CE 教师 vs TT 学生才是两次不同模型的 forward），而是**同一个 bi-encoder 模型对 query 与 document 分别编码**——这是双塔检索的标准操作，不是蒸馏。

### Q11：Stage 1 的"单 token readout"是什么意思？训练是不是合成数据 query/doc 各自过模型、算内积、对比学习？

**"单 token readout"指在输入序列末尾追加一个专门的 retrieval token `<emb>`，取它经过整个 MLLM 骨干后的最终层隐状态作为整条序列的表征，再过投影+归一化得 embedding。**

**先讲"单 token readout"：**

输入序列 $\mathbf{u}=[\mathbf{u}_{\mathrm{inst}};\mathbf{u}_x;\mathbf{u}_{\mathrm{ret}}]$，Stage 1 的 $\mathbf{u}_{\mathrm{ret}}^{(1)}=[\texttt{<emb>}]$——**就一个** retrieval token（vs Stage 2 有多个 anchor + typed latent）。

- `<emb>` 是序列里放进去的一个 token（像普通词一样查 embedding 矩阵得 input embedding），放在内容之后；
- 它经过 N 层 transformer 后，在末尾位置的 **output hidden state** $\mathbf{h}_{\texttt{<emb>}}$ 能 attend 前面所有 instruction + 多模态内容 token（因 causal attention 末 token 可见全前文），故聚合整条序列信息；
- readout representation $\mathbf{r}^{(1)}=\mathbf{h}_{\texttt{<emb>}}$，再 $\mathbf{z}^{(1)}=\mathrm{norm}(W_{\mathrm{emb}}\mathbf{r}^{(1)})$ 得 embedding。

**本质和 [two_tower 的 EOS pooling](summary_llm_native_two_tower.md) 一样**：decoder-only LLM 里末尾 token 的 output hidden state 聚合全序列，适合做表征。区别只是 two_tower 用现成的 EOS token，DME 用一个专门的 retrieval token `<emb>`（语义上是"为检索设的哨兵"）。机制完全同：放末尾 → 过骨干 → 取末尾 output hidden state → 投影归一化。

**再讲 Stage 1 训练流程（基本对，但精化"合成数据"）：**

✅ 对的部分：
- query, document 各自独立 forward（bi-encoder，共享骨干，token 不交互）：$\mathbf{z}_{q_i}=E_\theta^q(T_q(\iota_i,q_i))$、$\mathbf{z}_{d_i^+}=E_\theta^d(T_d(\iota_i,d_i^+))$；
- 两侧 $\ell_2$ 归一化后内积打分 $s=\mathbf{z}_q^\top\mathbf{z}_d$（= 余弦相似度）；
- 用对比学习训练，标准 InfoNCE：$\mathcal{L}_{\mathrm{S1}}^{\mathrm{CL}}=-\frac{1}{B}\sum_i\log\frac{\exp(s(q_i,d_i^+)/\tau)}{\exp(s(q_i,d_i^+)/\tau)+\sum_{d^-\in\mathcal{D}_i^-}\exp(s(q_i,d^-)/\tau)}$，鼓励 query 离正样本比离负样本近。

⚠️ "合成数据"是误解（需精化）：
- Stage 1 **主要靠大规模弱监督对比数据**（文本/图像/视频的公开 + 自研数据）学基础跨模态语义对齐；合成数据只是**补充**（高质量 image caption 补细粒度图文对齐、长视频 caption + 帧/clip 级监督补时序），不是全部。论文原文明确："Stage 1 mainly relies on large-scale weakly supervised contrastive data"。
- **教师生成的合成监督在 Stage 2-A 才出现**（Seed-2.0-Pro 生成 structured anchor record + typed trajectory），Stage 1 没有教师 CoT 监督。

**Stage 1 不用的东西（全 Stage 2 才引入）：**
- anchor token、typed latent token（Stage 2-A）；
- 证据池、evidence-enhanced readout（Stage 2-A）；
- Cross-Conditional Reconstruction 的 NTP/MTP（Stage 2-B）；
- hard negatives（Stage 1 主要 in-batch negatives + 大 batch；hard neg 在 Stage 2 才加）。

**负样本构造（Stage 1）：** 主要 in-batch negatives——同 mini-batch 内其他 document 当负样本，靠大 batch（128→8192 提升明显，再大收益递减）扩有效负样本空间；并做 false-negative 过滤（相似度过高的 pseudo negative 移除，防误把真答案当负样本）。任务均衡 in-batch 混合采样（同源按比例连续抽 + 跨任务批内混合，混合比 0.25 最优）。

**Stage 1 的定位：** 简单、可扩展的标准 bi-encoder 对比预训练，目的是建统一多模态嵌入空间 + 广任务覆盖 + 给 ANN 稳定几何。监督仍 pair 级（只说"该近"不说"为何"），不显式建模证据/对端语义——这正是 Stage 2 要补的。readout 也从 Stage 1 的"单 `<emb>` token 末隐状态"演进到 Stage 2 的"typed latent 终止隐状态 + stop-grad 证据池"（见方法节 Stage 2-A 的 evidence-enhanced readout）。

### Q12：Stage 2 细节追问——数据形式举例、anchor/typed latent 是什么及如何训练成角色、2-A 是否多目标、2-B 物理含义、推理时 document 侧加哪些 token

**(1) 步骤 0 教师生成的数据形式与举例**

教师 Seed-2.0-Pro 对每个 query-positive-negative 三元组生成 **structured anchor record**（离线标注、不是模型 token），含 5 类字段：`query_anchor` / `pos_anchor` / `neg_anchor`（证据条目：文本 span / 图像区域 / 视频帧引用）、`local_summary`（item 级语义摘要）、`traj`（typed trajectory，每步含角色 + 引用证据 id + 状态描述）。预处理对齐到序列化输入得 token/patch/frame 级分布 $y_e(j)$，`local_summary` 与 state description 离线编码成缓存嵌入 $\mathbf{s}$、$\mathbf{g}$。

举例：query = 文本"一只仓鼠在吃食物"，document = 图像（仓鼠吃食物），hard neg = 图像（仓鼠但不吃）：

```
query_anchor: [
  {type: text_span, text:"仓鼠",   tokens:[2:4]},
  {type: text_span, text:"吃食物", tokens:[5:7]}
]
pos_anchor: [
  {type: image_region, patches:[...], label:"仓鼠"},
  {type: image_region, patches:[...], label:"食物+进食动作"}
]
neg_anchor: [
  {type: image_region, patches:[...], label:"仓鼠（无进食）"}   ← 只用于 reject_neg，不作 anchor 命中目标
]
local_summary: "一只仓鼠正在进食"   → 离线编码成嵌入 s
traj:
  {role: localize,   evidence_ids:[q0,q1,pos0], state:"定位 query '仓鼠'+'吃食物' 概念与 pos 图仓鼠区域"}  → g_1
  {role: align_pos,  evidence_ids:[pos0,pos1],  state:"对齐 pos 图仓鼠+食物区域与 query 概念"}            → g_2
  {role: reject_neg, evidence_ids:[neg0],       state:"neg 图有仓鼠但无进食，应拒绝"}                    → g_3
  {role: summarize,  state:"总结：仓鼠进食"}                                                              → g_4
```

**(2) anchor token / typed latent token 是什么、如何训练成角色、label 怎么来**

- **是可学习软 token（多个），不是"随机 token"**：像 BERT `[CLS]` 或 prefix-tuning 的可学习 prefix，初始化随机但**通过训练学习语义**。anchor 每模态 $R_m$ 个（多个），typed latent 少量、每个对应一个角色（多个，固定角色）。
- **"作用"不是写死的，靠"教师标注 + 对应损失"训练出来**：
  - anchor 的"看哪"：label = 教师标注的证据位置 $y_e(j)$——教师用强模型能力识别 query/pos 里哪些 span/区域/帧是相关证据，对齐成 token/patch/frame 级分布，$\mathcal{L}_{\mathrm{hit}}$ 用 CE 让 anchor 注意力 $p_{s,r}(j)$ 逼近教师证据位置；
  - typed latent 的角色：第 $k$ 个 latent 对应 traj 第 $k$ 步，角色 $\kappa_{i,k}$（来自教师 traj）决定用哪个损失——`localize`→$\mathcal{L}_{\mathrm{sem}}$（label=教师 state 描述嵌入 $\mathbf{g}_k$）、`align_pos`→$\mathcal{L}_{\mathrm{align}}$（label=正 doc 嵌入 $\mathbf{z}_{d^+}$，stop-grad）、`reject_neg`→$\mathcal{L}_{\mathrm{reject}}$（label=正 vs 负 doc margin ranking）。
- **关键洞察**：软 token 本身可学习，其"作用"由"教师生成结构化标注（证据位置 + traj 角色 + state 描述）+ 对应损失"训练而成——本质是**把更强教师模型（Seed-2.0-Pro）的知识蒸馏进软 token**，不是预先写死语义。

**(3) 步骤 2,3,4 是多目标吗？步骤 5 的物理含义**

- **步骤 2,3,4 是多目标训练**：✅ 是。$\mathcal{L}_{\mathrm{S2A}}=\mathcal{L}_{\mathrm{hit}}+\mathcal{L}_{\mathrm{sum}}+\mathcal{L}_{\mathrm{typed}}$（$=\mathcal{L}_{\mathrm{sem}}+\mathcal{L}_{\mathrm{align}}+\mathcal{L}_{\mathrm{reject}}$），多损失项加权求和联合优化，共享骨干，多目标同时反传。
- **步骤 5（2-B）物理含义**：把 embedding 变成**语义瓶颈**——query embedding 须含足够信息以解码 document 文本（反之亦然），对抗"对比训练让 MLLM 生成式理解退化成纯相似度几何"；使 embedding 从"判别向量"升级为"信息完备表征"（可从向量恢复内容）；MTP 预测多未来 token 逼 embedding 预规划长程语义，非只存短程表面线索；双向对称要求两侧都保留对端语义；最深一层——使"语义充分性"**可量化**（$\mathrm{acc}@K$ 度量从此而来）。

**(4) 推理时 document 侧加哪些 token？**

⚠️ 需纠正：**typed latent 只 query 侧，document 侧无 typed latent**。

- 推理 **query 编码**：anchor token + typed latent token（<1ms 延迟，**在线**算一次）；
- 推理 **document 编码**：anchor token（做证据锚定）+ 标准 readout + 证据池，**无 typed latent**；document **离线预算**一次性算所有 embedding 建 ANN；
- 丢弃的是 2-B 的解码通路 + MTP 模块；2-A 的 anchor（两侧）+ typed latent（query 侧）都保留。

**不对称原因**：typed latent 的角色（localize/align_pos/reject_neg）本质是"定义检索意图"，是 query 的职责；document 是"提供候选证据"，标准 readout + 证据池足够，且 document 要离线预算、可扩展。这与 [two_tower](summary_llm_native_two_tower.md) "latent 只放 user tower、item tower 标准"的取舍同构（详见 Q6）。

### Q13：Stage 2 是把 Teacher 数据输入小模型吗？每种 token 举例 + 步骤 2,3,4 训练样本

**(1) "把 Teacher 数据输入小模型"——部分对但有关键混淆**

✅ 对的部分：教师 Seed-2.0-Pro 确实生成监督信号训练 DME 小模型，从"知识大模型→小模型"看像蒸馏。

⚠️ 关键混淆：**教师数据不是作为输入 token 喂给小模型，而是作为监督 label/target**。要分清两条线：

- **输入线**：小模型输入 = instruction + 多模态内容（query/document 本身）+ **小模型自己的可学习软 token**（anchor、typed latent、readout）。**教师 structured anchor record 不进输入序列**。
- **监督线**：教师 structured anchor record（证据位置、`local_summary` 嵌入、traj 角色+state 嵌入）作 label，通过 $\mathcal{L}_{\mathrm{hit}}$/$\mathcal{L}_{\mathrm{sum}}$/$\mathcal{L}_{\mathrm{sem}}$/$\mathcal{L}_{\mathrm{align}}$/$\mathcal{L}_{\mathrm{reject}}$ 监督软 token 学到对应行为。`local_summary` 与 state description 离线编码成缓存嵌入 $\mathbf{s}$、$\mathbf{g}$ 作对齐目标——进的是**损失目标空间**，不是输入 token 序列。

"输入小模型"若指"作为训练数据进入训练流程"则对；若指"作为输入 token 序列喂 forward"则错。更准确：**Stage 2 = 用教师 Seed-2.0-Pro 生成的结构化标注作监督信号，训练 DME 的软 token + readout + 骨干**。教师数据是 label，不是 input。

**(2) 每种 token 举例 + 是否从教师来**

**否，anchor/typed latent/readout 都是小模型自己的可学习软 token，不从教师数据来。教师数据是 label。**

| token 类型 | 举例 | 来源 | 作用 |
|---|---|---|---|
| modality-aware anchor token | query=文本侧放 1 个"文本模态 anchor"软向量 $A_q$；document=图像侧放 1 个"图像模态 anchor" $A_d$（多模态则每模态 $R_m$ 个） | 小模型可学习 embedding，初始化随机 | 隐状态 $\mathbf{a}_{s,r}$ 与**同模态**内容 token 做注意力定位证据（$\mathcal{I}_{m(r)}$ 限定同模态） |
| typed latent token | query 侧 3 个软向量 $T_1/T_2/T_3$，分别对应 localize/align_pos/reject_neg（+可选 summarize） | 小模型可学习 embedding，初始化随机 | 每个带检索角色，按角色用不同损失训练；**仅 query 侧** |
| readout token | Stage 1 是 `<emb>`；Stage 2 query 侧 readout 取最后一个 typed latent 终止隐状态，document 侧用标准 readout token | 小模型可学习/结构指定 | 取其隐状态作 readout 表征 |

**关键**：教师 structured anchor record（query_anchor/pos_anchor/neg_anchor/local_summary/traj）是**离线标注**，对齐成 token/patch/frame 级分布 $y_e(j)$ 与缓存嵌入 $\mathbf{s}$、$\mathbf{g}$，作为**损失 label**，不进输入序列。

**(3) 步骤 2,3,4 训练样本例子（贯穿"仓鼠吃食物"，明确输入线 vs 监督线）**

设 query=文本"一只 仓鼠 在 吃 食物"（token 1-5），document=图像（仓鼠吃食物，patch 序列），hard neg=图像（仓鼠不吃）。

**小模型输入序列**：

query 侧
```
[instruction: "检索匹配文本的图像"] [query 文本 tokens 1-5] [A_q: 文本模态 anchor 软token] [T1: localize] [T2: align_pos] [T3: reject_neg]
```

document 侧
```
[instruction] [image patches] [A_d: 图像模态 anchor 软token] [readout token]
```

**教师 label（structured anchor record，不进输入）**：query_anchor→$y_e(j)$（"仓鼠"@token1-2、"吃食物"@token4-5 的 token 分布）；pos_anchor→patch 分布；local_summary 嵌入 $\mathbf{s}$；traj state 嵌入 $\mathbf{g}_1$(localize)/$\mathbf{g}_2$(align_pos)/$\mathbf{g}_3$(reject_neg)。

**步骤 2（证据锚定）**：
- 输入线：$A_q$ 隐状态 $\mathbf{a}_{q,A}$ 与 query 文本 token $\mathbf{x}_{q,j}$ 做注意力 → $p_{q,A}(j)$；加权求和得证据池 $\mathbf{e}_{q,\mathrm{pool}}$
- 监督线（教师 label）：$\mathcal{L}_{\mathrm{hit}}=\mathrm{CE}(p_{q,A}(j), y_e(j))$ 让 $A_q$ 注意力逼近"仓鼠/吃食物"位置；$\mathcal{L}_{\mathrm{sum}}=1-\cos(W_{\mathrm{sum}}\mathbf{e}_{q,\mathrm{pool}}, \mathbf{s})$ 让证据池语义对齐 local_summary 嵌入
- document 侧同理 $A_d$ 与 image patch 注意力，label=pos_anchor patch 分布

**步骤 3（类型化隐式推理，仅 query）**：
- 输入线：$T_1/T_2/T_3$ 隐状态 $\mathbf{h}_{q,1}^{\mathrm{traj}}/\mathbf{h}_{q,2}^{\mathrm{traj}}/\mathbf{h}_{q,3}^{\mathrm{traj}}$，投影 $\mathbf{q}_{q,k}^{\mathrm{traj}}$
- 监督线（教师 label + 样本对）：
  - $T_1$(localize)→$\mathcal{L}_{\mathrm{sem}}=1-\cos(P_{\mathrm{traj}}\mathbf{h}_{q,1}^{\mathrm{traj}}, \mathbf{g}_1)$（对齐教师 localize state 嵌入）
  - $T_2$(align_pos)→$\mathcal{L}_{\mathrm{align}}$ InfoNCE（让 $T_2$ 检索出正 doc 嵌入 $\mathbf{z}_{d^+}$，stop-grad）
  - $T_3$(reject_neg)→$\mathcal{L}_{\mathrm{reject}}$ margin（$\cos(T_3,\mathbf{z}_{d^+}) > \cos(T_3,\mathbf{z}_{d^-}) + \mu$）
- 角色由教师 traj 步骤 $\kappa$ 决定用哪个损失

**步骤 4（证据增强 readout）**：
- 输入线：query $\mathbf{r}_q = \mathbf{h}_{q,3}^{\mathrm{traj}}$（$T_3$ 终止隐状态）$+\alpha\cdot\mathrm{sg}(W_e\mathbf{e}_{q,\mathrm{pool}})$；document $\mathbf{r}_d = \mathbf{h}_{d,\mathrm{read}} + \alpha\cdot\mathrm{sg}(W_e\mathbf{e}_{d,\mathrm{pool}})$
- 输出：$\mathbf{z}_q=\mathrm{norm}(W_{\mathrm{emb}}\mathbf{r}_q)$、$\mathbf{z}_d=\mathrm{norm}(W_{\mathrm{emb}}\mathbf{r}_d)$
- 监督线：本步不引入新 label，把步骤 2 的证据池 + 步骤 3 的 typed latent 终止状态融合成 embedding，喂对比损失 $\mathcal{L}_{\mathrm{S2}}^{\mathrm{CL}}$（label=正负样本对，query 与正 doc 近、与负 doc 远）

### Q14：Teacher 的 label 在步骤 0 产生，具体是哪些字段？

✅ 对：教师 Seed-2.0-Pro 在步骤 0（离线，训练前）生成 structured anchor record，作为 Stage 2-A 的监督 label。按"原始字段 → 预处理后 label → 用于哪个损失 → 监督谁"列清单。

**教师生成的 5 个原始字段**（structured anchor record）：

| 原始字段 | 形式 | 预处理后 label | 用于哪个损失 | 监督谁 |
|---|---|---|---|---|
| `query_anchor` | query 侧证据条目（文本 span/图像区域/视频帧 + 位置 + 语义标签） | token/patch/frame 级分布 $y_e(j)$（query 侧） | $\mathcal{L}_{\mathrm{hit}}$ | query 侧 anchor token 注意力 $p_{q,A}(j)$ |
| `pos_anchor` | positive 侧证据条目 | patch/frame 级分布 $y_e(j)$（document 侧） | $\mathcal{L}_{\mathrm{hit}}$ | document 侧 anchor token 注意力 |
| `neg_anchor` | negative 侧证据条目（可选） | 用于构造 `reject_neg` 轨迹状态，**不作 anchor 命中目标** | 间接进 $\mathcal{L}_{\mathrm{reject}}$（构造拒绝状态） | reject_neg typed latent |
| `local_summary` | item 级语义摘要（短文本） | 离线编码成缓存嵌入 $\mathbf{s}$ | $\mathcal{L}_{\mathrm{sum}}$ | 证据池 $\mathbf{e}_{s,\mathrm{pool}}$ |
| `traj` | typed trajectory（每步：角色 $\kappa$ + 引用证据 id + state description） | 角色序列 $\kappa_{i,k}$ + state 嵌入 $\mathbf{g}_k$ | $\mathcal{L}_{\mathrm{sem}}$（对齐 $\mathbf{g}_k$）+ 决定 typed latent 用哪个损失 | typed latent token |

**仓鼠例子对应**：query_anchor→$y_e(j)$（"仓鼠"@token1-2、"吃食物"@token4-5）；pos_anchor→patch 分布（仓鼠区域、食物+进食区域）；neg_anchor→neg 图仓鼠区域（无进食），只用于 reject_neg；local_summary→$\mathbf{s}$（"一只仓鼠正在进食"嵌入）；traj→$\kappa=$[localize, align_pos, reject_neg, summarize]，$\mathbf{g}_1..\mathbf{g}_4$。

**注意：不是所有 label 都来自教师。** Stage 2 还有两类 label 来自数据自带（非教师生成），别误以为全部 label 都出自 Seed-2.0-Pro：

- **正样本 $d^+$、负样本 $d^-$**（检索数据自带的 relevance 标注）→ 用于 $\mathcal{L}_{\mathrm{align}}$（正 doc 嵌入 $\mathbf{z}_{d^+}$）、$\mathcal{L}_{\mathrm{reject}}$（正 vs 负 doc margin）、$\mathcal{L}_{\mathrm{S2}}^{\mathrm{CL}}$（对比损失）；
- **Stage 2-B 的 NTP/MTP label** = document/query 自己的文本 token（自回归 target，非教师生成）。

| label 来源 | 具体内容 | 监督什么 |
|---|---|---|
| **教师生成（步骤 0）** | anchor 证据、local_summary、traj | 2-A 的 anchor / typed latent / 证据池 |
| **数据自带** | 正负样本对、文本 token | 对比损失、$\mathcal{L}_{\mathrm{align}}$/$\mathcal{L}_{\mathrm{reject}}$、2-B 重建 |
