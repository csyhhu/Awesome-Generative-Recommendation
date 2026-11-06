# LoopFM: Learning frOm HistOrical RePresentations of Foundation Model for Recommendation

- **arXiv**: [2605.29280](https://arxiv.org/abs/2605.29280)
- **作者**: Hua Zheng\*, Shali Jiang\*, Boyang Liu\* 等 40+ 人，通讯作者 Huayu Li（AI at Meta）
- **领域**: 工业推荐 / 广告排序、FM→VM 知识迁移、外部蒸馏（ExFM）、FM 表征作为序列特征
- **一句话**: 把大基础模型（FM）的**中间层 embedding** 抽取 → 压缩（AE + INT4）→ 按用户聚合成**历史序列**，作为小型垂直模型（VM）的**输入特征**；与 scalar KD 正交互补，线上把 transfer ratio 约**翻倍**。

---

## 1. 问题与动机

工业推荐是典型的 **两层架构**：

- **FM**：万亿参数、跨域特征、离线训练，学到丰富表征；
- **VM**：百万参数、强延迟约束、按场景/阶段部署。

二者之间通常靠 **知识蒸馏 (KD)** 连接：FM 的 scalar 预测作为 soft label 监督 VM（Meta 内即 ExFM 的外部蒸馏 + auxiliary head）。

作者观察到：随着 FM 扩到数万亿参数，**Transfer Ratio**

$$\text{TR} = \frac{\Delta \text{NE}_{\text{VM}}}{\Delta \text{NE}_{\text{FM}}}$$

持续下降（与 "teacher-student capacity gap 越大 KD 越差" 的结论一致）。原因假设：

1. **带宽瓶颈**：一个 scalar 把 FM 学到的跨域特征、多层交互、上下文信号全部压成一个数，FM 越强，"学到的" 和 "能传递的" 差距越大；
2. **特征鸿沟 (feature gap)**：FM 用了成百上千个跨域特征，VM 受延迟限制只能用子集，scalar KD 无法弥补。

**LoopFM 的思路**：在 scalar 通道之外，再开一条**高带宽 embedding 通道**——把 FM 的中间表征"物化"(materialize) 为 VM 的结构化输入特征。关键约束：**只用历史 embedding**，因此 VM serving 时**不需要实时 FM 推理**，FM 与 VM 也**没有架构耦合**。

---

## 2. 方法：三阶段模块化框架

```
FM inference (与 KD 共享同一次推理)
   │  ① Extraction：选 K 层激活拼接  e ∈ R^D  (D ~ 1e5–1e6)
   ▼
   ② Compression：AE(co-trained, stop-grad) → z ∈ R^d, tanh → INT4 (k-means INT4)
   ▼
   ③ Structuring：按 key（user / item / SID）分组 → 时间序列 S_k（或图）
   ▼
   ④ VM：Seq Arch（pooling / DIN / DMIN / self-attn）→ concat → interaction → ŷ^V
        L = L_task + λ · L_KD
```

### Stage 1：Embedding Extraction
- 选取 FM 的 K 层激活拼接：$\mathbf{e}^{(i)} = [\mathbf{h}_{l_1}; \dots; \mathbf{h}_{l_K}] \in \mathbb{R}^D$，粒度为单条 example（如一次 (user, ad) 曝光）。
- 选层原则（信息瓶颈视角）：**足够浅以保留输入信息，足够深以包含学到的交互**。
- 与 KD 共享 FM 推理 pass，额外开销可忽略。

### Stage 2：Compression
- 自编码器与 FM **共同训练**，但 **stop-gradient** 不回传 FM 主干：$\mathcal{L}_{AE} = \|\mathbf{e} - f_{dec}(f_{enc}(\mathbf{e}))\|^2$。
- encoder 最后一层用 **tanh** 把输出限制到 $[-1,1]$，便于 **INT4 量化**：`round(z·8).clamp(-8,7)/8`，相对 FP16 存储降 4×。
- **Matryoshka AE**：对多个前缀维度 $d'$ 求和重建损失，任意前缀 $\mathbf{z}_{1:d'}$ 都可用，方便部署时在特征价值与存储/延迟间灵活取舍。
- 也可替换为 PCA、随机投影、learned codebook 等。

### Stage 3：Structuring
- 按 key 分组（本文聚焦 **user-keyed 时序序列**）：
  $\mathbf{S}_u = [\mathbf{z}_{u,t_1}, \dots, \mathbf{z}_{u,t_L}]$，$t_L < \dots < t_1 < t_{cur}$
- **排除当前样本**（否则需要实时 FM 推理）；保留窗口如 30 天，截断最近 L 条（如 200）。
- 更一般地可以组织成图（user-item 二部图，边上挂 FM embedding）。

### VM 侧集成
- 用 Seq Arch（mean/sum pooling、DIN/DMIN attention、self-attention encoder）编码 $\mathbf{S}_u$，与其他特征 concat 后进交互层。
- 损失：$\mathcal{L} = \mathcal{L}_{task}(\hat{y}^V, y) + \lambda \mathcal{L}_{KD}(\hat{y}^V, \hat{y}^F)$。

---

## 3. 理论分析

### 3.1 增益分解（Gain Decomposition）
定义 FM 看到 $\mathbf{x}_{FM} = (\mathbf{x}_{VM}, \mathbf{x}_{extra})$，用户原始 VM 侧历史 $\mathbf{H}_u$，LoopFM 序列 $\mathbf{S}_u^{(k)}$。以 Bayes risk $H(y|\cdot)$ 衡量，LoopFM 信息增益**精确分解**为：

$$\mathcal{I}_{\text{LoopFM}} = \underbrace{I(\mathbf{H}_u; y \mid \mathbf{x}_{VM})}_{\mathcal{I}_{temporal}} + \underbrace{I(\mathbf{S}_u; y \mid \mathbf{x}_{VM}, \mathbf{H}_u)}_{\mathcal{I}_{cross}\ (\text{feature-gap})} - \underbrace{I(\mathbf{H}_u; y \mid \mathbf{x}_{VM}, \mathbf{S}_u)}_{\mathcal{I}_{residual}\ (\text{压缩损失})}$$

（证明：对 $(\mathbf{S}_u, \mathbf{H}_u)$ 两种顺序用 MI 链式法则后相等。）由数据处理不等式，$\mathcal{I}_{cross}$ 上界为原始额外特征历史的信息量。

压缩损失可沿 FM → AE → 量化 进一步分解，引入两个"损失比例"：
- $\tau_k$：时间通道损失比例，$\mathcal{I}_{residual} \le \tau_k \mathcal{I}_{temporal}$；
- $\eta_k \in [0,1]$：跨特征通道损失比例，$\mathcal{I}_{cross} = (1-\eta_k)\mathcal{I}_{feature\text{-}raw}$。

二者都随 FM 更大、AE 维度 d 更大、量化更细而下降。

### 3.2 Transfer Ratio 下界
两代 FM：FM$_2$ 比 FM$_1$ 多 $\delta = m_2 - m_1$ 个特征、参数更多。假设 (A1) 过参数化 FM 的 excess risk 满足 benign overfitting 包络；(A2) 新特征信息量有上下界；(A3) 新 FM 的 pipeline 损失不劣于旧 FM。

- **首次上线**：TR ≥ 0（加特征只会降 Bayes risk），并给出定量下界。
- **负迁移**：若违背 A3（新 pipeline 更差，$\eta_2 > \eta_1$）可能出现 student 变差。
- **正迁移**：
  $$\mathrm{TR}_{LB}(\delta) = \frac{-\tau_2 \mathcal{I}_{temporal} + (1-\eta_1)\underline{\kappa}^{hist}_{gap}\delta}{\bar{\kappa}_{gap}\delta + \bar\kappa_{over}\xi_1 - \underline\kappa_{over}\xi_2}$$
- **推论**：$\mathrm{TR}_{LB}$ 随 feature gap δ **单调递增**，饱和于 $(1-\eta_1)\underline{\kappa}^{hist}_{gap}/\bar\kappa_{gap}$。
- **序列长度定理**：增益关于 L 单调不减且收敛（上界 $H(y|\mathbf{x}_{VM})$），即收益递减。

### 3.3 理论启示
1. Pipeline 质量（AE 维度、量化精度、FM 强度）直接控制 TR，pipeline 损失过大会负迁移 → 动机：Matryoshka AE；
2. FM 升级引入越多新特征，LoopFM 受益越多；
3. L 越长越好但边际递减；
4. 只扩 FM 参数时，分母持续增长而分子在表征损失饱和后不再增长 → **TR 可能随 FM 规模下降**。

> 局限：结论是总体（population）层面的 MI/Bayes risk，未考虑有限样本、优化与容量限制。

---

## 4. 实验

### 4.1 设置
- 数据：**TaobaoAd**（25M 样本、22 特征，主实验）、KuaiVideo（13.7M、9 特征）、Amazon Electronics（3M、6 特征）。
- 时序流式协议（沿用 ExFM）：FM（DMIN）在 day 1–4 训练，day 5–8 推理产出 soft label 与 embedding；VM 在 day 5–7 **one-pass 流式**训练，day 8 测试。
- FM：DMIN（emb 32，DNN [512,256,128]）+ 共训 AE（d=32）；VM：FM / FmFM / DeepFM（非序列）、DIEN / DMR / DMIN（序列），均加 DMIN 式序列模块（d=32, L=50）。
- KD 用 ExFM 的 auxiliary head 框架。基于 FuxiCTR 实现。公开实验用 FP32，内部用 INT4。

### 4.2 RQ1：LoopFM vs KD（TaobaoAd AUC）

| VM | w/o distill | KD | LoopFM-only | KD+LoopFM |
|---|---|---|---|---|
| FM | 0.5790 | 0.5855 | 0.6203 | **0.6205** |
| FmFM | 0.5833 | 0.5885 | 0.6066 | **0.6106** |
| DeepFM | 0.5886 | 0.5980 | 0.6245 | **0.6344** |
| DIEN | 0.5945 | 0.5958 | 0.6335 | **0.6354** |
| DMR | 0.5967 | 0.6002 | **0.6361** | 0.6356 |
| DMIN | 0.5932 | 0.6002 | 0.6319 | **0.6346** |

- 全部 VM 上 LoopFM 都显著超过 KD；**VM = FM（DMIN）时 KD 几乎无收益，LoopFM 仍 +6.52% AUC** → 强互补证据。
- 5 seed 方差实验（DeepFM）：KD+LoopFM 0.6342 ± 0.0001，方差最低，p < 0.001。

### 4.3 RQ2：对比其他 embedding 迁移方式（DeepFM, 均基于 KD）

| 方法 | AUC |
|---|---|
| KD | 0.5980 |
| + 当前样本 FM emb 作特征（需实时 FM 推理） | 0.5985 |
| + FitNets 式表征匹配 loss | 0.5984 |
| + 仅 entity（user）embedding | 0.5983 |
| **+ LoopFM** | **0.6344** |

→ 关键在于 **"历史" embedding 序列**，而非当前样本表征或实体表征。

### 4.4 RQ3：序列结构 vs pooling
Mean-pool 0.6118，**Sum-pool 0.6300（意外地强）**，DIN attention 0.6112，**DMIN attention 0.6344**。启示：即便只存一个 sum-pooled 历史 embedding 特征（存储/计算极低）也能拿到大部分收益；更强的序列模块带来进一步提升。

### 4.5 RQ4：FM 学到的序列 vs 手工 raw ID 序列（DMIN VM，无 KD）

| | VM=FM (emb 32) | 小 VM (emb 2) |
|---|---|---|
| Baseline | 0.5932 | 0.5843 |
| + Raw User Seq | 0.6302 | 0.6186 |
| + LoopFM | 0.6319 | 0.6233 |
| + Both | **0.6362** | **0.6305** |

- VM 与 FM 等大时，LoopFM 仅略胜 raw seq（VM 可自己从 ID 学到等效表征）；
- **VM 越小，LoopFM 优势越大**（叠加在 raw seq 之上的增益：小 VM +0.0119 vs 大 VM +0.0060）→ LoopFM 价值随 FM–VM 容量差增大，正是生产常态。

### 4.6 RQ5：超参消融（DeepFM, KD+LoopFM）
- **抽取层**：Emb 层 0.6338、Hidden-0 0.6340、Hidden-1 0.6328、Deep 0.6326、**All 联合压缩 0.6341**、仅 soft-label 1 维序列 0.6286、仅 item 侧 emb 0.6291。
  - 浅层迁移更好（信息瓶颈），但纯 emb 层略逊 Hidden-0 → 交互学习有价值；
  - **仅 FM 预测分数构成的 1 维序列也有很强收益** → FM 预测的时间结构本身就有信息；
  - 内部采用**跨浅到深若干层联合压缩**。
- **FM checkpoint 频率（反直觉）**：固定 ckpt 0.6344 > 每个 split 更新 0.6311。原因是 **embedding drift**：固定 ckpt 中心漂移 ~1e-5，更新 ckpt 漂移大 ~1000×，序列内 embedding 不在同一空间，encoder 难学 → **序列内一致性 > 单条 embedding 新鲜度**（2–3 天尺度内）。
- **序列长度**：L=10→100 AUC 0.6327→0.6352，单调递增、边际递减，与理论一致。
- **维度 d**：公开数据上 8–128 几乎无差（spread 0.0008）；内部大 FM 下高维 + 宽 encoder 才有显著收益。

### 4.7 其他公开数据
- **KuaiVideo**：+0.6–1.6% AUC；mean pool 常优于 DMIN（预训练视觉 embedding 主导空间）；只用 watched 样本建序列（skip 太多会稀释兴趣）；DeepFM 上出现负迁移。无序列 baseline 下 LoopFM-only +1.6%，说明 **VM 本身缺序列特征时 LoopFM 价值最大**。
- **Amazon Electronics**：KD 是主导信号，LoopFM 小幅增益；KD+LoopFM 4 个 VM 均最佳 AUC，但部分 LogLoss 略差（校准受影响）。
- 收益差异归因于**特征丰富度**：TaobaoAd 22 特征 → $\mathcal{I}_{cross}$ 大；Kuai/Amazon 特征少，已有行为序列已覆盖大部分信号。

### 4.8 FM 容量（公开）
DMIN 从 1.9M → 31.5M 参数，KD+LoopFM 的 ΔAUC 稳定在 0.033–0.038，无明显趋势（小 FM 的 embedding 也足够丰富）。作者承认这不能代表万亿参数场景。

### 4.9 Item-only vs 交互 embedding
仅用 6 个 item 侧特征 lookup（无 user/context、无 DNN）压缩成序列，可拿到交互 embedding **86%** 的收益（TaobaoAd FM 太浅，item ID 本身强预测）。作者认为工业深 FM 中二者差距会显著变大。

### 4.10 Embedding 分析
- AE 32 维满秩（top-4/top-8 奇异值解释 28%/47% 方差）；t-SNE 呈现 soft label 的平滑梯度；
- 各维度与 soft label 和真实 label 的相关性模式高度一致（ρ=0.95）；
- VM attention：**强 recency bias**（<10min 比 4–8h 高约 3×）；同类目 1.13–1.16× lift、同品牌 1.04–1.06×；历史点击过的条目注意力反而略低 → 也利用负反馈信号。

---

## 5. 工业结果（Meta Ads，万亿参数 FM、十亿级样本、百万参数 VM）

- **Transfer Ratio**：多种 FM–VM 组合和多次 FM 更新下，LoopFM 使 KD 的 TR **约翻倍**；KD+LoopFM 总是优于单一通道 → 两者迁移的知识基本正交。
- **量化**：INT8 近无损；朴素 INT4 有 0.03–0.07% NE gap（生产中 0.02% 即显著）；**k-means INT4**（16 个适配 tanh 分布的非均匀中心）降到 0.01–0.02%，4× 压缩。
- **维度 × encoder 宽度（附录）**：NE gain 0.09%–0.14%：
  - $d_{model}$ 太小时，加大 embedding 维度无收益（都 0.09%）；
  - embedding 维度小时，加大 $d_{model}$ 仅 0.09→0.10%；
  - **两者共同放大**才能拿到 0.13–0.14%（$d_{model}$ ×3）。
- **FM 算力 scaling**：LoopFM 的相对 NE gain 与 FM 训练 FLOPs 呈**幂律关系**。
- **线上**：
  - v1：Y1H1 单 surface 上线；
  - v2：扩展到另外 3 个 surface，INT4 带来 4× 存储降低；
  - v3：使用最新 FM；
  - 转化提升：首个半年 **+0.5%**；次半年两次独立 launch 分别 **+1.03%**、**+1.22%**。

---

## 5+. 内网补充（Meta Internal，⚠️ 仅限内部，请勿公开发布）

> 来源：内网 Workplace 帖子、wiki、diff、task 检索整理。成本、SEV 等敏感信息已省略；部分数字来自索引中格式错乱的表格，引用前请回原帖核对。

### 起源（2024-11，原名 "Learning frOm HistOry and Present with FM"）
- **动机**：当时 KD 的 transfer ratio 只有约 **25%**，预计扩到 O(10K) GPU 后还会降。KD 被认为有四个缺陷：忽略 FM 的历史输出、受 FM/VM 特征不匹配影响、把一切压成一个数、serving 时没有 soft label。目标是在 KD 之上把 TR 从 25% 提到 **>45%**，长期覆盖 O(100) 个 VM、O(1–10) 个 FM。
- **FM 是 GEM / OmniFM**（早期叫 ExLarge FM / FB FM），与 Kunlun 无关。
- **原型**：先在 FB FM v3（而不是 VM）上验证，embedding 来自 FB FM LWB v2。**click、conv NE 增益都约 0.22%**。
  - 配置：30 天回看、max len 500、1 天防泄漏 buffer、所有 DHEN 层 + click/conv 预测分数。
  - 原始维度 3228，所以需要 AE 压缩。
- **原型消融**：
  - max len 200→500 再 +0.05% NE；
  - 22 天 buffer 不如 1 天；
  - 用全部 DHEN 层优于只用最后一层（最后一层和历史 label 绑得太紧）；
  - DCN 比 MaskBlock 好 +0.015%；
  - 把 LoopFM concat 到更深层（shared-arch output）反而回退；
  - 只用 click 序列比用 impression 序列差 0.06%。
- **防泄漏规则**：历史样本的 commit 时间戳 < 当前请求时间戳。有泄漏风险的事件占比：1 天 buffer 0.19%，3 天 buffer 0%。

### 版本演进

| 版本 | 时间 | 源 FM | 压缩 | 结果 |
|---|---|---|---|---|
| v1 | 25H1 | FB FM（DHEN1–8 + DENSE_PROJ）、IG FM（第 4、7 层） | FP16 | FB CMF 上线，约 0.13% 估计 GAS |
| v2 | 25H2 | OmniFM v1，单层联合压缩 | **INT4** | 约 0.25% 估计 GAS |
| v3 | 26H1 | OmniFM v3 | INT4/INT8 | CMF：TR **14%→33%**；NE：click +0.035%、conv +0.074%、click_reels +0.065% |
| v4 / **SyncLoopFM** | 26H1 | OmniFM/GEM v4 | INT8，输入维度 256→320 | **TR >40%**；12 个 launch（CMF、IG、MAI、AF OC），归因 GAS **>1%**；CMF 上 SyncLoopFM+KD 总 NE >0.3% |
| v5 | 26H2 | OmniFM v5 | **FSQ**（存成 INT4）+ Matryoshka，768 维 | 提案阶段；预计贡献 GEM 26H2 承诺 GAS 的约一半 |

- 论文里的 "+0.5% / +1.03% / +1.22% conversion" 对应内部 25H1、25H2 两个半年的 launch。内部同时记录的归因 GAS：25H1 +0.24%、25H2 +0.37%（论文里这两个数被注释掉了）。
- **SyncLoopFM**：解决 LoopFM 与 KD 用的 GEM 版本不一致造成的跨版本滞后。
- 领域 FM 也接入了：DPA LoopFM v1 已上线；BM FM 的 NE 增益 >0.2%。

### 工程 / 存储
- **v1 数据流**：广告曝光 → AdLoggerLite → DAS 从 bulk eval 取中间层 embedding → FP16 编码 → 长度为 1 的 EBF（event-based feature）→ 聚合进 **Up2x** 供 serving，同时写入 `event_based_features` 供训练。平均序列长度约 117。
- **v5**：UDS 事件 `OMNIFM_V5_LOOP_FM_EVENT`，key 是 user，走实时链路；有 joint compression（384 维）和 FSQ joint compression（768 维）两种特征。
- **延迟要求**：在线 EBF P90 5 分钟 / P50 1 分钟；离线 EBF（供 FM 自身训练）P90 1 小时。还要求支持 **blockout**：坏 checkpoint 产出的 embedding 能告警、回滚、过滤、删除、回填。
- **k-means INT4**：16 个中心，16 个 index 打包进一个 int64；有静态和周期刷新两种。比普通 INT4 好约 +0.05% NE。
- **Hybrid EBF**：v3/v4 接入后约省 40% 的 D&F 容量。规模参考：v3（INT4）约 4000 亿条 embedding，约 50TB。
- **VM 侧结构**：
  - 原型用 HRP：解码 → MLP 投影 → DCN；
  - 生产环境作为 `loopfm` event tower 接入 UHM，`max_seq_len=200`，用 Interformer/Transformer 序列结构；
  - FB HI CTR 配置：`embedding_dim=128`、`input_dim=256`、GPU 端解码。

### 已知问题
- **QPS/延迟**：IG CTR 上线时出现过约 70% 的 QPS 回退，靠 GPU 端解码、非 KJT 路径、专用 router、PT2 修复。v4 在 IG CTR 上又出现过同类问题。
- **数据链路**：EBF validator 改动导致训练行丢失、模型新鲜度下降；配置冲突让 consolidated 模型的 LoopFM 收益受威胁；v5 出现过 Up2x 数据损坏，需要清理。
- **冷启动**：在 Taobao 复现中约 11–14% 的用户没有历史。
- **监控**：`loopfm_health_stats` dashboard，表 `fct_loopfm_embedding_stats`、`fct_loopfm_embedding_norm`。

### 后续工作（对应论文 Future Directions）
- **SelfLoopFM**：即论文里的 self-LoopFM。trainer 维护自己的实体 embedding store。
  - 实现：表 offload 到 UVM，按 `sid % hash_size` 寻址，max len 800，320 维，INT8/INT4/k-means INT4。
  - scaling：NE 随 store 大小近似 log-linear，1→8TB 对应 0.02%→0.09%。
  - 在 OmniFM v5 上用不到 LoopFM v4 10% 的存储就能拿到接近的效果。
  - 当前卡在 serving 内存，计划用 SSD offload；key 计划扩展到广告账户、广告主。
- **Generative LoopFM**：用轻量 GPT-2 式模型预测"下一个" FM embedding，在 v4 之上再 +0.06% NE（ETI 特征 320 维，7 天窗口内每个用户 24 个事件）。
- **Organic / Unified LoopFM（2026-09）**：即论文里的跨域迁移。用 Blue Reels RankFM 的 embedding，Reels +0.30% NE、CMF +0.10% NE（1000 个事件，INT4）。INT4 与 FP16 打平；在 user×item embedding 之上再加 item embedding 没有额外收益。
- **其他**：TR 目标 >50%；AppFM、WebFM、CVR FM、IG FM 接入共享量化 infra；压缩方式探索 RQ-VAE、PCA、Seq2Seq。

---

## 6. 与相关工作对比

| 方法 | 迁移内容 | 是否需要实时 FM | 特点 |
|---|---|---|---|
| KD / ExFM | 当前样本 scalar | 否（离线 logging） | 低带宽，capacity gap 大时变差 |
| FitNets / CRD | 当前样本表征 + 辅助 loss | 训练时需要 | 本文实验几乎无收益 |
| 实体 embedding（PinSage、SUM） | user/item 级 embedding | 否 | 缺交互级信息 |
| LATTE / VISTA | 序列模型表征异步缓存 | 否 | 偏 serving 效率 |
| **IAT**（ByteDance，并行工作） | 历史训练实例压缩为 token 序列 | 否 | 序列特征工程视角，思路高度相似 |
| **LoopFM** | **历史 FM 中间 embedding 序列** | **否** | FM→VM 迁移视角 + 理论分析，与 KD 正交 |

---

## 7. 个人点评 / Takeaways

1. **"把大模型的推理副产物变成小模型的特征"** 是一种低成本、解耦的迁移方式：FM 推理只需在 KD logging 时顺带写 embedding，VM 侧就是一个普通的 embedding 序列特征，无需 lookup table 和 ID embedding 内存。
2. **历史 vs 当前**：当前样本的 FM embedding 几乎无用、历史序列却巨大收益——本质上 LoopFM 是给 VM 提供了一条 "由 FM 用全量跨域特征加工过的用户行为序列"，相当于把 FM 的 feature gap 以序列形式补给 VM。
3. **一致性 > 新鲜度**：序列特征由不同 ckpt 产出时的 embedding drift 是工程上的关键坑，提示需要 ckpt 对齐（anchored training、线性投影对齐、EMA）或定期全量重刷。
4. **Encoder 容量与 embedding 维度必须同步扩**，否则互为瓶颈——与 IAT/GR 中 "更强的序列模块才能吃下更丰富 token" 的结论一致。
5. Sum-pool 就有大收益、soft label 1 维序列也有效，说明落地可分级：先上 pooled 特征 / score 序列，再上完整序列 + attention。
6. 未来方向：item-keyed 序列（广告侧需求信号）、图结构、跨域（organic FM → ads VM）、**self-LoopFM**（FM 消费自身历史 embedding 的自增强闭环）、向量量化（RQ-VAE）替代标量 INT4。
7. 局限：存储开销大、冷启动用户无历史、压缩必然损失信息、VM 推理延迟增加、公开 benchmark 规模（~6M 参数）与工业差距大。
