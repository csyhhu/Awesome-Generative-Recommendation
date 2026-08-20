# Long Sequence

| Publication                                                                                                                                  | Affiliation | Year                 |
| -------------------------------------------------------------------------------------------------------------------------------------------- | ----------- | -------------------- |
| [LONGER: Scaling Up Long Sequence Modeling in Industrial Recommenders](../../Summary/Ranking/summary_longer_long_sequence.md)                | ByteDance   | 2025.09 (RecSys '25) |
| [LLaTTE: Scaling Laws for Multi-Stage Sequence Modeling in Large-Scale Ads Recommendation]()                                                 | Meta        | <br />               |
| [Make It Long, Keep It Fast: End-to-End 10k-Sequence Modeling at Billion Scale on Douyin](../../Summary/Ranking/summary_stca_rlb_ext_10k.md) | ByteDance   | 2025.11 (WWW 2026)   |
| [HyFormer: Unified Architecture for Long Sequence Ranking](../../Summary/Ranking/summary_hyformer_unified_arch.md)                           | ByteDance   | 2026.08 (RecSys '26) |
| [Teacher Retains Full Tokens, Student Merges Efficiently](../../Summary/Ranking/summary_tm20k_token_merge.md)                                | ByteDance   | 2026.08              |

## Unification of LONGER, STCA, HyFormer

> 三者（LONGER、STCA、HyFormer）可视为同一框架下、不同块启用/禁用的特例。框架有 5 个可配置块，**某块为空（None）即退化为另一种方法**。

### 1. 统一问题定义

给定：

- 目标 $TG \in \mathbb{R}^{N\_t \times d}$：目标 item 表征 token；
- 非序列特征 $NS \in \mathbb{R}^{N\_{ns} \times d}$：user / context / cross 等非序列特征 token；
- 序列特征 $SS \in \mathbb{R}^{N\_{ss} \times d}$：用户行为序列 token，$N\_{ss}$ 很大。

### 2. 统一层结构（5 个可配置块）

每个统一层（layer $l$）由 5 个可配置块组成：

| Module      | 作用                                              | 为空（None）时的退化                 |
| ----------- | ----------------------------------------------- | ---------------------------- |
| QueryGen    | 由 $NS/TG/\overline{SS}$ 生成 $N\_q$ 个 query token | 复用上一层 query（`reuse`）         |
| SeqEnc      | 由 $SS$ 逐层产生 $(K\_l, V\_l)$                      | 不访问序列（LONGER 自注意力层）          |
| QueryDecode | query 对序列 $K/V$ 做 cross-attention（**三者共有**）     | 跳过解码（LONGER 自注意力层）           |
| HighOrder   | 在 decoded query（+$NS$）上做高阶交互                    | **空 → 退化为 STCA**（纯 cross 堆叠） |
| QueryFuse   | 把当前层输出融合回 query 供下一层使用                          | 直接传递                         |

统一层前向（$\overline{SS}=\text{MeanPool}(SS)$）：
$$
\begin{aligned}
Q^{(l)}        &= \text{QueryGen}\_l\big(NS, TG, \overline{SS}, Q^{(l-1)}\big) \in \mathbb{R}^{N\_q \times d} \\
(K\_l, V\_l)     &= \text{SeqEnc}\_l\big(SS\big) \in \mathbb{R}^{L\_l \times d} \\
\tilde Q^{(l)} &= \text{QueryDecode}\big(Q^{(l)}, K\_l, V\_l\big) \in \mathbb{R}^{N\_q \times d} \\
\hat Q^{(l)}   &= \text{HighOrder}\_l\big(\[\tilde Q^{(l)}; NS; TG]\big) \in \mathbb{R}^{N\_q \times d} \\
Q^{(l+1)}      &= \text{QueryFuse}\_l\big(\hat Q^{(l)}, Q^{(l)}\big) \in \mathbb{R}^{N\_q \times d}
\end{aligned}
$$

最终 $\hat y = \sigma\big(\text{PredHead}(Q^{(L)}\[\text{target 位}])\big)$。

### 3. Baseline / LONGER / STCA / HyFormer 的模块实现

> 每行为一种方法（Baseline 为全量 dense self-attention 参考；LONGER / STCA / HyFormer 为长序列方法），每列为 5 个模块的实现方式。仅描述怎么做，不写具体类名；有多种实现时取论文 report 效果最优的变体（LONGER 取 $K{=}4$ Token Merge；HyFormer SeqEnc 取 `longer_style` 部署档）。

| Method | ① QueryGen | ② SeqEnc | ③ QueryDecode | ④ HighOrder | ⑤ QueryFuse |
|---|---|---|---|---|---|
| **Baseline** | identity：$Q{=}$ 原始 $[TG;NS;SS]$ 全量，无单独生成 | $K,V{=}$ 原始全量投影（$W_K,W_V$），不压缩 | 全量 self-attention $+$ FFN，$[TG;NS;SS]$ 互 attend（$O(L^2)$） | 无（identity，交互已在 ③ 完成） | residual |
| **LONGER** | 全局 token（target$+$CLS$+$UID…）拼接 recent-$k$ 采样序列，$N_q{=}m{+}k$ | Token Merge（相邻 $K{=}4$ sum-pool）$+$ InnerTrans（组内 transformer），$L{\to}L/K$ | 仅 layer 1：cross-attention，$Q{=}[G;H_S]$ attend 压缩序列 $KV$（长度 $L/K$）；后续层跳过 | 后续 $N$ 层：self-causal attention 在 $m{+}k$ 压缩 token 上（保留时序） | residual |
| **STCA** | query $=$ target 单 token，$N_q{=}1$（严格 $O(L)$ 根因） | identity：$SS$ 不变，仅逐层 $W_K^l/W_V^l$ 投影不同 | 每层 single-query cross，$Q{=}$target attend history；计算重排避免物化 $XW_K/XW_V$（RLB 前提：history target-agnostic） | **空（∅）** → 纯 cross 堆叠，无 history-history 二阶关系 | target-conditioned fusion：$\text{concat}[\hat Q;Q_{prev}]{\to}\text{Linear}{+}Q_{prev}$，逐层累积细粒度 |
| **HyFormer** | $N$ 个独立 FFN 由 $\text{GlobalInfo}{=}[NS;\text{MeanPool}(SS)]$ 分头发射差异化 global query | `longer_style` 档：$S_{short}$ 作 Q 对 $SS$ cross-attention 压到 $L_s$ | 每层 cross-attention，$Q{=}$global queries attend $KV{=}$SeqEnc 输出 | Query Boosting $=$ MLP-Mixer（跨 token 子空间 mixing）$+$ per-token FFN，在 $[\tilde Q;NS]$ 上 | residual $+$ mixer 输出 |

### 4. 各块理论复杂度（per-sample FLOPs）

> 约定：matmul $(m,k)@(k,n)=2mkn$；$L{=}N_{ss}$，$N_{full}{=}N_t{+}N_{ns}{+}L$。辅助量：$\text{Linear}(n,a,b){=}2nab$，$\text{MHA}(n_q,n_{kv},d){=}4(n_q{+}n_{kv})d^2{+}4n_q n_{kv} d$（含 $W_Q/W_K/W_V/W_O$），$\text{FFN}(n,d){=}16nd^2$，$\text{Mixer}(d,T){=}2dT^2$。本框架 SeqEnc 与 QueryDecode 各投影一次 $K/V$（实现冗余，不影响相对比较与退化结论）；下列公式与 §5 数值同口径。LONGER 的 ①②③ 仅出现在 layer 0（$\times 1$），④ 出现在 layers 2..N（$\times N$）；Baseline / STCA / HyFormer 每层重复。

| Method | ① QueryGen | ② SeqEnc | ③ QueryDecode | ④ HighOrder | ⑤ QueryFuse |
|---|---|---|---|---|---|
| **Baseline** | $0$ | $4N_{full}d^2$ | $\text{MHA}(N_{full},N_{full},d){+}\text{FFN}(N_{full},d)$ **（$O(L^2)$）** | $0$ | $0$ |
| **LONGER** | $2N_t d^2$ | $\text{MHA}(\tfrac{L}{K},\tfrac{L}{K},d){+}\text{FFN}(\tfrac{L}{K},d){+}4\tfrac{L}{K}d^2$ | $\text{MHA}(N_q,\tfrac{L}{K},d){+}\text{FFN}(N_q,d)$，$N_q{=}N_t{+}k$ | $\big[\text{MHA}(N_q,N_q,d){+}\text{FFN}(N_q,d)\big]{\times}N$ | $0$ |
| **STCA** | $0$ | $4Ld^2$ | $\text{MHA}(1,L,d){+}\text{FFN}(1,d)$ **（$O(L)$，无 $L^2$ 项）** | $0$ | $4N_q d^2$ |
| **HyFormer** | $2Nd^2$ | $\text{MHA}(L_s,L,d){+}4L_s d^2$ | $\text{MHA}(N_q,L_s,d){+}\text{FFN}(N_q,d)$ | $\text{Mixer}(d,T){+}\text{FFN}(T,d)$，$T{=}N_q{+}N_{ns}{+}N_t$ | $0$ |

**关键洞察**：
- **STCA 严格 $O(L)$**：$N_q{=}1$ 使 ③ $\text{MHA}(1,L,d)$ 中 $4n_q n_{kv} d\to 4Ld$（线性）且无 $n_q^2$ 项，仅剩 $4(1{+}L)d^2{+}4Ld$，二者均线性于 $L$；系统侧再靠 RLB 把线性 FLOPs 转为真实带宽收益。
- **LONGER 把 $L^2$ 消灭在 ②**：Token Merge 先 $L\to L/K$，③ 的 KV 长度变 $L/K$，故 $n_q n_{kv}\propto(N_t{+}k)\cdot L/K$；④ 自注意力只在 $N_q{=}N_t{+}k$ 上做，$O((N_t{+}k)^2)$ 与 $L$ 无关。
- **HyFormer** ② `longer_style` 把 KV 压到 $L_s\ll L$，③ 变 $O(N_q\cdot L_s)$；④ Mixer 的 $2dT^2$ 是 $T{=}N_q{+}N_{ns}{+}N_t$（常数级，不含 $L$），主导项在 ②。
- **Baseline** ③ 的 $4N_{full}^2 d$ 是唯一 $O(L^2)$ 项，三者分别用"压缩 ②（LONGER）/ 单 query ③（STCA）/ 压缩 ②（HyFormer）"消除之。

### 5. 复杂度交叉验证示例（$d{=}64, N\_t{=}1, N\_{ns}{=}13, N\_{ss}{=}512$）

> 各 cell 为该 method 对应 module 的 per-sample FLOPs（与 §4 公式同口径）。LONGER / STCA / HyFormer 的 Total 与 [WorkSpace/unified_long_seq.py](../../WorkSpace/unified_long_seq.py) 的 `FlopCounter` 运行时记录**完全一致**（理论 $=$ 实际）；Baseline 为解析参考（单 dense block，不在 .py 中）。$N_{full}{=}526$。

| Method | ① QueryGen | ② SeqEnc | ③ QueryDecode | ④ HighOrder | ⑤ QueryFuse | PredHead | **Total** |
|---|---|---|---|---|---|---|---|
| **Baseline**（1 dense block） | 0 | 8,617,984 | 122,536,960 | 0 | 0 | 8,320 | **131,163,264** |
| **LONGER**（1 cross $+$ 3 self） | 8,192 | 18,874,368 | 13,680,640 | 37,620,480 | 0 | 8,320 | **70,192,000** |
| **STCA**（8 层纯 cross 堆叠） | 0 | 67,108,864 | 68,812,800 | 0 | 131,072 | 8,320 | **136,061,056** |
| **HyFormer**（4 层全开） | 98,304 | 201,326,592 | 18,546,688 | 4,604,416 | 0 | 8,320 | **224,584,320** |

**观察**：
- $L{=}512$ 时，Baseline 单个 dense block（$\sim$131M）已与 STCA 8 层（$\sim$136M）相当——即 STCA 用 8 层 cross 的代价约等于 1 次全量 self-attention。
- ③ 是各方法的主战场：Baseline ③ 占自身 93%（$O(L^2)$）；STCA ③ 占 51%（$\text{MHA}(1,512,64)\approx$ 8.5M，线性于 $L$）；LONGER ③ 仅 13.6M（KV 被压到 $L/K{=}128$）；HyFormer ③ 占 8%（KV 压到 $L_s{=}256$，但 ② SeqEnc 反而占 90%——因 `longer_style` 仍对全长 $L$ 做 cross）。
- 规模外推：$L{=}10^4$ 时 Baseline ③ $\approx 4{\times}10^8{\times}64\approx 2.56{\times}10^{10}$，而 STCA 8 层 ③ $\approx 1.33{\times}10^9$（$\sim$19× 差距，且随 $L^2$ 持续扩大）。