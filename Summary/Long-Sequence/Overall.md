# 长序列压缩专题总结

本目录收录**以"让超长用户行为序列可用且便宜"为核心贡献**的论文。正如 [Long Sequence Tutorial](../../Tutorials/Long-Sequence.md) 所说，长序列处理的本质是**把序列变短**：要么在数据源侧缩减（按 target / 语义检索子序列），要么在模型内压缩（token merge、learned query、sketch），要么把压缩挪到请求链路之外（上游异步预计算、缓存）。

把用户历史离散成 token 的 FOUNDv2 也可以看作一种长期行为压缩，它放在 [`../SID/`](../SID/summary_foundv2_user_tokenizer.md)。MixFormer 是 SequenceO1 的融合骨干，但其贡献主要在 dense 与序列的 co-scaling，仍留在 [`../Ranking/`](../Ranking/summary_mixformer_co_scaling.md)。

## 收录论文

| 论文 | 机构 | 阶段 | 最大长度 | 压缩方式 | 与 target 的关系 | 总结 |
|---|---|---|---|---|---|---|
| LONGER | 字节 | 精排 (CVR) | 2K~10K | 全局 token + recent-k 采样 query 做一次 cross-causal 压缩；Token Merge + InnerTrans | 序列侧 target 无关，KV 可缓存 | [link](summary_longer_long_sequence.md) |
| STCA + RLB (Make It Long, Keep It Fast) | 抖音 | 精排 | 10K | 不压缩：单 query（target）堆叠 cross attention，$O(L)$；训练稀疏、推理稠密 | target 相关，不可跨 target 复用 | [link](summary_stca_rlb_ext_10k.md) |
| SequenceO1 | 抖音 | 精排 | 100K | Sketch Attention：$k$ 个 prototype，**按 prototype 维 softmax**，100K → 1K sketch，按用户缓存 | sketch target 无关；下游再做 STCA | [link](summary_sequenceo1_100k_sketch_attention.md) |
| TM20K | 字节 | 精排（广告） | 20K | 规则式 Token Merge（同 ID 局部合并 / 按位置分段 / 按层金字塔）+ 全量 Teacher 蒸馏 | full attention，target token 不合并 | [link](summary_tm20k_token_merge.md) |
| HyFormer | 字节搜索 | 精排 (CTR) | 3K | N 个 global query 逐层 cross-attend 序列，并与 NS token 做 MLP-Mixer；序列侧可选 LONGER-style 压缩 | query 含全局信息 | [link](summary_hyformer_unified_arch.md) |
| ChronicleRec | 腾讯 | 精排 (CTR/CVR) | 2K~4K（截断） | 近/中/远多粒度 merge + **穿插式因果** query token + 多 horizon 分支 + Chronicle Alignment 预训练 | target 无关，可缓存 | [link](summary_chronicle_tokens_lifelong.md) |
| IAT (Instance-As-Token) | 字节 | 精排（广告） | 256 个实例 | 不缩短长度，而是把每条历史**训练实例**（约 6000 维）压成 64 维 InsEmb，提高每个 token 的信息密度 | 上游源模型压缩，下游消费 | [link](summary_iat_instance_as_token.md) |
| LLaTTE | Meta | 精排（广告） | 数千 | 异步上游大模型把长历史压成 2048 维 user embedding；在线小模型做金字塔裁剪 | 上游 target 无关，在线 target 相关 | [link](summary_llatte_scaling_laws.md) |
| MUSE | 淘宝 | 精排（展示广告） | 100K | GSU 用多模态余弦相似度检索 top-K；ESU 用 SimTier 直方图 + SA-TA（ID 与多模态融合的 attention） | target 相关检索 | [link](summary_taobao_muse_lifelong_gsu_esu.md) |
| GEMs | 快手 | 召回（GR） | 100K+ | 三流分治：近期 full attention / 中期轻量 indexer 做 top-K 稀疏注意力 / 终身离线 QLA 压缩；无参数求和融合 | 召回无 target | [link](summary_gems-multi-stream-gr.md) |
| GLASS | 快手 / 清华 | 召回（GR） | ~1K | SID-Tier：码字原型 × 长期历史的兴趣直方图；SHS：用每条 beam 生成的 sid₁ 在长期历史中 hard search | 用 SID 前缀充当 target | [link](summary_glass_coarse_to_fine_long_term_interest.md) |

## 一、四类压缩范式

```text
              ┌─ A. 检索式缩减（数据源侧）      MUSE, GLASS, GEMs-mid
 超长历史 ────┼─ B. 不压缩，降复杂度到 O(L)     STCA
              ├─ C. 模型内压缩（端到端）        LONGER, HyFormer, TM20K, SequenceO1, ChronicleRec, GEMs-lifecycle
              └─ D. 上游预计算（两阶段迁移）    LLaTTE, IAT
```

### A. 检索式缩减：GSU/ESU 的新形态
- **MUSE**："GSU 从简、ESU 从丰"。GSU 用高质量多模态 embedding（SCL）做内积检索即可，attention 或 ID 融合检索都没有收益；收益来自 ESU 的 SimTier + SA-TA。上线 100K 版本后，相对 5K SIM 基线 CTR +12.6%、RPM +5.1%。
- **GLASS**：召回阶段没有 target，SIM 无法直接用。GLASS 的关键观察是：**GR 已生成的前缀 sid₁ 就是 target 的粗粒度代理**，于是对每条 beam 跑一次 SIM（GSU = 按 ĉ₀ 取桶，ESU = cross-attn）。另外，一级码本大小在 hard search 下变成了"检索桶的密度"。
- **GEMs 中期流**：把 GSU 换成可学习的 indexer。训练时用 KL 蒸馏全注意力分数，推理时取 top-K 做稀疏注意力；T=5000 时延迟从 103.68ms 降到 32.2ms，Recall 反而更高（去噪）。

**局限**：检索与最终目标割裂（MUSE 用冻结表征），并且检索依赖 target，同一用户的不同候选无法共享计算。

### B. STCA：不压缩，只把复杂度降到线性
以 target 为唯一 query，逐层 cross attend 全部历史，复杂度 $O(L)$。重排计算顺序后不再物化 $XW_K / XW_V$。配套两个系统设计：
- **RLB**：同一用户的多个 target 共享历史侧计算，带宽减少 77%~84%；
- **长度外推**：训练平均约 2K、推理 10K，用 U 形 Beta 分布随机截断，以约 1/3 的训练成本获得 10K 窗口约 80% 的增益。

**局限**（SequenceO1 指出）：外推比只有 4~5 倍，到 100K 训练长度仍需 20K+；单 query 重排后没有可缓存的 KV，**无法跨 target 复用**。

### C. 模型内压缩：主战场
| 工作 | 压缩算子 | 压缩比 | 关键设计 |
|---|---|---|---|
| LONGER | cross-causal：$Q=[\text{global};\text{recent-}k]$，KV = 全序列；之后在 $m+k$ 个 token 上做 self-attn | $L \to m+k$ | recent-k 真实行为做 query 远优于 learnable query；只采样 40% 序列就保留 95%+ 收益 |
| HyFormer | global query 逐层 cross-attn，再做 MLP-Mixer boosting，逐层回喂 | $L \to N$ | 序列压缩与特征交互**交替**进行，而不是先压缩再交互 |
| TM20K | 规则 merge：LITM（同 ID 局部合并）、PATM（近 3K 不压缩，远处 K=2/3/4）、LPTM（每 2 层减半） | 平均长度 8.8K → 1.8K | 压缩造成的损失由全量 20K Teacher 蒸馏补回 |
| SequenceO1 | Sketch Attention：$A=\text{softmax}_{\text{prototype}}(QK^\top)$，$\widetilde X=AX$，堆叠 2 层 | 100K → 1K | **分配而非选择**：每个 token 必须贡献单位质量，冷门兴趣不会在看到 target 之前被丢掉 |
| ChronicleRec | 多粒度 merge + 穿插式 query + 因果编码 | 2K~4K → 55 个 token | "sequence-in, sequence-out"：每个 token 有明确的时间感受野，多 horizon 分支显式去冗余 |
| GEMs 终身流 | 拟线性注意力（QLA）离线压缩，配重建损失 | 10 万+ → 压缩表示存入 KV | 离线算、在线查 |

### D. 上游预计算：把算力挪出请求链路
- **LLaTTE**：在线模型每源只能容纳约 400 个 token、层数也浅；新增算力投入异步上游模型（序列 FLOPs 为在线的 45 倍以上），输出 2048 维 embedding 写入 Feature Store。上游提升约 **50%** 迁移到下游（transfer ratio），线上 NE −0.25%。LLaTTE 还发现：上游缺少候选上下文，**序列长度的 scaling 斜率只有下游的一半**。
- **IAT**：压缩的对象不是"长度"而是"宽度"。每条历史实例的数千维特征被压成 InsEmb，下游序列的每个 token 信息量远高于手工的 item ID / 类目序列。User-Order 源模型（按用户聚合、因果 SIT）明显优于逐条独立压缩，而且 InsEmb 可以跨域迁移。

## 二、关键设计维度

### 1. target 相关还是 target 无关：决定能否复用
| | target 相关 | target 无关 |
|---|---|---|
| 代表 | STCA、MUSE、LLaTTE 在线部分 | LONGER 序列侧、SequenceO1 sketch、ChronicleRec、LLaTTE 上游、GEMs 终身流 |
| 优点 | 信息损失小，可以精确挑出与候选相关的行为 | 一次计算，跨候选、跨样本、跨请求复用；可以前移到请求入口（Pipeline Lift / 异步预取） |
| 代价 | 成本随候选数线性增长 | 压缩发生在看到 target 之前，必须保证**覆盖**，否则冷门兴趣丢失 |

主流方案已收敛为**两段式组合**：target 无关的轻量压缩器（可缓存）+ target 相关的重量级推理器（只在压缩后的短序列上运行）。SequenceO1 用 sketch + STCA，LLaTTE 用上游 embedding + 在线 LLaTTE，ChronicleRec 用 Chronicle Tokens + target-attention 读出，LONGER 用 KV cache + global token。

### 2. 近期保真、远期粗化
几乎所有工作都给出同一个归纳偏置，只是实现不同：
- **TM20K**：注意力分析显示最新 10% 的 token 贡献约一半注意力，所以 PATM 对近期不压缩、远期压缩 4 倍；
- **ChronicleRec**：near 段 stride=1，far 段大 kernel 粗聚合；query 锚点越近越密；
- **GEMs**：近期 full attention / 中期稀疏 / 终身离线压缩；
- **LLaTTE**：在线模型逐层裁剪旧行为（pyramidal）；
- **LONGER**：query 用 recent-k 行为；**SequenceO1**：近期 10K 用 STCA 全分辨率建模，同时整段 100K 进入 sketch（两个分支在时间上重叠）。

但也有反面证据提醒**远期不能丢**：ChronicleRec 去掉最近 1000 条后仍有 0.5499 GAUC（Short-Attn 只有 0.5307）；LLaTTE 的注意力在 200~1600 范围内都不可忽略，还呈现约 24 小时的周期峰值；GEMs 指出可学习融合会**坍缩到近期流**，于是改用无参数求和。

### 3. 压缩保留了多少增益
| 工作 | 参照上界 | 保留比例 | 成本 |
|---|---|---|---|
| SequenceO1 | 直接 STCA 100K | **83%** | 训练 / 推理 FLOPs 降约 50× / 64× |
| TM20K (+KD) | 20K Teacher | **约 85%** | 吞吐 83K vs Teacher 11K（基线 5K 模型 88K） |
| STCA 外推 | 全量 10K 训练 | **约 80%** | 训练成本约 1/3 |
| LLaTTE | 上游 ΔNE | **约 50%**（transfer ratio） | 在线 P99 延迟无可测变化 |
| ChronicleRec | Full-Attn | GAUC 差 0.0021（KuaiRand） | 训练加速约 4.8× |

几个数字口径不同（同一模型的不同长度、师生、跨阶段迁移），不能直接横比，但传递了一致的信息：**端到端可训练的压缩器能保留 80%+ 的增益；解耦的上游迁移大约只能保留一半**。

### 4. 系统设计往往比算子更重要
| 系统手段 | 代表 | 作用 |
|---|---|---|
| KV Cache Serving | LONGER | 多候选打分的吞吐退化从 −40% 降到 −6.8% |
| RLB / MRLB | STCA / SequenceO1 | 同一用户的多 target（或短时间内的多个请求）共享用户侧计算 |
| 用户侧缓存 + Pipeline Lift | SequenceO1、MUSE | 序列计算前移到请求入口，与召回并行；命中缓存时 100K 原始特征无需物化和传输 |
| 自定义 kernel | SequenceO1 (FlashSA) | prototype-wise softmax 无法复用 FlashAttention，需要自写；按 token 分块累加即可 |
| 变长打包 | TM20K (Stack Sequence)、STCA (ragged) | 20K 序列可节省约 10GB 显存 |
| 稀疏 ID 去重 | HyFormer (GPU Pooling) | 唯一 ID 只占约 25%，降低 H2D 传输 |
| 异步上游 + Feature Store | LLaTTE、IAT | 把大模型推理移出请求链路 |

### 5. 把训练成本也压下去
| 手段 | 代表 |
|---|---|
| 长度外推（训练短、推理长） | STCA |
| 一次性 Teacher + 蒸馏（Teacher 部署 8 个月未重训，服务数十个 Student） | TM20K |
| 压缩器预训练（mask 近期行为并预测） | ChronicleRec |
| 训练时蒸馏 indexer，推理时稀疏 | GEMs |
| 缓存 sketch（训练侧 local KVCache，命中率约 0.5） | SequenceO1 |

## 三、演进脉络（字节 / 抖音一线）

```text
LONGER (2K~10K, RecSys'25)        cross 压缩 + self-attn，KV cache
   └─ HyFormer (RecSys'25)        压缩与特征交互逐层交替
STCA + RLB (10K, WWW'26)           不压缩，单 query O(L)，长度外推
   └─ SequenceO1 (100K, RecSys'26) O(L) 也不够 → target 无关 sketch + 缓存 + MRLB，复用 STCA 做推理
TM20K (20K)                         规则 merge + Teacher 蒸馏，作为 LONGER / HyFormer 的高效替代
IAT                                 换一个维度：提升每个 token 的信息密度，而非延长序列
```

SequenceO1 对 10K → 100K 的诊断很有代表性：**到 100K 时瓶颈从 FLOPs 变成系统性的**，特征存储、样本体积、主机-设备通信、训练和推理都随长度膨胀。所以必须引入一个固定大小、可缓存的用户侧状态，把与长度相关的工作变成可摊销的成本。

## 四、总结与开放问题

**共识**
1. **先压缩、再推理（Compress-then-Reason）**：target 无关的轻量压缩器加上 target 相关的重量级推理器，是 10K 以上长度的主流答案。
2. **target 无关的压缩必须保证覆盖**：SequenceO1 的 prototype-wise softmax、ChronicleRec 的穿插式因果 query 和多 horizon、GEMs 的无参数融合，都在防止压缩或融合塌缩到近期 / 显著区域。
3. **用真实信号初始化 query 优于纯可学习 query**：LONGER 的 recent-k 优于 learnable；HyFormer / LLaTTE 的 query 来自非序列特征。SequenceO1 是例外，它的增益主要来自归一化方向，而不是 query 来源。
4. **信息质量与长度相乘**：LLaTTE 发现 ID-only 模型加深几乎无效，加入内容 embedding 后深度和长度的收益明显变大；MUSE 发现表征质量对 ESU 的影响远大于 GSU；IAT 用更丰富的实例 token 替代稀疏手工特征。

**开放问题**
- **缓存新鲜度对效果的影响**：SequenceO1 只报告了命中率（0.5 / 0.6）对 FLOPs 的影响；LLaTTE 的上游 embedding 由事件触发更新，staleness 对效果的敏感性都没有量化。
- **sketch 的 slot 利用率**：prototype-wise softmax 可能出现空 slot 或负载极不均衡，论文没有报告。
- **召回阶段长序列与 SID 的结合**：GEMs（三流 + SID 生成）和 GLASS（SID 前缀检索）刚开始探索，用 sid₁+sid₂ 做更细粒度的逐层检索、多级 sketch 与 SID 层级对齐等方向尚未验证。
- **上游迁移率的天花板**：LLaTTE 约 50% 的 transfer ratio 能否通过多 token 输出（而不是单个 2048 维向量）或与下游联合训练继续提升？ChronicleRec 的多 token 前缀和 SequenceO1 的端到端联合训练给出了两个可能方向。
