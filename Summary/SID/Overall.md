# Semantic ID (SID) 专题总结

本目录收录**以 SID 本身为研究对象**的论文：SID 怎么构造、怎么处理碰撞、怎么解码、怎么作为特征使用、上线后怎么更新。只把 SID 当作现成词表来用的工作（如 OneRanker / UniSGR / UniR² 的召回-排序统一、TokenMinds 的用户预训练）仍留在 `../Ranking/`；把 SID 层级用于长序列检索的 GLASS 放在 [`../Long-Sequence/`](../Long-Sequence/summary_glass_coarse_to_fine_long_term_interest.md)。

## 收录论文

| 论文 | 机构 | 场景 | SID 的角色 | 一句话贡献 | 总结 |
|---|---|---|---|---|---|
| Purely Semantic Indexing | 学术 | 序列推荐 / 商品搜索 / 文档检索 | 生成目标 | 放宽"最近质心"约束（ECM / RRS），不追加非语义 token 就能得到唯一 SID | [link](summary_purely_semantic_indexing.md) |
| CRID: Beyond Semantic IDs | 淘宝 | 搜索召回（GR） | 生成目标 | DocID = 语义簇前缀 + 簇内业务价值排名，构造上无碰撞，且支持增量更新 | [link](summary_crid_cluster_ranked_identifier.md) |
| TSGR (QP-SID + VRM) | 淘宝 | 搜索召回 + 预排序 | 生成目标 | 第三层按点击/查询条件排序（4 种并行排序），VRM 在生成模型内做价值重排 | [link](summary_value_aware_gr.md) |
| Don't Rewrite the SID Map | 淘宝 / 天猫 | 搜索召回（GR） | 生成目标 | 受控实验：重映射老商品 SID 会单调损伤效果，warm-start 也修不回来；应当冻结老 SID、只插入新品 | [link](summary_sid_topology_preservation_dynamic_update.md) |
| GEAR (BasisVQ) | 抖音广告 | 广告召回（GR） | 生成目标 | 正交基重参数化码本（BasisVQ）+ prefix-aware 仿射；tokenizer 在流式训练中端到端更新；多解码一步做桶内重排 | [link](summary_gear_basisvq_e2e_ad_retrieval.md) |
| BARGE | 腾讯 PCG | 生成式推荐（TIGER 式） | 生成目标 | 编码端 ICA 恢复 item 边界；解码端 HPR 路径重排 + DPD 正交双通道 SID，压制层级语义漂移 | [link 1](summary_barge_structural_gap_gr.md) / [link 2](summary_barge_semantic_drift_dual_path_decoding.md)（同一篇论文的两份总结） |
| Prefix-ngram SemID | Meta Ads | 判别式精排 | sparse / 序列特征 | 每层前缀各查一行 embedding 后做 sum-pool，解决哈希碰撞、长尾和 ID 漂移 | [link](summary_semantic_id_prefix_ngram_stability.md) |
| SiDec: Dual-purpose SID | YouTube | 精排 + 召回 | 输入特征 | 同一组 SID 同时走"可学习协同 embedding"与"冻结码本 + 解码器重建内容向量"两路，替代 dense 内容特征的 IO | [link](summary_sidec_dual_semantic_id.md) |
| FOUNDv2 | 蚂蚁 | 用户表征（支付 / 广告） | 用户侧 token | 多源用户行为经 Qwen3 Embedding 后，用 shared + source-specific 的 MRQ-VAE 离散化为可增量追加的用户 token | [link](summary_foundv2_user_tokenizer.md) |

## 一、SID 生命周期：一张图看各篇工作的位置

```text
 内容/协同 embedding ──▶ ① Tokenizer ──▶ ② 唯一性/碰撞 ──▶ ③ 消费 ──────────────▶ ④ 更新
                        RQ-VAE/RQ-KMeans   追加随机 ID?      生成式：beam search     新品插入
                        GEAR, BARGE(OSQ),  PSI, CRID, TSGR,  BARGE, GEAR, CRID, TSGR 老品重映射?
                        FOUNDv2(MRQ)       HC, GEAR rerank   判别式：prefix-ngram,   SID Map, CRID,
                                                             SiDec, FOUNDv2          GEAR, prefix-ngram
```

## 二、按问题拆解

### 1. Tokenizer：码本怎么学

| 工作 | 做法 | 解决的问题 | 关键证据 |
|---|---|---|---|
| GEAR | 码本右乘可学习正交矩阵 $\hat W$（Newton-Schulz 正交化），所有码随活跃码一起刚性旋转；第 $l$ 层码本按前缀查表做逐元素仿射 | 流式分布漂移下的死码（表征坍塌）；vanilla RQ 各层不知道前缀 | 相比 vanilla RQ+EMA：利用率 4.22% → 8.83%，碰撞率 84.2% → 52.3%；tokenizer 换回 vanilla RQ 后，线上 ADSS 增益从 +0.446% 降到 +0.288% |
| BARGE (OSQ-VAE) | Householder 正交旋转后切成两个正交子空间，各自残差量化，得到两套 SID | 单一量化轴锁死"轴外语义" | 两通道 top-K 的 Jaccard 只有 0.17~0.18；15%~24% 的命中只由一个通道贡献 |
| FOUNDv2 (MRQ-VAE) | 前几层用跨来源共享码本，后几层用来源专属码本 | 异构多源用户数据缺少统一离散空间 | 表示存储 240 GB → 8.2 GB，训练约 7h → 2h |

**共性观察**：
- **递减码本优于均匀码本**：GEAR (2048, 512, 128) 优于 (512, 512, 512)；BARGE (512, 256, 128, 64) 优于均匀 4 层，而且首层不能太小。两篇都把这一点解释为残差能量逐层衰减。
- **"正交"被两篇独立用上，但目的不同**：GEAR 用正交保持码本拓扑、让死码能被"转回"数据流形；BARGE 用正交保证两个通道互补，从而压低 $\kappa=\Pr[E^{(B)}\mid E^{(A)}]$。
- **SID 不一定是 item 侧的**：FOUNDv2 把同一套 RQ 机制用于用户行为切片，输出一串随时间 append 的用户 code。它与"把长期兴趣压缩成可缓存用户表示"的方向（参见 [Long-Sequence 专题](../Long-Sequence/Overall.md)）直接相关。

### 2. 碰撞：去掉、利用，还是交给下游

| 策略 | 代表 | 做法 | 评价 |
|---|---|---|---|
| 追加随机冲突 ID | TIGER 等基线 | `(c1, c2, c3, random)` | 多篇明确反对：PSI 认为它扩大搜索空间、损害冷启动；BARGE 发现最后一层随机 bit 会破坏 ICA/HPR 依赖的层级信号，把这个 slot 换成语义层效果相当 |
| 重新分配，保证唯一 | Purely Semantic Indexing | ECM 全局枚举候选组合；RRS 逐层回溯 | 唯一性比"严格最近质心"更重要；冷启动收益最明显 |
| 最后一层换成业务排名 | CRID、TSGR | 前缀 = 语义簇，末位 = 簇内按转化/点击排序的名次 | 构造上无碰撞，排名还带有统计先验 |
| 可控碰撞 | SID Map 论文 (HC) | 簇内 Top-100 独占 token，长尾共享 1024 个桶 | 与完全无碰撞的 FS 相比各指标略好，**碰撞不是瓶颈** |
| 接受高碰撞 + 桶内重排 | GEAR | 碰撞率约 52%，在完整 SID 后多解码一步得到序列条件化的用户向量，与物品向量点积 | 去掉重排、改为随机截断后，ADSS 从 +0.446% 降到 +0.358% |

可以看到一个趋势：从"消灭碰撞"转向"**让碰撞有意义**"（按价值排名、可控分桶、桶内重排）。判别式场景的 prefix-ngram 也是同一思路：随机哈希碰撞变成语义碰撞，粗粒度行由大量 item 共享。

### 3. 把业务价值写进 SID（淘宝搜索一脉）

CRID → TSGR → SID Map 是同一条技术线（CRID 与 TSGR 作者重叠，SID Map 直接沿用 CRID 的价值层）：
- **CRID**：首次把簇内转化率排名作为最后一级 token。末位不是"按价值排序"，而是 $P(r\mid c_s, q, h)$，统计先验和个性化偏好的混合。HR@1000 从 FORGE 的 72.50% 提升到 82.25%，线上 GMV +1.06%。
- **TSGR**：在 CRID 上加入**查询条件排序**（每个簇 1 种默认排序 + 3 种按代表词项的排序），并用 VRM 在生成骨干内部做 PV×CTR 重排，让一个模型同时完成召回和预排序。HR@1000 比 FORGE 高 9.16%，线上 GMV +1.64%。
- **SID Map**：价值层天然随时间变化，于是研究"价值刷新后该不该重映射"（见下文第 6 节）。

### 4. 解码：层级 SID 的误差级联

层级 SID 的核心难点是**前缀一错，整棵子树不可达**。BARGE 在 TIGER 上量化了这一点：$c_3$ 在前缀正确时准确率 77.0%，前缀错误时只有 0.6%（128×）。各篇的应对：

| 手段 | 代表 | 机制 |
|---|---|---|
| 路径级重排 | BARGE HPR | 每层用 decoder 初始状态 $h_0$ 和累积路径 embedding 做双塔余弦打分，与 NTP 概率融合，beam 宽度不变 |
| 多通道 OR 融合 | BARGE DPD | 两套正交 SID 各自 beam search，在 item 空间做 soft-OR（LSE） |
| 动态 beam | CRID (100→400→1500)、TSGR (400→400→10000)、GEAR (256→256→512) | 越往后分布越分散，beam 越大；末层去掉 trie 约束以省内存 |
| 后缀引入检索上下文 | GLASS（见 Long-Sequence） | 用已生成的 sid₁ 在长期历史中 hard search，缓解第 2/3 层的 rank degradation |

**评估警示**（SID Map 论文）：宽 beam 会掩盖问题。映射刷新后 HR@1 下降 42%~62%，HR@1000 只下降 8%~11%。评估 SID 方案的改动时要看 HR@1 / MRR，不能只看 HR@1000。

### 5. SID 作为判别式模型的特征

| 工作 | 查表方式 | 替代对象 | 主要收益 |
|---|---|---|---|
| Prefix-ngram | $e(x)=e[c_1]+e[c_1c_2]+e[c_1c_2c_3]+\dots$，每个前缀是独立的一行 | 随机哈希的 item ID | 同为 2048×3，prefix-3gram 的 NE 增益约是 trigram 的 5 倍（−0.141% vs −0.028%）；新 item NE −0.41%；同素材副本预测差异（A/A variance）−43% |
| SiDec | 协同流：Unigram / Bigram / Nested N-gram / SPM；内容流：冻结码本求和 → 轻量解码器 $f_\theta$ | dense 内容 embedding 的日志 / feature store IO | 直接灌 dense 训练速度降 28.2%；换成 SID 重建后，速度仍比 dense 快 20.4%，且 HR@100 更高 |
| FOUNDv2 | 用户 code lookup 后接 pooling / 序列模型 | 多源用户连续表征 | 存储下降约 30×，支持每日只编码新增行为 |

两点共识：
- **查表粒度比 tokenizer 本身更关键**：DLRM 的 sparse 模块不做上下文建模，前缀条件必须写进 ID（prefix-ngram 优于 all-bigrams / trigram）。SiDec 的 Nested N-gram 也是同一思想。
- **RQ 码本向量与协同 embedding 是两套空间**：prefix-ngram 完全不用 RQ 码本向量，只继承 trie 结构；SiDec 两套都用，但分别冻结（内容）和学习（协同）。码本是在"重建内容"的目标下学出来的，不适合直接承载协同信号。

### 6. 时间稳定性：SID 上线后怎么更新

这是近期最受关注、也最有分歧的一点：

| 工作 | 立场 | 证据 |
|---|---|---|
| Prefix-ngram | SID 提供**稳定的 ID 空间**：广告下线后，新广告共享前缀，粗粒度行可直接复用 | 对分布漂移的敏感度与无碰撞的 IE 相当（0.0073 vs RH 0.0083） |
| SID Map | **不要改模型已学过的 SID**：冻结老品、只插入新品 | 重映射比例 γ 与退化严格单调；只改 1.65% 的老商品，HR@1000 也从 0.7043 降到 0.6769；Full-Re + warm-start 甚至不如不重训的 Frozen |
| CRID | 支持增量更新：新品按距离归入最近簇，按日级统计**全量重排序** | 全重排序 HR@1000 80.05% > 无更新 77.52% > 仅插入 70.44% |
| GEAR | tokenizer 在线持续更新，SID 随之漂移，靠 warm-up 和分钟级索引刷新兜底 | 只有线上 A/B，没有专门讨论 SID 一致性 |

**CRID 与 SID Map 的结论看似矛盾**：一个说全重排序最好，一个说重映射有害。两者的评估协议不同。CRID 的 "仅插入" 会让簇内排名与新统计不一致，而它的重排序是否伴随模型重训，总结中没有写明；SID Map 则把映射策略和训练策略作为两个正交变量严格控制。在拿到 CRID 更新实验的训练协议之前，更稳妥的做法是采用 SID Map 的原则：**插入新品、冻结老品、周期性 warm-start**。GEAR 的"在线更新 SID"能工作，可能是因为生成器与 tokenizer 在同一流式训练中联合更新，代价是需要配套的近线索引系统。

## 三、总结与开放问题

**共识**
1. **SID 的价值主要在长尾、冷启动和 ID 漂移**：prefix-ngram、PSI、GEAR、SiDec 的收益都集中在新 item / 长尾上，head 基本持平。
2. **层级结构必须被显式利用**：prefix-ngram 把它写进 ID；BARGE / GLASS 利用前缀做重排或检索；CRID / TSGR 把末层赋予不同语义。
3. **工业 GR 中 SID 末层正在"去语义化"**：末层越来越多地编码业务价值、排名或桶内序号，而不是继续细分语义。
4. **SID 是 IO 友好的表示**：SiDec 和 FOUNDv2 说明，离散 token 能把"磁盘 / 网络绑定的 dense 传输"转成"加速器内的查表计算"。

**开放问题**
- **何时必须全量重建**：冻结老 SID 会让价值排名过时、头部 token 被越来越多新品共享。SID Map 没有回答何时重建、重建后如何迁移。
- **在线更新 tokenizer 与 SID 稳定性如何兼得**：GEAR 的流式 tokenizer 与 SID Map 的拓扑保持原则如何调和，目前缺少受控对比。
- **碰撞到底多少合适**：HC 的可控碰撞略优于无碰撞，GEAR 52% 的碰撞率依赖重排头。碰撞率、桶大小与下游重排能力之间的关系没有系统研究。
- **用户侧 SID 的内在评估**：FOUNDv2 没有报告重建误差、码本困惑度或未来行为检索准确率，用户 token 的质量目前只能靠下游 probe 间接衡量。
