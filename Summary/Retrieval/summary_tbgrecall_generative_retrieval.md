# TBGRecall: 面向电商推荐场景的生成式召回模型

**论文信息**
- **标题**：TBGRecall: A Generative Retrieval Model for E-commerce Recommendation Scenarios
- **作者**：Zida Liang, Changfa Wu, Dunxian Huang 等（上海交通大学 & 阿里巴巴）
- **arXiv**：[2508.11977](https://arxiv.org/abs/2508.11977)
- **发表时间**：2025年8月（v1），2025年11月修订（v2）

---

## 一、论文动机与问题陈述

推荐系统的**召回阶段（Retrieval Stage）**负责从海量商品库中快速筛选出数千个候选集，供后续精排模块使用。该过程本质上类似一个生成任务，因此近年来生成式模型被引入推荐领域。

然而，现有生成式推荐方法存在一个根本性矛盾：

> **自回归生成（Next Token Prediction, NTP）强制在同一 Session 的 item 之间引入顺序依赖关系，而在真实推荐系统中，同一 Session 内返回的商品是同时无序展示给用户的，彼此之间并不存在因果关系。**

具体来说：
- 用户的行为反馈（点击/购买）发生在商品曝光之后，因此因果关系仅存在于跨 Session 之间；
- 同一 Session 内各 item 应当是相互独立的，不应互相影响；
- 传统 token 级别自回归生成在推断时需要逐个生成 item，无法高效满足召回阶段的需求。

---

## 二、核心方法与贡献

### 2.1 整体框架：TBGRecall

**TBGRecall（TaoBao Generative Recall）** 是一个以 Session 为单位的自回归生成模型，基于 Meta HSTU 架构构建。

**样本构建（Sample Construction）**：
- 将用户历史行为序列按照**请求 Session** 进行划分，构建多 Session 序列；
- 每个 Session 以一个 **Context Token（$c^{(k)}$）** 开头，后跟多个 **Item Token**；
- 模型输入序列形如：$[c^{(1)}, i^{(1)}_1, i^{(1)}_2, c^{(2)}, \ldots, c^{(K)}, i^{(K)}_1, i^{(K)}_2, c^{(K+1)}]$；
- 每个 token 的向量由 item ID embedding、用户行为 embedding、商品属性（side info）embedding、场景 embedding 四部分相加得到。

### 2.2 Next Session Prediction（NSP）

NSP 是本文的**核心创新**，实现了 Session 级别的自回归推断，而非 token 级别：

- **Session Mask**：在标准因果 Mask 基础上，额外屏蔽同一 Session 内各 Item 之间的注意力，使 item 彼此不可见，消除无效的顺序依赖；
- **Session-wise RoPE（sw-RoPE）**：同一 Session 内所有 token 共享相同的位置编码，避免位置 embedding 引入顺序信息；
- **推断方式**：在推断时，将新 Session 的 Context Token $c^{(K+1)}$ 追加到序列末尾，输出对应位置的 embedding，通过 **ANN 搜索**检索商品池中的 Top-K 候选集；
- 无需逐步迭代生成，推断效率高，且与现有召回基础设施（ANN）天然适配。

**NSP 的 Attention Mask 示意：**

```
            c1  i1_1 i1_2  c2  i2_1  c3
c1        [ ✓                            ]
i1_1      [ ✓   ✗                        ]   ← i1_1 只能看到 c1
i1_2      [ ✓   ✗    ✗                   ]   ← i1_2 也只能看到 c1
c2        [ ✓   ✓    ✓    ✓              ]   ← c2 能看到 Session 1 的所有
i2_1      [ ✓   ✓    ✓    ✓   ✗         ]   ← i2_1 只能看到 c2 之前和 c2 本身
c3        [ ✓   ✓    ✓    ✓   ✓    ✓    ]
```

### 2.3 多会话预测（Multi-Session Prediction, MSP）

类比 LLM 中的 Multi-Token Prediction（MTP），NSP 中每个 Context Token 被要求预测未来**多个 Session** 的结果（而非仅下一个 Session），从而：
- 引入更长时序训练信号；
- 让模型更好地捕捉用户远距离的行为依赖关系；
- 消融实验表明 MSP 是单个贡献最大的组件。

### 2.4 Token-Specific Network（TSN）

由于 Context Token 和 Item Token 的语义和分布存在较大差异，引入 **TSN**：
- 在 Embedding 层和第一个 Transformer Block 处，分别对两类 token 使用独立的线性变换层；
- 推断阶段无额外计算开销（仅将共享投影替换为两个独立投影）。

### 2.5 整体损失函数

损失函数由三部分构成（Cascade Loss 设计）：

$$\mathcal{L}_{NSP} = \sum_{s \in \mathcal{S}}\frac{1}{N_s}\left(\mathcal{L}_{NCE}^{(s)} + \mathcal{L}_{click}^{(s)} + \mathcal{L}_{pay}^{(s)}\right)$$

- **$\mathcal{L}_{NCE}$**：对比学习损失，将 Context Token 向曝光商品（正样本）拉近，向随机负样本推远；
- **$\mathcal{L}_{click}$**：在曝光商品中，将点击商品作为正样本，进一步强化高价值交互；
- **$\mathcal{L}_{pay}$**：在点击商品中，将购买商品作为正样本，捕捉最高价值交互；
- **Multi-Scene Normalization**：按场景分别计算损失后归一化求和，避免场景分布不均导致的性能退化。

三级优先级的对比学习信号：`曝光 > 随机负样本`，`点击 > 曝光未点击`，`购买 > 点击未购买`。

### 2.6 随机部分增量训练（Partial Incremental Training, PIT）

针对大模型增量训练时间长（完整训练需 5 天）、延迟部署的问题，提出 PIT：

- 将所有用户随机分入 10 个桶（Bucket）；
- 每次增量训练只使用**一个桶**中最近 10 天的数据；
- 仅用 1/10 数据量，同等 GPU 资源（128 卡）下训练时间缩短至 11 小时，并能使用最新数据；
- 由于轮流覆盖所有桶，最终不会有数据损失，且每桶数据量足够大，分布代表性充分。

| 方法 | 训练时长 | GPU 数量 | 使用最新数据 | HR@4000 |
|------|---------|---------|------------|---------|
| Normal | 5天 | 128 | ❌ | 23.76% |
| Incre（理想） | 16小时 | 1280 | ✅ | 29.76% |
| SL（随机长度） | 48小时 | 128 | ❌ | 26.50% |
| **PIT（本文）** | **11小时** | **128** | **✅** | **29.45%** |

---

## 三、工程实现

### 训练框架
基于 TorchRec 构建，支持：
- **分布式负采样**：将亿级商品目录分片到各节点，实现低延迟采样；
- **异步数据加载**：CPU DataLoader 利用流水线并行，GPU 利用率 >90%；
- **Sharded Embedding + FSDP**：稀疏参数行级分片，稠密参数跨设备切分；
- 日常重训练、容错检查点、指标监控等端到端能力。

### 在线服务（Nearline 架构）
- **近线化推断**：用户实时行为触发异步生成用户 embedding，并通过 ANN 搜索预计算 Top-K 候选；
- 结果缓存为 `<userid, TopK Items>` 键值对，在线请求直接读取，避免实时计算；
- 定期更新 interest vector，平衡新鲜度与计算开销。

---

## 四、实验结果

### 数据集
| 数据集 | 交互数 | 用户数 | 商品数 |
|--------|--------|--------|--------|
| RecFlow（快手） | 3025 万 | 3.1 万 | 324 万 |
| TaoBao（工业级） | 1.2 万亿 | 5 亿 | 5 亿 |

### 基线对比（HR@K，越高越好）

在 TaoBao 工业数据集上 TBGRecall vs 主要基线：

| 模型 | HR@20 | HR@100 | HR@500 | HR@4000 |
|------|-------|--------|--------|---------|
| HSTU | 0.54% | 1.78% | 4.88% | 13.11% |
| Online (DT，日更生产模型) | 1.26% | 4.28% | 11.03% | 26.45% |
| **TBGRecall** | **1.53%** | **4.65%** | **11.81%** | **29.45%** |

在 RecFlow 数据集上同样全面领先所有基线。

### 消融实验（TaoBao 数据集）

| 模型变体 | HR@20 | HR@500 | HR@4000 |
|----------|-------|--------|---------|
| TBGRecall（完整） | **1.53%** | **11.81%** | **29.45%** |
| w/o TSN | 1.51% | 11.67% | 28.88% |
| w/o MSP | 1.35% | 10.59% | 27.28% |
| w/o MoE | 1.46% | 11.41% | 28.47% |
| w/o sw-RoPE | 1.32% | 10.67% | 27.36% |

MSP 对性能提升贡献最大（w/o MSP 损失约 8%）。

### Scaling Law

实验通过逐步增大隐层维度（128 → 256 → 512 → 1024 → MoE 架构），观察到：
- **模型性能与参数量的对数呈线性关系**（清晰的 Scaling Law）；
- 证明了大规模参数化生成模型在工业推荐系统中部署的可行性。

### 在线 A/B 测试

在淘宝首页「猜你喜欢」场景（每日亿级曝光），5% 流量运行 7 天：

| 指标 | 结果 |
|------|------|
| 曝光占比（PVR） | 23.94% |
| 成交笔数 | **+0.60%** |
| 成交金额 | **+2.16%** |

---

## 五、主要结论

1. **NSP 范式**解决了生成式推荐中自回归模型在召回场景的根本矛盾（Session 内 item 顺序依赖问题），是本文最核心的贡献；
2. **Session Mask + sw-RoPE** 是 NSP 的关键实现，确保 Session 内 item 独立性；
3. **MSP**（Multi-Session Prediction）引入长距离训练信号，是性能提升最显著的辅助模块；
4. **PIT**（Partial Incremental Training）以少量 GPU 资源实现高效近实时训练，具有很强的工业可行性；
5. 稀疏 ID 模型首次在工业规模推荐召回任务中展示出清晰的 **Scaling Law** 趋势；
6. 线上 A/B 验证显示成交金额提升 **+2.16%**，具有显著业务价值。

---

## 六、批判性讨论

### TBGRecall 是否真的是"生成式"推荐？

**本文并未真正生成出 retrieval item**，推断流程本质上是：

```
序列 → Transformer → 用户 embedding → ANN 检索
```

与经典的 DSSM / 双塔 / YoutubeDNN 的推断流程形式相同，只是用户侧 encoder 换成了更强的 Transformer。与"正统"生成式推荐方法的对比：

| 方法 | 是否生成 item | 推断方式 |
|------|-------------|---------|
| TIGER（Google） | ✅ 是 | Beam search 逐步生成 item 的语义 ID token 序列 |
| OneRec（Kuaishou） | ✅ 是 | Decoder 直接生成 item，统一召回和排序 |
| **TBGRecall（本文）** | ❌ 否 | Context Token embedding → ANN 检索 |

### 与双塔、HSTU 的本质区别

| 对比维度 | 双塔 / YoutubeDNN | HSTU（NTP） | TBGRecall（NSP） |
|----------|------------------|------------|-----------------|
| 用户侧建模 | MLP + pooling | Transformer（token-wise AR） | Transformer（session-wise AR） |
| 训练信号 | user-item 交互标签 | 下一个 item 预测 | Session 内曝光/点击/购买的对比学习 |
| Session 内 item 关系 | 忽略顺序 | 强制顺序依赖（错误假设） | 显式消除顺序依赖 |
| 推断 | ANN | 需逐步生成或取最后 token ANN | Context Token → ANN |

**TBGRecall vs HSTU 的核心改进**：HSTU 用 NTP 训练，等于假设"Session 内的商品有顺序因果关系"——这个假设在召回场景下是错的。TBGRecall 用 Session Mask 把这个假设去掉，同时换成更符合召回任务的对比损失。

### 综合评价

本文真正有价值的贡献是：
1. **发现并修正了 NTP 在召回场景下的错误假设**（核心洞察）；
2. **Session Mask + sw-RoPE** 的具体实现机制；
3. **Cascade Loss** 的三级对比学习设计；
4. **PIT** 的工业增量训练方案。

但将其冠以 "Generative Retrieval" 的名号，在学术定义上有一定过度包装的成分，其本质更准确的描述是"**用 Decoder-only Transformer 做表示学习的召回模型，通过 Session 级建模修正了 NTP 范式的不适配问题**"。

---

## 七、潜在应用与启发

- **召回阶段范式创新**：NSP 将自回归序列建模与 ANN 召回基础设施完美融合，可直接迁移到其他电商或内容推荐平台的召回模块；
- **Cascade Loss 设计**：NCE + Click + Pay 的层次化对比损失设计，对于需要区分不同行为强度的推荐场景具有参考价值；
- **大模型增量训练效率**：PIT 的分桶随机增量策略为大规模推荐模型的在线部署提供了实用解决方案；
- **近线化推断架构**：将重量级模型推断与在线服务解耦的 Nearline 架构，对低延迟要求的工业系统有广泛借鉴意义；
- **Scaling Law 的召回应用**：首次证明稀疏 ID 推荐模型遵循 Scaling Law，为推荐系统未来持续扩参数量提供了理论支撑。
