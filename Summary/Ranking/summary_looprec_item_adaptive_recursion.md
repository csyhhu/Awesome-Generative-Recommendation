# Think Harder Where It Matters: Item-Level Adaptive Recursion for Sequential Recommendation（LoopRec）

- **arXiv**: [2610.30128](https://arxiv.org/abs/2610.30128)
- **作者**: Zhanshuo Cao\*, Jiakai Tang\*, Yuqi Zhou, Sunhao Dai；通讯作者 Jun Xu（人大高瓴）
- **代码**: https://github.com/Elpmis117/LoopRec
- **领域**: 序列推荐 / Looped Transformer / 自适应深度（Adaptive Depth, Early Exit）/ Mixture-of-Recursions
- **一句话**: 质疑"历史序列中每个 item 都用同样深度计算"的默认设计，先用两个诊断实验证明 item 的计算需求高度不均匀、固定深度在 95.4% 的序列上并非最优；然后提出 **LoopRec**——**权重共享的递归 block + 每个递归边界上的逐 item 二元 exit/continue 门**，让深度成为每个 item 自己学出来的属性。在 4 类 backbone（SASRec / BSARec / NextItNet / LRURec）× 3 个 Amazon 数据集上 48/48 个指标全部提升（44 个显著）。

> 与同组的 [LoopCTR](summary_loopctr_loop_scaling.md) 对照：LoopCTR 是**样本级**的 loop scaling + 过程监督，推理可零 loop，并且在 Oracle 分析里发现"逐样本选深度"还有很大余量；LoopRec 把自适应深度做到了**序列内 item 级**，并且把它真正学出来了。

---

## 1. 动机：两个诊断实验（3 层 SASRec，Amazon Office）

### Probe 1：Item 级饱和深度（Saturation Depth）
**定义**：对训练好的 $L$ 层模型，item $i$ 的饱和深度 $\ell_i^\star$ 是最早的层 $\ell$，使得**只冻结该 item 在 $\ell$ 层之后的表示**（其他位置照常前向）时，最终 Top-$K$ 推荐集合（$K=30$）**完全不变**：

$$\ell_i^\star=\min\{\ell : \mathcal{R}_{\text{trunc}}^{(i,\ell)}=\mathcal{R}_{\text{full}}\}$$

**结果**：
- **76.14%** 的 item 在第 1 层就饱和，**93.28%** 的 $\ell^\star\le2$；
- 但 **6.72%** 必须跑满 3 层；
- 浅饱和和深饱和的 item **经常同时出现在同一条序列里** → 任何单一固定深度都无法同时照顾两类 item。

### Probe 2：受限 Tail-5 Oracle
Probe 1 只说明"不同 item 需求不同"，Probe 2 要回答"固定深度是否真的损害了推荐质量"。

**设置**：
- 冻结模型参数，只改**用户历史序列中最近 5 个 item（输入位置 $T-4,\dots,T$）** 的更新深度——不是候选 item，候选集仍是全量 item、打分方式不变：每个尾部位置选 $d\in\{1,2,3\}$，第 $d$ 层之后不再更新（与 Probe 1 的冻结方式相同：只停止该位置的更新，其他位置照常前向）；更早的位置照常跑满 3 层；
- 论文里的"candidate"指的是**深度分配方案**：共 $3^5=243$ 种组合，原始固定分配 $(3,3,3,3,3)$ 也在其中；论文还声明**排除尾部层计算量超过 baseline 的分配**；
- **在 causal mask 下实际影响的范围**：更早的位置看不到尾部，所以它们的表示完全不受影响；被改变的只有尾部 5 个位置自身。SASRec 用最后一个位置 $T$ 的输出作为用户表示去给全量 item 打分，所以这个实验本质上是在搜索"最后一个位置的表示怎么算"：$d_T$ 直接决定用户向量停在第几层，$d_{T-4..T-1}$ 则通过 attention 间接影响它；
- 对每条测试序列，Oracle 用 ground-truth target 在 243 个候选里选出让目标排名最好的那个；
- **相对尾部计算量** = 5 个尾部 item 的层计算次数之和 / 15。例如 40% = 6/15，对应深度和为 6（如 1,1,1,1,2）。
- 只看尾部 5 个：一是搜索空间随长度指数增长（全序列 50 个位置是 $3^{50}$）；二是最近的 item 对下一个 item 预测影响最大（4.3 节也显示最近四分位的饱和深度最高）。

**结果**：
- 相对固定深度：NDCG@20 **+7.1%**、MRR@20 **+6.5%**、HR@20 **+8.8%**（HR 提升说明有一部分 target 被推进了 Top-20，而不只是名次微调）；
- 固定深度只在 **4.6%** 的序列上是 Oracle 的最优选择；其余 **95.4%** 都是非均匀分配更好；
- 样例：rank 24→13（相对计算 40%）、7→4（66.7%）、7→5（53.3%）、12→10（60%）、19→18（60%）。

**解读时要注意**：
- **计算约束实际上不起作用**：每个 item 的深度上限就是 3，任何组合的尾部计算都 ≤ 15 = baseline，"排除计算量更大的分配"什么也没排除；反过来，"Oracle 分配用更少计算"也是必然的——只要不是全 3，计算就更少。所以"更好且更省"里，"更省"是构造决定的，有信息量的只是"更好"；
- **Oracle 选择本身有乐观偏差**：用测试 target 在 243 个扰动里挑最优，即使扰动没有任何结构，max 也几乎总能超过原值。所以 95.4% 主要说明"存在更好的深度分配"，**并不说明这种分配可以在不看答案的情况下找到**。论文没有做对照（例如等量随机扰动 + Oracle 选择，或在验证集上选分配再迁移到测试集），这是 Probe 2 证据力的主要缺口；
- **平手如何计数没说明**：若有多个候选与固定分配同名次，"固定深度最优仅 4.6%"取决于平手规则，论文没有交代；
- **分布外的前向**：模型训练时从未见过"部分位置在第 1 层就冻结"的混合状态，Oracle 仍能找到更好的分配，说明表示对深度扰动是有一定冗余的；
- **与 LoopRec 的关系**：Oracle 是看答案的上界（+7.1% NDCG@20），LoopRec 是不看答案、靠路由器学出来的（SASRec-Office 上 +2.2% NDCG@20），量级关系合理；第 4.3 节再用位置/频率的一致性去论证路由器学到的方向与诊断一致。

**结论**：统一深度**既在已收敛的 item 上浪费算力，又让关键的少数 item 算得不够** → 应该把深度当作**逐 item 的计算分配变量**（但 Probe 2 证明的是"存在余量"，而不是"余量可被学到"，后者要靠主实验来说明）。

---

## 2. 两个技术挑战

1. **跨深度的表示对齐**：不同 item 在不同深度 exit，但它们的输出要在同一个特征空间里，后续计算和打分才可比；
2. **渐进式深度决定**：item 是否"想清楚了"要在它不断吸收上下文的过程中才能看出来，不能一开始就定死（这正是它与 MoR token-choice router 的区别）。

对应方案：**权重共享递归**解决 (1)（所有深度都过同一组参数，天然处在同一空间）；**每个递归边界上的二元门**解决 (2)。

---

## 3. 方法：LoopRec

记号 **`LxRy`**：共享 block $\mathcal{B}$ 含 $x$ 层，最多递归 $y$ 轮。在第 $r$ 轮停止的位置实际经过 $r\cdot x$ 层编码。

**参数共享的粒度是"轮"，不是"层"**：block 内部的 $x$ 层各有各的参数（代码里是 `deepcopy` 出 `num_layers` 个独立的层），整个 block 在每一轮被原样复用。例如 `L2R3` 的有效深度是 6 层，但只有 2 层的参数（第 1、3、5 层共享一套，第 2、4、6 层共享另一套）；`L1R4` 才是真正的"每层参数都一样"。退出门（router + 轮次 embedding $\boldsymbol\rho_r$）也在所有边界上共享，靠 $\boldsymbol\rho_r$ 区分当前是第几轮。官方 release 的 SASRec 配置是 `L2R3`。

### 3.1 Initial Context Encoding
$$\mathbf{H}^{(1)}=\mathcal{B}(\mathbf{X}),\quad \mathbf{x}_t=\phi(\mathbf{e}_{v_t}+\mathbf{p}_t)$$
第一轮所有位置都跑，给后续路由决策提供上下文化后的状态。

### 3.2 Depth-Routed Recursion
在每个递归边界 $r\in\{1,\dots,R-1\}$，对每个仍活跃的位置 $i$：

$$s_i^{(r)}=g(\mathbf{h}_i^{(r)}+\boldsymbol{\rho}_r),\qquad q_{i,\text{exit}}^{(r)}=1-\sigma(s_i^{(r)})$$

- $g$：轻量 router head；$\boldsymbol{\rho}_r$：可学习的递归轮次 embedding。代码默认 $g$ 是两层 MLP（$d\to2d\to1$，GELU，无 bias），也可以选单层线性。实现上 logits 是 $[0,\ s]$ 两类做 softmax（exit 类的 logit 固定为 0），所以 $q_{\text{exit}}=1-\sigma(s)$，$s$ 就是"继续"的 logit；
- $q_{\text{exit}}\ge0.5$（等价于 $s\le0$，代码里是对两类取 argmax）则停止，否则进入下一轮。这个硬决策本身不可导，训练中也**没有采样或噪声探索**，是确定性的 argmax；
- 活跃集合单调收缩：$\mathcal{A}^{(1)}\supseteq\mathcal{A}^{(2)}\supseteq\dots$，**只对活跃位置**重新施加共享 block：$\mathbf{H}^{(r+1)}_{\mathcal{A}}=\mathcal{B}(\mathbf{H}^{(r)}_{\mathcal{A}})$；
- 活跃子序列**保持原始时间顺序并使用 causal mask**；已停止的位置保留当前状态、不再更新。用 mask 的视角看，它等价于在完整长度的 attention 上，对已退出位置**同时 mask 掉行和列**：行被 mask，所以它不再更新；列被 mask，所以它也不能再被别的 item 看到。按官方代码（`src/utils/looprec.py`），两类 backbone 的实现不同：
  - **SASRec / BSARec（attention）**：`recursion_wise`，把活跃位置 gather 成短序列，用按原位置构造的 causal mask 跑共享 block，再 scatter 回去。**退出的 item 不再作为 K/V 被后续轮次看到**，活跃 item 只能 attend 到其他活跃 item；配置里写死 `exited_token_memory=active_only`，换别的值会直接报错。这样才真正省 FLOPs（attention 长度变短）；
  - **NextItNet / LRURec（卷积 / 线性递归）**：`kv_sharing`，因为卷积和递归没法把子序列"打包"，所以每轮都对**整条序列**重新跑 block，再用 `torch.where(active_mask, next_h, current_h)` 丢掉已退出位置的新输出。**退出的 item 冻结在退出时的状态，但仍作为输入被后面的位置读到**——只 mask 行，不 mask 列，接近 MoD 的做法。代价是这两个 backbone 上**完全不省计算**（论文的计算量/延迟只在 SASRec 上测）；
  - 两个诊断实验中的"冻结"（只停止该位置更新、其余位置照常前向）语义上也更接近后者，而不是 SASRec 版 LoopRec 的做法。
- 最后一轮强制所有剩余位置停止。

### 3.3 Depth-Specific Readout
$$\bar{\mathbf{z}}_i=\mathbf{x}_i+\omega_i\,\mathbf{h}_i^{(d_i)}$$

前向是硬路由（hard depth $d_i$），但读出权重 $\omega_i$ 保留了对 router 概率的可微性，让推荐损失能更新 router：

$$c_i^{(1)}=1,\quad c_i^{(r+1)}=c_i^{(r)}(1-q^{(r)}_{i,\text{exit}}),\qquad
\omega_i=\begin{cases}c_i^{(d_i)}q^{(d_i)}_{i,\text{exit}} & d_i<R\\ c_i^{(R)} & d_i=R\end{cases}$$

即 $\omega_i$ 是"沿实际路径到达 $d_i$ 并在此 exit 的概率质量"（类似 ACT/PonderNet 的 halting 概率，但只取被选中的那一项）。

**门是怎么从 $\mathcal{L}_{\text{rec}}$ 学到东西的**：思路和 MoE 一样，把门概率乘到被选中的输出上，让梯度经由这个乘积流回门。以 $d_i<R$ 为例，$\omega_i=\prod_{j<d_i}(1-q^{(j)})\cdot q^{(d_i)}$：
- 如果在深度 $d_i$ 读出的 $\mathbf h^{(d_i)}$ 对预测有帮助，梯度要求 $\omega_i$ 变大 → $q^{(d_i)}$ 上升（在这一轮停得更坚决），$q^{(j<d_i)}$ 下降（前面几轮更倾向继续）→ **巩固当前深度**；
- 如果 $\mathbf h^{(d_i)}$ 没用甚至有害，梯度要求 $\omega_i$ 变小 → $q^{(d_i)}$ 下降，或者前面某轮的 $q^{(j)}$ 上升 → 门被**推离当前深度**，往更深或更浅移动都有可能，具体取决于各项梯度的相对大小；
- 强制退出（$d_i=R$）时 $\omega_i=\prod_j(1-q^{(j)})$，有用就让各轮都更倾向继续。

局限：梯度只告诉门"当前这条路径要不要加强"，**不会告诉它"换一个深度会怎样"**，因为没走的深度没有被计算，也就没有反事实信号。所以多样性要靠 $\mathcal{L}_{\text{bal}}$ 和初始化来保证。

### 3.4 训练目标
$$\mathcal{L}=\mathcal{L}_{\text{rec}}+\lambda_{\text{bal}}\mathcal{L}_{\text{bal}}+\lambda_z\mathcal{L}_z$$

- **$\mathcal{L}_{\text{rec}}$**：标准 next-item 交叉熵，是唯一的 item 预测监督，**不需要深度标签**；
- **$\mathcal{L}_{\text{bal}}$（防深度坍缩）**：Switch-Transformer 风格的负载均衡。沿实际路径定义 soft depth mass
  $$\alpha_{i,r}=\begin{cases}c_i^{(r)}q^{(r)}_{i,\text{exit}} & r<d_i\\ c_i^{(d_i)} & r=d_i\\ 0 & r>d_i\end{cases}$$
  $P_r=\frac1N\sum_i\alpha_{i,r}$（soft 使用率），$f_r=\frac1N\sum_i\mathbb{I}[d_i=r]$（hard 频率），$\mathcal{L}_{\text{bal}}=R\sum_r P_rf_r$；
- **$\mathcal{L}_z$（防门过早饱和）**：沿用 ST-MoE / MoR 的 router z-loss，对二元门 log-partition 惩罚：
  $$\mathcal{L}_z=\frac{1}{|\mathcal{R}_+|}\sum_{r\in\mathcal{R}_+}\frac{1}{|\mathcal{A}^{(r)}|}\sum_{i\in\mathcal{A}^{(r)}}\operatorname{softplus}(s_i^{(r)})^2$$
  $\lambda_z$ 全局固定为 $10^{-3}$。
  注意这里是**不对称**的：exit 类 logit 固定为 0，$\operatorname{softplus}(s)^2$ 在 $s\to-\infty$ 时趋于 0，在 $s>0$ 时近似 $s^2$ 增长。所以它只压制"继续"的 logit 变大，不阻止门饱和到"退出"一侧。和多专家 MoE 里对称的 z-loss 不同，它在稳定 logit 的同时**还带有一点"倾向提前退出"的计算惩罚**（$\lambda_z$ 很小，影响有限）。另外代码实现是对所有门决策直接求平均，而不是论文公式里先按轮平均再对轮平均，两者对各轮的加权略有不同；
- **$\mathcal{L}_{\text{bal}}$ 的梯度**：$f_r$ 是硬计数，不传梯度；梯度只经过 $\alpha$，作用是压低被过度使用的深度上的 soft 质量，把一部分 item 推向冷门深度；
- **Routing Warm-Up**：前 $W_{\text{router}}$ 个 epoch 所有位置强制跑满 $R$ 轮、$\omega_i=1$、关闭路由和正则，退化成固定最大深度的 looped 模型；之后再开启路由。$W_{\text{router}}\in\{5,8,10,12,16\}$，按验证集 NDCG@20 选（代码里把 epoch 数换算成 step 数）。warm-up 期间门完全不被调用、也没有梯度；router 初始化为 std 0.02 的小权重，所以开启路由时 $s\approx0$、$q_{\text{exit}}\approx0.5$，每个边界上大约一半 item 退出，初始深度分布近似几何分布（`R=4` 时约 50% / 25% / 12.5% / 12.5%），之后再由上面三个梯度来源调整。

### 3.5 与 MoR 的区别
- MoR token-choice：**一开始**就给 token 分配总递归深度；expert-choice：每步按预设容量做 top-$k$ 选择；
- LoopRec：**共享的二元停止门**，以 item 当前上下文化状态 + 轮次为条件，**每轮重新决策**，不需要逐步 top-$k$ 和容量设定。

---

## 4. 实验

### 4.1 设置
- **数据**：Amazon Reviews 2023 的 Software（13.4K 用户 / 9.7K item）、Office（11.5K / 28.3K）、Patio（14.5K / 37.2K，最稀疏，99.96%）；≥10 次交互，leave-one-out，max len 50；
- **指标**：全量排序 MRR@10/20、NDCG@10/20，5 个 seed 取均值，paired t-test；
- **Backbone**：SASRec（attention）、BSARec（frequency-aware attention）、NextItNet（因果卷积）、LRURec（线性递归）；
- SASRec 搜索 `LxRy` ∈ {L1R4, L2R2, L1R6, L2R3, L1R8, L2R4}；NextItNet ∈ {L3R2, L3R3, L6R2}。
- **各实验用的 backbone**：只有主结果表覆盖了 4 个 backbone；两个诊断实验（3 层 SASRec + Office）、深度分配分析（SASRec，主要是 Office，位置分析加了 Patio）、消融、启发式路由对比、延迟测量**全部只在 SASRec 上做**。所以"学到的路由对齐需求"这一结论只在 SASRec 上验证过，其他 3 个 backbone 只有精度数字。
- **任务范式**：4 个 backbone 都是**基于 item ID 的 next-item prediction**。训练时像语言模型一样，causal 编码，每个位置预测下一个 item（本文用全量 softmax 交叉熵）。推理时**只做一步**：编码历史后取最后一个位置的输出作为用户向量，与全部 item embedding 点积，排序取 Top-K。它不会把预测出的 item 拼回序列、逐个生成推荐列表，也没有 Semantic ID 或 beam search，和 TIGER 这类自回归生成 Semantic ID 的生成式推荐不同，更接近 HSTU 的检索用法。数据是 Amazon 评论序列（作为交互的代理），不是曝光日志，评估的是离线 leave-one-out 的下一个交互 item，不涉及曝光、点击这类排序目标。LoopRec 在带候选侧特征 / target attention 的工业排序模型上能否奏效，本文没有验证。

### 4.2 主结果（NDCG@20）

| Backbone | Software | Office | Patio |
|---|---|---|---|
| SASRec → +LoopRec | .0944 → .0998 (+5.68%) | .0582 → .0595 (+2.20%) | .0448 → .0458 (+2.19%) |
| BSARec → +LoopRec | .0987 → .0998 (+1.13%) | .0596 → .0602 (+0.97%) | .0453 → .0469 (+3.48%) |
| NextItNet → +LoopRec | .0782 → .0817 (+4.37%) | .0313 → .0354 (+13.18%) | .0192 → .0263 (+37.02%) |
| LRURec → +LoopRec | .0958 → .1001 (+4.51%) | .0573 → .0584 (+1.85%) | .0461 → .0470 (+1.87%) |

- **48/48 个对比全部正向，44 个显著**，相对提升 0.69%–49.07%；
- 对 attention、卷积、线性递归三类 backbone 都有效；
- **越稀疏收益越大**：各 backbone 平均 NDCG@20 提升 Software 3.92%、Office 4.55%、**Patio 11.14%**（不过 Patio 的均值很大程度上被 NextItNet 的 +37% 拉高，NextItNet 本身 baseline 较弱）。

### 4.3 学到的深度分配由什么决定？（SASRec, Office）
- **不坍缩**：所有 `LxRy` 配置都用到多个深度，浅层和深层都有可观占比；
- **位置**：越靠近当前（最近的 decile）平均递归轮数越高；
- **频率**：训练频率越低，深度越深——

| 训练频次 | Token 数 | 平均深度 | Depth 1 | Depth 4 |
|---|---|---|---|---|
| 1 | 505 | 2.160 | 50.3% | 29.1% |
| 20–49 | 4,123 | 1.818 | 69.2% | 24.1% |
| 100+ | 11,659 | 1.330 | 88.6% | 10.7% |

- **与饱和深度诊断一致**：在固定深度 SASRec 上，最近四分位的平均 $\ell^\star$=1.76（前三个四分位约 1.16–1.20），$\ell^\star=3$ 的样本几乎全集中在最近四分位（30.3%）；单次出现的 item 平均 $\ell^\star$=1.873，100+ 的为 1.197 → LoopRec 学到的路由是 **demand-aligned**（与需求对齐）的；
- **单条序列案例**：高频的 poly mailers / Avery labels 在 depth 1 exit；低频的 Aviditi box mailers / Pro Supply mailers 跑到 2–3 轮；但 PACKZON labels（freq 34）、TECKNET mouse pad（freq 86）跑满 4 轮，说明路由并不只由位置或频率单独决定，**上下文和类目语义也在起作用**；
- **对比启发式路由**（相同共享递归骨干，用位置或 log 频率规则映射深度）：Position NDCG@20 .0585、Frequency .0584、**LoopRec .0595**（分别高 1.64% / 1.88%），而 vanilla 为 .0582——启发式只有微小收益。

### 4.4 消融（SASRec, M@20 / N@20）

| 变体 | Software | Office | Patio |
|---|---|---|---|
| **LoopRec** | **.0659 / .0998** | **.0530 / .0595** | **.0416 / .0458** |
| Seq-level Loop（共享 block 跑 R 遍全序列，无 router） | .0634 / .0950 | .0525 / .0594 | .0390 / .0447 |
| Vanilla | .0629 / .0944 | .0513 / .0582 | .0399 / .0448 |
| w/o warm-up | .0615 / .0942 | .0520 / .0583 | .0401 / .0450 |
| w/o $\mathcal{L}_{\text{bal}}$ | .0655 / .0995 | .0507 / .0576 | .0385 / .0431 |
| w/o $\mathcal{L}_z$ | .0639 / .0973 | .0507 / .0574 | .0387 / .0432 |
| w/o 两者 | .0632 / .0963 | .0505 / .0574 | .0386 / .0431 |

- **权重共享递归本身贡献一部分收益**（Seq-level Loop 在 Software/Office 上优于 Vanilla），**item 级深度分配带来额外收益**；Office 上二者几乎打平（.0594 vs .0595），Patio 上 Seq-level Loop 甚至不如 Vanilla；
- 去掉 warm-up：平均 MRR@20 −4.03%、NDCG@20 −3.11%；
- 去掉两个正则：平均 MRR@20 −5.36%、NDCG@20 −4.25%；去掉任何一个正则在 Office/Patio 上的掉点都接近去掉两者，Software 上掉得少。

### 4.5 计算与延迟（Office，单卡 A6000，仅 encoder）

| 配置 | 对照 | 计算量 | 延迟 |
|---|---|---|---|
| L2R4 | SASRec-8L | 0.407× | 0.869×（91.2→79.3ms） |
| L2R3 | SASRec-6L | 0.680× | 0.907× |
| L2R2 | SASRec-4L | 0.755× | 0.995× |

- 计算量节省只**部分**转化为墙钟收益（动态的 gather/scatter 在 GPU 上不友好）；
- 注意对照是"**相同最大有效深度的固定深度 SASRec**"，而不是主表里的 vanilla（vanilla 层数是在 1–6 层里搜出来的），所以这里的"省计算"是相对 8L/6L/4L 深模型而言。

---

## 5. 局限与思考

- **规模很小**：三个 Amazon 子类目，每个约 1–1.5 万用户，max len 50，绝对指标很低（NDCG@20 约 0.05–0.10）；在工业长序列（上千长度）、大 batch 场景下的收益和动态执行开销都没有验证；
- **延迟收益有限**：0.87–0.995×；论文自己也提到需要 sparsity-aware kernel；
- **R 需要预先固定**：最大递归数作为超参在训练前定死，作者把"学习递归预算"列为未来工作；
- **退出后能否被看到，没有消融**：attention 版本里，退出的 item 不再作为 K/V（行和列都 mask）；卷积 / 递归版本里，它冻结但仍可见（只 mask 行）。论文正文只描述了前者，没有在同一个 backbone 上对比两种做法。代码里留着 `exited_token_memory` 这个开关，但 release 版本只允许 `active_only`，看起来作者探索过其他选项但没有报告；
- **读出尺度**：$\omega$ 对提前退出的 item 落在 $[0.5c, c]$，会压低 $h$ 相对 $x$ 的权重。不过代码在读出 $x+\omega h$ 之后还有一层 final LayerNorm（pre-norm 配置），整体尺度会被重新归一化，影响主要体现在 $x$ 与 $h$ 的相对比例上；
- **工业启示**：
  1. 与 LoopCTR 的 Oracle 结论相互印证——自适应深度确实有可观余量，而 item 级是比样本级更细的切入点；
  2. "高频 item 浅、低频 item 深、近期 item 深" 是一个可迁移的先验，在用户行为序列编码（如 HSTU / long-sequence 场景）中可以用来做**按 token 的计算预算分配**或长序列裁剪；
  3. 硬路由 + 沿路径的可微读出权重 + 负载均衡 + z-loss + warm-up 是一套可直接复用的训练配方。
