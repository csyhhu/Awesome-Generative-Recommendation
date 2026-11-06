# SequenceO1：基于低秩缓存（Low-Rank Caching）的端到端 100K 超长序列推荐建模

来源：arXiv: [2609.08443](https://arxiv.org/abs/2609.08443)（RecSys 2026，ByteDance / 抖音）

前作：[Make It Long, Keep It Fast（STCA + RLB，10K）](summary_stca_rlb_ext_10k.md)；部署融合骨干：[MixFormer](../Ranking/summary_mixformer_co_scaling.md)

## 综合理解（我的表述）及点评
> 本文设置了一个全局 Fix-length Query Matrix（prototype），对部分序列做压缩处理。该部分序列作为 kv，prototype 作为 query，进行第一次 attention。然后经过 residual、ffn 等基本操作，产生一个叫 sketch 的 fix-length matrix；从第二层开始，使用 sketch 作为 query，序列作为 kv 继续计算。最终将 sketch 作为该部分序列的表达进入计算。

**点评：骨架正确，缺一个关键点，有一处措辞需要修正。**
- ✅ **正确**：
  - 全局固定长度的可学习 prototype（$k\times d$，$k=1024$）作为第一层的 query，序列作为 KV；
  - 每层在 attention 之后接 residual、LN、FFN、LN，输出固定长度的 sketch；
  - 从第二层起用上一层的 sketch 作为 query，KV 仍是同一段原始序列；
  - 最后一层的 sketch 作为序列的表示参与后续计算。
- ⚠️ **缺失的关键点：attention 的归一化方向**。只说"prototype 作为 query、序列作为 KV 做 attention"，就和 Perceiver / Q-Former 的 learned-query pooling 没有区别了。SA 的本质是 **softmax 沿 prototype 维度做**（$\sum_j A_{j,i}=1$）：每个 token 把单位质量**分配**给各 prototype，而不是每个 prototype 在 token 上**挑选**。正因为这样，每条历史都必然进入 sketch，target 无关的压缩不会过早丢掉信息。消融中，换 query 来源的变体只有 +0.72% / +0.78%，SA 为 +1.07%。
- ❌ **措辞修正："部分序列"应为"整段超长历史"**。SA 压缩的是完整的 100K 历史，**包括**最近 10K；最近 10K 另由 STCA_10k 做全分辨率建模。两个分支在时间上是重叠的，而不是"近 10K + 远 90K"的切分。
- 📝 **细节补充**：
  - KV 中只有 K 做了投影（$XW_K$），V 直接用 $X$；
  - 每层有独立的 $W_Q^{(\ell)},W_K^{(\ell)}$ 和 FFN；
  - "进入计算"具体指：最后一层的 sketch（$k\times128$）**按用户缓存**，再经 $W_A$ 升维到 $k\times1024$，作为 STCA 的 history 输入，与 candidate query 交互。

**修正后的版本**：用一组全局固定长度的可学习 prototype（$k\times d$）作为 query，以整段 100K 历史为 key（$XW_K$）和 value（$X$），做一次 **prototype-wise 归一化**的 attention（每个 token 把质量分配给各 prototype），再经过 residual、LN、FFN、LN，得到固定长度的 sketch。从第二层起，用上一层的 sketch 作为 query，对同一段原始序列重复这一过程。最后一层的 sketch 作为整段历史的表示：按用户缓存，并经 $W_A$ 升维后作为 STCA 的 history 输入，与 candidate query 交互。

## 研究动机
- 抖音用户一年的消费行为就能达到 $10^5$ 量级。在 100K 规模下，瓶颈是**系统性**的：特征存储、样本体积、主机-设备以及分布式通信、训练和推理计算都会随长度膨胀，不只是单层 attention 的复杂度问题。
- 现有路线各有缺陷：
  - **截断 / 两阶段检索（SIM、TWIN V2）**：压缩和检索与最终排序目标分离，无法端到端优化，系统也更复杂。线上基线中，TWIN V2 先离线层次聚类成约 10K 条，再由 GSU 检索。
  - **长度外推（HSTU 随机长度、STCA 的 train sparsely / infer densely）**：外推比大约只有 4~5 倍（ULTRA-HSTU 训练 4.4K→推理 16K；STCA 训练 2~2.5K→推理 10K）。要推到 100K，训练长度仍需约 20K+，存储、通信、计算依然很重。STCA 推理代价也随 $L$ 线性增长，而且 cross attention 依赖 query，**无法跨 target 复用**。
  - **解耦式上游用户模型（LLaTTE、SOLARIS）**：上游模型异步产出用户表征再迁移给 ranker，存在表征瓶颈、目标不一致和新鲜度问题。
- 核心观察：超长历史既**可压缩**（Linformer / Perceiver / Set Transformer 的启示），又**可复用**。只要得到一个固定大小、只依赖用户的表征，就能一次计算、缓存，并跨训练样本、target 和相邻请求复用。

## 方法：Compress-then-Reason

### 1) Sketch Attention（SA）
- 学习 $k$ 个全局共享的 prototype $P^{(0)}\in\mathbb{R}^{k\times d}$（k 从几百到几千，生产中用 1K）。
- 亲和度：$S = \frac{(P W_Q)(X W_K)^\top}{\sqrt d}\in\mathbb{R}^{k\times n}$
- **Prototype-wise 归一化（关键差异）**：softmax 沿 prototype 维度做，即 $\sum_j A_{j,i}=1$，$A_{j,i}=p(j\mid i)$。每个历史 token 都把单位质量**分配（allocate）**到各个 prototype 上，而不是像 Perceiver / Poly-encoder 那样让每个 latent query 在 token 上做 softmax 去**选择（select）** token。
  - 原因：压缩发生在看到 target 之前，必须覆盖全部历史。token-wise softmax 会让多个 prototype 塌缩到相同的高显著区域，导致大量历史在 target conditioning 之前就被丢弃。
  - **"沿哪个轴归一化"的具体含义**：$S\in\mathbb R^{k\times n}$，行对应 prototype，列对应 token。
    - *Token-wise*（标准 attention / Perceiver / Q-Former）：**每行**做 softmax，`softmax(S, dim=-1)`，$\sum_i A_{j,i}=1$。每个 prototype 把总量为 1 的权重分给 $n$ 个 token，即 prototype 选 token，输出是加权平均。
    - *Prototype-wise*（SA）：**每列**做 softmax，`softmax(S, dim=0)`（batch 形式下是 `dim=-2`），$\sum_j A_{j,i}=1$。每个 token 把总量为 1 的权重分给 $k$ 个 prototype，即 token 选 prototype，输出是求和。
    - *实现上是否只差一个 `dim`？* 核心数学上是的，但真正落地时还有三处必须跟着改（个人分析）：
      1. **Padding mask 的位置**：token-wise 的做法是在 softmax 之前把 padding 列填成 $-\infty$。prototype-wise 如果照搬，整列都是 $-\infty$，再沿这一列做 softmax 会得到 **NaN**。正确做法是先 softmax，**再把 padding 列乘 0**。
      2. **尺度**：列和为 1 意味着 $AX$ 是求和，量级约为 $n/k$，并且随用户历史长度变化。在 Perceiver 实现上只改 `dim` 而不加 LN（论文中 ⑥ 的 Add&LN），数值会不稳定。
      3. **Kernel**：FlashAttention 和 SDPA 都把 softmax 固定在 key 维上。转置的技巧（token 作为 query、prototype 作为 key）只能得到 $A^\top$，而需要的输出是沿 token 维聚合的 $AX$，不是 $A^\top V$，所以无法直接复用现成算子。朴素实现会物化 $k\times n$ 的矩阵（约 200MB/层/用户），这就是需要自定义 **FlashSA** 的原因。好在每列的归一化是局部的，按 token 分块累加即可，写起来比 FlashAttention 简单。
      ```python
      # 单层 SA 参考实现。Xt: (B,k,d) prototype/sketch；X: (B,n,d)；mask: (B,n)，True 表示有效
      Q = Xt @ Wq                               # (B,k,d)
      K = X @ Wk                                # (B,n,d)
      S = Q @ K.transpose(1, 2) / d ** 0.5      # (B,k,n)
      A = S.softmax(dim=1)                      # 沿 k 归一化（token-wise 为 dim=-1）
      A = A * mask[:, None, :]                  # softmax 之后再屏蔽 padding，不能填 -inf
      Xh = A @ X                                # (B,k,d)，求和而非平均
      Xb = ln1(Xt + Xh)
      Xt_next = ln2(Xb + ffn(Xb))               # (B,k,d)，本层 sketch
      ```
    - *玩具例子（个人构造）*：2 个 prototype（$P_1$ 偏游戏，$P_2$ 偏美食），4 个 token（$t_{1..3}$ 为游戏视频，$t_4$ 为美食视频）。$S=\begin{bmatrix}4&4&4&1\\3&3&3&2\end{bmatrix}$，游戏视频比较"显著"，对两个 prototype 的得分都高。
      - Token-wise（按行）：$P_1$ 行得到 $[.33,.33,.33,.02]$，$P_2$ 行得到 $[.30,.30,.30,.11]$。**两个 slot 都塌缩到游戏上**，美食在 $P_2$ 中只占 11%，$t_4$ 的总贡献只有 0.13。
      - Prototype-wise（按列）：$t_{1..3}$ 每列得到 $[.73,.27]$，$t_4$ 列得到 $[.27,.73]$。$t_4$ 的总贡献恒为 1，在 $P_2$ 中占比约 48%，**美食兴趣被保留下来**。
    - *机制（个人分析）*：softmax 对同一归一化组内加常数不变。prototype-wise 下，**token 级别的整体偏移**（某条行为对所有 prototype 得分都高，例如热门或显著内容，key 范数大）会被约掉，只看它更倾向哪个 prototype，所以显著 token 无法独占所有 slot。token-wise 下被约掉的是 prototype 级别的偏移，显著 token 会在每一行都胜出，这就是塌缩的来源。
    - *代价：守恒量互换（个人分析）*：token-wise 保证**每个 slot** 恰好拿到 1 份权重，但 token 可能拿到 0；prototype-wise 保证**每个 token** 恰好贡献 1 份，但 slot 的质量 $m_j$ 可能极不均衡，甚至出现空 slot（所有 token 都不选它），即另一侧的塌缩。论文没有报告 slot 负载或利用率，也没有使用负载均衡正则，这是复现时值得监控的指标。
  - 在归一化思路上最接近 Slot Attention，但目标场景不同：Slot Attention 做迭代式的物体中心学习，SA 服务于强时延约束下的超长排序。
- Sketch：$\widetilde X = A X\in\mathbb{R}^{k\times d}$。它是加权聚合而不是选择，大小固定、与 target 无关、可微、可缓存。
- **直观理解："用一组全局可学习的 query 去读整段序列"**。这个说法大体成立，与 Perceiver / Set Transformer 的 PMA / BLIP-2 的 Q-Former 这类 learned-query pooling 同属一类：同一组 query 适用于任意长度的序列，所以输出固定为 $k$ 条。但有三点需要修正：
  1. *名词*：全局共享的是 **prototype $P^{(0)}$**；**sketch $\widetilde X$ 是每个用户的输出**，不是全局矩阵。
  2. *只有第 0 层的 query 是全局的*：Stacked Refinement 从第 1 层起用该用户自己的 sketch 作为 query，所以完整的 SA 是"从全局 query 出发，逐层个性化"。
  3. *归一化方向不同（本质区别）*：Perceiver 等是每个 query 在 token 上做 softmax，**query 挑选 token**，某些 token 可能被所有 query 忽略；SA 是每个 token 在 query 上做 softmax，**token 分配给 query**，每个 token 都必然贡献单位质量。由此带来两个连带差异：输出是求和而不是加权平均，并且 V 不做投影。
  压缩方法的消融也支持这一点：换 query 的来源（position-bucket query +0.72%、用最近 1K 行为初始化 query +0.78%）都不如 SA 的 +1.07%，说明增益主要来自归一化方向和逐层精化，而不只是"有一组 query"。
- **低秩视角**：$A$ 可看作数据相关的长度维投影（$n\to k$），类似 Linformer 的 $E,F$。区别在于参数量是 $O(kd+d^2)$，与 $n$ 无关，天然支持变长输入，输出又是可缓存的用户状态。作者强调这只是结构上的类比，不是 "Linformer for Rec"。

### 2) Stacked Refinement 与完整实现（逐步维度）
从 $\widetilde X^{(0)}=P^{(0)}$ 出发迭代：用当前 sketch 作为 query 重新计算 $A^{(\ell)}$，聚合得到 $\widehat X^{(\ell+1)} = A^{(\ell)}X$，然后经过 Add&LN、SwiGLU FFN、LN。这个过程可以理解为类 EM 的"分配-重估计"（只是概念类比）。$N_{sa}=2$ 就已足够。

**与 SA 的关系**：
- *定义上*：§3.1 的 SA 指单次"分配-聚合"算子（prototype-wise 的 cross attention，$\widetilde X=AX$）；Stacked Refinement 把这个算子按 Transformer block 的方式堆叠 $N_{sa}$ 次，每层外面套上 Add&LN 和 SwiGLU FFN。论文中"SA 模块"和"final SA"实际指的是整个堆叠后的模块，即 stacked SA（$N_{sa}=2$）。$N_{sa}=1$ 并且去掉残差和 FFN 时，就退化为 §3.1 的单块 SA。
- *相同点*：每层都是同一套机制，即 prototype-wise softmax 加上对**原始 $X$** 的聚合。
- *变化点*：Query 从全局固定的 $P^{(0)}$ 换成上一层的 sketch；每层有独立的 $W_Q^{(\ell)},W_K^{(\ell)}$ 和 FFN；只有 prototype 侧被精化，$X$ 不变。这类似 Perceiver 反复对同一输入做 cross attention，而不像标准 Transformer 那样两侧都更新。
- *为什么需要堆叠（个人分析）*：第 0 层的 query 是所有用户共享的 $P^{(0)}$，分配 $A^{(0)}$ 相当于用一组**全局固定的聚类中心**去划分每个用户的历史。从第 1 层起，query 是该用户自己的 sketch，分配会根据这个用户的历史分布重新调整，相当于**个性化的聚类中心**。这就是"refinement"的实际含义。这与 TWIN V2 的离线全局 KMeans（+0.30%）形成对比。
- *与 Slot Attention 的区别*：Slot Attention 各轮迭代**共享权重**，用 GRU 更新 slot，并按 $m_j$ 加权平均；SA 每层权重独立，用残差加 FFN，并且是直接求和后再接 LN。
- *代价*：每多一层就要再扫一遍 $X$（$2nd^2+4knd$），所以与 $n$ 相关的成本随 $N_{sa}$ 线性增加。$N_{sa}$ 从 2 增到 3 只多 +0.02%，因此不值得；去掉每层的 Add&Norm 和 FFN 则损失 -0.10%，说明 FFN 带来的非线性精化是有用的。

生产配置：$n\le 100\text{K}$（变长），$d=128$，$k=1024$，单头，$N_{sa}=2$，STCA 宽度 $D=1024$。

**Step 0：行为 embedding 与 action-side fusion**（在 SA 之前）
$$\mathbf e_i\in\mathbb R^{d},\ \mathbf a_i\in\mathbb R^{d_a}\ \xrightarrow{\text{concat}}\ [\mathbf e_i;\mathbf a_i]\in\mathbb R^{d+d_a}\ \xrightarrow{\text{SwiGLU}}\ \mathbb R^{d},\qquad \mathbf x_i=\mathbf e_i+\mathrm{FFN}_{\text{SwiGLU}}([\mathbf e_i;\mathbf a_i])$$
$$X=[\mathbf x_1,\dots,\mathbf x_n]^\top\in\mathbb R^{n\times d}\quad(100\text{K}\times128)$$
同一个 item 在 finish / skip / like 等不同反馈下会得到不同表示。消融中去掉这一步损失 -0.20%，是所有消融项里最大的。

**Step 1：初始化**：$\widetilde X^{(0)}=P^{(0)}\in\mathbb R^{k\times d}$（$1024\times128$），这是全局共享的可学习参数，所有用户相同。

**Step 2：第 $\ell$ 层（$\ell=0,\dots,N_{sa}-1$），每层有独立的 $W_Q^{(\ell)},W_K^{(\ell)}\in\mathbb R^{d\times d}$ 和 FFN 参数**

| 子步骤 | 公式 | 维度 | FLOPs |
|---|---|---|---|
| ① Query 投影 | $Q^{(\ell)}=\widetilde X^{(\ell)}W_Q^{(\ell)}$ | $(k\times d)(d\times d)\to k\times d$（1024×128） | $2kd^2$ |
| ② Key 投影 | $K^{(\ell)}=XW_K^{(\ell)}$ | $(n\times d)(d\times d)\to n\times d$（100K×128） | $2nd^2$ |
| ③ 亲和度 | $S^{(\ell)}=Q^{(\ell)}K^{(\ell)\top}/\sqrt d$ | $(k\times d)(d\times n)\to k\times n$（1024×100K） | $2knd$ |
| ④ **Prototype-wise softmax** | $A^{(\ell)}_{j,i}=\dfrac{\exp S^{(\ell)}_{j,i}}{\sum_{j'=1}^{k}\exp S^{(\ell)}_{j',i}}$ | $k\times n$，**每一列**（每个 token）和为 1 | 逐元素运算 |
| ⑤ 聚合 | $\widehat X^{(\ell+1)}=A^{(\ell)}X$，即 $\widehat{\mathbf x}_j=\sum_{i=1}^{n}A_{j,i}\mathbf x_i$ | $(k\times n)(n\times d)\to k\times d$ | $2knd$ |
| ⑥ 残差 + LN | $\overline X^{(\ell+1)}=\mathrm{LN}(\widetilde X^{(\ell)}+\widehat X^{(\ell+1)})$ | $k\times d$ | 逐元素运算 |
| ⑦ SwiGLU FFN | $\mathrm{FFN}(U)=(\mathrm{swish}(UW_1)\odot UW_g)W_2$，$W_1,W_g\in\mathbb R^{d\times 2d}$，$W_2\in\mathbb R^{2d\times d}$ | $k\times d\to k\times 2d\to k\times d$ | $12kd^2$ |
| ⑧ 残差 + LN | $\widetilde X^{(\ell+1)}=\mathrm{LN}(\overline X^{(\ell+1)}+\mathrm{FFN}(\overline X^{(\ell+1)}))$ | $k\times d$ | 逐元素运算 |

单层合计 $2nd^2+14kd^2+4knd$。当 $n=100\text{K}$ 时，主导项 $4knd\approx52$ GFLOPs/层。

**哪一个是 sketch**：论文用不同的符号区分各阶段。$\widehat X$（hat）是 ⑤ 的原始聚合结果，$\overline X$（bar）是 ⑥ 残差 + LN 之后的中间态，$\widetilde X$（tilde）才是 sketch。
- 在单块 SA（§3.1）中，⑤ 的 $AX$ 直接就是 sketch，这就是 "⑤ 之后就是 sketch" 这种理解的来源。
- 在堆叠版本中，⑤ 只是本层的"读出"。经过 ⑥–⑧ 得到的 $\widetilde X^{(\ell+1)}$ 才是**本层的 sketch 状态**，它同时作为下一层的 query。**最终被缓存、送入 STCA 的是最后一层 ⑧ 的输出 $\widetilde X^{(N_{sa})}$**。
- 注意：论文在 STCA 分支里又用 $\widehat X=\widetilde XW_A$ 表示 adapter 输出，与 ⑤ 的 $\widehat X^{(\ell+1)}$ 符号重名，阅读时容易混淆。

⑥–⑧ 的作用（结构上就是 Transformer 的 attention 子层 + FFN 子层，post-LN）：
- ⑥ 残差 $\widetilde X^{(\ell)}+\widehat X$：保留 slot 原有的身份（第 0 层是 prototype 本身），只把新读出的内容叠加上去，避免每层重新洗牌；LN 同时吸收 ⑤ "求和"带来的 $n/k$ 量级尺度。
- ⑦ FFN：⑤ 在给定 $A$ 时只是 $X$ 的线性组合；FFN 在每个 slot 内部做非线性变换，提升表达力，并把 slot 变成下一层更好的 query。
- ⑧ LN：稳定数值，为下一层或 adapter 做准备。
- 消融中去掉每层的 Add&Norm 和 FFN 损失 -0.10%。
- *成本几乎可以忽略（个人分析）*：⑥–⑧ 只作用于 $k$ 条 slot，FLOPs 为 $12kd^2$，与 $n$ 无关；相对于 $4knd$ 的比值是 $3d/n\approx0.4\%$（$n=100\text{K}$ 时）。压缩之后再做处理，计算量基本不用担心。
- *slot 之间没有直接交互（个人观察）*：⑥–⑧ 都是逐 slot 的运算，没有 sketch 内部的 self-attention。slot 之间只通过 ④ 中"同一个 token 在各 slot 间竞争"间接耦合。在 $k$ 条 slot 上加一层 self-attention 的成本很低（与 $n$ 无关），是一个可以尝试的扩展。

**Step 3：输出与衔接 STCA**：缓存 $\widetilde X=\widetilde X^{(N_{sa})}\in\mathbb R^{k\times d}$（1024×128，fp16 约 256KB/用户）。之后 $\widehat X=\widetilde XW_A$，$W_A\in\mathbb R^{d\times D}$，得到 $k\times D$（1024×1024），作为 sketch-STCA 的 history 输入。$W_A$ 的输出不做持久缓存。

完整数据流（$N_{sa}=2$）：
$$X\,(n\times d)\xrightarrow[\text{Q}=P^{(0)}]{\text{SA 层 0}}\widetilde X^{(1)}\,(k\times d)\xrightarrow[\text{Q}=\widetilde X^{(1)}]{\text{SA 层 1}}\widetilde X^{(2)}\,(k\times d)\ \xRightarrow{\text{缓存}}\ \xrightarrow{W_A}\,(k\times D)\xrightarrow[\text{Q}=\phi_t(\mathbf x_t)\in\mathbb R^D]{\text{STCA}\times N_{stca}}\mathbf z_u\in\mathbb R^{D}\to\text{与 }\mathbf z_r\text{ 一起送入 MixFormer}$$

"每层 SA 读一遍序列，再与本层输入的 sketch 做 residual + FFN + LN，得到本层 sketch，最后作为 candidate 侧 query 的 KV"——这个理解基本正确，补充三点：
1. **只有最后一层的 sketch 交给 candidate 侧**。中间层的 sketch 只作为下一层 SA 的 query，不会作为 KV 提供给 candidate。residual 的另一项就是本层的 query 输入（第 0 层即 $P^{(0)}$）。
2. **sketch 并不直接作为 KV**，而是 STCA 中 K/V 的来源：先经过 $W_A$ 升维到 $D$，STCA 每层再对 sketch 做 history-side pre-FFN（$12kD^2$），然后才进入 attention。在单 query 重排下，$\widehat XW_K$ 和 $\widehat XW_V$ 也不会显式物化：先算 $\mathbf u=(\mathbf qW_Q)W_K^\top$，得到 $\alpha=\mathrm{softmax}(\mathbf u\widehat X^\top/\sqrt{d_h})$，输出为 $(\alpha\widehat X)W_V$。
3. **缓存边界**：持久缓存的只有 $\widetilde X^{(N_{sa})}$（宽度 $d=128$，比 $k\times D$ 小 8 倍）；adapter 和 pre-FFN 这类与 target 无关的部分在每个 MRLB 组内算一次，由组内所有 target 共享；只有 target query 的 attention 部分需要按 target 逐个计算。

**实现要点**：
- **Q 和 K 分别是什么**：
  - *Q 来自 prototype / sketch 一侧*：$Q^{(\ell)}=\widetilde X^{(\ell)}W_Q^{(\ell)}\in\mathbb R^{k\times d}$，第 $j$ 行是第 $j$ 个 slot（可以理解为"兴趣簇中心"）在匹配空间里的表示。第 0 层的 $Q^{(0)}=P^{(0)}W_Q^{(0)}$ 是与用户无关的常量，之后各层的 Q 依赖该用户的 sketch。
  - *K 来自历史行为 token*：$K^{(\ell)}=XW_K^{(\ell)}\in\mathbb R^{n\times d}$，第 $i$ 行是第 $i$ 条行为在匹配空间里的表示。每层都要重算（$2nd^2$），是 SA 中除 $4knd$ 之外唯一随 $n$ 增长的投影。
  - *V 就是 $X$ 本身*，不做投影。
  - *与 STCA 的角色对比*：STCA 的 Q 是 target item（1 个），K/V 是历史；SA 的 Q 是 $k$ 个 prototype，K 是历史，**完全不涉及 target**，因此可以缓存。
  - *另一种等价视角（个人分析）*：softmax 沿 $j$ 做，所以 $A_{:,i}=\mathrm{softmax}_j(\mathbf k_i\cdot\mathbf q_j/\sqrt d)$，这恰好是"**以 token 为 query、prototype 为 key**"的标准 attention，表示 token $i$ 在 $k$ 个 slot 中挑选归属。只是随后的聚合 $AX$ 是沿 token 维求和（转置方向），把 token 写入 slot。所以命名上 prototype 是 Q，但从归一化语义看，token 才是"提问者"。
  - *范数的作用（个人分析）*：按列 softmax 时，只依赖 token 的项会被约掉。$\|\mathbf k_i\|$ 相当于 token $i$ 的**温度**，范数越大分配越尖锐、越接近硬聚类；$\|\mathbf q_j\|$ 和方向则决定 slot $j$ 对所有 token 的**先验吸引力**，类似混合模型的混合权重。token-wise softmax 下两者的作用正好互换。
- **没有 $W_V$**：value 直接用 $X$（宽度 $d$）。K 每层用新的 $W_K^{(\ell)}$ 重算，但 $X$ 本身在层间**不更新**，只有 prototype 侧被逐层精化。
- **第 0 层的 Query 与用户无关**：$Q^{(0)}=P^{(0)}W_Q^{(0)}$ 对所有用户相同，推理时可以预先算好当常量（论文未提，属推断）。
- **为什么不做 token-wise softmax**：token-wise 做法是 $A_{j,:}$ 对 $i$ 归一化，每个 prototype 在 $n$ 个 token 上竞争，容易塌缩到相同的高显著区域；prototype-wise 做法让每个 token 都有单位质量注入 sketch，保证覆盖。
- **尺度问题（个人观察，论文未讨论）**：prototype-wise softmax 只保证列和为 1。行和 $m_j=\sum_iA_{j,i}$ 满足 $\sum_j m_j=n$，所以 $\widehat{\mathbf x}_j$ 是**求和而非平均**，量级约为 $n/k$ 倍（100K/1K ≈ 100 倍），并且随用户历史长度变化。论文只靠 ⑥ 的 LN 吸收尺度。这样做的副作用是：$n$ 很大时残差项 $\widetilde X^{(\ell)}$ 会被 $\widehat X$ 淹没，而且 sketch 的"强度"信息（某个兴趣占多少行为）在 LN 之后部分丢失。Slot Attention 的做法是按 $m_j$ 做加权平均；如果要复现，可以对比"按 $m_j$ 归一化，并把 $\log m_j$ 作为额外特征"的版本。
- **padding / 变长**：每个 token 的归一化只涉及自身那一列，padding token 直接跳过即可，不会影响其他 token 的权重。这也是 ragged batch 容易实现的原因。
- **显存**：朴素实现中，$S$ 和 $A$ 每层每用户各需 $1024\times100\text{K}\approx10^8$ 个元素（fp16 约 200MB），这正是需要 FlashSA 的原因（见系统设计表）。

### 3) 双时间尺度 STCA 推理
- **近期分支**：$\mathbf z_r=\mathrm{STCA}_{10k}(\mathbf x_t, X_r)$，直接对最近 10K 行为建模，捕捉近期偏好。注意：SA 压缩的是**完整的 100K 历史**（论文原文为 "compresses the full $n=100$K history"），**包含**这最近 10K。两个分支在时间上是**重叠**的，而不是"最近 10K + 更早的 90K"这样的切分。最近的行为既以全分辨率出现在近期分支中，也以压缩形式出现在 sketch 中。
- **超长分支**：先用 width adapter $W_A$ 把 sketch 从 $d=128$ 映射到 STCA 宽度 $D=1024$，再做 $\mathbf z_u=\mathrm{STCA}_{sketch}(\hat{\mathbf x}_t,\widehat X)$。这一分支的长度固定为 $k$，与 $n$ 无关。
- **融合**：交给下游 ranker，线上用的是 MixFormer，但方法本身不绑定特定骨干。

## 系统设计：以 Amortization 为核心
| 组件 | 作用 |
|---|---|
| **训练侧 local KVCache** | 缓存 $d$ 宽度的用户 sketch $\widetilde X_u$。key = user id + 模型版本 + 历史时间戳 + sketch 配置。TTL 3h，命中率约 0.5。近期 10K 和候选特征不缓存 |
| **服务侧缓存** | 与训练共用 get/put 接口（减少 train/serve skew），TTL 1h，命中率约 0.6 |
| **MRLB（Multi-Request Level Batching）** | 把同一用户短时间内的多个请求聚成一组，sketch 只算一次，组内所有 target 复用。限制聚合度以稳定显存；近期分支加 request-aware mask 防止跨请求泄漏 |
| **Pipeline Lift** | SA 只依赖用户侧信息，可以提前到请求入口计算，与召回并行。命中时直接查表，100K 原始特征根本不需要物化和传输 |
| **FlashSA** | 类 FlashAttention 的融合 kernel：分块计算亲和度，在线维护 prototype-wise softmax 统计量（max / LSE），直接累加出 $\widetilde X$，不把 $k\times n$ 矩阵写回 HBM。支持 ragged batch，混合精度下数值稳定。*推断的实现方式*：沿 token 维分块（块大小 $B$），每块在片上算出 $S_{blk}=QK_{blk}^\top$（$k\times B$），对 $k$ 维求 max 和 LSE 得到 $A_{blk}$，再累加 $\widetilde X\mathrel{+}=A_{blk}X_{blk}$（$k\times d$）。因为每个 token 的归一化只依赖自己那一列，**跨块累加是普通求和，不需要 FlashAttention 那样的 online rescaling**。prototype-wise 归一化在这一点上反而比 token-wise 更容易做流式计算（前提是 $k=1024$ 个 prototype 能一次放进片上；如果 prototype 也要分块，则需要两遍扫描） |

缓存命中时省掉的是 100K 原始特征的存储、通信，以及 $n$ 相关的 SA 计算；未命中时只多出一次轻量的 SA。无论命中与否，重量级的 STCA 都只在 k 长度上运行。

## 复杂度分析（附录，GEMM FLOPs）
配置：SA 为单头、$d=128$、$N_{sa}=2$、$L_s=1024$；STCA 为 $h=16$、$d_h=64$、$D=1024$、$N_{stca}=4$。
- STCA：$N_{stca}(12LD^2 + 4LDh + 20D^2)$。单 query 重排后不再物化 $XW_K/XW_V$，所以**没有可跨 target 复用的 KV**，MRLB 只能摊销 history-side pre-FFN 这一项（$12LD^2$）。
- SA：$N_{sa}(2Ld^2 + 14L_sd^2 + 4L_sLd)$。同样随 $L$ 线性增长，但宽度 $d$ 小得多，而且**整个 SA 都能被 R 摊销**。
- 期望代价：$\mathbb E = \text{FLOPs}_{hit} + (1-p_{hit})\cdot\text{FLOPs}_{SA}$；在 MRLB 下 SA 项再除以 $R$。

100K 下超长分支的 FLOPs 对比（不含公共的近期 10K 分支和 ranker）：
| 场景 | STCA (GFLOPs/target) | SequenceO1 | 倍数 |
|---|---|---|---|
| 无 MRLB，$p_{hit}=0.5$ | 5059.46 | 108.10 | **46.8×** |
| 训练 MRLB，$R=40$，$p_{hit}=0.5$ | 152.13 | 3.046 | **49.9×** |
| 推理，$R=300$，$p_{hit}=0.6$ | 43.08 | 0.674 | **63.9×** |

设计哲学：两者都是 $O(L)$，但 SA 是**轻量压缩器**（宽度小、可复用），STCA 是**重量级推理器**（宽度大、依赖 target）。把长度相关的工作推给可摊销的轻量阶段，把算力集中到 target-history 精细交互上。

## 实验结论

### SA 与 vanilla（token-wise）attention 对比（100K 轻量设置）
只替换归一化方式，多任务 UAUC 全部提升：like +0.10%、follow +0.20%、clicmmt +0.09%、comment +0.19%、share +0.14%、finish +0.01%。

### SA 架构消融（相对 final SA；final SA 比 STCA(512) 的 Finish AUC 高 +1.07%）
| 变体 | 变化 |
|---|---|
| k: 1K→512 / 1K→2K | -0.04% / +0.03% |
| $N_{sa}$: 2→3 | +0.02% |
| d: 128→64 | -0.10% |
| 去掉 action-side SwiGLU fusion | **-0.20%** |
| 去掉每层 Add&Norm + SwiGLU FFN | -0.10% |

扩大 k 或层数的收益很小，因此保留较小的生产配置；action-aware 的 sketch 最关键。

**head 数没有做消融**：SA 固定为单头（附录 FLOPs 推导中写的是 "SA is single-head"，"We assume SA is single-head in our instantiation"），上表只扫了 $k$、$N_{sa}$、$d$ 三个超参。注意 STCA 的 $h=16$ 是 STCA 的配置，不是 SA 的。多头 SA（例如把 $d=128$ 切成若干头，每个头独立做 prototype-wise 分配）是否有收益，论文没有验证。

**层数的调参范围**：
- **SA**：$N_{sa}=2$，只做了 2→3 这一个点（+0.02%），没有测 $N_{sa}=1$。作者把 2→3 和 $k$ 1K→2K 的差异定位为"成本-质量参考，不作为显著性证据"。另外，"去掉每层 Add&Norm + FFN"（-0.10%）改的是每层的结构，不是层数。
- **STCA**：$N_{stca}=4$ 只出现在附录的 FLOPs 实例化配置里（"Hyperparameter instantiation for analysis"），没有做层数消融。$h=16$、$D=1024$ 同样没有扫过。

### 压缩方法对比（统一压缩到 1K×128，Finish AUC 相对 STCA(512) 的增益）
| 方法 | 增益 |
|---|---|
| **SA** | **+1.07%** |
| Lightning-Attention 压缩 | +0.83% |
| 最近 1K 行为初始化 query（LONGER 风格） | +0.78% |
| Position-bucket queries（每约 100 个行为一个桶） | +0.72% |
| Chunk mean pooling | +0.46% |
| TWIN V2 KMeans 聚类 | +0.30% |

增益来自压缩质量而不是表征尺寸。SA 明显好于 position bucket，作者据此认为推荐历史的**局部结构性弱于语言**，相关行为可能相隔很远。

### 与直接扩展 STCA 的对比
在 100K 截断（平均长度 85K）下，SequenceO1 的增益为 +1.07%，直接 STCA 为 +1.29%，**保留了 83% 的增益**，而训练和推理 FLOPs 分别降低约 50 倍和 64 倍。作为跨系统参照（不可直接比较）：LLaTTE 的上游到下游迁移比约 50~53%，SOLARIS 约 42~44%。

### 线上生产离线评估（基线：STCA 10K + TWIN V2；SequenceO1 去掉 TWIN V2，换成 100K sketch 分支）
| | Finish | Skip | Head | Like | Follow | Comment | Clicmmt | Share | Favourite | Dislike |
|---|---|---|---|---|---|---|---|---|---|---|
| ΔAUC | +0.29% | +0.23% | +0.18% | +0.09% | +0.05% | +0.04% | +0.16% | +0.18% | +0.30% | +1.29% |
| ΔUAUC | +0.40% | +0.43% | +0.46% | +0.72% | +0.58% | +0.43% | +0.61% | +0.94% | +1.47% | +3.63% |

所有指标都有提升，偏好类目标（Like / Share / Favourite / Dislike）的 UAUC 提升最大。

### 在线 A/B（一个月，全部统计显著）
| | 30 日活跃 | 时长 | Finish | Comment | Like | Dislike↓ |
|---|---|---|---|---|---|---|
| 抖音 | +0.197% | +1.50% | +2.33% | +3.59% | +2.36% | -6.98% |
| 抖音极速版 | +0.234% | +1.67% | +3.49% | +8.61% | +2.94% | -6.00% |

各活跃度分层都稳定为正。低活用户的活跃度和时长提升最明显（约 +0.55~0.66% 活跃），作者解读为召回流失用户的效果；高活用户则在 Finish 和 Like 上提升明显。

## 与相关工作的定位
- **对比前作 STCA 10K（[2511.06077](summary_stca_rlb_ext_10k.md)，同一团队）**：
  - *继承*：STCA 和单 query 重排原样沿用，用作近期 10K 分支和 sketch 分支的推理算子；RLB 扩展为 MRLB（从单请求内复用变为跨请求复用）；两篇都用端到端序列建模替换检索模块（TWIN(10k) → TWIN V2）；实验协议一致（抖音和抖音极速版一个月 A/B、按活跃度分层），本文的线上基线就是 "STCA 10K + TWIN V2"。
  - *控制训练成本的思路不同*：10K 版靠**长度外推**（训练平均约 2K → 推理 10K，随机长度 + U 形 Beta 采样）；本文认为外推比只有 4~5 倍，到 100K 仍需约 20K+ 的训练长度，因此改用 **SA 压缩加缓存**。
  - *复用能力不同*：STCA 依赖 query，单 query 重排后没有可缓存的 KV，跨请求只能摊销 pre-FFN；SA 的 sketch 与 target 无关，可以跨 target、跨样本、跨请求复用，并支撑 local KVCache、服务侧缓存、Pipeline Lift 和 FlashSA 这套系统组件。
  - *推理代价不同*：10K 版随 $L$ 线性增长；本文缓存命中时与 $n$ 无关，未命中时只多一次轻量 SA。
  - *精度取舍不同*：10K 版的设计哲学是"不压缩，在完整历史上做精确 attention"；本文承认需要有损压缩（保留直接 STCA 100K 约 83% 的增益），再用 prototype-wise 归一化减少压缩损失。从压缩方法的消融看，这种归一化正是 SA 胜出的关键。
  - *瓶颈判断不同*：10K 版的瓶颈是 FLOPs 加带宽（RLB 减少 77~84% 带宽）；100K 时瓶颈变成系统性的，特征存储、通信、训练和推理计算都随长度膨胀。
  - 一句话：10K 版通过改注意力结构（$O(L^2)\to O(L)$）把精确建模推到 10K；本文发现到 100K 时 $O(L)$ 也不够，而且无法复用，于是加一个与 target 无关的固定大小压缩层，把随长度增长的部分变成可缓存、可摊销的用户侧状态。
- **对比 TWIN V2 / SIM 等两阶段方法**：SequenceO1 端到端可微，压缩器与 ranker 联合优化，一个模块即可替代 TWIN V2。
- **对比 UIC/MIMN、HPMN 等增量用户状态**：这些方法可以常数时间读取，但不是在原始超长序列上端到端优化。
- **对比 Transformer KV / prefix cache（RelayGR、CollectiveKV 等）**：它们缓存逐位置状态，缓存体积随长度增长；SequenceO1 缓存**固定大小**的 sketch（100K→k）。
- **对比 LLaTTE / SOLARIS 等上游用户模型迁移**：SequenceO1 没有单独的上游模型，sketch 与最终排序目标联合训练。
- **对比 Q-Former（BLIP-2）/ Perceiver 等 learned-query pooling**（论文只引用了 Perceiver 和 Poly-encoder，Q-Former 的对比是讨论中补充的；Q-Former 的细节按 BLIP-2 原文）：
  - *相同的骨架*：一组固定数量的可学习 query；对**不变的输入**做多层 cross attention，只有 query 一侧逐层更新；输出固定长度；末端有一个线性层映射到下游宽度（Q-Former 映射到 LLM 宽度，对应 SA 的 $W_A$）。

| 维度 | Q-Former（BLIP-2） | Sketch Attention |
|---|---|---|
| **softmax 方向** | 沿输入 token（每个 query 在 257 个图像 token 上归一化）：**query 挑选 token** | 沿 prototype（每个 token 在 $k$ 个 slot 上归一化）：**token 分配给 slot** |
| 聚合 | 加权平均（行和为 1） | 求和（列和为 1），靠 LN 处理尺度 |
| query 之间的交互 | 每层都有 **query self-attention** | **没有**，只通过 softmax 中的竞争间接耦合 |
| 头数 / 投影 | 多头，有 $W_V$ | 单头，不投影 V（V 直接用 $X$） |
| 压缩比 | 257 → 32（约 8 倍），输入长度固定 | 最多 100K → 1024（约 100 倍），**输入变长** |
| 规模 | BERT-base 初始化，12 层（每隔一层有 cross attention），768 维，约 1 亿参数 | 2 层，$d=128$，参数量很小 |
| 训练方式 | 冻结图像编码器，分两阶段预训练（ITC / ITM / ITG，再接 LLM 生成） | 与 embedding 和 ranker **端到端**联合训练，只用排序损失 |
| 条件依赖 | 第一阶段 query 会和文本交互；InstructBLIP 让 query 感知指令，即**依赖条件** | **刻意保持与 target 无关**，以便缓存和复用 |
| 定位 | 模态桥接（视觉 → LLM） | 超长序列的可缓存压缩前端 |

  - *为什么归一化方向在推荐场景下更关键（个人分析）*：图像 token 少（257 个）且信息密集，用"挑选"也不会漏掉太多。100K 行为序列高度冗余并且长尾：头部兴趣占据大部分行为，冷门兴趣只有零星几条。在"挑选"模式下，多个 query 容易都塌缩到头部兴趣上，冷门兴趣在看到 target 之前就被丢掉了；而推荐恰恰常需要用这些冷门兴趣去匹配某个特定 candidate。"分配"模式保证每条行为都有单位质量进入 sketch。论文中 SA 与 vanilla attention 的对比（只换归一化方向）显示 UAUC 普遍提升，like +0.10%、follow +0.20%、comment +0.19%，可以看作对这一点的直接验证（那个 vanilla 变体可以视为不带 self-attention 的 Q-Former 式归一化）。
  - *可以借鉴的方向（个人分析）*：Q-Former 的 query self-attention 正好补上了 SA "slot 之间没有直接交互"的空缺。在 $k$ 条 slot 上加 self-attention，成本与 $n$ 无关，可以和 SA 的 prototype-wise 归一化结合起来尝试。

## 个人点评 / Takeaways
1. **Prototype-wise softmax** 是全文最值得借鉴的小改动。凡是"先于 target、需要复用"的压缩模块，都应优先保证覆盖（allocation），把选择（selection）留给下游 target-aware 模块。
2. **"Compress-then-Reason"加缓存**：把 $n$ 相关的计算集中到一个与 target 无关的轻量模块里，系统收益（存储、通信、pipeline lift）往往比单纯省 FLOPs 更大。
3. **复用边界**：STCA 单 query 重排后失去了可复用的 KV，而 SA 的 sketch 天然可复用。这说明评估"能否摊销"时，要同时看算子重排和缓存策略。
4. **局限**：缓存命中率（0.5 / 0.6）和 TTL 带来了新鲜度与正确性之间的取舍，文中没有给出命中率对效果（而非 FLOPs）的敏感性。与直接 STCA 100K 相比仍有约 17% 的增益差距。作者提出的未来方向包括多分辨率 / 层次化 sketch、按用户自适应调整 k，以及基于 prototype 分配的 sketch-to-behavior 检索和局部精化（文中尚未验证）。
