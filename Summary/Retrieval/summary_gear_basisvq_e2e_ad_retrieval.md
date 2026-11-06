# GEAR: Generative End-to-end Ad Retrieval at Douyin

## 基本信息

- **论文标题**: Generative End-to-end Ad Retrieval at Douyin
- **arXiv ID**: 2609.39327
- **作者**: Shaowen Zeng, Yanhua Huang, Jiacheng Sun, Jiarui Liu, Qian Dai, Zhikai Yang, Hancheng Li, Boya Wu, Tuoyu Zhang, Yekui Chen, Xiang Sun
- **单位**: 字节跳动（ByteDance）
- **场景**: 抖音广告召回（Douyin Ads），作为新增召回通路服务数亿 DAU
- **关键词**: Generative Retrieval, Semantic ID, 端到端 Tokenizer, Representation Collapse, Item Collision, BasisVQ, Prefix-aware RQ

---

## 核心问题

生成式召回把推荐变成"生成物品离散 token 序列"的任务。但在工业场景落地时有两个瓶颈：

1. **表征坍塌（Representation Collapse）**：真实推荐系统存在**持续的分布漂移**（新物品、新兴趣不断出现）。在流式数据上直接反传更新 tokenizer，梯度不稳定，码本陷入低利用率的次优状态（大量"死码"）。现有方案：
   - 冻结 tokenizer（OneRec、OneRec-V2、GR4AD）→ 无法适应分布漂移；
   - 启发式策略（EMA、死码重初始化）→ 只是优化稳定性的弱替代品。
2. **物品碰撞（Item Collision）**：**候选池巨大**，语义相似或码本容量有限导致不同物品共享相同 token 序列。现有方案：
   - 交给下游精排消歧 → 计算开销大、优化割裂；
   - 追加唯一 token（TIGER、GR4AD、LIGER）→ 唯一 token 无语义结构，损害泛化。

**关键洞察**：这两个瓶颈是**耦合**的——为缓解碰撞而扩大码本容量，会进一步加剧坍塌。因此需要统一框架同时解决。

---

## 方法：GEAR 框架

GEAR 联合端到端优化 **Tokenizer + Generator + Reranker** 三个组件，并做训练-服务协同设计。

### 1. Tokenizer：Prefix-aware BasisRQ

**物品表示**：使用标准双塔，取物品塔的中间隐层 embedding 作为待量化的稠密表示。
#### BasisVQ（核心创新）

- 引入可学习基矩阵 $W \in \mathbb{R}^{d\times d}$，通过 **Newton-Schulz 迭代**（可微）近似正交化：
  $W \leftarrow \frac{1}{2}W(3I - W^\top W)$，得到 $\hat W = h(W)$。
- 码本重参数化为 $\hat C = C\hat W$：**无论选中哪个码，梯度都会通过 $\hat W$ 影响整个码本**（全局梯度共享）。
  - **SID 的获取流程**：
    - **训练时**：每次前向都先算出 $\hat C = C\,h(W)$。隐层 embedding $z_i$（或残差 $x_l$）与 $\hat C$ 做最近邻匹配，得到码索引 $t_l$；各层的 $t_l$ 拼起来就是 SID。梯度反传更新 $C$ 和 $W$。原始 $C$ 从不直接用于匹配。
    - **推理和建索引时**：$\hat C$ 离线预计算并缓存，物品塔前向得到 $z_i$ 后，在 $\hat C$ 上查最近邻即可。由于 tokenizer 在流式训练中持续更新，物品的 SID 会定期重算（见系统部署中的 instance watcher）。
    - 这不是"先学码本、再分配 SID"的两阶段流程：重参数化与 SID 分配在每一步训练中同时发生。
    - 加上 prefix-aware 后，第 $l\ge2$ 层匹配的是 $\gamma\odot\hat C_l+\beta$，而不是 $\hat C_l$ 本身。
  - **变换的对象是码本，不是隐层 embedding**：$\hat W$ 右乘的是码本 $C$，残差 $x_l$ 保持原样；重建和残差都在原始 $z$ 空间中计算。
  - **在最近邻分配上等价于反向旋转输入**：因为 $\hat W$ 正交，$\|x - c_k\hat W\|_2 = \|x\hat W^\top - c_k\|_2$。把 $x$ 旋转 $\hat W^\top$ 后去匹配原始 $C$，选出的码完全相同。所以"对 embedding 做正交变换"只在分配这一步是一种等价视角。
  - **本质是重参数化，可表达的码本集合不变**：取 $\hat W = I$ 就退回 vanilla VQ，任意 $\hat C$ 也都能由 $C$ 单独表示。BasisVQ 改变的只是**优化动力学**：codebook loss $\|\mathrm{sg}[x]-c_k\hat W\|^2$ 的梯度既更新被选中的 $c_k$，也更新共享的 $\hat W$，进而带着所有码一起刚性旋转。commitment loss 则把 $x$ 拉向 $\mathrm{sg}[c^*]$。
  - 推断：每个 RQ 层应各有一个 $\hat W_l$（原文是"applying it to all codebook layers"）。
- **收敛保证**：Newton-Schulz 收敛到正交矩阵的充要条件是谱范数 $< \sqrt{3}$；先做 $W \leftarrow W/\max(\|W\|_F,\epsilon)$，因 $\|W\|_2 \le \|W\|_F$，谱范数被限制在 $\le 1$。
- **几何直觉**：
  - SimVQ 用无约束线性变换 $CW$ 也能共享梯度，但会扭曲非活跃码，甚至将其推到无穷远，永远无法恢复；且 SimVQ 冻结基础码本，限制容量。
  - BasisVQ 的正交约束使得优化等价于对整个坐标系做**刚性旋转**：非活跃码随活跃码一起旋转，**保持范数与两两拓扑不变**，相当于在隐空间中"扫描"，在流式分布漂移下更容易与新出现的数据流形相交，从而被"复活"。
- **复杂度**：训练时 5 次迭代足够，额外开销 $O(d^3)$，与 batch size 无关；参数开销 $O(d^2)$；推理时 $\hat C$ 可离线预计算，**零额外延迟**。
- 扩展到所有残差层即为 **BasisRQ**。

#### Prefix-aware 扩展

- 传统 RQ（RQ-VAE、RQ-KMeans）各层独立，第 2 层不知道第 1 层选了什么，表达力受限。QINCo 用 MLP 条件化码本，效果好但开销巨大。
- GEAR 对当前层码本做**逐元素仿射变换**，参数由之前所有层的量化结果查表得到：每层 $k$ 有 $\Gamma_k, B_k \in \mathbb{R}^{N_k\times d}$，按已选码查出 $\gamma_k,\beta_k$，再组合：
  $$\gamma = \bigodot_{k=1}^{l-1}\gamma_k,\quad \beta = \sum_{k=1}^{l-1}\Big(\beta_k \odot \bigodot_{k'=k+1}^{l-1}\gamma_{k'}\Big)$$
  （即逐层仿射变换的复合）。
  - **$\gamma_k,\beta_k$ 从哪来**：它们**不是从码本里查出来的**，而是从每层独立的可学习参数表 $\Gamma_k, B_k \in \mathbb{R}^{N_k\times d}$ 中，用第 $k$ 层已选码的索引 $t_k$ 查行：$\gamma_k = \Gamma_k[t_k]$，$\beta_k = B_k[t_k]$。$\Gamma_k, B_k$ 与码本 $C_k$ 行数相同，但参数是分开的。
  - **作用在谁身上**：仿射变换作用在**当前第 $l$ 层的码本** $\hat C_l$ 上，不是作用在前几层的量化结果或输入上。每个样本得到自己的码本 $\tilde C_l = \gamma\odot\hat C_l + \beta$，第 $l$ 层选中的重建码为 $\gamma\odot\hat c_{l,t_l} + \beta$。第 1 层没有前缀，不做变换。
  - **组合方式**：设 $f_k(c) = \gamma_k\odot c + \beta_k$，则 $f_2(f_1(c)) = \gamma_1\gamma_2 c + \beta_1\gamma_2 + \beta_2$，和上式一致。也就是说，作用在码本上的是 $f_{l-1}\circ\cdots\circ f_1$，越早的前缀越在内层。
  - **不只取决于上一层**：第 $l$ 层的变换取决于**前面所有层**的选择 $(t_1,\dots,t_{l-1})$，而不只是 $t_{l-1}$。查表确实是按层进行的：每个前缀层 $k$ 用自己的 $t_k$ 查一组 $(\gamma_k,\beta_k)$，然后全部复合起来。也可以写成递推形式：$\gamma^{(l)} = \gamma^{(l-1)}\odot\gamma_{l-1}$，$\beta^{(l)} = \beta^{(l-1)}\odot\gamma_{l-1} + \beta_{l-1}$。即"上一层累积下来的变换"再叠加"用 $t_{l-1}$ 新查出的一组"。所以"根据上一层的选择查表"只在这个递推意义下成立，最终效果是整条前缀路径决定当前层的码本。
  - **"改变当前层的表达"**：具体是指每个样本按自己的前缀路径，把第 $l$ 层码本整体缩放、平移到不同位置。相当于前缀不同的物品，在第 $l$ 层面对的是不同的码本（类似一棵树：每个父节点下有专属的子码本），但底层共享同一份 $\hat C_l$ 参数。相比之下，vanilla RQ 中所有前缀共用完全相同的第 $l$ 层码本。
  - 由公式可知，$\gamma_k$ 只取决于 $k$ 和 $t_k$，与目标层 $l$ 无关。因此第 $k$ 层的 $\Gamma_k, B_k$ 被后续所有层共享。
  - 逐元素缩放会破坏 BasisVQ 的保范数性质：正交旋转只作用在共享的 $\hat C_l$ 上，样本级的码向量之间不再是刚性关系。
- **快速距离计算**：朴素实现需要 $(b, N_l, d)$ 的样本级码本张量，内存爆炸。令 $\tilde x = x - \beta$，展开 L2 距离：
  $$D = (\tilde x\odot\tilde x)\mathbf{1}_d\mathbf{1}_{N}^\top + (\gamma\odot\gamma)(\hat C\odot\hat C)^\top - 2(\tilde x\odot\gamma)\hat C^\top$$
  全部是标准矩阵乘，时间复杂度 $O(bNd)$，与 vanilla VQ 渐近相同。

### 2. Generator：Encoder-Decoder

- **Encoder（轻量）**：非序列特征 embedding 随机分为 $L_g$ 组，各自 FFN 投影为 token；点击序列（长 $L_c$）、转化序列（长 $L_e$）分别经 FFN 投影。拼接为 $H = \mathrm{Concat}(H_g, H_c, H_e)$，作为所有 decoder 层 cross-attention 共享的 KV。
- **Decoder**：N 层 Transformer，每层依次为 Cross-Attn → GatedFFN → Causal Self-Attn → GatedFFN，Pre-RMSNorm，GatedFFN 为 SwiGLU。每一步使用独立的输出投影 $W_{out}^{(l)}$ 预测第 $l$ 层码。
- **与 tokenizer 的关系**：
  - **这里的 tokenizer 指什么**：tokenizer 不在生成器内部，而是与生成器并列的独立模块，即"双塔（用户塔 + 物品塔）+ Prefix-aware BasisRQ"。它的职责是把物品映射为 SID：物品特征 → 物品塔中间隐层 $z_i$ → BasisRQ → $(t_1,t_2,t_3)$。论文把整个双塔都算作 tokenizer 的一部分，因为用户塔只为 $\mathcal{L}_{LTR}$ 提供监督，让物品表示学到偏好信号。生成器和 tokenizer 只通过 SID 交互：训练时 SID 是监督标签，推理时 SID 是倒排索引的 key。
  - **是 encoder-decoder，但不是"先产出码本再训练生成器"**：tokenizer 与生成器在同一个训练过程中联合训练，只是用渐进 warm-up 错开启动时间。tokenizer 实时产出的 SID 作为生成器 teacher forcing 的目标。
  - **Encoder 非常轻**：没有 Transformer 编码层，只用 FFN 把特征组和行为序列投影成 KV token（$28+200+100$ 个），即 $H$。真正的建模集中在 2 层 decoder 的 cross-attention 里。
  - **码本和 prefix-aware 变换不进入生成器**：它们只负责把物品映射为 SID $(t_1,t_2,t_3)$。decoder 只接触离散 token，每一步的输出是 $N_l$ 个类别上的 softmax（2048/512/128）。decoder 的 token 输入 embedding 是否复用码向量，论文没有说明（推断：应该是独立学习的 embedding）。
  - **梯度怎么"端到端"**：SID 是 argmin 得到的离散标签，$\mathcal{L}_{Gen}$ 的梯度不会回传到 tokenizer。端到端体现在两处：（1）SID 随 tokenizer 在线更新，而不是冻结；（2）reranker 使用 tokenizer 物品塔的 $v_i$，$\mathcal{L}_{Rerank}$ 的梯度会回传到物品塔。

### 3. Reranker：上下文条件化的轻量重排头

- 生成完 $(t_1,\dots,t_m)$ 后**再多解码一步**，输出融合了层级 token 与用户特征的交互，经 MLP 得到 $u_{\text{rerank}}$，与 tokenizer 物品塔的物品 embedding $v_i$ 做点积：$s_i = u_{\text{rerank}}^\top v_i$。
- 关键点：$u_{\text{rerank}}$ 同时依赖用户和**生成的 token 序列**，每个候选序列都有专属用户表示，只需负责区分该序列下的少量碰撞物品，因此简单点积即可达到接近重 MLP 交互模型的细粒度区分能力。
- 开销仅为一次额外 decoder 步 + 点积，复用 encoder 输出 $H$ 与物品 embedding。
- **多解码的一步相当于什么**：相当于一个**以完整 SID 路径为条件的用户表示**。$u_{\text{rerank}} = \mathrm{MLP}(\text{decoder}(H, [\text{BOS}, t_1, t_2, t_3]))$，既包含用户特征 $H$，也"知道"自己要在哪个 SID 桶里做排序。
- **为什么需要多这一步**：预测 $t_3$ 时，decoder 的输入是 [BOS, $t_1$, $t_2$]，隐状态还没看到 $t_3$。只有把 $t_3$ 也喂进去再走一步，隐状态才覆盖完整路径 $(t_1,t_2,t_3)$，从而针对这个具体的桶生成专属用户向量。
- **Reranker 的作用**：
  - **桶内消歧**（主要作用）：碰撞率约 52%，同一个 SID 下往往有多个广告，生成器给它们的概率完全相同，无法区分，必须有物品级的分数。
  - **桶内截断与多样性**：线上每个 SID 只保留 rerank 分最高的若干个，避免单个 SID 占满结果。
  - **训练信号**：$\mathcal{L}_{Rerank}$ 通过 $v_i$ 回传到物品塔。
  - 消融显示，去掉 rerank、改为随机截断后，ADSS 从 +0.446% 降到 +0.358%。

### 4. 训练目标

四个损失联合训练：$\mathcal{L} = \lambda_{LTR}\mathcal{L}_{LTR} + \lambda_{RQ}\mathcal{L}_{RQ} + \lambda_{Gen}\mathcal{L}_{Gen} + \lambda_{Rerank}\mathcal{L}_{Rerank}$，权重为 $(1,1,10,1)$。

| 损失 | 作用 | 形式 |
|---|---|---|
| $\mathcal{L}_{LTR}$ | 训练双塔 | LambdaLoss（pairwise logistic × \|ΔNDCG\|） |
| $\mathcal{L}_{RQ}$ | 量化 | codebook loss + commitment loss（$\eta=0.1$） |
| $\mathcal{L}_{Gen}$ | 生成器 | teacher forcing 的加权 CE，权重 $w=\log(1+\text{secpm})$（仅曝光样本，secpm 为精排阶段的收入指标） |
| $\mathcal{L}_{Rerank}$ | 重排头 | LambdaLoss |

**渐进式 warm-up**：先只训练双塔（$\mathcal{L}_{LTR}$）→ 再开启 $\mathcal{L}_{RQ}$ → 最后开启 $\mathcal{L}_{Gen}$ 与 $\mathcal{L}_{Rerank}$，避免生成器依赖不稳定的 tokenizer。

**训练过程示例**（数值为虚构，标注"推断"的部分是论文没有写明的细节）：

一条流式样本 = 用户 u 的一次请求，采样到 4 个广告，标签 $y$ 按价值排序：A 转化（$y$=2，secpm=30），B 点击（$y$=1，secpm=8），C 仅曝光（$y$=0，secpm=3），D 未曝光（$y$=0）。处于 warm-up 的第三阶段，四个损失全部开启。

1. **双塔 → $\mathcal{L}_{LTR}$**：用户塔输出 $u$（96 维），物品塔对 A–D 分别输出中间隐层 $z$（128 维，供量化）和最终输出 $v$（96 维）。计算 $s_i = u^\top v_i$，用 LambdaLoss 让 $s_A > s_B > s_C, s_D$。
2. **BasisRQ 量化 → $\mathcal{L}_{RQ}$**（以 A 为例）：
   - 先算 $\hat C_l = C_l\,h(W_l)$。
   - **第 1 层**：在 $\hat C_1$（2048 个码）中找离 $z_A$ 最近的码，得到 $t_1 = 17$；残差 $x_2 = z_A - \hat c_{1,17}$。
   - **第 2 层**：查 $\gamma_1 = \Gamma_1[17]$、$\beta_1 = B_1[17]$。用快速距离公式在 $\gamma_1\odot\hat C_2 + \beta_1$（512 个码）上匹配 $x_2$，得到 $t_2 = 305$；残差 $x_3 = x_2 - (\gamma_1\odot\hat c_{2,305} + \beta_1)$。
   - **第 3 层**：再查 $\Gamma_2[305]$、$B_2[305]$，与第 1 层的复合得到 $(\gamma,\beta)$。在变换后的 $\hat C_3$（128 个码）上匹配，得到 $t_3 = 42$。
   - A 的 SID = (17, 305, 42)。对每层累加 codebook loss + commitment loss，梯度更新 $C_l$、$W_l$、$\Gamma_k$、$B_k$，commitment loss 也会回传到物品塔。
3. **生成器 → $\mathcal{L}_{Gen}$**：
   - Encoder 把 u 的特征投影成 $H$（28+200+100 个 KV token）。
   - Decoder 用 teacher forcing 输入 [BOS, 17, 305]，三步分别预测 17 / 305 / 42。
   - 每个样本的 CE 权重为 $w=\log(1+\text{secpm})$：A 为 $\log 31\approx3.43$，B 为 $\log 9\approx2.20$，C 为 $\log 4\approx1.39$；D 未曝光，$w=0$，不参与。高价值的 A 主导生成目标。
   - SID 是离散标签，这部分梯度只更新 encoder 和 decoder，不回传到 tokenizer。
4. **Reranker → $\mathcal{L}_{Rerank}$**：
   - 在 [BOS, 17, 305, 42] 之后多解码一步，经 MLP 得到 $u_{\text{rerank}}^{(A)}$，打分 $s_A^{\text{rerank}} = u_{\text{rerank}}^{(A)\top} v_A$。
   - 推断：B、C、D 各自以自己的 SID 走同样流程，得到各自的分数，再在这 4 个物品上算 LambdaLoss。
   - 梯度回传到 decoder、MLP，以及通过 $v_i$ 回传到物品塔。
5. **合并更新**：$\mathcal{L} = 1\cdot\mathcal{L}_{LTR} + 1\cdot\mathcal{L}_{RQ} + 10\cdot\mathcal{L}_{Gen} + 1\cdot\mathcal{L}_{Rerank}$，一次反传更新全部参数。下一批数据到来时，$C$、$W$、$\Gamma$、$B$ 和物品塔都已变化，A 的 SID 可能变成 (17, 311, 42)。生成器跟着新 SID 继续学，这就是"tokenizer 随分布漂移在线适应"。PS 中存的 SID 和 embedding 由 instance watcher 定期刷新。

### 5. 推理

Beam search 逐层保留 top $K_l$ 序列（beam = 256, 256, 512），得到 $K_m$ 个候选序列 → 展开到对应物品 → reranker 打分消歧。

- **用户侧走的是生成器，不是双塔的用户塔**：在线链路为用户特征 → 生成器 encoder 得到 $H$ → decoder beam search 解出 512 个 SID → 查 CSR 索引，展开为各 SID 下的广告及预计算的 $v_i$ → 对每个 SID 多解码一步得到 $u_{\text{rerank}}$，与 $v_i$ 做 INT4 点积 → 每个 SID 内按分截断 → 输出召回结果。SID 是"桶"，不是单个 item，桶内物品由 reranker 决定取舍。
- **双塔在推理中的作用**：
  - **物品塔**（近线、离线使用）：instance watcher 对每个广告跑一次物品塔 + tokenizer，得到 SID（建索引的 key）和 $v_i$（reranker 用的物品向量），存入 PS 和索引。
  - **用户塔**：论文的在线链路中没有出现，推断它只在训练时为 $\mathcal{L}_{LTR}$ 提供监督，让物品表示学到偏好信号，从而让 SID 和 $v_i$ 对排序有意义。
  - 注意：GEAR 是新增的召回通路，线上原有的 dense 双塔召回系统仍然保留，那是另一套独立的生产系统。

---

## 系统部署

作为现有召回系统之外的**额外召回通路**。所有与物品相关的计算移出在线关键路径，序列生成、序列→物品展开、物品打分集成在同一计算图中。

- **流式训练 + 近线索引**：参数服务器（PS）为每个物品存储 token 序列 + 物品 embedding，随流式训练持续更新；instance watcher 定期刷新存量物品、增量编码新激活物品（只需一次物品塔+tokenizer前向）；近线 index builder 构建 **CSR 风格索引**（排序目录：token 序列 → 对齐的物品列表 + 预计算 embedding），**分钟级**刷新，随模型一起下发。
- **在线推理**：KV-cache beam search → 查索引取物品及 embedding → reranker 计算序列专属用户 embedding，用 **packed INT4** 低精度点积打分 → 每个序列内按分排序并截断到上限，防止单一序列垄断结果，提升多样性。

---

## 实验结果

**配置**：三层 Prefix-aware BasisRQ，码本大小 (2048, 512, 128)（递减设计匹配逐层残差能量衰减），码向量维度 128；双塔输出 96 维；encoder $L_g=28$、$L_c=200$、$L_e=100$；decoder 2 层、6 头 × 64 维、FFN 中间维 384。

### 在线 A/B（抖音广告，7 天，每组 5% 流量）

| 方法 | ADSS | ADVV |
|---|---|---|
| **子集评估** | | |
| GEAR | **+0.446%** | **+0.498%** |
| w/o prefix-aware BasisRQ（换成 vanilla RQ） | +0.288% | +0.219% |
| w/o rerank（碰撞物品随机截断） | +0.358% | +0.341% |
| w/o secpm reweight | +0.395% | +0.354% |
| **全量部署** | | |
| GEAR | **+0.563%** | **+0.658%** |

- tokenizer 贡献最大，其次是 rerank 头，secpm 加权有额外增益。
- **冷启动**：新广告主冷启动激活率 **+1.46%**（离散语义 token 比 ID embedding 对无交互新广告泛化更好）。

### Tokenizer 分析

| 配置 | 利用率 Util.↑ | 碰撞率 Coll.↓ | MSE (×1e-3)↓ |
|---|---|---|---|
| Prefix-aware BasisRQ (2048,512,128) | **8.83%** | **52.3%** | **1.82** |
| Prefix-aware BasisRQ (512,512,512) | 6.16% | 75.6% | 2.04 |
| BasisRQ (2048,512,128) | 7.45% | 63.8% | 2.03 |
| Vanilla RQ + EMA (2048,512,128) | 4.22% | 84.2% | 2.32 |

- 递减码本 > 均匀码本（同等容量）；
- 去掉 prefix-aware：碰撞率 +11.5pp；
- BasisRQ vs vanilla RQ：利用率 +3.23pp、碰撞率 −20.4pp。

### 合成数据实验（附录）

8 维空间 512 个高斯簇（12.8 万点），码本初始化在原点附近 $[-0.1,0.1]$ 小立方体内（模拟分布漂移后旧码本远离新数据流形）。t-SNE 可视化显示：SimVQ 能激活更多码，但因码本冻结无法覆盖整个流形；BasisVQ 通过刚性旋转复活死码，更快且更均匀地覆盖全部流形。

---

## 个人点评

**亮点**
- **BasisVQ 思路简洁优雅**：正交基重参数化 = 全局梯度共享 + 拓扑保持，直击"死码不可恢复"的根因，且推理零开销，工程上极易移植到任何 RQ tokenizer。
- **Prefix-aware 仿射 + 快速距离公式**：以 QINCo 一小部分成本获得前缀感知能力，复杂度不变，实用性强。
- **Reranker 设计巧妙**：多解码一步得到"序列条件化"的用户向量，把碰撞消歧问题局部化，避免追加无语义的唯一 token。
- 真正做到 tokenizer 在流式训练中端到端可微更新，并配套 PS + 分钟级近线索引刷新，是从"冻结 SID"走向"动态 SID"的较完整工业方案。

**不足 / 疑问**
- 只有线上 A/B 和 tokenizer 指标，缺少离线召回指标（Recall@K 等）与公开数据集对比，难以与 TIGER/OneRec/GR4AD 等横向比较。
- 碰撞率仍高达 52.3%、码本利用率仅 8.83%（按完整码序列空间计），对 reranker 依赖较重；未分析每个序列下的平均物品数和截断上限的影响。
- 正交约束意味着 $\hat W$ 只能旋转不能缩放，码本整体表达力是否受限、与 SimVQ 的直接线上对比缺失（仅有 toy 实验）。
- tokenizer 持续更新时物品 SID 会漂移，生成器训练目标随之变化，论文对 SID 稳定性 / 新旧 SID 不一致问题讨论不足（仅靠 warm-up 与分钟级索引刷新）。
- 作为额外通路，增益（ADSS +0.56%）中有多少来自"多一路召回"本身而非方法设计，需结合子集消融理解。

**相关工作关联**：TIGER、OneRec / OneRec-V2、GR4AD、GPR、NEZHA、PIT、MTGRec、SimVQ、GRIT-VQ、QINCo、PSRQ、Gryphon、LIGER、COBRA。
