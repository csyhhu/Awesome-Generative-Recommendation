# LoopCTR: Unlocking the Loop Scaling Power for Click-Through Rate Prediction

- **arXiv**: [2604.19550](https://arxiv.org/abs/2604.19550)（NeurIPS 2026 投稿，preprint）
- **作者**: Jiakai Tang\*, Runfeng Zhang\*, Weiqiu Wang\* 等；通讯作者 Xu Chen（人大高瓴）、Yeqiu Yang（阿里巴巴）
- **领域**: CTR 排序 / Transformer 推荐模型 Scaling / Looped Transformer / 参数复用
- **一句话**: 提出 **loop scaling**：用**共享参数的 Loop Block 反复迭代**来提升训练时计算量，使计算与参数解耦；配合 **HCR（Hyper-Connected Residuals）+ MoE** 扩表达力、**逐 loop 深度的过程监督**，实现 **"train-multi-loop, infer-zero-loop"**——推理时完全跳过 Loop Block，单次前向即超过所有 baseline。

---

## 1. 问题与动机

现有 CTR Scaling（HSTU、MTGR、OneTrans、RankMixer、TokenMixer-Large 等）沿三条轴：

- **深度**：堆层；
- **宽度**：加大 token embedding 维度；
- **输入**：拉长用户行为序列。

共同点是 **参数和计算一同增长**，与工业部署的存储/延迟约束矛盾越来越大。

作者提出第四条轴：**通过递归复用来扩展计算（computation scaling through recursive reuse）**。同一层反复跑多次（loop latent reasoning），计算增加但参数不变；且权重共享本身是一种**归纳偏置**，能缓解稀疏推荐数据上的过拟合。

直接用标准 Transformer 层做 loop 有两个瓶颈：

1. **表达力瓶颈**：固定的计算流（固定 1:1 残差）下，单个共享层反复应用能做的事情有限；
2. **效率瓶颈**：推理时跑多次 loop，延迟线性增长，在线排序扛不住。

---

## 2. 方法：Sandwich 架构

```
输入 T（短序列 item token + 长序列压缩 query token + 全局 token）
   │
   ▼ Entry Block：分组异构投影 + 组内 self-attention（HCR + MoE）     → l=0 → Exit → ŷ(0)
   ▼ Loop Block × L（共享参数，Prefix Attention，HCR + MoE）           → l=1..L → Exit → ŷ(l)
   ▼ Exit Block：global→seq cross-attn + MLP tower（MoE）             → ŷ
```

### 2.1 输入预处理
- **短期序列** $\mathcal{S}^s$：保留原始 item token；
- **长期序列** $\mathcal{S}^l$（上千长度）：借鉴 **Q-Former**，用一组可学习 query token（InHouse 上为 16 个）通过 cross-attention 压缩，端到端训练；
- **全局特征**（user/item/context/cross）：每个字段 tokenize 为一个 token。

### 2.2 Entry Block
- **组特定投影**：第 $g$ 组 token 用独立 $\mathbf{W}_g, \mathbf{b}_g$ 映射到公共空间（每条序列一组、每个全局 token 单独一组），避免共享投影混淆异构语义；
- **组内独立 full self-attention**：$\mathbf{H}_{entry} = [\text{SelfAttn}(G_1);\dots;\text{SelfAttn}(G_K)]$，完全并行，attention 代价 $O(\sum_j m_j^2 d)$。

### 2.3 Loop Block（核心）
同一共享层迭代 $L$ 次：

$$\mathbf{H}^{(l)} = \text{PrefixAttn}\big([\mathbf{H}_{seq}^{(l-1)}; \mathbf{H}_{glb}^{(l-1)}],\ \mathbf{M}\big)$$

**非对称 mask**：序列 token 只看序列 token；全局 token 看全部。即"全局 token 聚合序列上下文，但不让全局特征污染序列表示"。

### 2.4 Exit Block
全局 token 对序列 token 做 cross-attention，拼接后过 MLP 得到 $\hat y$。

> **工程关键**：整个架构中**序列 token 从不 attend 全局 token**，因此用户序列侧表示（Entry 的序列组 + Loop 中 seq-to-seq attention）与候选 item 无关，可**每请求算一次、KV cache 跨候选共享**。在线代价 $C_{user} + N\cdot C_{item}$，均摊约等于 $C_{item}$。

---

## 3. Hyper-Connected Residuals（HCR）

动机：标准残差 $\mathbf{h}+f(\mathbf{h})$ 是单流 + 固定 1:1 混合，共享层在不同迭代下无法调整信息流。借鉴 Hyper-Connections / mHC，把隐状态复制成 $n$ 条流 $\mathbf{H}\in\mathbb{R}^{n\times d}$：

$$\hat{\mathbf{H}} = \underbrace{\mathbf{A}_r^\top \mathbf{H}}_{\text{残差混合}} + \underbrace{\mathbf{B}^\top \cdot \mathcal{T}\big((\mathbf{H}^\top \mathbf{A}_m)^\top\big)}_{\text{子层贡献}}$$

- $\mathbf{A}_m\in\mathbb{R}^{n\times1}$：把 $n$ 条流融合成子层输入；
- $\mathbf{B}\in\mathbb{R}^{1\times n}$：把子层输出分发回各流；
- $\mathbf{A}_r\in\mathbb{R}^{n\times n}$：流间残差混合。

三者都是 **静态可学习参数 + 输入相关的动态扰动**（$\bar{\mathbf H}=\text{RMSNorm}(\mathbf H)$）：

$$\tilde{\mathbf A}_m = \mathbf A_m + s_\alpha \odot \tanh(\bar{\mathbf H}\mathbf W_m),\quad \tilde{\mathbf A}_r = \mathbf A_r + s_\alpha \odot \tanh(\bar{\mathbf H}\mathbf W_r),\quad \tilde{\mathbf B} = \mathbf B + s_\beta \odot \tanh(\bar{\mathbf H}\mathbf W_\beta)^\top$$

- **初始化**：$\mathbf W_{m,r,\beta}=0$，$\mathbf B=\mathbf 1$，$\mathbf A_m=\mathbf e_{s \bmod n}$，$\mathbf A_r=\mathbf I$ → 起点等价于标准 Pre-Norm Transformer；
- **意义**：输入相关系数让残差 **instance-adaptive 且隐式 loop-aware**——不需要显式喂 loop index，共享层也能在不同迭代下表现不同；
- 实验中 $n=2$（attention 一条流、FFN 一条流），每个子层仅增加 $4d+10$ 个参数，开销可忽略。

---

## 4. MoE-Augmented Transformer

单个共享层的参数容量仍可能不足，在所有 block 中加入稀疏 MoE：

- **Attention MoE**：把 $\mathbf W_V$、$\mathbf W_O$ 替换为 MoE，**二者共享同一个 router**（同一 token 在 V/O 选同一组专家）；$\mathbf W_Q, \mathbf W_K$ 保持共享，以保证相似度计算一致；
- **FFN MoE**：标准 top-$k$ 路由；
- **负载均衡损失**（Switch Transformer 形式）：$\mathcal L_{bal} = E\sum_e f_e p_e$，总损失 $\mathcal L = \mathcal L_{total} + \lambda\mathcal L_{bal}$。

额外参数：$(E-1)(2d^2 + 2d\,d_{ff})$；计算只随激活专家数 $k$ 增长。

---

## 5. 训练目标：多深度过程监督

每个 loop 深度 $l\in\{0,\dots,L\}$（$l=0$ 即 Entry 输出）都过 Exit Block 得到 $\hat y^{(l)}$，平均 BCE：

$$\mathcal L_{total} = \frac{1}{L+1}\sum_{l=0}^{L}\mathcal L^{(l)}_{BCE}$$

- **Zero-loop 推理**：由于 $l=0$ 被直接监督，推理时只跑 Entry + Exit，完全绕过 Loop Block（部署时 Loop Block 可直接从子网络中删掉）；
- **权重共享的归纳偏置**：Loop Block 参数必须对所有深度都有用 → 学到更泛化的表示；
- **隐式自蒸馏**：深层 loop 通过共享参数充当"老师"，把浅层表示往更好的方向拉（相邻 loop 的余弦相似度随训练上升但不到 1）。

训练代价：$C_{entry} + L\cdot C_{loop} + (L+1)\cdot C_{exit}$；推理（$i$ 次 loop）：$C_{entry} + i\cdot C_{loop} + C_{exit}$。

---

## 6. 实验

### 6.1 设置
- **数据集**：Amazon Electronics（3M 交互）、TaobaoAds（25M）、KuaiVideo（13.7M）、**InHouse**（阿里某电商平台 9 天日志，6.1M 交互，含短序列 50 + **长序列 1024**）；
- **Baseline**：DNN 类（DLRM / DIN / DCNv2 / Wukong）、Transformer 特征交互（DHEN / AutoInt / HiFormer）、序列+特征统一（InterFormer / OneTrans / HSTU / MTGR），以及 **StackCTR(3)**：把 Loop Block 换成 3 层独立参数、与 LoopCTR(3/3) **iso-FLOPs** 的对照；
- **训练**：8×H20，AdamW lr=1e-3，batch 2048，emb dim 64；MoE 默认 2/4。
- 记号 **LoopCTR(i/L)**：训练 $L$ 次 loop，推理 $i$ 次。

### 6.2 主结果（AUC）

| 方法 | Amazon | TaobaoAds | KuaiVideo | InHouse |
|---|---|---|---|---|
| 最强 baseline | .8689 (OneTrans) | .6430 (AutoInt) | .7430 (DIN) | .7000 (DIN) |
| StackCTR(3) | .8690 | .6403 | .7423 | .6999 |
| **LoopCTR(0/3)** | .8715 | .6439 | .7448 | **.7007** |
| **LoopCTR(1/3)** | **.8728** | **.6441** | **.7450** | **.7007** |
| LoopCTR(3/3) | .8726 | .6436 | .7448 | .7002 |
| *Oracle（逐样本选最优深度）* | *.8858* | *.6672* | *.7588* | *.7195* |

关键结论：

1. **SOTA**：LoopCTR 在 4 个数据集的 AUC/NE 上包揽前列，Amazon 上比 OneTrans 高 0.0039 AUC；
2. **Zero-loop 已超所有 baseline**：InHouse 上 LoopCTR(0/3) 仅 **13.38M FLOPs / 9.26ms**，对比 HSTU 2150M / 775.72ms（**160× FLOPs、84× 延迟**）、OneTrans 417.97M / 494.58ms；
3. **共享参数 > 堆叠层**：iso-FLOPs 下 LoopCTR(3/3) 在 4 个数据集 AUC 都优于 StackCTR(3)，且参数更少（Amazon 1.27M vs 1.95M）；TaobaoAds 的 GAUC 上 StackCTR 略好；
4. **Oracle 余量大**：逐样本选最优 loop 深度可额外带来 0.013–0.023 AUC（相对最佳实际值），相对 $L=0$ 总共 0.02–0.04 AUC。

### 6.3 Loop Scaling（RQ2）
扫描 $L\in\{0,1,2,3\}$ × $i\in\{0,1,2,3\}$：

- **训练 loop 越多，实际效果越好**：Amazon 0.8662→0.8728，InHouse 0.6966→0.7007；loss landscape 显示 $L$ 越大极小值越**宽、平**，泛化越好；
- **推理 loop 边际递减**：第一个 loop 收益最大，之后持平甚至轻微下降（Amazon L=3：.8715→.8728→.8728→.8726）；推理 $i>L$ 时（外推）效果下降；
- **反直觉：训练 loop 越少，Oracle 天花板越高**：Amazon Oracle L=1/2/3 = .8885/.8865/.8858；InHouse .7306 vs .7195。解释：多 loop 训练得到平坦极小值，但也让各深度表示**同质化**；少 loop 训练的极小值更尖锐，但深度间表示更多样，留给逐样本自适应选择的空间更大。

### 6.4 消融（RQ3，Amazon / KuaiVideo，LoopCTR(3/3)）
四个组件都有正贡献：
- Amazon 上 **HCR 最关键**（去掉 AUC −0.0201）；
- KuaiVideo 上 **MoE 最关键**（−0.0060），序列模式更丰富的数据更依赖参数容量；
- 过程监督（PS）和异构特征投影（MP）在两个数据集上都稳定有效。

### 6.5 其它分析
- **Oracle 最优深度分布呈双峰**：loop 0 占 36.8%、loop 3 占 30.9%、loop 1/2 占 17.2%/15.0%。一部分样本 Entry 就够了，多跑反而变差（类比 reasoning 模型的 **overthinking**），另一部分需要完整迭代 → 天然适合 **adaptive inference**；
- **MoE 超参**：2/4 最佳；1/4（无路由多样性）、4/4（无稀疏）都更差；总专家数增至 5 不再提升；
- **专家路由随 loop 变化**：Attention MoE 在第 1 次迭代集中于专家 2、4（44.0% / 42.6%），第 3 次接近均匀（22–28%）；FFN MoE 一开始就更均衡但仍随迭代漂移 → 共享层并非重复同一计算；
- **HCR 系数**在 attention / FFN 子层分布不同，且随 loop 迭代变化，验证 HCR 实现了"隐式 loop-aware"。

### 6.6 效率
- FLOPs 与延迟随推理 loop 数**线性增长**，便于按延迟预算选配置；
- 全部 Loop 迭代共用参数，非零 loop 时参数量与 $L$ 无关；zero-loop 部署时可删除 Loop Block，参数更少。
- 注意：在小的公开数据集上，LoopCTR 的 FLOPs 并不低（Amazon 0/3 为 62.61M，比 OneTrans 20.88M 还高），**zero-loop 的效率优势主要在 InHouse（长序列 + Q-Former 压缩 + 分组 attention）上体现**。

---

## 7. 与 Looped Transformer 相关工作的关系

- Universal Transformer（+ ACT）、Looped Transformer 的图灵完备性/长度泛化、MoEUT、LoopLM、ETD 等都来自 NLP，且**推理时必须跑多次 loop**；
- LoopCTR 的差异：① 首次把 loop scaling 引入 CTR；② sandwich + HCR + MoE 解决共享层表达力问题；③ 过程监督把多 loop 的收益"烧进"共享参数，**推理可以零 loop**，解决了延迟问题。

---
