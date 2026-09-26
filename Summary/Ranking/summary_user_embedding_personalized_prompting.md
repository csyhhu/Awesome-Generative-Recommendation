# User Embedding Model for Personalized Language Prompting：用用户历史嵌入生成个性化 Soft Prompt

> 论文：[User Embedding Model for Personalized Language Prompting](https://arxiv.org/abs/2401.04858)
>
> arXiv：2401.04858
>
> 作者：Sumanth Doddapaneni, Krishna Sayana, Ambarish Jash, Sukhdeep Sodhi, Dima Kuzmin（Google Research）
>
> 方向：用户建模、长历史压缩、个性化语言模型、Soft Prompt

---

## 一、论文要解决的问题

当语言模型用于个性化推荐时，最直接的做法是把用户历史转成文本并拼接到任务指令前。但一条电影记录可能包含标题、类型、评分和较长描述，几十条历史就会产生上万 token，带来三个问题：

- 输入容易超过普通语言模型的上下文窗口；
- Self-Attention 成本随文本长度二次增长；
- 为了控制成本而只检索少量历史，可能丢失长期偏好。

本文提出 **User Embedding Module（UEM）**：先把每条历史压缩成语义向量，再把整段历史映射成一组与语言模型 embedding 维度相同的用户 soft-prompt tokens。语言模型不再读取每条电影的完整文本，而是读取这些连续向量，并与任务级 soft prompt 和文本指令共同完成偏好预测。

论文真正验证的任务不是 item retrieval 或 next-item recommendation，而是：**根据同一用户的历史电影与评分，生成该用户最喜欢和最不喜欢的电影类型**。因此它更接近“用户画像/偏好总结”。

### 对方法流程的直观理解

可以把本文理解成下面的流程：

```text
用户历史中的多部电影
  └─ 每部电影的标题与类型、评分、描述
       ↓ 分别通过 Sentence-T5 编码并拼接
  每部电影一个复合 embedding
       ↓ UEM Transformer 建模电影之间的关系
  每部电影一个个性化 user soft token
       ↓ 与 20 个任务级 soft-prompt tokens、文本任务指令拼接
  Flan-T5
       ↓
  生成用户喜欢和不喜欢的电影类型
```

这个理解需要补充三点边界：

1. 本文使用电影数据验证方法，但目标是**电影类型偏好理解**，没有直接生成或排序具体电影，因此严格来说不是完整的 movie recommendation。
2. 每部电影不是只对一段文本做一次编码，而是分别编码“标题与类型”“评分”“描述”三部分，再拼接为该电影的输入表示。
3. 额外加入的 20 个 tokens 是所有用户共享、训练得到的**任务级 soft prompt**；用户个性化信息来自 UEM 根据历史生成的 `p` 个 user soft tokens，其中 `p` 是输入的历史电影数。

---

## 二、核心方法

### 2.1 总体输入输出

给定：

- 用户历史 $H = \{h_i\}_{i=1}^{p}$；
- 文本任务指令 $X$，其 LM embedding 为 $X_e \in \mathbb{R}^{n \times e}$；
- 目标文本 $Y$，描述用户喜欢和不喜欢的 genres。

UEM 根据用户历史生成个性化 soft prompt：

$$
Pr_{UEM}(U) \in \mathbb{R}^{p \times e}
$$

模型输入为：

$$
[P_e; Pr_{UEM}(U); X_e] \in \mathbb{R}^{(k+p+n) \times e}
$$

其中 $P_e \in \mathbb{R}^{k \times e}$ 是与用户无关、可训练的任务级 soft prompt。训练目标为：

$$
\max_{\theta, UEM, P_e} \log P_{\theta}\left(Y \mid [P_e; Pr_{UEM}(U); X_e]\right)
$$

论文使用 `k=20` 个任务级 soft-prompt tokens，并联合更新 UEM、任务 soft prompt 和 Flan-T5 参数。

### 2.2 单条历史如何编码

每部电影被拆成三段文本：

1. **标题与类型**：`The movie {title} is listed with genres {genres}`；
2. **评分**：`The movie is rated with {rating} stars`；
3. **描述**：电影剧情简介。

三段文本分别通过 Sentence-T5 得到 embedding，再沿特征维拼接：

$$
u_i = [u_i^{title+genre}; u_i^{rating}; u_i^{description}] \in \mathbb{R}^{3s}
$$

于是长度为 $p$ 的用户历史被表示为：

$$
U = [u_1, u_2, \ldots, u_p] \in \mathbb{R}^{p \times 3s}
$$

### 2.3 User Embedding Module

UEM 是一个小型 Transformer，负责在历史 item 之间建模关系。论文默认配置为：

| 配置 | 数值 |
|---|---:|
| Transformer 层数 | 3 |
| Attention heads | 12 |
| Hidden dimension | 768 |
| MLP dimension | 2048 |
| 新增参数量 | 约 65M |

UEM 输出再经过线性投影，与语言模型的 embedding 维度 $e$ 对齐。每条历史最终对应一个 LM soft-prompt token：

```text
电影标题/类型 ─┐
电影评分       ├─ Sentence-T5 embeddings ─ concat ─┐
电影描述       ┘                                    │
                                                     ▼
历史中的 p 部电影 ─────────────────────────────── UEM Transformer
                                                     │
                                                     ▼
                                           p 个用户 soft tokens
                                                     │
20 个任务 soft tokens + 用户 soft tokens + 文本指令 ─┤
                                                     ▼
                                                  Flan-T5
                                                     │
                                                     ▼
                                   喜欢/不喜欢的电影类型文本
```

需要注意：这里的“压缩”是**按 item 压缩**，不是把任意长度历史压成固定数量的 tokens。`p` 条历史仍生成 `p` 个用户 tokens；主要收益来自把每条可能占几百文本 token 的记录压成一个连续 token。

---

## 三、数据集与任务构造

论文将 MovieLens 与 Rotten Tomatoes 的电影描述合并，并过滤历史少于 20 条的用户：

| 统计项 | 数值 |
|---|---:|
| Reviews | 14.4M |
| Movies | 8.2K |
| Users | 127K |
| Train / Dev / Test users | 117K / 5K / 5K |

标签由用户历史聚合得到：

- 只考虑至少出现过 3 次的 genre；
- 平均评分高于 3.5 的 genre 中，取最偏好的 3 个；
- 平均评分低于 3.0 的 genre 中，取最不偏好的 3 个；
- 输出自然语言模板：`The user likes ... and doesn't like ...`。

虽然模型以 text-to-text 方式训练，最终评估会通过 verbalizer 抽取 genre，并按 19 个类别做 multi-label classification。由于类别分布偏斜，论文报告 weighted precision、recall 和 F1，而不是 BLEU/ROUGE。

这个任务的标签直接由输入历史中的 genre 和 rating 聚合而来，所以实验主要衡量模型能否对长历史进行**压缩、统计和偏好归纳**，不能直接证明其 next-item 推荐或候选排序能力。

---

## 四、实验设置

- Backbone：Flan-T5 Base / Large；另比较 T5 1.1、LM-adapted T5 和 LongT5。
- 训练步数：10K。
- Batch size：128。
- Text-history learning rate：`1e-2`。
- Embedding-history learning rate：`5e-3`。
- 默认 UEM：3 层、65M 参数。
- Text baseline：把历史字段对应的原始文本直接拼接给 LM。
- Counting baseline：统计整段历史中出现最频繁的 3 个 genres。

LongT5 使用 16K-token 输入处理 50 条文本历史。论文报告它需要 TPU v3-32，而普通 Flan-T5 可在 TPU v3-8 上训练；LongT5 训练时间约为同规模 Flan-T5 的 4 倍，Serving latency 也更高。

---

## 五、主要结果

### 5.1 不同历史输入方式

| 方法 | 历史条数 | Base F1 | Large F1 |
|---|---:|---:|---:|
| Counting baseline | 全历史 | 0.192 | 0.192 |
| Text history | 5 | 0.273 | 0.261 |
| Embedding history | 5 | 0.252 | 0.215 |
| LongT5 text history | 50 | **0.529** | **0.557** |
| Embedding history | 50 | 0.396 | 0.381 |
| Embedding history | 100 | 0.404 | 0.444 |

结论需要分两层理解：

1. **在普通 Flan-T5 的上下文预算下，UEM 能注入更多历史，并明显优于只拼接 5 条文本历史。** 从 5 条 text history 到 100 条 embedding history，Base/Large 的 F1 分别提高 `0.131/0.183`。
2. **如果允许更长上下文和更大计算预算，LongT5 仍然明显更强。** LongT5 读取 50 条完整文本时，F1 比读取 100 条历史的 UEM 高 `0.125/0.113`。

论文摘要和正文声称相对 text-based prompting baseline 提升 `0.21/0.25` F1，但表中 `Text Hist. 5` 到 `Emb. Hist. 100` 的实际差值是 `0.131/0.183`。`0.212/0.252` 恰好对应 `Emb. Hist. 100` 相对 `Counting baseline` 的差值。因此，论文的文字表述与表格数据存在口径不一致。

### 5.2 历史长度消融

| Embedding history length | Base F1 | Large F1 |
|---:|---:|---:|
| 5 | 0.273 | 0.261 |
| 20 | 0.275 | 0.281 |
| 30 | 0.337 | 0.367 |
| 50 | 0.396 | 0.381 |
| 100 | **0.404** | **0.444** |

按附录表格，Base 和 Large 的 F1 都随历史长度单调上升；Base 在 5 到 20 条之间提升很小，主要收益出现在 30 条之后。

论文估计 50 条完整文本历史约为 16K tokens，而 UEM 只向 LM 注入 50 个用户 tokens，因此单条历史的压缩比很高。

需要留意主结果表与附录在 `Embedding history = 5` 上存在另一处不一致：主表将 `0.252/0.215` 标为 Embedding Hist. 5，而附录历史长度表给出 `0.273/0.261`，后者与主表的 Text Hist. 5 完全相同。论文没有解释该差异。

### 5.3 Backbone 选择

固定使用 50 条 embedding history：

| LM | Base F1 | Large F1 |
|---|---:|---:|
| T5 1.1 | 0.208 | 0.267 |
| T5 LM-adapted | 0.338 | 0.378 |
| Flan-T5 | **0.396** | **0.381** |

Flan-T5 整体最好，作者认为原因是任务本身采用自然语言指令格式，instruction-tuned backbone 更容易适配。

### 5.4 UEM 深度

固定使用 50 条历史：

| UEM 层数 | Base F1 | Large F1 |
|---:|---:|---:|
| 1 | 0.346 | 0.347 |
| 2 | 0.384 | 0.365 |
| 3 | **0.396** | **0.381** |

更深的 UEM 持续提升效果，说明历史 item 之间的交互建模有价值。但论文只测试到 3 层，也没有比较更轻量的 pooling、Perceiver-style latent compression 或线性序列模块，因此不能判断 65M Transformer 是否是最佳性价比方案。

---

## 六、效率分析

### 6.1 相对直接文本拼接的优势

假设每条历史平均展开成 $m$ 个文本 tokens，直接拼接的 LM 输入长度约为 $pm+n$；UEM 方案给 LM 的长度约为 $p+k+n$。当电影描述较长、即 $m \gg 1$ 时，LM 主干的注意力和激活开销会显著下降。

此外，Sentence-T5 的 item embeddings 可以预计算并缓存，用户历史更新时只需要组织对应向量。这使该方法自然适合多模态扩展：文本、图像和音频可以各自编码，再由 UEM 融合成 LM soft prompt。

### 6.2 它不是固定长度、线性复杂度的历史压缩

论文将 UEM 描述为比直接文本输入更便宜，这个方向成立，但复杂度需要更精确地理解：

- UEM 对 `p` 个历史 tokens 自身执行 Transformer self-attention，仍有 $O(p^2)$ 计算；
- UEM 输出数量等于历史条数 `p`，LM 输入长度仍随历史线性增长；
- 论文没有报告端到端 latency、吞吐、显存或 FLOPs，只用 TPU 规格和训练时长定性比较 LongT5；
- 新增 65M 参数对小于 1B 的 backbone 并非可以忽略。

因此，该方法消除的是“每个 item 的长文本展开”，并没有解决无限长用户历史。若要面向数千到数万行为，仍需要 latent bottleneck、层级汇聚、检索或 token merging。

---

## 七、优点

1. **接口简单**：只要 LM 支持 `inputs_embeds`，就可以把个性化历史作为 soft prompt 注入，不需要修改语言建模目标。
2. **历史表示可学习**：UEM 与 LM 联合训练，用户向量不是通用语义 embedding 的简单平均，而是针对下游偏好任务进行上下文化。
3. **避免长文本展开**：将一条电影的标题、评分和描述压成一个 token，显著扩大普通上下文窗口能覆盖的历史条数。
4. **模态扩展自然**：不同模态可先用专用 encoder 转成 embedding，再由 UEM 统一融合。
5. **实验包含关键消融**：覆盖历史长度、LM 初始化和 UEM 深度，并提供强但昂贵的 LongT5 对照。

---

## 八、局限与风险

1. **任务范围较窄**：只验证 genre preference summarization，没有测试 rating prediction、CTR、ranking、retrieval 或 next-item generation。
2. **标签与输入高度同源**：liked/disliked genres 由输入中的 genre 和 rating 直接聚合，任务可能主要考察统计汇总，而不是更开放的用户理解。
3. **强基线仍占优**：LongT5-50 的 F1 明显高于 UEM-100，说明 embedding 压缩存在信息损失，尤其是 Sentence-T5 预编码可能丢失任务相关细节。
4. **缺少同预算比较**：没有给出 UEM 与 LongT5 的严格 FLOPs、延迟、显存或成本归一化曲线，难以量化性能/效率 Pareto frontier。
5. **不是固定容量表示**：一条历史对应一个 soft token，超长历史仍会遇到二次注意力成本和上下文上限。
6. **仅使用小型 LM**：实验 backbone 均小于 1B 参数，结论是否适用于现代大模型或更强 instruction models 尚未验证。
7. **结果表存在口径问题**：两处 F1 数字与正文或附录不一致，削弱了对增益幅度的信心。
8. **缺少用户时间建模**：方法描述没有显式时间间隔、行为顺序衰减或兴趣漂移机制，也没有时间切分评估。
9. **缺少部署验证**：没有在线实验、缓存更新策略、冷启动分析以及用户 embedding freshness 研究。

---

## 九、与后续用户建模工作的关系

这篇论文可以看作“**行为历史作为 LM soft prompt**”的早期直接实现：

```text
原始历史文本
    ↓ 预训练语义编码器
逐 item 连续表示
    ↓ 小型历史 Transformer
个性化 soft-prompt tokens
    ↓ 注入语言模型
自然语言偏好预测
```

它与后续生成式用户表示方法的共同点是：都试图把长行为历史转换成可被下游模型重复消费的用户表示。区别在于：

- 本文输出与历史等长的连续 soft tokens，并针对单个偏好生成任务端到端训练；
- 更后续的用户 token 方法往往引入固定容量 latent tokens、离散 Semantic IDs、多目标未来行为预测和异步缓存，以提高跨任务复用与工业服务能力；
- 本文保留较强的语言语义接口，适合连接文本生成任务，但还没有证明表示能够脱离当前 LM 和任务独立复用。

---

## 十、核心结论

本文的核心价值不是证明 embedding 压缩比完整文本更准确，而是展示了一条实用的折中路线：**先把每条用户历史压成语义向量，再用可训练的历史模块把这些向量转换成个性化 soft prompts，从而让普通上下文长度的语言模型读取更多历史。**

实验支持三个相对稳健的结论：

- 历史从 5 条扩大到 100 条时，偏好分类 F1 整体提升；
- instruction-tuned Flan-T5 比普通或仅 LM-adapted T5 更适合该任务；
- 更深的 UEM 能更好地建模历史 item 间关系。

但证据边界也很明确：LongT5 读取完整文本仍然效果最好，UEM 尚未形成固定长度用户表示，实验任务与真实推荐目标有明显距离，并且部分结果数字存在内部不一致。更准确的定位是：**UEM 证明了 personalized soft prompting 可以作为长历史文本拼接的低成本替代方案，但尚未证明它是长历史推荐建模的最终方案。**
