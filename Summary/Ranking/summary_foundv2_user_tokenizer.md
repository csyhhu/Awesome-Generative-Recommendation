# FOUNDv2: Learning Unified User Quantized Tokenizers for User Representation

- **论文**：[FOUNDv2: Learning Unified User Quantized Tokenizers for User Representation](https://arxiv.org/abs/2508.00956)
- **arXiv ID**：2508.00956
- **会议**：KDD 2026
- **作者**：Chuan He、Yang Chen、Bin Dou 等（蚂蚁集团、浙江大学、浙江工业大学）
- **代码**：[chuanhe1999/FOUNDv2](https://github.com/chuanhe1999/FOUNDv2)
- **关键词**：用户建模、离散用户表示、RQ-VAE、多源融合、长序列、通用用户表征

---

## 1. 总结

FOUNDv2 先用 Qwen3 Embedding 将账单、App、小程序、搜索和结构化属性等异构用户数据压成连续向量，再用带有“共享码本 + 来源专属码本”的 Multi-View RQ-VAE 将向量离散化为用户 Token；随后通过 token、时间窗口和未来行为语义三种粒度的预训练目标，使这些 Token 不只能够重建历史，还能够预测未来行为。

它的核心价值不是把用户唯一标识成一个离散 ID，而是把长期、多源用户行为转成一串可复用、可增量更新、存储成本较低的离散表征。论文报告相对上一代 FOUND 将 20M 用户数据的表示存储从 240 GB 降到 8.2 GB、训练时间从约 7 小时降到约 2 小时，并已用于支付宝支付和广告场景。

| Method | Input | Training Methods | Output | How to use | Intermediate Performance Validation | Resource Consumption |
|---|---|---|---|---|---|---|
| **FOUNDv2** | 多源用户历史：PayBill、SPM、MiniProgram、App、Search 及 Tabular attributes。文本来源由 Qwen3-Embedding-0.6B 编为 1024-D 向量，表格来源由 MLP 编码；alignment 消融报告约 1,100/2,200 个 input-trajectory Token，但未定义其 token 类型。 | 三阶段：`多源数据 → Qwen/MLP embedding`；`embedding → shared + source-specific MRQ-VAE`，以 reconstruction 和 RQ loss 学习离散码；随后在用户 Token 序列上联合优化 token-level next-code alignment、window-level periodic contrastive alignment 和 semantic-level future-behavior alignment。论文未明确三阶段是否联合反传。 | 主要输出是按来源/行为生成的离散 user token IDs：正式配置为 4 层 shared code + 2 层 source-specific code，并对少于 0.5% 的 collision 添加 special token。它形成可增量追加的长期用户 Token 序列；每用户最终缓存长度未披露。 | 将 Token ID lookup 为 embedding，作为额外 user-side feature 接入 One4all、MSDP、MaskNet、PEPNet 或广告召回模型；离线缓存用户 Token，并支持只编码新增行为，无需重算完整历史。 | 主要是四类约 50 万样本任务上的 **linear probe AUC/KS**，用于验证预训练表示的跨任务质量；另有 MRQ-VAE/对齐任务消融、codebook utilization 和 t-SNE。推荐实验用 MaskNet 的 AUC/HR 做 downstream validation。论文没有报告 tokenizer-intrinsic retrieval、reconstruction MSE、perplexity 或 collision bucket quality。 | 预训练数据约 20M 用户/记录，alignment 实验报告约 1.1K/2.2K input tokens（token 类型未定义）；16 张 100 GB PPU-810E，20M 数据训练约 2 小时。90 天表示存储 8.2 GB，即约 410 bytes/user，线性外推 1B 用户约 410 GB 单副本。batch=256 时 60/90/180 天分别约 3/5/10 GB GPU memory。线上每天刷新约 400M–500M DAU；每用户 code 数、QPS、延迟、复制与 KV 元数据均未报告。 |

### Input

FOUNDv2 的输入不是已经构造好的 item SID，而是原始的多源用户信息。论文列出六类来源，其中行为文本和搜索文本通过带 Query 的 Qwen3 Embedding 编码，结构化属性通过 MLP adapter 编码。每个来源先独立形成连续向量，再进入共同的量化框架。

从 Interest Cache 视角看，输入存在两个粒度：

- **Tokenizer 构建阶段**：一条来源数据或行为窗口对应一个 1024-D 连续 embedding。
- **用户预训练阶段**：多个来源或时间窗口产生的离散码可按时间拼接；alignment 消融表另外报告约 1,100/2,200 个 `Token Number`，但没有定义这里是 Qwen 输入 subwords、离散 user codes，还是其他 trajectory-token 口径。

不能把 `≈1,100/≈2,200` 直接理解成每个用户在线缓存的离散 code 数。论文同时报告 20M 用户的 90 天 Token 表示只占 8.2 GB，即约 410 bytes/user；如果每用户真有 2,200 个、每层 1024 entries 的 codes，即使做理论上的 10-bit packing 也要约 2,750 bytes/user，明显不一致。因此更可信的判断是：1.1K/2.2K 是训练输入轨迹长度或未解释的实验口径，而不是最终缓存长度。

### Training Methods

训练链路可拆成三个阶段：

```text
Stage A: Raw multi-source data
         → frozen or pretrained Qwen3 Embedding / MLP
         → source embeddings

Stage B: Source embeddings
         → MRQ-VAE shared codebooks
         → MRQ-VAE source-specific codebooks
         → reconstruction loss + residual quantization loss

Stage C: Historical user-token sequence
         → Transformer
         → token / window / future-semantic alignment losses
```

与 TokenMinds 的差异是，FOUNDv2 不以自回归生成未来 item SID 作为唯一训练目标。它先学习“如何压缩当前多源历史”，再用三种 alignment 让压缩后的用户 Token 对未来具有预测性。论文没有清楚披露 Stage B 与 Stage C 之间是冻结、顺序微调还是联合训练，这是复现时最大的训练协议缺口。

### Output

#### 对当前理解的校正

“把多场景文本行为 embedding 成 token，再取 RQ-VAE 多层 prefix codebook 作为 user token”已经接近论文主线，但更精确的描述是：

1. Qwen3 首先把一个来源/时间切片中的长文本压成**一个 1024-D 连续 embedding**；表格数据则由 MLP 得到连续 embedding。
2. MRQ-VAE encoder 把该向量映射为 latent \(z\)。随后 residual quantizer 在每一层 codebook 中选择最近的 code vector。
3. 真正保存为 user token 的是每层选中的 **code index**，例如 `[12, 87, 4, 66, 31, 8]`，不是整张 codebook，也不是 MRQ-VAE encoder 的多层 hidden states。
4. 前几层是 shared codebook，后几层是 source-specific codebook。它们具有“先公共语义、再残差细化”的层次性，但论文没有像树状 item SID 那样严格证明每个前缀对应稳定、可解释的语义簇，因此称为 residual code path 比称为 prefix SID 更准确。
5. 下游必须把离散 ID 转回连续向量才能进入 MaskNet/PEPNet 等网络。`embedding lookup → pooling/sequence modeling → downstream network` 是合理实现，但论文没有明确说明下游复用 MRQ-VAE codebook vectors，还是为每个任务重新训练独立 embedding table。

#### Input 是否为纯文本、每个 element 是否为单个事件

Input 不是纯文本，而是两类数据：

- **文本化行为与搜索**：PayBill、SPM、MiniProgram、App 等行为被整理成文本，Search 本身也是文本，随后由 Qwen3 Embedding 编码。
- **统计/结构化特征**：Tabular attributes 以 \(T\in\mathbb R^{N\times F\times D}\) 表示，不转成自然语言，而是直接经过 MLP adapter。

论文附录中的 SPM prompt 一次包含多个页面或组件行为及其发生次数，例如某组件访问 2 次、另一个页面访问 3 次。因此更合理的数据单元是**一个来源在某个时间窗口内的聚合行为 slice**，而不是“一个 element 严格等于一次原子点击”。生产章节进一步说明系统每天只处理当天活动，暗示部署粒度可能是 `user × day × modality`；但论文没有精确定义训练阶段每个 slice 的窗口边界。

所以正确的数据流不是：

```text
每个历史事件 → 一个 embedding → embedding sequence 整体进入 MRQ-VAE
```

而更接近：

```text
Bill 当日多条行为文本 ──Qwen──→ 1 个 Bill embedding ──MRQ-VAE──→ 4/6 codes
Search 当日多个 query  ──Qwen──→ 1 个 Search embedding ─MRQ-VAE──→ 4/6 codes
Tabular 当日统计特征  ──MLP───→ 1 个 Tabular embedding ─MRQ-VAE─→ 4/6 codes

不同来源、不同日期的 code blocks
  → 按时间拼接
  → 长期 User Token Sequence
```

因此，长期序列中的一个 element 是一个 **RQ code ID**；连续的 4 或 6 个 elements 才共同表示一个来源/时间 slice。它不是原始历史事件和 Token 的一一映射。

#### 普通 RQ-VAE 与 MRQ-VAE 的区别

两者都采用多层 residual quantization。普通 RQ-VAE 的“multi-layer”含义是：第一层近似原 latent，后续每层继续量化上一层尚未解释的 residual：

\[
c^l=\arg\min_k\|r^l-v_k^l\|_2^2,
\qquad
r^{l+1}=r^l-v_{c^l}^l.
\]

如果所有来源都使用同一条普通 RQ-VAE 路径，那么 Bill、Search、App 等输入都在相同的 \(L\) 层通用码本中选码。它能够压缩向量，但没有显式区分“跨来源公共信息”和“来源独有信息”。

MRQ-VAE 的 `M` 表示 **Multi-View**，不是简单地“比 RQ-VAE 多几层”。它把量化路径拆成两段：

| 维度 | 普通 RQ-VAE | FOUNDv2 MRQ-VAE |
|---|---|---|
| 多层 residual quantization | 有 | 有 |
| Shared codebooks | 所有层通常统一使用 | 前 \(L_c\) 层供所有来源共同使用 |
| Source-specific codebooks | 无 | 后 \(L_u\) 层按 Bill/Search/App 等来源分别设置 |
| Decoder | 通常一个通用 decoder | 每个来源使用 source-specific MLP decoder |
| 学习目标 | 重建输入并量化 residual | 同时提取跨来源共性并保留来源特有 residual |

正式配置是 4 层 shared codebooks，加上每个来源 2 层 specific codebooks。每层有 1024 个 256-D code vectors。一个来源 slice 只走其中 6 层：

```text
source embedding
  → Shared-1 → Shared-2 → Shared-3 → Shared-4
  → 当前 source 的 Specific-1 → Specific-2
  → 6 个 code indices
```

Shared 的含义是多个来源从同一套字典中选择，而不是所有来源必须选择相同 code index。Bill 和 Search 可以在 `Shared-1` 分别选择 12 和 731，只是它们引用的是同一张 `Shared-1` codebook。

#### Encoder、Quantizer 与 Decoder 的输入输出：一个具体例子

这里的 “Encoder/Decoder” 不能按 TokenMinds 的自回归 Encoder-Decoder 理解。FOUNDv2 实际有两个不同层次的 Encoder，以及一个只在量化训练中负责重建的 Decoder：

```text
Prompted source text
  → Qwen3-Embedding Encoder
  → 1024-D source embedding
  → MRQ-VAE Encoder
  → latent z
  → residual quantizer
  → code IDs + quantized vector z_hat
  → source-specific MLP Decoder
  → reconstructed 1024-D source embedding
```

假设用户 Alice 在 Day 1 的 Bill slice 中有以下聚合行为。这个内容是为解释数据流而构造的简化例子；论文附录的真实 SPM prompt 同样在一条 Source Data 中放入多个页面/组件行为及访问次数。

```text
午餐外卖 35 元，支付成功
便利店消费 18 元，支付成功
电影票 80 元，支付成功
```

**第一步：Qwen3-Embedding Encoder 的输入。** 系统把固定任务 Query、当前来源的聚合数据和 `[EOS]` 拼成一条文本序列：

```text
Query:
  提取支付宝用户数据的文本特征

Source Data:
  [Day 1][Bill]
  午餐外卖 35 元，支付成功；
  便利店消费 18 元，支付成功；
  电影票 80 元，支付成功。

Serialized Qwen input:
  [Query tokens, Source-Data tokens, EOS]
```

Qwen3-Embedding-0.6B 输出最后一层 `[EOS]` hidden state，而不是输出离散 code：

```text
[Query, Bill source data, EOS]
  → Qwen3-Embedding-0.6B
  → h_bill ∈ R^1024
```

对 Tabular 来源没有这条文本 Prompt；原始结构化属性直接通过 MLP 得到同维连续 embedding。

**第二步：MRQ-VAE Encoder 与 Quantizer 的输入。** `h_bill` 先经过 pooling/共享 MLP 得到维度对齐后的 `h_bar_bill`，后者才是 MRQ-VAE Encoder 的直接输入；Encoder 再将其映射到与 256-D code vectors 对齐的 latent `z_bill`：

```text
Alignment input:        h_bill ∈ R^1024
MRQ-VAE Encoder input:  h_bar_bill = M(P(h_bill)) ∈ R^d_c
MRQ-VAE Encoder output: z_bill ∈ latent space
Quantizer input:        r^0 = z_bill
```

论文没有单独列出 `d_c` 和 MRQ-VAE Encoder 的结构；正式配置明确披露的是每个 code vector 为 256-D。

Residual Quantizer 不是读取文本或 Token ID sequence，而是逐层量化当前 residual。下面的数值是为解释选码过程而构造的示例，不是论文公开的真实 code：

| Quantization layer | 当前量化对象 | 最近的 code entry | 产生的 Token ID | 含义 |
|---|---|---:|---|---|
| Shared-1 | \(r^0=z_{bill}\) | 12 | `<S1:12>` | 第一层跨来源粗粒度公共模式 |
| Shared-2 | \(r^1=r^0-v_{12}^{S1}\) | 87 | `<S2:87>` | 对公共 residual 的进一步近似 |
| Shared-3 | \(r^2\) | 4 | `<S3:4>` | 更细的公共 residual |
| Shared-4 | \(r^3\) | 66 | `<S4:66>` | 最后一层公共 residual |
| Bill-specific-1 | \(r^4\) | 31 | `<B1:31>` | Bill 来源特有模式 |
| Bill-specific-2 | \(r^5\) | 8 | `<B2:8>` | Bill 特有 residual 的进一步细化 |

Alice 的 Day 1 Bill User Token block 就是：

```text
[<S1:12>, <S2:87>, <S3:4>, <S4:66>, <B1:31>, <B2:8>]
```

对应的量化连续向量是六个 code vectors 的和：

\[
\hat z_{bill}
=v_{12}^{S1}+v_{87}^{S2}+v_{4}^{S3}+v_{66}^{S4}
+v_{31}^{B1}+v_{8}^{B2}.
\]

**第三步：source-specific MLP Decoder 的输入与监督目标。** Bill Decoder 接收的不是原始文本，也不是六个 code IDs 的自回归序列，而是这些 code IDs 对应 vectors 的和 `z_hat_bill`：

```text
Decoder input:
  z_hat_bill
  = v[S1:12] + v[S2:87] + v[S3:4] + v[S4:66]
    + v[B1:31] + v[B2:8]                         # 256-D

Bill-specific MLP Decoder output:
  h_tilde_bill ∈ R^1024

Reconstruction target:
  h_bill ∈ R^1024                                # Qwen3 source embedding

Loss:
  L_re = ||h_tilde_bill - h_bill||_2^2
```

因此，FOUNDv2 Decoder 的功能是检查离散 codes 是否保留了足够信息来重建来源 embedding。它没有 teacher forcing、没有 shifted target，也不负责生成未来 behavior codes；未来预测能力来自后续 token/window/semantic alignment tasks。完成 Tokenizer 训练后，线上生成和缓存 code IDs 只需要 Qwen/MLP Encoder、MRQ-VAE Encoder 与 Quantizer，重建 Decoder 可以不位于下游 serving path。

这里必须保留 codebook/layer namespace。`<S1:12>` 与 `<S2:12>` 虽然 entry index 都是 12，但来自不同码本、对应不同向量，不能只存成一个无上下文的整数 `12`。工程上可以存 `(codebook_id, entry_id)`，也可以给每张码本分配 offset 后编码成全局整数；论文没有披露实际序列化方案。

按正式配置推算，逻辑 vocabulary 包含：

\[
4\times1024\ \text{shared symbols}
+6\ \text{sources}\times2\times1024\ \text{specific symbols}
=16{,}384\ \text{symbols},
\]

另外还有处理少量 collision 的 special tokens。这是由论文配置推算出的命名空间大小，不是论文明确报告的线上 vocabulary 数字。

如果 Alice 同一天还有 Search slice，可能得到另一个 block：

```text
Search Day 1
→ [<S1:731>, <S2:22>, <S3:418>, <S4:91>, <R1:72>, <R2:11>]
```

Day 2 又产生 App block：

```text
App Day 2
→ [<S1:105>, <S2:87>, <S3:9>, <S4:203>, <A1:9>, <A2:52>]
```

最终缓存的 Alice User Token Sequence 是这些 blocks 的时间拼接：

```text
[
  Day1/Bill:   <S1:12>,  <S2:87>, <S3:4>,   <S4:66>,  <B1:31>, <B2:8>,
  Day1/Search: <S1:731>, <S2:22>, <S3:418>, <S4:91>,  <R1:72>, <R2:11>,
  Day2/App:    <S1:105>, <S2:87>, <S3:9>,   <S4:203>, <A1:9>,  <A2:52>,
  ...
]
```

这串 Token 不是 Alice 的唯一用户 ID。每个 block 是对某个来源/时间 slice 的语义量化，完整序列才是 Alice 的长期离散用户表示；不同用户仍可能产生相同 codes。论文称完整 Token 表示的碰撞率低于 0.5%，并在发生碰撞时追加 special token 进行区分。

#### 修正后的整体理解

可以把整体理解写成：

> 一个用户在某个来源、某个时间窗口中的多条原始记录，先被聚合成一个 source slice；该 slice 被编码为一个连续 embedding，再由 MRQ-VAE 量化为一个 4/6-code tuple。多个来源或增量时间窗口的 tuples 可以拼接形成 User Token Sequence。论文没有公开每个用户最终缓存多少个 atomic code IDs；消融表的 1.1K/2.2K 不能直接等同于线上输出长度。

因此，下面三个说法需要避免：

- “每条原子记录都映射成 6 个 Token”：论文证据更支持每个 `source × time slice` 映射成 4/6 个 Token，一个 slice 内可以包含多条行为。
- “6 个 Token 是一个唯一 SID”：它更像一个 residual code tuple，表达该 slice 的量化语义，不保证唯一定位该行为或用户。
- “每个用户最终固定 2,200 Token”：1,100/2,200 只是消融表中的 `Token Number`，token 类型未定义；论文自报的 8.2 GB 存储量与每用户 2,200 个离散 codes 明显不一致。

#### 6 个 codes 如何在 reconstruction 中回到连续空间

这一步论文描述得比较完整。每个 code ID 直接索引其所在码本中的 256-D code vector，再把 6 个向量相加：

```text
6 code IDs
  → lookup 6 pretrained codebook vectors, each 256-D
  → element-wise sum
  → one quantized latent z_hat, 256-D
  → source-specific MLP decoder
  → reconstructed Qwen/source embedding, 1024-D
```

公式为：

\[
\hat z^{(x)}=
\sum_{l=1}^{L_c}v_{c_l}^{l,(S)}+
\sum_{l=L_c+1}^{L_c+L_u}v_{c_l}^{l,(x)}.
\]

这意味着 6 个 codes 不是分别重建 6 份内容，而是共同近似一个 source slice 的 latent。前层解释主要结构，后层逐步补上 residual。

#### User codes 如何变成下游 embedding

这里需要区分论文明确披露的部分和没有披露的部分。

**论文明确披露：**

1. 一个 source slice 的 4/6 个 code IDs 可以 lookup 对应的 256-D codebook vectors，并通过求和恢复一个 quantized source latent。
2. Token-level alignment 使用 Transformer 读取离散 Token 序列，为每个位置产生 contextual hidden state \(h_k\)，并预测下一个 code embedding。
3. Window-level alignment 使用窗口聚合表示 \(h_i^W\)，Semantic-level alignment 使用融合表示 \(e_i^f\)。
4. 下游推荐把 `UserRep=[Code_1,...,Code_n]` 作为 MaskNet 的用户侧特征。

**论文没有披露：**

- 最终线上每个用户缓存的 \(n\) 是多少。
- 下游直接复用 MRQ-VAE 的 256-D codebook vectors，还是训练新的 task-specific embedding table。
- 是否先将每 4/6 个 codes 相加成 source-slice vector，再进行时序建模。
- 使用 mean、last-token、attention、`[CLS]` 或其他 pooling 得到单个 user embedding。
- 最终 user embedding 的维度，以及 MaskNet 是否一定只接收一个 pooled vector。

因此不能把“2,200 个离散 Token 通过某个已知 pooling 变成一个 embedding”写成论文结论。更稳妥的流程是：

```text
N 个缓存 code IDs（N 未披露）
  → codebook 或 task-specific embedding lookup
  → N × d token embeddings
  → Transformer / pooling / feature interaction（具体结构未披露）
  → 下游 user representation 或 prediction score
```

另一种与 reconstruction 路径兼容的工程实现，是先把每个 4/6-code tuple 求和恢复为一个 256-D source-slice vector，再对多个 slice vectors 做时序聚合。论文没有给出足够细节判断实际使用哪一种路径。

#### 单次量化到底输出多少 Token

论文同时出现了三种数量口径：

| 粒度 | Token 数 | 依据与解释 |
|---|---:|---|
| 论文方法示例中的一个来源 | 4 | 示例使用 `2 shared + 2 specific` |
| 正式实验配置中的一个来源 | 6 | Implementation Detail 使用 `4 shared + 2 specific` |
| 多尺度预训练的 trajectory input | ≈1,100 或 ≈2,200 | 消融表只写 `Token Number`，未说明是 Qwen subwords 还是离散 codes；不能据此推断缓存长度 |

若按正式实验的六个数据来源，并假设一次用户快照中六个来源都各量化一次，则理论输出为：

\[
N_{snapshot}=6\ \text{sources}\times(4\ \text{shared}+2\ \text{specific})=36\ \text{tokens}.
\]

如果采用论文示意图和部署附录中的 `2 shared + 2 specific`，则六个来源是：

\[
N_{snapshot}=6\times4=24\ \text{tokens}.
\]

但这两个值都只是“六个来源各产生一次表示”的推算。真实长度还取决于某天有多少活跃来源、每个来源按天聚合一次还是切成多个窗口，以及缺失来源是否跳过。论文没有披露这些数据构造细节。

论文附录明确写道“每天处理当天行为，每个 modality 生成 4 个 discrete tokens，再追加到历史序列”；这与正式实验的 6 code/source 配置不一致。最合理的解释是生产使用了较轻的 4-code 配置，或者论文不同章节沿用了不同版本。不能把单次输出武断地固定为 4 或 6，引用时应同时保留这一口径差异。

#### 为什么会逐渐加长

长度增长不是因为 RQ-VAE 的 prefix 层数每天变深。每个来源切片的量化深度始终固定为 4 或 6 层；增长来自**不断追加新的时间切片**：

\[
H_d=\operatorname{Concat}\left(H_{d-1},\{Q(x_{d,m})\}_{m\in A_d}\right),
\]

其中 \(H_d\) 是第 \(d\) 天结束后的用户 Token history，\(A_d\) 是当天活跃的数据来源，\(Q(x_{d,m})\) 返回该来源当天的 4 或 6 个 code IDs。

例如使用生产章节的 4-code 口径：

```text
Day 1: Bill   → [S1, S8, B3, B9]
       Search → [S2, S7, R4, R6]
       History length = 8

Day 2: App    → [S1, S5, A2, A8]
       History = Day 1 tokens ++ Day 2 tokens
       History length = 12

...

After many days/modalities
       History length grows with retained slices; exact maximum is not reported
```

若每天六个来源都活跃且每个来源输出 4 个码，理论增长上限是约 24 tokens/day；输出 6 个码时则约 36 tokens/day。但这是便于理解的上限示例，不是论文披露的实际日均增长率。

虽然 `6 modalities × 4 codes × 90 days = 2,160` 在数字上接近 2,200，但该解释与论文报告的 8.2 GB 总存储不相容：20M 用户若各保存 2,200 codes，理论最低存储也远大于 8.2 GB。因此不能用这一巧合证明 2,200 是缓存 code 数。

论文没有说明历史序列是否采用 FIFO、固定天数窗口、按重要性裁剪或其他 compaction，也没有给出 append 后的平均/最大 code 数。生产系统不可能真正无限增长；它大概率维持某个 retention window 或压缩单元，但具体策略是论文留下的系统缺口。

#### 最终 Output 的准确表述

FOUNDv2 的核心输出是**固定码数的来源/时间切片表示，按时间累积后形成可变长度用户 Token 序列**：

```text
One source/time slice
  → [shared code IDs, source-specific code IDs]  # 4 or 6 IDs

Long-term user representation
  → [slice_1 codes, slice_2 codes, ..., slice_n codes,
     optional collision code]                    # cached length not reported
```

因此它与 Google Doc 中两类输出有明显区别：

- TokenMinds 输出未来兴趣 SID sequences 和一个 pooled dense embedding。
- LLaTTE 输出固定维度、可缓存的 upstream dense embedding。
- FOUNDv2 输出不断随新行为 append 的离散历史表示；codebook vectors 可以转回连续 embedding，但论文没有把独立的固定维度 dense user embedding 作为主要缓存产物。

### How to use

论文验证了四种消费方式：

1. **Linear probe**：固定用户表示，在不同业务标签上训练轻量预测头。
2. **Existing user models**：将 Tokenizer 表示作为补充特征加入 One4all 或 MSDP。
3. **Ranking**：Token lookup 后与 item features 一起输入 MaskNet；支付场景中接入 PEPNet。
4. **Recall**：广告场景用 FOUNDv2 表示替换 OmniRec 的用户表示，但论文没有披露具体召回打分结构。

生产更新采用 append-only 思路：每天只编码 DAU 当日新增行为，并追加到已有 Token history。这使 FOUNDv2 更像一个“压缩后的长期历史缓存”，下游仍需 embedding lookup、pooling/sequence model 或其他 adapter 才能得到任务分数。

### Intermediate Performance Validation

最接近 Google Doc 所说 intermediate validation 的实验，是四类任务上的 linear probe AUC/KS。它验证的是“不依赖复杂下游模型时，预训练用户表示是否保留可迁移信息”。

此外还有三类诊断证据：

- `shared-only / specific-only / shared+specific` 码本消融，验证量化结构。
- `Semantic / Token / Window` 任务消融，验证预训练目标。
- codebook utilization 与 t-SNE，观察容量使用和来源分离。

但 FOUNDv2 没有类似 TokenMinds Decoder Recall@10 的原生输出验收指标。论文也没有报告重建误差、codebook perplexity、碰撞 bucket 分布或 Token 对未来行为的直接 retrieval accuracy。因此当前的 intermediate validation 本质上仍依赖带标签的下游 probe，而不是 Tokenizer 自身可独立监控的线上质量指标。

### Resource Consumption

资源口径可整理为：

| Phase | Paper-reported consumption | Missing information |
|---|---|---|
| Precompute embeddings | Qwen3-Embedding-0.6B；20M 多源数据 | Qwen 是否冻结、总 token/FLOPs、embedding 生成耗时与硬件占比 |
| Train FOUNDv2 | 16 × PPU-810E 100 GB；约 2 小时 | PPU 与 GPU 的等价吞吐、Stage B/C 各自耗时、模型参数量 |
| Sequence memory | batch 256 下，60/90/180 天约 3/5/10 GB | 指标是训练还是推理显存、参数/激活/输入各自占比 |
| Representation storage | 20M 用户、90 天约 8.2 GB；约 410 bytes/user 的粗略均值 | KV key、版本、TTL、复制、collision special token、每日 append 后的截断策略 |
| Daily refresh | 只处理约 400M–500M DAU，而非全部 1B 用户 | 日刷新 wall time、QPS、机器数、失败率和 freshness SLA |
| Downstream serving | Token embedding 作为附加特征 | lookup/sequence aggregation 延迟、下游显存和吞吐开销 |

如果机械地把 `8.2 GB / 20M users` 线性外推到 10 亿用户，单副本 payload 约为 410 GB；三副本约为 1.23 TB。但这一估计隐含每个用户都使用相同 90 天长度、相同 Token 数且无元数据开销，实际系统容量应同时受 DAU、保留窗口、序列截断和复制策略影响。

#### Inference 是否保存每用户 2,200 个 codes

从论文的自报存储量看，答案大概率是否定的：

\[
8.2\ \text{GB}/20\ \text{M users}\approx410\ \text{bytes/user}.
\]

正式配置每层有 1024 entries，单个 entry index 至少需要 10 bit。假设每用户保存 2,200 个 codes，不计任何 key、时间、来源和序列化开销：

| Code 编码方式 | 每用户 2,200 codes | 20M 用户 | 1B 用户单副本 |
|---|---:|---:|---:|
| 理论 10-bit bit packing | 2,750 bytes | 55 GB | 2.75 TB |
| `uint16` | 4,400 bytes | 88 GB | 4.4 TB |
| `uint32` | 8,800 bytes | 176 GB | 8.8 TB |

即便采用理论最紧凑的 10-bit packing，20M 用户也需要约 55 GB，是论文 8.2 GB 的约 6.7 倍。因此 `2,200 Token Number` 不可能在不引入其他强压缩或完全不同 codebook 配置的情况下，直接等于每个用户保存的离散 code 数。

反过来，用 410 bytes/user 估计最大 code 数：

- 10-bit packing：最多约 328 codes/user。
- `uint16`：最多约 205 codes/user。
- `uint32`：最多约 102 codes/user。

这些仍是忽略所有元数据的上限。若还要保存 user key、时间/来源边界、版本、collision token 和校验信息，真实 code 数会更少。论文早期/示例配置使用 256-entry codebook 时可以用 1 byte/code，410 bytes 才可能容纳约 410 codes；这也再次暴露了论文 256-entry/1024-entry、4-code/6-code 多套配置之间的口径不一致。

因此推理侧最可靠的结论是：

> FOUNDv2 会保存每个用户的紧凑离散 code 表示并支持增量更新，但论文没有报告每用户平均/最大 code 数。应优先采用作者测得的约 410 bytes/user，而不是假定每用户保存 2,200 个 codes。

### 对 Interest Cache 的直接判断

FOUNDv2 很适合作为“长期行为的离散缓存”候选，因为它解决了增量更新和存储密度问题；但若目标是得到一个 request 时可直接送入 MLP 的固定宽度 user feature，它没有 LLaTTE 那样直接，还需要额外 pooling、sequence encoder 或 task-specific embedding aggregation。若目标是用中间指标独立评估缓存质量，则需要补充论文没有提供的 intrinsic validation，例如 future-item retrieval、masked-token accuracy、reconstruction error、codebook perplexity、collision distribution 和 freshness degradation。

---

## 2. 论文要解决什么问题

传统工业用户表示通常将每个来源分别编码，再把多个连续向量拼接或晚期融合。作者认为这种范式有三个瓶颈：

1. **多源数据没有统一表示空间**：文本、行为序列和表格属性由不同编码器独立处理，跨来源的共性与互补关系难以充分建模。
2. **连续表示的信息密度低**：保存长时间跨度的高维连续向量或原始行为序列需要大量存储和显存，限制了可利用历史长度。
3. **缺乏多尺度结构**：单个连续向量是整体且静态的，不容易同时表达单次行为内部的局部依赖、跨周/月的周期规律以及未来高层意图。

FOUNDv2 的目标是学习一个统一函数：

\[
U_n = g_{\theta^*}(X_n),
\]

其中用户输入 \(X_n\) 包含多个来源，输出 \(U_n\) 是一串离散用户 Token。下游任务只需要训练自己的预测头：

\[
\hat y_n^i = \mathcal P_{\omega_i}(U_n).
\]

因此，上游 Tokenizer 可以独立预训练和周期性刷新，下游风控、画像、行为预测、召回或排序模型共同消费同一份用户表示。

---

## 3. 整体架构

FOUNDv2 的处理链路可以概括为：

```text
多源原始数据
  ├── 行为文本：Bill / SPM / MiniProgram / App 等
  ├── 搜索文本
  └── 表格属性
          ↓
Stage 1：连续语义编码
  ├── 文本来源 → Qwen3-Embedding-0.6B → 1024-D embedding
  └── 表格来源 → MLP adapter → embedding
          ↓
Stage 2：MRQ-VAE 离散量化
  ├── Shared codebooks：提取跨来源共性
  └── Source-specific codebooks：保留来源特有信息
          ↓
每条行为的离散 Token 序列
          ↓
多尺度预训练
  ├── Token-level：行为内部依赖
  ├── Window-level：跨时间窗口周期性
  └── Semantic-level：当前用户表示与未来行为语义对齐
          ↓
缓存并提供给下游任务
```

这里实际包含两个不同层次的训练：

- **Tokenizer 学习**：依靠重建损失与 residual quantization 损失，把连续 embedding 可靠地压成离散码。
- **用户序列预训练**：在离散 Token 序列上加入三类未来预测/对齐目标，使 Token 组合具有时序预测能力。

---

## 4. Stage 1：多源数据进入统一语言空间

### 4.1 文本与行为数据

对于行为文本 \(V_n\) 和搜索文本 \(R_n\)，作者把任务 Query 与来源数据拼成 Prompt，输入 Qwen3 Embedding，并取最后一层 `[EOS]` hidden state：

\[
\hat H_n^{(V)} = \operatorname{LLM}(Query,V_n,[EOS]),
\]

\[
\hat H_n^{(R)} = \operatorname{LLM}(Query,R_n,[EOS]).
\]

论文给出的 Query 示例是“提取支付宝用户数据的文本特征”，Source Data 则是带有日期、页面、组件和访问次数的用户日志文本。

### 4.2 表格数据

结构化表格特征不直接交给 LLM，而是通过 MLP：

\[
\hat H_n^{(T)} = \operatorname{MLP}(T_n).
\]

原因是 LLM 对原始结构化数值的理解并不稳定。最终各种来源都被映射成同维度的连续语义向量。

### 4.3 对“统一编码”的准确理解

论文把这一阶段描述为将数据映射到统一语言空间，但不同来源仍分别产生 embedding，表格数据还使用独立 MLP。真正显式共享参数并产生跨来源耦合的关键位置，是下一阶段的 shared codebooks，而不是在 Qwen 输入端把所有来源拼成一次联合前向。因此，它更准确地说是“统一语义空间 + 共享量化层”，而不是原始特征级的完全 early fusion。

---

## 5. Stage 2：Multi-View RQ-VAE

### 5.1 为什么不是普通 RQ-VAE

普通 RQ-VAE 对全部来源使用同一套码本，容易让来源特有模式互相干扰；如果每个来源完全使用独立码本，又无法共享跨来源知识。

FOUNDv2 将每个来源的量化路径拆成两段：

- 前 \(L_c\) 层使用 **shared codebooks**，编码跨来源共性。
- 后 \(L_u\) 层使用 **source-specific codebooks**，编码来源独有残差。

这相当于在离散空间中实现“公共表示 + 私有表示”。

### 5.2 Residual Quantization

来源 \(x\) 的连续向量先经过 pooling 和共享 MLP，再由 encoder 映射到 latent \(z_n^{(x)}\)。每一层从码本中选择距离当前 residual 最近的向量：

\[
c_n^l = \arg\min_k \left\|r_n^l-v_k^l\right\|_2^2,
\]

\[
r_n^{l+1}=r_n^l-v_{c_n^l}^l,
\qquad r_n^0=z_n.
\]

最终量化向量为共享码与专属码之和：

\[
\hat z_n^{(x)} =
\sum_{l=1}^{L_c}v_{c_n^l}^{l,(S)}+
\sum_{l=L_c+1}^{L_c+L_u}v_{c_n^l}^{l,(x)}.
\]

直观上，前几层先回答“这个行为跨来源共有的高层语义是什么”，后几层再回答“它在 Bill、Search 或 App 等具体来源中有什么独特模式”。

### 5.3 Tokenizer 训练损失

每个来源有自己的 MLP decoder，用量化向量重建原始 Qwen/MLP embedding：

\[
\mathcal L_{re}=\|\hat X_n-X_n\|_2^2.
\]

Residual quantization 使用 codebook loss 与 commitment loss：

\[
\mathcal L_{rq}=\sum_{l=1}^{L}
\left(
\|\operatorname{sg}[r_n^l]-v_{c_n^l}^l\|_2^2
+\alpha\|r_n^l-\operatorname{sg}[v_{c_n^l}^l]\|_2^2
\right).
\]

Tokenizer 的训练目标是：

\[
\mathcal L_{tokenizer}=\mathcal L_{re}+\mathcal L_{rq}.
\]

### 5.4 具体配置

论文正式实验采用：

| 组件 | 配置 |
|---|---|
| 文本 Encoder | Qwen3-Embedding-0.6B |
| 输入 embedding | 1024 维 |
| Shared codebook | 4 层，每层 1024 entries，code 维度 256 |
| Source-specific codebook | 6 个来源各 2 层，每层 1024 entries，code 维度 256 |
| 优化器 | AdamW，learning rate = `1e-3` |
| 计算资源 | 16 张 PPU-810E，每张 100 GB |

因此，一条属于来源 \(x\) 的行为通常由 4 个共享码和 2 个来源专属码表示。共享码本对所有来源复用，而专属码本按来源区分。

### 5.5 Token collision 处理

不同用户可能产生相同 Token 序列。论文称碰撞率低于 0.5%，并使用一个 fallback：按业务 engagement 排序，为发生碰撞的表示分配额外 special token，以保证下游能够区分。

这个设计说明 FOUNDv2 Token 的首要职责是语义压缩，而非天然的一一用户标识。唯一性由额外 token 补足，而不是要求 RQ-VAE 本身无碰撞。

---

## 6. 三种尺度的未来行为对齐

仅依靠重建目标，Token 可能只保留“能够复原当前 embedding”的信息，不一定对未来预测最有价值。FOUNDv2 因此加入三类预训练任务。

### 6.1 Token-level alignment：局部行为结构

给定一个行为拆出的 Token 序列 \(\mathcal Z=\{z_1,\ldots,z_K\}\)，Transformer 根据前序 Token 预测下一个 Token 的 codebook embedding，并使用 cosine distance：

\[
\mathcal L_{token}=\sum_{k=1}^{K-1}\omega_k
\left(1-\varphi(h_k,T_\theta(z_{k+1}))\right),
\]

\[
\omega_k=1+k/\tau_t.
\]

位置越靠后的 Token 权重越大，用来强调近期、细粒度 residual code 的信息。

### 6.2 Window-level alignment：长期周期性

把用户长期轨迹分成多个时间窗口 \(W_1,\ldots,W_N\)，选择具有周期对应关系的窗口，例如不同周的同一天，并用 InfoNCE 将同一用户的对应窗口拉近：

\[
\mathcal L_{win}=-\frac{1}{B}\sum_i
\log\frac{\exp(s(h_i^{W_a},h_i^{W_b})/\tau_w)}
{\sum_j\exp(s(h_i^{W_a},h_j^{W_b})/\tau_w)}.
\]

它希望捕获周/月级消费规律，而不只预测相邻动作。

### 6.3 Semantic-level alignment：未来高层意图

作者把未来时刻的购买商品、消费金额和支付状态等属性组织成自然语言描述，再由 LoRA 微调的 LLM 编码为 \(e_{i,t_2}^{nl}\)。当前用户 Token 的融合表示 \(e_{i,t_1}^f\) 与该未来语义做 InfoNCE 对齐：

\[
\mathcal L_{sem}=-\frac{1}{B}\sum_i
\log\frac{\exp(s(e_{i,t_1}^{f},e_{i,t_2}^{nl})/\tau_s)}
{\sum_j\exp(s(e_{i,t_1}^{f},e_{j,t_2}^{nl})/\tau_s)}.
\]

这一步不是预测精确的未来 item ID，而是让当前 Token 对未来行为的高层概念具有判别力。

### 6.4 总目标

\[
\mathcal L_{total}=
\lambda_1\mathcal L_{token}+
\lambda_2\mathcal L_{win}+
\lambda_3\mathcal L_{sem},
\]

其中实验配置为：

```text
lambda_token = 0.2
lambda_window = 0.4
lambda_semantic = 0.4
```

消融结果显示 semantic alignment 是最大增益来源，token 与 window alignment 在其基础上提供互补提升。

---

## 7. 一个简化例子

假设某用户一天中有三类数据：

```text
Bill:   午餐外卖 35 元，支付成功
Search: “周末亲子电影”
App:    打开地图并搜索电影院
```

各来源先得到连续向量，再分别经过同一组 shared codebooks 和自己的 specific codebooks：

```text
Bill   → [S12, S87, S04, S66, BILL_31, BILL_08]
Search → [S12, S87, S19, S03, SRCH_72, SRCH_11]
App    → [S12, S87, S19, S41, APP_09,  APP_52]
```

这里 `S12, S87` 的重复可以理解为不同来源共享了某种用户意图，但真实 code 没有人工定义的固定语义标签，不能直接解释为“消费”或“电影”。后续层逐渐补充来源特有信息。

长期用户历史会变成这些行为 Token 的时间序列。预训练任务分别学习：

- 同一行为的后续 residual code 应与前序 code 保持结构一致。
- 本周六与下周六的行为窗口可能具有周期相似性。
- 今天的搜索、App 和消费行为应能预测未来的“周末娱乐消费”语义。

下游排序模型读取离散 Token 的 embedding，而不必重新处理全部原始日志。

---

## 8. 实验设置

### 8.1 数据

- 预训练集：约 2000 万条支付宝多来源、多场景用户记录。
- 测试任务：每个约 50 万样本。
- 任务包括：外卖意愿、购买力、洗钱风险、游戏偏好。
- 评估方式：在线性探针上报告 AUC 与 KS。
- 推荐任务：前 21 天约 155 万样本训练，后 3 天约 10 万样本分为验证和测试集，使用 AUC、HR@10、HR@20。

### 8.2 基线

- **One4all**：通用用户表示预训练。
- **MSDP**：用多时间尺度行为分布预测学习序列表示。
- **FOUND**：上一代用户自监督与用户文本对齐框架。
- **RQ-VAE variant**：用普通 RQ-VAE 替换 MRQ-VAE。
- **MaskNet**：验证 Token 作为用户侧特征时的推荐效果。

---

## 9. 主要结果

### 9.1 用户表示任务

| 方法 | 外卖意愿 AUC / KS | 购买力 AUC / KS | 洗钱风险 AUC / KS | 游戏偏好 AUC / KS |
|---|---:|---:|---:|---:|
| One4all | 0.7663 / 0.3998 | 0.8348 / 0.4885 | 0.8855 / 0.6266 | 0.9750 / 0.8600 |
| MSDP | 0.7891 / 0.4361 | 0.8464 / 0.4989 | 0.8810 / 0.6194 | 0.9731 / 0.8505 |
| FOUND | 0.8315 / 0.5144 | 0.9381 / 0.7228 | 0.9026 / 0.6952 | 0.9425 / 0.8064 |
| FOUNDv2（普通 RQ-VAE） | 0.8338 / 0.5174 | 0.9429 / 0.7313 | 0.9237 / 0.7079 | 0.9765 / 0.8631 |
| **FOUNDv2（MRQ-VAE）** | **0.8399 / 0.5262** | **0.9516 / 0.7585** | **0.9639 / 0.8342** | **0.9797 / 0.8747** |

作者报告 FOUNDv2 相对 FOUND 平均提高约 3% AUC、9% KS。更值得注意的是，普通 RQ-VAE 已能接近或超过 FOUND，而 shared+specific 的 MRQ-VAE 在四个任务上继续稳定提升，说明收益并不只来自离散化，也来自多来源码本设计。

将 Unified Tokenizer 作为额外特征加入旧模型同样有效：

- One4all 平均提高约 1.2% AUC、3.0% KS。
- MSDP 平均提高约 0.8% AUC、2.3% KS。

这支持了“Token 可被不同下游架构复用”的主张。

### 9.2 码本结构消融

| 码本结构 | 每层容量 | avg. AUC | avg. KS |
|---|---:|---:|---:|
| 4 shared | 256 × 128 | 0.7578 | 0.4213 |
| 4 specific | 256 × 128 | 0.7606 | 0.4251 |
| 2 shared + 2 specific | 256 × 128 | 0.7662 | 0.4373 |
| 4 shared + 2 specific | 256 × 128 | 0.7677 | 0.4410 |
| **4 shared + 2 specific** | **1024 × 256** | **0.7746** | **0.4538** |

结论有两层：

1. 纯共享或纯专属码本都不如混合设计。
2. 增加层数、entry 数和 code 维度仍能提高性能，表明码本容量存在 scaling 趋势。

不过该实验同时改变了 entry 数与 code 维度，无法严格分离两者各自的贡献。

### 9.3 多尺度任务消融

| 任务组合 | 序列 Token 数 | avg. AUC | avg. KS |
|---|---:|---:|---:|
| 无对齐任务 | ≈1100 | 0.6985 | 0.3185 |
| Semantic | ≈1100 | 0.7679 | 0.4406 |
| Semantic + Window | ≈1100 | 0.7707 | 0.4445 |
| Semantic + Token | ≈1100 | 0.7708 | 0.4449 |
| Semantic + Token + Window | ≈1100 | 0.7732 | 0.4489 |
| **Semantic + Token + Window** | **≈2200** | **0.7746** | **0.4538** |

主要观察：

- Semantic alignment 贡献最大，说明对未来语义进行显式监督比单纯重建 Token 更重要。
- Token-level 与 Window-level 增益接近且可以叠加。
- 将表中 `Token Number` 从约 1100 扩展到约 2200 仍有提升，但边际收益较小；论文未定义这些 Token 是原始/Qwen 输入还是离散 user codes，不能将其直接当作线上缓存长度。

### 9.4 数据来源消融

移除任何来源都会降低性能，其中 Bill 数据最关键：移除 Bill 后，外卖意愿 AUC 从 0.8399 降到 0.6200，购买力 AUC 从 0.9516 降到 0.7883。这说明统一表示并不意味着各来源可互相完全替代；与任务直接相关的强特征仍然主导结果。

### 9.5 下游推荐

| 用户特征 | Backbone | AUC | HR@10 | HR@20 |
|---|---|---:|---:|---:|
| User Profile | MaskNet | 0.6476 | 0.2308 | 0.2968 |
| User Profile + Tokenizer | MaskNet | 0.6527 | 0.2678 | 0.4913 |

加入用户 Token 后，论文报告相对提升约 0.7% AUC、16.0% HR@10、65.5% HR@20。这里 Tokenizer 是补充特征，不是用生成式模型替换 MaskNet。

---

## 10. 效率与“信息密度”

### 10.1 存储和训练时间

| 模型 | 数据规模 | 训练时间 | 表示/数据存储 |
|---|---:|---:|---:|
| FOUND | 20M | 约 7 小时 | 240 GB |
| FOUNDv2 | 20M | 约 2 小时 | 8.2 GB |

论文据此报告：

- 训练速度约 **3.5 倍**。
- 表示存储约缩小 **30 倍**。

需要注意，240 GB 在正文中也被称为 90 天行为日志的 raw data，而 8.2 GB 是压缩后的 tokenized form。这一比较很好地展示了系统收益，但并不是两个同维连续用户 embedding 的纯量化误差对比。

### 10.2 更长历史

固定 batch size 为 256 时：

| 模型 | 历史跨度 | GPU 显存 | avg. AUC | avg. KS |
|---|---:|---:|---:|---:|
| FOUND | 60 天 | ≈60 GB | 0.7613 | 0.4259 |
| FOUNDv2 | 60 天 | ≈3 GB | 0.7633 | 0.4291 |
| FOUNDv2 | 90 天 | ≈5 GB | 0.7732 | 0.4489 |
| FOUNDv2 | 180 天 | ≈10 GB | 0.7746 | 0.4538 |

这组结果体现了作者所说的 densing：在有限显存中装入更多历史信息。FOUNDv2 从 60 天扩展到 180 天时仍只使用约 10 GB，但从 90 天到 180 天的收益明显小于从 60 天到 90 天，说明继续延长历史存在边际递减。

---

## 11. 工业部署

### 11.1 增量更新

生产系统覆盖约 10 亿用户。每天只编码当日新增行为，每个模态为一条行为生成 4 个离散 Token，再追加到已有历史 Token 序列，不重新计算完整历史。

相比上一代 FOUND 需要刷新全部 10 亿用户，FOUNDv2 只处理约 4 亿至 5 亿日活用户，并避免重复编码历史行为，因此降低了每日离线推理成本和延迟。

这里“每个模态 4 个 Token”与正文正式配置的“4 层 shared + 2 层 specific”没有被论文进一步对齐，可能对应不同部署配置或不同计数口径。

### 11.2 在线 A/B 结果

- **Tap-to-Pay**：优惠券成本降低 15%。
- **Scan-to-Pay**：定价成本降低 4.71%。
- **广告 CPM**：提高 0.56%。
- **广告消耗**：提高 0.50%。
- **广告带来的交易额**：提高 0.72%。
- **客户价值**：提高 3.13%。

支付场景将 FOUNDv2 表示作为 PEPNet 的核心特征；广告召回则用 FOUNDv2 替换 OmniRec 的用户表示。

---

## 12. Codebook 分析

作者观察到：

- Shared codebook 第一层利用率较低，表示少量 code 已能概括跨来源的公共模式。
- 更深 shared 层与 source-specific codebooks 利用率更高，承担细粒度重组和来源特化。
- Tabular codebook 利用率最低，作者将其归因于数值属性的语义多样性低于 App、Bill 等文本来源。
- t-SNE 中，shared+specific 架构形成更紧密、分离更清晰的来源簇；纯 shared 架构的 Search 与 SPM 等来源边界较模糊。

这为 shared+specific 设计提供了直观证据，但 t-SNE 只是一种投影后的定性可视化，不能单独证明表示已经实现语义解耦。更有说服力的证据仍然是码本消融和下游指标。

---

## 13. 与常见用户 Token/SID 方法的区别

FOUNDv2 的 Token 对象是**用户行为/用户表示**，并非商品或文档 ID：

| 方法类型 | 被量化对象 | Token 的主要作用 |
|---|---|---|
| TIGER、RQ-VAE item SID | 商品 embedding | 让生成式模型通过 Token 序列召回具体商品 |
| TokenMinds | 未来 item SID / 用户兴趣 | 生成多个离散未来兴趣，并同时保留 dense user embedding |
| **FOUNDv2** | 多源用户数据的连续 embedding | 压缩长期历史，形成跨任务复用的用户 Token 序列 |

FOUNDv2 的下游推荐实验仍然是 `User Token → embedding lookup → MaskNet`，没有把推荐直接改写成自回归生成用户 Token 或 item SID。因此，它更接近“离散化的通用用户特征基础设施”，而不是端到端生成式召回模型。

---

## 14. 论文最重要的贡献

1. **把离散量化从 item 表示扩展到多源用户表示**：目标不只是索引 item，而是压缩可跨任务复用的用户历史。
2. **Shared + specific residual codebooks**：在离散空间同时保留跨来源共性和来源独有模式，优于全部共享或全部独立。
3. **让 Token 具有预测性而非仅有重建性**：通过局部、周期和未来语义三种尺度进行对齐。
4. **系统层收益明确**：Token 适合缓存、增量追加，并显著降低长历史的存储、训练和刷新成本。
5. **具有工业证据**：论文不仅报告离线探针指标，也给出支付宝多个场景的在线收益。

---

## 15. 局限性与阅读时需要保留的判断

### 15.1 实验完全依赖私有数据

四个表示任务、推荐任务和线上实验都来自支付宝，没有公开数据集上的同口径结果。即使代码公开，外部研究者也很难复现论文的主要结论，尤其是多来源字段、时间窗口构造和线上增量更新。

### 15.2 “统一多源编码”并不是完全联合编码

Qwen 对各文本来源分别编码，表格由 MLP 编码；不同来源在 MRQ-VAE 的 shared codebooks 处才显式共享。因此论文对“early fusion”或“统一编码”的表述略强，模型并未在原始 token 层做全来源 self-attention。

### 15.3 训练阶段连接关系交代不足

论文分别给出 Tokenizer 的 `reconstruction + RQ` 损失和用户序列的三种 alignment 损失，但没有完整说明：

- 两阶段是严格顺序训练还是联合微调。
- Alignment 阶段是否冻结 Qwen、MRQ-VAE encoder 和 codebooks。
- 执行三种 alignment 的 Transformer 层数、隐藏维度和参数量。
- Token embedding 是直接读取 codebook vector，还是另建 embedding table。

这些细节会显著影响复现成本和对收益来源的判断。

### 15.4 部分 Token 数口径不一致

正式 MRQ-VAE 配置是每个来源 4 个 shared levels 加 2 个 specific levels，但部署章节称每个模态生成 4 个 Token；alignment 消融又报告约 1100/2200 个 `Token Number`。后者的 token 类型未定义，而且 8.2 GB/20M users 的存储数字排除了“每用户缓存 2200 个离散 codes”的朴素解释。论文没有明确给出原始输入长度、每日输出 code 数、最终缓存长度与历史截断策略之间的转换关系。

### 15.5 Collision fallback 描述较模糊

论文称碰撞低于 0.5%，但没有报告用户规模下的碰撞分布、最大 bucket、special-token vocabulary 大小、增量稳定性或额外存储。按 engagement 排序补特殊码也会把业务信号引入原本的语义表示，其长期漂移如何处理没有展开。

### 15.6 效率对比需注意口径

“240 GB → 8.2 GB”比较的是原始行为数据与压缩 Token，不是两个等价输出格式；“60 GB → 3 GB”也没有拆分模型参数、激活、输入序列和 embedding table 的显存组成。结果仍然说明压缩很有效，但 30 倍不能简单理解为模型自身显存减少 30 倍。

### 15.7 Baseline 公平性与因果归因有限

论文没有充分说明所有表示模型是否使用完全相同的来源、历史跨度、预训练样本量和下游调参预算。FOUNDv2 的优势同时来自 Qwen3 encoder、离散化、更长历史、MRQ-VAE 和未来对齐任务，当前实验不能完全拆分每一部分的独立贡献。

### 15.8 在线实验缺少统计信息

论文没有给出 A/B 测试时长、流量比例、置信区间和显著性检验，也没有披露延迟、日更新吞吐、峰值资源或失败回退指标。因此在线结果具有工业参考价值，但证据完整性不如包含详细实验协议的生产论文。

---

## 16. 对推荐系统实践的启示

- **用户表示不一定必须是单个 dense vector**。离散 Token 序列可以保留更多行为结构，又允许传统 ranker 通过 embedding lookup 使用。
- **共享与专属容量需要同时存在**。对异构行为强行全部共享会污染公共空间，完全隔离又失去跨域迁移；分层 residual codebook 是一个简洁的折中。
- **压缩目标应与未来预测目标结合**。只优化 reconstruction 容易保存与业务无关的细节，语义和时间对齐能让有限码本容量优先承载预测性信息。
- **压缩的最大收益可能来自系统，而非单点模型指标**。更长历史、更小缓存、增量更新和多任务复用共同决定工业价值。
- **长历史不是无限有效**。论文中 90 天到 180 天仍有提升，但边际增益明显收窄；实际系统应联合评估信息增益、freshness 和缓存成本。

---

## 17. 总结

FOUNDv2 提供了一条务实的工业用户建模路线：先用预训练语言 embedding 统一异构来源的语义接口，再用 shared+specific MRQ-VAE 把连续表示压成分层离散 Token，最后用多尺度未来行为对齐提升 Token 的预测能力。

论文最有说服力的部分是 shared/specific 码本消融、长历史下的显存收益，以及“每日新增 Token 追加到历史”的部署模式。它说明用户 Tokenization 的价值不仅是让数据看起来像语言，更重要的是建立一个高信息密度、可缓存、可增量更新、可被多种下游模型消费的统一用户特征层。

与此同时，对“统一融合”、30 倍效率以及工业在线增益的解读应保持克制：数据和协议完全私有，多个训练与部署细节没有披露，部分比较口径也不完全等价。更准确的结论是，FOUNDv2 为大规模多源用户表示给出了一个有强工业证据的离散化设计，而不是已经充分证明可跨平台通用的标准方案。
