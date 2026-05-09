# LongCat-Next：将多模态词汇化为离散 Token（中文摘要）

- **论文**：LongCat-Next: Lexicalizing Modalities as Discrete Tokens  
- **arXiv**：[2603.27538](https://arxiv.org/abs/2603.27538)  
- **团队**：美团 LongCat Team（技术报告形式）  
- **开源**：[GitHub — meituan-longcat/LongCat-Next](https://github.com/meituan-longcat/LongCat-Next) · [Hugging Face — meituan-longcat/LongCat-Next](https://huggingface.co/meituan-longcat/LongCat-Next)

---

## 1. 背景与动机

大语言模型的成功建立在「下一 token 预测」（Next-Token Prediction, NTP）与离散自回归建模之上，但主流多模态系统仍以语言为中心，把图像、语音等当作外挂连续特征或附加模块，架构割裂、融合不充分。

本文提出 **Discrete Native Autoregressive（DiNA）**：在**统一的离散符号空间**中表示文本、视觉与音频，使多模态都能用同一套自回归目标训练，最大限度减少模态专用分支。核心工程问题变为：如何为各模态设计高质量的 **tokenizer / detokenizer 配对**，把原始信号可靠地映射为离散 ID 并可重建。

---

## 2. DiNA 范式要点

DiNA 期望满足三类工业级标准：

1. **性能可对齐全模态专用模型**：理解与生成都不能明显掉队。
2. **模态协同而非牺牲语言**：扩展多模态不应显著损伤纯文本能力。
3. **基础设施友好**：沿用 decoder-only、大规模 LLM 训练栈，额外归纳偏置尽量少。

在此设定下，视觉与音频不再作为「投影进 LM 的连续向量」，而是与文本一样成为 **离散 token 序列**，由同一 backbone 做多任务学习。

---

## 3. 视觉：dNaViT（Discrete Native Any-resolution Visual Transformer）

离散视觉建模常被质疑存在「天花板」，作者归纳为两重瓶颈：**表征容量**与**量化带来的信息损失**。

### 3.1 语义完备性（Semantic Completeness）

离散视觉编码 z 应近似充当原图 I 的代理：对广泛图像任务 \mathcal{Q}，有  
P(A \mid z,\mathcal{Q}) \approx P(A \mid I,\mathcal{Q})。  
这蕴含两方面：**判别不变性**（语义保留）与 **生成充分性**（可由 detokenizer 重建 I' \approx D(z)）。

### 3.2 Semantic-and-Aligned Encoder（SAE）

作者强调 **语言对齐的语义编码器** 适合作为量化前的连续流形：语义丰富、且与 LM 空间亲和。实践中可直接采用已有 VLM 编码器族（如 QwenViT、MoonViT、AIMv2 等）作为 SAE 近似，无需从零训练专用 SAE。

**本文实现**：视觉 tokenizer 侧采用 **Qwen2.5-ViT**，空间压缩约 **28×**。

### 3.3 分层离散：RVQ

在 SAE 特征上做 **残差向量量化（Residual Vector Quantization, RVQ）**：多级码本逐级拟合残差，缓解单次量化失真。码本用 **EMA** 更新，并配合 commitment 与语义重构（如特征余弦相似）等损失。

### 3.4 原生分辨率（Native Resolution）

在编码器的 **可变长度 native-resolution latent** 上操作（配合类似 Pack-n-Pack 的序列化与变长 FlashAttention），避免固定 resize 带来的细节损失；支持任意分辨率 tokenize / detokenize。

### 3.5 Detokenizer

Tokenizer 训练完成后 **冻结 SAE 与码本**。Detokenizer 以 **ViT 式像素解码器**（文中约 400M 参数）从离散码嵌入重建图像，损失含像素、感知与对齐项；再用 **flow matching** 的轻量 refiner 增强高频细节。

### 3.6 与 LM 的衔接：多级 token 与 DepthTransformer

视觉码本规模为 **8 层 × 16384**；多级嵌入 **相加** 融合（各级嵌入独立学习）。LM 仍做单步自回归预测，多级输出由 **DepthTransformer** 并行解码以恢复视觉细节——从而在一步自回归内展开指数级多级组合空间。

### 3.7 「intrinsic information recovery」讨论

作者观察到：**残差网络结构**本身为低层视觉信息保留了传播路径，即使缺乏重建监督，若干语义编码器仍具备一定重建能力；这与「语义完备性」互补，说明离散化前的连续表征已隐含可恢复性。

---

## 4. 音频 Tokenizer

- **编码**：Whisper encoder 提特征 → 下采样 4× → **8 层 RVQ**（码本尺寸递减：8k→4k→2k→1k×5）。  
- **Tokenizer 训练分支一**：离散 token 送入 **冻结的小 LM（Qwen3-1.7B）** 做多任务音频理解，使表示可对齐文本嵌入空间（后续可丢弃该 LM）。  
- **分支二**：对称解码器重建粗 Mel，再用 **flow matching** 细化，最后 vocoder 出波形。  
- **损失**：Mel 重建 + RVQ commit + LLM 理解损失加权。

**内部语言引导（文本引导语音）**：对齐的文本 token 与音频 token 经专用嵌入后 **逐元素相加**，得到「text-guided audio」模态；与用户实时输入的「pure audio」区分。生成策略包括：**并行生成**（音频相对文本延迟若干步，利于全双工）与 **串行生成**（先文后音，语言质量更稳）。训练时对延迟随机采样，统一两种极端情形。

---

## 5. 语言骨干与多模态一体化

- **骨干**：**LongCat-Flash-Lite A3B**——总参数量约 **68.5B**，激活约 **3B**（随上下文约 2.9B–4.5B），MoE 含 Zero-Expert、Shortcut MoE 等设计。  
- **路径统一**：文本、视觉、声学 token **不经模态专用分支**（对比部分工作的 modality-aware MoE、3D RoPE 等），共享单一 decoder 通路。  
- **嵌入**：视觉与音频的码本嵌入 **随机初始化**，与词表嵌入 **端到端联合训练**；预量化特征仅用于 RVQ 聚类指派，不直接固定嵌入取值。  
- **训练总量**：统一多模态阶段约 **2T tokens**。

训练分两阶段大类：**各模态 tokenizer 独立训练** → **原生多模态训练**（Pre-align 预热码本与 DepthTransformer，主干冻结 → 再全量解冻端到端，tokenizer 保持冻结）。

---

## 6. 实验结论（概括）

### 6.1 视觉理解

与 **Qwen3-Omni-A3B-Instruct**、**GPT5-minimal**、**Gemini 2.5 Flash-Lite** 及专用 MLLM（如 **Qwen3-VL-A3B-Instruct**）对比：

- **数学推理**：MathVista **83.1**、MathVision **64.7** 等在对比集中表现突出。  
- **综合学科**：MMMU **70.6**、MMMU-Pro **60.3** 等优于 Qwen3-Omni-A3B-Instruct。  
- **文档 / OCR / 图表**：OmniDocBench、OCRBench、CharXiv-RQ、ChartQA 等多项领先或极具竞争力。  
- **GUI**：OSWorld-G、ScreenSpot-V2 与强基线接近。

### 6.2 视觉生成（文生图）

与专用 T2I（如 FLUX.1-dev、Qwen-Image、Gemini 2.5 Flash Image 等）及统一多模态模型对比：**GenEval、DPG、LongText、TIFF、CVTG** 等基准上，统一架构下的 LongCat-Next 在 **长文本理解与文字渲染** 等维度优势明显，整体生成质量可与专用系统媲美（模型规模更小前提下仍有竞争力）。

### 6.3 音频

与 Gemini Flash-Lite 系列、Qwen3-Omni、MiMo-Audio、Kimi-Audio、Step-Audio-2-mini 等对比：ASR/TTS WER、MMAU 等理解任务及 OpenAudioBench 相关对话评测上整体 **很强**，部分数据集互有胜负。

### 6.4 纯文本与 Agent

在 **Tau2** 工具使用、**SWE-Bench** 等上相对 Kimi-Linear、Qwen3-Omni 等展现优势；MMLU / MMLU-Pro / C-Eval 等与纯文本优化模型仍有差距但 **多模态扩展未严重拖垮文本基线**（文中强调缓解「multimodal tax」）。

---

## 7. 消融与方法洞察（离散 vs 连续）

在较小骨干（如 Qwen-7B）上的预对齐实验表明：

1. 离散表征初期对齐更难、损失更高；
2. 引入轻量 **Pre-Buffer**（对多级码本查表求和后的 FFN 重编码）可显著加速收敛、缩小与连续特征的差距；
3. **延长训练与扩大数据**后，离散版本在 OCR/文档/图表与 STEM 等任务上可逼近连续版本（约 **1%** 量级差距），说明离散视觉并非固有天花板，**数据规模与训练配方**是关键。

---

## 8. 工程与数据（简要）

- **视觉 tokenizer 数据**：LAION、COYO、DataComp、TextAtlas、室内理解子集及高质量合成图等，约 **5000 万** 图像级训练，最大约 **1736×1736**。  
- **Detokenizer 数据**：在 Stage 1 语料基础上增加 SAM-1B、RenderedText、IDL 等。  
- **音频**：约 **250 万小时** 级语料，分阶段训练解码器、语义-声学联合、DiT 精细化等。  
- **多模态主训练**：四阶段（Pre-align → Pre-train → Mid-train → SFT），序列长度从 8K 逐步增至 **64K**（SFT）；Mid-train 强化长 CoT、任意分辨率生成等。

文中还涉及 **VHalf 流水线并行** 等基础设施与 RL、未来方向讨论，属规模化落地的配套叙述。

---

## 9. 延伸讨论：与 ERNIE 5.0 对比

对照对象为百度 **[ERNIE 5.0 Technical Report](https://arxiv.org/abs/2602.04705)**（arXiv:2602.04705）；ERNIE 要点总结见本仓库 `[summary_ernie5_unified_multimodal.md](summary_ernie5_unified_multimodal.md)`。以下仅为方法论对照，非论文原文。

### 9.1 统一多模态训练：相同点

- **单一主干承载多模态**：二者都把文本与感知模态纳入同一套自回归式训练，避免典型的「纯文本 LM + 外挂连续编码器、仅文本解码」式 late-fusion。
- **MoE 扩容**：ERNIE 为万亿级超稀疏 MoE + **模态无关专家路由**；LongCat 为 **LongCat-Flash-Lite A3B** 量级 MoE，**文本 / 视觉 / 音频 token 共用同一 decoder 通路**，不做 modality-aware 分支。
- **感知离散化**：ERNIE 视觉含位量化与 tokenizer 设计，音频为 codec + RVQ；LongCat 以 **SAE + RVQ（视觉）**、**Whisper + RVQ（音频）** 将连续信号变为离散码。

### 9.2 统一多模态训练：核心差异


| 维度          | ERNIE 5.0                                                                                                              | LongCat-Next（DiNA）                                                                                                 |
| ----------- | ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **统一目标的形式** | **Next-Group-of-Tokens**：文本 NTP+MTP；图像/视频 **NFSP**（帧内多尺度 + 帧间）；音频 **NCP**（按 RVQ 层级在深度方向自回归）。共享 token 空间 + **分模态预测接口**。 | **同一套 NTP**：各模态先变为离散 ID，与词表 token **同一自回归接口**；视觉靠 **DepthTransformer** 在单步内展开多级 RVQ。                               |
| **表征与归纳偏置** | **双路混合视觉**（CNN + ViT）、**Uni-RoPE (t,h,w)**、**级联扩散精化**；保留与模态结构绑定的设计。                                                    | **「词汇化」**：强调连续感知 → 码本 / RVQ，再与 LM **共享嵌入端到端**；精化侧重 **flow matching** 等 refiner。                                    |
| **训练起点**    | 强调 **从零联合预训练**，不依赖先训好的语言主干。                                                                                            | **Tokenizer 与主干分期**：先训 tokenizer/detokenizer，再 Pre-align → 端到端；视觉 SAE 用 **Qwen2.5-ViT**，音频 tokenizer 期可挂 **小 LM**。 |
| **视频**      | **原生纳入**（因果 3D tokenizer、NFSP）。                                                                                        | 摘要主线为图文音；视频非叙述重点。                                                                                                  |
| **系统级扩展**   | **弹性训练**（单次 run 一族子网络）、**统一多模态 RL（UMRL）** 等。                                                                           | 规模与配方侧重 tokenizer / 离散对齐；RL 偏展望。                                                                                   |


ERNIE 的「统一」是 **共享路由与 token 空间 + NFSP/NCP 等模态化预测结构 + 弹性训练与 UMRL**；LongCat 的「统一」是 **尽量把各模态变成「词」再单一 NTP**，差异主要靠 tokenizer 与 DepthTransformer、文本引导语音等衔接。

### 9.3 各模态数据编码：两者如何对照

下表概括「原始数据 → 进入主干前的离散/结构化表示」（ERNIE 细节来自其技术报告总结；LongCat 来自本文）。


| 模态          | ERNIE 5.0（编码要点）                                                                                                                           | LongCat-Next（编码要点）                                                                           |
| ----------- | ----------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **文本**      | **UTF-16BE + BPE Dropout** → 共享 token；预测侧 **NTP + MTP**。                                                                                  | 常规范畴 **子词 token** → 离散 ID，直接进入 MoE decoder。                                                  |
| **图像 / 视频** | **因果 2D 多尺度** tokenizer → **因果 3D** 统一图/视频；**bit-wise 量化** + 渐进切换；**CNN + ViT 双路** + 注意力式 Patch Merger（图 K=4，视频 K=16 跨 4 帧）；**Uni-RoPE**。 | **Qwen2.5-ViT（SAE）** → 投影 → **多层 RVQ**；**原生分辨率变长** latent；多级 embedding **相加** 入 LM。          |
| **音频**      | ~**12.5 Hz** codec + **RVQ**；首层语义 **蒸馏 Whisper**，残差层声学细节；主干侧 **NCP** 层级预测。                                                                | **Whisper encoder** → 下采样 4× → **8 层 RVQ**；tokenizer 期可对齐 **小 LM**；文本引导时 **文本嵌入与音频嵌入逐元素相加**。 |


**音频共性**：均用 **RVQ + Whisper 相关语义**；ERNIE 强调 **codec 帧率与 NCP**，LongCat 强调 **Whisper→RVQ→（可选）LM 对齐** 及 **与文本嵌入融合的交互格式**。

---

## 10. 总结

LongCat-Next 将 **DiNA** 落地为可开源的工业级原生多模态模型：**dNaViT** 把图像变为层级离散「视觉词」并支持任意分辨率理解与生成；音频侧用 Whisper + RVQ + flow 细化；**单一 MoE decoder** 在共享离散空间上做 NTP，并通过 DepthTransformer、文本延迟对齐等机制协调生成细节与交互形态。实验表明，在精心设计的 tokenizer 与训练策略下，**离散多模态表征可与连续专用模型竞争**，并在统一框架内同时做强「看、画、说」。

---

*摘要依据论文 TeX 源码（arXiv:2603.27538）整理；第 9 节延伸讨论参考 arXiv:2602.04705 及仓库内 ERNIE 5.0 摘要。*