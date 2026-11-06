# MUSE: A Simple Yet Effective Multimodal Search-Based Framework for Lifelong User Interest Modeling

- **机构**: 武汉大学、阿里巴巴集团
- **arXiv**: 2512.07216
- **代码/数据**: https://taobao-mm.github.io (数据集 HuggingFace: TaoBao-MM/Taobao-MM；代码 GitHub: TaoBao-MM/MUSE)

## 背景与动机

工业推荐系统的超长用户行为序列（单用户可达 $10^5$ 量级）建模通常采用两阶段框架（SIM、TWIN 等）：
1. **GSU（General Search Unit，通用检索单元）**：从用户全量历史行为中快速检索出与目标 item 最相关的 top-K 子序列；
2. **ESU（Exact Search Unit，精排检索单元）**：对这 K 条子序列做精细的用户兴趣建模（如 Target Attention），输出用于 CTR 预测的用户兴趣表征。

现有方法（SIM、TWIN）在两个阶段都**几乎只用 ID 特征**，带来两个问题：
- ID embedding 对长尾/过时 item 学习不充分，导致 GSU 检索质量差；
- ESU 只能捕捉共现信号，缺乏语义泛化能力。

已有工作 MISS 尝试在 GSU 引入多模态表征（增加一路 MM-GSU），但 **ESU 阶段仍然只用 ID 特征**，第二个问题未被解决。

本文的核心问题：**如何在 GSU 和 ESU 两个阶段都充分利用多模态信息？** 通过在工业级数据上做系统性实验，作者提炼出三条关键洞察，并据此提出 MUSE 框架。

## 三条关键洞察

| 阶段 | 结论 |
|---|---|
| **GSU** | 多模态相似度检索优于 ID-based 检索，但**检索机制的复杂度不重要**——简单内积（cosine similarity）足以达到最优效果，attention 或 ID-多模态联合检索等复杂方案收益可忽略甚至变差 |
| **ESU** | **显式的多模态序列建模**（如 SimTier）能带来显著增益，在此基础上进一步做 **ID-多模态融合**（本文提出的 SA-TA）还能继续提升 |
| **表征质量** | 多模态表征的质量（细粒度语义程度）对 **ESU 的影响远大于对 GSU 的影响**；细粒度表征（SCL）全面优于粗粒度通用表征（OpenCLIP）和协同过滤表征（I2I） |

### 关于三种多模态表征
- **OpenCLIP**（ChineseCLIP）：2 亿图文对比学习得到的通用语义表征，不含交互信号；
- **I2I**：基于淘宝 item-item 共现构造正样本对、MoCo 负采样训练的表征，注入了协同信号；
- **SCL（Semantic-aware Contrastive Learning）**：沿用作者此前工作（SimTier/MAKE 论文, 2407.19467）的预训练方法，用"搜索 query → 购买 item"构造正样本对，InfoNCE 训练，兼具语义与行为相关性，效果最好。

### GSU 实验发现
- 用 GSU 单独替换检索用的表征（ESU 保持不变）：SCL(+0.33%) > I2I(+0.22%) > OpenCLIP(+0.14%) > ID baseline (GAUC 0.6356)；
- 在 SCL 基础上尝试提升 GSU 检索复杂度：① 用可学习 MLP 变换多模态 embedding 后再算相似度（"multimodal attention score"）反而**掉点**（-0.13%）；② 融合 ID 相似度与多模态相似度（无论是简单相加还是可学习融合权重）收益极小（+0.03%/-0.01%），且带来额外计算开销。**结论：GSU 用简单多模态内积检索即可，不需要复杂化。**

### ESU 实验发现
- 纯 ID Target Attention baseline: GAUC 0.6301；
- 引入 SimTier（多模态相似度序列→直方图特征，与 ID 路并行拼接）：+0.70%；
- 提出的 SA-TA（ID attention 分数与多模态相似度融合）替换纯 ID attention：+0.60%；
- **SA-TA + SimTier 同时使用**：+1.21%，效果最好，验证两种多模态利用方式（序列统计建模 + attention 融合）是互补的。

## MUSE 框架设计

基于以上洞察，MUSE 遵循三条设计原则：(1) 采用高质量 SCL 表征；(2) GSU 用简单的相似度检索；(3) ESU 做显式多模态序列建模 + ID-多模态融合。

### GSU：多模态余弦相似度检索
给定目标 item $a$ 与用户历史序列 $\mathbf{B}_u=[b_1,\dots,b_L]$，查表得到冻结的 SCL 多模态向量 $v_a$、$\mathbf{V}_u=[v_1,\dots,v_L]$，计算

$$r_i = \langle v_a, v_i \rangle$$

取 top-K 相似度最高的行为构成子序列 $\mathbf{B}_u^{*}$，直接作为 GSU 输出，无需其它可学习模块。

### ESU：双路（Dual-Path）建模

**路径一：SimTier（多模态序列统计建模）**
不直接用原始 embedding，而是对 GSU 检索出的相似度序列 $\mathbf{R}=[r_1,\dots,r_K]$ 做统计：将 $[-1,1]$ 均匀划分为 $N$ 个 tier（档位），统计每个 tier 内相似度值的个数，得到 $N$ 维直方图 $h^{MM}=\text{Histogram}(\mathbf{R})$，作为用户多模态兴趣分布的紧凑表征。（该模块沿用了作者此前 SimTier/MAKE 论文中的 SimTier 方法，此处直接施加在 GSU 检索出的 top-K 相似度序列上。）

**路径二：SA-TA（Semantic-Aware Target Attention，ID-多模态融合 attention）**
标准 Target Attention 计算 ID 侧的 attention 分数：

$$\boldsymbol{\alpha}^{ID} = \frac{(\mathbf{E}_uW^k)(e_aW^q)^\top}{\sqrt{D}}$$

SA-TA 将其与多模态相似度向量 $\mathbf{R}$ 融合，得到融合后的 attention 分数（$\gamma_1,\gamma_2,\gamma_3$ 为可学习标量）：

$$\boldsymbol{\alpha}^{Fusion} = \gamma_1\boldsymbol{\alpha}^{ID} + \gamma_2\mathbf{R} + \gamma_3(\boldsymbol{\alpha}^{ID}\odot\mathbf{R})$$

再做 softmax 加权求和得到 ID 侧的用户兴趣表征 $u_l^{ID} = \text{Softmax}(\boldsymbol{\alpha}^{Fusion})^\top(\mathbf{E}_uW^v)$。多模态相似度直接参与 attention 权重的计算，起到"用语义相关性校正/增强 ID attention"的作用，尤其有利于长尾 item（ID embedding 学习不充分）。

**最终表征**：两路拼接 $u_l=[h^{MM}, u_l^{ID}]$，送入预测塔做 CTR 预测。

## 系统部署（淘宝展示广告系统）

工业场景延迟预算仅几百毫秒，而 GSU 需要拉取用户 100K 长度行为序列及其多模态 embedding，网络通信开销是主要瓶颈。由于该开销**只依赖用户、与候选 item 无关**，系统设计将其与排序主链路解耦：
- **异步预取**：GSU 服务与匹配（matching）阶段并行，异步从远端存储拉取行为序列和多模态 embedding，缓存进 GPU 显存，通常在匹配阶段完成前就已就绪，延迟被完全隐藏；
- **Top-K 检索**：排序阶段用缓存好的 embedding 算相似度、选 top-K，可与其它特征处理并行，几乎不增加延迟；
- **ESU 建模**：对选出的 top-K 子序列做 SimTier + SA-TA 计算，输出最终 CTR 预测。

该设计使 MUSE 能以**可忽略的额外延迟**支持 100K 长度序列建模，自 2025 年年中起已承载淘宝展示广告的主要流量。

## 数据集

现有公开推荐数据集普遍缺乏"超长行为序列 + 高质量多模态特征"的组合，作者构建了两个数据集：

| 字段 | 工业生产数据集 | 开源学术数据集 |
|---|---|---|
| 样本数 | 37.1 亿 | 9,900 万 |
| 用户数 | 1.9 亿 | 879 万 |
| 去重 item 数 | 236 亿 | 3,540 万 |
| 最大行为序列长度 | 100K | 1K |

- **工业数据集**：淘宝展示广告一周曝光日志（前 6 天训练，第 7 天测试），每条样本含数百个特征，用于验证 MUSE 在真实工业场景下的效果；
- **开源学术数据集**：出于版权原因不含原始图片/标题，改为提供每个 item 的 **128 维 SCL 多模态 embedding**；包含用户特征（脱敏 ID、年龄、性别、城市、省份）、item 特征（ID、类目、城市、省份）、最长 1K 的行为序列（仅 item ID + SCL embedding）及点击标签；这是**首个公开的"超长行为序列 + 高质量多模态 embedding"配对数据集**，用于推动学术界研究。

## 实验结果

**Baseline**：DIN、SIM-Hard、SIM-Soft、TWIN、MISS。为公平比较，除长序列建模模块外网络结构统一，均使用 SCL 表征，训练一个 epoch。DIN 用最近 50 条行为；两阶段方法 GSU 检索 50 条，全序列长度生产集截断为 5K、开源集截断为 1K。

**整体效果**（GAUC，Base=DIN）：

| 方法 | Production-5K | Open-Source-1K |
|---|---|---|
| Base(DIN) | 0.6271 | 0.6082 |
| SIM-Hard | 0.6351 (+1.27%) | 0.6139 (+0.94%) |
| SIM-Soft | 0.6356 (+1.35%) | 0.6135 (+0.87%) |
| TWIN | 0.6304 (+0.53%) | 0.6098 (+0.26%) |
| MISS | 0.6308 (+0.59%) | 0.6097 (+0.25%) |
| **MUSE** | **0.6377 (+1.69%)** | **0.6154 (+1.18%)** |

MUSE 全面超越所有 baseline，验证三点：① 多模态 GSU 优于 ID-based GSU；② GSU 复杂化（如 TWIN 的 attention 检索）收益有限甚至更差；③ ESU 引入多模态（相比只在 GSU 用多模态的 MISS）带来显著提升。

**序列长度影响**：序列从 5K 扩展到 100K，GAUC 提升 +0.38%；且在所有长度下，多模态增强的 ESU 都显著优于纯 ID ESU。

**用户/item 分组分析**：按行为序列长度将用户分 9 组，行为越长（更丰富的行为上下文）相对提升越大；按曝光频次将 item 分 9 组，长尾（低频）item 的相对提升更大，说明多模态信息有效缓解了长尾 item 的泛化问题。

**GSU 案例分析（Appendix）**：SCL 检索能精准捕捉款式、颜色、材质等细粒度视觉细节；OpenCLIP 检索粒度较粗，容易被无关背景/水印/文字干扰；SIM-hard（类目检索）能对齐类目但忽略细粒度属性；SIM-soft（ID 共现检索）偏向流行度、语义相关性低。

**线上 A/B 测试**：2025 年中上线 100K 长度版本 MUSE，相较 5K 长度的 SIM 生产基线，长期 A/B 测试取得 **CTR +12.6%、RPM +5.1%、ROI +11.4%** 的显著提升。

## 与前作（SimTier/MAKE 论文）的关系

本文与团队此前工作《Enhancing Taobao Display Advertising with Multimodal Representations》（CIKM 2024, arXiv 2407.19467）一脉相承：SCL 预训练方法、SimTier 模块均直接复用自前作。区别在于：
- 前作聚焦"单阶段"CTR 模型中如何接入多模态表征（SimTier vs MAKE 两种接入方式的对比），未涉及 GSU/ESU 两阶段超长序列框架；
- 本文将 SimTier 迁移到两阶段超长序列场景，把它作用于 **GSU 检索出的相似度子序列**而非全量序列，并新增了 **GSU 多模态检索**、**ESU 的 ID-多模态融合（SA-TA）** 等新组件，系统性回答了"多模态该如何分别赋能 GSU 与 ESU"这一更完整的问题，同时支持了 100K 级别的序列长度（前作未强调超长序列）。

## 结论

论文系统性回答了"如何在两阶段超长序列建模框架中充分利用多模态信息"这一问题，得出核心设计原则——**GSU 从简、ESU 从丰**：GSU 用高质量多模态 embedding 做简单内积检索即可，无需复杂机制；ESU 则需要显式的多模态序列建模（SimTier）与 ID-多模态融合（SA-TA）才能充分释放多模态信息的价值。基于此提出的 MUSE 框架已在淘宝展示广告系统稳定支持 100K 长度的用户行为建模，带来可观的线上收益，同时开源了首个超长序列 + 高质量多模态 embedding 配对的大规模数据集，为学术界研究提供基础设施。
