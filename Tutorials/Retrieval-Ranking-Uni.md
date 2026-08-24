# 三者对比：OneRanker vs UniSGR vs UniR²

## 1. 基础信息与场景

| 维度 | OneRanker | UniSGR | UniR² |
|---|---|---|---|
| **机构 / 时间** | 腾讯 / 2026.03 | 阿里国际 / 2026.07 | 快手&中科院 / 2026.07 |
| **业务场景** | 广告推荐（微信视频号广告） | 电商推荐（Lazada 首页） | 短视频/直播（快手） |
| **多目标** | 兴趣（点击/转化）+ 商业价值（eCPM） | click / atc / pay（电商漏斗） | CTR / LVTR(长观看) / GTR(送礼) |
| **训练范式** | 三阶段串行（生成→增强→排序） | 两阶段（多场景预训练→场景对齐） | 两阶段（先 L_gen→联合 L_gen+L_rank） |

## 2. 联合范式（本质差异）

| 范式 | OneRanker | UniSGR | UniR² |
|---|---|---|---|
| **联合层级** | **多模块串联**（模块级耦合） | **单模型双头**（表示级共享） | **单序列多段**（序列级融合） |
| **召回-排序如何物理连接** | Step1/2（生成+增强）→ Step3（R-Decoder 排序），三模块串接 | 生成器 decoder 顶部直接挂 PLE 排序头，同一前向 | 用户段\|生成段\|排序段 拼在**同一条 decoder 序列**里 |
| **耦合桥梁** | **K/V 贯通**（Step1/2 输出直接作 Step3 的 Key/Value） | **三路表示共享**（语义ID/编码器/解码器表示） | **h^gen_L**（生成段末态 hidden state 作为排序段输入） |

## 3. 召回侧架构

| 维度 | OneRanker | UniSGR | UniR² |
|---|---|---|---|
| **Decoder 骨架** | HSTU Decoder-only | Sparse MoE Decoder（GQA + SwiGLU-MoE，7% 激活） | Decoder-only Transformer（3层，75M） |
| **生成方式** | 多路径并行 MTP（单次前向生成多条 SID 路径） | Beam Search（512, 512, 1024）三层不同 beam width | Beam Search 生成 3 层 SID |
| **推理加速** | 单次前向并行多路径 | **STARK** 树注意力 + 重组 KV Cache，吞吐 +200% | KV Cache 复用 + 排序/策略过滤并行 |

## 4. 排序侧架构

| 维度 | OneRanker | UniSGR | UniR² |
|---|---|---|---|
| **排序模块** | R-Decoder（1层，候选 token 作 Query，Step1/2 作 K/V） | Target Attention（DIN 风格，候选语义ID 作 Query，M 作 KV）+ PLE（CGC） | MMoE 多目标塔 |
| **排序输入特征** | 候选 item token + 排序任务 token T_r | 用户信息 + 语义ID表示 + Target Attention 输出 + decoder hidden states | 物品特征 + h^gen_L + 排序隐藏状态 |
| **排序损失** | BPR（候选对，标签为 eCPM 等商业价值） | BCE（多目标加权） | BCE（多目标加权） |

## 5. 任务信号注入与梯度处理（关键差异）

| 维度 | OneRanker | UniSGR | UniR² |
|---|---|---|---|
| **任务信号注入** | 任务 token 序列（兴趣+价值）+ **Fake Item Token**（簇锚点粗粒度感知）+ 因果 mask | **Task-Aware Tokens**（e_click/e_atc/e_pay 预置 BOS 前）+ **FACL** 漏斗对比学习 | 无显式任务 token，**双查询机制**（生成查询/排序查询）+ 独立 FFN 建立任务屏障 |
| **梯度处理哲学** | **软对齐**：DC 损失把排序 softmax 分布压回生成分布，形成排序→生成梯度回路 | **主动反哺**：联合损失 L_gen + α·L_rank，排序梯度强化生成表示的多目标感知 | **硬隔离**：LoRA 注入排序侧 Q/K/V，生成查询用原始投影 + detach，保护生成主干 |
| **总损失** | α·L_MTP + β·L_rank + γ·L_DC | L_gen + α·L_rank + L_aux(FACL) | L_gen（加权CE）+ L_rank（BCE） |

## 6. 效果对比

| 维度 | OneRanker | UniSGR | UniR² |
|---|---|---|---|
| **离线主指标** | HR@1 0.2639（vs GPR 0.1824，**+44.7%**） | HR@100 0.2195（vs OneRec 0.2151，+2.0%） | HR@128 0.8741（vs PROMISE 0.8416，+3.86%） |
| **线上主指标** | GMV-Normal **+1.34%**（微信视频号广告） | IPV **+3.36%** / GMV **+5.68%**（Lazada） | 播放量 **+1.177%** / 送礼总金额 **+2.569%**（快手） |

---

# 三者本质差异总结

**1. 联合范式三个层级（递进抽象）**

| 层级 | 方法 | 召回与排序的关系 |
|---|---|---|
| 模块级耦合 | OneRanker | 生成模块 → 增强模块 → 排序模块，三模块**串联**，靠 K/V 贯通 |
| 表示级共享 | UniSGR | 生成器与排序头**并联**在同一 decoder 上，共享三路表示 |
| 序列级融合 | UniR² | 召回段和排序段拼在**同一条序列**里，hidden state 自然桥接 |

**2. 梯度处理三种哲学**

| 哲学 | 方法 | 出发点 |
|---|---|---|
| 软对齐 | OneRanker | 不阻断梯度，用 DC 损失约束**分布一致**，让排序信号软性影响生成 |
| 主动反哺 | UniSGR | **希望**排序梯度改进生成表示，主动让多目标感知渗透到生成 |
| 硬隔离 | UniR² | **保护**生成主干不被排序干扰，用 LoRA + detach 物理隔离梯度 |
