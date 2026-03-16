# Stack：单细胞上下文学习基础模型 — 完整技术文档

> **本文档以问答（Q&A）形式系统解析 Stack 的训练架构、损失函数、输入数据、模块组成及下游任务，所有技术描述均结合官方论文 [[1]](#ref-1) 及官方教程 [[4]](#ref-4)[[5]](#ref-5) 进行说明与引证。**

---

## 目录

1. [Stack 是什么？](#1-stack-是什么)
2. [如何安装和获取模型权重？](#2-如何安装和获取模型权重)
3. [Stack 使用的是什么整体架构？](#3-stack-使用的是什么整体架构)
4. [模型以什么数据作为输入？](#4-模型以什么数据作为输入)
5. [模型一共有几个模块？各自的作用是什么？](#5-模型一共有几个模块各自的作用是什么)
6. [注意力机制的内部结构是什么？](#6-注意力机制的内部结构是什么)
7. [模型构建的整体原理是什么？](#7-模型构建的整体原理是什么)
8. [预训练使用的损失函数是什么？](#8-预训练使用的损失函数是什么)
9. [微调使用的损失函数是什么？](#9-微调使用的损失函数是什么)
10. [训练和微调的超参数配置是什么？](#10-训练和微调的超参数配置是什么)
11. [有几个下游任务？定义和原理是什么？](#11-有几个下游任务定义和原理是什么)
12. [评估指标是什么？](#12-评估指标是什么)
13. [模块与任务总览](#13-模块与任务总览)
14. [代码文件索引](#14-代码文件索引)
15. [参考文献](#15-参考文献)

---

## 1. Stack 是什么？

**A：Stack 是一个在 1.5 亿个均匀预处理的单细胞上训练的大规模编码器-解码器基础模型。**

根据官方论文 [[1]](#ref-1) 及 README [[2]](#ref-2) 的描述：

> *"Stack is a large-scale encoder-decoder foundation model trained on 150 million uniformly-preprocessed single cells. It introduces a novel tabular attention architecture that enables both intra- and inter-cellular information flow, setting cell-by-gene matrix chunks as the basic input data unit. Through in-context learning, Stack offers substantial performance improvements in generalizing biological effects and enables generation of unseen cell profiles in novel contexts."*

Stack 的三大核心创新点：

1. **Tabular Attention 架构**：以"细胞 × 基因"矩阵块为基本输入单元，交替进行细胞内（intra-cellular）和跨细胞（inter-cellular）注意力计算，突破了传统单细胞基础模型仅处理单个细胞的局限。

2. **上下文学习（In-Context Learning）**：模型在推理时无需重新训练，通过观察同批次的"上下文细胞"，零样本泛化到新的生物学场景（新供体、新扰动条件、新细胞类型）。

3. **大规模预训练**：在来自 CellxGene 等数据库的 1.5 亿个人类单细胞转录组数据上预训练，并通过冻结教师蒸馏进行微调，覆盖药物扰动、跨供体等多种场景 [[1]](#ref-1)。

---

## 2. 如何安装和获取模型权重？

**A：通过 PyPI 安装软件包，从 HuggingFace 下载预训练权重。**

### 2.1 安装

按照官方 README [[2]](#ref-2) 的说明，支持两种安装方式：

```bash
# 方式一：pip 安装（推荐）
pip install arc-stack

# 方式二：从源码安装
git clone https://github.com/ArcInstitute/stack.git
cd stack
pip install -e .
```

> **依赖说明**：YAML 格式的配置文件需要额外安装 `pyyaml`（`pip install pyyaml`）。官方 README [[2]](#ref-2) 的 Quick Start 部分同时提供了 `uv` 包管理器的安装方式。

### 2.2 预训练模型权重

官方提供两个预训练模型，均托管于 HuggingFace Hub [[3]](#ref-3)：

| 模型 | HuggingFace ID | 用途 |
|------|----------------|------|
| Stack-Large（预训练） | [`arcinstitute/Stack-Large`](https://huggingface.co/arcinstitute/Stack-Large) | 细胞嵌入提取 |
| Stack-Large-Aligned（微调） | [`arcinstitute/Stack-Large-Aligned`](https://huggingface.co/arcinstitute/Stack-Large-Aligned) | 扰动预测与上下文生成 |

```python
# 使用 huggingface_hub 下载模型权重（参见官方教程 [4]）
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="arcinstitute/Stack-Large",
    repo_type="model",
    local_dir="stack-large-model",
)
```

---

## 3. Stack 使用的是什么整体架构？

**A：Stack 采用编码器-解码器式的 Tabular Attention 基础模型，模型类名为 `StateICLModel`（State In-Context Learning Model）。**

官方论文 [[1]](#ref-1) 将该架构命名为 **Tabular Attention**（表格注意力），其核心思路是：将一批单细胞数据组织成一个"细胞 × 基因"矩阵块（chunk），同时处理多个细胞和多个基因，实现细胞间（inter-cellular）和细胞内（intra-cellular）的双向信息流动。

架构数据流如下（对应 `src/stack/models/core/base.py`）：

```
原始计数矩阵
(batch_size, n_cells, n_genes)
         │
         ▼ log1p 变换 + 矩形遮罩（预训练）/ 时间调度遮罩（微调）
         │
         ▼
gene_reduction：Linear(n_genes → n_hidden × token_dim) + GELU + Dropout
         │  reshape: (batch, n_cells, n_hidden, token_dim)
         ▼
gene_pos_embedding（可学习，形状 (n_hidden, token_dim)，广播加到每层输入）
         │
         ▼ ×n_layers  TabularAttentionLayer
         │   ├── Cell-Attention：(batch×n_cells, n_hidden, token_dim) 上的 MHA
         │   │    + LayerNorm（残差连接）
         │   ├── Gene-Attention：(batch, n_cells, n_hidden×token_dim) 上的 MHA
         │   │    + LayerNorm（残差连接，支持可选因果掩码）
         │   └── MLP：token 维 Feed-Forward + LayerNorm（残差连接）
         │
         ▼
output_mlp：Linear(n_hidden×token_dim → n_hidden×token_dim×2) + GELU
            Linear(n_hidden×token_dim×2 → n_genes×2)
         │
         ▼
px_scale（softmax）× lib_size → nb_mean
softplus(output[...,1])      → nb_dispersion
```

---

## 4. 模型以什么数据作为输入？

**A：输入为单细胞 RNA-seq 原始计数矩阵，以"细胞 × 基因"矩阵块为基本数据单元。**

官方论文 [[1]](#ref-1) 强调将矩阵块（chunk）而非单个细胞作为基本处理单元，这是 Stack 区别于其他单细胞基础模型的关键设计。

### 4.1 输入张量格式

```
Tensor shape: (batch_size, n_cells, n_genes)
```

| 维度 | 含义 | 典型值 |
|------|------|--------|
| `batch_size` | 一次前向传播中的矩阵块数量 | 8–32 |
| `n_cells` | 每块的细胞数（预训练 / 微调不同） | 预训练: 128–256；微调: 512 |
| `n_genes` | 高变基因（HVG）数量 | 1,000–15,000 |

### 4.2 数据类型和预处理

- **原始格式**：整数计数（raw counts），存储在 HDF5 / H5AD（AnnData）文件中
- **内部预处理**：模型在 `forward()` 中自动执行 `log1p(x)` 变换（`src/stack/models/core/base.py`）
- **库大小**：`lib_size = features.sum(dim=-1, keepdim=True)`，用于归一化 NB 分布的均值参数

### 4.3 数据集配置格式（参见 README [[2]](#ref-2)）

官方 README [[2]](#ref-2) 的 "Dataset Configuration Format" 小节定义了两类数据集格式：

```
# 人类数据集格式
human:/path:donor_col:cell_type_col[:filter_organism[:gene_col]]

# 药物扰动数据集格式
drug:/path:condition_col:cell_line_col:control_condition[:filter_organism[:gene_col]]
```

### 4.4 高变基因（HVG）计算（参见 README [[2]](#ref-2)）

官方 README [[2]](#ref-2) 的 "Data Preparation" 小节提供了 HVG 计算方法：

```python
from stack.data.datasets import DatasetConfig, compute_hvg_union

configs = [DatasetConfig(path="/data/path", filter_organism=True)]
hvg_genes = compute_hvg_union(configs, n_top_genes=1000, output_path="hvg.pkl")
```

---

## 5. 模型一共有几个模块？各自的作用是什么？

**A：模型核心共有 4 大可学习模块，另含 1 个辅助正则化组件（非 nn.Module）。**

所有模块均在 `StateICLModelBase.__init__()` 中初始化（`src/stack/models/core/base.py`）。

---

### 模块 1：基因降维层（`gene_reduction`）

```python
# src/stack/models/core/base.py
self.gene_reduction = nn.Sequential(
    nn.Linear(n_genes, n_hidden * token_dim),
    nn.GELU(),
    nn.Dropout(dropout),
)
```

**作用**：将每个细胞的基因表达向量（维度 `n_genes`）压缩到 `n_hidden × token_dim` 的隐空间，再 reshape 为形如 `(n_hidden, token_dim)` 的 token 矩阵。此步骤相当于将"基因空间"映射到"token 空间"，`n_hidden` 个 token 各自是 `token_dim` 维的向量，代表基因组的不同"语义槽"（slot）。

**意义**：这是从高维基因空间（通常 1000–15000 维）向低维 token 空间（通常 100×16 = 1600 维）的有效降维，后续注意力层均在此 token 空间运算。

---

### 模块 2：基因位置嵌入（`gene_pos_embedding`）

```python
# src/stack/models/core/base.py
self.gene_pos_embedding = nn.Parameter(torch.randn(n_hidden, token_dim))
```

**作用**：可学习的位置嵌入，形状与 token 矩阵中的基因维度一致（`n_hidden × token_dim`）。在每个 `TabularAttentionLayer` 执行 cell-attention 之前，此嵌入会广播加到所有细胞的 token 上：

```python
# src/stack/modules/attention.py  TabularAttentionLayer.forward()
x_cell_with_pos = x_cell + gene_pos_emb.unsqueeze(0)
```

**意义**：赋予模型区分不同"基因槽"的能力，类比 Transformer 中的位置编码（Positional Encoding），但这里编码的是"基因维度的位置"而非序列位置。

---

### 模块 3：Tabular Attention 层堆叠（`layers`）

```python
# src/stack/models/core/base.py
self.layers = nn.ModuleList([
    TabularAttentionLayer(
        token_dim=token_dim,
        n_cells=n_cells,
        n_hidden=n_hidden,
        n_heads=n_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
    )
    for _ in range(n_layers)  # 默认 n_layers=9（Stack-Large 配置）
])
```

每个 `TabularAttentionLayer`（`src/stack/modules/attention.py`）包含以下三个子模块：

#### 3a. Cell-Attention（`cell_attn`）—— 细胞内注意力（intra-cellular）

```python
self.cell_attn = MultiHeadAttention(d_model=token_dim, n_heads=cell_attn_heads)
self.cell_norm = nn.LayerNorm(token_dim)
```

- **操作维度**：将输入 reshape 为 `(batch×n_cells, n_hidden, token_dim)`，对每个细胞的 `n_hidden` 个基因 token 做多头自注意力。
- **意义**：捕捉同一细胞内不同"基因槽"之间的依赖关系，学习细胞内的基因共表达模式（intra-cellular gene-gene relationships）。

#### 3b. Gene-Attention（`gene_attn`）—— 跨细胞注意力（inter-cellular）

```python
self.gene_attn = MultiHeadAttention(d_model=n_hidden * token_dim, n_heads=n_heads)
self.gene_norm = nn.LayerNorm(n_hidden * token_dim)
```

- **操作维度**：将输入 reshape 为 `(batch, n_cells, n_hidden×token_dim)`，对 `n_cells` 个细胞的扁平嵌入做多头自注意力。
- **意义**：实现不同细胞之间的信息交换，是 Stack 上下文学习能力的核心——让"查询细胞"（待预测）能够参考"上下文细胞"（已观测）的表达谱。
- **因果掩码支持**：在微调推理阶段，通过传入布尔注意力掩码（`gene_attn_mask`），限制查询细胞只能注意上下文细胞，防止信息泄漏。

#### 3c. Token-level MLP（前馈网络）

```python
self.mlp = nn.Sequential(
    nn.Linear(token_dim, token_dim * mlp_ratio),  # 默认扩展比 mlp_ratio=4
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(token_dim * mlp_ratio, token_dim),
    nn.Dropout(dropout),
)
self.mlp_norm = nn.LayerNorm(token_dim)
```

- **操作维度**：在 token 维度（`token_dim`）独立作用于每个 token。
- **意义**：增加非线性表达能力，完成标准 Transformer 层的 Feed-Forward 部分。

所有三个子模块均采用**残差连接（Residual Connection）+ 层归一化（LayerNorm）**的标准 Pre-Norm 结构。

---

### 模块 4：输出 MLP（`output_mlp`，解码器）

```python
# src/stack/models/core/base.py
self.output_mlp = nn.Sequential(
    nn.Linear(n_hidden * token_dim, n_hidden * token_dim * 2),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(n_hidden * token_dim * 2, n_genes * 2),
)
```

**作用**：将每个细胞的聚合嵌入向量（`n_hidden × token_dim` 维）解码为每个基因的负二项（NB）分布参数。输出最后一维按 `n_genes × 2` 拆分：

```python
# src/stack/models/core/base.py  _compute_nb_parameters()
output = output.reshape(batch_size, n_cells, self.n_genes, 2)

px_scale_logits = output[..., 0]                   # 归一化均值的 logit
nb_dispersion  = F.softplus(output[..., 1])         # 离散度参数（> 0）
px_scale = F.softmax(px_scale_logits, dim=-1)       # 对基因维度 softmax
nb_mean  = px_scale * observed_lib_size             # 乘以库大小还原绝对均值
```

**意义**：NB 分布是 scRNA-seq 数据的经典统计模型（处理过离散计数数据），解码器直接输出分布参数而非点估计，可通过采样生成计数数据。

---

### 辅助组件：Sliced Wasserstein Distance（`SlicedWassersteinDistance`）

```python
# src/stack/modules/regularizers.py
@dataclass
class SlicedWassersteinDistance:
    n_proj: int = 64  # 随机投影数量

    def __call__(self, x, y, n_proj=None):
        projections = torch.randn(num_proj, latent_dim, device=x.device)
        projections = projections / projections.norm(dim=1, keepdim=True)
        x_proj = x @ projections.t()
        y_proj = y @ projections.t()
        x_sorted, _ = torch.sort(x_proj, dim=1)
        y_sorted, _ = torch.sort(y_proj, dim=1)
        return ((x_sorted - y_sorted) ** 2).mean()
```

**作用**：不是 `nn.Module`，而是一个计算正则化项的可调用辅助类。通过在随机投影方向上近似 Wasserstein 距离，将细胞嵌入分布约束为接近标准高斯分布，防止嵌入空间退化。

---

### 微调专用模块（仅 `ICL_FinetunedModel` 中存在）

微调模型（`src/stack/models/finetune/mixins.py`）在上述 4 个基础模块之外，还新增了：

| 模块 | 用途 |
|------|------|
| `query_pos_embedding`（`nn.Parameter`） | 注入到 query cells 位置，区分上下文细胞与查询细胞 |
| `cls`（3 层 MLP 分类头） | 判断细胞是否为查询细胞（辅助分类损失） |
| `nblog_sampler`（`ReparamNBLogSampler`） | 从 NB 分布参数可微采样，用于 MMD 损失计算 |
| `mmd_loss`（`SamplesLoss(loss="energy")`） | 计算预测分布与真实分布之间的 Energy MMD 损失 |

---

## 6. 注意力机制的内部结构是什么？

**A：Stack 使用标准多头自注意力（Multi-Head Self-Attention），实现了不带位置偏置的 Scaled Dot-Product Attention。**

`MultiHeadAttention` 实现位于 `src/stack/modules/attention.py`：

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5        # 1/√d_head 缩放
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)  # 合并 Q/K/V 投影
        self.proj = nn.Linear(d_model, d_model)  # 输出投影

    def forward(self, x, attn_mask=None, return_attn=False):
        qkv = self.qkv(x).reshape(B, T, 3, H, D)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:  # 布尔掩码 → -inf
            attn_scores = attn_scores.masked_fill(attn_mask, float("-inf"))
        attn = softmax(attn_scores, dim=-1)
        out = (attn @ v).reshape(B, T, d_model)
        return self.proj(out), attn  # 可选返回注意力权重
```

关键设计细节：
- **QKV 合并投影**：Q/K/V 用单个 `nn.Linear` 一次性计算，效率更高
- **无偏置**：`qkv = nn.Linear(..., bias=False)`，遵循现代 LLM 的常见实践
- **布尔因果掩码**：使用 `masked_fill(..., float("-inf"))` 实现，微调时用于防止 query cells 相互注意
- **注意力权重可返回**：支持 `return_attn=True`，`get_attn()` 方法（`src/stack/models/core/inference.py`）使用此接口进行可视化分析

---

## 7. 模型构建的整体原理是什么？

**A：Stack 的核心构建原理是"基于 Tabular Attention 的上下文学习（In-Context Learning）+ 自监督遮罩重建预训练"。**

官方论文 [[1]](#ref-1) 概括了 Stack 的设计哲学：通过在海量单细胞数据上预训练，学习跨细胞、跨组织的通用细胞表示；通过上下文学习，在无需额外训练的情况下泛化到新的生物学背景。

### 原理 1：自监督遮罩重建预训练（Masked Self-Supervised Pretraining）

参考 BERT 的 Masked Language Model（MLM）设计 [[1]](#ref-1)：

```
输入矩阵块 (batch, n_cells, n_genes)
  ↓ 随机遮罩 mask_rate ∈ [mask_rate_min, mask_rate_max]（默认 10%–80%）
    采用"矩形遮罩"：同一批次所有细胞的同一组基因被遮罩
  ↓ 模型从未被遮罩的基因预测被遮罩基因的 NB 分布参数
  ↓ 通过 NB 负对数似然损失 + SW 正则损失进行训练
```

**意义**：迫使模型学习基因间的条件依赖关系，并借助 Gene-Attention 从邻近细胞获取补充信息。

### 原理 2：上下文学习（In-Context Learning）

在微调阶段引入细胞级别的上下文学习机制 [[1]](#ref-1)：

```
细胞按时间步 t ∈ [0, 1) 随机划分为三段：
  ├── kept cells（0 到 n_kept_cell）：固定参考上下文（对照条件）
  ├── context cells（n_kept_cell 到 n_context_cell）：随时间增加的观测细胞
  └── query cells（n_context_cell 到 n_cells）：待预测的目标细胞

因果注意力掩码：
  attn_mask[:n_kept_cell, n_kept_cell:] = True  # kept cells 看不到后面
  query cells 的 Gene-Attention 只能注意 kept/context cells

查询位置嵌入（query_pos_embedding）注入到 query cells 的 token 上，
让模型区分"我是需要被预测的查询细胞"。
```

**意义**：训练模型学会"类比推断"——给定参考条件下的细胞表达谱，推断目标条件下的细胞应有什么表达，从而实现零样本（zero-shot）扰动预测。

### 原理 3：教师-学生 EMA 蒸馏（Teacher-Student Distillation）

微调时引入冻结的教师模型（`src/stack/finetune/lightning.py`）：

```python
# EMA 更新教师模型（每 500 步执行一次）
for t, s in zip(teacher_model.parameters(), model.parameters()):
    t.data.mul_(ema).add_(s.data, alpha=1.0 - ema)  # ema=0.95
```

- 教师模型以 EMA 方式跟踪学生模型，权重缓慢移动
- 教师模型的输出嵌入（`target_embeddings`）作为 MMD 损失的参考分布
- 提供稳定、低方差的训练目标，防止训练崩溃

### 原理 4：权重初始化

所有参数在 `StateICLModelBase.__init__()` 中通过 `self.apply(self._init_weights)` 统一初始化：

```python
nn.Linear  → Xavier 均匀初始化权重，偏置初始化为 0
nn.LayerNorm → 偏置初始化为 0，权重初始化为 1
nn.Embedding → 均值 0、标准差 0.2 的正态分布初始化
```

---

## 8. 预训练使用的损失函数是什么？

**A：预训练损失 = 负二项重建损失 + Sliced Wasserstein 正则损失。**

（对应 `src/stack/models/core/losses.py` 和 `src/stack/models/core/base.py`）

**总损失公式**：

$$\mathcal{L}_{\text{pretrain}} = \mathcal{L}_{\text{recon}} + \lambda_{\text{sw}} \cdot \mathcal{L}_{\text{sw}}$$

其中 `sw_weight`（即 $\lambda_{\text{sw}}$）默认为 `0.01`（可通过 `--sw_weight` 参数调整）。

---

### 损失 1：负二项重建损失（Negative Binomial Reconstruction Loss）

```python
# src/stack/models/core/losses.py
from scvi.distributions import NegativeBinomial

nb_dist = NegativeBinomial(mu=nb_mean, theta=nb_dispersion)
recon_loss_all = -nb_dist.log_prob(targets)
# 只在被遮罩基因上计算损失（mask=True 处）
recon_loss = (recon_loss_all * mask.float()).sum() / mask.float().sum()
```

**原理**：
- scRNA-seq 原始计数数据具有稀疏性、高噪声和过离散（overdispersion）的统计特性，负二项（NB）分布是对此类数据最成熟的概率建模工具，被广泛用于 scVI [[6]](#ref-6) 等主流单细胞框架。
- 损失为被遮罩基因的平均负对数似然（NLL），以自监督的方式迫使模型从可见基因预测被遮罩基因的表达。
- **矩形遮罩策略**：随机选取 `mask_rate ∈ [mask_rate_min, mask_rate_max]`（默认 10%–80%）比例的基因，对所有细胞的这些基因同时置零，形成"矩形"遮罩模式。

---

### 损失 2：Sliced Wasserstein Distance 正则损失（SW Regularization）

```python
# src/stack/models/core/losses.py  _compute_sw_loss()
prior_samples = torch.randn_like(final_cell_embeddings_subsampled)
centered_embeddings = final_cell_embeddings_subsampled - final_cell_embeddings_subsampled.mean(dim=1, keepdim=True)
sw_loss = self.sw_distance(centered_embeddings, prior_samples)
```

**原理**：
- Sliced Wasserstein Distance（SWD）是 Wasserstein-2 距离的高效近似，通过在随机单位向量方向上投影并对一维 Wasserstein 距离取平均来估计高维分布间的差异 [[7]](#ref-7)。
- 此损失将细胞嵌入的分布约束为接近各向同性的标准高斯分布 $\mathcal{N}(0, I)$，类似于变分自编码器（VAE）中的 KL 散度正则项，但计算更高效（无需 reparameterization trick）。
- 防止嵌入空间塌陷（collapse）或出现极端的各向异性分布，提升嵌入的泛化性。

---

## 9. 微调使用的损失函数是什么？

**A：微调损失 = NB 重建损失 + Energy MMD 损失 + SW 正则损失 + 分类辅助损失。**

（对应 `src/stack/models/finetune/mixins.py` 和 `src/stack/models/finetune/model.py`）

**总损失公式**：

$$\mathcal{L}_{\text{finetune}} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{mmd}} + \lambda_{\text{sw}} \cdot \mathcal{L}_{\text{sw}} + \mathcal{L}_{\text{cls}}$$

---

### 损失 1：NB 重建损失（`L_recon`）

与预训练相同，但遮罩比例更低（10%–30%），且**只在 context cells 上计算**（不在 query cells 上计算，避免监督泄漏）：

```python
# src/stack/models/finetune/model.py
if n_context_cell < n_cells and mask[:, :n_context_cell, :].float().sum() > 0:
    recon_loss, _ = self._compute_reconstruction_loss(
        nb_mean[:, :n_context_cell, :],
        nb_dispersion[:, :n_context_cell, :],
        ground_truth_features[:, :n_context_cell, :],
        mask[:, :n_context_cell, :],
    )
```

---

### 损失 2：Energy MMD 损失（`L_mmd`）

```python
# src/stack/models/finetune/mixins.py
from geomloss import SamplesLoss
mmd_loss_fn = SamplesLoss(loss="energy")

# 预测分布（从 NB 参数采样，可微）
pred_dist_all = self.nblog_sampler(mu=pred_mean, theta=pred_dispersion, N=rep_lib_size / 1e4)
# 真实分布（log1p 归一化计数）
true_dist_all = torch.log1p(1e4 * ground_truth_features[:, n_context_cell:, :] / rep_lib_size)

mmd_loss = mmd_loss_fn(pred_dist_all, true_dist_all)
# 按时间归一化（上下文越少，损失权重越高）
mmd_loss = (0.5 * mmd_loss + 0.5 * mmd_embed_loss) / (1 - time)
```

**原理**：
- 能量距离（Energy Distance）是 MMD 的一种核，定义为预测分布与真实分布之间的统计距离，可视为两个分布的"重心距离"。
- 使用 `geomloss` 库 [[8]](#ref-8) 计算，支持批量样本间的高效最优传输距离近似。
- 时间变量 `time ∈ [0, 0.9375)` 控制上下文比例（`n_context_cell = n_kept_cell + (n_cells - n_kept_cell) × time`），`/ (1 - time)` 归一化确保上下文较少时损失权重增大，迫使模型在低上下文条件下仍能做出合理预测。
- 损失同时施加在"基因表达分布"（`pred_dist_all`）和"细胞嵌入分布"（`pred_embed_all`）上，各占 50%。

---

### 损失 3：SW 正则损失（`L_sw`）

与预训练相同，防止微调时嵌入空间退化。

---

### 损失 4：分类辅助损失（Context vs. Query Classification Loss）

```python
# src/stack/models/finetune/mixins.py
# 构建 MLP 分类头
self.cls = nn.Sequential(
    nn.Linear(self.n_hidden * self.token_dim * 2, self.n_hidden),
    nn.GELU(),
    nn.Linear(self.n_hidden, 1),
)

# 计算分类损失
bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # 类别不平衡加权
# context cells 标签=0，query cells 标签=1
cls_loss = bce(logits_flat, labels_flat)
```

**原理**：
- 辅助监督任务：判断每个细胞是"已观测的上下文细胞"（标签 0）还是"待预测的查询细胞"（标签 1）。
- 输入是"上下文细胞平均嵌入"与"目标细胞嵌入"的拼接（因此输入维度为 `n_hidden × token_dim × 2`）。
- 使用类别权重（`pos_weight = neg_count / pos_count`）平衡类别不均衡。
- 通过梯度缩放钩子（`scale_pos_grad_hook: grad × 10`）加大分类头参数的梯度，加速其收敛。
- **意义**：迫使嵌入空间中上下文细胞与查询细胞的表示产生可区分的差异，辅助学习"已观测 vs. 待生成"的语义边界。

---

## 10. 训练和微调的超参数配置是什么？

**A：Stack 通过命令行参数或 YAML 配置文件管理超参数（参见 README [[2]](#ref-2) 和 `configs/` 目录）。**

### 10.1 预训练关键超参数（`stack-train` / `src/stack/cli/launch_training.py`）

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--n_hidden` | 100 | 基因槽（token）数量 |
| `--token_dim` | 8 | 每个 token 的维度 |
| `--n_layers` | 6 | Tabular Attention 层数 |
| `--n_heads` | 8 | Gene-Attention 头数 |
| `--sample_size` | 128 | 每个矩阵块的细胞数（`n_cells`） |
| `--mask_rate_min` | 0.1 | 最小遮罩比例 |
| `--mask_rate_max` | 0.8 | 最大遮罩比例 |
| `--sw_weight` | 0.01 | SW 正则损失权重 |
| `--n_proj` | 64 | SW 距离的随机投影数 |
| `--learning_rate` | 1e-4 | AdamW 学习率 |
| `--weight_decay` | 3e-3 | AdamW 权重衰减 |
| `--scheduler` | cosine | 学习率调度器类型 |
| `--precision` | bf16-mixed | 混合精度训练 |

Stack-Large 使用的预训练配置（`configs/training/bc_large.yaml`，参见 [[2]](#ref-2)）：

```yaml
n_hidden: 100
token_dim: 16
n_layers: 9
sample_size: 256
batch_size: 32
learning_rate: 0.0001
weight_decay: 0.003
max_epochs: 10
scheduler: cosine
```

### 10.2 微调关键超参数（`stack-finetune` / `src/stack/cli/launch_finetuning.py`）

Stack-Large-Aligned 使用的微调配置（`configs/finetuning/ft_parsecg.yaml`）：

```yaml
sample_size: 512      # 微调时使用更大的矩阵块
batch_size: 8
accumulate_grad_batches: 4   # 等效批次大小 = 32
learning_rate: 0.00002       # 微调学习率更小
weight_decay: 0.003
replacement_ratio: 0.75      # query cells 占比
max_epochs: 8
scheduler: cosine
```

### 10.3 优化器配置（`src/stack/finetune/lightning.py`）

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,     # 默认 1e-4（预训练）/ 2e-5（微调）
    weight_decay=weight_decay,  # 默认 3e-3
)
# 支持：Cosine Annealing（with Warmup）、ReduceLROnPlateau
```

---

## 11. 有几个下游任务？定义和原理是什么？

**A：Stack 定义了 2 个下游推理任务，分别对应 2 个 CLI 工具和 2 个官方教程 Notebook [[4]](#ref-4)[[5]](#ref-5)。**

---

### 下游任务 1：细胞嵌入提取（Cell Embedding Extraction）

**CLI 命令**：`stack-embedding`（`src/stack/cli/embedding.py`）

**官方教程**：`notebooks/tutorial-embed.ipynb` [[4]](#ref-4)

**定义**：对给定的 AnnData 文件，使用预训练 Stack-Large 模型提取每个细胞的固定维度嵌入向量（`final_cell_embeddings`，维度为 `n_hidden × token_dim`），输出为 `.npy` 文件或 `.h5ad` 文件。

**推理流程**（对应 `InferenceMixin.get_latent_representation()`，`src/stack/models/core/inference.py`）：

```
1. 将细胞数据按 donor_id 排序后组织为矩阵块
2. 对每个矩阵块执行 log1p 变换和基因降维（gene_reduction）
3. 经过 n_layers × TabularAttentionLayer 前向传播
4. 取输出 x 的细胞维度嵌入（reshape 为 n_cells × (n_hidden × token_dim)）
5. 拼接所有批次的嵌入，返回 (n_total_cells, n_hidden × token_dim) 数组
```

**官方使用示例**（参见 `notebooks/tutorial-embed.ipynb` [[4]](#ref-4)）：

```bash
# 从 HuggingFace 下载 Stack-Large 权重
# (使用 huggingface_hub.snapshot_download，repo_id="arcinstitute/Stack-Large")

stack-embedding \
    --checkpoint "stack-large-model/bc_large.ckpt" \
    --adata "lung_ordered.h5ad" \
    --genelist "stack-large-model/basecount_1000per_15000max.pkl" \
    --gene-name-col feature_name \
    --batch-size 16 \
    --output "lung_embeddings.npy"

# 在 Python 中加载并用于下游分析
import numpy as np
import scanpy as sc
adata.obsm['stack-embed'] = np.load('lung_embeddings.npy')
sc.pp.neighbors(adata, use_rep='stack-embed')
sc.tl.umap(adata)
sc.pl.umap(adata, color='cell_type')
```

> **说明**：官方教程 [[4]](#ref-4) 使用 Tabula Sapiens 肺部数据集（来自 CellxGene）进行演示。Stack 嵌入可直接输入 Scanpy 的标准分析流程（`sc.pp.neighbors` → `sc.tl.umap`），像其他降维结果一样使用。

**应用场景**：细胞类型分类、批次效应校正、细胞状态分析、零样本细胞类型注释。

---

### 下游任务 2：上下文生成 / 扰动预测（In-Context Generation / Perturbation Prediction）

**CLI 命令**：`stack-generation`（`src/stack/cli/generation.py`）

**官方教程**：`notebooks/tutorial-predict.ipynb` [[5]](#ref-5)

**定义**：给定一组"参考细胞"（context cells，已观测的对照条件或参考供体的细胞），零样本预测"查询细胞"（query cells，目标扰动条件或目标供体下未被观测的细胞）的基因表达谱。

**推理流程**（对应 `InferenceMixin.get_prediction()`，`src/stack/models/core/inference.py`）：

```
1. 加载 context AnnData（参考细胞）和 test AnnData（目标细胞）
2. 按 split_column（如 drug condition）划分批次
3. 将 context + query 细胞拼接组成矩阵块
4. 对 query cells 位置注入 query_pos_embedding
5. 构建因果注意力掩码（query cells 只能注意 context cells）
6. 模型前向传播，输出每个 query cell 的 NB 分布参数（nb_mean, nb_dispersion）
7. 从 NB 分布采样得到预测计数，输出为每个 condition 的 .h5ad 文件
```

微调模型采用**渐进式遮罩调度**（`masking_ratio_schedule` 和 `context_ratio_schedule`）逐步增加上下文信息：

```
masking_ratio_schedule: [0.8, 0.6, 0.4, 0.2, 0.0]
context_ratio_schedule:  [0.2, 0.25, 0.3, 0.35, 0.4]
```

**官方使用示例**（参见 `notebooks/tutorial-predict.ipynb` [[5]](#ref-5)）：

```bash
# 从 HuggingFace 下载 Stack-Large-Aligned 权重（repo_id="arcinstitute/Stack-Large-Aligned"）
# 下载示例数据（repo_id="arcinstitute/Stack-DrugPBMC-Example"，repo_type="dataset"）

stack-generation \
    --checkpoint "stack-aligned-model/bc_large_aligned.ckpt" \
    --base-adata "openproblems_context.h5ad" \
    --test-adata  "openproblems_test.h5ad" \
    --genelist "stack-aligned-model/basecount_1000per_15000max.pkl" \
    --output-dir "./generations" \
    --split-column "sm_name"
# 每个 drug condition 输出一个 .h5ad 文件到 ./generations/
```

> **说明**：官方教程 [[5]](#ref-5) 使用 2023 OpenProblems.Bio 竞赛的 Donor 1 PBMC 数据集演示药物扰动预测。以 T 细胞的药物反应作为上下文（context），预测骨髓细胞和 B 细胞在相同药物下的表达谱，无需任何微调。

**应用场景**：
- 零样本药物扰动预测（跨细胞系、跨供体）
- 跨条件细胞表达谱泛化
- 虚拟细胞生成与数字孪生
- 稀有扰动条件的数据增强

---

## 12. 评估指标是什么？

**A：Stack 使用三个主要评估指标，在验证/测试阶段自动计算。**

（对应 `LossComputationMixin._compute_eval_metrics()`，`src/stack/models/core/losses.py`）

| 指标 | 公式 | 含义 |
|------|------|------|
| `masked_mae` | 遮罩基因上的平均绝对误差 | 预测计数与真实计数的平均偏差 |
| `masked_corr` | 遮罩基因上的 Pearson 相关系数（按细胞取平均） | 预测表达模式与真实模式的线性相关度 |
| `mask_rate` | 被遮罩基因占总基因的比例 | 用于追踪实际遮罩率 |

微调阶段额外使用：

| 指标 | 含义 |
|------|------|
| `cls_acc` | 分类头对 context/query cells 的二分类准确率 |
| `sw_predict` | 预测 NB 分布与真实分布之间的 SWD（分布级评估） |

---

## 13. 模块与任务总览

| 类别 | 组成 | 数量 |
|------|------|------|
| **架构层** | Gene Reduction / Gene Pos Embedding / TabularAttentionLayer × n / Output MLP | **4 大模块** |
| **注意力子层** | Cell-Attention（intra-cellular）+ Gene-Attention（inter-cellular）+ Token MLP | 每层 **3 个子层** |
| **预训练损失** | NB 重建损失 + SW 正则损失 | **2 个** |
| **微调损失** | NB 重建 + Energy MMD + SW + 分类（CLS） | **4 个** |
| **微调专用模块** | query_pos_embedding + cls 分类头 + nblog_sampler + mmd_loss | **4 个** |
| **下游推理任务** | 细胞嵌入提取（`stack-embedding`）+ 上下文生成（`stack-generation`） | **2 个** |
| **评估指标** | masked_mae / masked_corr / mask_rate / cls_acc / sw_predict | **5 个** |

---

## 14. 代码文件索引

| 内容 | 文件路径 |
|------|----------|
| 模型基础架构与前向传播 | `src/stack/models/core/base.py` |
| 预训练损失函数与评估指标 | `src/stack/models/core/losses.py` |
| 推理方法（embedding / prediction） | `src/stack/models/core/inference.py` |
| Tabular Attention 模块 | `src/stack/modules/attention.py` |
| SW 正则化组件 | `src/stack/modules/regularizers.py` |
| 微调模型（整合架构） | `src/stack/models/finetune/model.py` |
| 微调模块和损失 Mixin | `src/stack/models/finetune/mixins.py` |
| NB 分布可微采样器 | `src/stack/models/utils.py` |
| 预训练 Lightning 模块 | `src/stack/training/lightning.py` |
| 微调 Lightning 模块（教师-学生） | `src/stack/finetune/lightning.py` |
| 预训练 CLI 入口 | `src/stack/cli/launch_training.py` |
| 微调 CLI 入口 | `src/stack/cli/launch_finetuning.py` |
| 嵌入提取 CLI | `src/stack/cli/embedding.py` |
| 上下文生成 CLI | `src/stack/cli/generation.py` |
| 预训练数据集 | `src/stack/data/training/datasets.py` |
| 微调数据集 | `src/stack/data/finetuning/datasets.py` |
| 预训练配置示例 | `configs/training/bc_large.yaml` |
| 微调配置示例 | `configs/finetuning/ft_parsecg.yaml` |
| 细胞嵌入教程 Notebook | `notebooks/tutorial-embed.ipynb` |
| 扰动预测教程 Notebook | `notebooks/tutorial-predict.ipynb` |

---

## 15. 参考文献

<a id="ref-1"></a>**[1]** Stack 官方论文（预印本）：  
> *"Stack: In-context learning of single-cell biology"*  
> bioRxiv, 2026. DOI: 10.64898/2026.01.09.698608v1  
> 在线地址：<https://www.biorxiv.org/content/10.64898/2026.01.09.698608v1>

<a id="ref-2"></a>**[2]** Stack 官方 README（GitHub）：  
> ArcInstitute/stack GitHub Repository README.md  
> <https://github.com/ArcInstitute/stack>

<a id="ref-3"></a>**[3]** Stack 预训练模型权重（HuggingFace Hub）：  
> `arcinstitute/Stack-Large`：<https://huggingface.co/arcinstitute/Stack-Large>  
> `arcinstitute/Stack-Large-Aligned`：<https://huggingface.co/arcinstitute/Stack-Large-Aligned>  
> `arcinstitute/Stack-DrugPBMC-Example`（示例数据）：<https://huggingface.co/datasets/arcinstitute/Stack-DrugPBMC-Example>

<a id="ref-4"></a>**[4]** 官方细胞嵌入教程 Notebook：  
> `notebooks/tutorial-embed.ipynb`（本仓库）  
> <https://github.com/ArcInstitute/stack/blob/main/notebooks/tutorial-embed.ipynb>

<a id="ref-5"></a>**[5]** 官方扰动预测教程 Notebook：  
> `notebooks/tutorial-predict.ipynb`（本仓库）  
> <https://github.com/ArcInstitute/stack/blob/main/notebooks/tutorial-predict.ipynb>

<a id="ref-6"></a>**[6]** scVI（负二项分布单细胞建模基础框架）：  
> Lopez, R. et al. *"Deep generative modeling for single-cell transcriptomics"* Nature Methods, 2018.  
> <https://www.nature.com/articles/s41592-018-0229-2>

<a id="ref-7"></a>**[7]** Sliced Wasserstein Distance：  
> Rabin, J. et al. *"Wasserstein Barycenter and Its Application to Texture Mixing"* SSVM, 2011.  
> Kolouri, S. et al. *"Sliced Wasserstein Distance for Learning Gaussian Mixture Models"* CVPR, 2018.

<a id="ref-8"></a>**[8]** geomloss 库（Energy Distance / MMD 损失计算）：  
> Feydy, J. et al. *"Interpolating between Optimal Transport and MMD using Sinkhorn Divergences"* AISTATS, 2019.  
> <https://www.kernel-operations.io/geomloss/>
