# Stack 模型架构详解（Q&A 形式）

> 本文档以 **1 对 1 问答** 的方式，系统解析 Stack 代码库中的模型训练架构、损失函数、输入数据、模块组成以及下游任务。

---

## Q1：Stack 使用的是什么整体架构？

**A：Stack 采用"编码器-解码器"式的基础模型（Foundation Model），核心架构为 Tabular Attention（表格注意力）。**

模型类名为 `StateICLModel`（State In-Context Learning Model），继承自 `StateICLModelBase`（`src/stack/models/core/base.py`）。整体思路是把一批单细胞数据组织成一个"细胞 × 基因"矩阵块（chunk），让模型同时看到多个细胞、多个基因，从而支持细胞间（inter-cellular）和细胞内（intra-cellular）的信息流动。

架构数据流如下：

```
原始计数矩阵
(batch, n_cells, n_genes)
        │
        ▼
log1p 变换 + 矩形遮罩
        │
        ▼
Gene Reduction  →  Token 矩阵 (batch, n_cells, n_hidden, token_dim)
        │
        ▼
Gene Positional Embedding（可学习）
        │
        ▼
×n_layers TabularAttentionLayer
        ├── Cell-Attention（基因维度：intra-cellular）
        ├── Gene-Attention（细胞维度：inter-cellular）
        └── MLP
        │
        ▼
Output MLP  →  NB 参数 (nb_mean, nb_dispersion) + px_scale
                (batch, n_cells, n_genes)
```

---

## Q2：模型以什么数据作为输入？

**A：输入为单细胞 RNA-seq 原始计数矩阵，以"细胞 × 基因"矩阵块为基本数据单元。**

具体格式：
- **张量形状**：`(batch_size, n_cells, n_genes)`
  - `batch_size`：一次传入的矩阵块数量
  - `n_cells`：每块采样的细胞数，预训练默认 128，微调默认 512
  - `n_genes`：高变基因（HVG）数量，由 `hvg_genes.pkl` 决定（通常为 1000–15000 个）
- **数据格式**：原始整数计数（raw counts），存储在 HDF5 / H5AD 文件（AnnData 格式）中
- **预处理**：读取后在模型内部自动做 `log1p(x)` 变换，库大小（library size）通过对基因维求和得到，用于后续 NB 分布参数的归一化
- **数据集类型**：
  - `human` 类型：需提供 `donor_id` 列（批次/供体）和 `cell_type` 列（细胞类型）
  - `drug` 类型：需提供 `condition` 列（扰动条件）、`cell_line` 列（细胞系）和 `control_condition`（对照条件）

---

## Q3：模型一共有几个模块？各自的作用是什么？

**A：模型共有 4 大可学习模块，另含 1 个辅助正则化组件。**

### 模块 1：基因降维层（`gene_reduction`）

```python
nn.Sequential(
    nn.Linear(n_genes, n_hidden * token_dim),
    nn.GELU(),
    nn.Dropout(dropout),
)
```

- **作用**：将每个细胞的基因表达向量（长度 `n_genes`）投影到 `n_hidden × token_dim` 的隐空间，再 reshape 为 `(n_hidden, token_dim)` 的 token 矩阵，相当于为每个"基因槽（gene slot）"生成一个 `token_dim` 维的嵌入向量。
- **意义**：这一步是从基因空间向 token 空间的降维，后续注意力层在此 token 空间上运算。

---

### 模块 2：基因位置嵌入（`gene_pos_embedding`）

```python
nn.Parameter(torch.randn(n_hidden, token_dim))
```

- **作用**：可学习的位置嵌入，形状与 token 矩阵中的基因维对齐（`n_hidden × token_dim`），在每层注意力前加到 token 上，赋予模型区分不同"基因槽"的能力。

---

### 模块 3：Tabular Attention 层堆叠（`layers`，共 `n_layers` 层，默认 9 层）

每层 `TabularAttentionLayer`（`src/stack/modules/attention.py`）包含三个子模块：

**3a. Cell-Attention（`cell_attn`）—— 细胞内注意力（intra-cellular）**

```python
MultiHeadAttention(d_model=token_dim, n_heads=cell_attn_heads)
```

- 操作维度：把 `(batch×n_cells, n_hidden, token_dim)` 中的 `n_hidden` 个基因 token 做自注意力。
- **意义**：让同一细胞内不同基因槽之间交换信息，捕捉基因共表达关系。

**3b. Gene-Attention（`gene_attn`）—— 跨细胞注意力（inter-cellular）**

```python
MultiHeadAttention(d_model=n_hidden * token_dim, n_heads=n_heads)
```

- 操作维度：把 `(batch, n_cells, n_hidden × token_dim)` 中的 `n_cells` 个细胞做自注意力。
- **意义**：让不同细胞之间交换信息，实现上下文学习（in-context learning），即让"查询细胞"参考"上下文细胞"的表达谱。

**3c. MLP（前馈网络）**

```python
nn.Sequential(
    nn.Linear(token_dim, token_dim * mlp_ratio),
    nn.GELU(), nn.Dropout(dropout),
    nn.Linear(token_dim * mlp_ratio, token_dim),
    nn.Dropout(dropout),
)
```

- 每个 token 独立经过前馈网络，增加非线性表达能力。

---

### 模块 4：输出 MLP（`output_mlp`，解码器）

```python
nn.Sequential(
    nn.Linear(n_hidden * token_dim, n_hidden * token_dim * 2),
    nn.GELU(), nn.Dropout(dropout),
    nn.Linear(n_hidden * token_dim * 2, n_genes * 2),
)
```

- **作用**：将每个细胞的嵌入向量解码为每个基因的负二项（NB）分布参数：
  - `output[..., 0]` → `px_scale`（softmax 归一化后乘以库大小得 `nb_mean`）
  - `output[..., 1]` → `nb_dispersion`（softplus 激活）

---

### 辅助组件：Sliced Wasserstein Distance（`SlicedWassersteinDistance`）

```python
@dataclass
class SlicedWassersteinDistance:
    n_proj: int = 64  # 随机投影数量
```

- 并非 `nn.Module`，而是一个计算正则化项的可调用辅助类，用于度量细胞嵌入分布与标准高斯先验之间的差异。

---

## Q4：模型训练使用的损失函数是什么？原理是什么？

**A：预训练和微调阶段各有不同的损失函数组合。**

### 预训练损失（`StateICLModel`）

**总损失 = 重建损失 + SW 权重 × SW 正则损失**

```
L_total = L_recon + sw_weight × L_sw
```

#### 损失 1：负二项重建损失（`L_recon`，Negative Binomial Reconstruction Loss）

```python
nb_dist = NegativeBinomial(mu=nb_mean, theta=nb_dispersion)
recon_loss = -nb_dist.log_prob(targets)
recon_loss = (recon_loss * mask).sum() / mask.sum()  # 只在被遮罩基因上计算
```

- **原理**：单细胞测序数据呈现离散计数、高噪声和过离散特性，负二项分布是对 scRNA-seq 数据最经典的统计建模。模型预测每个基因的 NB 分布参数，损失为遮罩基因的负对数似然（NLL），鼓励模型从未遮罩基因预测遮罩基因的表达。
- **遮罩策略**：对基因维度随机选取 10%–80% 比例的基因置零（矩形遮罩，即所有细胞的同一组基因被遮罩），自监督方式训练。

#### 损失 2：Sliced Wasserstein Distance 正则损失（`L_sw`）

```python
prior_samples = torch.randn_like(embeddings)  # 标准高斯先验
L_sw = SW_distance(centered_embeddings, prior_samples)
```

- **原理**：Sliced Wasserstein Distance 是 Wasserstein 距离的一种高效近似，通过在随机方向上投影后比较一维分布差异（排序后 L2 距离）来度量两个高维分布的距离。此损失将细胞嵌入的分布约束为接近标准正态分布 N(0,I)，起到变分自编码器（VAE）式的先验正则化作用，防止嵌入空间塌陷。

---

### 微调损失（`ICL_FinetunedModel`）

**总损失 = 重建损失 + MMD 损失 + SW 损失 + 分类损失**

```
L_total = L_recon + L_mmd + sw_weight × L_sw + L_cls
```

#### 损失 1：重建损失（`L_recon`）

同预训练，但遮罩比例较低（10%–30%），且只在"上下文细胞"（context cells）上计算。

#### 损失 2：Energy MMD 损失（`L_mmd`，Maximum Mean Discrepancy）

```python
mmd_loss_fn = SamplesLoss(loss="energy")  # geomloss 库
L_mmd = mmd_loss_fn(pred_dist, true_dist)
```

- **原理**：能量距离（Energy Distance）是 MMD 的一种核，衡量预测细胞分布与真实细胞分布之间的统计距离。微调阶段的目标是让模型在给定"上下文细胞"后，对"查询细胞"（未被观测到、需要生成的细胞）的分布预测尽量匹配真实分布。损失按时间变量 `(1 - time)` 归一化，鼓励模型在上下文较少时仍能做出合理预测。

#### 损失 3：Sliced Wasserstein 正则损失（`L_sw`）

同预训练的 SW 损失。

#### 损失 4：分类损失（`L_cls`，Context vs. Query Classification）

```python
bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
L_cls = bce(logits, labels)  # 上下文细胞标签 0，查询细胞标签 1
```

- **原理**：辅助任务，通过一个轻量 MLP 分类头（`self.cls`）判断每个细胞是"上下文细胞"（已观测）还是"查询细胞"（待预测），迫使模型在嵌入空间中区分两类细胞的表示。加速收敛并提升嵌入质量。

---

## Q5：模型构建的整体原理是什么？

**A：Stack 的构建原理是"上下文学习（In-Context Learning）+ 自监督遮罩重建"。**

1. **自监督预训练**：给模型一个"细胞 × 基因"矩阵块，随机遮罩部分基因，让模型预测被遮罩基因的表达（类似 BERT 的 Masked Language Model）。通过 Tabular Attention 在细胞间共享信息，使模型从"邻居细胞"中学到互补信息。
2. **上下文学习（In-Context Learning）**：在微调阶段，将细胞分为"上下文细胞"（context）和"查询细胞"（query）。模型看到上下文细胞的真实表达，预测查询细胞的表达。通过因果注意力掩码（causal attention mask）防止查询细胞"看到"彼此的信息，只能从上下文细胞中学习，从而实现零样本（zero-shot）扰动预测。
3. **教师-学生蒸馏（Teacher-Student Distillation）**：微调中引入冻结的教师模型（通过 EMA 更新），为学生模型提供稳定的嵌入目标（`t_cell_embeddings`），提升训练稳定性。
4. **初始化**：所有线性层用 Xavier 均匀初始化，偏置初始化为 0；LayerNorm 参数标准初始化；Embedding 用均值 0、标准差 0.2 的正态分布初始化。

---

## Q6：有几个下游任务？每个任务的定义和原理是什么？

**A：Stack 定义了 2 大类下游推理任务，对应 2 个 CLI 工具。**

---

### 下游任务 1：细胞嵌入提取（Cell Embedding Extraction）

**CLI**：`stack-embedding`（调用 `InferenceMixin.get_prediction`）

**定义**：对给定的 AnnData 文件，使用预训练/微调后的 Stack 模型提取每个细胞的固定维度嵌入向量（`final_cell_embeddings`，维度为 `n_hidden × token_dim`），输出为 H5AD 文件（`X` 矩阵为嵌入，每列对应一个潜在维度）或 NumPy `.npy` 文件。

**原理**：
- 将细胞数据组织为矩阵块，经过完整的 Tabular Attention 前向传播。
- 取输出层中每个细胞对应的隐向量（`batch × n_cells × (n_hidden × token_dim)`）作为嵌入。
- 这些嵌入整合了细胞内基因共表达信息（cell-attention）和跨细胞上下文信息（gene-attention），可用于下游分析：聚类、降维（UMAP）、细胞类型注释等。

**应用场景**：细胞类型分类、批次效应校正、细胞状态分析。

---

### 下游任务 2：上下文生成（In-Context Generation / Perturbation Prediction）

**CLI**：`stack-generation`（调用 `InferenceMixin.get_prediction` 配合 `cell_ratio`、`context_ratio` 参数）

**定义**：给定一组"参考细胞"（reference/context cells，已观测的扰动前或对照细胞），零样本预测"查询细胞"（query cells，目标扰动条件下的细胞）的基因表达谱。

**原理**：
1. 将细胞按位置分为三段：**kept cells**（控制组，固定上下文）、**context cells**（部分观测，用于训练或推理）、**query cells**（待预测，目标条件下的细胞）。
2. 对 query cells 位置注入可学习的"查询位置嵌入"（`query_pos_embedding`），并使用因果注意力掩码（causal mask）使 query cells 只能注意 kept/context cells，而不能互相注意。
3. 模型从已观测细胞的表达模式中"类比推断"（analogical reasoning）出 query cells 在目标条件下应有的基因表达谱，输出 NB 分布参数（均值 `nb_mean` 和离散度 `nb_dispersion`），支持从中采样生成计数数据。

**应用场景**：零样本药物扰动预测、跨供体/跨细胞系的表达谱泛化、虚拟细胞生成。

---

## 模块与任务总览

| 类别 | 名称 | 数量 |
|------|------|------|
| 可学习模块 | gene_reduction / gene_pos_embedding / layers（TabularAttentionLayer × n） / output_mlp | **4 大模块** |
| 预训练损失 | NB 重建损失 + SW 正则损失 | **2 个** |
| 微调损失 | NB 重建 + MMD（Energy）损失 + SW 正则 + 分类（CLS）损失 | **4 个** |
| 下游推理任务 | 细胞嵌入提取 + 上下文生成（扰动预测） | **2 个** |

---

## 参考文件索引

| 内容 | 文件路径 |
|------|----------|
| 模型基础架构 | `src/stack/models/core/base.py` |
| 损失函数 | `src/stack/models/core/losses.py` |
| Tabular Attention 模块 | `src/stack/modules/attention.py` |
| SW 正则化 | `src/stack/modules/regularizers.py` |
| 微调模型 | `src/stack/models/finetune/model.py` |
| 微调损失 Mixin | `src/stack/models/finetune/mixins.py` |
| 推理/下游任务 | `src/stack/models/core/inference.py` |
| 训练数据集 | `src/stack/data/training/datasets.py` |
| 微调数据集 | `src/stack/data/finetuning/datasets.py` |
| 预训练 Lightning 模块 | `src/stack/training/lightning.py` |
| 微调 Lightning 模块 | `src/stack/finetune/lightning.py` |
