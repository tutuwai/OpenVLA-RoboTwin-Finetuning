# OpenVLA-OFT 微调显存需求分析

## 1. 当前训练配置

以下参数来自 `finetune_aloha.sh`：

| 参数 | 值 | 说明 |
|------|------|------|
| `vla_path` | OpenVLA-7B | 基座模型，7B 参数 |
| `torch_dtype` | bf16 | 模型以 bfloat16 精度加载 |
| `use_lora` | True | 使用 LoRA 微调 |
| `lora_rank` | 32 | LoRA 秩 |
| `use_film` | True | FiLM 视觉-语言融合模块 |
| `num_images_in_input` | 3 | 输入 3 张相机图片（head + 2 wrist） |
| `use_proprio` | True | 输入本体感觉状态 |
| `use_l1_regression` | True | L1 回归连续动作头 |
| `batch_size` | 2 | 每卡 batch size |
| `grad_accumulation_steps` | 1 | 梯度累积步数 |

> **注意**：脚本注释中要求全局 batch size >= 16，否则训练不收敛。
> 全局 batch size = `batch_size * nproc_per_node * grad_accumulation_steps`。
> 当前配置：2 x 1 x 1 = **2**，远小于 16。说明原始配置是为 **8 卡** 设计的（2 x 8 x 1 = 16）。

---

## 2. 模型参数量（来自实际训练日志）

### 2.1 LoRA 注入后（FiLM / action_head 添加前）

```
trainable params: 110,828,288 || all params: 7,652,065,472 || trainable%: 1.4483
```

- **模型总参数**：76.5 亿（含 LoRA 新增的低秩矩阵）
- **LoRA 可训练参数**：1.11 亿

### 2.2 额外模块

| 模块 | 可训练参数 | 说明 |
|------|-----------|------|
| vision_backbone (原始) | 28,697,344 | LoRA 前的可训练视觉参数 |
| vision_backbone (FiLM 包装后) | 484,939,264 | FiLM 新增约 4.56 亿参数 |
| proprio_projector | 16,842,752 | 本体感觉投影层 |
| action_head (L1 regression) | 268,644,366 | 连续动作预测头 |
| **total trainable** | **852,557,326** | 所有可训练参数总计 |

### 2.3 参数汇总

```
总参数量 = 76.5 亿 (VLA+LoRA)
         + 4.56 亿 (FiLM 新增)
         + 2.69 亿 (action_head)
         + 0.17 亿 (proprio_projector)
         = 约 83.9 亿参数

其中:
  冻结参数: 83.9 - 8.53 = 75.4 亿
  可训练参数:               8.53 亿
```

---

## 3. 显存占用详细计算

### 3.1 模型权重（bf16，每参数 2 字节）

所有参数（冻结 + 可训练）都以 bf16 存储在 GPU 上：

```
83.9 亿 x 2 bytes = 16.78 GB = 15.63 GiB
```

### 3.2 梯度（bf16，仅可训练参数）

PyTorch 为每个 `requires_grad=True` 的参数保存一份梯度张量：

```
8.53 亿 x 2 bytes = 1.71 GB = 1.59 GiB
```

### 3.3 AdamW 优化器状态（fp32，仅可训练参数）

AdamW 为每个可训练参数保存两个 **fp32** 状态：

| 状态 | 计算 | 大小 |
|------|------|------|
| 一阶动量 m | 8.53 亿 x 4 bytes | 3.41 GB |
| 二阶动量 v | 8.53 亿 x 4 bytes | 3.41 GB |
| **小计** | | **6.82 GB = 6.35 GiB** |

### 3.4 静态显存合计（不含激活值）

| 项目 | 显存 (GiB) |
|------|-----------|
| 模型权重 (bf16) | 15.63 |
| 梯度 (bf16) | 1.59 |
| 优化器状态 (fp32) | 6.35 |
| **合计** | **23.57 GiB** |

### 3.5 RTX 4090 显存容量

```
GPU 总显存: 23.52 GiB（实际可用更少，驱动/系统另占少量）
```

### 3.6 结论：仅静态开销已超出容量

```
静态需求:  23.57 GiB
GPU 容量:  23.52 GiB
--------------------------
缺口:      +0.05 GiB  <-- 连模型 + 优化器都放不下！
```

**激活值（forward pass 产生的中间张量）还没算。**
3 张 224x224 图片、batch_size=2，经过 ViT + LLM 多层 Transformer，
激活值通常再占 **数 GiB**。

### 3.7 实际 OOM 日志验证

```
torch.cuda.OutOfMemoryError: Tried to allocate 20.00 MiB.
  GPU total capacity: 23.52 GiB
  PyTorch allocated:  22.90 GiB
  Free:               8.69 MiB
```

OOM 发生在 LoRA 前向传播的注意力 QKV 投影层（`peft/tuners/lora/layer.py`），
说明模型权重 + 优化器已经把显存吃满，forward pass 一计算激活值就崩了。

---

## 4. 为什么单卡 24GB 不够？一图总结

```
|<------------- RTX 4090 显存 23.52 GiB ------------->|
|                                                      |
| 模型权重 bf16  | 优化器 fp32  | 梯度  |  激活值      |
| 15.63 GiB     | 6.35 GiB    | 1.59  |  ~3-6 GiB   |
|                |             |       |              |
|<--- 静态合计 23.57 GiB ---->|       |              |
|                              ^      |              |
|                       已超过上限！   |  完全放不下   |
```

核心矛盾: **7B 模型在 bf16 下的冻结权重就占了约 15 GiB**，
加上 8.5 亿可训练参数的梯度和 AdamW fp32 优化器状态，
仅静态开销就已经超过 24GB 卡的容量。

---

## 5. 多卡并行方案

### 5.1 DDP vs FSDP

| | DDP | FSDP |
|---|---|---|
| 原理 | 每卡存**完整**模型副本 | 模型参数/梯度/优化器**分片**到各卡 |
| 每卡静态显存 (4 卡) | 23.57 GiB（不变） | **约 5.9 GiB** |
| RTX 4090 可行性 | 即使 batch_size=1 也可能 OOM | 充裕 |
| 适用场景 | 大显存 GPU（A100 40GB+） | **24GB 卡多卡并行** |

`finetune.py` 已同时支持 DDP 和 FSDP，通过 `--use_fsdp` 参数切换。

### 5.2 使用方法

**DDP（大显存 GPU，如 A100）**

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/finetune.py \
  --batch_size 4 \
  --grad_accumulation_steps 2 \
  ...
# 全局 batch = 4 x 2 x 2 = 16
```

**FSDP（推荐用于 24GB 卡，如 RTX 4090）**

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune.py \
  --use_fsdp True \
  --gradient_checkpointing True \
  --batch_size 2 \
  --grad_accumulation_steps 2 \
  ...
# 全局 batch = 2 x 4 x 2 = 16
```

### 5.3 新增参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_fsdp` | False | 启用 FSDP 替代 DDP，将模型分片到各卡 |
| `--gradient_checkpointing` | False | 梯度检查点，不保存中间激活值，反向时重算（约慢 30%，省 3-5 GiB） |

### 5.4 全局 batch size 计算

```
全局 batch = batch_size x GPU 数 x grad_accumulation_steps
```

官方建议全局 batch size >= 16，否则训练不收敛。

### 5.5 推荐配置速查

| GPU 配置 | 模式 | batch_size | nproc | grad_accum | 全局 batch |
|---------|------|-----------|-------|-----------|-----------|
| 1x RTX 4090 | - | - | - | - | **OOM** |
| 4x RTX 4090 | FSDP | 2 | 4 | 2 | 16 |
| 1x A100 40GB | DDP | 2 | 1 | 8 | 16 |
| 2x A100 40GB | DDP | 4 | 2 | 2 | 16 |
| 1x A100 80GB | DDP | 8 | 1 | 2 | 16 |
