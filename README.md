# OpenVLA-RoboTwin-Finetuning

基于 RoboTwin 仿真平台和 OpenVLA-oft 的双臂机器人端到端控制微调实践仓库。  
仓库重点覆盖完整工程链路：数据采集、数据转换、RLDS 构建、分布式训练、LoRA 合并与评测。

## 一句话介绍（简历可用）

在 RoboTwin 仿真环境中复现并工程化 OpenVLA-oft 微调流程，完成从专家数据采集到 7B VLA 分布式训练与评测闭环，并针对末端 action-chunk 数据缺失问题做数据管线修复。

## 项目亮点

- **端到端工程闭环**：数据采集 -> 数据转换 -> RLDS 构建 -> 分布式训练 -> LoRA 合并 -> 仿真评测。
- **大模型训练工程**：围绕 7B VLA 的显存瓶颈，采用 LoRA + 多卡 FSDP 训练方案。
- **多模态输入适配**：支持多视角图像与 proprio 输入接入 OpenVLA-oft。
- **数据质量修复**：修复轨迹末尾样本被裁剪问题，提升末段状态学习覆盖度。
- **可解释文档化**：提供流程图与 shape 流图（`doc/*.d2`），便于复现和面试讲解。

## 技术栈

- 仿真与数据：RoboTwin、HDF5、RLDS、TensorFlow Data pipeline
- 训练与模型：PyTorch、Transformers、PEFT(LoRA)、FSDP、FlashAttention
- 工程与实验：Shell、Conda、ModelScope、TensorBoard

## 项目内容

- OpenVLA-oft 微调流程实现（含 LoRA、多视角图像与 proprio 输入）。
- RoboTwin 专家策略数据采集与预处理。
- RoboTwin 原始数据到 OpenVLA 训练所需 RLDS 格式转换。
- 单机多卡训练（推荐 FSDP）与评测脚本。
- 训练流程图与数据形状流图（`doc/*.d2`）。

## 目录概览

```text
.
├─ data/
│  └─ README_rawdata.md
├─ doc/
│  ├─ finetune_simple_flow.d2
│  └─ total.d2
├─ policy/openvla-oft/
│  ├─ datasets/
│  │  └─ robotwin_builder.py
│  ├─ prismatic/vla/datasets/rlds/
│  │  └─ traj_transforms.py
│  ├─ vla-scripts/
│  │  └─ finetune_simple.py
│  ├─ finetune_aloha.sh
│  ├─ merge_lora.sh
│  └─ eval.sh
└─ README.md
```

## 环境与算力要求

- 操作系统：Linux（推荐 Ubuntu）或等价云环境。
- Python：3.10（建议 conda 环境）。
- GPU：训练推荐 **4x RTX 4090 (24GB)**；7B 模型单卡/少卡容易 OOM。
- 说明：若仅验证流程，训练跑通少量 step（例如 100 step）即可。

## 面试讲解主线（建议）

- **为什么做**：验证 VLA 在双臂 manipulation 任务上的端到端可行性与稳定性。
- **怎么做**：先做数据闭环，再做训练闭环，最后定位并修复末段失败问题。
- **难点在哪**：7B 模型显存压力、数据格式对齐、action chunk 监督覆盖不足。
- **你的贡献**：打通训练链路、完成 FSDP 训练实践、修复 `traj_transforms.py` 末尾样本逻辑。

## 快速开始

### 1) 安装 RoboTwin 环境

参考官方文档完成 Vulkan 与 RoboTwin 安装：  
`https://robotwin-platform.github.io/doc/usage/robotwin-install.html`

示例：

```bash
conda create -n RoboTwin python=3.10 -y
conda activate RoboTwin

git clone https://github.com/Aiclass2026/RoboTwin.git
cd RoboTwin

pip install -r script/requirements.txt
bash script/_install.sh
```

### 2) 安装 OpenVLA-oft 依赖

```bash
cd policy/openvla-oft
pip install -e .

pip install packaging ninja
ninja --version
pip install "flash-attn==2.5.5" --no-build-isolation
```

### 3) 数据采集

任务列表：`https://robotwin-platform.github.io/doc/tasks/index.html`

```bash
# 参数：任务名 配置名 渲染GPU id
bash collect_data.sh beat_block_hammer demo_clean 0
```

可选配置：
- `demo_clean`：无域随机化
- `demo_randomized`：有域随机化

配置说明：`https://robotwin-platform.github.io/doc/usage/configurations.html`

### 4) 数据转换（RoboTwin -> RLDS）

```bash
cd policy/openvla-oft

python preprocess_aloha.py \
  --dataset_path /path/to/raw/data \
  --out_base_dir /path/to/oft-processed \
  --instruction_dir /path/to/instructions

python datasets/robotwin_builder.py \
  --task_name beat_block_hammer \
  --data_dir /path/to/oft-processed \
  --save_path /path/to/oft-rlds
```

### 5) 启动训练

先在 `finetune_aloha.sh` 中修改模型路径、数据路径和超参数，再运行：

```bash
cd policy/openvla-oft
bash finetune_aloha.sh
```

### 6) 合并 LoRA 权重

```bash
cd policy/openvla-oft
bash merge_lora.sh \
  /path/to/openvla \
  /path/to/lora_ckpt_dir \
  /path/to/merged_ckpt_dir
```

### 7) 评测模型

```bash
cd policy/openvla-oft
bash eval.sh beat_block_hammer demo_clean /path/to/merged_or_downloaded_ckpt 0 0 aloha_beat_block_hammer
```

## 关键修复：轨迹末尾数据保留

### 问题背景

在 action-chunk 训练中，原始实现对轨迹末尾时刻做了裁剪：

```python
effective_traj_len = traj_len - future_action_window_size
```

这会导致每条轨迹最后若干时刻无法成为训练锚点，可能削弱末段决策能力。

### 修复方式

文件：`policy/openvla-oft/prismatic/vla/datasets/rlds/traj_transforms.py`

将有效轨迹长度改为全长：

```python
effective_traj_len = traj_len
```

在未来动作不足一个完整 chunk 时，索引逻辑会使用最后有效动作补齐（repeat-last 行为），从而保持张量形状与下游接口一致。

## 可视化与文档

- 训练流程与数据形状流：`doc/finetune_simple_flow.d2`
- 项目整体流程图：`doc/total.d2`

## License

本仓库使用 MIT License，详见 `LICENSE`。
