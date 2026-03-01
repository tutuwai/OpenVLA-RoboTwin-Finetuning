"""
finetune_simple.py

这个脚本用于对 OpenVLA-OFT 做 LoRA 微调，核心特点：
1) 主干 VLA 模型使用 FSDP（Fully Sharded Data Parallel）节省显存；
2) 动作头使用 L1 回归（直接回归连续动作，而非分类）；
3) 用 TensorBoard 记录训练/验证指标。

整体流程：
配置解析 -> 模型/处理器加载 -> 数据集与 DataLoader ->
前向计算 loss -> 反向传播 -> 日志记录 -> 保存 checkpoint。
"""

import gc
import os
import random
import time
from datetime import datetime
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional, Tuple, Type

import draccus
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import tqdm
from accelerate import PartialState
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForVision2Seq,
    AutoProcessor,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from experiments.robot.openvla_utils import (
    check_model_logic_mismatch,
    model_is_on_hf_hub,
    update_auto_map,
)
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import (
    PrismaticImageProcessor,
    PrismaticProcessor,
)
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from prismatic.models.projectors import ProprioProjector
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
)
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ======================== Configuration ========================


@dataclass
class FinetuneConfig:
    """训练配置。

    说明：
    - 该 dataclass 中每个字段都可通过命令行参数覆盖（由 draccus.wrap 提供）。
    - 大多数路径参数支持相对路径（相对于当前工作目录）。
    """
    # fmt: off
    vla_path: str = "openvla/openvla-7b"             # Path to OpenVLA model (HuggingFace Hub or local)

    # Dataset
    data_root_dir: Path = Path("datasets/rlds")      # Directory containing RLDS datasets
    dataset_name: str = "aloha_scoop_x_into_bowl"    # Name of fine-tuning dataset
    run_root_dir: Path = Path("runs")                # Directory to store checkpoints
    shuffle_buffer_size: int = 100_000               # Dataloader shuffle buffer size

    # Architecture
    use_film: bool = False                           # Use FiLM to infuse language into visual features
    num_images_in_input: int = 1                     # Number of images in VLA input
    use_proprio: bool = False                        # Include robot proprioceptive state in input

    # Training
    batch_size: int = 8                              # Batch size per device
    learning_rate: float = 5e-4
    lr_warmup_steps: int = 0                         # Steps to warm up LR (10% -> 100%)
    num_steps_before_decay: int = 100_000            # Steps before LR decays by 10x
    grad_accumulation_steps: int = 1
    max_steps: int = 200_000
    use_val_set: bool = False                        # Use validation set
    val_freq: int = 10_000                           # Validation frequency in steps
    val_time_limit: int = 180                        # Time limit (seconds) for validation
    save_freq: int = 10_000                          # Checkpoint save frequency (-1 = no mid-training saves)
    resume: bool = False                             # Resume from checkpoint
    resume_step: Optional[int] = None                # Step number to resume from
    resume_base_model_path: Optional[str] = None     # Base model path when resuming
    image_aug: bool = True                           # Train with image augmentations

    # Parallelism & memory
    gradient_checkpointing: bool = False             # Trade compute for memory

    # LoRA
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0

    # Logging
    log_dir: Path = Path("logs")                     # TensorBoard log directory
    log_freq: int = 10                               # TensorBoard logging frequency in steps
    run_id_note: Optional[str] = None                # Extra note to append to run_id (e.g. timestamp)
    # fmt: on


# ======================== Helpers ========================


def remove_ddp_prefix(state_dict: dict) -> dict:
    """去掉 DDP 保存权重时常见的 `module.` 前缀。

    Args:
        state_dict: 形如 {参数名: Tensor} 的权重字典。

    Returns:
        新字典。若 key 以 `module.` 开头则去掉该前缀。
    """
    return {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}


def load_checkpoint(module_name: str, path: str, step: int, device: str = "cpu") -> dict:
    """加载某个子模块 checkpoint，并自动兼容 DDP key 前缀。

    Args:
        module_name: 子模块名字，例如 `action_head`、`vision_backbone`。
        path: checkpoint 根目录路径。
        step: 训练步数，用于拼接文件名。
        device: `torch.load(..., map_location=...)` 的设备字符串。

    Returns:
        state_dict（dict[str, Tensor]）。
    """
    checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    return remove_ddp_prefix(state_dict)


def get_run_id(cfg: FinetuneConfig) -> str:
    """根据配置生成实验 run_id（用于目录命名和日志区分）。"""
    if cfg.resume:
        run_id = cfg.vla_path.split("/")[-1]
        if "chkpt" in run_id.split("--")[-1]:
            run_id = "--".join(run_id.split("--")[:-1])
    else:
        run_id = (
            f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
            f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
            f"+lr-{cfg.learning_rate}"
        )
        if cfg.use_lora:
            run_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
        if cfg.image_aug:
            run_id += "--image_aug"
        if cfg.run_id_note is not None:
            run_id = f"{cfg.run_id_note}--{run_id}"
    return run_id


def count_parameters(module: nn.Module, name: str) -> None:
    """打印模块中可训练参数量（`requires_grad=True`）。"""
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"# trainable params in {name}: {num_params}")


def set_seed(seed: int) -> None:
    """设置随机种子以尽量复现结果。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] Random seed set to {seed}")


# ======================== Module Wrappers ========================


class IdentityWrapper(nn.Module):
    """单卡时模拟 DDP 接口的小包装器。

    作用：
    - DDP/FSDP 场景下，通常通过 `wrapped.module` 访问原模型；
    - 单卡不做分布式时，包装为同样的访问方式，减少分支代码。
    """
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def _get_llm_transformer_layer_cls():
    """自动找到 LLM 的 Decoder Layer 类，用于 FSDP auto-wrap。"""
    try:
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        return LlamaDecoderLayer
    except ImportError:
        raise RuntimeError("Cannot detect LLM decoder layer class for FSDP wrapping.")


def wrap_fsdp(module: nn.Module, device_id: int) -> FSDP:
    """用 FSDP 包装模型（FULL_SHARD + bf16 混合精度）。"""
    layer_cls = _get_llm_transformer_layer_cls()
    auto_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={layer_cls})
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    return FSDP(
        module,
        auto_wrap_policy=auto_policy,
        mixed_precision=mp_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=device_id,
        use_orig_params=True,
        limit_all_gathers=True,
    )


def wrap_ddp(module: nn.Module, device_id: int, find_unused: bool = False) -> nn.Module:
    """分布式环境用 DDP 包装，否则返回单卡 IdentityWrapper。"""
    if dist.is_available() and dist.is_initialized():
        return DDP(module, device_ids=[device_id], find_unused_parameters=find_unused, gradient_as_bucket_view=True)
    else:
        print("[INFO] Distributed not initialized. Using single-GPU IdentityWrapper.")
        return IdentityWrapper(module)


def init_module(
    module_class: Type[nn.Module],
    module_name: str,
    cfg: FinetuneConfig,
    device_id: int,
    module_args: dict,
    to_bf16: bool = False,
    find_unused_params: bool = False,
) -> nn.Module:
    """初始化小模块（如 action_head/proprio_projector）并进行设备/并行封装。

    流程：
    1) 实例化模块；
    2) 如需 resume，加载该模块 checkpoint；
    3) 可选转 bf16；
    4) 放到当前 GPU；
    5) DDP 或 IdentityWrapper 包装。
    """
    module = module_class(**module_args)
    count_parameters(module, module_name)

    if cfg.resume:
        state_dict = load_checkpoint(module_name, cfg.vla_path, cfg.resume_step)
        module.load_state_dict(state_dict)

    if to_bf16:
        module = module.to(torch.bfloat16)
    module = module.to(device_id)

    return wrap_ddp(module, device_id, find_unused_params)


# ======================== Forward Pass (L1 Regression Only) ========================


def move_to_device(x, device):
    """递归地把嵌套结构里的 Tensor 移到指定 device。"""
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    elif isinstance(x, list):
        return [move_to_device(v, device) for v in x]
    elif isinstance(x, torch.Tensor):
        return x.to(device)
    return x


# important by haiwei
def run_forward_pass(
    vla,
    action_head,
    proprio_projector,
    batch,
    device_id,
    use_proprio,
    use_film,
    num_patches,
) -> Tuple[torch.Tensor, float]:
    """执行一次训练前向计算，并返回 loss（Tensor）与 loss 数值（float）。

    Args:
        vla: FSDP 包装后的 OpenVLA 主模型。
        action_head: DDP/Identity 包装后的 L1 回归动作头。
        proprio_projector: 本体感觉投影模块（可为 None）。
        batch: 一个 mini-batch 的字典，核心 key-value 为：
            - `input_ids`: LongTensor, shape = [B, T]
              文本 token id（包含 prompt、动作 token 等序列）。
            - `attention_mask`: Bool/Long Tensor, shape = [B, T]
              1 表示有效 token，0 表示 padding。
            - `pixel_values`: FloatTensor，
              常见 shape 为 [B, N_img, C, H, W]（多图）或 [B, C, H, W]（单图）。
              含义是输入图像像素（通常已归一化）。
            - `labels`: LongTensor, shape = [B, T]
              语言模型训练标签；其中动作 token 位置会被后续掩码选中。
            - `actions`: FloatTensor, shape = [B, NUM_ACTIONS_CHUNK, ACTION_DIM]
              连续动作监督信号（真实机械臂动作）。
            - `proprio`(可选): FloatTensor, 常见 shape = [B, PROPRIO_DIM]
              机器人本体状态（关节角、夹爪状态等）。
        device_id: 当前进程使用的 GPU id。
        use_proprio: 是否启用 proprio 分支。
        use_film: 是否启用 FiLM 视觉调制。
        num_patches: 视觉 token 数量（外加可选 proprio token）。

    Returns:
        loss: 标量 Tensor（L1）。
        loss_value: Python float，对应 `loss.item()`，用于日志记录。
    """
    # 监督动作（连续值），shape: [B, NUM_ACTIONS_CHUNK, ACTION_DIM]
    ground_truth_actions = batch["actions"].to(device_id).to(torch.bfloat16)

    device = next(vla.parameters()).device
    batch = move_to_device(batch, device)

    # autocast: 前向用 bf16，降低显存与带宽压力
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output: CausalLMOutputWithPast = vla(
            # input_ids / attention_mask / labels: [B, T]
            input_ids=batch["input_ids"].to(device_id),
            attention_mask=batch["attention_mask"].to(device_id),
            # pixel_values: [B, N_img, C, H, W] 或 [B, C, H, W]
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
            labels=batch["labels"],
            output_hidden_states=True,
            proprio=batch["proprio"] if use_proprio else None,
            proprio_projector=proprio_projector if use_proprio else None,
            use_film=use_film,
        )

    # labels[:, 1:] 对齐 next-token 预测，shape: [B, T-1]
    ground_truth_token_ids = batch["labels"][:, 1:].to(device_id)
    
    # 两个函数都会返回 bool mask，shape 都是 [B, T-1]：
    # - get_current_action_mask(...): 选中“当前动作”对应的 ACTION_DIM 个动作 token 位置。
    #   实现上通过累计计数(cumsum)截取前 ACTION_DIM 个有效动作位，再过滤出动作 token。
    # - get_next_actions_mask(...): 选中“后续动作块”对应的位置（累计计数 > ACTION_DIM 的动作 token）。
    # 这两个 mask 后续会做或运算 `current_action_mask | next_actions_mask`，
    # 用于从 `text_hidden_states` 中抽取所有用于动作回归的隐向量。
    current_action_mask = get_current_action_mask(ground_truth_token_ids)
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids)

    # 取出 VLA 的最后一层隐藏状态，shape 为 [B, seq_len, D]
    last_hidden_states = output.hidden_states[-1]
    
    # 这里做了两件事（与 labels[:, 1:] 的 next-token 训练目标严格对齐）：
    # 1) 去掉前 num_patches 个前缀 token：这些是视觉 token；若 use_proprio=True，
    #    proprio token 也作为前缀一起计入 num_patches（见上游 NUM_PATCHES += 1），
    # 2) 去掉最后一个 token 的隐状态（:-1）：自回归 LM 中位置 t 的 hidden_state 预测 t+1，
    #    最后一个位置没有对应的下一个监督 token（类比 <eos> token）。
    # 对齐关系说明：
    # - labels[:, 1:] 的 shape 是 [B, T-1]
    # - text_hidden_states 的 shape 是 [B, T-1, D]
    # 二者已在 seq_len 维度对齐
    text_hidden_states = last_hidden_states[:, num_patches:-1]
    
    batch_size = batch["input_ids"].shape[0]
    actions_hidden_states = (
        # 根据动作 mask 取出动作相关 token 的隐藏状态：
        # 1) 布尔索引后形状会变成 [B*K, D]（K=NUM_ACTIONS_CHUNK*ACTION_DIM）；
        # 2) reshape 成 [B, K, D]，恢复 batch 维度，供动作头逐 token 回归。
        text_hidden_states[current_action_mask | next_actions_mask]
        .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
        .to(torch.bfloat16)
    )  # (B, NUM_ACTIONS_CHUNK * ACTION_DIM, D)

    # L1 regression: predict actions and compute loss
    # 动作头输出连续动作，shape: [B, NUM_ACTIONS_CHUNK, ACTION_DIM]
    predicted_actions = action_head.module.predict_action(actions_hidden_states)
    
    # 标量 loss，比较预测动作与真实动作
    loss = torch.nn.L1Loss()(ground_truth_actions, predicted_actions)

    return loss, loss.item()


# ======================== Checkpoint Saving (FSDP) ========================


def save_training_checkpoint(
    cfg: FinetuneConfig,
    run_dir: Path,
    log_step: int,
    vla,
    processor,
    action_head,
    proprio_projector,
    train_dataset,
    distributed_state,
    optimizer,
    scheduler,
) -> None:
    """保存训练 checkpoint。

    保存内容：
    - VLA（FSDP）完整权重（聚合到 rank0 后保存）；
    - processor；
    - LoRA adapter；
    - 小模块权重（action_head / proprio_projector / 可选 vision_backbone）；
    - 每个 rank 自己的 optimizer/scheduler 状态（FSDP 分片态）。
    """
    checkpoint_dir = Path(str(run_dir) + f"--{log_step}_chkpt")
    checkpoint_name_suffix = f"{log_step}_checkpoint.pt"
    adapter_dir = checkpoint_dir / "lora_adapter"

    if distributed_state.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(adapter_dir, exist_ok=True)
        save_dataset_statistics(train_dataset.dataset_statistics, checkpoint_dir)
        print(f"Saving checkpoint for step {log_step}")

    if distributed_state.num_processes > 1:
        dist.barrier()

    # Gather FSDP-sharded VLA params to rank 0
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(vla, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state = vla.state_dict()

    if distributed_state.is_main_process:
        processor.save_pretrained(checkpoint_dir)
        vla.module.save_pretrained(adapter_dir, state_dict=cpu_state)

        if cfg.use_film:
            vb_prefix = "base_model.model.vision_backbone."
            vb_sd = {k[len(vb_prefix):]: v for k, v in cpu_state.items() if k.startswith(vb_prefix)}
            torch.save(vb_sd, checkpoint_dir / f"vision_backbone--{checkpoint_name_suffix}")

        if cfg.use_proprio and proprio_projector is not None:
            torch.save(proprio_projector.state_dict(), checkpoint_dir / f"proprio_projector--{checkpoint_name_suffix}")

        if action_head is not None:
            torch.save(action_head.state_dict(), checkpoint_dir / f"action_head--{checkpoint_name_suffix}")

    # Each rank saves its own optimizer/scheduler state (FSDP-sharded)
    torch.save(
        {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
        checkpoint_dir / f"training_state_rank{dist.get_rank()}.pt",
    )

    del cpu_state
    gc.collect()
    torch.cuda.empty_cache()

    if distributed_state.num_processes > 1:
        dist.barrier()


# ======================== Validation ========================


def run_validation(
    vla,
    action_head,
    proprio_projector,
    val_dataloader,
    device_id,
    cfg: FinetuneConfig,
    num_patches: int,
    log_step: int,
    distributed_state,
    writer: Optional[SummaryWriter],
) -> None:
    """运行验证集评估并写入 TensorBoard。"""
    val_start_time = time.time()
    vla.eval()
    all_val_losses = []

    with torch.no_grad():
        for batch in val_dataloader:
            _, loss_value = run_forward_pass(
                vla=vla,
                action_head=action_head,
                proprio_projector=proprio_projector if cfg.use_proprio else None,
                batch=batch,
                device_id=device_id,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                num_patches=num_patches,
            )
            all_val_losses.append(loss_value)
            if time.time() - val_start_time > cfg.val_time_limit:
                break

    # 仅统计验证 loss 平均值
    avg_val_loss = sum(all_val_losses) / len(all_val_losses) if len(all_val_losses) > 0 else float("nan")

    if distributed_state.is_main_process:
        print(f"[Val step {log_step}] batches={len(all_val_losses)}, loss={avg_val_loss:.4f}")
        if writer is not None:
            writer.add_scalar("val/loss", avg_val_loss, log_step)


# ======================== Main Training Loop ========================


# important by haiwei
@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    """训练主入口：基于 LoRA + L1 回归动作头对 OpenVLA 做微调。

    关键训练数据流（按一次 batch）：
    1) DataLoader 产生 `batch` 字典（见 `run_forward_pass` 中详细 key/shape 注释）；
    2) VLA 输出隐藏状态 `hidden_states[-1]`，shape 为 [B, T, D]；
    3) 用动作 token mask 选出动作相关隐向量；
    4) action_head 回归得到 [B, NUM_ACTIONS_CHUNK, ACTION_DIM]；
    5) 与 `batch["actions"]` 做 L1 得到 loss，反向传播更新参数。
    """
    assert cfg.use_lora, "Only LoRA fine-tuning is supported. Please set --use_lora=True!"

    cfg.vla_path = cfg.vla_path.rstrip("/")
    print(f"Fine-tuning OpenVLA `{cfg.vla_path}` on `{cfg.dataset_name}`")

    run_id = get_run_id(cfg)
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)

    # ==================== 1) GPU / 分布式环境初始化 ====================
    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    # ==================== 2) 日志系统（仅主进程） ====================
    writer = None
    if distributed_state.is_main_process:
        tb_log_dir = cfg.log_dir / run_id
        os.makedirs(tb_log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_log_dir))
        print(f"TensorBoard logs: {tb_log_dir}")

    print(
        f"Constants: NUM_ACTIONS_CHUNK={NUM_ACTIONS_CHUNK}, ACTION_DIM={ACTION_DIM}, "
        f"PROPRIO_DIM={PROPRIO_DIM}, NORM_TYPE={ACTION_PROPRIO_NORMALIZATION_TYPE}"
    )

    # ==================== 3) 加载/注册模型类 ====================
    # HuggingFace 的 Auto* 工厂（AutoConfig/AutoProcessor/AutoModel...）
    # 需要“模型类型 -> Python 类”的映射关系才能实例化自定义模型。
    # - 若是 hub 上的标准仓库，通常可直接读取其配置并加载；
    # - 若是本地/自定义路径，这里手动 register，告诉 Auto*：
    #   `model_type=openvla` 对应 OpenVLAConfig / PrismaticProcessor /
    # OpenVLAForActionPrediction 等类。
    if model_is_on_hf_hub(cfg.vla_path):
        cfg.vla_path = snapshot_download(repo_id=cfg.vla_path)
    else:
        # 注册配置类：当 config.json 中 model_type 为 "openvla" 时使用该配置类。
        AutoConfig.register("openvla", OpenVLAConfig)
        # 注册图像处理器和多模态处理器（文本+图像预处理）。
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        # 注册模型类：AutoModelForVision2Seq.from_pretrained 将返回该模型。
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    if distributed_state.is_main_process:
        update_auto_map(cfg.vla_path)
        check_model_logic_mismatch(cfg.vla_path)

    if distributed_state.num_processes > 1:
        dist.barrier()

    # ==================== 4) 加载 Processor 与 VLA ====================
    # 注意：先在 CPU 构建模型，再交给 FSDP 包装与分片
    _base_path = cfg.resume_base_model_path if cfg.resume else cfg.vla_path
    processor = AutoProcessor.from_pretrained(_base_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        _base_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True,
    )

    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    # ==================== 5) LoRA 配置 ====================
    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=min(cfg.lora_rank, 16),
        lora_dropout=cfg.lora_dropout,
        target_modules="all-linear",
        init_lora_weights="gaussian",
    )
    vla = get_peft_model(vla, lora_config)
    vla.print_trainable_parameters()

    if cfg.resume:
        assert cfg.resume_step is not None, "resume_step must be set when resume=True"
        adapter_dir = os.path.join(cfg.vla_path, "lora_adapter")
        if os.path.exists(adapter_dir):
            print(f"[Resume] Loading LoRA adapter from: {adapter_dir}")
            vla.load_adapter(adapter_dir, adapter_name="default", is_trainable=True)
            vla.set_adapter("default")
        else:
            print(f"[WARNING] lora_adapter not found at: {adapter_dir}")

    # ==================== 6) 可选 FiLM 视觉调制 ====================
    if cfg.use_film:
        count_parameters(vla.vision_backbone, "vision_backbone (original)")
        vla.model.vision_backbone = FiLMedPrismaticVisionBackbone(
            vision_backbone=vla.model.vision_backbone, llm_dim=vla.llm_dim,
        )
        count_parameters(vla.vision_backbone, "vision_backbone (post-FiLM)")
        if cfg.resume:
            state_dict = load_checkpoint("vision_backbone", cfg.vla_path, cfg.resume_step)
            vla.model.vision_backbone.load_state_dict(state_dict)

    # ==================== 7) 可选梯度检查点 ====================
    # 必须在 FSDP 包装前开启，否则某些模块不会按预期生效
    if cfg.gradient_checkpointing:
        vla.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print("[INFO] Gradient checkpointing enabled")

    # ==================== 8) FSDP 包装主模型 ====================
    vla = vla.to(dtype=torch.bfloat16)
    vla = wrap_fsdp(vla, device_id)
    print("[INFO] VLA wrapped with FSDP (FULL_SHARD)")

    # ==================== 9) 初始化小模块（DDP/单卡包装） ====================
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = init_module(
            ProprioProjector, "proprio_projector", cfg, device_id,
            {"llm_dim": vla.module.llm_dim, "proprio_dim": PROPRIO_DIM},
        )

    action_head = init_module(
        L1RegressionActionHead, "action_head", cfg, device_id,
        {"input_dim": vla.module.llm_dim, "hidden_dim": vla.module.llm_dim, "action_dim": ACTION_DIM},
        to_bf16=True,
    )

    # ==================== 10) 计算视觉 token 数 ====================
    NUM_PATCHES = (
        vla.module.vision_backbone.get_num_patches()
        * vla.module.vision_backbone.get_num_images_in_input()
    )
    if cfg.use_proprio:
        NUM_PATCHES += 1

    # ==================== 11) 优化器与学习率计划 ====================
    trainable_params = [p for p in vla.parameters() if p.requires_grad]
    trainable_params += [p for p in action_head.parameters() if p.requires_grad]
    if cfg.use_proprio:
        trainable_params += [p for p in proprio_projector.parameters() if p.requires_grad]
    print(f"# total trainable params: {sum(p.numel() for p in trainable_params)}")

    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)
    original_lr = optimizer.param_groups[0]["lr"]

    scheduler = MultiStepLR(optimizer, milestones=[cfg.num_steps_before_decay], gamma=0.1)

    # ==================== 12) 恢复优化器/调度器状态（若断点续训） ====================
    if cfg.resume:
        rank = dist.get_rank()
        train_state_path = Path(cfg.vla_path) / f"training_state_rank{rank}.pt"
        if train_state_path.exists():
            print(f"[Resume] Loading optimizer/scheduler from: {train_state_path}")
            ckpt = torch.load(train_state_path, map_location=f"cuda:{device_id}")
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
        else:
            print(f"[Warning] {train_state_path.name} not found")
        print(f"[Resume] Resumed from step {cfg.resume_step}")

    # ==================== 13) 数据集与 DataLoader ====================
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    use_wrist_image = cfg.num_images_in_input > 1

    batch_transform = RLDSBatchTransform(
        action_tokenizer, processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=use_wrist_image,
        use_proprio=cfg.use_proprio,
    )
    train_dataset = RLDSDataset(
        cfg.data_root_dir, cfg.dataset_name, batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )
    if cfg.use_val_set:
        val_dataset = RLDSDataset(
            cfg.data_root_dir, cfg.dataset_name, batch_transform,
            resize_resolution=tuple(vla.module.config.image_sizes),
            shuffle_buffer_size=cfg.shuffle_buffer_size // 10,
            image_aug=cfg.image_aug,
            train=False,
        )

    if distributed_state.is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)

    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right",
    )
    # collator 输出的 `batch` 是字典，训练时最关键字段：
    # - input_ids: [B, T]，文本 token id
    # - attention_mask: [B, T]，padding 掩码
    # - labels: [B, T]，LM 监督标签
    # - pixel_values: [B, N_img, C, H, W] 或 [B, C, H, W]，图像输入
    # - actions: [B, NUM_ACTIONS_CHUNK, ACTION_DIM]，连续动作监督
    # - proprio(可选): [B, PROPRIO_DIM]，机器人本体状态
    dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, sampler=None, collate_fn=collator, num_workers=0)
    val_dataloader = None
    if cfg.use_val_set:
        val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, sampler=None, collate_fn=collator, num_workers=0)

    # ===================== 14) 训练循环 =====================
    log_step = 0
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader):
            loss, loss_value = run_forward_pass(
                vla=vla,
                action_head=action_head,
                proprio_projector=proprio_projector if cfg.use_proprio else None,
                batch=batch,
                device_id=device_id,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                num_patches=NUM_PATCHES,
            )

            # 梯度累积：每小步只反传 1/N 的 loss，N 步后再 optimizer.step()
            normalized_loss = loss / cfg.grad_accumulation_steps
            normalized_loss.backward()

            # gradient_step_idx: 真正参数更新步（不是 dataloader 的 batch 索引）
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps
            log_step = gradient_step_idx if not cfg.resume else (cfg.resume_step or 0) + gradient_step_idx

            # LR warmup：前若干步从 10% 线性升到 100%
            if cfg.lr_warmup_steps > 0:
                lr_progress = min((gradient_step_idx + 1) / cfg.lr_warmup_steps, 1.0)
                current_lr = original_lr * (0.1 + 0.9 * lr_progress)
                for pg in optimizer.param_groups:
                    pg["lr"] = current_lr

            # 每累计 grad_accumulation_steps 个 batch，更新一次参数
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress.update()

            # 记录当前 step 的 loss
            if distributed_state.is_main_process and log_step % cfg.log_freq == 0:
                if writer is not None:
                    writer.add_scalar("train/loss", loss_value, log_step)
                    writer.add_scalar("train/lr", scheduler.get_last_lr()[0], log_step)
                progress.set_postfix({"loss": f"{loss_value:.4f}"})

            # 定期保存 checkpoint，便于恢复训练和中途评估
            if cfg.save_freq > 0 and gradient_step_idx > 0 and log_step % cfg.save_freq == 0:
                save_training_checkpoint(
                    cfg=cfg, run_dir=run_dir, log_step=log_step, vla=vla,
                    processor=processor, action_head=action_head,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    train_dataset=train_dataset, distributed_state=distributed_state,
                    optimizer=optimizer, scheduler=scheduler,
                )

            # 定期验证；验证后记得切回 train 模式
            if cfg.use_val_set and log_step > 0 and log_step % cfg.val_freq == 0:
                run_validation(
                    vla=vla, action_head=action_head,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    val_dataloader=val_dataloader, device_id=device_id,
                    cfg=cfg, num_patches=NUM_PATCHES, log_step=log_step,
                    distributed_state=distributed_state, writer=writer,
                )
                vla.train()

            # Stop at max_steps
            if log_step >= cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break

    # Save final checkpoint
    final_step = log_step
    print(f"Saving final checkpoint at step {final_step}...")
    save_training_checkpoint(
        cfg=cfg, run_dir=run_dir, log_step=final_step, vla=vla,
        processor=processor, action_head=action_head,
        proprio_projector=proprio_projector if cfg.use_proprio else None,
        train_dataset=train_dataset, distributed_state=distributed_state,
        optimizer=optimizer, scheduler=scheduler,
    )

    if writer is not None:
        writer.close()
    print("Training complete.")


if __name__ == "__main__":
    set_seed(seed=0)
    finetune()
