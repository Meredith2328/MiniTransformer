import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.init import trunc_normal_
from math import sqrt
from einops import einsum, rearrange, reduce
from jaxtyping import Float, Integer, Int
from collections.abc import Callable, Iterable
from typing import Optional, Tuple, Union, BinaryIO, IO, Any
import numpy as np
import os

def get_lr_cosine_schedule(
    t: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    余弦学习率调度 - 严格按数学公式实现
    """
    if t < warmup_iters:
        # 预热: α_t = (t / Tw) * α_max
        return (t / warmup_iters) * max_learning_rate
    
    elif t <= cosine_cycle_iters:
        # 余弦退火: α_t = α_min + 1/2 * (1 + cos(π * (t - Tw)/(Tc - Tw))) * (α_max - α_min)
        import math
        progress = (t - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        cosine_term = math.cos(progress * math.pi)
        return min_learning_rate + 0.5 * (1 + cosine_term) * (max_learning_rate - min_learning_rate)
    
    else:
        # 稳定阶段: α_t = α_min
        return min_learning_rate

def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6
) -> None:
    """
    梯度裁剪：如果梯度的L2范数超过阈值，则按比例缩放
    
    Args:
        parameters: 可迭代的参数集合
        max_l2_norm: 最大允许的L2范数 M
        eps: 数值稳定性常数，防止除零
        
    Algorithm:
        1. 计算所有参数梯度的L2范数: ||g||₂ = √(Σ g_i²)
        2. 如果 ||g||₂ > M:
              缩放因子 = M / (||g||₂ + ε)
              每个梯度: g_i = g_i × 缩放因子
    """
    # 收集所有非空梯度
    grads = []
    for p in parameters:
        if p.grad is not None:
            grads.append(p.grad.detach())
    
    if not grads:
        return
    
    # 在同一个设备上计算总范数
    device = grads[0].device
    
    # 计算总范数: ||g||₂ = √(Σ ||g_i||₂²)
    total_norm = 0.0
    for g in grads:
        # 将梯度移到同一设备计算（如果需要）
        if g.device != device:
            g = g.to(device)
        total_norm += g.norm().item() ** 2
    
    total_norm = total_norm ** 0.5
    
    # 如果范数超过阈值，进行裁剪
    if total_norm > max_l2_norm:
        # 计算缩放因子: M / (||g||₂ + ε)
        clip_coef = max_l2_norm / (total_norm + eps)
        
        # 对每个梯度进行缩放
        for g in grads:
            g.mul_(clip_coef)

def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从数据集采样batch
    
    Args:
        dataset: 1D numpy数组，包含所有token IDs
        batch_size: 批次大小 B
        context_length: 上下文长度 m
        device: PyTorch设备字符串
    
    Returns:
        (inputs, targets): 两个形状为 (batch_size, context_length) 的张量
    """
    dataset_len = len(dataset)
    inputs = torch.empty(batch_size, context_length, dtype=torch.long)
    targets = torch.empty(batch_size, context_length, dtype=torch.long)
    
    for i in range(batch_size):
        start_idx = torch.randint(0, dataset_len-context_length, (1, )).item()
        input_seq = dataset[start_idx: start_idx+context_length]
        input_target = dataset[start_idx+1: start_idx+context_length+1]
        
        inputs[i] = torch.tensor(input_seq, dtype=torch.long)
        targets[i] = torch.tensor(input_target, dtype=torch.long)
    
    inputs = inputs.to(device=device)
    targets = targets.to(device=device)
    return (inputs, targets)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: Union[str, os.PathLike, BinaryIO, IO[bytes]],
    extra_state: Optional[dict[str, Any]] = None,
) -> None:
    """
    保存训练检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        iteration: 当前迭代步数
        out: 输出路径或文件对象
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    if extra_state is not None:
        checkpoint['extra_state'] = dict(extra_state)
    
    torch.save(checkpoint, out)


def load_checkpoint(
    src: Union[str, os.PathLike, BinaryIO, IO[bytes]],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    return_checkpoint: bool = False,
) -> Union[int, dict[str, Any]]:
    """
    加载训练检查点
    
    Args:
        src: 检查点路径或文件对象
        model: 模型（用于加载状态）
        optimizer: 优化器（用于加载状态）
    
    Returns:
        int: 保存时的迭代步数
    """
    checkpoint = torch.load(src)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if return_checkpoint:
        return checkpoint
    return checkpoint['iteration']
