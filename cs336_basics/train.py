import os
import sys
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from typing import Optional, Tuple, Dict, Any

# 导入你的模型实现
from model import TransformerLM
from utils import get_batch, get_lr_cosine_schedule, save_checkpoint, load_checkpoint

'''
# 训练GPT-2 small
python train.py \
    --train-data ./data/train.bin \
    --val-data ./data/val.bin \
    --vocab-size 50257 \
    --d-model 768 \
    --num-heads 12 \
    --d-ff 3072 \
    --num-layers 12 \
    --context-length 1024 \
    --learning-rate 6e-4 \
    --batch-size 32 \
    --total-iters 100000 \
    --save-dir ./checkpoints/gpt2-small

# 从检查点恢复
python train.py \
    --train-data ./data/train.bin \
    --resume ./checkpoints/gpt2-small/checkpoint_0050000.pt \
    # ... 其他参数
'''

def estimate_loss(
    model: nn.Module,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    eval_iters: int,
    device: str
) -> float:
    """
    评估模型损失（无梯度）
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for _ in range(eval_iters):
            inputs, targets = get_batch(dataset, batch_size, context_length, device)
            logits = model(inputs)  # [batch, seq_len, vocab_size]
            
            # 计算损失
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
            total_loss += loss.item()
    
    model.train()
    return total_loss / eval_iters


def train(
    # 数据参数
    train_data_path: str,
    val_data_path: Optional[str],
    # 模型参数
    vocab_size: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
    num_layers: int,
    context_length: int,
    # 优化器参数
    learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    total_iters: int,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    # 训练参数
    batch_size: int,
    eval_interval: int,
    eval_iters: int,
    log_interval: int,
    save_interval: int,
    save_dir: str,
    device: str,
    resume_checkpoint: Optional[str] = None,
):
    """
    主训练循环
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载数据集（内存映射）
    print(f"加载训练数据: {train_data_path}")
    train_dataset = np.memmap(train_data_path, dtype=np.uint16, mode='r')
    
    if val_data_path:
        print(f"加载验证数据: {val_data_path}")
        val_dataset = np.memmap(val_data_path, dtype=np.uint16, mode='r')
    else:
        val_dataset = None
    
    # 创建模型
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        theta=10000.0,  # RoPE参数
    ).to(device)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 创建优化器
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=eps,
        weight_decay=weight_decay
    )
    
    # 恢复检查点
    start_iter = 0
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"恢复检查点: {resume_checkpoint}")
        start_iter, _ = load_checkpoint(resume_checkpoint, model, optimizer, device)
        print(f"从步数 {start_iter} 恢复训练")
    
    # 训练循环
    print("\n开始训练...")
    print("=" * 80)
    
    train_losses = []
    iter_time = 0.0
    best_val_loss = float('inf')
    
    for iter_num in range(start_iter, total_iters):
        iter_start = time.time()
        
        # 1. 调整学习率
        lr = get_lr_cosine_schedule(
            iter_num,
            learning_rate,
            min_learning_rate,
            warmup_iters,
            total_iters
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # 2. 采样batch
        inputs, targets = get_batch(train_dataset, batch_size, context_length, device)
        
        # 3. 前向传播
        logits = model(inputs)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )
        
        # 4. 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 5. 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # 6. 优化器步进
        optimizer.step()
        
        # 计时
        iter_time += time.time() - iter_start
        
        # 记录训练损失
        train_losses.append(loss.item())
        
        # 日志
        if iter_num % log_interval == 0:
            avg_loss = np.mean(train_losses[-log_interval:])
            print(f"步数 {iter_num:6d} | 损失 {avg_loss:.4f} | 学习率 {lr:.2e} | 时间 {iter_time:.2f}s")
            iter_time = 0.0
        
        # 评估
        if iter_num % eval_interval == 0 and val_dataset is not None:
            val_loss = estimate_loss(
                model, val_dataset, batch_size, context_length,
                eval_iters, device
            )
            print(f"步数 {iter_num:6d} | 验证损失 {val_loss:.4f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(save_dir, f'best_model.pt')
                save_checkpoint(
                    model, optimizer, iter_num, best_path,
                    config={
                        'vocab_size': vocab_size,
                        'd_model': d_model,
                        'num_heads': num_heads,
                        'd_ff': d_ff,
                        'num_layers': num_layers,
                        'context_length': context_length
                    },
                    val_loss=val_loss
                )
        
        # 定期保存检查点
        if iter_num % save_interval == 0 and iter_num > 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_{iter_num:07d}.pt')
            save_checkpoint(
                model, optimizer, iter_num, checkpoint_path,
                config={
                    'vocab_size': vocab_size,
                    'd_model': d_model,
                    'num_heads': num_heads,
                    'd_ff': d_ff,
                    'num_layers': num_layers,
                    'context_length': context_length
                }
            )
    
    print("\n训练完成!")
    print("=" * 80)
    
    # 保存最终模型
    final_path = os.path.join(save_dir, 'final_model.pt')
    save_checkpoint(
        model, optimizer, total_iters, final_path,
        config={
            'vocab_size': vocab_size,
            'd_model': d_model,
            'num_heads': num_heads,
            'd_ff': d_ff,
            'num_layers': num_layers,
            'context_length': context_length
        }
    )


def main():
    parser = argparse.ArgumentParser(description='训练Transformer语言模型')
    
    # 数据参数
    parser.add_argument('--train-data', type=str, required=True,
                        help='训练数据路径 (numpy memmap)')
    parser.add_argument('--val-data', type=str, default=None,
                        help='验证数据路径 (numpy memmap)')
    
    # 模型参数
    parser.add_argument('--vocab-size', type=int, default=50257,
                        help='词表大小')
    parser.add_argument('--d-model', type=int, default=768,
                        help='模型维度')
    parser.add_argument('--num-heads', type=int, default=12,
                        help='注意力头数')
    parser.add_argument('--d-ff', type=int, default=3072,
                        help='FFN内部维度')
    parser.add_argument('--num-layers', type=int, default=12,
                        help='Transformer层数')
    parser.add_argument('--context-length', type=int, default=1024,
                        help='上下文长度')
    
    # 优化器参数
    parser.add_argument('--learning-rate', type=float, default=6e-4,
                        help='最大学习率')
    parser.add_argument('--min-learning-rate', type=float, default=6e-5,
                        help='最小学习率')
    parser.add_argument('--warmup-iters', type=int, default=2000,
                        help='预热步数')
    parser.add_argument('--total-iters', type=int, default=100000,
                        help='总训练步数')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.95,
                        help='Adam beta2')
    parser.add_argument('--eps', type=float, default=1e-8,
                        help='Adam epsilon')
    parser.add_argument('--weight-decay', type=float, default=0.1,
                        help='权重衰减')
    
    # 训练参数
    parser.add_argument('--batch-size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--eval-interval', type=int, default=1000,
                        help='评估间隔')
    parser.add_argument('--eval-iters', type=int, default=200,
                        help='评估迭代次数')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='日志间隔')
    parser.add_argument('--save-interval', type=int, default=5000,
                        help='保存间隔')
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                        help='保存目录')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='设备')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    
    args = parser.parse_args()
    
    # 打印配置
    print("=" * 80)
    print("训练配置:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("=" * 80)
    
    # 开始训练
    train(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        context_length=args.context_length,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        warmup_iters=args.warmup_iters,
        total_iters=args.total_iters,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.eps,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        save_dir=args.save_dir,
        device=args.device,
        resume_checkpoint=args.resume
    )


if __name__ == '__main__':
    main()