import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union


def apply_temperature(
    logits: torch.Tensor,
    temperature: float
) -> torch.Tensor:
    """
    应用温度缩放
    
    softmax(v, τ) = exp(v/τ) / Σ exp(v/τ)
    
    Args:
        logits: 原始logits [vocab_size]
        temperature: 温度参数 τ
        
    Returns:
        缩放后的logits
    """
    if temperature == 0:
        # 温度=0时，直接取argmax（确定性的）
        return logits
    return logits / temperature


def top_p_filtering(
    probs: torch.Tensor,
    top_p: float
) -> torch.Tensor:
    """
    Top-p (nucleus) 采样过滤
    
    保留累积概率 >= top_p 的最小集合
    
    Args:
        probs: 概率分布 [vocab_size]
        top_p: 阈值 p (0 < p ≤ 1)
        
    Returns:
        过滤后的概率分布（未选择的词概率设为0）
    """
    # 按概率降序排序
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    # 计算累积概率
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # 找到累积概率 >= top_p 的最小集合
    # 移除累积概率超过 top_p 的词
    remove_mask = cumsum_probs > top_p
    # 但至少保留一个词
    remove_mask[..., 1:] = remove_mask[..., :-1].clone()
    remove_mask[..., 0] = False
    
    # 创建过滤后的分布
    filtered_probs = probs.clone()
    filtered_probs[sorted_indices[remove_mask]] = 0
    
    # 重新归一化
    if filtered_probs.sum() > 0:
        filtered_probs = filtered_probs / filtered_probs.sum()
    
    return filtered_probs


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: Optional[float] = None
) -> int:
    """
    从logits中采样下一个token
    
    Args:
        logits: 模型输出的logits [vocab_size]
        temperature: 温度参数
        top_p: top-p采样阈值
        
    Returns:
        采样的token ID
    """
    # 1. 应用温度
    if temperature > 0:
        logits = apply_temperature(logits, temperature)
    else:
        # 温度=0：直接取argmax
        return torch.argmax(logits).item()
    
    # 2. 转换为概率
    probs = F.softmax(logits, dim=-1)
    
    # 3. 应用top-p过滤
    if top_p is not None and 0 < top_p < 1:
        probs = top_p_filtering(probs, top_p)
    
    # 4. 采样
    return torch.multinomial(probs, num_samples=1).item()


def generate(
    model: nn.Module,
    prompt: Union[str, List[int]],
    tokenizer=None,  # 如果有tokenizer，可以传进来处理字符串
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    eos_token_id: Optional[int] = None,
    device: str = 'cpu'
) -> List[int]:
    """
    从语言模型生成文本
    
    Args:
        model: Transformer语言模型
        prompt: 提示词（字符串或token ID列表）
        tokenizer: 分词器（用于处理字符串输入）
        max_new_tokens: 最大生成token数
        temperature: 温度参数
        top_p: top-p采样阈值
        eos_token_id: 结束符token ID
        device: 设备
        
    Returns:
        生成的token ID列表（包含prompt）
    """
    model.eval()
    
    # 1. 处理prompt
    if isinstance(prompt, str):
        assert tokenizer is not None, "需要tokenizer来处理字符串prompt"
        input_ids = tokenizer.encode(prompt)
    else:
        input_ids = prompt.copy()
    
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    # 2. 逐token生成
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 前向传播
            logits = model(input_ids)  # [1, seq_len, vocab_size]
            
            # 取最后一个位置的logits
            next_token_logits = logits[0, -1, :]  # [vocab_size]
            
            # 采样下一个token
            next_token = sample_next_token(
                next_token_logits,
                temperature=temperature,
                top_p=top_p
            )
            
            # 添加到序列
            input_ids = torch.cat([
                input_ids,
                torch.tensor([[next_token]], device=device)
            ], dim=1)
            
            # 检查是否遇到结束符
            if eos_token_id is not None and next_token == eos_token_id:
                break
    
    return input_ids[0].tolist()


def generate_batched(
    model: nn.Module,
    prompts: List[Union[str, List[int]]],
    tokenizer=None,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    eos_token_id: Optional[int] = None,
    device: str = 'cpu'
) -> List[List[int]]:
    """
    批量生成（带填充）
    """
    model.eval()
    
    # 处理所有prompt
    input_ids_list = []
    for prompt in prompts:
        if isinstance(prompt, str):
            assert tokenizer is not None
            ids = tokenizer.encode(prompt)
        else:
            ids = prompt.copy()
        input_ids_list.append(ids)
    
    # 找到最长prompt
    max_prompt_len = max(len(ids) for ids in input_ids_list)
    
    # 填充到相同长度
    padded_ids = []
    attention_masks = []
    for ids in input_ids_list:
        padding_len = max_prompt_len - len(ids)
        padded = ids + [0] * padding_len  # 假设0是padding token
        mask = [1] * len(ids) + [0] * padding_len
        padded_ids.append(padded)
        attention_masks.append(mask)
    
    input_ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
    attention_mask = torch.tensor(attention_masks, dtype=torch.long, device=device)
    
    batch_size = len(prompts)
    finished = [False] * batch_size
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 前向传播
            logits = model(input_ids)  # [batch, seq_len, vocab_size]
            
            # 取最后一个位置
            next_token_logits = logits[:, -1, :]  # [batch, vocab_size]
            
            # 对每个样本采样
            next_tokens = []
            for i in range(batch_size):
                if finished[i]:
                    # 已经结束的样本，用padding token填充
                    next_tokens.append(0)
                else:
                    token = sample_next_token(
                        next_token_logits[i],
                        temperature=temperature,
                        top_p=top_p
                    )
                    next_tokens.append(token)
                    
                    # 检查是否结束
                    if eos_token_id is not None and token == eos_token_id:
                        finished[i] = True
            
            # 添加到序列
            next_tokens = torch.tensor(next_tokens, device=device).unsqueeze(1)
            input_ids = torch.cat([input_ids, next_tokens], dim=1)
            
            # 如果全部结束，提前退出
            if all(finished):
                break
    
    # 返回所有生成的序列
    return [seq.tolist() for seq in input_ids]


def run_generation(
    model: nn.Module,
    prompt: Union[str, List[int]],
    tokenizer=None,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    eos_token_id: Optional[int] = None,
    device: str = 'cpu'
) -> Union[str, List[int]]:
    """
    适配器函数：生成文本
    """
    generated_ids = generate(
        model, prompt, tokenizer,
        max_new_tokens, temperature, top_p,
        eos_token_id, device
    )
    
    # 如果提供了tokenizer，返回字符串；否则返回token IDs
    if tokenizer is not None and isinstance(prompt, str):
        return tokenizer.decode(generated_ids)
    return generated_ids
    