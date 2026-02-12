from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from cs336_basics import bpe, model

def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """

    linear_layer = model.Linear(d_in, d_out)
    state_dict = {'weight': weights}
    linear_layer.load_state_dict(state_dict, strict=False)
    return linear_layer(in_features)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    embedding_layer = model.Embedding(vocab_size, d_model)
    embedding_layer.load_state_dict({'weight': weights})
    return embedding_layer(token_ids)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    swiglu = model.SwiGLU(d_model, d_ff)
    swiglu.load_state_dict({'w1.weight': w1_weight, 'w2.weight': w2_weight, 'w3.weight': w3_weight})
    return swiglu(in_features)


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    scaled_dot_product_attention = model.ScaledDotProductAttention()
    return scaled_dot_product_attention(Q, K, V, mask)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    mha = model.MultiHeadSelfAttention(
        d_model=d_model,
        num_heads=num_heads,
        use_rope=False
    )
    
    state_dict = {
        'W_q.weight': q_proj_weight,
        'W_k.weight': k_proj_weight, 
        'W_v.weight': v_proj_weight,
        'W_o.weight': o_proj_weight
    }
    mha.load_state_dict(state_dict)
    
    with torch.no_grad():
        output = mha(in_features)
    
    return output


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    mha = model.MultiHeadSelfAttention(
        d_model=d_model,
        num_heads=num_heads,
        use_rope=True,
        theta=theta,
        max_seq_len=max_seq_len
    )
    
    state_dict = {
        'W_q.weight': q_proj_weight,
        'W_k.weight': k_proj_weight,
        'W_v.weight': v_proj_weight,
        'W_o.weight': o_proj_weight
    }
    mha.load_state_dict(state_dict)
    
    with torch.no_grad():
        output = mha(in_features, token_positions)
    
    return output


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    rope = model.RoPE(theta, d_k, max_seq_len)
    with torch.no_grad():
        output = rope(in_query_or_key, token_positions)
    return output


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """

    block = model.TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        theta=theta,
        max_seq_len=max_seq_len
    )
    
    state_dict = {
        'norm1.weight': weights['ln1.weight'],
        'norm2.weight': weights['ln2.weight'],
        'attention.W_q.weight': weights['attn.q_proj.weight'],
        'attention.W_k.weight': weights['attn.k_proj.weight'],
        'attention.W_v.weight': weights['attn.v_proj.weight'],
        'attention.W_o.weight': weights['attn.output_proj.weight'],
        'ffn.w1.weight': weights['ffn.w1.weight'],
        'ffn.w2.weight': weights['ffn.w2.weight'],
        'ffn.w3.weight': weights['ffn.w3.weight'],
    }
    
    block.load_state_dict(state_dict)
    
    batch_size, seq_len = in_features.shape[0], in_features.shape[1]
    token_positions = torch.arange(seq_len, device=in_features.device)
    token_positions = token_positions.unsqueeze(0).expand(batch_size, -1)
    
    with torch.no_grad():
        output = block(in_features, token_positions)
    
    return output


def debug_transformer_weights(model, ref_weights, num_layers):
    """
    独立调试函数：对比TransformerLM模型权重与参考权重
    
    Args:
        model: 你的TransformerLM模型实例
        ref_weights: 测试传进来的weights字典
        num_layers: 层数
    """
    print("\n" + "=" * 80)
    print("🔍 TRANSFORMER LM 权重调试")
    print("=" * 80)
    
    total_mismatches = 0
    
    # 1. Token Embedding
    print("\n【1. Token Embedding】")
    print("-" * 50)
    model_w = model.token_embedding.weight
    ref_w = ref_weights['token_embeddings.weight']
    print(f"  模型: shape {tuple(model_w.shape)}")
    print(f"  参考: shape {tuple(ref_w.shape)}")
    if model_w.shape == ref_w.shape:
        diff = (model_w - ref_w).abs().max().item()
        print(f"  ✅ 形状匹配 | 最大差异: {diff:.6f}")
        if diff < 1e-5:
            print(f"  ✅ 数值一致")
        else:
            print(f"  ⚠️ 数值差异较大")
            total_mismatches += 1
    else:
        print(f"  ❌ 形状不匹配！")
        total_mismatches += 1
    
    # 2. 遍历每一层
    for layer_idx in range(num_layers):
        print(f"\n{'='*60}")
        print(f"【2.{layer_idx} Layer {layer_idx}】")
        print(f"{'='*60}")
        
        layer = model.layers[layer_idx]
        prefix = f'layers.{layer_idx}'
        
        # ----- Attention 权重 -----
        print("\n  📍 MultiHeadSelfAttention")
        
        # Q
        model_q = layer.attention.W_q.weight
        ref_q = ref_weights[f'{prefix}.attn.q_proj.weight']
        print(f"\n    [Q]")
        print(f"      模型: {tuple(model_q.shape)}")
        print(f"      参考: {tuple(ref_q.shape)}")
        if model_q.shape == ref_q.shape:
            diff = (model_q - ref_q).abs().max().item()
            print(f"      ✅ 形状匹配 | 最大差异: {diff:.6f}")
        elif model_q.shape == ref_q.T.shape:
            diff = (model_q - ref_q.T).abs().max().item()
            print(f"      🔄 需要转置 | 转置后差异: {diff:.6f}")
            total_mismatches += 1
        else:
            print(f"      ❌ 形状不匹配 (模型{model_q.shape} vs 参考{ref_q.shape})")
            total_mismatches += 1
        
        # K
        model_k = layer.attention.W_k.weight
        ref_k = ref_weights[f'{prefix}.attn.k_proj.weight']
        print(f"\n    [K]")
        print(f"      模型: {tuple(model_k.shape)}")
        print(f"      参考: {tuple(ref_k.shape)}")
        if model_k.shape == ref_k.shape:
            diff = (model_k - ref_k).abs().max().item()
            print(f"      ✅ 形状匹配 | 最大差异: {diff:.6f}")
        elif model_k.shape == ref_k.T.shape:
            diff = (model_k - ref_k.T).abs().max().item()
            print(f"      🔄 需要转置 | 转置后差异: {diff:.6f}")
            total_mismatches += 1
        else:
            print(f"      ❌ 形状不匹配")
            total_mismatches += 1
        
        # V
        model_v = layer.attention.W_v.weight
        ref_v = ref_weights[f'{prefix}.attn.v_proj.weight']
        print(f"\n    [V]")
        print(f"      模型: {tuple(model_v.shape)}")
        print(f"      参考: {tuple(ref_v.shape)}")
        if model_v.shape == ref_v.shape:
            diff = (model_v - ref_v).abs().max().item()
            print(f"      ✅ 形状匹配 | 最大差异: {diff:.6f}")
        elif model_v.shape == ref_v.T.shape:
            diff = (model_v - ref_v.T).abs().max().item()
            print(f"      🔄 需要转置 | 转置后差异: {diff:.6f}")
            total_mismatches += 1
        else:
            print(f"      ❌ 形状不匹配")
            total_mismatches += 1
        
        # O
        model_o = layer.attention.W_o.weight
        ref_o = ref_weights[f'{prefix}.attn.output_proj.weight']
        print(f"\n    [O]")
        print(f"      模型: {tuple(model_o.shape)}")
        print(f"      参考: {tuple(ref_o.shape)}")
        if model_o.shape == ref_o.shape:
            diff = (model_o - ref_o).abs().max().item()
            print(f"      ✅ 形状匹配 | 最大差异: {diff:.6f}")
        elif model_o.shape == ref_o.T.shape:
            diff = (model_o - ref_o.T).abs().max().item()
            print(f"      🔄 需要转置 | 转置后差异: {diff:.6f}")
            total_mismatches += 1
        else:
            print(f"      ❌ 形状不匹配")
            total_mismatches += 1
        
        # ----- FFN 权重 -----
        print("\n  📍 SwiGLU")
        
        # w1
        model_w1 = layer.ffn.w1.weight
        ref_w1 = ref_weights[f'{prefix}.ffn.w1.weight']
        print(f"\n    [w1]")
        print(f"      模型: {tuple(model_w1.shape)}")
        print(f"      参考: {tuple(ref_w1.shape)}")
        if model_w1.shape == ref_w1.shape:
            diff = (model_w1 - ref_w1).abs().max().item()
            print(f"      ✅ 形状匹配 | 最大差异: {diff:.6f}")
        elif model_w1.shape == ref_w1.T.shape:
            diff = (model_w1 - ref_w1.T).abs().max().item()
            print(f"      🔄 需要转置 | 转置后差异: {diff:.6f}")
            total_mismatches += 1
        else:
            print(f"      ❌ 形状不匹配 (期望 {ref_w1.shape} 或 {ref_w1.T.shape})")
            total_mismatches += 1
        
        # w2
        model_w2 = layer.ffn.w2.weight
        ref_w2 = ref_weights[f'{prefix}.ffn.w2.weight']
        print(f"\n    [w2]")
        print(f"      模型: {tuple(model_w2.shape)}")
        print(f"      参考: {tuple(ref_w2.shape)}")
        if model_w2.shape == ref_w2.shape:
            diff = (model_w2 - ref_w2).abs().max().item()
            print(f"      ✅ 形状匹配 | 最大差异: {diff:.6f}")
        elif model_w2.shape == ref_w2.T.shape:
            diff = (model_w2 - ref_w2.T).abs().max().item()
            print(f"      🔄 需要转置 | 转置后差异: {diff:.6f}")
            total_mismatches += 1
        else:
            print(f"      ❌ 形状不匹配")
            total_mismatches += 1
        
        # w3
        model_w3 = layer.ffn.w3.weight
        ref_w3 = ref_weights[f'{prefix}.ffn.w3.weight']
        print(f"\n    [w3]")
        print(f"      模型: {tuple(model_w3.shape)}")
        print(f"      参考: {tuple(ref_w3.shape)}")
        if model_w3.shape == ref_w3.shape:
            diff = (model_w3 - ref_w3).abs().max().item()
            print(f"      ✅ 形状匹配 | 最大差异: {diff:.6f}")
        elif model_w3.shape == ref_w3.T.shape:
            diff = (model_w3 - ref_w3.T).abs().max().item()
            print(f"      🔄 需要转置 | 转置后差异: {diff:.6f}")
            total_mismatches += 1
        else:
            print(f"      ❌ 形状不匹配")
            total_mismatches += 1
        
        # ----- RMSNorm -----
        print("\n  📍 RMSNorm")
        
        # norm1
        model_norm1 = layer.norm1.weight
        ref_norm1 = ref_weights[f'{prefix}.ln1.weight']
        print(f"\n    [norm1]")
        print(f"      模型: {tuple(model_norm1.shape)}")
        print(f"      参考: {tuple(ref_norm1.shape)}")
        if model_norm1.shape == ref_norm1.shape:
            diff = (model_norm1 - ref_norm1).abs().max().item()
            print(f"      ✅ 形状匹配 | 最大差异: {diff:.6f}")
        else:
            print(f"      ❌ 形状不匹配")
            total_mismatches += 1
        
        # norm2
        model_norm2 = layer.norm2.weight
        ref_norm2 = ref_weights[f'{prefix}.ln2.weight']
        print(f"\n    [norm2]")
        print(f"      模型: {tuple(model_norm2.shape)}")
        print(f"      参考: {tuple(ref_norm2.shape)}")
        if model_norm2.shape == ref_norm2.shape:
            diff = (model_norm2 - ref_norm2).abs().max().item()
            print(f"      ✅ 形状匹配 | 最大差异: {diff:.6f}")
        else:
            print(f"      ❌ 形状不匹配")
            total_mismatches += 1
    
    # 3. Final Layer Norm
    print(f"\n{'='*60}")
    print("【3. Final Layer Norm】")
    print(f"{'='*60}")
    model_final = model.final_norm.weight
    ref_final = ref_weights['ln_final.weight']
    print(f"  模型: {tuple(model_final.shape)}")
    print(f"  参考: {tuple(ref_final.shape)}")
    if model_final.shape == ref_final.shape:
        diff = (model_final - ref_final).abs().max().item()
        print(f"  ✅ 形状匹配 | 最大差异: {diff:.6f}")
    else:
        print(f"  ❌ 形状不匹配")
        total_mismatches += 1
    
    # 4. LM Head
    print(f"\n{'='*60}")
    print("【4. LM Head】")
    print(f"{'='*60}")
    model_head = model.lm_head.weight
    ref_head = ref_weights['lm_head.weight']
    print(f"  模型: {tuple(model_head.shape)}")
    print(f"  参考: {tuple(ref_head.shape)}")
    if model_head.shape == ref_head.shape:
        diff = (model_head - ref_head).abs().max().item()
        print(f"  ✅ 形状匹配 | 最大差异: {diff:.6f}")
    elif model_head.shape == ref_head.T.shape:
        diff = (model_head - ref_head.T).abs().max().item()
        print(f"  🔄 需要转置 | 转置后差异: {diff:.6f}")
        total_mismatches += 1
    else:
        print(f"  ❌ 形状不匹配")
        total_mismatches += 1
    
    # 5. 总结
    print("\n" + "=" * 80)
    print("📊 调试总结")
    print("=" * 80)
    if total_mismatches == 0:
        print("✅ 所有权重形状匹配！")
        print("   如果输出仍然不对，检查:")
        print("   - RoPE是否生效")
        print("   - 因果掩码是否正确")
        print("   - forward是否传递了token_positions")
    else:
        print(f"❌ 发现 {total_mismatches} 处形状不匹配")
        print("   需要根据上述 🔄 标记添加 .T 转置")
    
    print("\n" + "=" * 80)

def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    transformer_model = model.TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        theta=rope_theta
    )
    
    state_dict = {}
    
    # Token Embedding
    state_dict['token_embedding.weight'] = weights['token_embeddings.weight']
    
    # Transformer Block
    for layer_idx in range(num_layers):
        prefix = f'layers.{layer_idx}'
        
        # attention
        state_dict[f'layers.{layer_idx}.attention.W_q.weight'] = weights[f'{prefix}.attn.q_proj.weight']
        state_dict[f'layers.{layer_idx}.attention.W_k.weight'] = weights[f'{prefix}.attn.k_proj.weight']
        state_dict[f'layers.{layer_idx}.attention.W_v.weight'] = weights[f'{prefix}.attn.v_proj.weight']
        state_dict[f'layers.{layer_idx}.attention.W_o.weight'] = weights[f'{prefix}.attn.output_proj.weight']
        
        # FFN
        w1_weight = weights[f'{prefix}.ffn.w1.weight']  # (d_ff, d_model)
        state_dict[f'layers.{layer_idx}.ffn.w1.weight'] = w1_weight  # 转置成 (d_model, d_ff)
        
        w2_weight = weights[f'{prefix}.ffn.w2.weight']  # (d_model, d_ff)
        state_dict[f'layers.{layer_idx}.ffn.w2.weight'] = w2_weight  # 转置成 (d_ff, d_model)
        
        w3_weight = weights[f'{prefix}.ffn.w3.weight']  # (d_ff, d_model)
        state_dict[f'layers.{layer_idx}.ffn.w3.weight'] = w3_weight  # 转置成 (d_model, d_ff)
        
        # RMSNorm
        state_dict[f'layers.{layer_idx}.norm1.weight'] = weights[f'{prefix}.ln1.weight']
        state_dict[f'layers.{layer_idx}.norm2.weight'] = weights[f'{prefix}.ln2.weight']
    
    state_dict['final_norm.weight'] = weights['ln_final.weight']

    lm_head_weight = weights['lm_head.weight']
    state_dict['lm_head.weight'] = lm_head_weight
    
    # 加载权重
    transformer_model.load_state_dict(state_dict, strict=False)
    # debug_transformer_weights(transformer_model, weights, num_layers) # DEBUG2
    
    # DEBUG1: 对比state_dict和named_parameters
    # for name, _ in transformer_model.named_parameters():
    #     if name in state_dict:
    #         print(f"✅ {name}")
    #     else:
    #         print(f"❌ {name} 不在state_dict中")

    # 生成位置索引
    batch_size, seq_len = in_indices.shape
    device = in_indices.device
    token_positions = torch.arange(seq_len, device=device)
    token_positions = token_positions.unsqueeze(0).expand(batch_size, -1)
    
    # 前向传播
    with torch.no_grad():
        logits = transformer_model(in_indices, token_positions)
    
    return logits


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    rmsnorm = model.RMSNorm(d_model, eps=eps)
    rmsnorm.load_state_dict({'weight': weights})
    return rmsnorm(in_features)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    silu = model.SiLU()
    return silu(in_features)


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    raise NotImplementedError


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    softmax = model.Softmax()
    return softmax(in_features, dim)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    raise NotImplementedError


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    raise NotImplementedError


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    raise NotImplementedError


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    raise NotImplementedError


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    raise NotImplementedError


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    raise NotImplementedError


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    tokenizer = bpe.ByteLevelBPE()
    return tokenizer.train(input_path, vocab_size, special_tokens)
