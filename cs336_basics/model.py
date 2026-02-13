import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.init import trunc_normal_
from math import sqrt
from einops import einsum, rearrange, reduce
from jaxtyping import Float, Integer, Int

class Linear(nn.Module):
    '''Construct a (out_features, in_features) linear layer.
    forward: y = xW^T.
    '''


    def __init__(self, in_features, out_features, device=None, dtype=None):
        # 注意对外提供的接口是: in在左, out在右.
        # 内部存储是反的, forward实际实现是反的, 但外部不需要知道这个事情.
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        std = sqrt(2.0 / (in_features + out_features))
        self.weight = trunc_normal_(
            nn.Parameter(
                torch.empty(out_features, in_features, device=device, dtype=dtype)
            ),
            mean = 0.0,
            std = std,
            a = -3.0 * std,
            b = 3.0 * std
        )

    def forward(self, x):
        if x.device != self.weight.device:
            self.weight = self.weight.to(x.device)

        # return x @ self.weight.T
        return einsum(x, self.weight, '... in_features, out_features in_features -> ... out_features')

class Embedding(nn.Module):
    """
    Embedding layer that maps integer token IDs to dense vectors.
    """
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Create embedding matrix parameter (vocab_size, d_model)
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using truncated normal distribution N(0,1) truncated at [-3,3]."""
        trunc_normal_(
            self.weight,
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0
        )
    
    def forward(self, token_ids: Integer[Tensor, "batch seq_len"]) -> Float[Tensor, "batch seq_len embedding_dim"]:
        """
        Lookup embedding vectors for the given token IDs.
        
        Args:
            token_ids: Long tensor of token IDs with shape (batch_size, sequence_length)
            
        Returns:
            Embedding vectors with shape (batch_size, sequence_length, embedding_dim)
        """
        # Check that token IDs are within vocabulary range
        if token_ids.max() >= self.num_embeddings or token_ids.min() < 0:
            raise ValueError(f"Token IDs must be between 0 and {self.num_embeddings - 1}")
        
        # 直接索引查出token_ids对应的embedding向量
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    """RMSNorm(a_i) = a_i * g_i / RMS(a)
    RMS(a) is the std without "-avg" step.
    g_i: (d_model) is a learnable "gain" parameter.
    eps: 1e-5. Hyperparameter.
    """

    def __init__(self, d_model, eps = 1e-5, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.d_model = d_model
        self.eps = 1e-5

    def forward(self, x):
        # upcast input to torch.float32
        # to prevent overflow in x^2
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # mean in d_model
        rms_a = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        result = x / rms_a * self.weight
        return result.to(in_dtype)

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff = None, device=None,dtype=None):
        super().__init__()
        if d_ff == None:
            # 8 / 3 * d_model并向上取整到64的倍数
            d_ff = int((8 / 3) * d_model)
            d_ff = ((d_ff + 63) // 64) * 64

        # in在左, out在右
        # 内部存储是反的, 但 **在SwiGLU层, 不需要考虑这个事情**
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x):
        orig_dtype = x.dtype
        x = x.float()
        silu = SiLU()
        output = self.w2((silu(self.w1(x)) * self.w3(x)))
        return output.to(orig_dtype)

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # precalc frequencies
        # freqs[k] = Θ^(-2k/d_k) for 2k = 0, 2, ..., d_k - 2
        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        positions = torch.arange(max_seq_len, device=device).float()

        # Cartesian product
        angles = einsum(positions, freqs, 'seq, half -> seq half')

        self.register_buffer('cos', torch.cos(angles), persistent=False)
        self.register_buffer('sin', torch.sin(angles), persistent=False)
        
    def forward(self, x, token_positions):
        orig_dtype = x.dtype
        device = x.device

        if self.cos.device != device:
            self.cos = self.cos.to(device)
            self.sin = self.sin.to(device)

        x_reshaped = rearrange(x, '... seq (half two) -> ... seq half two', two=2)

        positions = token_positions.long()
        cos = self.cos[positions]
        sin = self.sin[positions]

        x1, x2 = x_reshaped[..., 0], x_reshaped[..., 1]
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        x_rotated = rearrange(
            [rotated_x1, rotated_x2],
            'two ... seq half -> ... seq (half two)',
            two=2
        )
        return x_rotated.to(orig_dtype)

class Softmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dim):
        '''Numerical stability trick:
        softmax(x) = softmax(x - max(x))
        since
        exp(x_i) / Σexp(x_j) = exp(x_i - c) / Σexp(x_j - c)'''
        x -= x.max(dim=dim, keepdim=True)[0] # [0] for max val, [1] for index
        return torch.exp(x) / (torch.exp(x).sum(dim=dim, keepdim=True))

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask):
        d_k = Q.shape[-1]
        scores = einsum(Q, K, '... q d_k, ... k d_k -> ... q k') / (d_k ** 0.5)
        if mask is not None: # [seq_len, seq_len]
            while mask.dim() < scores.dim():
                mask = mask.unsqueeze(0)
            scores = scores.masked_fill(~mask, float('-inf'))
        softmax = Softmax()
        attn_weights = softmax(scores, dim=-1)
        return einsum(attn_weights, V, '... q k, ... k d -> ... q d')
    
class MultiHeadSelfAttention(nn.Module):
    """
    MHA. Use RoPE or not.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        use_rope: bool = False,
        theta: float = 10000.0,
        max_seq_len: int = 2048,
        device=None,
        dtype=None
    ):
        super().__init__()
        
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.use_rope = use_rope
        
        # 线性投影层（无偏置）
        self.W_q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_k = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_v = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_o = Linear(d_model, d_model, device=device, dtype=dtype)
        
        # RoPE（如果需要）
        if use_rope:
            self.rope = RoPE(
                theta=theta,
                d_k=self.d_k,
                max_seq_len=max_seq_len,
                device=device
            )
        
        # 因果掩码
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).bool()
        self.register_buffer('causal_mask', causal_mask, persistent=False)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            std = (2.0 / (module.in_features + module.out_features)) ** 0.5
            trunc_normal_(
                module.weight,
                mean=0.0,
                std=std,
                a=-3 * std,
                b=3 * std
            )
    
    def forward(
        self,
        x: Float[Tensor, "... seq_len d_model"],
        token_positions: Int[Tensor, "... seq_len"] | None = None
    ) -> Float[Tensor, "... seq_len d_model"]:
        """
        前向传播
        """
        device = x.device
        seq_len = x.shape[-2]
        
        # 1. Q/K/V投影
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        
        # 2. 多头拆分
        q = rearrange(q, '... s (h d) -> ... h s d', h=self.num_heads)
        k = rearrange(k, '... s (h d) -> ... h s d', h=self.num_heads)
        v = rearrange(v, '... s (h d) -> ... h s d', h=self.num_heads)
        
        # 3. RoPE（如果需要）
        if self.use_rope:
            if token_positions is None:
                # 自动生成位置索引
                batch_shape = x.shape[:-2]
                token_positions = torch.arange(seq_len, device=device)
                token_positions = token_positions.expand(*batch_shape, -1)
            
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
            # V不应用RoPE
        
        # 4. 缩放点积注意力 + 因果掩码
        scale = self.d_k ** 0.5
        scores = einsum(q, k, '... h q d, ... h k d -> ... h q k') / scale
        
        mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(~mask, float('-inf'))
        
        attn_weights = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        attn_output = einsum(attn_weights, v, '... h q k, ... h k d -> ... h q d')
        
        # 5. 多头合并
        attn_output = rearrange(attn_output, '... h s d -> ... s (h d)')
        
        # 6. 输出投影
        output = self.W_o(attn_output)
        
        return output

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float = 10000.0,
        max_seq_len: int = 2048,
        device=None,
        dtype=None
    ):
        super().__init__()
        
        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attention = MultiHeadSelfAttention(
            d_model, num_heads, True, theta, max_seq_len, device, dtype
        )
        
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(self, x, token_positions=None):
        x = x + self.attention(self.norm1(x), token_positions)
        x = x + self.ffn(self.norm2(x))
        return x

class TransformerLM(nn.Module):
    """
    Transformer Language Model
    """
    
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        theta: float = 10000.0,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        
        # Token Embedding
        self.token_embedding = Embedding(
            vocab_size, 
            d_model, 
            device=device, 
            dtype=dtype
        )
        
        # Transformer Blocks - 注意变量名改为layers以匹配adapter
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                theta=theta,
                max_seq_len=context_length,
                device=device,
                dtype=dtype
            )
            for _ in range(num_layers)
        ])
        
        # Final Layer Norm
        self.final_norm = RMSNorm(
            d_model, 
            device=device, 
            dtype=dtype
        )
        
        # Output Projection (权重绑定)
        self.lm_head = Linear(
            d_model, 
            vocab_size, 
            device=device, 
            dtype=dtype
        )
        
        # 罪魁祸首, 不应该用权重绑定!
        # self.lm_head.weight = self.token_embedding.weight
    
    def forward(
        self,
        token_ids: Int[Tensor, "batch seq_len"],
        token_positions: Int[Tensor, "batch seq_len"] | None = None
    ) -> Float[Tensor, "batch seq_len vocab_size"]:
        """
        前向传播
        """
        x = self.token_embedding(token_ids)  # [batch, seq_len, d_model]
        
        for layer in self.layers:
            x = layer(x, token_positions)
        
        x = self.final_norm(x)
        logits = self.lm_head(x)  # [batch, seq_len, vocab_size]
        
        return logits
        
class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        # 1) calc log-sum-exp
        # batch "1" since 1 only automatically added in the front
        max_vals = reduce(inputs, 'batch vocab -> batch 1', reduction='max')
        inputs_stable = inputs - max_vals # (batch, vocab)
        
        log_sum_exp = reduce(
            inputs_stable.exp(),
            'batch vocab -> batch',
            reduction='sum'
        ).log() + max_vals
        
        # 2) get the predicted vals for correct labels
        # eg. targets = torch.tensor([2, 0, 3])
        # logits = torch.tensor([
        #     [0.1, 0.2, '0.3', 0.4, 0.5],
        #     ['1.0', 1.1, 1.2, 1.3, 1.4],
        #     [2.0, 2.1, 2.2, '2.3', 2.4]
        # ])
        # we take torch.tensor([0.3, 1.0, 2.3])
        target_logits = inputs.gather(
            dim=-1,
            index=targets.unsqueeze(-1)
        ).squeeze(-1)

        # 3) calc batch mean loss
        losses = log_sum_exp - target_logits
        return losses.mean()
