import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.init import trunc_normal_
from math import sqrt
from einops import einsum, rearrange
from jaxtyping import Float, Integer

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
    RMS(a) = \sqrt(\sum_{i=1}^{d_model} a_i^2 + eps)
    which is the std without "-avg" step.
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
        silu = SiLU()
        return self.w2((silu(self.w1(x)) * self.w3(x)))

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # precalc frequencies
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
