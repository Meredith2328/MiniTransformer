import torch
import torch.nn as nn
import torch.nn.init as init
from jaxtyping import Float, Integer
from torch import Tensor


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
        init.trunc_normal_(
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
