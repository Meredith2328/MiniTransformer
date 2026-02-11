import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from math import sqrt
from einops import einsum

class Linear(nn.Module):
    '''Construct a (out_features, in_features) linear layer.
    forward: y = xW^T.
    '''

    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        std = sqrt(2.0 / (in_features + out_features))
        self.weights = trunc_normal_(
            nn.Parameter(
                torch.empty(out_features, in_features, device=device, dtype=dtype)
            ),
            mean = 0.0,
            std = std,
            a = -3.0 * std,
            b = 3.0 * std
        )

    def forward(self, x):
        if x.device != self.weights.device:
            self.weights = self.weights.to(x.device)

        # return x @ self.weights.T
        return einsum(x, self.weights, '... in_features, out_features in_features -> ... out_features')
