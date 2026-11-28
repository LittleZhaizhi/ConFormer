"""Conditional attention across spatial and historical axes."""

import torch
from torch import nn
from torch.nn import functional as F


class DualAxisAttention(nn.Module):
    """ConFormer Eq. (4) and (7) for [batch, time, node, feature]."""

    def __init__(self, model_dim, condition_dim, num_heads):
        super().__init__()
        if num_heads <= 0 or model_dim % num_heads:
            raise ValueError("The model width must be divisible by the number of heads")
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.query = nn.Linear(model_dim, model_dim, bias=False)
        self.key_value = nn.Linear(model_dim + condition_dim, 2 * model_dim, bias=False)
        self.fusion = nn.Sequential(
            nn.Linear(2 * model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim),
        )

    def project(self, x, condition):
        condition = condition.expand(*x.shape[:-1], condition.shape[-1])
        key, value = self.key_value(torch.cat((x, condition), dim=-1)).chunk(2, dim=-1)
        return tuple(
            tensor.unflatten(-1, (self.num_heads, self.head_dim))
            for tensor in (self.query(x), key, value)
        )

    @staticmethod
    def attend(query, key, value):
        # SDPA expects [groups, heads, sequence, head features].
        result = F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        )
        return result.transpose(1, 2).flatten(-2)

    def attend_space(self, query, key, value):
        # One independent sequence of nodes for each batch item and time.
        batch, steps = query.shape[:2]
        return self.attend(
            *(tensor.flatten(0, 1) for tensor in (query, key, value))
        ).unflatten(0, (batch, steps))

    def attend_time(self, query, key, value):
        # One independent sequence of history steps for each batch item and node.
        batch, _, nodes = query.shape[:3]
        return self.attend(
            *(tensor.transpose(1, 2).flatten(0, 1) for tensor in (query, key, value))
        ).unflatten(0, (batch, nodes)).transpose(1, 2)

    def forward(self, x, condition):
        qkv = self.project(x, condition)
        space = self.attend_space(*qkv)
        time = self.attend_time(*qkv)
        return self.fusion(torch.cat((space, time), dim=-1))
