#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import torch
from models.ffn import FeedForwardNet
from models.attention import MultiHeadAttention


class TransformerEncoder(torch.nn.Module):
  """Transformer编码器"""
  def __init__(self, n_head:int, word_dim:int):
    super(TransformerEncoder, self).__init__()
    # 多头注意力
    self.multi_head_attn = MultiHeadAttention(n_head=n_head, d_in=word_dim, d_out=word_dim)
    # 多头注意力层归一化
    self.multi_head_attn_layer_norm = torch.nn.LayerNorm(normalized_shape=word_dim)
    # 前馈神经网络
    self.ffn = FeedForwardNet(d_out=word_dim, n_hidden=word_dim*4)
    # 前馈层归一化
    self.ffn_layer_norm = torch.nn.LayerNorm(normalized_shape=word_dim)

  def forward(self, pos:torch.Tensor) -> torch.Tensor:
    # 设pos：[batch_size, T, word_dim]
    # 1. 多头注意力
    multi_head_attn_output = self.multi_head_attn(Q=pos, K=pos, V=pos) # [batch_size, T, word_dim]
    # 2. Add & Norm
    multi_head_attn_output = self.multi_head_attn_layer_norm(pos + multi_head_attn_output) # [batch_size, T, word_dim]
    # 3. 前馈神经网络
    ffn_output = self.ffn(multi_head_attn_output)  # [batch_size, T, word_dim]
    # 4. Add & Norm
    output = self.ffn_layer_norm(multi_head_attn_output+ffn_output)  # [batch_size, T, word_dim]
    return output

