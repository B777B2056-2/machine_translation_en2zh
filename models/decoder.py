#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import torch
from typing import Tuple
from models.ffn import FeedForwardNet
from models.attention import MultiHeadAttention


class KVCacheManager(object):
  """推理过程中KV缓存管理器"""
  def __init__(self):
    self.cache = None

  def update(self, key_value:Tuple[torch.Tensor, torch.Tensor]) -> None:
    self.cache = key_value

  def get_latest(self) -> Tuple[torch.Tensor, torch.Tensor]:
    return self.cache


class TransformerDecoder(torch.nn.Module):
  """Transformer解码器"""
  def __init__(self, n_head:int, word_dim:int):
    super(TransformerDecoder, self).__init__()
    # kv cache管理器
    self.attn_kv_cache = KVCacheManager()
    # 多头注意力（掩码）
    self.masked_multi_head_attn = MultiHeadAttention(n_head=n_head, d_in=word_dim, d_out=word_dim)
    # 多头注意力（掩码）层归一化
    self.masked_multi_head_attn_layer_norm = torch.nn.LayerNorm(normalized_shape=word_dim)
    # 多头注意力
    self.multi_head_attn = MultiHeadAttention(n_head=n_head, d_in=word_dim, d_out=word_dim)
    # 多头注意力层归一化
    self.multi_head_attn_layer_norm = torch.nn.LayerNorm(normalized_shape=word_dim)
    # 前馈神经网络
    self.ffn = FeedForwardNet(d_out=word_dim, n_hidden=word_dim*4)
    # 前馈层归一化
    self.ffn_layer_norm = torch.nn.LayerNorm(normalized_shape=word_dim)

  def forward(self, x:torch.Tensor, encoder_outputs:torch.Tensor, mask=None, use_cache:bool=False) -> torch.Tensor:
    # x：[batch_size, T, word_dim]，encoder_outputs：[batch_size, S, word_dim]
    # 1. 多头注意力（掩码）
    residual = x
    if use_cache:
      cache = self.attn_kv_cache.get_latest()
      masked_multi_head_attn_output, new_cache = self.masked_multi_head_attn(Q=x, K=x, V=x, mask=mask, past_key_value=cache) # [batch_size, T, word_dim]
      self.attn_kv_cache.update(new_cache)
    else:
      masked_multi_head_attn_output, new_cache = self.masked_multi_head_attn(Q=x, K=x, V=x, mask=mask, past_key_value=None)
    # 2. Add & Norm
    x = self.masked_multi_head_attn_layer_norm(residual + masked_multi_head_attn_output) # [batch_size, T, word_dim]
    # 3. 多头自注意力
    # 注意此处Q为解码器注意力分数，K与V均为编码器输出
    # 目的：在Decoder的时候，每一位单词都可以利用到Encoder所有单词的信息
    residual = x
    multi_head_attn_output, _ = self.multi_head_attn(Q=x, K=encoder_outputs, V=encoder_outputs) # [batch_size, T, word_dim]
    # 4. Add & Norm
    multi_head_attn_output = self.multi_head_attn_layer_norm(residual + multi_head_attn_output) # [batch_size, T, word_dim]
    # 5. 前馈神经网络
    ffn_output = self.ffn(multi_head_attn_output)  # [batch_size, T, word_dim]
    # 6. Add & Norm
    output = self.ffn_layer_norm(multi_head_attn_output+ffn_output)  # [batch_size, T, word_dim]
    return output

