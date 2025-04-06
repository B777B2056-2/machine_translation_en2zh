#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import torch


class LearnedPositionalEmbeddingWithWordEmbedding(torch.nn.Module):
  """词嵌入 + 可学习位置编码"""
  def __init__(self, vocab_size: int, word_dim: int, max_seq_len: int):
    super().__init__()
    # 词嵌入层（输入token索引）
    self.word_embedding = torch.nn.Embedding(vocab_size, word_dim)
    # 位置编码层（输入位置索引）
    self.pos_embedding = torch.nn.Embedding(max_seq_len, word_dim)

  def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
    # 词嵌入
    word_emb = self.word_embedding(input_ids)  # [batch, seq, dim]
    # 生成位置索引 [0, 1, 2, ..., seq_len-1]
    positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
    # 位置编码
    pos_emb = self.pos_embedding(positions)  # [1, seq, dim]
    return word_emb + pos_emb  # 按元素相加
