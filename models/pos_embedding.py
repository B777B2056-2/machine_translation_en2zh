#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import torch


class LearnedPositionalEmbeddingWithWordEmbedding(torch.nn.Module):
  """绝对位置编码：位置编码与词嵌入矩阵均通过反向传播学习得到"""
  def __init__(self, max_position:int, word_dim:int):
    super(LearnedPositionalEmbeddingWithWordEmbedding, self).__init__()
    self.pos_embedding = torch.nn.Embedding(max_position, word_dim)

  def forward(self, input_vectors:torch.Tensor) -> torch.Tensor:
    """前向传播"""
    return self.pos_embedding(input_vectors)
