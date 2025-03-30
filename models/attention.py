#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import torch
import torch.nn.functional as F


class ScaleDotProductAttention(torch.nn.Module):
  """注意力机制"""
  def __init__(self):
    super(ScaleDotProductAttention, self).__init__()
    self.softmax = torch.nn.Softmax(dim=-1) # 在每一行，即句子维度实施softmax

  def forward(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, mask=None):
    """注意力机制前向传播（自注意力：Q == K == V；交叉注意力：Q = 解码器产生的Q，K、V = 编码器产生的K、V）"""
    # 1. A = Q · K转置，A为每个词在上下文中的的注意力得分，A的每一行都符合N ~ (0, d_out)标准正态分布
    A = torch.matmul(Q, K.transpose(-1,-2)) # [batch_size, T, T]
    # 2. 对A每一行进行归一化
    # 即A1 = A/根号下d_out ，使得A1每一行都符合N ~ (0, 1)正态分布
    d_out = Q.size(-1)
    A1 = torch.div(A, torch.sqrt(torch.tensor(d_out, dtype=torch.float32)))  # [batch_size, T, T]
    # 3, 如果有掩码，则将注意力分数矩阵中对应掩码位置的值设为负无穷大
    if mask is not None:
      A1 = A1.masked_fill(mask == 0, -1e9)
    # 4. A2 = softmax(A1)，对A1每一行进行softmax转换，为每个词在上下文中的注意力系数
    A2 = self.softmax(A1) # [batch_size, T, T]
    # 5. 输出矩阵Z = A2 · V，R为经过注意力得分掩码后的词向量
    Z = torch.matmul(A2, V)  # [batch_size, T, d_out]
    return Z


class MultiHeadAttention(torch.nn.Module):
  """多头注意力机制"""
  def __init__(self, n_head:int, d_in:int, d_out:int):
    super(MultiHeadAttention, self).__init__()
    self.n_head = n_head
    self.d_out = d_out
    assert d_in % n_head == 0
    self.attention = ScaleDotProductAttention()
    self.Wq = torch.nn.Linear(d_in, d_out, bias=False)  # [d_in, d_out]
    self.Wk = torch.nn.Linear(d_in, d_out, bias=False)  # [d_in, d_out]
    self.Wv = torch.nn.Linear(d_in, d_out, bias=False)  # [d_in, d_out]
    self.Wo = torch.nn.Linear(d_out, d_out, bias=False)  # [d_out, d_out]

  def __split_heads(self, x: torch.Tensor):
    """将张量拆分为多头"""
    batch_size, T, _ = x.shape
    # [batch_size, n_head, T, d_out/n_head]
    return x.view(batch_size, T, self.n_head, self.d_out // self.n_head).transpose(1, 2)

  def __expand_mask(self, mask: torch.Tensor):
    """扩展掩码以适配多头注意力"""
    expanded_mask = mask.unsqueeze(1)  # [batch, n_head, T, T]
    return expanded_mask

  def forward(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, mask=None):
    """多头注意力机制前向传播"""
    # 1. 输入的Q、K、V矩阵与Wq、Wk、Wv相乘，Q、K、V拆分为多头
    Q = self.__split_heads(self.Wq(Q))  # [batch_size, n_head, T, d_out/n_head]
    K = self.__split_heads(self.Wk(K))  # [batch_size, n_head, T, d_out/n_head]
    V = self.__split_heads(self.Wv(V))  # [batch_size, n_head, T, d_out/n_head]
    # 2. 计算注意力
    multi_head_mask = None
    if mask is not None:
      multi_head_mask = self.__expand_mask(mask)
    Z = self.attention(Q=Q, K=K, V=V, mask=multi_head_mask)  # [batch_size, n_head, T, d_out/n_head]
    # 3. 拼接所有头的注意力得分（按d_out/n_head维度拼接）
    batch_size, _, T, _ = Z.shape
    Z = Z.transpose(1, 2).contiguous().view(batch_size, T, -1)  # [batch_size, T, d_out]
    # 4. 线性投影
    output = self.Wo(Z) # [batch_size, T, d_out]
    return output

