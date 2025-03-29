#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import torch


class FeedForwardNet(torch.nn.Module):
  """前馈神经网络"""
  def __init__(self, d_out:int, n_hidden:int):
    super(FeedForwardNet, self).__init__()
    self.linear1 = torch.nn.Linear(d_out, n_hidden)
    self.relu = torch.nn.ReLU()
    self.linear2 = torch.nn.Linear(n_hidden, d_out)

  def forward(self, x:torch.Tensor) -> torch.Tensor:
    x = self.linear1(x)
    x = self.relu(x)
    x = self.linear2(x)
    return x
