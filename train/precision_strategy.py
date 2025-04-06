#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import torch
from torch.cuda.amp import GradScaler
from abc import ABC, abstractmethod
from typing import Tuple, Union, Dict, Any


class AbstractPrecisionStrategy(ABC):
  """训练时使用的精度策略"""
  def __init__(self, net:torch.nn.Module, criterion:Any, optimizer:torch.optim, tgt_vocab_size:int):
    self.net = net
    self.criterion = criterion
    self.optimizer = optimizer
    self.tgt_vocab_size = tgt_vocab_size

  @abstractmethod
  def do_train_one_batch(self, encoder_inputs:torch.Tensor,
                         decoder_inputs:torch.Tensor, decoder_outputs:torch.Tensor) -> Tuple[torch.Tensor, float]:
    raise NotImplementedError


class BF16PrecisionStrategy(AbstractPrecisionStrategy):
  """bf16精度训练策略"""
  def __init__(self, *args, **kwargs):
    super(BF16PrecisionStrategy, self).__init__(*args, **kwargs)
    self.scaler = GradScaler()

  def load_scaler_state(self, scaler_state:Dict[str, Any]) -> None:
    """加载梯度缩放器状态"""
    self.scaler = GradScaler()
    self.scaler.load_state_dict(scaler_state)

  def do_train_one_batch(self, encoder_inputs:torch.Tensor,
                         decoder_inputs:torch.Tensor, decoder_outputs:torch.Tensor) -> Tuple[torch.Tensor, float]:
    """bf16训练时反向传播"""
    self.optimizer.zero_grad()
    device_type = next(self.net.parameters()).device.type
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
      probs = self.net(encoder_inputs, decoder_inputs)
      loss = self.criterion(
        probs.contiguous().view(-1, self.tgt_vocab_size),
        decoder_outputs.contiguous().view(-1)
      )
    self.scaler.scale(loss).backward()
    self.scaler.step(self.optimizer)
    self.scaler.update()
    return probs, loss.item()


class FP32PrecisionStrategy(AbstractPrecisionStrategy):
  """fp32精度训练策略"""
  def __init__(self, *args, **kwargs):
    super(FP32PrecisionStrategy, self).__init__(*args, **kwargs)

  def do_train_one_batch(self, encoder_inputs:torch.Tensor,
                         decoder_inputs:torch.Tensor, decoder_outputs:torch.Tensor) -> Tuple[torch.Tensor, float]:
    """fp32训练时反向传播"""
    self.optimizer.zero_grad()
    probs = self.net(encoder_inputs, decoder_inputs)
    loss = self.criterion(
      probs.contiguous().view(-1, self.tgt_vocab_size),
      decoder_outputs.contiguous().view(-1)
    )
    loss.backward()
    torch.nn.utils.clip_grad_value_(self.net.parameters(), clip_value=0.5)  # 梯度裁剪，防止梯度爆炸
    self.optimizer.step()
    return probs, loss.item()


def precision_strategy_factory(precision, *args, **kwargs) -> Union[BF16PrecisionStrategy, FP32PrecisionStrategy, None]:
  """训练精度策略工厂"""
  if precision == torch.bfloat16:
    return BF16PrecisionStrategy(*args, **kwargs)
  elif precision == torch.float32:
    return FP32PrecisionStrategy(*args, **kwargs)
  return None