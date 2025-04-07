#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import unittest
import torch
from models.attention import ScaleDotProductAttention


class TestScaleDotProductAttention(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    torch.manual_seed(42)
    cls.test_params = {
      'batch_size': 2,
      'seq_len': 4,
      'd_model': 8
    }
  
  def _create_inputs(self):
    shape = (self.test_params['batch_size'],
             self.test_params['seq_len'],
             self.test_params['d_model'])
    return (torch.randn(shape),
            torch.randn(shape),
            torch.randn(shape))
  
  def _create_causal_mask(self):
    """创建因果掩码"""
    seq_len = self.test_params['seq_len']
    causal_mask = ~torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return causal_mask
  
  def _run_test_case(self, use_mask=False):
    q, k, v = self._create_inputs()
    mask = self._create_causal_mask() if use_mask else None
    
    # 自定义实现
    custom_attn = ScaleDotProductAttention()
    custom_output = custom_attn(q, k, v, mask)
    # 官方实现（带数值稳定性检查）
    with torch.no_grad():
      official_output = torch.nn.functional.scaled_dot_product_attention(
        q, k, v,
        is_causal=True if use_mask else False,
        scale=1.0 / torch.sqrt(torch.tensor(self.test_params['d_model'])),
        dropout_p=0.0  # 显式关闭dropout
      )
    
    # 验证结果
    self._validate_outputs(custom_output, official_output)
  
  def _validate_outputs(self, custom, official):
    # 检查NaN
    self.assertFalse(torch.isnan(custom).any(), "Custom output contains NaN")
    self.assertFalse(torch.isnan(official).any(), "Official output contains NaN")
    
    # 检查形状
    self.assertEqual(custom.shape, official.shape)
    
    # 检查数值有效性
    torch.testing.assert_close(custom, official, rtol=1e-4, atol=1e-6,
                               msg="Output values mismatch")
  
  def test_no_mask(self):
    """测试无mask情况"""
    self._run_test_case(use_mask=False)
  
  def test_with_mask(self):
    """测试带mask情况"""
    self._run_test_case(use_mask=True)


if __name__ == '__main__':
  unittest.main()
