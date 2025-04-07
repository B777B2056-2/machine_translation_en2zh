#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import unittest
import torch
from models.decoder import *


class TestKVCache(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    torch.manual_seed(42)
    cls.test_params = {
      'batch_size': 2,
      'n_head': 8,
      'seq_len': 4,
      'd_model': 256,
    }

  def _create_inputs(self):
    shape = (self.test_params['batch_size'],
             self.test_params['seq_len'],
             self.test_params['d_model'])
    return (torch.randn(shape),
            torch.randn(shape))

  def _create_causal_mask(self, seq_len):
    causal_mask = ~torch.triu(torch.ones(self.test_params["batch_size"] , seq_len, seq_len), diagonal=1).bool()
    return causal_mask

  def _validate_outputs(self, x1, x2):
    # 检查NaN
    self.assertFalse(torch.isnan(x1).any(), "Custom output contains NaN")
    self.assertFalse(torch.isnan(x2).any(), "Official output contains NaN")

    # 检查形状
    self.assertEqual(x1.shape, x2.shape)

    # 检查数值有效性
    torch.testing.assert_close(x1, x2, rtol=1e-4, atol=1e-6,
                               msg="Output values mismatch")

  def test(self):
    """测试带mask情况"""
    x, encoder_outputs = self._create_inputs()
    mask = self._create_causal_mask(self.test_params['seq_len'])

    attn_without_cache = TransformerDecoder(n_head=self.test_params['n_head'], word_dim=self.test_params['d_model'])
    attn_with_cache = TransformerDecoder(n_head=self.test_params['n_head'], word_dim=self.test_params['d_model'])
    attn_with_cache.load_state_dict(attn_without_cache.state_dict())

    with torch.no_grad():
      output_without_cache = attn_without_cache(x, encoder_outputs, mask=mask, use_cache=False)
      output_with_cache = attn_with_cache(x, encoder_outputs, mask=mask, use_cache=True)
    self._validate_outputs(output_without_cache, output_with_cache)


if __name__ == '__main__':
  unittest.main()
