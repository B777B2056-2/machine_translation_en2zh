#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import torch
from models.embedding import LearnedPositionalEmbeddingWithWordEmbedding
from models.encoder import TransformerEncoder
from models.decoder import TransformerDecoder


class Transformer(torch.nn.Module):
  """Transformer"""
  def __init__(self, src_vocab_size:int, src_max_seq_len:int, tgt_vocab_size:int, tgt_max_seq_len:int,
               n_head:int, word_dim:int, n_block:int=1):
    """
    Args:
      src_vocab_size: 源语言词表大小
      src_max_seq_len：源语言序列最长长度
      tgt_vocab_size：目标语言词表大小
      tgt_max_seq_len：目标语言序列最长长度
      n_head: 多头注意力的头数量
      word_dim: 词嵌入维度
      n_block: 编码器/解码器个数
    Returns:
      预测概率 [batch_size, output_len, vocab_size]
    """
    super(Transformer, self).__init__()
    self.input_pos_embedding = LearnedPositionalEmbeddingWithWordEmbedding(src_vocab_size, word_dim, src_max_seq_len)
    self.output_pos_embedding = LearnedPositionalEmbeddingWithWordEmbedding(tgt_vocab_size, word_dim, tgt_max_seq_len)
    self.encoders = torch.nn.ModuleList([TransformerEncoder(n_head=n_head, word_dim=word_dim) for _ in range(n_block)])
    self.decoders = torch.nn.ModuleList([TransformerDecoder(n_head=n_head, word_dim=word_dim) for _ in range(n_block)])
    self.linear = torch.nn.Linear(word_dim, tgt_vocab_size)

  def __generate_causal_mask(self, seq_len:int) -> torch.Tensor:
    """
    因果掩码：防止解码器在训练时看到未来信息
    示例mask = [
      [1, 0, 0, 0],  # 第1个词只能看到自己
      [1, 1, 0, 0],  # 第2个词可以看到前2个词
      [1, 1, 1, 0],  # 第3个词可以看到前3个词
      [1, 1, 1, 1],  # 第4个词可以看到全部（如果未填充）
    ]
    """
    causal_mask = ~torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return causal_mask

  def __generate_padding_mask(self, input_ids) -> torch.Tensor:
    """
    padding掩码：忽略输入序列中的填充符号（如 <pad>），防止模型关注无效位置
    若输入序列中存在填充符号（如用 0 表示），则填充位置标记为0，非填充位置为1。
    """
    padding_mask = (input_ids != 0).unsqueeze(1)  # 扩展维度 [batch, 1, seq_len]
    return padding_mask

  def __concat_masks(self, causal_mask, padding_mask) -> torch.Tensor:
    """合并因果掩码与padding掩码"""
    combined_mask = causal_mask.to(padding_mask.device) & padding_mask
    return combined_mask

  def forward(self, input_embeddings:torch.Tensor, output_embeddings:torch.Tensor) -> torch.Tensor:
    """
    Args:
        input_embeddings: 编码器输入 [batch_size, input_len]
        output_embeddings: 解码器输入 [batch_size, output_len]
    Returns:
        预测概率 [batch_size, output_len, vocab_size]
    """
    # 1. 生成解码器输入向量的组合掩码（因果掩码+padding掩码），此后所有解码器共享该掩码
    _, seq_len = output_embeddings.shape
    causal_mask = self.__generate_causal_mask(seq_len)
    padding_mask = self.__generate_padding_mask(input_ids=output_embeddings)
    mask = self.__concat_masks(causal_mask=causal_mask, padding_mask=padding_mask)
    # 2. 位置编码
    input_pos = self.input_pos_embedding(input_embeddings)  # input词向量位置编码
    output_pos = self.output_pos_embedding(output_embeddings) # output词向量位置编码
    # 3. 编解码
    encoder_input = input_pos
    decoder_input = output_pos
    decoder_outputs = None
    for i in range(len(self.encoders)):
      encoder = self.encoders[i]
      decoder = self.decoders[i]
      # 编码
      encoder_outputs = encoder(encoder_input)
      # 解码
      decoder_outputs = decoder(decoder_input, encoder_outputs, mask=mask)
      # 重置编码器/解码器输入
      encoder_input = encoder_outputs
      decoder_input = decoder_outputs
    # 4. 线性层转换
    outputs = self.linear(decoder_outputs)
    # 5. 输出词向量预测值
    # outputs = F.softmax(outputs,dim=-1) # 交叉熵损失函数包含了softmax
    return outputs

