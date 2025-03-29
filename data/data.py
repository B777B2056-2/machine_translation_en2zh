#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from data.tokenizer import Tokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class EnZhDataSet(Dataset):
  """中英文机器翻译数据集"""
  def __init__(self, en_tokenizer:Tokenizer, zh_tokenizer:Tokenizer):
    # 获取词索引（不提前进行词嵌入，因为在当前Transformer实现中，已经将词嵌入融合在位置编码模块了）
    en_tokens = en_tokenizer.tokenize(language="en")
    zh_tokens = zh_tokenizer.tokenize(language="zh")
    # 检查句子对齐
    assert en_tokenizer.sentences_num == zh_tokenizer.sentences_num, "EN/ZH sentences mismatch"
    self.len = en_tokenizer.sentences_num
    assert self.len != 0, "Dataset length is 0"
    # 保存词表
    en_tokenizer.save(language="en")
    zh_tokenizer.save(language="zh")
    # 生成解码器、编码器输入与输出
    self.encoder_inputs = en_tokens     # 编码器输入：待翻译语言（英语）的词索引序列
    self.decoder_inputs, self.decoder_outputs = self.__decoder_vectors_shift_right(zh_tokens)  # 解码器输入与标签

  @staticmethod
  def __decoder_vectors_shift_right(tokens):
    """解码器输入与标签构造"""
    decoder_inputs = []
    decoder_outputs = []
    for sentence in tokens:
      decoder_inputs.append(sentence[:-1])  # 解码器输入，包含 <sos> A B
      decoder_outputs.append(sentence[1:])  # 解码器标签，包含 A B <eos>
    return decoder_inputs, decoder_outputs

  def __getitem__(self, i):
    return torch.tensor(self.encoder_inputs[i]), torch.tensor(self.decoder_inputs[i]), torch.tensor(self.decoder_outputs[i])

  def __len__(self):
    return self.len


def build_train_data_loader(batch_size:int=32):
  """构建训练集data loader"""
  import os
  import kagglehub
  dir_path = kagglehub.dataset_download("concyclics/machine-translation-chinese-and-english")
  # 句子分词
  en_tokenizer = Tokenizer(file_path=os.path.join(dir_path, "english.en"))
  zh_tokenizer = Tokenizer(file_path=os.path.join(dir_path, "chinese.zh"))
  # 构建数据集
  train_dataset = EnZhDataSet(en_tokenizer, zh_tokenizer)
  vocab_size = len(train_dataset)

  def seq_collate_fn(batch):
    """动态填充批次数据"""
    encoder, decoder_in, decoder_out = zip(*batch)
    encoder_padded = pad_sequence([torch.LongTensor(x) for x in encoder], batch_first=True, padding_value=0)
    decoder_in_padded = pad_sequence([torch.LongTensor(x) for x in decoder_in], batch_first=True, padding_value=0)
    decoder_out_padded = pad_sequence([torch.LongTensor(x) for x in decoder_out], batch_first=True, padding_value=0)
    return encoder_padded, decoder_in_padded, decoder_out_padded

  return DataLoader(train_dataset, batch_size=batch_size, collate_fn=seq_collate_fn, shuffle=True), vocab_size
