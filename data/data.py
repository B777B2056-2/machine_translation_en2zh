#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from data.tokenizer import EnglishTokenizer, ChineseTokenizer, TokenizerMode, Tokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class EnZhDataSet(Dataset):
  """中英文机器翻译数据集"""
  def __init__(self, en_tokenizer:EnglishTokenizer, zh_tokenizer:ChineseTokenizer):
    # 获取词索引（不提前进行词嵌入，因为在当前Transformer实现中，已经将词嵌入融合在位置编码模块了）
    en_tokens = en_tokenizer.sentences()
    zh_tokens = zh_tokenizer.sentences()
    # 检查句子对齐
    assert len(en_tokens) == len(zh_tokens), "EN/ZH sentences mismatch"
    self.len = len(en_tokens)
    assert self.len != 0, "Dataset length is 0"
    # 保存词表
    en_tokenizer.save()
    zh_tokenizer.save()
    # 生成解码器、编码器输入与输出
    self.encoder_inputs = [torch.LongTensor(t) for t in en_tokens]     # 编码器输入：待翻译语言（英语）的词索引序列
    self.decoder_inputs, self.decoder_outputs = self.__decoder_vectors_shift_right(zh_tokens)  # 解码器输入与标签

  def __decoder_vectors_shift_right(self, tokens):
    """解码器输入与标签构造"""
    decoder_inputs = []
    decoder_outputs = []
    for sentence in tokens:
      decoder_inputs.append(torch.LongTensor(sentence[:-1]))  # 解码器输入，包含 <sos> A B
      decoder_outputs.append(torch.LongTensor(sentence[1:]))  # 解码器标签，包含 A B <eos>
    return decoder_inputs, decoder_outputs

  def __getitem__(self, i):
    return self.encoder_inputs[i], self.decoder_inputs[i], self.decoder_outputs[i]

  def __len__(self):
    return self.len


class DataLoaderBuilder(object):
  """数据加载器构建工厂"""
  def __init__(self, batch_size:int, num_workers:int, val_size:float, pin_memory:bool):
    """初始化：加载数据集"""
    import os
    import kagglehub
    dir_path = kagglehub.dataset_download("concyclics/machine-translation-chinese-and-english")
    # 句子分词
    if os.path.exists(os.path.join(dir_path, "en_tokenized.pkl")):
      self.en_tokenizer = EnglishTokenizer(mode=TokenizerMode.LOAD_FROM_DISK,
                                      file_path=os.path.join(dir_path, "en_tokenized.pkl"))
    else:
      self.en_tokenizer = EnglishTokenizer(mode=TokenizerMode.FROM_TEXT_DATA, file_path=os.path.join(dir_path, "english.en"))
    if os.path.exists(os.path.join(dir_path, "zh_tokenized.pkl")):
      self.zh_tokenizer = ChineseTokenizer(mode=TokenizerMode.LOAD_FROM_DISK,
                                      file_path=os.path.join(dir_path, "zh_tokenized.pkl"))
    else:
      self.zh_tokenizer = ChineseTokenizer(mode=TokenizerMode.FROM_TEXT_DATA, file_path=os.path.join(dir_path, "chinese.zh"))
    # 构建数据集
    dataset = EnZhDataSet(self.en_tokenizer, self.zh_tokenizer)
    # 划分训练集与验证集
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.pin_memory = pin_memory
    self.train_set, self.val_set = torch.utils.data.random_split(
      dataset,
      [1.0 - val_size, val_size],
      generator=torch.Generator().manual_seed(42)
    )

  def en_vocab_size(self):
    """返回英文词表大小"""
    return self.en_tokenizer.vocab_size()

  def zh_vocab_size(self):
    """返回中文词表大小"""
    return self.zh_tokenizer.vocab_size()

  def en_max_seq_len(self):
    """返回英文最大句长"""
    return self.en_tokenizer.max_seq_len()

  def zh_max_seq_len(self):
    """返回中文最大句长"""
    return self.zh_tokenizer.max_seq_len()

  @staticmethod
  def seq_collate_fn(batch):
    """动态填充批次数据"""
    encoder, decoder_in, decoder_out = zip(*batch)
    encoder_padded = pad_sequence(encoder, batch_first=True, padding_value=Tokenizer.WORD_PADDING_IDX)
    decoder_in_padded = pad_sequence(decoder_in, batch_first=True, padding_value=Tokenizer.WORD_PADDING_IDX)
    decoder_out_padded = pad_sequence(decoder_out, batch_first=True, padding_value=Tokenizer.WORD_PADDING_IDX)
    return encoder_padded, decoder_in_padded, decoder_out_padded

  def train_data_loader(self):
    """构建训练集data loader"""
    return DataLoader(self.train_set, batch_size=self.batch_size, collate_fn=DataLoaderBuilder.seq_collate_fn,
                      shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory)

  def val_data_loader(self):
    """构建验证集data loader"""
    val_data_loader = DataLoader(self.val_set, batch_size=self.batch_size, collate_fn=DataLoaderBuilder.seq_collate_fn,
                                 shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)
    return val_data_loader
