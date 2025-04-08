#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import pickle
from typing import List, Tuple
from sklearn.feature_extraction.text import CountVectorizer
from enum import Enum
from abc import ABC, abstractmethod


class TokenizerMode(Enum):
  FROM_TEXT_DATA = 0          # 从文本初始化词表
  LOAD_FROM_DISK = 1  # 从磁盘恢复状态


class Tokenizer(ABC):
  """将文本转换为词向量：读取文本 -> tokenized分词 -> token编码：转成input ids向量"""
  SENTENCE_START_PLACEHOLDER = "<sos>" # 句子起始符
  SENTENCE_END_PLACEHOLDER = "<eos>"   # 句子结束符
  WORD_PADDING_IDX = 0                 # 词填充索引

  def __init__(self, language:str, file_path:str, mode:TokenizerMode):
    self.__language = language
    if mode == TokenizerMode.FROM_TEXT_DATA:
      self.__init_from_text(file_path)
    elif mode == TokenizerMode.LOAD_FROM_DISK:
      self.__load_from_disk(file_path)
    else:
      raise ValueError(f"Unknown Mode: {mode}")

  def __init_from_text(self, file_path: str):
    """从文本初始化词表"""
    assert file_path != ""
    with open(file_path, "r", encoding='UTF-8') as f:
      self.__build_vocab(f.readlines())

  def __load_from_disk(self, file_path: str):
    """从磁盘加载"""
    assert file_path != "" and ".pkl" in file_path
    data = pickle.load(open(file_path, 'rb'))
    self.__st_num = data["sentence_num"]
    self.__max_sequence_length = data["max_seq_len"]
    self.__word2idx = data["word2idx"]
    self.__idx2word = data["idx2word"]
    self.__tokenized_sentences = data["tokenized_sentences"]
    self.__start_symbol_idx = data["start_symbol_idx"]
    self.__end_symbol_idx = data["end_symbol_idx"]
    self.__language = data["language"]
    self.vectorizer = CountVectorizer(vocabulary=data["vectorizer"])

  def save(self, dir_path:str= "output/vocabs") -> None:
    """保存词表（只保存一次）"""
    if os.path.exists(dir_path):
      files = [f for f in os.listdir(dir_path) if "vectorizer" in f and self.__language in f and f.endswith(".pkl")]
      if len(files) != 0:
        return
    else:
      os.makedirs(dir_path, exist_ok=True)
    with open(os.path.join(dir_path, f"{self.__language}_tokenizer.pkl"), 'wb') as f:
      data = {
        "sentence_num": self.__st_num,
        "max_seq_len": self.__max_sequence_length,
        "word2idx": self.__word2idx,
        "idx2word": self.__idx2word,
        "tokenized_sentences": self.__tokenized_sentences,
        "start_symbol_idx": self.__start_symbol_idx,
        "end_symbol_idx": self.__end_symbol_idx,
        "language": self.__language,
        "vectorizer": self.vectorizer.vocabulary_,
      }
      pickle.dump(data, f)

  @abstractmethod
  def _words_cut(self, sentence: str) -> List[str]:
    """句子分词方式，中英文不同"""
    raise NotImplementedError

  def __tokenized(self, lines: List[str]) -> Tuple[List[str], int]:
    """文本分词，例如：单句为 1929 or 1989? -> 分词后 [[1929, or, 1989, ?]]"""
    assert len(lines) != 0
    corpus = []
    max_seq_len = 0
    for line in lines:
      sentence = [Tokenizer.SENTENCE_START_PLACEHOLDER]
      words = self._words_cut(line)
      for word in words:
        sentence.append(word)
      sentence.append(Tokenizer.SENTENCE_END_PLACEHOLDER)
      max_seq_len = max(max_seq_len, len(sentence))
      corpus.append(" ".join(sentence))
    return corpus, max_seq_len

  def __word_to_idx(self, corpus: List[str]) -> List[List[int]]:
    """将分词产出的token，转换为数值向量（即词索引）"""
    self.vectorizer = CountVectorizer(lowercase=False)
    _ = self.vectorizer.fit_transform(corpus)
    vocab = self.vectorizer.vocabulary_
    self.__word2idx = vocab.copy()
    self.__idx2word = {}
    for (word, idx) in self.__word2idx.items():
      self.__idx2word[idx] = word
    tokenized_sentences = []
    for doc in corpus:
      tokens = self.vectorizer.build_analyzer()(doc)
      indices = [self.__word2idx[token] for token in tokens if token in self.__word2idx]
      tokenized_sentences.append(indices)
    # 获取起始符、结束符索引
    self.__start_symbol_idx = self.__word2idx[Tokenizer.SENTENCE_START_PLACEHOLDER.lstrip("<").rstrip(">")]
    self.__end_symbol_idx = self.__word2idx[Tokenizer.SENTENCE_END_PLACEHOLDER.lstrip("<").rstrip(">")]
    return tokenized_sentences

  def __build_vocab(self, lines:List[str]):
    """构建词表（训练前进行）"""
    # 分词
    corpus, max_seq_len = self.__tokenized(lines=lines)
    self.__st_num = len(corpus)
    self.__max_sequence_length = max_seq_len
    # token编码：转成input ids向量
    self.__tokenized_sentences = self.__word_to_idx(corpus)

  def tokenize(self, prompts) -> List[List[int]]:
    # 已有词表，分词后直接查表，并添加起始、结束符索引位置
    corpus, _ = self.__tokenized(lines=prompts)  # 分词
    vector = []
    for doc in corpus:
      tokens = self.vectorizer.build_analyzer()(doc)
      indices = [self.__word2idx[token] for token in tokens]
      vector.append(indices)
    return vector

  def detokenize(self, tokens: List[int]) -> List[str]:
    """从传入token反向构建词"""
    words = []
    for word_idx in tokens:
      words.append(self.__idx2word[word_idx] if word_idx in self.__idx2word else Tokenizer.SENTENCE_END_PLACEHOLDER)
    return words

  def start_flag_id(self) -> int:
    return self.__start_symbol_idx

  def end_flag_id(self) -> int:
    return self.__end_symbol_idx

  def sentence_num(self) -> int:
    return self.__st_num

  def sentences(self) -> List[List[int]]:
    return self.__tokenized_sentences

  def vocab_size(self) -> int:
    return len(self.__word2idx)

  def max_seq_len(self) -> int:
    return self.__max_sequence_length

class ChineseTokenizer(Tokenizer):
  """中文分词器"""
  def __init__(self, file_path:str, mode:TokenizerMode=TokenizerMode.FROM_TEXT_DATA):
    super(ChineseTokenizer, self).__init__('zh', file_path, mode)

  def _words_cut(self, sentence: str) -> List[str]:
    import jieba
    return jieba.lcut(sentence)


class EnglishTokenizer(Tokenizer):
  """英文分词器"""
  def __init__(self, file_path:str, mode:TokenizerMode=TokenizerMode.FROM_TEXT_DATA):
    super(EnglishTokenizer, self).__init__('en', file_path, mode)

  def _words_cut(self, sentence: str) -> List[str]:
    import nltk
    return nltk.word_tokenize(sentence)