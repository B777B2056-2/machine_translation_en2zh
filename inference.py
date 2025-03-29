#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import torch
from data.data import Tokenizer
from models.transformer import Transformer
from checkpoint import CheckpointManager


class Inference:
  """Transformer推理器"""
  def __init__(self, model_path:str, tokenizer_dir_path:str, max_seq_len:int=50):
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.max_seq_len = max_seq_len
    # 加载词表
    self.__load_tokenizer(tokenizer_dir_path)
    # 加载模型
    self.__load_model(model_path)

  def __load_model(self, model_path:str):
    """载入模型"""
    checkpoint = CheckpointManager.load_from_specified_path(model_path, device=self.device)
    self.hyper_param = checkpoint.hyperparameters
    vocab_size = self.hyper_param["vocab_size"]
    n_head = self.hyper_param["n_head"]
    word_dim = self.hyper_param["word_dim"]
    self.net = Transformer(n_head=n_head, word_dim=word_dim, vocab_size=vocab_size)
    self.net.load_state_dict(checkpoint.model_state)

    # 转移到可用设备
    self.net = self.net.to(self.device)
    self.net.eval()

  def __load_tokenizer(self, tokenizer_dir_path:str) -> None:
    self.en_tokenizer = Tokenizer.load_from_disk("en", tokenizer_dir_path)
    self.zh_tokenizer = Tokenizer.load_from_disk("zh", tokenizer_dir_path)
    assert self.en_tokenizer.sentences_num != 0, "分词后句子数量不能为0"

  def __build_decoder_inputs(self, decoder_outputs=None) -> torch.Tensor:
    """构造解码器输入"""
    if decoder_outputs is None: # 初始时只有 <sos>
      return torch.tensor([[self.en_tokenizer.start_flag_id()]], dtype=torch.long)
    # 将历史输出与 <sos> 组合，如 [<sos>, A, B]
    sequence = [self.en_tokenizer.start_flag_id()] + decoder_outputs
    return torch.tensor([sequence], dtype=torch.long)

  def __call__(self, prompt:str):
    # 对prompt进行分词
    tokens = self.en_tokenizer.tokenize(language="en", prompts=[prompt])
    # 根据tokens构造编码器输入
    encoder_input = torch.tensor(tokens, dtype=torch.long).to(self.device)  # [1, src_len]
    # 循环输出解码器预测结果
    decoder_outputs = []
    for _ in range(self.max_seq_len):
      # 构造解码器输入
      decoder_input = self.__build_decoder_inputs(decoder_outputs).to(self.device) # 解码器输入
      # 前向传播
      with torch.no_grad():
        probs = self.net(encoder_input, decoder_input).cpu()
      # 预测下一个词（取概率最大的词索引）
      next_token = probs[:, -1, :].argmax(dim=-1).item()
      if next_token == self.zh_tokenizer.end_flag_id():  # 终止条件：生成出终止符索引
        break
      decoder_outputs.append(next_token)
    # 将词索引转换为文本
    words = self.zh_tokenizer.detokenize(tokens=decoder_outputs)
    words.remove(Tokenizer.SENTENCE_START_PLACEHOLDER)
    words.remove(Tokenizer.SENTENCE_END_PLACEHOLDER)
    answer = "".join(words)
    return answer


if __name__ == "__main__":
  inference = Inference(
    model_path="output/checkpoints/epoch_10_checkpoint.pt",
    tokenizer_dir_path="output/vocabs",
  )

  while True:
    prompt = input("> ")
    if prompt == "exit":
      break
    print(inference(prompt="Paris"))
