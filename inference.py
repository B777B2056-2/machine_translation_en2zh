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
    hyper_param = checkpoint.hyperparameters
    self.net = Transformer(
      n_head=hyper_param["n_head"],
      word_dim=hyper_param["word_dim"],
      vocab_size=hyper_param["vocab_size"]
    )
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
    if Tokenizer.SENTENCE_START_PLACEHOLDER in words:
      words.remove(Tokenizer.SENTENCE_START_PLACEHOLDER)
    if Tokenizer.SENTENCE_END_PLACEHOLDER in words:
      words.remove(Tokenizer.SENTENCE_END_PLACEHOLDER)
    answer = "".join(words)
    return answer

def inference_main():
  import argparse
  # 创建参数解析器
  parser = argparse.ArgumentParser(description='模型推理参数设置')

  # 模型参数组
  model_group = parser.add_argument_group('模型参数')
  model_group.add_argument('--model_path', type=str, required=True,
                           help='模型检查点路径 (必需)')
  model_group.add_argument('--tokenizer_dir', type=str, required=True,
                           help='分词器目录路径 (必需)')

  # 推理参数组
  infer_group = parser.add_argument_group('推理参数')
  infer_group.add_argument('--max_length', type=int, default=50,
                           help='生成文本最大长度 (默认：50)')

  args = parser.parse_args()

  # 初始化推理引擎
  inference = Inference(
    model_path=args.model_path,
    tokenizer_dir_path=args.tokenizer_dir,
  )

  # 交互式推理循环
  print("输入提示词开始生成（输入exit退出）:")
  while True:
    try:
      prompt = input("> ")
      if prompt.lower() == "exit":
        break
      print(inference(prompt=prompt))
    except KeyboardInterrupt:
      print("\n退出推理")
      break


if __name__ == "__main__":
  inference_main()
