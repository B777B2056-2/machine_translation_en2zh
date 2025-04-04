#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import torch
from models.transformer import Transformer
from data.data import build_train_data_loader
from checkpoint import CheckpointMetaInfo, CheckpointManager


class Trainer(object):
  """机器翻译Transformer训练器"""
  def __init__(self, vocab_size:int, lr:float, n_head:int, word_dim:int, ckpt_save_interval:int=1):
    """
    Args:
    :param vocab_size:        词表大小
    :param lr:                学习率
    :param n_head:            多头注意力机制的头个数
    :param word_dim:          词嵌入维度
    :return:                  None
    """
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.net = Transformer(n_head=n_head, word_dim=word_dim, vocab_size=vocab_size).to(self.device)
    self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
    # 初始化ckpt管理器
    self.ckpt = CheckpointManager(save_interval=ckpt_save_interval)
    # 保存超参数字典
    self.hyper_param = {
      "vocab_size": vocab_size,
      "n_head": n_head,
      "word_dim": word_dim,
    }

  def __need_save_checkpoint(self) -> bool:
    """检查用户是否需要保存ckpt"""
    return self.ckpt.save_interval > 0

  def __save_checkpoint(self, epoch:int):
    """到达指定间隔，保存checkpoint"""
    if not self.__need_save_checkpoint() or ((epoch + 1) % self.ckpt.save_interval != 0):
      return
    self.ckpt.save(
      CheckpointMetaInfo(
        epoch=epoch,
        model_state=self.net.state_dict(),
        optimizer_state=self.optimizer.state_dict(),
        hyperparameters=self.hyper_param,
      )
    )
    print(f"save checkpoint for epoch {epoch}")

  def __resume_from_latest_checkpoint(self) -> int:
    """从最新检查点恢复训练"""
    checkpoint = self.ckpt.get_latest_ckpt(device=self.device)
    if not checkpoint:
      print("No checkpoint found, starting fresh training.")
      return 0  # 从第一轮开始

    # 恢复网络状态
    self.hyper_param = checkpoint.hyperparameters
    self.net = Transformer(
      n_head=self.hyper_param["n_head"],
      word_dim=self.hyper_param["word_dim"],
      vocab_size=self.hyper_param["vocab_size"]
    )
    self.net.load_state_dict(checkpoint.model_state)
    self.optimizer.load_state_dict(checkpoint.optimizer_state)

    # 转移到可用设备
    self.net = self.net.to(self.device)

    # 返回最后训练位置
    last_epoch = checkpoint.epoch
    print(f"Resuming training from Epoch {last_epoch}")
    return last_epoch + 1  # 从下一个epoch开始

  def train(self, train_data_loader, n_epoch:int) -> None:
    """
        Args:
        :param train_data_loader: 训练数据集dataloader
        :param n_epoch:           训练轮次
        :return:                  None
    """
    self.net.train()
    for epoch in range(n_epoch):
      train_loss = 0.0
      for batch_idx, (encoder_inputs, decoder_inputs, decoder_outputs) in enumerate(train_data_loader):
        encoder_inputs, decoder_inputs, decoder_outputs = encoder_inputs.to(self.device), decoder_inputs.to(
          self.device), decoder_outputs.to(self.device)

        self.optimizer.zero_grad()
        probs = self.net(encoder_inputs, decoder_inputs)
        loss = self.criterion(
          probs.contiguous().view(-1, self.hyper_param["vocab_size"]),  # [batch*(seq_len-1), vocab_size]
          decoder_outputs.contiguous().view(-1)  # [batch*(seq_len-1)]
        )
        loss.backward()
        self.optimizer.step()

        train_loss += loss.item()
      print(f"epoch: {epoch}, train loss: {train_loss}")
      # 到达指定间隔，保存checkpoint
      self.__save_checkpoint(epoch=epoch)


def train_main():
  import argparse
  parser = argparse.ArgumentParser(description='Transformer模型训练参数')

  train_group = parser.add_argument_group('训练参数')
  train_group.add_argument('--n_epoch', type=int, default=10,
                           help='训练轮次 (默认：10)')
  train_group.add_argument('--lr', type=float, default=0.1,
                           help='初始学习率 (默认：0.1)')
  train_group.add_argument('--ckpt', type=int, default=1,
                           help='ckpt保存间隔 (默认：1)')

  model_group = parser.add_argument_group('模型参数')
  model_group.add_argument('--n_head', type=int, default=8,
                           help='注意力头数 (默认：8)')
  model_group.add_argument('--word_dim', type=int, default=64,
                           help='词向量维度 (默认：64)')

  data_group = parser.add_argument_group('数据参数')
  data_group.add_argument('--batch_size', type=int, default=32,
                          help='批次大小 (默认：32)')
  args = parser.parse_args()

  # 构建训练集
  print("start load training data...")
  train_dataloader, vocab_size = build_train_data_loader(batch_size=args.batch_size)

  # 构建训练器
  trainer = Trainer(
    vocab_size=vocab_size,
    lr=args.lr,
    n_head=args.n_head,
    word_dim=args.word_dim,
    ckpt_save_interval=args.ckpt,
  )

  # 开始训练
  print("start training...")
  trainer.train(train_dataloader, args.n_epoch)
  print("finish training...")


if __name__ == "__main__":
  train_main()
