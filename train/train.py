#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from data.tokenizer import Tokenizer
from models.transformer import Transformer
from data.data import DataLoaderBuilder
from checkpoint import CheckpointMetaInfo, CheckpointManager
from precision_strategy import *


def setup_train_seed(seed):
  """设置随机种子"""
  import random, numpy as np
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)
  torch.backends.cudnn.deterministic = True


def calculate_model_metrics(probs: torch.Tensor, targets: torch.Tensor,
                            pad_token_id: int = 0) -> Tuple[float, float]:
  """
  计算模型输出的指标（词级准确率、句级准确率）

  Args:
      probs: 模型输出的概率分布 [batch, seq_len, vocab_size]
      targets: 目标token IDs [batch, seq_len]
      pad_token_id: 用于计算mask的padding token ID

  Returns:
      token_acc: 词级准确率（忽略padding）
      seq_acc: 句级准确率
  """
  # 获取预测的token (argmax)
  pred_tokens = probs.argmax(dim=-1)  # [batch, seq_len]

  # Token-level Accuracy（忽略padding）
  mask = (targets != pad_token_id)
  token_correct = ((pred_tokens == targets) & mask).sum().item()
  token_total = mask.sum().item()
  token_acc = token_correct / token_total if token_total > 0 else 0.0

  # Sequence-level Accuracy（整个句子完全正确）
  seq_correct = (pred_tokens == targets).all(dim=-1).sum().item()
  seq_total = targets.size(0)
  seq_acc = seq_correct / seq_total if seq_total > 0 else 0.0

  return token_acc, seq_acc


class Trainer(object):
  """机器翻译Transformer训练器"""
  def __init__(self, data_loader_builder:DataLoaderBuilder, random_seed:int, precision:str,
               lr:float, n_head:int, word_dim:int, enable_data_parallel:bool, ckpt_save_interval:int=1):
    """
    Args:
    :param data_loader_builder: 数据加载器构建器
    :param random_seed:         随机种子
    :param precision:           精度，可选"fp32"或"bf16"
    :param lr:                  学习率
    :param n_head:              多头注意力机制的头个数
    :param word_dim:            词嵌入维度
    :return:                    None
    """
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    # 设置随机种子
    setup_train_seed(random_seed)
    # 数据初始化
    self.train_loader = data_loader_builder.train_data_loader()
    self.val_loader = data_loader_builder.val_data_loader()
    src_vocab_size = data_loader_builder.en_vocab_size()
    src_max_seq_len = data_loader_builder.en_max_seq_len()
    tgt_vocab_size = data_loader_builder.zh_vocab_size()
    tgt_max_seq_len = data_loader_builder.zh_max_seq_len()
    # 保存超参数字典
    self.hyper_param = {
      "enable_data_parallel": enable_data_parallel,
      "precision": precision,
      "seed": random_seed,
      "src_vocab_size": src_vocab_size,
      "src_max_seq_len": src_max_seq_len,
      "tgt_vocab_size": tgt_vocab_size,
      "tgt_max_seq_len": tgt_max_seq_len,
      "n_head": n_head,
      "word_dim": word_dim,
    }
    # 网络初始化
    self.net = self.__create_model(hyper_param=self.hyper_param)
    self.criterion = torch.nn.CrossEntropyLoss(ignore_index=Tokenizer.WORD_PADDING_IDX)
    self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)
    self.precision_strategy = precision_strategy_factory(precision=torch.float32 if precision == "fp32" else torch.bfloat16,
                                                         net=self.net,
                                                         criterion=self.criterion,
                                                         optimizer=self.optimizer,
                                                         tgt_vocab_size=tgt_vocab_size)
    # 初始化ckpt管理器
    self.ckpt = CheckpointManager(save_interval=ckpt_save_interval)

  def __create_model(self, hyper_param:Dict[str, Any], model_state=None) -> Transformer:
    """创建模型"""
    model = Transformer(
      src_vocab_size=hyper_param["src_vocab_size"],
      src_max_seq_len=hyper_param["src_max_seq_len"],
      tgt_vocab_size=hyper_param["tgt_vocab_size"],
      tgt_max_seq_len=hyper_param["tgt_max_seq_len"],
      n_head=hyper_param["n_head"],
      word_dim=hyper_param["word_dim"]
    )
    if hyper_param["enable_data_parallel"]:
      model = torch.nn.DataParallel(model)
    if model_state is not None:
      model.load_state_dict(model_state)
    return model.to(self.device)

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
        scaler_state=self.precision_strategy.scaler.state_dict(),
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
    setup_train_seed(self.hyper_param["seed"])
    self.net = self.__create_model(hyper_param=self.hyper_param, model_state=checkpoint.model_state)
    if isinstance(self.precision_strategy, BF16PrecisionStrategy):
      self.precision_strategy.load_scaler_state(scaler_state=checkpoint.scaler_state)
    self.optimizer.load_state_dict(checkpoint.optimizer_state)

    # 返回最后训练位置
    last_epoch = checkpoint.epoch
    return last_epoch + 1  # 从下一个epoch开始

  def _run_epoch(self, data_loader, is_training: bool) -> Tuple[float, float, float]:
    """统一训练/验证循环"""
    total_loss = 0.0
    total_token_acc = 0.0
    total_seq_acc = 0.0

    for batch_idx, (encoder_inputs, decoder_inputs, decoder_outputs) in enumerate(data_loader):
      encoder_inputs, decoder_inputs, decoder_outputs = encoder_inputs.to(self.device), decoder_inputs.to(
        self.device), decoder_outputs.to(self.device)

      if is_training:
        # 反向传播和优化（仅在训练模式）
        probs, loss_value = self.precision_strategy.do_train_one_batch(encoder_inputs=encoder_inputs,
                                                                       decoder_inputs=decoder_inputs,
                                                                       decoder_outputs=decoder_outputs)
      else:
        # 禁用梯度计算（仅在验证模式）
        with torch.no_grad():
          probs = self.net(encoder_inputs, decoder_inputs)
          loss_value = self.criterion(
            probs.contiguous().view(-1, self.hyper_param["tgt_vocab_size"]),
            decoder_outputs.contiguous().view(-1)
          ).item()

      # 计算指标
      token_acc, seq_acc = calculate_model_metrics(probs, decoder_outputs)
      total_loss += loss_value
      total_token_acc += token_acc
      total_seq_acc += seq_acc

    # 计算平均指标
    n_batches = len(data_loader)
    return (
      total_loss / n_batches,
      total_token_acc / n_batches,
      total_seq_acc / n_batches
    )

  def __call__(self, n_epoch:int) -> None:
    """
        Args:
        :param n_epoch:           训练轮次
        :return:                  None
    """
    self.net.train()
    epoch_start = self.__resume_from_latest_checkpoint()
    print(f"Starting Training From Epoch {epoch_start}")
    for epoch in range(epoch_start, n_epoch):
      train_loss, train_token_acc, train_seq_acc = self._run_epoch(self.train_loader, is_training=True) # 训练
      val_loss, val_token_acc, val_seq_acc = self._run_epoch(self.val_loader, is_training=False)        # 验证
      print(f"Epoch {epoch} | \
            Train Loss: {train_loss:.4f} | Token Acc: {train_token_acc:.4f} | Seq Acc: {train_seq_acc:.4f} | \
            Val Loss: {val_loss:.4f} | Token Acc: {val_token_acc:.4f} | Seq Acc: {val_seq_acc:.4f}")
      # 到达指定间隔，保存checkpoint
      self.__save_checkpoint(epoch=epoch)

def train_main():
  import argparse
  parser = argparse.ArgumentParser(description='Transformer模型训练参数')

  train_group = parser.add_argument_group('训练参数')
  train_group.add_argument('--random_seed', type=int, default=1012,
                           help='随机数种子（默认：1012）')
  train_group.add_argument('--precision', type=str, choices=["fp32", "bf16"], default="bf16",
                           help='训练精度（默认：bf16）')
  train_group.add_argument('--n_epoch', type=int, default=10,
                           help='训练轮次 (默认：10)')
  train_group.add_argument('--lr', type=float, default=0.0001,
                           help='初始学习率 (默认：0.0001)')
  train_group.add_argument('--enable_data_parallel', type=bool, default=True,
                           help='开启数据并行 (默认：开启)')
  train_group.add_argument('--ckpt', type=int, default=1,
                           help='ckpt保存间隔 (默认：1)')

  model_group = parser.add_argument_group('模型参数')
  model_group.add_argument('--n_head', type=int, default=8,
                           help='注意力头数 (默认：8)')
  model_group.add_argument('--word_dim', type=int, default=256,
                           help='词向量维度 (默认：256)')

  data_group = parser.add_argument_group('数据参数')
  data_group.add_argument('--batch_size', type=int, default=256,
                          help='批次大小 (默认：256)')
  data_group.add_argument('--val_size', type=float, default=0.2,
                          help='验证集大小 (默认：0.2)')
  data_group.add_argument('--num_workers', type=int, default=4,
                          help='数据加载进程数量 (默认：4)')
  args = parser.parse_args()

  # 构建训练集 & 验证集加载器
  print("start load training data...")
  data_loader_builder = DataLoaderBuilder(batch_size=args.batch_size,
                                          num_workers=args.num_workers,
                                          val_size=args.val_size)
  print("finish load training data...")

  # 构建训练器
  trainer = Trainer(
    data_loader_builder=data_loader_builder,
    random_seed=args.random_seed,
    precision=args.precision,
    lr=args.lr,
    n_head=args.n_head,
    word_dim=args.word_dim,
    enable_data_parallel=args.enable_data_parallel,
    ckpt_save_interval=args.ckpt,
  )

  # 开始训练
  print("start training...")
  trainer(args.n_epoch)
  print("finish training...")


if __name__ == "__main__":
  train_main()
