#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import torch.optim
from typing import Optional, Dict, Any


class CheckpointMetaInfo(object):
  """检查点元信息"""
  def __init__(self, epoch:int, model_state:Dict[str, Any], scaler_state:Optional[Dict[str, Any]],
               optimizer_state: Dict[str, Any], hyperparameters:Dict[str, Any]):
    self.epoch = epoch
    self.model_state = model_state
    self.scaler_state = scaler_state
    self.optimizer_state = optimizer_state
    self.hyperparameters = hyperparameters

  @classmethod
  def from_dict(cls, state_dict:Dict[str, Any]) -> "CheckpointMetaInfo":
    """从字典重建对象"""
    return cls(
      epoch=state_dict['epoch'],
      model_state=state_dict['model_state'],
      scaler_state=state_dict['scaler_state'],
      optimizer_state=state_dict['optimizer_state'],
      hyperparameters=state_dict['hyperparameters'],
    )


class CheckpointManager(object):
  """模型训练checkpoint管理器"""
  def __init__(self, storage_path:str="output/checkpoints", save_interval:int=1):
    self.storage_path = storage_path
    self.save_interval = save_interval
    if not os.path.exists(self.storage_path):
      os.makedirs(storage_path, exist_ok=True)
    else:
      self.clean_history_checkpoints()

  def save(self, meta_info: CheckpointMetaInfo) -> str:
    """保存检查点"""
    if not os.path.exists(self.storage_path):
      os.makedirs(self.storage_path, exist_ok=True)
    filename = f"epoch_{meta_info.epoch}_checkpoint.pt"
    path = os.path.join(self.storage_path, filename)
    torch.save({
      'checkpoint_type': 'meta_info',
      'data': meta_info.__dict__,
    }, path)
    return path

  def get_latest_ckpt(self, device:str) -> Optional[CheckpointMetaInfo]:
    """获取最新检查点"""
    checkpoints = [f for f in os.listdir(self.storage_path) if "checkpoint" in f and f.endswith(".pt")]
    if not checkpoints or len(checkpoints) == 0:
      return None
    latest_path = os.path.join(self.storage_path, checkpoints[0])
    return CheckpointManager.load_from_specified_path(latest_path, device=device)

  def clean_history_checkpoints(self) -> None:
    """清理除最新checkpoint之外的所有checkpoint，节省磁盘空间"""
    checkpoints = [f for f in os.listdir(self.storage_path) if "checkpoint" in f and f.endswith(".pt")]
    if not checkpoints or len(checkpoints) < 2:
      return None
    checkpoints.sort(key=lambda x: int(x.replace("epoch_", "").replace("_checkpoint.pt", "")), reverse=True)
    old_checkpoints = checkpoints[1:]
    for old_checkpoint in old_checkpoints:
      os.remove(os.path.join(self.storage_path, old_checkpoint))

  @staticmethod
  def load_from_specified_path(file_path:str, device:str) -> Optional[CheckpointMetaInfo]:
    """从指定路径加载checkpoint"""
    loaded = torch.load(file_path, map_location=device)
    if loaded.get('checkpoint_type') != 'meta_info':
      raise ValueError("Invalid checkpoint format")
    return CheckpointMetaInfo.from_dict(loaded['data'])
