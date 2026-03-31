import os
from typing import Any, Dict, Optional

import torch
from torch import nn


def save_full_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    save_path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    summary: 保存完整 checkpoint 以支持断点续训
    param model: 当前模型
    param optimizer: 当前优化器
    param epoch: 当前训练轮数
    param best_val_loss: 当前最佳验证损失
    param save_path: checkpoint 保存路径
    return: 无
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "metadata": metadata or {},
    }

    torch.save(checkpoint, save_path)


def load_checkpoint_flexible(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    fallback_start_epoch: int = 1,
    fallback_best_val_loss: float = float("inf")
) -> Dict[str, Any]:
    """
    summary: 兼容读取旧版权重文件与新版完整 checkpoint
    param model: 待加载权重的模型
    param checkpoint_path: checkpoint 文件路径
    param device: 当前设备
    param optimizer: 可选优化器，用于恢复优化器状态
    param fallback_start_epoch: 旧版权重文件的默认起始 epoch
    param fallback_best_val_loss: 旧版权重文件的默认最佳验证损失
    return: 包含加载信息的字典
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])

        optimizer_loaded = False
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            optimizer_loaded = True

        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_loss = checkpoint.get("best_val_loss", fallback_best_val_loss)

        print(f"已加载新版完整 checkpoint: {checkpoint_path}")
        print(f"下次训练将从 epoch {start_epoch} 开始")
        print(f"optimizer 状态已恢复: {optimizer_loaded}")

        return {
            "start_epoch": start_epoch,
            "best_val_loss": best_val_loss,
            "is_full_checkpoint": True,
            "optimizer_loaded": optimizer_loaded,
            "metadata": checkpoint.get("metadata", {}),
        }

    if isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)

        print(f"已加载旧版权重文件: {checkpoint_path}")
        print("该文件仅包含模型权重，optimizer 状态无法恢复。")
        print(f"将从 epoch {fallback_start_epoch} 开始继续训练。")

        return {
            "start_epoch": fallback_start_epoch,
            "best_val_loss": fallback_best_val_loss,
            "is_full_checkpoint": False,
            "optimizer_loaded": False,
            "metadata": {},
        }

    raise ValueError(f"无法识别的 checkpoint 格式: {checkpoint_path}")


def save_checkpoint(model: nn.Module, save_path: str) -> None:
    """
    summary: 保存模型参数
    param model: 待保存模型
    param save_path: 保存路径
    return: 无
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)


def peek_checkpoint_metadata(checkpoint_path: str, device: torch.device | str = "cpu") -> Dict[str, Any]:
    """
    summary: 读取 checkpoint 中的元数据而不恢复模型
    param checkpoint_path: checkpoint 路径
    param device: torch.load 使用的 map_location
    return: 元数据字典
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint.get("metadata", {})

    return {}
