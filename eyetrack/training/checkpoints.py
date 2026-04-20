import os
from typing import Any, Dict, Optional

import torch
from torch import nn

from eyetrack.config import (
    DEFAULT_BASE_CHANNELS,
    DEFAULT_IN_CHANNELS,
    DEFAULT_INPUT_HEIGHT,
    DEFAULT_INPUT_WIDTH,
    DEFAULT_NUM_CLASSES,
    DEFAULT_PREPROCESS_MODE,
    DEFAULT_QAT_BACKEND,
    DEFAULT_QUANTIZATION_MODE,
    DEFAULT_USE_AMP,
    DEFAULT_USE_MASK,
    build_model_metadata,
)


def _load_state_dict_flexible(
    model: nn.Module,
    state_dict: Dict[str, Any],
    allow_partial_state_dict: bool,
) -> Dict[str, Any]:
    """
    summary: 加载 state_dict，必要时允许 strict=False 兼容加载
    param model: 目标模型
    param state_dict: 待加载参数
    param allow_partial_state_dict: 是否允许 partial load
    return: 加载结果信息
    """
    try:
        model.load_state_dict(state_dict)
        return {
            "partial_load": False,
            "missing_keys": [],
            "unexpected_keys": [],
        }
    except RuntimeError:
        if not allow_partial_state_dict:
            raise

    incompatible = model.load_state_dict(state_dict, strict=False)
    return {
        "partial_load": True,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


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
    fallback_best_val_loss: float = float("inf"),
    allow_partial_state_dict: bool = True,
) -> Dict[str, Any]:
    """
    summary: 兼容读取旧版权重文件与新版完整 checkpoint
    param model: 待加载权重的模型
    param checkpoint_path: checkpoint 文件路径
    param device: 当前设备
    param optimizer: 可选优化器，用于恢复优化器状态
    param fallback_start_epoch: 旧版权重文件的默认起始 epoch
    param fallback_best_val_loss: 旧版权重文件的默认最佳验证损失
    param allow_partial_state_dict: 是否允许 strict=False 兼容加载
    return: 包含加载信息的字典
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        load_state_info = _load_state_dict_flexible(
            model=model,
            state_dict=checkpoint["model_state_dict"],
            allow_partial_state_dict=allow_partial_state_dict,
        )

        optimizer_loaded = False
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            optimizer_loaded = True

        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_loss = checkpoint.get("best_val_loss", fallback_best_val_loss)

        print(f"已加载新版完整 checkpoint: {checkpoint_path}")
        print(f"下次训练将从 epoch {start_epoch} 开始")
        print(f"optimizer 状态已恢复: {optimizer_loaded}")
        if load_state_info["partial_load"]:
            print(
                "模型参数按 strict=False 兼容加载，"
                f"missing_keys={len(load_state_info['missing_keys'])}, "
                f"unexpected_keys={len(load_state_info['unexpected_keys'])}"
            )

        return {
            "start_epoch": start_epoch,
            "best_val_loss": best_val_loss,
            "is_full_checkpoint": True,
            "optimizer_loaded": optimizer_loaded,
            "metadata": checkpoint.get("metadata", {}),
            "partial_load": load_state_info["partial_load"],
            "missing_keys": load_state_info["missing_keys"],
            "unexpected_keys": load_state_info["unexpected_keys"],
        }

    if isinstance(checkpoint, dict):
        load_state_info = _load_state_dict_flexible(
            model=model,
            state_dict=checkpoint,
            allow_partial_state_dict=allow_partial_state_dict,
        )

        print(f"已加载旧版权重文件: {checkpoint_path}")
        print("该文件仅包含模型权重，optimizer 状态无法恢复。")
        print(f"将从 epoch {fallback_start_epoch} 开始继续训练。")
        if load_state_info["partial_load"]:
            print(
                "模型参数按 strict=False 兼容加载，"
                f"missing_keys={len(load_state_info['missing_keys'])}, "
                f"unexpected_keys={len(load_state_info['unexpected_keys'])}"
            )

        return {
            "start_epoch": fallback_start_epoch,
            "best_val_loss": fallback_best_val_loss,
            "is_full_checkpoint": False,
            "optimizer_loaded": False,
            "metadata": {},
            "partial_load": load_state_info["partial_load"],
            "missing_keys": load_state_info["missing_keys"],
            "unexpected_keys": load_state_info["unexpected_keys"],
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


def resolve_model_metadata(
    checkpoint_path: str | None = None,
    device: torch.device | str = "cpu",
    in_channels: int = DEFAULT_IN_CHANNELS,
    num_classes: int = DEFAULT_NUM_CLASSES,
    base_channels: int = DEFAULT_BASE_CHANNELS,
    input_width: int = DEFAULT_INPUT_WIDTH,
    input_height: int = DEFAULT_INPUT_HEIGHT,
    preprocess_mode: str = DEFAULT_PREPROCESS_MODE,
    amp: bool = DEFAULT_USE_AMP,
    use_mask: bool = DEFAULT_USE_MASK,
    quantization_mode: str = DEFAULT_QUANTIZATION_MODE,
    qat_backend: str = DEFAULT_QAT_BACKEND,
) -> Dict[str, Any]:
    """
    summary: 基于 fallback 配置和 checkpoint 元数据解析最终模型配置
    param checkpoint_path: 可选 checkpoint 路径
    param device: 读取 checkpoint 时的 map_location
    param in_channels: fallback 输入通道数
    param num_classes: fallback 类别数
    param base_channels: fallback 基础通道数
    param input_width: fallback 输入宽度
    param input_height: fallback 输入高度
    param preprocess_mode: fallback 预处理模式
    param amp: fallback AMP 开关
    param use_mask: fallback mask 开关
    param quantization_mode: fallback 量化模式
    param qat_backend: fallback QAT backend
    return: 解析后的模型配置元数据
    """
    resolved = build_model_metadata(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        input_width=input_width,
        input_height=input_height,
        preprocess_mode=preprocess_mode,
        amp=amp,
        use_mask=use_mask,
        quantization_mode=quantization_mode,
        qat_backend=qat_backend,
    )

    if checkpoint_path is None:
        return resolved

    checkpoint_metadata = peek_checkpoint_metadata(checkpoint_path=checkpoint_path, device=device)
    if len(checkpoint_metadata) == 0:
        return resolved

    resolved["in_channels"] = int(checkpoint_metadata.get("in_channels", resolved["in_channels"]))
    resolved["num_classes"] = int(checkpoint_metadata.get("num_classes", resolved["num_classes"]))
    resolved["base_channels"] = int(checkpoint_metadata.get("base_channels", resolved["base_channels"]))
    resolved["input_width"] = int(checkpoint_metadata.get("input_width", resolved["input_width"]))
    resolved["input_height"] = int(checkpoint_metadata.get("input_height", resolved["input_height"]))
    resolved["preprocess_mode"] = str(checkpoint_metadata.get("preprocess_mode", resolved["preprocess_mode"]))
    resolved["amp"] = bool(checkpoint_metadata.get("amp", resolved["amp"]))
    resolved["use_mask"] = bool(checkpoint_metadata.get("use_mask", resolved["use_mask"]))
    resolved["quantization_mode"] = str(checkpoint_metadata.get("quantization_mode", resolved["quantization_mode"]))
    resolved["qat_backend"] = str(checkpoint_metadata.get("qat_backend", resolved["qat_backend"]))
    return resolved
