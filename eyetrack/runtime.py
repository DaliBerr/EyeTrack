from contextlib import nullcontext

import torch


def resolve_device(device: str) -> torch.device:
    """
    summary: 根据字符串解析运行设备
    param device: auto/cpu/cuda 或具体设备名
    return: torch 设备对象
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前环境不可用 CUDA，但传入了 --device cuda")

    return torch.device(device)


def should_enable_amp(device: torch.device, use_amp: bool) -> bool:
    """
    summary: 判断当前设备上是否应该启用 AMP
    param device: 当前设备
    param use_amp: 用户配置的 AMP 开关
    return: 是否启用 AMP
    """
    return bool(use_amp and device.type == "cuda")


def autocast_context(device: torch.device, use_amp: bool):
    """
    summary: 返回适合当前设备的 autocast 上下文
    param device: 当前设备
    param use_amp: AMP 是否启用
    return: autocast 或空上下文
    """
    if should_enable_amp(device=device, use_amp=use_amp):
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    return nullcontext()
