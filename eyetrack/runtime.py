from contextlib import nullcontext

import torch


def resolve_device(device: str) -> torch.device:
    """
    summary: parse
    param device: auto/cpu/cuda
    return: torch
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the current environment, but --device cuda was passed")

    return torch.device(device)


def should_enable_amp(device: torch.device, use_amp: bool) -> bool:
    """
    summary: current enable AMP
    param device: current
    param use_amp: AMP
    return: enable AMP
    """
    return bool(use_amp and device.type == "cuda")


def autocast_context(device: torch.device, use_amp: bool):
    """
    summary: return current autocast
    param device: current
    param use_amp: AMP enable
    return: autocast
    """
    if should_enable_amp(device=device, use_amp=use_amp):
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    return nullcontext()
