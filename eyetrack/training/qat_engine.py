import copy
import warnings
from typing import Any, Dict

import torch
from torch import nn


VALID_QAT_BACKENDS = {"qnnpack", "fbgemm"}


def list_supported_quantized_engines() -> list[str]:
    """
    summary: 获取当前 PyTorch 构建支持的 quantized engines
    param 无: 无
    return: 支持列表（小写）
    """
    return [str(engine).lower() for engine in torch.backends.quantized.supported_engines]


def choose_supported_qat_backend(requested_backend: str) -> str:
    """
    summary: 在当前环境中选择可用 backend，必要时自动回退
    param requested_backend: 请求的 backend
    return: 最终可用 backend
    """
    requested = normalize_qat_backend(requested_backend)
    supported = list_supported_quantized_engines()

    if requested in supported:
        return requested

    fallback_order = ["fbgemm", "x86", "onednn"]
    for candidate in fallback_order:
        if candidate in supported:
            warnings.warn(
                "quantized engine "
                f"{requested.upper()} is not supported in current PyTorch build; "
                f"fallback to {candidate}. supported={supported}",
                RuntimeWarning,
            )
            return candidate

    raise RuntimeError(
        "当前 PyTorch 构建没有可用的 quantized engine，"
        f"requested={requested} supported={supported}"
    )


def normalize_qat_backend(backend: str) -> str:
    """
    summary: 规范化 QAT backend 名称并校验合法性
    param backend: backend 名称
    return: 规范化后的 backend
    """
    normalized = backend.strip().lower()
    if normalized not in VALID_QAT_BACKENDS:
        raise ValueError(f"不支持的 qat_backend: {backend}，可选值: {sorted(VALID_QAT_BACKENDS)}")
    return normalized


def set_qat_backend(backend: str) -> str:
    """
    summary: 设置 PyTorch quantized engine
    param backend: backend 名称
    return: 规范化后的 backend
    """
    selected = choose_supported_qat_backend(backend)
    torch.backends.quantized.engine = selected
    return selected


def prepare_model_for_qat(model: nn.Module, backend: str = "qnnpack") -> nn.Module:
    """
    summary: 将模型准备为 QAT 训练形态
    param model: 目标模型
    param backend: qnnpack 或 fbgemm
    return: 已准备好的模型
    """
    normalized_backend = set_qat_backend(backend)

    model.train()
    if hasattr(model, "fuse_model"):
        model.fuse_model(is_qat=True)

    model.qconfig = torch.ao.quantization.get_default_qat_qconfig(normalized_backend)
    torch.ao.quantization.prepare_qat(model, inplace=True)
    return model


def update_qat_epoch_state(
    model: nn.Module,
    stage_epoch: int,
    total_stage_epochs: int,
    disable_observer_last_n_epochs: int = 1,
    freeze_bn_after_epoch: int = 2,
) -> Dict[str, Any]:
    """
    summary: 根据当前 QAT 阶段轮次切换 observer 与 BN 统计状态
    param model: QAT 模型
    param stage_epoch: QAT 阶段内当前轮次，从 1 开始
    param total_stage_epochs: QAT 阶段总轮次
    param disable_observer_last_n_epochs: 最后多少轮关闭 observer
    param freeze_bn_after_epoch: 从第几轮开始冻结 BN 统计
    return: 当前状态信息
    """
    observer_enabled = True
    if disable_observer_last_n_epochs > 0:
        disable_threshold = max(total_stage_epochs - disable_observer_last_n_epochs + 1, 1)
        observer_enabled = stage_epoch < disable_threshold

    if observer_enabled:
        model.apply(torch.ao.quantization.enable_observer)
    else:
        model.apply(torch.ao.quantization.disable_observer)

    freeze_bn_fn = getattr(torch.ao.quantization, "freeze_bn_stats", None)
    update_bn_fn = getattr(torch.ao.quantization, "update_bn_stats", None)

    bn_frozen = False
    if freeze_bn_after_epoch > 0 and stage_epoch >= freeze_bn_after_epoch and freeze_bn_fn is not None:
        model.apply(freeze_bn_fn)
        bn_frozen = True
    elif update_bn_fn is not None:
        model.apply(update_bn_fn)

    return {
        "observer_enabled": observer_enabled,
        "bn_frozen": bn_frozen,
        "stage_epoch": stage_epoch,
        "total_stage_epochs": total_stage_epochs,
    }


def convert_prepared_qat_model(model: nn.Module, backend: str = "qnnpack") -> nn.Module:
    """
    summary: 将已 prepare_qat 的模型转换为量化推理模型副本
    param model: 已 prepare_qat 的模型
    param backend: qnnpack 或 fbgemm
    return: 量化推理模型（CPU）
    """
    set_qat_backend(backend)
    converted = copy.deepcopy(model).cpu().eval()
    return torch.ao.quantization.convert(converted, inplace=True)


def _strip_qat_modules_to_float_inplace(module: nn.Module) -> None:
    """
    summary: 递归将 QAT 模块替换为对应的浮点模块
    param module: 待处理模块
    return: 无
    """
    for name, child in list(module.named_children()):
        _strip_qat_modules_to_float_inplace(child)
        if hasattr(child, "to_float") and "qat" in child.__class__.__module__:
            setattr(module, name, child.to_float())


def strip_prepared_qat_model_to_float(model: nn.Module) -> nn.Module:
    """
    summary: 将已 prepare_qat 的模型转换为便于导出的浮点模型副本
    param model: 已 prepare_qat 且已加载权重的模型
    return: 去除 fake quant 的浮点模型（CPU）
    """
    stripped = copy.deepcopy(model).cpu().eval()
    _strip_qat_modules_to_float_inplace(stripped)
    return stripped
