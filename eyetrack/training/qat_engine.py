import copy
import warnings
from typing import Any, Dict

import torch
from torch import nn


VALID_QAT_BACKENDS = {"qnnpack", "fbgemm"}


def list_supported_quantized_engines() -> list[str]:
    """
    summary: current PyTorch supports quantized engines
    param none: none
    return: supportslist()
    """
    return [str(engine).lower() for engine in torch.backends.quantized.supported_engines]


def choose_supported_qat_backend(requested_backend: str) -> str:
    """
    summary: current backend, fallback
    param requested_backend: backend
    return: backend
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
        "current PyTorch quantized engine, "
        f"requested={requested} supported={supported}"
    )


def normalize_qat_backend(backend: str) -> str:
    """
    summary: QAT backend
    param backend: backend
    return: backend
    """
    normalized = backend.strip().lower()
    if normalized not in VALID_QAT_BACKENDS:
        raise ValueError(f"unsupported qat_backend: {backend}, optional: {sorted(VALID_QAT_BACKENDS)}")
    return normalized


def set_qat_backend(backend: str) -> str:
    """
    summary: PyTorch quantized engine
    param backend: backend
    return: backend
    """
    selected = choose_supported_qat_backend(backend)
    torch.backends.quantized.engine = selected
    return selected


def prepare_model_for_qat(model: nn.Module, backend: str = "qnnpack") -> nn.Module:
    """
    summary: model QAT training
    param model: model
    param backend: qnnpack fbgemm
    return: model
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
    summary: current QAT observer BN
    param model: QAT model
    param stage_epoch: QAT current, 1
    param total_stage_epochs: QAT
    param disable_observer_last_n_epochs: observer
    param freeze_bn_after_epoch: BN
    return: current
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
    summary: prepare_qat model quantizationinferencemodel
    param model: prepare_qat model
    param backend: qnnpack fbgemm
    return: quantizationinferencemodel(CPU)
    """
    set_qat_backend(backend)
    converted = copy.deepcopy(model).cpu().eval()
    return torch.ao.quantization.convert(converted, inplace=True)


def _strip_qat_modules_to_float_inplace(module: nn.Module) -> None:
    """
    summary: QAT
    param module: process
    return: none
    """
    for name, child in list(module.named_children()):
        _strip_qat_modules_to_float_inplace(child)
        if hasattr(child, "to_float") and "qat" in child.__class__.__module__:
            setattr(module, name, child.to_float())


def strip_prepared_qat_model_to_float(model: nn.Module) -> nn.Module:
    """
    summary: prepare_qat model model
    param model: prepare_qat model
    return: fake quant model(CPU)
    """
    stripped = copy.deepcopy(model).cpu().eval()
    _strip_qat_modules_to_float_inplace(stripped)
    return stripped
