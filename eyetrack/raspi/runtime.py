from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from eyetrack.config import DEFAULT_INPUT_HEIGHT, DEFAULT_INPUT_WIDTH
from eyetrack.deployment.onnx_tools import build_onnx_session, require_onnx


def ensure_quantized_onnx_model(model_path: str) -> None:
    onnx = require_onnx()
    model = onnx.load(model_path)
    op_types = {node.op_type for node in model.graph.node}
    if "QuantizeLinear" not in op_types and "DequantizeLinear" not in op_types:
        raise RuntimeError(f"树莓派实时脚本只接受量化 ONNX 模型，但该模型不像 QDQ INT8 ONNX: {model_path}")


@dataclass(frozen=True)
class PiOnnxModelInfo:
    model_path: str
    input_name: str
    input_width: int
    input_height: int
    input_channels: int
    providers: tuple[str, ...]


def _resolve_dim(shape: list[Any], index: int, fallback: int) -> int:
    if index >= len(shape):
        return fallback
    value = shape[index]
    return int(value) if isinstance(value, int) else fallback


class PiOnnxSegmentationRuntime:
    def __init__(self, model_path: str, backend: str = "cpu"):
        model_file = Path(model_path)
        if model_file.suffix.lower() != ".onnx":
            raise RuntimeError(f"树莓派实时脚本仅支持 .onnx 模型，当前路径: {model_path}")
        if not model_file.exists():
            raise FileNotFoundError(f"未找到 ONNX 模型: {model_file}")
        if backend != "cpu":
            raise RuntimeError(f"树莓派实时脚本当前仅支持 ONNX Runtime CPU backend，收到: {backend}")

        ensure_quantized_onnx_model(str(model_file))
        self.session = build_onnx_session(model_path=str(model_file), backend=backend, enable_profiling=False)
        input_meta = self.session.get_inputs()[0]
        input_shape = list(input_meta.shape)
        metadata = self.session.get_modelmeta().custom_metadata_map
        self.info = PiOnnxModelInfo(
            model_path=str(model_file),
            input_name=input_meta.name,
            input_width=int(metadata.get("input_width", _resolve_dim(input_shape, 3, DEFAULT_INPUT_WIDTH))),
            input_height=int(metadata.get("input_height", _resolve_dim(input_shape, 2, DEFAULT_INPUT_HEIGHT))),
            input_channels=int(metadata.get("in_channels", _resolve_dim(input_shape, 1, 1))),
            providers=tuple(self.session.get_providers()),
        )

        if self.info.input_channels != 1:
            raise RuntimeError(f"树莓派实时脚本只支持单通道灰度模型，当前模型通道数: {self.info.input_channels}")

    @property
    def input_width(self) -> int:
        return self.info.input_width

    @property
    def input_height(self) -> int:
        return self.info.input_height

    @property
    def input_name(self) -> str:
        return self.info.input_name

    @property
    def providers(self) -> tuple[str, ...]:
        return self.info.providers

    def predict_label_map(self, input_tensor: np.ndarray) -> np.ndarray:
        logits = self.session.run(None, {self.info.input_name: input_tensor.astype(np.float32, copy=False)})[0]
        return np.argmax(logits, axis=1)[0].astype(np.uint8)
