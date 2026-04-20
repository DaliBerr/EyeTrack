from typing import Any, Dict


DEFAULT_IN_CHANNELS = 1
DEFAULT_NUM_CLASSES = 4
DEFAULT_BASE_CHANNELS = 16
DEFAULT_INPUT_WIDTH = 384
DEFAULT_INPUT_HEIGHT = 240
DEFAULT_PREPROCESS_MODE = "raw_resize"
DEFAULT_USE_AMP = True
DEFAULT_USE_MASK = True
DEFAULT_QUANTIZATION_MODE = "fp32"
DEFAULT_QAT_BACKEND = "qnnpack"

DEFAULT_CHECKPOINT_PATH = "./checkpoints/best_unet_b16_384x240_amp.pth"
DEFAULT_ONNX_PATH = "./checkpoints/unet_b16_384x240_fp32.onnx"
DEFAULT_INT8_ONNX_PATH = "./checkpoints/unet_b16_384x240_int8_qdq.onnx"

DEFAULT_B8_CHECKPOINT_PATH = "./checkpoints/best_unet_b8_384x240_amp.pth"
DEFAULT_B8_ONNX_PATH = "./checkpoints/unet_b8_384x240_fp32.onnx"
DEFAULT_B8_INT8_ONNX_PATH = "./checkpoints/unet_b8_384x240_int8_qdq.onnx"


def build_model_metadata(
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
    summary: 构建模型配置元数据，便于 checkpoint 与导出流程复用
    param in_channels: 输入通道数
    param num_classes: 输出类别数
    param base_channels: U-Net 基础通道数
    param input_width: 模型输入宽度
    param input_height: 模型输入高度
    param preprocess_mode: 预处理模式
    param amp: 是否使用 AMP 作为训练基线
    param use_mask: 训练时是否读取 mask
    param quantization_mode: 量化模式标记，默认 fp32
    param qat_backend: QAT backend，默认 qnnpack
    return: 元数据字典
    """
    return {
        "in_channels": in_channels,
        "num_classes": num_classes,
        "base_channels": base_channels,
        "input_width": input_width,
        "input_height": input_height,
        "preprocess_mode": preprocess_mode,
        "amp": amp,
        "use_mask": use_mask,
        "quantization_mode": quantization_mode,
        "qat_backend": qat_backend,
    }
