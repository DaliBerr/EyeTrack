import argparse
from pathlib import Path

import torch

from eyetrack.config import (
    DEFAULT_BASE_CHANNELS,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_IN_CHANNELS,
    DEFAULT_INPUT_HEIGHT,
    DEFAULT_INPUT_WIDTH,
    DEFAULT_NUM_CLASSES,
    DEFAULT_ONNX_PATH,
    build_model_metadata,
)
from eyetrack.deployment.onnx_tools import preprocess_onnx_model_file, require_onnx
from eyetrack.models.unet import UNet
from eyetrack.runtime import resolve_device
from eyetrack.training.checkpoints import load_checkpoint_flexible, resolve_model_metadata


def export_checkpoint_to_onnx(
    checkpoint_path: str,
    onnx_path: str = DEFAULT_ONNX_PATH,
    in_channels: int = DEFAULT_IN_CHANNELS,
    num_classes: int = DEFAULT_NUM_CLASSES,
    base_channels: int = DEFAULT_BASE_CHANNELS,
    input_width: int = DEFAULT_INPUT_WIDTH,
    input_height: int = DEFAULT_INPUT_HEIGHT,
    device: str = "cpu",
    opset_version: int = 17,
) -> None:
    """
    summary: 将 PyTorch checkpoint 导出为固定输入尺寸的 ONNX 模型
    param checkpoint_path: PyTorch checkpoint 路径
    param onnx_path: 导出的 ONNX 路径
    param in_channels: 输入通道数
    param num_classes: 输出类别数
    param base_channels: U-Net 基础通道数
    param input_width: 固定输入宽度
    param input_height: 固定输入高度
    param device: 导出设备
    param opset_version: ONNX opset 版本
    return: 无
    """
    torch_device = resolve_device(device)
    resolved_model_metadata = resolve_model_metadata(
        checkpoint_path=checkpoint_path,
        device="cpu",
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        input_width=input_width,
        input_height=input_height,
    )
    print("model metadata:", resolved_model_metadata)

    model = UNet(
        in_channels=int(resolved_model_metadata["in_channels"]),
        num_classes=int(resolved_model_metadata["num_classes"]),
        base_channels=int(resolved_model_metadata["base_channels"]),
    ).to(torch_device)

    load_info = load_checkpoint_flexible(
        model=model,
        checkpoint_path=checkpoint_path,
        device=torch_device,
        optimizer=None,
    )
    print("checkpoint 信息:", load_info)

    model.eval()

    dummy_input = torch.randn(
        1,
        int(resolved_model_metadata["in_channels"]),
        int(resolved_model_metadata["input_height"]),
        int(resolved_model_metadata["input_width"]),
        device=torch_device,
    )

    output_path = Path(onnx_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=None,
    )

    processed_path = preprocess_onnx_model_file(str(output_path))
    onnx = require_onnx()
    model_proto = onnx.load(processed_path)
    metadata = build_model_metadata(
        in_channels=int(resolved_model_metadata["in_channels"]),
        num_classes=int(resolved_model_metadata["num_classes"]),
        base_channels=int(resolved_model_metadata["base_channels"]),
        input_width=int(resolved_model_metadata["input_width"]),
        input_height=int(resolved_model_metadata["input_height"]),
        preprocess_mode=str(resolved_model_metadata["preprocess_mode"]),
        amp=bool(resolved_model_metadata["amp"]),
        use_mask=bool(resolved_model_metadata["use_mask"]),
    )
    onnx.helper.set_model_props(model_proto, {key: str(value) for key, value in metadata.items()})
    onnx.save(model_proto, processed_path)

    print(f"已导出 ONNX 模型: {processed_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将轻量 U-Net checkpoint 导出为固定尺寸 ONNX 模型")
    parser.add_argument("--checkpoint_path", type=str, default=DEFAULT_CHECKPOINT_PATH, help="PyTorch checkpoint 路径")
    parser.add_argument("--onnx_path", type=str, default=DEFAULT_ONNX_PATH, help="ONNX 输出路径")
    parser.add_argument("--in_channels", type=int, default=DEFAULT_IN_CHANNELS, help="模型输入通道数")
    parser.add_argument("--num_classes", type=int, default=DEFAULT_NUM_CLASSES, help="模型输出类别数")
    parser.add_argument("--base_channels", type=int, default=DEFAULT_BASE_CHANNELS, help="U-Net 基础通道数")
    parser.add_argument("--input_width", type=int, default=DEFAULT_INPUT_WIDTH, help="固定输入宽度")
    parser.add_argument("--input_height", type=int, default=DEFAULT_INPUT_HEIGHT, help="固定输入高度")
    parser.add_argument("--device", type=str, default="cpu", help="导出设备，默认 cpu")
    parser.add_argument("--opset_version", type=int, default=17, help="ONNX opset 版本")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_checkpoint_to_onnx(
        checkpoint_path=args.checkpoint_path,
        onnx_path=args.onnx_path,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        base_channels=args.base_channels,
        input_width=args.input_width,
        input_height=args.input_height,
        device=args.device,
        opset_version=args.opset_version,
    )


if __name__ == "__main__":
    main()
