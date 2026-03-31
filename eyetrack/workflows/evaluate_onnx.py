import argparse
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from eyetrack.config import (
    DEFAULT_BASE_CHANNELS,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_IN_CHANNELS,
    DEFAULT_INPUT_HEIGHT,
    DEFAULT_INPUT_WIDTH,
    DEFAULT_NUM_CLASSES,
    DEFAULT_ONNX_PATH,
    DEFAULT_USE_AMP,
)
from eyetrack.data.openeds import OpenEDSSegDataset
from eyetrack.deployment.onnx_tools import evaluate_onnx_segmentation
from eyetrack.metrics.segmentation import average_dict_values, compute_dice_per_class
from eyetrack.models.unet import UNet
from eyetrack.runtime import autocast_context, resolve_device
from eyetrack.training.checkpoints import load_checkpoint_flexible


def compare_onnx_with_pytorch(
    model_path: str,
    checkpoint_path: str,
    root_dir: str,
    split: str = "validation",
    input_width: int = DEFAULT_INPUT_WIDTH,
    input_height: int = DEFAULT_INPUT_HEIGHT,
    in_channels: int = DEFAULT_IN_CHANNELS,
    num_classes: int = DEFAULT_NUM_CLASSES,
    base_channels: int = DEFAULT_BASE_CHANNELS,
    use_amp: bool = DEFAULT_USE_AMP,
    device: str = "auto",
    batch_size: int = 1,
    num_workers: int = 0,
    limit: int | None = None,
) -> Dict[str, float]:
    """
    summary: 比较 PyTorch checkpoint 与 ONNX 模型的输出一致性
    param model_path: ONNX 模型路径
    param checkpoint_path: PyTorch checkpoint 路径
    param root_dir: 数据集根目录
    param split: 数据划分
    param input_width: 模型输入宽度
    param input_height: 模型输入高度
    param in_channels: 输入通道数
    param num_classes: 类别数量
    param base_channels: U-Net 基础通道数
    param use_amp: 是否启用 AMP
    param device: 运行设备
    param batch_size: DataLoader 批大小
    param num_workers: DataLoader 进程数
    param limit: 最多比较多少个样本
    return: 一致性指标
    """
    import onnxruntime as ort

    dataset = OpenEDSSegDataset(
        root_dir=root_dir,
        split=split,
        input_width=input_width,
        input_height=input_height,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    torch_device = resolve_device(device)
    model = UNet(in_channels=in_channels, num_classes=num_classes, base_channels=base_channels).to(torch_device)
    load_info = load_checkpoint_flexible(model=model, checkpoint_path=checkpoint_path, device=torch_device, optimizer=None)
    print("PyTorch checkpoint 信息:", load_info)
    model.eval()

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    total_agreement = 0.0
    total_batches = 0
    dice_list = []

    with torch.inference_mode():
        for batch in dataloader:
            images = batch["image"].to(torch_device)
            onnx_inputs = batch["image"].cpu().numpy().astype(np.float32)

            with autocast_context(device=torch_device, use_amp=use_amp):
                torch_logits = model(images)
            torch_preds = torch.argmax(torch_logits, dim=1).cpu()

            onnx_logits = session.run(None, {input_name: onnx_inputs})[0]
            onnx_preds = torch.from_numpy(np.argmax(onnx_logits, axis=1).astype(np.int64))

            agreement = (torch_preds == onnx_preds).float().mean().item()
            dice = compute_dice_per_class(onnx_preds, torch_preds, num_classes=num_classes, ignore_background=True)

            total_agreement += agreement
            total_batches += 1
            dice_list.append(dice)

            if limit is not None and batch_size * total_batches >= limit:
                break

    avg_dice = average_dict_values(dice_list)
    return {
        "prediction_agreement": total_agreement / max(total_batches, 1),
        "compare_dice_1": avg_dice.get(1, 0.0),
        "compare_dice_2": avg_dice.get(2, 0.0),
        "compare_dice_3": avg_dice.get(3, 0.0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 ONNX 分割模型，并可选与 PyTorch checkpoint 对比")
    parser.add_argument("--model_path", type=str, default=DEFAULT_ONNX_PATH, help="ONNX 模型路径")
    parser.add_argument("--root_dir", type=str, required=True, help="OpenEDS 风格数据集根目录")
    parser.add_argument("--split", type=str, default="validation", help="数据划分")
    parser.add_argument("--backend", type=str, default="cpu", choices=["cpu", "nnapi"], help="ONNX Runtime backend")
    parser.add_argument("--input_width", type=int, default=DEFAULT_INPUT_WIDTH, help="模型输入宽度")
    parser.add_argument("--input_height", type=int, default=DEFAULT_INPUT_HEIGHT, help="模型输入高度")
    parser.add_argument("--batch_size", type=int, default=1, help="评估批大小")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader 进程数")
    parser.add_argument("--limit", type=int, default=None, help="最多评估多少个样本")

    parser.add_argument("--checkpoint_path", type=str, default=None, help="可选 PyTorch checkpoint，用于一致性对比")
    parser.add_argument("--in_channels", type=int, default=DEFAULT_IN_CHANNELS, help="模型输入通道数")
    parser.add_argument("--num_classes", type=int, default=DEFAULT_NUM_CLASSES, help="模型输出类别数")
    parser.add_argument("--base_channels", type=int, default=DEFAULT_BASE_CHANNELS, help="U-Net 基础通道数")
    parser.add_argument("--device", type=str, default="auto", help="PyTorch 对比时的运行设备")
    parser.add_argument("--amp", action="store_true", default=DEFAULT_USE_AMP, help="PyTorch 对比时启用 AMP")
    parser.add_argument("--no-amp", action="store_false", dest="amp", help="PyTorch 对比时禁用 AMP")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    onnx_metrics = evaluate_onnx_segmentation(
        model_path=args.model_path,
        root_dir=args.root_dir,
        split=args.split,
        input_width=args.input_width,
        input_height=args.input_height,
        num_classes=args.num_classes,
        backend=args.backend,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        limit=args.limit,
    )
    print("ONNX metrics:", onnx_metrics)

    if args.checkpoint_path is None:
        return

    compare_metrics = compare_onnx_with_pytorch(
        model_path=args.model_path,
        checkpoint_path=args.checkpoint_path,
        root_dir=args.root_dir,
        split=args.split,
        input_width=args.input_width,
        input_height=args.input_height,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        base_channels=args.base_channels,
        use_amp=args.amp,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        limit=args.limit,
    )
    print("PyTorch vs ONNX compare metrics:", compare_metrics)


if __name__ == "__main__":
    main()
