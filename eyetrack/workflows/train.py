import argparse
import math
import os
import random
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from eyetrack.config import (
    DEFAULT_BASE_CHANNELS,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_IN_CHANNELS,
    DEFAULT_INPUT_HEIGHT,
    DEFAULT_INPUT_WIDTH,
    DEFAULT_NUM_CLASSES,
    DEFAULT_PREPROCESS_MODE,
    DEFAULT_USE_AMP,
    DEFAULT_USE_MASK,
    build_model_metadata,
)
from eyetrack.data.openeds import OpenEDSSegDataset
from eyetrack.models.unet import UNet
from eyetrack.training.checkpoints import load_checkpoint_flexible, save_full_checkpoint
from eyetrack.training.engine import train_one_epoch, validate_one_epoch
from eyetrack.runtime import resolve_device, should_enable_amp


def set_seed(seed: int = 42) -> None:
    """
    summary: 固定随机种子以保证结果可复现
    param seed: 随机种子
    return: 无
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_metrics(epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
    """
    summary: 打印训练与验证指标
    param epoch: 当前轮数
    param train_metrics: 训练指标
    param val_metrics: 验证指标
    return: 无
    """
    def render_metric(value: float) -> str:
        if math.isnan(value):
            return "n/a"

        return f"{value:.4f}"

    print(f"\nEpoch [{epoch}]")
    print(
        f"Train | loss={train_metrics['loss']:.4f} "
        f"acc={train_metrics['acc']:.4f} "
        f"masked_acc={render_metric(train_metrics['masked_acc'])} "
        f"dice1={train_metrics['dice_1']:.4f} "
        f"dice2={train_metrics['dice_2']:.4f} "
        f"dice3={train_metrics['dice_3']:.4f}"
    )
    print(
        f"Val   | loss={val_metrics['loss']:.4f} "
        f"acc={val_metrics['acc']:.4f} "
        f"masked_acc={render_metric(val_metrics['masked_acc'])} "
        f"dice1={val_metrics['dice_1']:.4f} "
        f"dice2={val_metrics['dice_2']:.4f} "
        f"dice3={val_metrics['dice_3']:.4f}"
    )


def run_training(
    root_dir: str,
    batch_size: int = 4,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    num_workers: int = 0,
    save_checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
    resume_checkpoint_path: str | None = None,
    in_channels: int = DEFAULT_IN_CHANNELS,
    num_classes: int = DEFAULT_NUM_CLASSES,
    base_channels: int = DEFAULT_BASE_CHANNELS,
    input_width: int = DEFAULT_INPUT_WIDTH,
    input_height: int = DEFAULT_INPUT_HEIGHT,
    use_amp: bool = DEFAULT_USE_AMP,
    use_mask: bool = DEFAULT_USE_MASK,
    device: str = "auto",
) -> None:
    """
    summary: 训练 OpenEDS 分割模型并支持 AMP 与轻量配置
    param root_dir: 数据集根目录
    param batch_size: 批大小
    param num_epochs: 训练轮数
    param learning_rate: 学习率
    param num_workers: DataLoader 进程数
    param save_checkpoint_path: checkpoint 保存路径
    param resume_checkpoint_path: 可选续训 checkpoint
    param in_channels: 输入通道数
    param num_classes: 类别数量
    param base_channels: U-Net 基础通道数
    param input_width: 模型输入宽度
    param input_height: 模型输入高度
    param use_amp: 是否启用 AMP
    param use_mask: 是否读取 mask 并计算 masked_acc
    param device: 运行设备
    return: 无
    """
    set_seed(42)

    torch_device = resolve_device(device)
    amp_enabled = should_enable_amp(device=torch_device, use_amp=use_amp)
    print("device:", torch_device)
    print("amp:", amp_enabled)

    train_dataset = OpenEDSSegDataset(
        root_dir=root_dir,
        split="train",
        input_width=input_width,
        input_height=input_height,
        use_mask=use_mask,
    )
    val_dataset = OpenEDSSegDataset(
        root_dir=root_dir,
        split="validation",
        input_width=input_width,
        input_height=input_height,
        use_mask=use_mask,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch_device.type == "cuda",
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch_device.type == "cuda",
    )

    model = UNet(in_channels=in_channels, num_classes=num_classes, base_channels=base_channels).to(torch_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    metadata = build_model_metadata(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        input_width=input_width,
        input_height=input_height,
        preprocess_mode=DEFAULT_PREPROCESS_MODE,
        amp=use_amp,
        use_mask=use_mask,
    )

    start_epoch = 1
    best_val_loss = float("inf")

    if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
        load_info = load_checkpoint_flexible(
            model=model,
            checkpoint_path=resume_checkpoint_path,
            device=torch_device,
            optimizer=optimizer,
        )

        start_epoch = load_info["start_epoch"]
        best_val_loss = load_info["best_val_loss"]

    for epoch in range(start_epoch, num_epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=torch_device,
            use_amp=use_amp,
            num_classes=num_classes,
            scaler=scaler,
        )

        val_metrics = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=torch_device,
            use_amp=use_amp,
            num_classes=num_classes,
        )

        print_metrics(epoch, train_metrics, val_metrics)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]

            save_full_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_loss=best_val_loss,
                save_path=save_checkpoint_path,
                metadata=metadata,
            )

            print("已保存当前最佳完整 checkpoint。")


def parse_args() -> argparse.Namespace:
    """
    summary: 解析训练命令行参数
    param 无: 无
    return: 参数对象
    """
    parser = argparse.ArgumentParser(description="训练轻量 U-Net 眼部分割模型")

    parser.add_argument("--root_dir", type=str, required=True, help="OpenEDS 风格数据集根目录")
    parser.add_argument("--batch_size", type=int, default=4, help="训练批大小")
    parser.add_argument("--num_epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="优化器学习率")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader 进程数")

    parser.add_argument("--save_checkpoint_path", type=str, default=DEFAULT_CHECKPOINT_PATH, help="最佳模型 checkpoint 保存路径")
    parser.add_argument("--resume_checkpoint_path", type=str, default=None, help="可选续训 checkpoint 路径")

    parser.add_argument("--in_channels", type=int, default=DEFAULT_IN_CHANNELS, help="模型输入通道数")
    parser.add_argument("--num_classes", type=int, default=DEFAULT_NUM_CLASSES, help="模型输出类别数")
    parser.add_argument("--base_channels", type=int, default=DEFAULT_BASE_CHANNELS, help="U-Net 基础通道数")
    parser.add_argument("--input_width", type=int, default=DEFAULT_INPUT_WIDTH, help="模型输入宽度")
    parser.add_argument("--input_height", type=int, default=DEFAULT_INPUT_HEIGHT, help="模型输入高度")
    parser.add_argument("--device", type=str, default="auto", help="运行设备: auto/cpu/cuda")
    parser.add_argument("--amp", action="store_true", default=DEFAULT_USE_AMP, help="启用 CUDA AMP")
    parser.add_argument("--no-amp", action="store_false", dest="amp", help="禁用 CUDA AMP")
    parser.add_argument("--use_mask", action="store_true", default=DEFAULT_USE_MASK, help="读取 mask 并计算 masked_acc")
    parser.add_argument("--no-use_mask", action="store_false", dest="use_mask", help="不读取 mask，加快数据加载")

    return parser.parse_args()


def main() -> None:
    """
    summary: 训练入口
    param 无: 无
    return: 无
    """
    args = parse_args()

    run_training(
        root_dir=args.root_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        save_checkpoint_path=args.save_checkpoint_path,
        resume_checkpoint_path=args.resume_checkpoint_path,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        base_channels=args.base_channels,
        input_width=args.input_width,
        input_height=args.input_height,
        use_amp=args.amp,
        use_mask=args.use_mask,
        device=args.device,
    )
