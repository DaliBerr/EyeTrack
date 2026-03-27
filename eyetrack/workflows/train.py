import os
import random
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from eyetrack.data.openeds import OpenEDSSegDataset
from eyetrack.models.unet import UNet
from eyetrack.training.checkpoints import load_checkpoint_flexible, save_full_checkpoint
from eyetrack.training.engine import train_one_epoch, validate_one_epoch


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
    print(f"\nEpoch [{epoch}]")
    print(
        f"Train | loss={train_metrics['loss']:.4f} "
        f"acc={train_metrics['acc']:.4f} "
        f"masked_acc={train_metrics['masked_acc']:.4f} "
        f"dice1={train_metrics['dice_1']:.4f} "
        f"dice2={train_metrics['dice_2']:.4f} "
        f"dice3={train_metrics['dice_3']:.4f}"
    )
    print(
        f"Val   | loss={val_metrics['loss']:.4f} "
        f"acc={val_metrics['acc']:.4f} "
        f"masked_acc={val_metrics['masked_acc']:.4f} "
        f"dice1={val_metrics['dice_1']:.4f} "
        f"dice2={val_metrics['dice_2']:.4f} "
        f"dice3={val_metrics['dice_3']:.4f}"
    )


def main() -> None:
    """
    summary: 训练 OpenEDS 分割模型并支持自动续训
    param 无: 无
    return: 无
    """
    set_seed(42)

    root_dir = r"D:\Code\DataSet\OpenEDS\openEDS\openEDS"
    batch_size = 4
    num_epochs = 10
    learning_rate = 1e-3
    num_workers = 0

    resume_checkpoint_path = "./checkpoints/best_unet_openeds.pth"
    save_checkpoint_path = "./checkpoints/best_unet_openeds_full.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_dataset = OpenEDSSegDataset(root_dir=root_dir, split="train")
    val_dataset = OpenEDSSegDataset(root_dir=root_dir, split="validation")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = UNet(in_channels=1, num_classes=4, base_channels=32).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    start_epoch = 1
    best_val_loss = float("inf")

    if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
        load_info = load_checkpoint_flexible(
            model=model,
            checkpoint_path=resume_checkpoint_path,
            device=device,
            optimizer=optimizer,
            fallback_start_epoch=3,
            fallback_best_val_loss=0.0094,
        )

        start_epoch = load_info["start_epoch"]
        best_val_loss = load_info["best_val_loss"]

    for epoch in range(start_epoch, num_epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        val_metrics = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
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
            )

            print("已保存当前最佳完整 checkpoint。")
