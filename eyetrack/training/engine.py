from typing import Dict

import torch
import torch.nn as nn
from tqdm import tqdm

from eyetrack.metrics.segmentation import (
    average_dict_values,
    compute_dice_per_class,
    compute_masked_pixel_accuracy,
    compute_pixel_accuracy,
)


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    summary: 执行一轮训练
    param model: 分割模型
    param dataloader: 训练数据加载器
    param optimizer: 优化器
    param criterion: 损失函数
    param device: 训练设备
    return: 本轮训练指标字典
    """
    model.train()

    total_loss = 0.0
    total_acc = 0.0
    total_masked_acc = 0.0
    total_batches = 0
    dice_list = []

    progress_bar = tqdm(dataloader, desc="Train", leave=False)

    for batch in progress_bar:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)

            acc = compute_pixel_accuracy(preds, labels)
            masked_acc = compute_masked_pixel_accuracy(preds, labels, masks)
            dice = compute_dice_per_class(preds, labels, num_classes=4, ignore_background=True)

        total_loss += loss.item()
        total_acc += acc
        total_masked_acc += masked_acc
        dice_list.append(dice)
        total_batches += 1

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{acc:.4f}",
            "m_acc": f"{masked_acc:.4f}",
        })

    avg_dice = average_dict_values(dice_list)

    return {
        "loss": total_loss / total_batches,
        "acc": total_acc / total_batches,
        "masked_acc": total_masked_acc / total_batches,
        "dice_1": avg_dice.get(1, 0.0),
        "dice_2": avg_dice.get(2, 0.0),
        "dice_3": avg_dice.get(3, 0.0),
    }


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    summary: 执行一轮验证
    param model: 分割模型
    param dataloader: 验证数据加载器
    param criterion: 损失函数
    param device: 训练设备
    return: 本轮验证指标字典
    """
    model.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_masked_acc = 0.0
    total_batches = 0
    dice_list = []

    progress_bar = tqdm(dataloader, desc="Val", leave=False)

    for batch in progress_bar:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        masks = batch["mask"].to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)

        acc = compute_pixel_accuracy(preds, labels)
        masked_acc = compute_masked_pixel_accuracy(preds, labels, masks)
        dice = compute_dice_per_class(preds, labels, num_classes=4, ignore_background=True)

        total_loss += loss.item()
        total_acc += acc
        total_masked_acc += masked_acc
        dice_list.append(dice)
        total_batches += 1

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{acc:.4f}",
            "m_acc": f"{masked_acc:.4f}",
        })

    avg_dice = average_dict_values(dice_list)

    return {
        "loss": total_loss / total_batches,
        "acc": total_acc / total_batches,
        "masked_acc": total_masked_acc / total_batches,
        "dice_1": avg_dice.get(1, 0.0),
        "dice_2": avg_dice.get(2, 0.0),
        "dice_3": avg_dice.get(3, 0.0),
    }
