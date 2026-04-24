import math
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
from eyetrack.runtime import autocast_context, should_enable_amp


def format_metric_for_postfix(value: float) -> str:
    """
    summary:
    param value:
    return:
    """
    if math.isnan(value):
        return "n/a"

    return f"{value:.4f}"


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = False,
    num_classes: int = 4,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> Dict[str, float]:
    """
    summary: training
    param model: model
    param dataloader: training
    param optimizer:
    param criterion:
    param device: training
    return: training dict
    """
    model.train()

    total_loss = 0.0
    total_acc = 0.0
    masked_acc_values = []
    total_batches = 0
    dice_list = []
    amp_enabled = should_enable_amp(device=device, use_amp=use_amp)

    progress_bar = tqdm(dataloader, desc="Train", leave=False)

    for batch in progress_bar:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        masks = batch.get("mask")
        if masks is not None:
            masks = masks.to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device=device, use_amp=amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)

        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)

            acc = compute_pixel_accuracy(preds, labels)
            if masks is not None:
                masked_acc = compute_masked_pixel_accuracy(preds, labels, masks)
                masked_acc_values.append(masked_acc)
            else:
                masked_acc = float("nan")
            dice = compute_dice_per_class(preds, labels, num_classes=num_classes, ignore_background=True)

        total_loss += loss.item()
        total_acc += acc
        dice_list.append(dice)
        total_batches += 1

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{acc:.4f}",
            "m_acc": format_metric_for_postfix(masked_acc),
        })

    avg_dice = average_dict_values(dice_list)

    return {
        "loss": total_loss / total_batches,
        "acc": total_acc / total_batches,
        "masked_acc": sum(masked_acc_values) / len(masked_acc_values) if len(masked_acc_values) > 0 else float("nan"),
        "dice_1": avg_dice.get(1, 0.0),
        "dice_2": avg_dice.get(2, 0.0),
        "dice_3": avg_dice.get(3, 0.0),
    }


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = False,
    num_classes: int = 4,
) -> Dict[str, float]:
    """
    summary: validation
    param model: model
    param dataloader: validation
    param criterion:
    param device: training
    return: validation dict
    """
    model.eval()

    total_loss = 0.0
    total_acc = 0.0
    masked_acc_values = []
    total_batches = 0
    dice_list = []
    amp_enabled = should_enable_amp(device=device, use_amp=use_amp)

    progress_bar = tqdm(dataloader, desc="Val", leave=False)

    for batch in progress_bar:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        masks = batch.get("mask")
        if masks is not None:
            masks = masks.to(device)

        with autocast_context(device=device, use_amp=amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)

        acc = compute_pixel_accuracy(preds, labels)
        if masks is not None:
            masked_acc = compute_masked_pixel_accuracy(preds, labels, masks)
            masked_acc_values.append(masked_acc)
        else:
            masked_acc = float("nan")
        dice = compute_dice_per_class(preds, labels, num_classes=num_classes, ignore_background=True)

        total_loss += loss.item()
        total_acc += acc
        dice_list.append(dice)
        total_batches += 1

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{acc:.4f}",
            "m_acc": format_metric_for_postfix(masked_acc),
        })

    avg_dice = average_dict_values(dice_list)

    return {
        "loss": total_loss / total_batches,
        "acc": total_acc / total_batches,
        "masked_acc": sum(masked_acc_values) / len(masked_acc_values) if len(masked_acc_values) > 0 else float("nan"),
        "dice_1": avg_dice.get(1, 0.0),
        "dice_2": avg_dice.get(2, 0.0),
        "dice_3": avg_dice.get(3, 0.0),
    }
