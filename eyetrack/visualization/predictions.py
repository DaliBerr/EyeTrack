import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from eyetrack.runtime import autocast_context


def colorize_label_map(label_map: np.ndarray) -> np.ndarray:
    """
    summary: classlabel
    param label_map: classlabel
    return:
    """
    color_table = np.array([
        [0, 0, 0],
        [80, 80, 80],
        [180, 180, 180],
        [255, 255, 255],
    ], dtype=np.uint8)

    color_image = color_table[label_map]
    return color_image


def build_error_map(
    pred: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    summary:,,
    param pred: predictionlabel
    param target: label
    param mask: optionalvalid mask
    return:
    """
    error = (pred != target).astype(np.uint8)

    if mask is not None:
        error = error * mask.astype(np.uint8)

    return error


@torch.no_grad()
def visualize_predictions(
    model: nn.Module,
    dataloader,
    device: torch.device,
    save_dir: str,
    num_samples: int = 12,
    use_amp: bool = False,
) -> None:
    """
    summary: savevalidation prediction
    param model: trainingmodel
    param dataloader: validation DataLoader
    param device: current
    param save_dir: savedirectory
    param num_samples: save sample
    return: none
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    saved_count = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        masks = batch["mask"].to(device)
        sample_ids = batch["id"]

        with autocast_context(device=device, use_amp=use_amp):
            logits = model(images)
        preds = torch.argmax(logits, dim=1)

        batch_size = images.size(0)

        for i in range(batch_size):
            image = images[i, 0].detach().cpu().numpy()
            label = labels[i].detach().cpu().numpy()
            pred = preds[i].detach().cpu().numpy()
            mask = masks[i].detach().cpu().numpy().astype(np.uint8)
            sample_id = sample_ids[i]

            label_color = colorize_label_map(label)
            pred_color = colorize_label_map(pred)

            pred_in_mask = pred.copy()
            pred_in_mask[mask == 0] = 0
            pred_in_mask_color = colorize_label_map(pred_in_mask)

            error_in_mask = build_error_map(pred, label, mask=mask)

            fig = plt.figure(figsize=(16, 8))

            ax1 = plt.subplot(2, 3, 1)
            ax1.imshow(image, cmap="gray")
            ax1.set_title("image")
            ax1.axis("off")

            ax2 = plt.subplot(2, 3, 2)
            ax2.imshow(label_color)
            ax2.set_title("label")
            ax2.axis("off")

            ax3 = plt.subplot(2, 3, 3)
            ax3.imshow(pred_color)
            ax3.set_title("prediction")
            ax3.axis("off")

            ax4 = plt.subplot(2, 3, 4)
            ax4.imshow(mask, cmap="gray")
            ax4.set_title("mask")
            ax4.axis("off")

            ax5 = plt.subplot(2, 3, 5)
            ax5.imshow(pred_in_mask_color)
            ax5.set_title("prediction_in_mask")
            ax5.axis("off")

            ax6 = plt.subplot(2, 3, 6)
            ax6.imshow(error_in_mask, cmap="gray")
            ax6.set_title("error_in_mask")
            ax6.axis("off")

            plt.suptitle(f"sample_id = {sample_id}")
            plt.tight_layout()

            save_path = os.path.join(save_dir, f"{sample_id}_vis.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            saved_count += 1
            print(f"Saved visualization: {save_path}")

            if saved_count >= num_samples:
                print(f"Visualization finished, total saved {saved_count}.")
                return

    print(f"Visualization finished, total saved {saved_count}.")
