from typing import Dict

import torch


def compute_pixel_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    summary: image
    param pred: predictionclass, shape BxHxW
    param target: class, shape BxHxW
    return:
    """
    correct = (pred == target).float().sum().item()
    total = target.numel()
    return correct / total


def compute_masked_pixel_accuracy(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """
    summary: mask
    param pred: predictionclass, shape BxHxW
    param target: class, shape BxHxW
    param mask: valid mask, shape BxHxW
    return: mask
    """
    valid = mask.bool()
    valid_count = valid.sum().item()

    if valid_count == 0:
        return 0.0

    correct = (pred[valid] == target[valid]).float().sum().item()
    return correct / valid_count


def compute_dice_per_class(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_background: bool = False
) -> Dict[int, float]:
    """
    summary: class Dice
    param pred: predictionclass, shape BxHxW
    param target: class, shape BxHxW
    param num_classes: classcount
    param ignore_background:
    return: class Dice dict
    """
    dice_dict = {}
    start_class = 1 if ignore_background else 0

    for cls in range(start_class, num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()

        intersection = (pred_cls * target_cls).sum().item()
        union = pred_cls.sum().item() + target_cls.sum().item()

        if union == 0:
            dice = 1.0
        else:
            dice = (2.0 * intersection) / union

        dice_dict[cls] = dice

    return dice_dict


def average_dict_values(dict_list: list[Dict[int, float]]) -> Dict[int, float]:
    """
    summary: dict
    param dict_list: dictlist
    return: dict
    """
    if len(dict_list) == 0:
        return {}

    keys = dict_list[0].keys()
    avg_dict = {}

    for key in keys:
        avg_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)

    return avg_dict
