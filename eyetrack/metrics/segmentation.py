from typing import Dict

import torch


def compute_pixel_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    summary: 计算全图像素准确率
    param pred: 预测类别图，shape 为 BxHxW
    param target: 真实类别图，shape 为 BxHxW
    return: 准确率
    """
    correct = (pred == target).float().sum().item()
    total = target.numel()
    return correct / total


def compute_masked_pixel_accuracy(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """
    summary: 计算 mask 区域内的像素准确率
    param pred: 预测类别图，shape 为 BxHxW
    param target: 真实类别图，shape 为 BxHxW
    param mask: 有效区域掩码，shape 为 BxHxW
    return: mask 区域内准确率
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
    summary: 计算每个类别的 Dice
    param pred: 预测类别图，shape 为 BxHxW
    param target: 真实类别图，shape 为 BxHxW
    param num_classes: 类别数量
    param ignore_background: 是否忽略背景类
    return: 每个类别对应的 Dice 字典
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
    summary: 对多个字典中的同名键取平均
    param dict_list: 字典列表
    return: 平均后的字典
    """
    if len(dict_list) == 0:
        return {}

    keys = dict_list[0].keys()
    avg_dict = {}

    for key in keys:
        avg_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)

    return avg_dict
