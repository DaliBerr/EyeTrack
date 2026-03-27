from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def read_gray_image(image_path: str) -> np.ndarray:
    """
    summary: 读取灰度图并归一化到 0~1
    param image_path: 图像路径
    return: float32 的二维数组，shape 为 HxW
    """
    image = Image.open(image_path).convert("L")
    image = np.array(image, dtype=np.float32) / 255.0
    return image


def read_label_npy(label_path: str) -> np.ndarray:
    """
    summary: 读取分割标签 npy
    param label_path: 标签路径
    return: int64 的二维数组，shape 为 HxW
    """
    label = np.load(label_path)
    label = label.astype(np.int64)
    return label


def read_mask_png(mask_path: str) -> np.ndarray:
    """
    summary: 读取二值 mask 并转为 0/1
    param mask_path: mask 路径
    return: uint8 的二维数组，shape 为 HxW
    """
    mask = Image.open(mask_path).convert("L")
    mask = np.array(mask, dtype=np.uint8)
    mask = (mask > 0).astype(np.uint8)
    return mask


class OpenEDSSegDataset(Dataset):
    """
    summary: OpenEDS 单帧语义分割数据集
    param root_dir: 数据集根目录
    param split: 数据划分，如 train/validation/test
    return: 可供 PyTorch 读取的数据集对象
    """

    def __init__(self, root_dir: str, split: str):
        """
        summary: 初始化数据集并收集样本 id
        param root_dir: 数据集根目录
        param split: 数据划分名称
        return: 无
        """
        self.root_dir = Path(root_dir)
        self.split = split

        self.image_dir = self.root_dir / split / "images"
        self.label_dir = self.root_dir / split / "labels"
        self.mask_dir = self.root_dir / split / "masks"

        if not self.image_dir.exists():
            raise FileNotFoundError(f"找不到图像目录: {self.image_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"找不到标签目录: {self.label_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"找不到 mask 目录: {self.mask_dir}")

        self.sample_ids = self._collect_sample_ids()

        if len(self.sample_ids) == 0:
            raise RuntimeError(f"{split} 中没有找到可用样本")

    def _collect_sample_ids(self) -> List[str]:
        """
        summary: 收集同时拥有 image/label/mask 的样本编号
        param self: 数据集实例
        return: 样本编号列表
        """
        image_paths = sorted(self.image_dir.glob("*.png"))
        sample_ids = []

        for image_path in image_paths:
            sample_id = image_path.stem
            label_path = self.label_dir / f"{sample_id}.npy"
            mask_path = self.mask_dir / f"{sample_id}.png"

            if label_path.exists() and mask_path.exists():
                sample_ids.append(sample_id)

        return sample_ids

    def __len__(self) -> int:
        """
        summary: 返回样本数量
        param self: 数据集实例
        return: 数据集长度
        """
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        summary: 读取单个样本并转为张量
        param index: 样本索引
        return: 包含 image、label、mask、id 的字典
        """
        sample_id = self.sample_ids[index]

        image_path = self.image_dir / f"{sample_id}.png"
        label_path = self.label_dir / f"{sample_id}.npy"
        mask_path = self.mask_dir / f"{sample_id}.png"

        image = read_gray_image(str(image_path))
        label = read_label_npy(str(label_path))
        mask = read_mask_png(str(mask_path))

        image_tensor = torch.from_numpy(image).unsqueeze(0).float()
        label_tensor = torch.from_numpy(label).long()
        mask_tensor = torch.from_numpy(mask).bool()

        return {
            "image": image_tensor,
            "label": label_tensor,
            "mask": mask_tensor,
            "id": sample_id,
        }
