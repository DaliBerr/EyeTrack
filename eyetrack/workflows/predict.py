import argparse
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from eyetrack.models.unet import UNet
from eyetrack.training.checkpoints import load_checkpoint_flexible


VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def natural_key(text: str) -> List[Any]:
    """
    summary: 生成自然排序键
    param text: 输入字符串
    return: 可用于排序的键列表
    """
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


class SegmentationInferenceDataset(Dataset):
    """
    summary: 仅用于推理的灰度图数据集
    param image_dir: 输入图像目录
    return: 可供 DataLoader 使用的数据集对象
    """

    def __init__(self, image_dir: str):
        """
        summary: 初始化推理数据集并收集图像路径
        param image_dir: 输入图像目录
        return: 无
        """
        self.image_dir = Path(image_dir)

        if not self.image_dir.exists():
            raise FileNotFoundError(f"找不到图像目录: {self.image_dir}")

        self.image_paths = self._collect_image_paths()

        if len(self.image_paths) == 0:
            raise RuntimeError(f"目录中没有找到可用图像: {self.image_dir}")

    def _collect_image_paths(self) -> List[Path]:
        """
        summary: 收集目录中所有可用图像路径
        param self: 数据集实例
        return: 图像路径列表
        """
        image_paths = []

        for path in self.image_dir.iterdir():
            if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTS:
                image_paths.append(path)

        image_paths.sort(key=lambda p: natural_key(p.name))
        return image_paths

    def __len__(self) -> int:
        """
        summary: 返回图像数量
        param self: 数据集实例
        return: 数据集长度
        """
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        summary: 读取单张灰度图并转为张量
        param index: 图像索引
        return: 包含 image 和 id 的字典
        """
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("L")
        image = np.array(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image).unsqueeze(0).float()

        return {
            "image": image_tensor,
            "id": image_path.stem,
        }


def resolve_device(device: str) -> torch.device:
    """
    summary: 根据字符串解析运行设备
    param device: auto/cpu/cuda 或具体设备名
    return: torch 设备对象
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前环境不可用 CUDA，但传入了 --device cuda")

    return torch.device(device)


def resolve_image_dir(image_dir: str | None, root_dir: str | None, split: str) -> Path:
    """
    summary: 解析实际用于推理的图像目录
    param image_dir: 显式传入的图像目录
    param root_dir: 数据集根目录
    param split: 数据划分名称
    return: 最终图像目录路径
    """
    if image_dir is not None:
        return Path(image_dir)

    if root_dir is not None:
        return Path(root_dir) / split / "images"

    raise ValueError("必须提供 --image_dir，或提供 --root_dir 与 --split。")


def run_prediction_to_npy(
    checkpoint_path: str,
    output_dir: str,
    image_dir: str | None = None,
    root_dir: str | None = None,
    split: str = "validation",
    batch_size: int = 1,
    num_workers: int = 0,
    in_channels: int = 1,
    num_classes: int = 4,
    base_channels: int = 32,
    device: str = "auto"
) -> None:
    """
    summary: 执行分割模型推理并将预测标签保存为 npy
    param checkpoint_path: 模型 checkpoint 路径
    param output_dir: 预测标签输出目录
    param image_dir: 输入图像目录
    param root_dir: 数据集根目录
    param split: 数据划分名称
    param batch_size: 批大小
    param num_workers: DataLoader 进程数
    param in_channels: 模型输入通道数
    param num_classes: 模型输出类别数
    param base_channels: U-Net 基础通道数
    param device: 运行设备
    return: 无
    """
    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    torch_device = resolve_device(device)
    print("device:", torch_device)

    resolved_image_dir = resolve_image_dir(image_dir=image_dir, root_dir=root_dir, split=split)
    print("image_dir:", resolved_image_dir)

    dataset = SegmentationInferenceDataset(image_dir=str(resolved_image_dir))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch_device.type == "cuda",
    )

    model = UNet(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
    ).to(torch_device)

    load_info = load_checkpoint_flexible(
        model=model,
        checkpoint_path=checkpoint_path,
        device=torch_device,
        optimizer=None,
    )
    print("checkpoint 信息:", load_info)

    model.eval()
    saved_count = 0

    with torch.inference_mode():
        for batch in dataloader:
            images = batch["image"].to(torch_device)
            sample_ids = batch["id"]

            logits = model(images)
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.uint8)

            for pred, sample_id in zip(preds, sample_ids):
                save_path = save_dir / f"{sample_id}.npy"
                np.save(save_path, pred)
                saved_count += 1
                print(f"[{saved_count}/{len(dataset)}] 已保存预测: {save_path}")

    print(f"\n预测完成，共保存 {saved_count} 个 .npy 文件到: {save_dir}")


def parse_args() -> argparse.Namespace:
    """
    summary: 解析命令行参数
    param 无: 无
    return: 参数对象
    """
    parser = argparse.ArgumentParser(description="执行分割模型推理并将结果保存为 .npy 标签图")

    parser.add_argument("--checkpoint_path", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument("--output_dir", type=str, required=True, help="预测结果 .npy 输出目录")
    parser.add_argument("--image_dir", type=str, default=None, help="输入灰度图目录")
    parser.add_argument("--root_dir", type=str, default=None, help="数据集根目录，若提供则读取 root_dir/split/images")
    parser.add_argument("--split", type=str, default="validation", help="数据划分名称，默认 validation")

    parser.add_argument("--batch_size", type=int, default=1, help="推理批大小")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader 进程数")
    parser.add_argument("--in_channels", type=int, default=1, help="模型输入通道数")
    parser.add_argument("--num_classes", type=int, default=4, help="模型输出类别数")
    parser.add_argument("--base_channels", type=int, default=32, help="U-Net 基础通道数")
    parser.add_argument("--device", type=str, default="auto", help="运行设备: auto/cpu/cuda")

    return parser.parse_args()


def main() -> None:
    """
    summary: 主函数，执行预测并保存 npy
    param 无: 无
    return: 无
    """
    args = parse_args()

    run_prediction_to_npy(
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        image_dir=args.image_dir,
        root_dir=args.root_dir,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        base_channels=args.base_channels,
        device=args.device,
    )


if __name__ == "__main__":
    main()
