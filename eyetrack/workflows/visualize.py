import torch
from torch.utils.data import DataLoader

from eyetrack.config import (
    DEFAULT_BASE_CHANNELS,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_IN_CHANNELS,
    DEFAULT_INPUT_HEIGHT,
    DEFAULT_INPUT_WIDTH,
    DEFAULT_NUM_CLASSES,
    DEFAULT_USE_AMP,
)
from eyetrack.data.openeds import OpenEDSSegDataset
from eyetrack.models.unet import UNet
from eyetrack.runtime import resolve_device
from eyetrack.training.checkpoints import load_checkpoint_flexible, resolve_model_metadata
from eyetrack.visualization.predictions import visualize_predictions


def run_visualization_only(
    root_dir: str,
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
    save_dir: str = "./vis_val",
    batch_size: int = 4,
    num_workers: int = 0,
    num_samples: int = 12,
    in_channels: int = DEFAULT_IN_CHANNELS,
    num_classes: int = DEFAULT_NUM_CLASSES,
    base_channels: int = DEFAULT_BASE_CHANNELS,
    input_width: int = DEFAULT_INPUT_WIDTH,
    input_height: int = DEFAULT_INPUT_HEIGHT,
    use_amp: bool = DEFAULT_USE_AMP,
    device: str = "auto",
) -> None:
    """
    summary: 仅加载模型并在验证集上保存预测可视化
    param root_dir: 数据集根目录
    param checkpoint_path: 模型 checkpoint 路径
    param save_dir: 可视化结果目录
    param batch_size: 验证批大小
    param num_workers: DataLoader 进程数
    param num_samples: 最多保存样本数
    return: 无
    """
    torch_device = resolve_device(device)
    resolved_model_metadata = resolve_model_metadata(
        checkpoint_path=checkpoint_path,
        device="cpu",
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        input_width=input_width,
        input_height=input_height,
        amp=use_amp,
    )
    print("device:", torch_device)
    print("model metadata:", resolved_model_metadata)

    val_dataset = OpenEDSSegDataset(
        root_dir=root_dir,
        split="validation",
        input_width=int(resolved_model_metadata["input_width"]),
        input_height=int(resolved_model_metadata["input_height"]),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch_device.type == "cuda",
    )

    model = UNet(
        in_channels=int(resolved_model_metadata["in_channels"]),
        num_classes=int(resolved_model_metadata["num_classes"]),
        base_channels=int(resolved_model_metadata["base_channels"]),
    ).to(torch_device)

    load_info = load_checkpoint_flexible(
        model=model,
        checkpoint_path=checkpoint_path,
        device=torch_device,
        optimizer=None,
    )

    print("checkpoint 信息:", load_info)

    visualize_predictions(
        model=model,
        dataloader=val_loader,
        device=torch_device,
        save_dir=save_dir,
        num_samples=num_samples,
        use_amp=bool(resolved_model_metadata["amp"]),
    )
