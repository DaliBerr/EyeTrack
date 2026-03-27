import torch
from torch.utils.data import DataLoader

from eyetrack.data.openeds import OpenEDSSegDataset
from eyetrack.models.unet import UNet
from eyetrack.training.checkpoints import load_checkpoint_flexible
from eyetrack.visualization.predictions import visualize_predictions


def run_visualization_only(
    root_dir: str,
    checkpoint_path: str,
    save_dir: str = "./vis_val",
    batch_size: int = 4,
    num_workers: int = 0,
    num_samples: int = 12
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    val_dataset = OpenEDSSegDataset(root_dir=root_dir, split="validation")
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = UNet(in_channels=1, num_classes=4, base_channels=32).to(device)

    load_info = load_checkpoint_flexible(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
        optimizer=None,
    )

    print("checkpoint 信息:", load_info)

    visualize_predictions(
        model=model,
        dataloader=val_loader,
        device=device,
        save_dir=save_dir,
        num_samples=num_samples,
    )
