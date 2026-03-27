from eyetrack.training.checkpoints import (
    load_checkpoint_flexible,
    save_checkpoint,
    save_full_checkpoint,
)
from eyetrack.workflows.train import main, print_metrics, set_seed
from eyetrack.workflows.visualize import run_visualization_only

__all__ = [
    "load_checkpoint_flexible",
    "main",
    "print_metrics",
    "run_visualization_only",
    "save_checkpoint",
    "save_full_checkpoint",
    "set_seed",
]


if __name__ == "__main__":
    run_visualization_only(
        root_dir=r"D:\Code\DataSet\OpenEDS\openEDS\openEDS",
        checkpoint_path="./checkpoints/best_unet_openeds.pth",
        save_dir="./vis_val",
        batch_size=6,
        num_workers=0,
        num_samples=12,
    )
