import argparse
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from eyetrack.config import (
    DEFAULT_BASE_CHANNELS,
    DEFAULT_IN_CHANNELS,
    DEFAULT_INPUT_HEIGHT,
    DEFAULT_INPUT_WIDTH,
    DEFAULT_NUM_CLASSES,
    DEFAULT_USE_AMP,
)
from eyetrack.data.preprocessing import preprocess_gray_image
from eyetrack.models.unet import UNet
from eyetrack.runtime import autocast_context, resolve_device
from eyetrack.training.checkpoints import load_checkpoint_flexible, resolve_model_metadata


VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def natural_key(text: str) -> List[Any]:
    """
    summary:
    param text: input
    return: list
    """
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


class SegmentationInferenceDataset(Dataset):
    """
    summary: inference dataset
    param image_dir: inputimagedirectory
    return: DataLoader dataset
    """

    def __init__(self, image_dir: str, input_width: int = DEFAULT_INPUT_WIDTH, input_height: int = DEFAULT_INPUT_HEIGHT):
        """
        summary: inferencedataset imagepath
        param image_dir: inputimagedirectory
        return: none
        """
        self.image_dir = Path(image_dir)
        self.input_width = input_width
        self.input_height = input_height

        if not self.image_dir.exists():
            raise FileNotFoundError(f"not foundimagedirectory: {self.image_dir}")

        self.image_paths = self._collect_image_paths()

        if len(self.image_paths) == 0:
            raise RuntimeError(f"directory image: {self.image_dir}")

    def _collect_image_paths(self) -> List[Path]:
        """
        summary: directory imagepath
        param self: dataset
        return: imagepathlist
        """
        image_paths = []

        for path in self.image_dir.iterdir():
            if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTS:
                image_paths.append(path)

        image_paths.sort(key=lambda p: natural_key(p.name))
        return image_paths

    def __len__(self) -> int:
        """
        summary: returnimagecount
        param self: dataset
        return: dataset
        """
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        summary: read
        param index: imageindex
        return: image id dict
        """
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("L")
        image = np.array(image, dtype=np.float32)
        image = preprocess_gray_image(image=image, input_width=self.input_width, input_height=self.input_height)
        image_tensor = torch.from_numpy(image).unsqueeze(0).float()

        return {
            "image": image_tensor,
            "id": image_path.stem,
        }

def resolve_image_dir(image_dir: str | None, root_dir: str | None, split: str) -> Path:
    """
    summary: parse inference imagedirectory
    param image_dir: imagedirectory
    param root_dir: dataset directory
    param split:
    return: imagedirectorypath
    """
    if image_dir is not None:
        return Path(image_dir)

    if root_dir is not None:
        return Path(root_dir) / split / "images"

    raise ValueError("You must provide --image_dir, or provide both --root_dir and --split.")


def run_prediction_to_npy(
    checkpoint_path: str,
    output_dir: str,
    image_dir: str | None = None,
    root_dir: str | None = None,
    split: str = "validation",
    batch_size: int = 1,
    num_workers: int = 0,
    in_channels: int = DEFAULT_IN_CHANNELS,
    num_classes: int = DEFAULT_NUM_CLASSES,
    base_channels: int = DEFAULT_BASE_CHANNELS,
    input_width: int = DEFAULT_INPUT_WIDTH,
    input_height: int = DEFAULT_INPUT_HEIGHT,
    use_amp: bool = DEFAULT_USE_AMP,
    device: str = "auto",
) -> None:
    """
    summary: modelinference predictionlabelsave npy
    param checkpoint_path: model checkpoint path
    param output_dir: predictionlabeloutputdirectory
    param image_dir: inputimagedirectory
    param root_dir: dataset directory
    param split:
    param batch_size: batch
    param num_workers: DataLoader
    param in_channels: modelinput
    param num_classes: modeloutputclass
    param base_channels: U-Net
    param device:
    return: none
    """
    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

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

    resolved_image_dir = resolve_image_dir(image_dir=image_dir, root_dir=root_dir, split=split)
    print("image_dir:", resolved_image_dir)

    dataset = SegmentationInferenceDataset(
        image_dir=str(resolved_image_dir),
        input_width=int(resolved_model_metadata["input_width"]),
        input_height=int(resolved_model_metadata["input_height"]),
    )
    dataloader = DataLoader(
        dataset,
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
    print("checkpoint info:", load_info)

    model.eval()
    saved_count = 0

    with torch.inference_mode():
        for batch in dataloader:
            images = batch["image"].to(torch_device)
            sample_ids = batch["id"]

            with autocast_context(device=torch_device, use_amp=bool(resolved_model_metadata["amp"])):
                logits = model(images)
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.uint8)

            for pred, sample_id in zip(preds, sample_ids):
                save_path = save_dir / f"{sample_id}.npy"
                np.save(save_path, pred)
                saved_count += 1
                print(f"[{saved_count}/{len(dataset)}] saveprediction: {save_path}")

    print(f"\npredictioncompleted, save {saved_count}.npy file: {save_dir}")


def parse_args() -> argparse.Namespace:
    """
    summary: parseCLIarguments
    param none: none
    return: arguments
    """
    parser = argparse.ArgumentParser(description=" modelinference save.npy label ")

    parser.add_argument("--checkpoint_path", type=str, required=True, help="model checkpoint path")
    parser.add_argument("--output_dir", type=str, required=True, help="prediction.npy outputdirectory")
    parser.add_argument("--image_dir", type=str, default=None, help="input directory")
    parser.add_argument("--root_dir", type=str, default=None, help="dataset directory, read root_dir/split/images")
    parser.add_argument("--split", type=str, default="validation", help=", default validation")

    parser.add_argument("--batch_size", type=int, default=1, help="inferencebatch ")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader ")
    parser.add_argument("--in_channels", type=int, default=DEFAULT_IN_CHANNELS, help="modelinput ")
    parser.add_argument("--num_classes", type=int, default=DEFAULT_NUM_CLASSES, help="modeloutputclass ")
    parser.add_argument("--base_channels", type=int, default=DEFAULT_BASE_CHANNELS, help="U-Net ")
    parser.add_argument("--input_width", type=int, default=DEFAULT_INPUT_WIDTH, help="modelinput ")
    parser.add_argument("--input_height", type=int, default=DEFAULT_INPUT_HEIGHT, help="modelinput ")
    parser.add_argument("--amp", action="store_true", default=DEFAULT_USE_AMP, help="enable CUDA AMP inference")
    parser.add_argument("--no-amp", action="store_false", dest="amp", help="disable CUDA AMP inference")
    parser.add_argument("--device", type=str, default="auto", help=": auto/cpu/cuda")

    return parser.parse_args()


def main() -> None:
    """
    summary: main function, prediction save npy
    param none: none
    return: none
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
        input_width=args.input_width,
        input_height=args.input_height,
        use_amp=args.amp,
        device=args.device,
    )


if __name__ == "__main__":
    main()
