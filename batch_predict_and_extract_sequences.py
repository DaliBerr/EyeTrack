import argparse
import fnmatch
import re
from pathlib import Path
from typing import Any, List, Optional

from extract_geometry_from_segmentation import process_prediction_directory
from eyetrack.config import (
    DEFAULT_BASE_CHANNELS,
    DEFAULT_IN_CHANNELS,
    DEFAULT_INPUT_HEIGHT,
    DEFAULT_INPUT_WIDTH,
    DEFAULT_NUM_CLASSES,
    DEFAULT_USE_AMP,
)
from eyetrack.workflows.predict import run_prediction_to_npy


def natural_key(text: str) -> List[Any]:
    """
    summary:
    param text: input
    return: list
    """
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def list_sequence_dirs(sequence_root: str, sequence_glob: str) -> List[Path]:
    """
    summary: process sequencedirectory
    param sequence_root: sequence directory
    param sequence_glob: sequence
    return: sequencedirectorylist
    """
    root_path = Path(sequence_root)

    if not root_path.exists():
        raise FileNotFoundError(f"not foundsequence directory: {root_path}")

    sequence_dirs = []
    for path in root_path.iterdir():
        if path.is_dir() and fnmatch.fnmatch(path.name, sequence_glob):
            sequence_dirs.append(path)

    sequence_dirs.sort(key=lambda p: natural_key(p.name))
    return sequence_dirs


def resolve_mask_dir(mask_root: Optional[str], sequence_name: str) -> Optional[str]:
    """
    summary: parsecurrentsequence valid directory
    param mask_root: mask directory
    param sequence_name: currentsequence
    return: return mask directory, return None
    """
    if mask_root is None:
        return None

    candidate = Path(mask_root) / sequence_name
    if candidate.exists() and candidate.is_dir():
        return str(candidate)

    return None


def process_all_sequences(
    sequence_root: str,
    checkpoint_path: str,
    output_root: str,
    sequence_glob: str = "S_*",
    mask_root: Optional[str] = None,
    batch_size: int = 1,
    num_workers: int = 0,
    in_channels: int = DEFAULT_IN_CHANNELS,
    num_classes: int = DEFAULT_NUM_CLASSES,
    base_channels: int = DEFAULT_BASE_CHANNELS,
    input_width: int = DEFAULT_INPUT_WIDTH,
    input_height: int = DEFAULT_INPUT_HEIGHT,
    use_amp: bool = DEFAULT_USE_AMP,
    device: str = "auto",
    iris_class_id: int = 2,
    pupil_class_id: int = 3,
    kernel_size: int = 3,
    iris_min_area: int = 100,
    pupil_min_area: int = 20,
    save_overlay: bool = False,
    save_clean_masks: bool = False,
    skip_existing: bool = False,
) -> None:
    """
    summary: batch S sequence prediction arguments
    param sequence_root: S sequence directory
    param checkpoint_path: model checkpoint path
    param output_root: batchoutput directory
    param sequence_glob: sequence
    param mask_root: optional mask directory
    param batch_size: inferencebatch
    param num_workers: DataLoader
    param in_channels: modelinput
    param num_classes: modeloutputclass
    param base_channels: U-Net
    param device:
    param iris_class_id: iris class
    param pupil_class_id: pupil class
    param kernel_size:
    param iris_min_area: iris minimum threshold
    param pupil_min_area: pupil minimum threshold
    param save_overlay: save
    param save_clean_masks: save
    param skip_existing:
    return: none
    """
    sequence_dirs = list_sequence_dirs(sequence_root=sequence_root, sequence_glob=sequence_glob)

    if len(sequence_dirs) == 0:
        raise RuntimeError(f" sequencedirectory: root={sequence_root}, glob={sequence_glob}")

    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    print(f" {len(sequence_dirs)} sequence process.")

    for index, sequence_dir in enumerate(sequence_dirs, start=1):
        sequence_name = sequence_dir.name
        sequence_output_dir = output_root_path / sequence_name
        pred_dir = sequence_output_dir / "predictions"
        output_csv = sequence_output_dir / "geometry.csv"
        overlay_dir = sequence_output_dir / "overlay" if save_overlay else None
        clean_mask_dir = sequence_output_dir / "clean_masks" if save_clean_masks else None
        mask_dir = resolve_mask_dir(mask_root=mask_root, sequence_name=sequence_name)

        if skip_existing and output_csv.exists():
            print(f"\n[{index}/{len(sequence_dirs)}] {sequence_name},: {output_csv}")
            continue

        print(f"\n[{index}/{len(sequence_dirs)}] processsequence: {sequence_name}")
        print(f"image_dir: {sequence_dir}")
        print(f"pred_dir : {pred_dir}")
        print(f"csv_path : {output_csv}")
        if mask_dir is not None:
            print(f"mask_dir : {mask_dir}")

        run_prediction_to_npy(
            checkpoint_path=checkpoint_path,
            output_dir=str(pred_dir),
            image_dir=str(sequence_dir),
            root_dir=None,
            split="validation",
            batch_size=batch_size,
            num_workers=num_workers,
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            input_width=input_width,
            input_height=input_height,
            use_amp=use_amp,
            device=device,
        )

        process_prediction_directory(
            pred_dir=str(pred_dir),
            output_csv=str(output_csv),
            image_dir=str(sequence_dir),
            mask_dir=mask_dir,
            overlay_dir=str(overlay_dir) if overlay_dir is not None else None,
            clean_mask_dir=str(clean_mask_dir) if clean_mask_dir is not None else None,
            iris_class_id=iris_class_id,
            pupil_class_id=pupil_class_id,
            kernel_size=kernel_size,
            iris_min_area=iris_min_area,
            pupil_min_area=pupil_min_area,
            save_overlay=save_overlay,
            save_clean_masks=save_clean_masks,
        )

    print(f"\n sequenceprocesscompleted, output directory: {output_root_path}")


def parse_args() -> argparse.Namespace:
    """
    summary: parseCLIarguments
    param none: none
    return: arguments
    """
    parser = argparse.ArgumentParser(description="batch S sequence prediction geometry")

    parser.add_argument("--sequence_root", type=str, required=True, help=" S_0, S_1 sequencedirectory directory")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="model checkpoint path")
    parser.add_argument("--output_root", type=str, required=True, help="batchoutput directory")

    parser.add_argument("--sequence_glob", type=str, default="S_*", help="sequence, default S_*")
    parser.add_argument("--mask_root", type=str, default=None, help="optional mask directory, mask_root/sequence ")
    parser.add_argument("--skip_existing", action="store_true", help=" geometry.csv sequence")

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

    parser.add_argument("--iris_class_id", type=int, default=2, help="iris class ")
    parser.add_argument("--pupil_class_id", type=int, default=3, help="pupil class ")
    parser.add_argument("--kernel_size", type=int, default=3, help=" ")
    parser.add_argument("--iris_min_area", type=int, default=100, help="iris minimum threshold")
    parser.add_argument("--pupil_min_area", type=int, default=20, help="pupil minimum threshold")

    parser.add_argument("--save_overlay", action="store_true", help=" save ")
    parser.add_argument("--save_clean_masks", action="store_true", help=" save ")

    return parser.parse_args()


def main() -> None:
    """
    summary: main function, batch prediction geometry
    param none: none
    return: none
    """
    args = parse_args()

    process_all_sequences(
        sequence_root=args.sequence_root,
        checkpoint_path=args.checkpoint_path,
        output_root=args.output_root,
        sequence_glob=args.sequence_glob,
        mask_root=args.mask_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        base_channels=args.base_channels,
        input_width=args.input_width,
        input_height=args.input_height,
        use_amp=args.amp,
        device=args.device,
        iris_class_id=args.iris_class_id,
        pupil_class_id=args.pupil_class_id,
        kernel_size=args.kernel_size,
        iris_min_area=args.iris_min_area,
        pupil_min_area=args.pupil_min_area,
        save_overlay=args.save_overlay,
        save_clean_masks=args.save_clean_masks,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
