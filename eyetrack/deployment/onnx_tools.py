import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from eyetrack.config import DEFAULT_INPUT_HEIGHT, DEFAULT_INPUT_WIDTH, DEFAULT_NUM_CLASSES
from eyetrack.data.openeds import OpenEDSSegDataset
from eyetrack.data.preprocessing import preprocess_gray_image
from eyetrack.metrics.segmentation import (
    average_dict_values,
    compute_dice_per_class,
    compute_masked_pixel_accuracy,
    compute_pixel_accuracy,
)

VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def require_onnx():
    """
    summary: onnx
    param none: none
    return: onnx
    """
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("missing onnx, requirements onnx.") from exc

    return onnx


def require_onnxruntime():
    """
    summary: onnxruntime
    param none: none
    return: onnxruntime
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("missing onnxruntime, requirements onnxruntime.") from exc

    return ort


def require_onnxruntime_quantization():
    """
    summary: onnxruntime.quantization
    param none: none
    return: onnxruntime.quantization
    """
    try:
        import onnxruntime.quantization as ort_quantization
    except ImportError as exc:
        raise RuntimeError(
            "missing onnxruntime.quantization, quantization onnxruntime."
        ) from exc

    return ort_quantization


def preprocess_onnx_model_file(input_path: str, output_path: Optional[str] = None) -> str:
    """
    summary: ONNX model shape inference
    param input_path: input ONNX path
    param output_path: optionaloutputpath, default input
    return: process modelpath
    """
    onnx = require_onnx()
    src_path = Path(input_path)
    dst_path = Path(output_path) if output_path is not None else src_path

    model = onnx.load(str(src_path))
    onnx.checker.check_model(model)
    inferred = onnx.shape_inference.infer_shapes(model)
    onnx.save(inferred, str(dst_path))
    return str(dst_path)


def get_onnx_input_name(model_path: str) -> str:
    """
    summary: read ONNX model input
    param model_path: modelpath
    return: input
    """
    onnx = require_onnx()
    model = onnx.load(model_path)
    if len(model.graph.input) == 0:
        raise RuntimeError(f"model input: {model_path}")

    return model.graph.input[0].name


def build_onnx_session(model_path: str, backend: str = "cpu", enable_profiling: bool = False):
    """
    summary: ONNX Runtime inference
    param model_path: ONNX modelpath
    param backend: cpu nnapi
    param enable_profiling: enable profiling
    return: ONNX Runtime session
    """
    ort = require_onnxruntime()

    sess_options = ort.SessionOptions()
    sess_options.enable_profiling = enable_profiling
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    available_providers = ort.get_available_providers()

    if backend == "cpu":
        providers: list[Any] = ["CPUExecutionProvider"]
    elif backend == "nnapi":
        if "NNAPIExecutionProvider" not in available_providers:
            raise RuntimeError(
                f"current NNAPIExecutionProvider. providers: {available_providers}"
            )
        providers = ["NNAPIExecutionProvider", "CPUExecutionProvider"]
    else:
        raise ValueError(f"unsupported backend: {backend}")

    return ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)


def load_manifest_paths(manifest_path: str) -> List[str]:
    """
    summary: readimagepathlist
    param manifest_path: filepath, imagepath
    return: imagepathlist
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"not found input manifest: {path}")

    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            items.append(stripped)

    if len(items) == 0:
        raise RuntimeError(f"input manifest imagepath: {path}")

    return items


class CalibrationImageReader:
    """
    summary: ONNX Runtime quantizationcalibration read
    """

    def __init__(
        self,
        image_dir: str,
        input_name: str,
        input_width: int = DEFAULT_INPUT_WIDTH,
        input_height: int = DEFAULT_INPUT_HEIGHT,
        limit: int = 256,
    ):
        self.image_dir = Path(image_dir)
        self.input_name = input_name
        self.input_width = input_width
        self.input_height = input_height
        self.limit = limit
        self.image_paths = self._collect_image_paths()
        self.range_start = 0
        self.range_end = len(self.image_paths)
        self.index = self.range_start

    def _collect_image_paths(self) -> List[Path]:
        if not self.image_dir.exists():
            raise FileNotFoundError(f"not foundcalibrationimagedirectory: {self.image_dir}")

        image_paths = sorted(
            [
                path
                for path in self.image_dir.iterdir()
                if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTS
            ]
        )
        if len(image_paths) == 0:
            raise RuntimeError(f"calibrationimagedirectory: {self.image_dir}")

        return image_paths[: self.limit]

    def __len__(self) -> int:
        return len(self.image_paths)

    def set_range(self, start_index: int = 0, end_index: int | None = None) -> None:
        """
        summary: current reader sample, ORT calibration
        param start_index: index,
        param end_index: index, ; None
        return: none
        """
        total = len(self.image_paths)
        resolved_end = total if end_index is None else min(end_index, total)
        if start_index < 0 or start_index > resolved_end:
            raise ValueError(
                f"none calibration range: start_index={start_index}, end_index={resolved_end}, total={total}"
            )
        self.range_start = start_index
        self.range_end = resolved_end
        self.index = self.range_start

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        if self.index >= self.range_end:
            return None

        image_path = self.image_paths[self.index]
        self.index += 1

        image = Image.open(image_path).convert("L")
        image = np.array(image, dtype=np.float32)
        image = preprocess_gray_image(image=image, input_width=self.input_width, input_height=self.input_height)
        image = np.expand_dims(np.expand_dims(image, axis=0), axis=0).astype(np.float32)
        return {self.input_name: image}

    def rewind(self) -> None:
        self.set_range(0, len(self.image_paths))


def resolve_image_dir(root_dir: str, split: str) -> str:
    """
    summary: parse OpenEDS dataset image directory
    param root_dir: dataset directory
    param split:
    return: image directorypath
    """
    image_dir = Path(root_dir) / split / "images"
    if not image_dir.exists():
        raise FileNotFoundError(f"not foundimagedirectory: {image_dir}")
    return str(image_dir)


def evaluate_onnx_segmentation(
    model_path: str,
    root_dir: str,
    split: str = "validation",
    input_width: int = DEFAULT_INPUT_WIDTH,
    input_height: int = DEFAULT_INPUT_HEIGHT,
    num_classes: int = DEFAULT_NUM_CLASSES,
    backend: str = "cpu",
    batch_size: int = 1,
    num_workers: int = 0,
    limit: int | None = None,
) -> Dict[str, float]:
    """
    summary: labeldataset ONNX model
    param model_path: ONNX modelpath
    param root_dir: dataset directory
    param split:
    param input_width: modelinput
    param input_height: modelinput
    param num_classes: class
    param backend: cpu nnapi
    param batch_size: DataLoader batch
    param num_workers: DataLoader
    param limit: sample
    return: dict
    """
    session = build_onnx_session(model_path=model_path, backend=backend, enable_profiling=False)
    input_name = session.get_inputs()[0].name

    dataset = OpenEDSSegDataset(
        root_dir=root_dir,
        split=split,
        input_width=input_width,
        input_height=input_height,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    total_acc = 0.0
    total_masked_acc = 0.0
    total_batches = 0
    dice_list = []

    for batch in dataloader:
        images = batch["image"].cpu().numpy().astype(np.float32)
        labels = batch["label"]
        masks = batch["mask"]

        logits = session.run(None, {input_name: images})[0]
        preds = torch.from_numpy(np.argmax(logits, axis=1).astype(np.int64))

        acc = compute_pixel_accuracy(preds, labels)
        masked_acc = compute_masked_pixel_accuracy(preds, labels, masks)
        dice = compute_dice_per_class(preds, labels, num_classes=num_classes, ignore_background=True)

        total_acc += acc
        total_masked_acc += masked_acc
        dice_list.append(dice)
        total_batches += 1

        if limit is not None and batch_size * total_batches >= limit:
            break

    avg_dice = average_dict_values(dice_list)
    return {
        "acc": total_acc / max(total_batches, 1),
        "masked_acc": total_masked_acc / max(total_batches, 1),
        "dice_1": avg_dice.get(1, 0.0),
        "dice_2": avg_dice.get(2, 0.0),
        "dice_3": avg_dice.get(3, 0.0),
    }


def compute_mean_foreground_dice(metrics: Dict[str, float]) -> float:
    """
    summary: class Dice
    param metrics: dict
    return: Dice
    """
    return (metrics.get("dice_1", 0.0) + metrics.get("dice_2", 0.0) + metrics.get("dice_3", 0.0)) / 3.0


def summarize_profile_providers(profile_path: str) -> Dict[str, int]:
    """
    summary: ORT profiling JSON provider
    param profile_path: profiling filepath
    return: provider -> event_count
    """
    path = Path(profile_path)
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as file:
        events = json.load(file)

    provider_counts: Dict[str, int] = {}

    for event in events:
        if not isinstance(event, dict):
            continue

        args = event.get("args", {})
        if not isinstance(args, dict):
            continue

        provider = args.get("provider") or args.get("execution_provider") or args.get("provider_type")
        if provider is None:
            continue

        provider_name = str(provider)
        provider_counts[provider_name] = provider_counts.get(provider_name, 0) + 1

    return provider_counts
