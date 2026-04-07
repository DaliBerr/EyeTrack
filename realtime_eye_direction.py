import argparse
import time
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import cv2
import numpy as np
import torch
from torch import nn

from eyetrack.config import (
    DEFAULT_IN_CHANNELS,
    DEFAULT_BASE_CHANNELS,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_INPUT_HEIGHT,
    DEFAULT_INPUT_WIDTH,
    DEFAULT_NUM_CLASSES,
    DEFAULT_USE_AMP,
)
from eyetrack.data.preprocessing import ResizeMeta, preprocess_bgr_frame
from eyetrack.models.unet import UNet
from eyetrack.runtime import autocast_context, resolve_device
from eyetrack.training.checkpoints import load_checkpoint_flexible, resolve_model_metadata


# =========================
# 配置区
# =========================
CHECKPOINT_PATH = DEFAULT_CHECKPOINT_PATH
CAMERA_INDEX = 0
WINDOW_NAME = "Realtime Eye Direction Demo"

PREFERRED_CAMERA_WIDTH = 1280
PREFERRED_CAMERA_HEIGHT = 720
PREFERRED_CAMERA_FPS = 60

MODEL_INPUT_WIDTH = DEFAULT_INPUT_WIDTH
MODEL_INPUT_HEIGHT = DEFAULT_INPUT_HEIGHT

MIN_SELECTED_ROI_SIZE = 40

BASE_CHANNELS = DEFAULT_BASE_CHANNELS
NUM_CLASSES = DEFAULT_NUM_CLASSES
USE_AMP = DEFAULT_USE_AMP

IRIS_CLASS_ID = 2
PUPIL_CLASS_ID = 3

EMA_ALPHA = 0.35
MAX_VALID_NORM_RADIUS = 0.85
ARROW_LENGTH = 120
SHOW_DEBUG_TEXT = True
USE_MIRROR_VIEW = False

ROI_PREVIEW_WIDTH = 220
ROI_PREVIEW_HEIGHT = 140
PRED_PREVIEW_WIDTH = 220
PRED_PREVIEW_HEIGHT = 140


# =========================
# 数据结构
# =========================
@dataclass
class EllipseResult:
    center_x: float
    center_y: float
    major_axis: float
    minor_axis: float
    angle_deg: float


@dataclass
class RegionGeometry:
    area: int
    center_x: Optional[float]
    center_y: Optional[float]
    ellipse: Optional[EllipseResult]


@dataclass
class ROIBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def clip(self, frame_width: int, frame_height: int) -> "ROIBox":
        return ROIBox(
            x1=max(0, min(self.x1, frame_width)),
            y1=max(0, min(self.y1, frame_height)),
            x2=max(0, min(self.x2, frame_width)),
            y2=max(0, min(self.y2, frame_height)),
        )

    def is_valid(self, min_size: int = MIN_SELECTED_ROI_SIZE) -> bool:
        return self.width >= min_size and self.height >= min_size

    def crop(self, frame: np.ndarray) -> np.ndarray:
        return frame[self.y1:self.y2, self.x1:self.x2]


@dataclass
class ROIInteractionState:
    active_roi: Optional[ROIBox] = None
    drag_start: Optional[Tuple[int, int]] = None
    drag_current: Optional[Tuple[int, int]] = None
    frame_width: int = 0
    frame_height: int = 0

    def set_frame_shape(self, frame: np.ndarray) -> None:
        self.frame_height, self.frame_width = frame.shape[:2]

    def clear(self) -> None:
        self.active_roi = None
        self.drag_start = None
        self.drag_current = None

    def get_drag_roi(self) -> Optional[ROIBox]:
        if self.drag_start is None or self.drag_current is None:
            return None

        x1 = min(self.drag_start[0], self.drag_current[0])
        y1 = min(self.drag_start[1], self.drag_current[1])
        x2 = max(self.drag_start[0], self.drag_current[0])
        y2 = max(self.drag_start[1], self.drag_current[1])
        return ROIBox(x1=x1, y1=y1, x2=x2, y2=y2).clip(self.frame_width, self.frame_height)


class OnlineEMAFilter:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.dx: Optional[float] = None
        self.dy: Optional[float] = None

    def reset(self) -> None:
        self.dx = None
        self.dy = None

    def update(self, dx: Optional[float], dy: Optional[float], valid: bool) -> Tuple[Optional[float], Optional[float]]:
        if not valid or dx is None or dy is None:
            return self.dx, self.dy

        if self.dx is None or self.dy is None:
            self.dx = dx
            self.dy = dy
        else:
            self.dx = self.alpha * dx + (1.0 - self.alpha) * self.dx
            self.dy = self.alpha * dy + (1.0 - self.alpha) * self.dy

        return self.dx, self.dy


ROI_STATE = ROIInteractionState()


@dataclass
class RuntimeModelConfig:
    checkpoint_path: str
    in_channels: int
    num_classes: int
    base_channels: int
    input_width: int
    input_height: int
    use_amp: bool


# =========================
# 鼠标交互
# =========================
def handle_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
    del flags, param

    if ROI_STATE.frame_width <= 0 or ROI_STATE.frame_height <= 0:
        return

    x = int(np.clip(x, 0, max(ROI_STATE.frame_width - 1, 0)))
    y = int(np.clip(y, 0, max(ROI_STATE.frame_height - 1, 0)))

    if event == cv2.EVENT_LBUTTONDOWN:
        ROI_STATE.drag_start = (x, y)
        ROI_STATE.drag_current = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and ROI_STATE.drag_start is not None:
        ROI_STATE.drag_current = (x, y)
    elif event == cv2.EVENT_LBUTTONUP and ROI_STATE.drag_start is not None:
        ROI_STATE.drag_current = (x, y)
        drag_roi = ROI_STATE.get_drag_roi()
        if drag_roi is not None and drag_roi.is_valid():
            ROI_STATE.active_roi = drag_roi
        ROI_STATE.drag_start = None
        ROI_STATE.drag_current = None


# =========================
# 相机与预处理
# =========================
def configure_camera(cap: cv2.VideoCapture) -> Tuple[int, int, float]:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, PREFERRED_CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PREFERRED_CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, PREFERRED_CAMERA_FPS)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    return width, height, fps


def preprocess_frame(frame_bgr: np.ndarray, input_width: int, input_height: int) -> Tuple[np.ndarray, torch.Tensor, ResizeMeta]:
    """
    summary: 将 ROI 图像预处理为模型输入，采用与训练一致的 raw-resize 流程
    param frame_bgr: 输入 ROI 的 BGR 图像
    return: 灰度预览图、模型张量、预处理元信息
    """
    return preprocess_bgr_frame(
        frame_bgr=frame_bgr,
        input_width=input_width,
        input_height=input_height,
    )


@torch.no_grad()
def predict_label_map(model: nn.Module, input_tensor: torch.Tensor, device: torch.device, use_amp: bool) -> np.ndarray:
    with autocast_context(device=device, use_amp=use_amp):
        logits = model(input_tensor.to(device))
    pred = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)
    return pred


# =========================
# 分割与几何
# =========================
def build_empty_geometry_result() -> Dict[str, Any]:
    empty_region = RegionGeometry(area=0, center_x=None, center_y=None, ellipse=None)
    return {
        "iris_mask": None,
        "pupil_mask": None,
        "iris_geometry": empty_region,
        "pupil_geometry": empty_region,
        "norm_dx": None,
        "norm_dy": None,
        "norm_radius": None,
        "valid": False,
    }


def extract_class_binary_mask(label_map: np.ndarray, class_id: int) -> np.ndarray:
    return (label_map == class_id).astype(np.uint8)


def keep_largest_connected_component(binary_mask: np.ndarray) -> np.ndarray:
    if binary_mask.sum() == 0:
        return binary_mask.copy()

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels <= 1:
        return binary_mask.copy()

    largest_label = 1
    largest_area = stats[1, cv2.CC_STAT_AREA]

    for label_id in range(2, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area > largest_area:
            largest_area = area
            largest_label = label_id

    return (labels == largest_label).astype(np.uint8)


def clean_binary_mask(binary_mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    if binary_mask.sum() == 0:
        return binary_mask.copy()

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    return cleaned


def compute_mask_centroid(binary_mask: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    if binary_mask.sum() == 0:
        return None, None

    moments = cv2.moments(binary_mask)
    if abs(moments["m00"]) < 1e-8:
        return None, None

    center_x = moments["m10"] / moments["m00"]
    center_y = moments["m01"] / moments["m00"]
    return float(center_x), float(center_y)


def fit_ellipse_to_mask(binary_mask: np.ndarray) -> Optional[EllipseResult]:
    if binary_mask.sum() == 0:
        return None

    contour_img = (binary_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(contour_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return None

    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5:
        return None

    ellipse = cv2.fitEllipse(contour)
    (center_x, center_y), (axis_a, axis_b), angle_deg = ellipse

    major_axis = float(max(axis_a, axis_b))
    minor_axis = float(min(axis_a, axis_b))
    fixed_angle = float(angle_deg)

    if axis_b > axis_a:
        fixed_angle = (fixed_angle + 90.0) % 180.0

    return EllipseResult(
        center_x=float(center_x),
        center_y=float(center_y),
        major_axis=major_axis,
        minor_axis=minor_axis,
        angle_deg=fixed_angle,
    )


def extract_region_geometry(binary_mask: np.ndarray) -> RegionGeometry:
    area = int(binary_mask.sum())
    if area == 0:
        return RegionGeometry(area=0, center_x=None, center_y=None, ellipse=None)

    center_x, center_y = compute_mask_centroid(binary_mask)
    ellipse = fit_ellipse_to_mask(binary_mask)
    return RegionGeometry(area=area, center_x=center_x, center_y=center_y, ellipse=ellipse)


def rotate_vector(dx: float, dy: float, angle_deg: float) -> Tuple[float, float]:
    theta = math.radians(angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    return cos_t * dx - sin_t * dy, sin_t * dx + cos_t * dy


def extract_eye_geometry_from_label_map(pred_label_map: np.ndarray) -> Dict[str, Any]:
    iris_mask = extract_class_binary_mask(pred_label_map, IRIS_CLASS_ID)
    pupil_mask = extract_class_binary_mask(pred_label_map, PUPIL_CLASS_ID)

    iris_mask = keep_largest_connected_component(iris_mask)
    pupil_mask = keep_largest_connected_component(pupil_mask)

    iris_mask = clean_binary_mask(iris_mask, kernel_size=3)
    pupil_mask = clean_binary_mask(pupil_mask, kernel_size=3)

    iris_geometry = extract_region_geometry(iris_mask)
    pupil_geometry = extract_region_geometry(pupil_mask)

    norm_dx = None
    norm_dy = None
    norm_radius = None

    if (
        iris_geometry.center_x is not None and iris_geometry.center_y is not None and
        pupil_geometry.center_x is not None and pupil_geometry.center_y is not None and
        iris_geometry.ellipse is not None
    ):
        offset_dx = float(pupil_geometry.center_x - iris_geometry.center_x)
        offset_dy = float(pupil_geometry.center_y - iris_geometry.center_y)

        local_dx, local_dy = rotate_vector(offset_dx, offset_dy, angle_deg=-iris_geometry.ellipse.angle_deg)
        semi_major = iris_geometry.ellipse.major_axis / 2.0
        semi_minor = iris_geometry.ellipse.minor_axis / 2.0

        if semi_major > 1e-6:
            norm_dx = float(local_dx / semi_major)
        if semi_minor > 1e-6:
            norm_dy = float(local_dy / semi_minor)
        if norm_dx is not None and norm_dy is not None:
            norm_radius = float(math.sqrt(norm_dx * norm_dx + norm_dy * norm_dy))

    valid = (
        iris_geometry.ellipse is not None and
        pupil_geometry.ellipse is not None and
        norm_dx is not None and
        norm_dy is not None and
        norm_radius is not None and
        norm_radius <= MAX_VALID_NORM_RADIUS
    )

    return {
        "iris_mask": iris_mask,
        "pupil_mask": pupil_mask,
        "iris_geometry": iris_geometry,
        "pupil_geometry": pupil_geometry,
        "norm_dx": norm_dx,
        "norm_dy": norm_dy,
        "norm_radius": norm_radius,
        "valid": valid,
    }


# =========================
# 可视化与坐标映射
# =========================
def map_point_to_frame(x: float, y: float, preprocess_meta: ResizeMeta, roi_box: ROIBox) -> Tuple[int, int]:
    source_x = float(np.clip(x * preprocess_meta.scale_x, 0.0, max(preprocess_meta.source_width - 1.0, 0.0)))
    source_y = float(np.clip(y * preprocess_meta.scale_y, 0.0, max(preprocess_meta.source_height - 1.0, 0.0)))

    frame_x = int(round(roi_box.x1 + source_x))
    frame_y = int(round(roi_box.y1 + source_y))
    return frame_x, frame_y


def map_axis_radii_to_frame(ellipse: EllipseResult, preprocess_meta: ResizeMeta) -> Tuple[int, int]:
    major_radius = max(1, int(round((ellipse.major_axis / 2.0) * preprocess_meta.scale_x)))
    minor_radius = max(1, int(round((ellipse.minor_axis / 2.0) * preprocess_meta.scale_y)))
    return major_radius, minor_radius


def draw_ellipse_on_frame(
    frame: np.ndarray,
    ellipse: Optional[EllipseResult],
    preprocess_meta: Optional[ResizeMeta],
    roi_box: Optional[ROIBox],
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> np.ndarray:
    output = frame.copy()
    if ellipse is None or preprocess_meta is None or roi_box is None:
        return output

    center = map_point_to_frame(ellipse.center_x, ellipse.center_y, preprocess_meta, roi_box)
    axes = map_axis_radii_to_frame(ellipse, preprocess_meta)

    cv2.ellipse(output, center=center, axes=axes, angle=ellipse.angle_deg, startAngle=0, endAngle=360, color=color, thickness=thickness)
    return output


def draw_center_on_frame(
    frame: np.ndarray,
    x: Optional[float],
    y: Optional[float],
    preprocess_meta: Optional[ResizeMeta],
    roi_box: Optional[ROIBox],
    color: Tuple[int, int, int],
    radius: int = 4,
) -> np.ndarray:
    output = frame.copy()
    if x is None or y is None or preprocess_meta is None or roi_box is None:
        return output

    center = map_point_to_frame(x, y, preprocess_meta, roi_box)
    cv2.circle(output, center, radius, color, -1)
    return output


def draw_direction_arrow(
    frame: np.ndarray,
    iris_geometry: RegionGeometry,
    preprocess_meta: Optional[ResizeMeta],
    roi_box: Optional[ROIBox],
    dx: Optional[float],
    dy: Optional[float],
) -> np.ndarray:
    output = frame.copy()

    if iris_geometry.center_x is None or iris_geometry.center_y is None:
        return output
    if dx is None or dy is None:
        return output
    if preprocess_meta is None or roi_box is None:
        return output

    start_pt = map_point_to_frame(iris_geometry.center_x, iris_geometry.center_y, preprocess_meta, roi_box)
    end_pt = (int(round(start_pt[0] + dx * ARROW_LENGTH)), int(round(start_pt[1] + dy * ARROW_LENGTH)))
    cv2.arrowedLine(output, start_pt, end_pt, (0, 255, 255), 3, tipLength=0.22)
    return output


def draw_corner_indicator(frame: np.ndarray, dx: Optional[float], dy: Optional[float]) -> np.ndarray:
    output = frame.copy()
    if dx is None or dy is None:
        return output

    frame_height, frame_width = output.shape[:2]
    center_x = frame_width - 160
    center_y = 90

    cv2.rectangle(output, (center_x - 60, center_y - 60), (center_x + 60, center_y + 60), (80, 80, 80), 1)
    cv2.circle(output, (center_x, center_y), 3, (255, 255, 255), -1)

    tip = (int(round(center_x + dx * 55)), int(round(center_y + dy * 55)))
    cv2.arrowedLine(output, (center_x, center_y), tip, (255, 255, 0), 2, tipLength=0.2)
    return output


def draw_roi_overlay(frame: np.ndarray, selected_roi: Optional[ROIBox]) -> np.ndarray:
    output = frame.copy()

    if selected_roi is not None:
        cv2.rectangle(output, (selected_roi.x1, selected_roi.y1), (selected_roi.x2, selected_roi.y2), (0, 255, 0), 2)
        cv2.putText(output, "selected ROI", (selected_roi.x1, max(20, selected_roi.y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

    drag_roi = ROI_STATE.get_drag_roi()
    if drag_roi is not None:
        cv2.rectangle(output, (drag_roi.x1, drag_roi.y1), (drag_roi.x2, drag_roi.y2), (255, 255, 0), 1)

    return output


def put_debug_text(
    frame: np.ndarray,
    geometry_result: Dict[str, Any],
    preprocess_meta: Optional[ResizeMeta],
    smoothed_dx: Optional[float],
    smoothed_dy: Optional[float],
    fps: float,
    camera_resolution: Tuple[int, int],
    selected_roi: Optional[ROIBox],
) -> np.ndarray:
    output = frame.copy()

    lines = [
        f"FPS: {fps:.1f}",
        f"camera: {camera_resolution[0]}x{camera_resolution[1]}",
        "ROI: drag left mouse to select | c clear | r reset ema | q quit",
    ]

    if selected_roi is not None:
        lines.append(f"selected_roi: {selected_roi.width}x{selected_roi.height}")
    else:
        lines.append("selected_roi: none")

    if preprocess_meta is not None:
        lines.append(f"preprocess: {preprocess_meta.source_width}x{preprocess_meta.source_height}->{preprocess_meta.input_width}x{preprocess_meta.input_height}")
        lines.append(f"scale_xy: {preprocess_meta.scale_x:.3f}, {preprocess_meta.scale_y:.3f}")
    else:
        lines.append("preprocess: none")
        lines.append("scale_xy: none")

    lines.extend([
        f"valid: {geometry_result['valid']}",
        f"norm_dx: {geometry_result['norm_dx']}",
        f"norm_dy: {geometry_result['norm_dy']}",
        f"norm_r : {geometry_result['norm_radius']}",
        f"sm_dx  : {smoothed_dx}",
        f"sm_dy  : {smoothed_dy}",
    ])

    y = 24
    for line in lines:
        cv2.putText(output, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        y += 23

    return output


def draw_preview_panel(frame: np.ndarray, image: np.ndarray, label: str, top_left: Tuple[int, int], size: Tuple[int, int], use_gray: bool) -> np.ndarray:
    output = frame.copy()
    panel_width, panel_height = size
    x, y = top_left

    if use_gray:
        panel = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        panel = image.copy()

    panel = cv2.resize(panel, (panel_width, panel_height), interpolation=cv2.INTER_NEAREST)

    frame_height, frame_width = output.shape[:2]
    x = int(np.clip(x, 0, max(frame_width - panel_width, 0)))
    y = int(np.clip(y, 0, max(frame_height - panel_height, 0)))

    output[y:y + panel_height, x:x + panel_width] = panel
    cv2.rectangle(output, (x, y), (x + panel_width, y + panel_height), (255, 255, 255), 1)
    cv2.putText(output, label, (x + 8, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def build_visualization(
    frame_bgr: np.ndarray,
    geometry_result: Dict[str, Any],
    preprocess_meta: Optional[ResizeMeta],
    selected_roi: Optional[ROIBox],
    smoothed_dx: Optional[float],
    smoothed_dy: Optional[float],
    fps: float,
    camera_resolution: Tuple[int, int],
) -> np.ndarray:
    vis = frame_bgr.copy()

    vis = draw_ellipse_on_frame(vis, geometry_result["iris_geometry"].ellipse, preprocess_meta, selected_roi, (0, 255, 0), 2)
    vis = draw_ellipse_on_frame(vis, geometry_result["pupil_geometry"].ellipse, preprocess_meta, selected_roi, (0, 0, 255), 2)

    vis = draw_center_on_frame(vis, geometry_result["iris_geometry"].center_x, geometry_result["iris_geometry"].center_y, preprocess_meta, selected_roi, (0, 255, 0), 4)
    vis = draw_center_on_frame(vis, geometry_result["pupil_geometry"].center_x, geometry_result["pupil_geometry"].center_y, preprocess_meta, selected_roi, (0, 0, 255), 4)

    vis = draw_direction_arrow(vis, geometry_result["iris_geometry"], preprocess_meta, selected_roi, smoothed_dx, smoothed_dy)
    vis = draw_corner_indicator(vis, smoothed_dx, smoothed_dy)
    vis = draw_roi_overlay(vis, selected_roi)

    if SHOW_DEBUG_TEXT:
        vis = put_debug_text(vis, geometry_result, preprocess_meta, smoothed_dx, smoothed_dy, fps, camera_resolution, selected_roi)

    return vis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实时眼动方向验证")
    parser.add_argument("--checkpoint_path", type=str, default=CHECKPOINT_PATH, help="待加载的 checkpoint 路径")
    parser.add_argument("--camera_index", type=int, default=CAMERA_INDEX, help="摄像头索引")
    parser.add_argument("--device", type=str, default="auto", help="运行设备: auto/cpu/cuda")

    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true", help="强制启用 CUDA AMP")
    amp_group.add_argument("--no-amp", dest="amp", action="store_false", help="强制禁用 AMP")
    parser.set_defaults(amp=None)

    return parser.parse_args()


def resolve_runtime_model_config(checkpoint_path: str, amp_override: Optional[bool]) -> RuntimeModelConfig:
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"未找到 checkpoint: {checkpoint_file}")

    metadata = resolve_model_metadata(
        checkpoint_path=str(checkpoint_file),
        device="cpu",
        in_channels=DEFAULT_IN_CHANNELS,
        num_classes=NUM_CLASSES,
        base_channels=BASE_CHANNELS,
        input_width=MODEL_INPUT_WIDTH,
        input_height=MODEL_INPUT_HEIGHT,
        amp=USE_AMP,
    )
    runtime_config = RuntimeModelConfig(
        checkpoint_path=str(checkpoint_file),
        in_channels=int(metadata.get("in_channels", DEFAULT_IN_CHANNELS)),
        num_classes=int(metadata.get("num_classes", NUM_CLASSES)),
        base_channels=int(metadata.get("base_channels", BASE_CHANNELS)),
        input_width=int(metadata.get("input_width", MODEL_INPUT_WIDTH)),
        input_height=int(metadata.get("input_height", MODEL_INPUT_HEIGHT)),
        use_amp=bool(metadata.get("amp", USE_AMP)),
    )

    if amp_override is not None:
        runtime_config.use_amp = amp_override

    return runtime_config


def print_runtime_config(runtime_config: RuntimeModelConfig, device: torch.device) -> None:
    print("checkpoint:", runtime_config.checkpoint_path)
    print(
        "model config:",
        {
            "in_channels": runtime_config.in_channels,
            "num_classes": runtime_config.num_classes,
            "base_channels": runtime_config.base_channels,
            "input_width": runtime_config.input_width,
            "input_height": runtime_config.input_height,
            "amp": runtime_config.use_amp and device.type == "cuda",
        },
    )


# =========================
# 主循环
# =========================
def main() -> None:
    args = parse_args()
    runtime_config = resolve_runtime_model_config(checkpoint_path=args.checkpoint_path, amp_override=args.amp)

    device = resolve_device(args.device)
    amp_enabled = runtime_config.use_amp and device.type == "cuda"
    print("device:", device)
    print_runtime_config(runtime_config, device)

    model = UNet(
        in_channels=runtime_config.in_channels,
        num_classes=runtime_config.num_classes,
        base_channels=runtime_config.base_channels,
    ).to(device)
    load_info = load_checkpoint_flexible(
        model=model,
        checkpoint_path=runtime_config.checkpoint_path,
        device=device,
        optimizer=None,
    )
    print("checkpoint 信息:", load_info)
    model.eval()

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头: {args.camera_index}")

    camera_width, camera_height, camera_fps = configure_camera(cap)
    print(f"camera resolution: {camera_width}x{camera_height}, fps={camera_fps:.1f}")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, handle_mouse)

    ema_filter = OnlineEMAFilter(alpha=EMA_ALPHA)
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("摄像头读取失败。")
            break

        if USE_MIRROR_VIEW:
            frame = cv2.flip(frame, 1)

        ROI_STATE.set_frame_shape(frame)

        selected_roi = ROI_STATE.active_roi
        geometry_result = build_empty_geometry_result()
        preprocess_meta: Optional[ResizeMeta] = None
        roi_gray_preview: Optional[np.ndarray] = None
        pred_label_map: Optional[np.ndarray] = None

        if selected_roi is not None and selected_roi.is_valid():
            roi_frame = selected_roi.crop(frame)
            if roi_frame.size > 0:
                roi_gray_preview, input_tensor, preprocess_meta = preprocess_frame(
                    roi_frame,
                    input_width=runtime_config.input_width,
                    input_height=runtime_config.input_height,
                )
                pred_label_map = predict_label_map(model, input_tensor, device, use_amp=amp_enabled)
                geometry_result = extract_eye_geometry_from_label_map(pred_label_map)

        smoothed_dx, smoothed_dy = ema_filter.update(
            geometry_result["norm_dx"],
            geometry_result["norm_dy"],
            geometry_result["valid"],
        )

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        vis = build_visualization(
            frame_bgr=frame,
            geometry_result=geometry_result,
            preprocess_meta=preprocess_meta,
            selected_roi=selected_roi,
            smoothed_dx=smoothed_dx,
            smoothed_dy=smoothed_dy,
            fps=fps,
            camera_resolution=(camera_width, camera_height),
        )

        if roi_gray_preview is not None:
            preview_y = vis.shape[0] - ROI_PREVIEW_HEIGHT - 10
            vis = draw_preview_panel(vis, roi_gray_preview, "ROI Gray", (10, preview_y), (ROI_PREVIEW_WIDTH, ROI_PREVIEW_HEIGHT), use_gray=True)

        if pred_label_map is not None:
            pred_preview = (pred_label_map * 85).astype(np.uint8)
            preview_y = vis.shape[0] - PRED_PREVIEW_HEIGHT - 10
            vis = draw_preview_panel(vis, pred_preview, "Segmentation", (20 + ROI_PREVIEW_WIDTH, preview_y), (PRED_PREVIEW_WIDTH, PRED_PREVIEW_HEIGHT), use_gray=True)

        cv2.imshow(WINDOW_NAME, vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            ema_filter.reset()
            print("已重置 EMA 状态。")
        elif key == ord("c"):
            ROI_STATE.clear()
            ema_filter.reset()
            print("已清除 ROI 并重置 EMA。")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
