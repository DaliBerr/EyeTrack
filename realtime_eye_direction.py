import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import cv2
import numpy as np
import torch
from torch import nn

from eyetrack.config import (
    DEFAULT_BASE_CHANNELS,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_IN_CHANNELS,
    DEFAULT_INPUT_HEIGHT,
    DEFAULT_INPUT_WIDTH,
    DEFAULT_NUM_CLASSES,
    DEFAULT_USE_AMP,
)
from eyetrack.data.preprocessing import ResizeMeta, preprocess_bgr_frame
from eyetrack.deployment.onnx_tools import build_onnx_session, require_onnxruntime
from eyetrack.gaze import (
    CalibrationSession,
    GazeFeatureResult,
    RegionGeometry,
    advance_calibration_session,
    begin_calibration_session,
    build_empty_gaze_feature_result,
    build_five_point_calibration_points,
    cancel_calibration_session,
    extract_gaze_features_from_label_map,
    predict_screen_point,
    resolve_tracking_features,
)
from eyetrack.models.unet import UNet
from eyetrack.realtime_gaze import (
    build_calibration_canvas,
    destroy_window,
    draw_screen_preview_panel,
    ensure_fullscreen_window,
    format_optional_pair,
)
from eyetrack.runtime import autocast_context, resolve_device
from eyetrack.training.checkpoints import load_checkpoint_flexible, resolve_model_metadata


# =========================
# 配置区
# =========================
CHECKPOINT_PATH = DEFAULT_CHECKPOINT_PATH
CAMERA_INDEX = 2
WINDOW_NAME = "Realtime Eye Direction Demo"
CALIBRATION_WINDOW_NAME = "Realtime Gaze Calibration"

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
KERNEL_SIZE = 3
IRIS_MIN_AREA = 100
PUPIL_MIN_AREA = 20

EMA_ALPHA = 0.35
MAX_VALID_NORM_RADIUS = 0.85
ARROW_LENGTH = 120
SHOW_DEBUG_TEXT = True
USE_MIRROR_VIEW = False
SCREEN_PREVIEW_SIZE = 220
STALE_SCREEN_POINT_TIMEOUT_MS = 300
DEFAULT_CALIBRATION_SETTLE_MS = 500
DEFAULT_CALIBRATION_CAPTURE_MS = 1000
DEFAULT_CALIBRATION_MIN_VALID_FRAMES = 15
DEFAULT_CALIBRATION_MARGIN = 0.1

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
    roi_revision: int = 0

    def set_frame_shape(self, frame: np.ndarray) -> None:
        self.frame_height, self.frame_width = frame.shape[:2]

    def set_active_roi(self, roi: ROIBox) -> None:
        if self.active_roi != roi:
            self.active_roi = roi
            self.roi_revision += 1

    def clear(self) -> None:
        had_roi = self.active_roi is not None
        self.active_roi = None
        self.drag_start = None
        self.drag_current = None
        if had_roi:
            self.roi_revision += 1

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
    model_path: str
    runtime_type: str
    in_channels: int
    num_classes: int
    base_channels: int
    input_width: int
    input_height: int
    use_amp: bool
    onnx_backend: str = "cpu"


@dataclass
class RuntimePredictor:
    runtime_type: str
    torch_model: Optional[nn.Module] = None
    onnx_session: Any = None
    onnx_input_name: Optional[str] = None


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
            ROI_STATE.set_active_roi(drag_roi)
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


def is_onnx_model_path(model_path: str) -> bool:
    return Path(model_path).suffix.lower() == ".onnx"


def resolve_onnx_runtime_config(model_path: str, onnx_backend: str) -> RuntimeModelConfig:
    ort = require_onnxruntime()
    session = build_onnx_session(model_path=model_path, backend=onnx_backend, enable_profiling=False)
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0] if len(session.get_outputs()) > 0 else None
    custom_metadata = session.get_modelmeta().custom_metadata_map

    input_shape = list(input_meta.shape)
    output_shape = list(output_meta.shape) if output_meta is not None else []

    def _shape_dim(shape: list[Any], index: int, fallback: int) -> int:
        if index >= len(shape):
            return fallback
        value = shape[index]
        return int(value) if isinstance(value, int) else fallback

    del ort
    return RuntimeModelConfig(
        model_path=str(model_path),
        runtime_type="onnx",
        in_channels=int(custom_metadata.get("in_channels", _shape_dim(input_shape, 1, DEFAULT_IN_CHANNELS))),
        num_classes=int(custom_metadata.get("num_classes", _shape_dim(output_shape, 1, NUM_CLASSES))),
        base_channels=int(custom_metadata.get("base_channels", BASE_CHANNELS)),
        input_width=int(custom_metadata.get("input_width", _shape_dim(input_shape, 3, MODEL_INPUT_WIDTH))),
        input_height=int(custom_metadata.get("input_height", _shape_dim(input_shape, 2, MODEL_INPUT_HEIGHT))),
        use_amp=False,
        onnx_backend=onnx_backend,
    )


@torch.no_grad()
def predict_label_map(
    predictor: RuntimePredictor,
    input_tensor: torch.Tensor,
    device: torch.device,
    use_amp: bool,
) -> np.ndarray:
    if predictor.runtime_type == "pytorch":
        if predictor.torch_model is None:
            raise RuntimeError("PyTorch predictor 未正确初始化。")
        with autocast_context(device=device, use_amp=use_amp):
            logits = predictor.torch_model(input_tensor.to(device))
        return torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)

    if predictor.runtime_type == "onnx":
        if predictor.onnx_session is None or predictor.onnx_input_name is None:
            raise RuntimeError("ONNX predictor 未正确初始化。")
        logits = predictor.onnx_session.run(
            None,
            {predictor.onnx_input_name: input_tensor.cpu().numpy().astype(np.float32)},
        )[0]
        return np.argmax(logits, axis=1)[0].astype(np.uint8)

    raise ValueError(f"不支持的 predictor runtime_type: {predictor.runtime_type}")


# =========================
# 分割与几何
# =========================
def build_empty_geometry_result() -> GazeFeatureResult:
    return build_empty_gaze_feature_result()


def extract_eye_geometry_from_label_map(pred_label_map: np.ndarray) -> GazeFeatureResult:
    return extract_gaze_features_from_label_map(
        pred_label_map=pred_label_map,
        valid_mask=None,
        iris_class_id=IRIS_CLASS_ID,
        pupil_class_id=PUPIL_CLASS_ID,
        kernel_size=KERNEL_SIZE,
        iris_min_area=IRIS_MIN_AREA,
        pupil_min_area=PUPIL_MIN_AREA,
        max_valid_norm_radius=MAX_VALID_NORM_RADIUS,
    )


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


def get_calibration_state(session: Optional[CalibrationSession]) -> Tuple[str, str, bool]:
    if session is None:
        return "idle", "none", False
    return session.state, session.calibration_step, session.is_calibrated


def put_debug_text(
    frame: np.ndarray,
    geometry_result: GazeFeatureResult,
    feature_mode: str,
    selected_feature_x: Optional[float],
    selected_feature_y: Optional[float],
    selected_feature_valid: bool,
    selected_feature_reasons: tuple[str, ...],
    preprocess_meta: Optional[ResizeMeta],
    smoothed_dx: Optional[float],
    smoothed_dy: Optional[float],
    fps: float,
    camera_resolution: Tuple[int, int],
    selected_roi: Optional[ROIBox],
    calibration_session: Optional[CalibrationSession],
    screen_uv: Optional[Tuple[float, float]],
    tracking_valid: bool,
    status_message: str,
) -> np.ndarray:
    output = frame.copy()
    calibration_state, calibration_step, calibrated = get_calibration_state(calibration_session)
    invalid_reason_text = "none" if len(geometry_result.invalid_reasons) == 0 else "|".join(geometry_result.invalid_reasons)
    feature_reason_text = "none" if len(selected_feature_reasons) == 0 else "|".join(selected_feature_reasons)

    lines = [
        f"FPS: {fps:.1f}",
        f"camera: {camera_resolution[0]}x{camera_resolution[1]}",
        "ROI: drag left mouse | c clear | r reset | s calibrate | x cancel | q quit",
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
        f"valid: {geometry_result.valid}",
        f"iris_area: {geometry_result.iris_geometry.area}",
        f"pupil_area: {geometry_result.pupil_geometry.area}",
        f"norm_dx: {geometry_result.norm_dx}",
        f"norm_dy: {geometry_result.norm_dy}",
        f"norm_r : {geometry_result.norm_radius}",
        f"iris_fx: {geometry_result.iris_feature_x}",
        f"iris_fy: {geometry_result.iris_feature_y}",
        f"feature_mode: {feature_mode}",
        f"feature_x: {selected_feature_x}",
        f"feature_y: {selected_feature_y}",
        f"feature_valid: {selected_feature_valid}",
        f"feature_invalid: {feature_reason_text}",
        f"sm_dx  : {smoothed_dx}",
        f"sm_dy  : {smoothed_dy}",
        f"invalid: {invalid_reason_text}",
        f"calibration_state: {calibration_state}",
        f"calibration_step : {calibration_step}",
        f"calibrated: {calibrated}",
        f"screen_uv: {format_optional_pair(screen_uv)}",
        f"tracking_valid: {tracking_valid}",
        f"message: {status_message}",
    ])

    if calibration_session is not None and calibration_session.is_active:
        lines.append(f"capture_samples: {len(calibration_session.current_point_samples)}/{calibration_session.min_valid_frames}")

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
    geometry_result: GazeFeatureResult,
    feature_mode: str,
    selected_feature_x: Optional[float],
    selected_feature_y: Optional[float],
    selected_feature_valid: bool,
    selected_feature_reasons: tuple[str, ...],
    preprocess_meta: Optional[ResizeMeta],
    selected_roi: Optional[ROIBox],
    smoothed_dx: Optional[float],
    smoothed_dy: Optional[float],
    fps: float,
    camera_resolution: Tuple[int, int],
    calibration_session: Optional[CalibrationSession],
    calibration_points: list,
    screen_uv: Optional[Tuple[float, float]],
    tracking_valid: bool,
    status_message: str,
) -> np.ndarray:
    vis = frame_bgr.copy()

    vis = draw_ellipse_on_frame(vis, geometry_result.iris_geometry.ellipse, preprocess_meta, selected_roi, (0, 255, 0), 2)
    vis = draw_ellipse_on_frame(vis, geometry_result.pupil_geometry.ellipse, preprocess_meta, selected_roi, (0, 0, 255), 2)

    vis = draw_center_on_frame(vis, geometry_result.iris_geometry.center_x, geometry_result.iris_geometry.center_y, preprocess_meta, selected_roi, (0, 255, 0), 4)
    vis = draw_center_on_frame(vis, geometry_result.pupil_geometry.center_x, geometry_result.pupil_geometry.center_y, preprocess_meta, selected_roi, (0, 0, 255), 4)

    vis = draw_direction_arrow(vis, geometry_result.iris_geometry, preprocess_meta, selected_roi, smoothed_dx, smoothed_dy)
    vis = draw_corner_indicator(vis, smoothed_dx, smoothed_dy)
    vis = draw_roi_overlay(vis, selected_roi)

    active_point_name = None
    calibrated = False
    if calibration_session is not None:
        active_point = calibration_session.current_point if calibration_session.is_active else None
        active_point_name = active_point.name if active_point is not None else None
        calibrated = calibration_session.is_calibrated

    vis = draw_screen_preview_panel(
        vis,
        calibration_points=calibration_points,
        screen_uv=screen_uv,
        tracking_valid=tracking_valid,
        calibrated=calibrated,
        active_point_name=active_point_name,
        panel_size=SCREEN_PREVIEW_SIZE,
    )

    if SHOW_DEBUG_TEXT:
        vis = put_debug_text(
            vis,
            geometry_result,
            feature_mode,
            selected_feature_x,
            selected_feature_y,
            selected_feature_valid,
            selected_feature_reasons,
            preprocess_meta,
            smoothed_dx,
            smoothed_dy,
            fps,
            camera_resolution,
            selected_roi,
            calibration_session,
            screen_uv,
            tracking_valid,
            status_message,
        )

    return vis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实时眼动方向验证")
    parser.add_argument("--model_path", type=str, default=None, help="待加载的模型路径，支持 .pth 与 .onnx")
    parser.add_argument("--checkpoint_path", type=str, default=CHECKPOINT_PATH, help="兼容旧参数名；若未传 --model_path，则使用该路径")
    parser.add_argument("--camera_index", type=int, default=CAMERA_INDEX, help="摄像头索引")
    parser.add_argument("--device", type=str, default="auto", help="运行设备: auto/cpu/cuda")
    parser.add_argument("--onnx_backend", type=str, default="cpu", choices=["cpu", "nnapi"], help="ONNX Runtime backend，仅在 .onnx 模型时生效")
    parser.add_argument("--feature_mode", type=str, choices=["pupil_iris", "iris_only"], default="pupil_iris", help="实时校准与跟踪使用的特征模式")
    parser.add_argument("--calibration_settle_ms", type=int, default=DEFAULT_CALIBRATION_SETTLE_MS, help="校准点切换后的稳定等待时长")
    parser.add_argument("--calibration_capture_ms", type=int, default=DEFAULT_CALIBRATION_CAPTURE_MS, help="每个校准点的采样时长")
    parser.add_argument("--calibration_min_valid_frames", type=int, default=DEFAULT_CALIBRATION_MIN_VALID_FRAMES, help="每个校准点要求的最少有效帧数")
    parser.add_argument("--calibration_margin", type=float, default=DEFAULT_CALIBRATION_MARGIN, help="四角校准点距边缘的归一化留白")

    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true", help="强制启用 CUDA AMP")
    amp_group.add_argument("--no-amp", dest="amp", action="store_false", help="强制禁用 AMP")
    parser.set_defaults(amp=None)

    return parser.parse_args()


def resolve_runtime_model_config(model_path: str, amp_override: Optional[bool], onnx_backend: str) -> RuntimeModelConfig:
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"未找到模型文件: {model_file}")

    if is_onnx_model_path(str(model_file)):
        runtime_config = resolve_onnx_runtime_config(model_path=str(model_file), onnx_backend=onnx_backend)
        return runtime_config

    metadata = resolve_model_metadata(
        checkpoint_path=str(model_file),
        device="cpu",
        in_channels=DEFAULT_IN_CHANNELS,
        num_classes=NUM_CLASSES,
        base_channels=BASE_CHANNELS,
        input_width=MODEL_INPUT_WIDTH,
        input_height=MODEL_INPUT_HEIGHT,
        amp=USE_AMP,
    )
    runtime_config = RuntimeModelConfig(
        model_path=str(model_file),
        runtime_type="pytorch",
        in_channels=int(metadata.get("in_channels", DEFAULT_IN_CHANNELS)),
        num_classes=int(metadata.get("num_classes", NUM_CLASSES)),
        base_channels=int(metadata.get("base_channels", BASE_CHANNELS)),
        input_width=int(metadata.get("input_width", MODEL_INPUT_WIDTH)),
        input_height=int(metadata.get("input_height", MODEL_INPUT_HEIGHT)),
        use_amp=bool(metadata.get("amp", USE_AMP)),
        onnx_backend=onnx_backend,
    )

    if amp_override is not None:
        runtime_config.use_amp = amp_override

    return runtime_config


def build_runtime_predictor(runtime_config: RuntimeModelConfig, device: torch.device) -> RuntimePredictor:
    if runtime_config.runtime_type == "pytorch":
        model = UNet(
            in_channels=runtime_config.in_channels,
            num_classes=runtime_config.num_classes,
            base_channels=runtime_config.base_channels,
        ).to(device)
        load_info = load_checkpoint_flexible(
            model=model,
            checkpoint_path=runtime_config.model_path,
            device=device,
            optimizer=None,
        )
        print("checkpoint 信息:", load_info)
        model.eval()
        return RuntimePredictor(runtime_type="pytorch", torch_model=model)

    if runtime_config.runtime_type == "onnx":
        session = build_onnx_session(
            model_path=runtime_config.model_path,
            backend=runtime_config.onnx_backend,
            enable_profiling=False,
        )
        print("onnx providers:", session.get_providers())
        return RuntimePredictor(
            runtime_type="onnx",
            onnx_session=session,
            onnx_input_name=session.get_inputs()[0].name,
        )

    raise ValueError(f"不支持的 runtime_type: {runtime_config.runtime_type}")


def print_runtime_config(runtime_config: RuntimeModelConfig, device: torch.device) -> None:
    print("model_path:", runtime_config.model_path)
    print(
        "model config:",
        {
            "runtime_type": runtime_config.runtime_type,
            "in_channels": runtime_config.in_channels,
            "num_classes": runtime_config.num_classes,
            "base_channels": runtime_config.base_channels,
            "input_width": runtime_config.input_width,
            "input_height": runtime_config.input_height,
            "amp": runtime_config.use_amp and runtime_config.runtime_type == "pytorch" and device.type == "cuda",
            "onnx_backend": runtime_config.onnx_backend if runtime_config.runtime_type == "onnx" else None,
        },
    )


# =========================
# 主循环
# =========================
def main() -> None:
    args = parse_args()
    model_path = args.model_path if args.model_path is not None else args.checkpoint_path
    runtime_config = resolve_runtime_model_config(model_path=model_path, amp_override=args.amp, onnx_backend=args.onnx_backend)

    device = resolve_device(args.device)
    amp_enabled = runtime_config.runtime_type == "pytorch" and runtime_config.use_amp and device.type == "cuda"
    print("device:", device)
    print_runtime_config(runtime_config, device)

    predictor = build_runtime_predictor(runtime_config=runtime_config, device=device)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头: {args.camera_index}")

    camera_width, camera_height, camera_fps = configure_camera(cap)
    print(f"camera resolution: {camera_width}x{camera_height}, fps={camera_fps:.1f}")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, handle_mouse)

    ema_filter = OnlineEMAFilter(alpha=EMA_ALPHA)
    calibration_session: Optional[CalibrationSession] = None
    calibration_window_open = False
    calibration_points = build_five_point_calibration_points(args.calibration_margin)
    displayed_screen_uv: Optional[Tuple[float, float]] = None
    last_valid_screen_ts_ms: Optional[float] = None
    tracking_valid = False
    status_message = "请选择 ROI，然后按 s 开始五点校准。"
    last_roi_revision = ROI_STATE.roi_revision
    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("摄像头读取失败。")
                break

            if USE_MIRROR_VIEW:
                frame = cv2.flip(frame, 1)

            ROI_STATE.set_frame_shape(frame)

            if ROI_STATE.roi_revision != last_roi_revision:
                last_roi_revision = ROI_STATE.roi_revision
                ema_filter.reset()
                calibration_session = None
                calibration_window_open = False
                destroy_window(CALIBRATION_WINDOW_NAME)
                displayed_screen_uv = None
                last_valid_screen_ts_ms = None
                tracking_valid = False
                status_message = "ROI 已更新，校准已失效。"

            selected_roi = ROI_STATE.active_roi
            geometry_result = build_empty_geometry_result()
            selected_feature_x: Optional[float] = None
            selected_feature_y: Optional[float] = None
            selected_feature_valid = False
            selected_feature_reasons: tuple[str, ...] = tuple()
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
                    pred_label_map = predict_label_map(predictor, input_tensor, device, use_amp=amp_enabled)
                    geometry_result = extract_eye_geometry_from_label_map(pred_label_map)

            selected_feature_x, selected_feature_y, selected_feature_valid, selected_feature_reasons = resolve_tracking_features(
                geometry_result,
                feature_mode=args.feature_mode,
            )

            smoothed_dx, smoothed_dy = ema_filter.update(
                selected_feature_x,
                selected_feature_y,
                selected_feature_valid,
            )

            now_ms = time.time() * 1000.0

            if calibration_session is not None and calibration_session.is_active:
                previous_state = calibration_session.state
                calibration_session = advance_calibration_session(
                    calibration_session,
                    now_ms=now_ms,
                    feature_dx=smoothed_dx,
                    feature_dy=smoothed_dy,
                    feature_valid=bool(selected_feature_valid and smoothed_dx is not None and smoothed_dy is not None),
                )

                if calibration_session.state != previous_state:
                    if calibration_session.state == "capturing":
                        status_message = f"开始采样 {calibration_session.calibration_step}"
                    elif calibration_session.state == "settling":
                        status_message = f"请注视 {calibration_session.calibration_step}"
                    elif calibration_session.state == "completed":
                        calibration_points = calibration_session.points
                        calibration_window_open = False
                        destroy_window(CALIBRATION_WINDOW_NAME)
                        status_message = "五点校准完成。"
                        print(status_message)
                    elif calibration_session.state == "failed":
                        calibration_window_open = False
                        destroy_window(CALIBRATION_WINDOW_NAME)
                        status_message = calibration_session.failure_reason or "校准失败。"
                        print(status_message)

            if calibration_session is not None and calibration_session.is_active:
                if not calibration_window_open:
                    ensure_fullscreen_window(CALIBRATION_WINDOW_NAME)
                    calibration_window_open = True
                calibration_canvas = build_calibration_canvas((camera_width, camera_height), calibration_session)
                cv2.imshow(CALIBRATION_WINDOW_NAME, calibration_canvas)
            elif calibration_window_open:
                destroy_window(CALIBRATION_WINDOW_NAME)
                calibration_window_open = False

            calibration_model = calibration_session.model if calibration_session is not None and calibration_session.is_calibrated else None
            tracking_valid = False
            if calibration_model is not None and selected_feature_valid and smoothed_dx is not None and smoothed_dy is not None:
                predicted_screen_uv = predict_screen_point(calibration_model, smoothed_dx, smoothed_dy)
                if predicted_screen_uv is not None:
                    displayed_screen_uv = predicted_screen_uv
                    last_valid_screen_ts_ms = now_ms
                    tracking_valid = True
            elif calibration_model is None:
                displayed_screen_uv = None
                last_valid_screen_ts_ms = None
            elif (
                displayed_screen_uv is None
                or last_valid_screen_ts_ms is None
                or (now_ms - last_valid_screen_ts_ms) > STALE_SCREEN_POINT_TIMEOUT_MS
            ):
                displayed_screen_uv = None
                last_valid_screen_ts_ms = None

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            vis = build_visualization(
                frame_bgr=frame,
                geometry_result=geometry_result,
                feature_mode=args.feature_mode,
                selected_feature_x=selected_feature_x,
                selected_feature_y=selected_feature_y,
                selected_feature_valid=selected_feature_valid,
                selected_feature_reasons=selected_feature_reasons,
                preprocess_meta=preprocess_meta,
                selected_roi=selected_roi,
                smoothed_dx=smoothed_dx,
                smoothed_dy=smoothed_dy,
                fps=fps,
                camera_resolution=(camera_width, camera_height),
                calibration_session=calibration_session,
                calibration_points=calibration_points,
                screen_uv=displayed_screen_uv,
                tracking_valid=tracking_valid,
                status_message=status_message,
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
                calibration_session = None
                calibration_window_open = False
                destroy_window(CALIBRATION_WINDOW_NAME)
                displayed_screen_uv = None
                last_valid_screen_ts_ms = None
                tracking_valid = False
                status_message = "已重置 EMA 与校准状态。"
                print(status_message)
            elif key == ord("c"):
                ROI_STATE.clear()
                ema_filter.reset()
                calibration_session = None
                calibration_window_open = False
                destroy_window(CALIBRATION_WINDOW_NAME)
                displayed_screen_uv = None
                last_valid_screen_ts_ms = None
                tracking_valid = False
                status_message = "已清除 ROI，校准已失效。"
                print(status_message)
            elif key == ord("s"):
                if selected_roi is None or not selected_roi.is_valid():
                    status_message = "请先框选有效 ROI，再开始校准。"
                    print(status_message)
                else:
                    ema_filter.reset()
                    calibration_session = begin_calibration_session(
                        now_ms=now_ms,
                        settle_ms=args.calibration_settle_ms,
                        capture_ms=args.calibration_capture_ms,
                        min_valid_frames=args.calibration_min_valid_frames,
                        margin=args.calibration_margin,
                    )
                    calibration_points = calibration_session.points
                    displayed_screen_uv = None
                    last_valid_screen_ts_ms = None
                    tracking_valid = False
                    ensure_fullscreen_window(CALIBRATION_WINDOW_NAME)
                    calibration_window_open = True
                    status_message = f"开始校准({args.feature_mode}): {calibration_session.calibration_step}"
                    print(status_message)
            elif key == ord("x"):
                if calibration_session is not None and calibration_session.is_active:
                    calibration_session = cancel_calibration_session(calibration_session, "已取消当前校准。")
                    calibration_window_open = False
                    destroy_window(CALIBRATION_WINDOW_NAME)
                    displayed_screen_uv = None
                    last_valid_screen_ts_ms = None
                    tracking_valid = False
                    status_message = calibration_session.failure_reason or "已取消当前校准。"
                    print(status_message)
    finally:
        cap.release()
        destroy_window(CALIBRATION_WINDOW_NAME)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
