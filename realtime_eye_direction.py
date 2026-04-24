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
    build_nine_point_calibration_points,
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
OUTER_BOUNDARY_CLASS_ID = 1
KERNEL_SIZE = 3
IRIS_MIN_AREA = 100
PUPIL_MIN_AREA = 20

MAX_VALID_NORM_RADIUS = 0.85
QUALITY_TRACKER_ALPHA = 0.18
MIN_TRACKING_CONFIDENCE = 0.22
MIN_CALIBRATION_CONFIDENCE = 0.30
KALMAN_PROCESS_NOISE = 0.002
KALMAN_BASE_MEASUREMENT_NOISE = 0.008
KALMAN_LOW_CONFIDENCE_NOISE = 0.06
KALMAN_JUMP_NOISE_SCALE = 0.10
KALMAN_MAX_COVARIANCE = 1.5
ARROW_LENGTH = 120
SHOW_DEBUG_TEXT = True
USE_MIRROR_VIEW = False
SCREEN_PREVIEW_SIZE = 220
STALE_SCREEN_POINT_TIMEOUT_MS = 300
DEFAULT_CALIBRATION_SETTLE_MS = 800
DEFAULT_CALIBRATION_CAPTURE_MS = 1500
DEFAULT_CALIBRATION_MIN_VALID_FRAMES = 15
DEFAULT_CALIBRATION_MARGIN = 0.1

ROI_PREVIEW_WIDTH = 220
ROI_PREVIEW_HEIGHT = 140
PRED_PREVIEW_WIDTH = 220
PRED_PREVIEW_HEIGHT = 140
DEFAULT_DEMO_RECORD_KEY = "v"
DEMO_RENDER_STATE_IDLE = "idle_calibration"
DEMO_RENDER_STATE_READY = "ready_to_render"
DEMO_RENDER_STATE_RENDERING = "rendering"
DEMO_RENDER_STATE_FINISHED = "finished"
DEMO_VIDEO_CODEC = "mp4v"
DEMO_OUTPUT_SUFFIX = "_gaze_demo.mp4"
DEMO_CALIBRATION_OUTPUT_STEM_SUFFIX = "_calibration"
DEMO_PANEL_MARGIN = 10
DEMO_PANEL_GAP = 10
DEMO_MIN_PANEL_WIDTH = 96
DEMO_MIN_PANEL_HEIGHT = 60
DEMO_PANEL_ANCHOR_BOTTOM_LEFT = "bottom_left"
DEMO_PANEL_ANCHOR_CENTER_LEFT = "center_left"
DEMO_CENTER_LEFT_PANEL_ANCHOR_U = 0.30
DEMO_CALIBRATION_RECORD_FPS_FALLBACK = 30.0
DEMO_GAZE_RING_RADIUS = 18
DEMO_GAZE_RING_THICKNESS = 4
DEMO_GAZE_OUTER_RING_RADIUS = 28
DEMO_GAZE_OUTER_RING_THICKNESS = 2
DEMO_GAZE_CENTER_RADIUS = 4
DEMO_GAZE_SHADOW_OFFSET = 1
DEMO_GAZE_SHADOW_COLOR = (0, 0, 0)
DEMO_GAZE_VALID_COLOR = (0, 255, 0)
DEMO_GAZE_INVALID_COLOR = (0, 165, 255)


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


class FeatureQualityTracker:
    def __init__(self, alpha: float = QUALITY_TRACKER_ALPHA):
        self.alpha = float(np.clip(alpha, 0.01, 0.99))
        self.area_ema: Optional[float] = None
        self.feature_x_ema: Optional[float] = None
        self.feature_y_ema: Optional[float] = None

    def reset(self) -> None:
        self.area_ema = None
        self.feature_x_ema = None
        self.feature_y_ema = None

    def compute_confidence(
        self,
        geometry_result: GazeFeatureResult,
        feature_mode: str,
        feature_x: Optional[float],
        feature_y: Optional[float],
        feature_valid: bool,
        feature_reasons: tuple[str, ...],
    ) -> float:
        if not feature_valid or feature_x is None or feature_y is None:
            return 0.0

        iris_area = float(max(geometry_result.iris_geometry.area, 0))
        area_reference = max(float(IRIS_MIN_AREA) * 2.0, 1.0)
        area_term = float(np.clip(iris_area / area_reference, 0.0, 1.0))

        if self.area_ema is None:
            area_stability = 1.0
        else:
            area_delta = abs(iris_area - self.area_ema) / max(self.area_ema, 1.0)
            area_stability = float(np.exp(-2.5 * area_delta))

        if self.feature_x_ema is None or self.feature_y_ema is None:
            motion_stability = 1.0
        else:
            motion_delta = float(np.hypot(feature_x - self.feature_x_ema, feature_y - self.feature_y_ema))
            motion_stability = float(np.exp(-6.0 * motion_delta))

        shape_term = 0.35
        if geometry_result.iris_geometry.ellipse is not None:
            major = max(float(geometry_result.iris_geometry.ellipse.major_axis), 1e-6)
            minor = float(geometry_result.iris_geometry.ellipse.minor_axis)
            axis_ratio = minor / major
            shape_term = float(np.clip((axis_ratio - 0.12) / 0.55, 0.0, 1.0))

        normalized_mode = feature_mode.strip().lower()
        mode_factor = 1.0
        if normalized_mode == "iris_only" and not geometry_result.iris_outer_valid:
            mode_factor = 0.65

        reason_penalty = 1.0 / (1.0 + 0.3 * len(feature_reasons))
        confidence = (0.20 + 0.25 * area_term + 0.25 * area_stability + 0.20 * motion_stability + 0.10 * shape_term)
        confidence = float(np.clip(confidence * mode_factor * reason_penalty, 0.0, 1.0))

        if self.area_ema is None:
            self.area_ema = iris_area
        else:
            self.area_ema = self.alpha * iris_area + (1.0 - self.alpha) * self.area_ema

        if self.feature_x_ema is None or self.feature_y_ema is None:
            self.feature_x_ema = float(feature_x)
            self.feature_y_ema = float(feature_y)
        else:
            self.feature_x_ema = self.alpha * float(feature_x) + (1.0 - self.alpha) * self.feature_x_ema
            self.feature_y_ema = self.alpha * float(feature_y) + (1.0 - self.alpha) * self.feature_y_ema

        return confidence


class AdaptiveKalmanFilter2D:
    def __init__(
        self,
        process_noise: float = KALMAN_PROCESS_NOISE,
        base_measurement_noise: float = KALMAN_BASE_MEASUREMENT_NOISE,
        low_confidence_noise: float = KALMAN_LOW_CONFIDENCE_NOISE,
        jump_noise_scale: float = KALMAN_JUMP_NOISE_SCALE,
        max_covariance: float = KALMAN_MAX_COVARIANCE,
    ):
        self.process_noise = float(max(process_noise, 1e-7))
        self.base_measurement_noise = float(max(base_measurement_noise, 1e-7))
        self.low_confidence_noise = float(max(low_confidence_noise, 0.0))
        self.jump_noise_scale = float(max(jump_noise_scale, 0.0))
        self.max_covariance = float(max(max_covariance, 1e-5))

        self.state_x: Optional[float] = None
        self.state_y: Optional[float] = None
        self.var_x: float = self.max_covariance
        self.var_y: float = self.max_covariance
        self.last_measurement: Optional[Tuple[float, float]] = None

    def reset(self) -> None:
        self.state_x = None
        self.state_y = None
        self.var_x = self.max_covariance
        self.var_y = self.max_covariance
        self.last_measurement = None

    def _predict(self) -> None:
        if self.state_x is None or self.state_y is None:
            return
        self.var_x = min(self.var_x + self.process_noise, self.max_covariance)
        self.var_y = min(self.var_y + self.process_noise, self.max_covariance)

    @staticmethod
    def _update_axis(state: float, variance: float, measurement: float, measurement_noise: float) -> Tuple[float, float]:
        gain = variance / (variance + measurement_noise)
        next_state = state + gain * (measurement - state)
        next_variance = (1.0 - gain) * variance
        return float(next_state), float(next_variance)

    def update(self, dx: Optional[float], dy: Optional[float], valid: bool, confidence: float) -> Tuple[Optional[float], Optional[float]]:
        self._predict()

        if not valid or dx is None or dy is None:
            return self.state_x, self.state_y

        measurement_x = float(dx)
        measurement_y = float(dy)

        if self.state_x is None or self.state_y is None:
            self.state_x = measurement_x
            self.state_y = measurement_y
            self.var_x = self.base_measurement_noise
            self.var_y = self.base_measurement_noise
            self.last_measurement = (measurement_x, measurement_y)
            return self.state_x, self.state_y

        confidence = float(np.clip(confidence, 0.0, 1.0))
        jump = 0.0
        if self.last_measurement is not None:
            jump = float(np.hypot(measurement_x - self.last_measurement[0], measurement_y - self.last_measurement[1]))

        measurement_noise = self.base_measurement_noise
        measurement_noise += (1.0 - confidence) * self.low_confidence_noise
        measurement_noise += jump * self.jump_noise_scale
        measurement_noise = float(np.clip(measurement_noise, 1e-6, 5.0))

        self.state_x, self.var_x = self._update_axis(self.state_x, self.var_x, measurement_x, measurement_noise)
        self.state_y, self.var_y = self._update_axis(self.state_y, self.var_y, measurement_y, measurement_noise)
        self.last_measurement = (measurement_x, measurement_y)

        return self.state_x, self.state_y


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


@dataclass(frozen=True)
class DemoVideoSourceInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int


@dataclass(frozen=True)
class DemoGazeSample:
    timestamp_s: float
    screen_uv: Optional[Tuple[float, float]]
    tracking_valid: bool


@dataclass(frozen=True)
class PreviewPanelSpec:
    image: np.ndarray
    label: str
    desired_size: Tuple[int, int]
    use_gray: bool


@dataclass(frozen=True)
class PreviewPanelPlacement:
    x: int
    y: int
    width: int
    height: int


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
        outer_boundary_class_id=OUTER_BOUNDARY_CLASS_ID,
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
    selected_feature_confidence: float,
    selected_feature_reasons: tuple[str, ...],
    preprocess_meta: Optional[ResizeMeta],
    smoothed_dx: Optional[float],
    smoothed_dy: Optional[float],
    fps: float,
    camera_resolution: Tuple[int, int],
    selected_roi: Optional[ROIBox],
    calibration_session: Optional[CalibrationSession],
    static_boundary_ellipse: Optional[EllipseResult],
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
        f"feature_conf: {selected_feature_confidence:.3f}",
        f"feature_invalid: {feature_reason_text}",
        f"conf_gate: track>={MIN_TRACKING_CONFIDENCE:.2f} cal>={MIN_CALIBRATION_CONFIDENCE:.2f}",
        f"sm_dx  : {smoothed_dx}",
        f"sm_dy  : {smoothed_dy}",
        f"invalid: {invalid_reason_text}",
        f"calibration_state: {calibration_state}",
        f"calibration_step : {calibration_step}",
        f"calibrated: {calibrated}",
        f"static_boundary: {static_boundary_ellipse is not None}",
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
    selected_feature_confidence: float,
    selected_feature_reasons: tuple[str, ...],
    preprocess_meta: Optional[ResizeMeta],
    selected_roi: Optional[ROIBox],
    smoothed_dx: Optional[float],
    smoothed_dy: Optional[float],
    fps: float,
    camera_resolution: Tuple[int, int],
    calibration_session: Optional[CalibrationSession],
    static_boundary_ellipse: Optional[EllipseResult],
    calibration_points: list,
    screen_uv: Optional[Tuple[float, float]],
    tracking_valid: bool,
    status_message: str,
) -> np.ndarray:
    vis = frame_bgr.copy()

    vis = draw_ellipse_on_frame(vis, geometry_result.iris_geometry.ellipse, preprocess_meta, selected_roi, (0, 255, 0), 2)
    vis = draw_ellipse_on_frame(vis, geometry_result.pupil_geometry.ellipse, preprocess_meta, selected_roi, (0, 0, 255), 2)
    vis = draw_ellipse_on_frame(vis, static_boundary_ellipse, preprocess_meta, selected_roi, (255, 128, 0), 1)

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
            selected_feature_confidence,
            selected_feature_reasons,
            preprocess_meta,
            smoothed_dx,
            smoothed_dy,
            fps,
            camera_resolution,
            selected_roi,
            calibration_session,
            static_boundary_ellipse,
            screen_uv,
            tracking_valid,
            status_message,
        )

    return vis


def resolve_demo_output_path(demo_fpv_video: str, demo_output: Optional[str]) -> str:
    if demo_output is not None and str(demo_output).strip() != "":
        return str(Path(demo_output).expanduser().resolve())

    source_path = Path(demo_fpv_video).expanduser().resolve()
    return str(source_path.with_name(f"{source_path.stem}{DEMO_OUTPUT_SUFFIX}"))


def append_output_stem_suffix(output_path: str, stem_suffix: str) -> str:
    resolved_output_path = Path(output_path).expanduser().resolve()
    return str(
        resolved_output_path.with_name(
            f"{resolved_output_path.stem}{stem_suffix}{resolved_output_path.suffix or '.mp4'}"
        )
    )


def resolve_demo_calibration_output_path(demo_output_path: str) -> str:
    return append_output_stem_suffix(demo_output_path, DEMO_CALIBRATION_OUTPUT_STEM_SUFFIX)


def inspect_demo_video_source(video_path: str) -> tuple[cv2.VideoCapture, DemoVideoSourceInfo]:
    source_path = Path(video_path).expanduser().resolve()
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开演示 FPV 视频: {source_path}")

    ret, frame = cap.read()
    if not ret or frame is None or frame.size == 0:
        cap.release()
        raise RuntimeError(f"演示 FPV 视频无法读取首帧: {source_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        height, width = frame.shape[:2]

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(max(cap.get(cv2.CAP_PROP_FRAME_COUNT), 0.0))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"演示 FPV 视频尺寸无效: {source_path}")
    if fps <= 0.0:
        cap.release()
        raise RuntimeError(f"演示 FPV 视频 FPS 无效: {source_path}")

    return cap, DemoVideoSourceInfo(
        path=str(source_path),
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
    )


def resolve_demo_render_state(
    current_state: str,
    demo_source_info: Optional[DemoVideoSourceInfo],
    selected_roi: Optional[ROIBox],
    calibration_session: Optional[CalibrationSession],
    rendering_active: bool,
) -> str:
    if rendering_active:
        return DEMO_RENDER_STATE_RENDERING

    ready = bool(
        demo_source_info is not None
        and selected_roi is not None
        and selected_roi.is_valid()
        and calibration_session is not None
        and calibration_session.is_calibrated
    )
    if not ready:
        return DEMO_RENDER_STATE_IDLE
    if current_state == DEMO_RENDER_STATE_FINISHED:
        return DEMO_RENDER_STATE_FINISHED
    return DEMO_RENDER_STATE_READY


def validate_demo_render_start_request(
    *,
    demo_source_info: Optional[DemoVideoSourceInfo],
    selected_roi: Optional[ROIBox],
    calibration_session: Optional[CalibrationSession],
    rendering_active: bool,
) -> Optional[str]:
    if rendering_active:
        return "Demo rendering is already active."
    if demo_source_info is None:
        return "Cannot start demo rendering because the FPV source video is unavailable."
    if selected_roi is None or not selected_roi.is_valid():
        return "Please select a valid ROI before starting demo rendering."
    if calibration_session is None or not calibration_session.is_calibrated:
        return "Demo rendering requires a completed calibration."
    return None


def create_mp4_video_writer(output_path: str, width: int, height: int, fps: float) -> cv2.VideoWriter:
    resolved_output_path = Path(output_path).expanduser().resolve()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(resolved_output_path),
        cv2.VideoWriter_fourcc(*DEMO_VIDEO_CODEC),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(
            f"无法创建演示视频输出: {resolved_output_path} "
            f"({int(width)}x{int(height)}@{float(fps):.2f}fps, codec={DEMO_VIDEO_CODEC})"
        )
    return writer


def create_demo_video_writer(source_info: DemoVideoSourceInfo, output_path: str) -> cv2.VideoWriter:
    return create_mp4_video_writer(
        output_path=output_path,
        width=source_info.width,
        height=source_info.height,
        fps=source_info.fps,
    )


def start_demo_rendering(
    source_capture: cv2.VideoCapture,
    source_info: DemoVideoSourceInfo,
    output_path: str,
) -> cv2.VideoWriter:
    source_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return create_demo_video_writer(source_info=source_info, output_path=output_path)


def close_demo_writer(writer: Optional[cv2.VideoWriter]) -> Optional[str]:
    if writer is None:
        return None

    try:
        writer.release()
    except Exception as exc:
        return str(exc)
    return None


def finalize_optional_video_recording(
    writer: Optional[cv2.VideoWriter],
    output_path: Optional[str],
    result_label: str,
) -> tuple[Optional[cv2.VideoWriter], Optional[str]]:
    if writer is None:
        return None, None

    close_error = close_demo_writer(writer)
    if close_error is not None:
        return None, f"{result_label}时 writer 收尾失败: {close_error}"

    output_name = "output.mp4" if output_path is None else Path(output_path).name
    return None, f"{result_label}: {output_name}"


def build_demo_preview_panels(
    roi_gray_preview: Optional[np.ndarray],
    pred_label_map: Optional[np.ndarray],
) -> list[PreviewPanelSpec]:
    panels: list[PreviewPanelSpec] = []
    if roi_gray_preview is not None:
        panels.append(
            PreviewPanelSpec(
                image=roi_gray_preview,
                label="ROI Gray",
                desired_size=(ROI_PREVIEW_WIDTH, ROI_PREVIEW_HEIGHT),
                use_gray=True,
            )
        )
    if pred_label_map is not None:
        panels.append(
            PreviewPanelSpec(
                image=(pred_label_map * 85).astype(np.uint8),
                label="Segmentation",
                desired_size=(PRED_PREVIEW_WIDTH, PRED_PREVIEW_HEIGHT),
                use_gray=True,
            )
        )
    return panels


def compute_demo_panel_layout(
    frame_width: int,
    frame_height: int,
    panels: list[PreviewPanelSpec],
    margin: int = DEMO_PANEL_MARGIN,
    gap: int = DEMO_PANEL_GAP,
    anchor: str = DEMO_PANEL_ANCHOR_BOTTOM_LEFT,
    prefer_horizontal: bool = True,
) -> list[PreviewPanelPlacement]:
    if len(panels) == 0:
        return []

    available_width = max(int(frame_width) - 2 * int(margin), 1)
    available_height = max(int(frame_height) - 2 * int(margin), 1)
    widths = [max(int(panel.desired_size[0]), 1) for panel in panels]
    heights = [max(int(panel.desired_size[1]), 1) for panel in panels]

    def _scaled_sizes(scale: float) -> tuple[list[int], list[int]]:
        scaled_widths = [max(1, int(round(width * scale))) for width in widths]
        scaled_heights = [max(1, int(round(height * scale))) for height in heights]
        return scaled_widths, scaled_heights

    def _resolve_group_x(group_width: int) -> int:
        if anchor == DEMO_PANEL_ANCHOR_CENTER_LEFT:
            anchor_center_x = int(round(frame_width * DEMO_CENTER_LEFT_PANEL_ANCHOR_U))
            return int(np.clip(anchor_center_x - group_width // 2, margin, max(frame_width - margin - group_width, margin)))
        return int(margin)

    horizontal_total_width = sum(widths) + max(len(widths) - 1, 0) * gap
    horizontal_max_height = max(heights)
    horizontal_scale = min(
        1.0,
        available_width / float(max(horizontal_total_width, 1)),
        available_height / float(max(horizontal_max_height, 1)),
    )
    horizontal_widths, horizontal_heights = _scaled_sizes(horizontal_scale)
    horizontal_is_large_enough = (
        min(horizontal_widths) >= DEMO_MIN_PANEL_WIDTH and min(horizontal_heights) >= DEMO_MIN_PANEL_HEIGHT
    )

    vertical_total_height = sum(heights) + max(len(heights) - 1, 0) * gap
    vertical_max_width = max(widths)
    vertical_scale = min(
        1.0,
        available_width / float(max(vertical_max_width, 1)),
        available_height / float(max(vertical_total_height, 1)),
    )
    vertical_widths, vertical_heights = _scaled_sizes(vertical_scale)

    use_horizontal = bool(prefer_horizontal)
    if not prefer_horizontal:
        use_horizontal = False
    elif horizontal_total_width > available_width or horizontal_max_height > available_height:
        use_horizontal = horizontal_is_large_enough

    placements: list[PreviewPanelPlacement] = []
    if use_horizontal:
        current_x = _resolve_group_x(sum(horizontal_widths) + max(len(horizontal_widths) - 1, 0) * gap)
        group_height = max(horizontal_heights)
        if anchor == DEMO_PANEL_ANCHOR_CENTER_LEFT:
            base_y = int(np.clip((frame_height - group_height) // 2, margin, max(frame_height - margin - group_height, margin)))
        else:
            base_y = frame_height - margin - group_height
        for width, height in zip(horizontal_widths, horizontal_heights):
            if anchor == DEMO_PANEL_ANCHOR_CENTER_LEFT:
                panel_y = base_y + (group_height - height) // 2
            else:
                panel_y = frame_height - margin - height
            placements.append(PreviewPanelPlacement(x=current_x, y=panel_y, width=width, height=height))
            current_x += width + gap
        return placements

    group_width = max(vertical_widths)
    group_x = _resolve_group_x(group_width)
    total_height = sum(vertical_heights) + max(len(vertical_heights) - 1, 0) * gap
    if anchor == DEMO_PANEL_ANCHOR_CENTER_LEFT:
        current_y = int(np.clip((frame_height - total_height) // 2, margin, max(frame_height - margin - total_height, margin)))
        for width, height in zip(vertical_widths, vertical_heights):
            panel_x = group_x + (group_width - width) // 2
            placements.append(PreviewPanelPlacement(x=panel_x, y=current_y, width=width, height=height))
            current_y += height + gap
        return placements

    current_y = frame_height - margin
    for width, height in zip(vertical_widths, vertical_heights):
        current_y -= height
        placements.append(PreviewPanelPlacement(x=group_x, y=current_y, width=width, height=height))
        current_y -= gap
    return placements


def build_demo_gaze_sample(
    timestamp_s: float,
    screen_uv: Optional[Tuple[float, float]],
    tracking_valid: bool,
) -> DemoGazeSample:
    normalized_uv = None if screen_uv is None else (float(screen_uv[0]), float(screen_uv[1]))
    return DemoGazeSample(
        timestamp_s=max(float(timestamp_s), 0.0),
        screen_uv=normalized_uv,
        tracking_valid=bool(tracking_valid),
    )


def interpolate_screen_uv(
    start_uv: Optional[Tuple[float, float]],
    end_uv: Optional[Tuple[float, float]],
    alpha: float,
) -> Optional[Tuple[float, float]]:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if start_uv is None and end_uv is None:
        return None
    if start_uv is None or end_uv is None:
        return start_uv if alpha < 0.5 else end_uv
    return (
        float((1.0 - alpha) * start_uv[0] + alpha * end_uv[0]),
        float((1.0 - alpha) * start_uv[1] + alpha * end_uv[1]),
    )


def resolve_interpolated_demo_gaze(
    previous_sample: Optional[DemoGazeSample],
    current_sample: Optional[DemoGazeSample],
    target_timestamp_s: float,
) -> tuple[Optional[Tuple[float, float]], bool]:
    if previous_sample is None and current_sample is None:
        return None, False
    if previous_sample is None:
        return current_sample.screen_uv, current_sample.tracking_valid
    if current_sample is None:
        return previous_sample.screen_uv, previous_sample.tracking_valid

    if current_sample.timestamp_s <= previous_sample.timestamp_s:
        return current_sample.screen_uv, current_sample.tracking_valid
    if target_timestamp_s <= previous_sample.timestamp_s:
        return previous_sample.screen_uv, previous_sample.tracking_valid
    if target_timestamp_s >= current_sample.timestamp_s:
        return current_sample.screen_uv, current_sample.tracking_valid

    alpha = (float(target_timestamp_s) - previous_sample.timestamp_s) / max(
        current_sample.timestamp_s - previous_sample.timestamp_s,
        1e-6,
    )
    screen_uv = interpolate_screen_uv(previous_sample.screen_uv, current_sample.screen_uv, alpha)
    tracking_valid = previous_sample.tracking_valid if alpha < 0.5 else current_sample.tracking_valid
    return screen_uv, tracking_valid


def map_demo_screen_uv_to_frame(
    screen_uv: Tuple[float, float],
    width: int,
    height: int,
) -> Tuple[int, int]:
    px = int(round(np.clip(screen_uv[0], 0.0, 1.0) * max(width - 1, 0)))
    py = int(round(np.clip(screen_uv[1], 0.0, 1.0) * max(height - 1, 0)))
    return px, py


def draw_demo_gaze_overlay(
    frame: np.ndarray,
    screen_uv: Optional[Tuple[float, float]],
    tracking_valid: bool,
) -> np.ndarray:
    output = frame.copy()
    if screen_uv is None:
        return output

    height, width = output.shape[:2]
    px, py = map_demo_screen_uv_to_frame(screen_uv, width=width, height=height)
    color = DEMO_GAZE_VALID_COLOR if tracking_valid else DEMO_GAZE_INVALID_COLOR
    shadow_center = (px + DEMO_GAZE_SHADOW_OFFSET, py + DEMO_GAZE_SHADOW_OFFSET)

    cv2.circle(
        output,
        shadow_center,
        DEMO_GAZE_OUTER_RING_RADIUS,
        DEMO_GAZE_SHADOW_COLOR,
        DEMO_GAZE_OUTER_RING_THICKNESS + 2,
        cv2.LINE_AA,
    )
    cv2.circle(
        output,
        shadow_center,
        DEMO_GAZE_RING_RADIUS,
        DEMO_GAZE_SHADOW_COLOR,
        DEMO_GAZE_RING_THICKNESS + 2,
        cv2.LINE_AA,
    )
    cv2.circle(
        output,
        shadow_center,
        DEMO_GAZE_CENTER_RADIUS + 2,
        DEMO_GAZE_SHADOW_COLOR,
        -1,
        cv2.LINE_AA,
    )

    cv2.circle(
        output,
        (px, py),
        DEMO_GAZE_OUTER_RING_RADIUS,
        color,
        DEMO_GAZE_OUTER_RING_THICKNESS,
        cv2.LINE_AA,
    )
    cv2.circle(
        output,
        (px, py),
        DEMO_GAZE_RING_RADIUS,
        color,
        DEMO_GAZE_RING_THICKNESS,
        cv2.LINE_AA,
    )
    cv2.circle(
        output,
        (px, py),
        DEMO_GAZE_CENTER_RADIUS,
        color,
        -1,
        cv2.LINE_AA,
    )
    return output


def compose_demo_output_frame(
    fpv_frame_bgr: np.ndarray,
    screen_uv: Optional[Tuple[float, float]],
    tracking_valid: bool,
    roi_gray_preview: Optional[np.ndarray],
    pred_label_map: Optional[np.ndarray],
    panel_anchor: str = DEMO_PANEL_ANCHOR_BOTTOM_LEFT,
    prefer_horizontal_panels: bool = True,
) -> np.ndarray:
    output = draw_demo_gaze_overlay(
        fpv_frame_bgr,
        screen_uv=screen_uv,
        tracking_valid=tracking_valid,
    )
    panels = build_demo_preview_panels(roi_gray_preview=roi_gray_preview, pred_label_map=pred_label_map)
    placements = compute_demo_panel_layout(
        frame_width=output.shape[1],
        frame_height=output.shape[0],
        panels=panels,
        anchor=panel_anchor,
        prefer_horizontal=prefer_horizontal_panels,
    )
    for panel, placement in zip(panels, placements):
        output = draw_preview_panel(
            output,
            panel.image,
            panel.label,
            (placement.x, placement.y),
            (placement.width, placement.height),
            use_gray=panel.use_gray,
        )
    return output


def compose_calibration_output_frame(
    calibration_canvas_bgr: np.ndarray,
    roi_gray_preview: Optional[np.ndarray],
    pred_label_map: Optional[np.ndarray],
) -> np.ndarray:
    output = calibration_canvas_bgr.copy()
    panels = build_demo_preview_panels(roi_gray_preview=roi_gray_preview, pred_label_map=pred_label_map)
    placements = compute_demo_panel_layout(
        frame_width=output.shape[1],
        frame_height=output.shape[0],
        panels=panels,
        anchor=DEMO_PANEL_ANCHOR_CENTER_LEFT,
        prefer_horizontal=False,
    )
    for panel, placement in zip(panels, placements):
        output = draw_preview_panel(
            output,
            panel.image,
            panel.label,
            (placement.x, placement.y),
            (placement.width, placement.height),
            use_gray=panel.use_gray,
        )
    return output


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实时眼动方向验证")
    parser.add_argument("--model_path", type=str, default=None, help="待加载的模型路径，支持 .pth 与 .onnx")
    parser.add_argument("--checkpoint_path", type=str, default=CHECKPOINT_PATH, help="兼容旧参数名；若未传 --model_path，则使用该路径")
    parser.add_argument("--camera_index", type=int, default=CAMERA_INDEX, help="摄像头索引")
    parser.add_argument("--demo_fpv_video", type=str, default=None, help="启用 PC 演示录制模式时使用的 FPV 视频路径")
    parser.add_argument("--demo_output", type=str, default=None, help="演示录制模式导出路径；默认输出到 FPV 视频同目录")
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

    return parser.parse_args(argv)


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
    demo_mode_enabled = args.demo_fpv_video is not None
    demo_output_path = None if not demo_mode_enabled else resolve_demo_output_path(args.demo_fpv_video, args.demo_output)
    demo_calibration_output_path = None if demo_output_path is None else resolve_demo_calibration_output_path(demo_output_path)
    demo_source_capture: Optional[cv2.VideoCapture] = None
    demo_source_info: Optional[DemoVideoSourceInfo] = None

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
    if demo_mode_enabled:
        demo_source_capture, demo_source_info = inspect_demo_video_source(args.demo_fpv_video)
        print(
            "demo fpv:",
            f"{demo_source_info.width}x{demo_source_info.height}, fps={demo_source_info.fps:.2f}, "
            f"frames={demo_source_info.frame_count}, path={demo_source_info.path}",
        )
        print("demo output:", demo_output_path)
        print("demo calibration output:", demo_calibration_output_path)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, handle_mouse)

    quality_tracker = FeatureQualityTracker(alpha=QUALITY_TRACKER_ALPHA)
    kalman_filter = AdaptiveKalmanFilter2D()
    calibration_session: Optional[CalibrationSession] = None
    calibration_window_open = False
    calibration_points = build_nine_point_calibration_points(args.calibration_margin)
    displayed_screen_uv: Optional[Tuple[float, float]] = None
    last_valid_screen_ts_ms: Optional[float] = None
    tracking_valid = False
    status_message = "请选择 ROI，然后按 s 开始九点校准。"
    if demo_mode_enabled:
        status_message += f" 校准完成后按 {DEFAULT_DEMO_RECORD_KEY} 从头导出演示视频。"
    last_roi_revision = ROI_STATE.roi_revision
    prev_time = time.time()
    demo_render_state = DEMO_RENDER_STATE_IDLE
    demo_writer: Optional[cv2.VideoWriter] = None
    demo_calibration_writer: Optional[cv2.VideoWriter] = None
    demo_calibration_recording_blocked = False
    demo_render_start_monotonic: Optional[float] = None
    demo_rendered_frames = 0
    demo_preview_frame: Optional[np.ndarray] = None
    calibration_preview_frame: Optional[np.ndarray] = None
    demo_previous_gaze_sample: Optional[DemoGazeSample] = None
    demo_current_gaze_sample: Optional[DemoGazeSample] = None

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
                stop_error = close_demo_writer(demo_writer)
                if stop_error is not None:
                    print(f"warning: failed to finalize demo writer during ROI update: {stop_error}")
                demo_writer = None
                demo_calibration_writer, calibration_video_message = finalize_optional_video_recording(
                    demo_calibration_writer,
                    demo_calibration_output_path,
                    "校准阶段视频已停止",
                )
                if calibration_video_message is not None:
                    print(calibration_video_message)
                demo_calibration_recording_blocked = False
                demo_render_start_monotonic = None
                demo_rendered_frames = 0
                demo_preview_frame = None
                demo_previous_gaze_sample = None
                demo_current_gaze_sample = None
                demo_render_state = DEMO_RENDER_STATE_IDLE
                quality_tracker.reset()
                kalman_filter.reset()
                calibration_session = None
                calibration_window_open = False
                destroy_window(CALIBRATION_WINDOW_NAME)
                displayed_screen_uv = None
                last_valid_screen_ts_ms = None
                tracking_valid = False
                status_message = "ROI 已更新，校准已失效。"
                if calibration_video_message is not None:
                    status_message = f"{status_message} {calibration_video_message}"

            selected_roi = ROI_STATE.active_roi
            geometry_result = build_empty_geometry_result()
            selected_feature_x: Optional[float] = None
            selected_feature_y: Optional[float] = None
            selected_feature_valid = False
            selected_feature_confidence = 0.0
            selected_feature_reasons: tuple[str, ...] = tuple()
            preprocess_meta: Optional[ResizeMeta] = None
            roi_gray_preview: Optional[np.ndarray] = None
            pred_label_map: Optional[np.ndarray] = None
            calibration_preview_frame = None

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

            calibration_model_for_feature = calibration_session.model if calibration_session is not None and calibration_session.is_calibrated else None
            static_boundary_ellipse = calibration_model_for_feature.static_boundary_ellipse if calibration_model_for_feature is not None else None

            selected_feature_x, selected_feature_y, selected_feature_valid, selected_feature_reasons = resolve_tracking_features(
                geometry_result,
                feature_mode=args.feature_mode,
                reference_boundary_ellipse=static_boundary_ellipse,
            )

            selected_feature_confidence = quality_tracker.compute_confidence(
                geometry_result=geometry_result,
                feature_mode=args.feature_mode,
                feature_x=selected_feature_x,
                feature_y=selected_feature_y,
                feature_valid=selected_feature_valid,
                feature_reasons=selected_feature_reasons,
            )

            smoothed_dx, smoothed_dy = kalman_filter.update(
                selected_feature_x,
                selected_feature_y,
                selected_feature_valid,
                selected_feature_confidence,
            )

            calibration_feature_valid = bool(
                selected_feature_valid
                and selected_feature_confidence >= MIN_CALIBRATION_CONFIDENCE
                and smoothed_dx is not None
                and smoothed_dy is not None
            )

            now_ms = time.time() * 1000.0

            if calibration_session is not None and calibration_session.is_active:
                previous_state = calibration_session.state
                calibration_session = advance_calibration_session(
                    calibration_session,
                    now_ms=now_ms,
                    feature_dx=smoothed_dx,
                    feature_dy=smoothed_dy,
                    feature_valid=calibration_feature_valid,
                    feature_confidence=selected_feature_confidence,
                    boundary_ellipse=geometry_result.outer_boundary_geometry.ellipse,
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
                        status_message = "九点校准完成。"
                        demo_calibration_writer, calibration_video_message = finalize_optional_video_recording(
                            demo_calibration_writer,
                            demo_calibration_output_path,
                            "校准阶段视频已保存",
                        )
                        demo_calibration_recording_blocked = False
                        if calibration_video_message is not None:
                            status_message = f"{status_message} {calibration_video_message}"
                        if demo_mode_enabled:
                            status_message += f" 按 {DEFAULT_DEMO_RECORD_KEY} 从头导出演示视频。"
                        print(status_message)
                    elif calibration_session.state == "failed":
                        calibration_window_open = False
                        destroy_window(CALIBRATION_WINDOW_NAME)
                        status_message = calibration_session.failure_reason or "校准失败。"
                        demo_calibration_writer, calibration_video_message = finalize_optional_video_recording(
                            demo_calibration_writer,
                            demo_calibration_output_path,
                            "校准阶段视频已保存",
                        )
                        demo_calibration_recording_blocked = False
                        if calibration_video_message is not None:
                            status_message = f"{status_message} {calibration_video_message}"
                        print(status_message)

            if calibration_session is not None and calibration_session.is_active:
                if not calibration_window_open:
                    ensure_fullscreen_window(CALIBRATION_WINDOW_NAME)
                    calibration_window_open = True
                calibration_canvas = build_calibration_canvas((camera_width, camera_height), calibration_session)
                calibration_preview_frame = calibration_canvas
                if demo_mode_enabled:
                    calibration_preview_frame = compose_calibration_output_frame(
                        calibration_canvas_bgr=calibration_canvas,
                        roi_gray_preview=roi_gray_preview,
                        pred_label_map=pred_label_map,
                    )
                    if (
                        demo_calibration_writer is None
                        and not demo_calibration_recording_blocked
                        and demo_calibration_output_path is not None
                    ):
                        calibration_record_fps = float(camera_fps) if camera_fps > 1.0 else DEMO_CALIBRATION_RECORD_FPS_FALLBACK
                        try:
                            demo_calibration_writer = create_mp4_video_writer(
                                output_path=demo_calibration_output_path,
                                width=calibration_preview_frame.shape[1],
                                height=calibration_preview_frame.shape[0],
                                fps=calibration_record_fps,
                            )
                            print(f"开始导出校准阶段视频: {Path(demo_calibration_output_path).name}")
                        except Exception as exc:
                            demo_calibration_writer = None
                            demo_calibration_recording_blocked = True
                            print(f"warning: 无法开始校准阶段视频导出: {exc}")
                    if demo_calibration_writer is not None:
                        demo_calibration_writer.write(calibration_preview_frame)
                cv2.imshow(CALIBRATION_WINDOW_NAME, calibration_preview_frame)
            elif calibration_window_open:
                destroy_window(CALIBRATION_WINDOW_NAME)
                calibration_window_open = False

            calibration_model = calibration_session.model if calibration_session is not None and calibration_session.is_calibrated else None
            tracking_valid = False
            tracking_gate_valid = bool(
                selected_feature_valid
                and selected_feature_confidence >= MIN_TRACKING_CONFIDENCE
                and smoothed_dx is not None
                and smoothed_dy is not None
            )
            if calibration_model is not None and tracking_gate_valid:
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
            demo_render_state = resolve_demo_render_state(
                current_state=demo_render_state,
                demo_source_info=demo_source_info,
                selected_roi=selected_roi,
                calibration_session=calibration_session,
                rendering_active=demo_writer is not None,
            )
            demo_render_elapsed_s: Optional[float] = None
            if demo_mode_enabled and demo_writer is not None and demo_render_start_monotonic is not None:
                demo_render_elapsed_s = max(time.monotonic() - demo_render_start_monotonic, 0.0)
                latest_gaze_sample = build_demo_gaze_sample(
                    timestamp_s=demo_render_elapsed_s,
                    screen_uv=displayed_screen_uv,
                    tracking_valid=tracking_valid,
                )
                if demo_current_gaze_sample is None:
                    demo_previous_gaze_sample = latest_gaze_sample
                    demo_current_gaze_sample = latest_gaze_sample
                else:
                    demo_previous_gaze_sample = demo_current_gaze_sample
                    demo_current_gaze_sample = latest_gaze_sample

            vis = build_visualization(
                frame_bgr=frame,
                geometry_result=geometry_result,
                feature_mode=args.feature_mode,
                selected_feature_x=selected_feature_x,
                selected_feature_y=selected_feature_y,
                selected_feature_valid=selected_feature_valid,
                selected_feature_confidence=selected_feature_confidence,
                selected_feature_reasons=selected_feature_reasons,
                preprocess_meta=preprocess_meta,
                selected_roi=selected_roi,
                smoothed_dx=smoothed_dx,
                smoothed_dy=smoothed_dy,
                fps=fps,
                camera_resolution=(camera_width, camera_height),
                calibration_session=calibration_session,
                static_boundary_ellipse=static_boundary_ellipse,
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

            if demo_mode_enabled and demo_writer is not None and demo_source_capture is not None and demo_source_info is not None:
                if demo_render_start_monotonic is None:
                    demo_render_start_monotonic = time.monotonic()
                if demo_render_elapsed_s is None:
                    demo_render_elapsed_s = max(time.monotonic() - demo_render_start_monotonic, 0.0)

                target_frame_count = max(int(demo_render_elapsed_s * demo_source_info.fps) + 1, 1)
                if demo_source_info.frame_count > 0:
                    target_frame_count = min(target_frame_count, demo_source_info.frame_count)
                frames_to_render = max(target_frame_count - demo_rendered_frames, 0)

                while frames_to_render > 0:
                    demo_ret, demo_frame = demo_source_capture.read()
                    if not demo_ret or demo_frame is None or demo_frame.size == 0:
                        stop_error = close_demo_writer(demo_writer)
                        demo_writer = None
                        demo_render_start_monotonic = None
                        demo_rendered_frames = 0
                        demo_previous_gaze_sample = None
                        demo_current_gaze_sample = None
                        demo_render_state = DEMO_RENDER_STATE_FINISHED
                        if stop_error is not None:
                            status_message = f"演示视频导出结束，但 writer 收尾失败: {stop_error}"
                        else:
                            status_message = f"演示视频导出已完成: {Path(demo_output_path).name}"
                        print(status_message)
                        break

                    frame_timestamp_s = demo_rendered_frames / float(max(demo_source_info.fps, 1e-6))
                    interpolated_screen_uv, interpolated_tracking_valid = resolve_interpolated_demo_gaze(
                        previous_sample=demo_previous_gaze_sample,
                        current_sample=demo_current_gaze_sample,
                        target_timestamp_s=frame_timestamp_s,
                    )
                    demo_preview_frame = compose_demo_output_frame(
                        fpv_frame_bgr=demo_frame,
                        screen_uv=interpolated_screen_uv,
                        tracking_valid=interpolated_tracking_valid,
                        roi_gray_preview=roi_gray_preview,
                        pred_label_map=pred_label_map,
                    )
                    demo_writer.write(demo_preview_frame)
                    demo_rendered_frames += 1
                    frames_to_render -= 1

                    if demo_source_info.frame_count > 0 and demo_rendered_frames >= demo_source_info.frame_count:
                        stop_error = close_demo_writer(demo_writer)
                        demo_writer = None
                        demo_render_start_monotonic = None
                        demo_rendered_frames = 0
                        demo_previous_gaze_sample = None
                        demo_current_gaze_sample = None
                        demo_render_state = DEMO_RENDER_STATE_FINISHED
                        if stop_error is not None:
                            status_message = f"演示视频导出结束，但 writer 收尾失败: {stop_error}"
                        else:
                            status_message = f"演示视频导出已完成: {Path(demo_output_path).name}"
                        print(status_message)
                        break

                if demo_preview_frame is not None:
                    vis = demo_preview_frame

            cv2.imshow(WINDOW_NAME, vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                stop_error = close_demo_writer(demo_writer)
                if stop_error is not None:
                    print(f"warning: failed to finalize demo writer during quit: {stop_error}")
                demo_writer = None
                demo_calibration_writer, calibration_video_message = finalize_optional_video_recording(
                    demo_calibration_writer,
                    demo_calibration_output_path,
                    "校准阶段视频已停止",
                )
                if calibration_video_message is not None:
                    print(calibration_video_message)
                demo_calibration_recording_blocked = False
                demo_previous_gaze_sample = None
                demo_current_gaze_sample = None
                break
            elif demo_mode_enabled and key == ord(DEFAULT_DEMO_RECORD_KEY):
                start_error = validate_demo_render_start_request(
                    demo_source_info=demo_source_info,
                    selected_roi=selected_roi,
                    calibration_session=calibration_session,
                    rendering_active=demo_writer is not None,
                )
                if demo_writer is not None:
                    stop_error = close_demo_writer(demo_writer)
                    demo_writer = None
                    demo_render_start_monotonic = None
                    demo_rendered_frames = 0
                    demo_render_state = DEMO_RENDER_STATE_FINISHED
                    demo_previous_gaze_sample = None
                    demo_current_gaze_sample = None
                    if stop_error is not None:
                        status_message = f"停止演示导出时 writer 收尾失败: {stop_error}"
                    else:
                        status_message = f"演示视频导出已停止: {Path(demo_output_path).name}"
                    print(status_message)
                elif start_error is not None:
                    status_message = start_error
                    print(status_message)
                else:
                    try:
                        demo_writer = start_demo_rendering(
                            source_capture=demo_source_capture,
                            source_info=demo_source_info,
                            output_path=demo_output_path,
                        )
                        demo_render_start_monotonic = time.monotonic()
                        demo_rendered_frames = 0
                        demo_preview_frame = None
                        initial_gaze_sample = build_demo_gaze_sample(
                            timestamp_s=0.0,
                            screen_uv=displayed_screen_uv,
                            tracking_valid=tracking_valid,
                        )
                        demo_previous_gaze_sample = initial_gaze_sample
                        demo_current_gaze_sample = initial_gaze_sample
                        demo_render_state = DEMO_RENDER_STATE_RENDERING
                        status_message = f"开始导出演示视频: {Path(demo_output_path).name}"
                    except Exception as exc:
                        demo_writer = None
                        demo_render_start_monotonic = None
                        demo_rendered_frames = 0
                        demo_previous_gaze_sample = None
                        demo_current_gaze_sample = None
                        demo_render_state = DEMO_RENDER_STATE_FINISHED
                        status_message = f"无法开始演示视频导出: {exc}"
                    print(status_message)
            elif key == ord("r"):
                stop_error = close_demo_writer(demo_writer)
                if stop_error is not None:
                    print(f"warning: failed to finalize demo writer during reset: {stop_error}")
                demo_writer = None
                demo_calibration_writer, calibration_video_message = finalize_optional_video_recording(
                    demo_calibration_writer,
                    demo_calibration_output_path,
                    "校准阶段视频已停止",
                )
                demo_calibration_recording_blocked = False
                demo_render_start_monotonic = None
                demo_rendered_frames = 0
                demo_preview_frame = None
                demo_previous_gaze_sample = None
                demo_current_gaze_sample = None
                demo_render_state = DEMO_RENDER_STATE_IDLE
                quality_tracker.reset()
                kalman_filter.reset()
                calibration_session = None
                calibration_window_open = False
                destroy_window(CALIBRATION_WINDOW_NAME)
                displayed_screen_uv = None
                last_valid_screen_ts_ms = None
                tracking_valid = False
                status_message = "已重置滤波与校准状态。"
                if calibration_video_message is not None:
                    status_message = f"{status_message} {calibration_video_message}"
                print(status_message)
            elif key == ord("c"):
                stop_error = close_demo_writer(demo_writer)
                if stop_error is not None:
                    print(f"warning: failed to finalize demo writer during ROI clear: {stop_error}")
                demo_writer = None
                demo_calibration_writer, calibration_video_message = finalize_optional_video_recording(
                    demo_calibration_writer,
                    demo_calibration_output_path,
                    "校准阶段视频已停止",
                )
                demo_calibration_recording_blocked = False
                demo_render_start_monotonic = None
                demo_rendered_frames = 0
                demo_preview_frame = None
                demo_previous_gaze_sample = None
                demo_current_gaze_sample = None
                demo_render_state = DEMO_RENDER_STATE_IDLE
                ROI_STATE.clear()
                quality_tracker.reset()
                kalman_filter.reset()
                calibration_session = None
                calibration_window_open = False
                destroy_window(CALIBRATION_WINDOW_NAME)
                displayed_screen_uv = None
                last_valid_screen_ts_ms = None
                tracking_valid = False
                status_message = "已清除 ROI，校准已失效。"
                if calibration_video_message is not None:
                    status_message = f"{status_message} {calibration_video_message}"
                print(status_message)
            elif key == ord("s"):
                if selected_roi is None or not selected_roi.is_valid():
                    status_message = "请先框选有效 ROI，再开始校准。"
                    print(status_message)
                else:
                    stop_error = close_demo_writer(demo_writer)
                    if stop_error is not None:
                        print(f"warning: failed to finalize demo writer before calibration: {stop_error}")
                    demo_writer = None
                    demo_calibration_writer, calibration_video_message = finalize_optional_video_recording(
                        demo_calibration_writer,
                        demo_calibration_output_path,
                        "校准阶段视频已停止",
                    )
                    if calibration_video_message is not None:
                        print(calibration_video_message)
                    demo_calibration_recording_blocked = False
                    demo_render_start_monotonic = None
                    demo_rendered_frames = 0
                    demo_preview_frame = None
                    demo_previous_gaze_sample = None
                    demo_current_gaze_sample = None
                    demo_render_state = DEMO_RENDER_STATE_IDLE
                    quality_tracker.reset()
                    kalman_filter.reset()
                    calibration_session = begin_calibration_session(
                        now_ms=now_ms,
                        settle_ms=args.calibration_settle_ms,
                        capture_ms=args.calibration_capture_ms,
                        min_valid_frames=args.calibration_min_valid_frames,
                        margin=args.calibration_margin,
                        point_pattern="nine",
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
                    stop_error = close_demo_writer(demo_writer)
                    if stop_error is not None:
                        print(f"warning: failed to finalize demo writer during calibration cancel: {stop_error}")
                    demo_writer = None
                    demo_calibration_writer, calibration_video_message = finalize_optional_video_recording(
                        demo_calibration_writer,
                        demo_calibration_output_path,
                        "校准阶段视频已停止",
                    )
                    demo_calibration_recording_blocked = False
                    demo_render_start_monotonic = None
                    demo_rendered_frames = 0
                    demo_preview_frame = None
                    demo_previous_gaze_sample = None
                    demo_current_gaze_sample = None
                    demo_render_state = DEMO_RENDER_STATE_IDLE
                    calibration_session = cancel_calibration_session(calibration_session, "已取消当前校准。")
                    calibration_window_open = False
                    destroy_window(CALIBRATION_WINDOW_NAME)
                    displayed_screen_uv = None
                    last_valid_screen_ts_ms = None
                    tracking_valid = False
                    status_message = calibration_session.failure_reason or "已取消当前校准。"
                    if calibration_video_message is not None:
                        status_message = f"{status_message} {calibration_video_message}"
                    print(status_message)
    finally:
        stop_error = close_demo_writer(demo_writer)
        if stop_error is not None:
            print(f"warning: failed to finalize demo writer during shutdown: {stop_error}")
        demo_calibration_writer, calibration_video_message = finalize_optional_video_recording(
            demo_calibration_writer,
            demo_calibration_output_path,
            "校准阶段视频已停止",
        )
        if calibration_video_message is not None:
            print(calibration_video_message)
        demo_previous_gaze_sample = None
        demo_current_gaze_sample = None
        demo_calibration_recording_blocked = False
        if demo_source_capture is not None:
            demo_source_capture.release()
        cap.release()
        destroy_window(CALIBRATION_WINDOW_NAME)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
