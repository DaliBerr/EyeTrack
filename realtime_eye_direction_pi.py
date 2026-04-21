from __future__ import annotations

import argparse
import os
import socket
import time
from typing import Optional

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - preview is optional
    cv2 = None

from eyetrack.config import DEFAULT_INPUT_HEIGHT, DEFAULT_INPUT_WIDTH
from eyetrack.gaze import (
    CalibrationSession,
    EllipseResult,
    GazeFeatureResult,
    advance_calibration_session,
    begin_calibration_session,
    cancel_calibration_session,
    extract_gaze_features_from_label_map,
    predict_screen_point,
    resolve_tracking_features,
)
from eyetrack.realtime_gaze import build_calibration_canvas, destroy_window, draw_screen_preview_panel
from eyetrack.raspi import (
    GazeMetadataPublisher,
    NonBlockingTerminalReader,
    PiOnnxSegmentationRuntime,
    PicameraStreamConfig,
    PicameraStreamWorker,
    RtspVideoServer,
    build_gaze_metadata_packet,
    convert_fpv_frame_to_bgr,
    discover_picamera_cameras,
    is_frame_stale,
    pair_eye_and_fpv_frames,
    prepare_eye_tensor_from_yuv420,
)


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
STALE_SCREEN_POINT_TIMEOUT_MS = 300
DEFAULT_CAMERA_STALE_MS = 500
DEFAULT_SYNC_TOLERANCE_MS = 40
DEFAULT_EYE_FPS = 30.0
DEFAULT_FPV_FPS = 30.0
DEFAULT_FPV_WIDTH = 1280
DEFAULT_FPV_HEIGHT = 720
DEFAULT_FPV_PIXEL_FORMAT = "BGR888"
DEFAULT_RTSP_HOST = "0.0.0.0"
DEFAULT_RTSP_PORT = 8554
DEFAULT_RTSP_PATH = "fpv"
DEFAULT_RTSP_BITRATE_KBPS = 4000
DEFAULT_RECONNECT_INTERVAL_MS = 2000
DEFAULT_CALIBRATION_SETTLE_MS = 500
DEFAULT_CALIBRATION_CAPTURE_MS = 1000
DEFAULT_CALIBRATION_MIN_VALID_FRAMES = 15
DEFAULT_CALIBRATION_MARGIN = 0.1
DEFAULT_EYE_PREVIEW_SCALE = 2.0
EYE_PREVIEW_WINDOW_NAME = "Pi Eye Tracking Preview"
CALIBRATION_WINDOW_NAME = "Pi Calibration Target"
DEFAULT_CALIBRATION_WINDOW_WIDTH = 1280
DEFAULT_CALIBRATION_WINDOW_HEIGHT = 720


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
        self.last_measurement: Optional[tuple[float, float]] = None

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
    def _update_axis(state: float, variance: float, measurement: float, measurement_noise: float) -> tuple[float, float]:
        gain = variance / (variance + measurement_noise)
        next_state = state + gain * (measurement - state)
        next_variance = (1.0 - gain) * variance
        return float(next_state), float(next_variance)

    def update(
        self,
        x: Optional[float],
        y: Optional[float],
        valid: bool,
        confidence: float,
    ) -> tuple[Optional[float], Optional[float]]:
        self._predict()

        if not valid or x is None or y is None:
            return self.state_x, self.state_y

        measurement_x = float(x)
        measurement_y = float(y)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raspberry Pi 5 dual-CSI real-time gaze metadata + FPV video")
    parser.add_argument("--model_path", type=str, required=True, help="Quantized ONNX model path (.onnx only)")
    parser.add_argument("--eye_only_mode", action="store_true", help="Use only the eye camera and skip FPV/RTSP.")
    parser.add_argument("--eye_camera_id", type=int, default=0, help="Near-eye IR camera ID")
    parser.add_argument("--fpv_camera_id", type=int, default=1, help="FPV camera ID")
    parser.add_argument("--eye_width", type=int, default=DEFAULT_INPUT_WIDTH, help="Near-eye camera width; must match model input")
    parser.add_argument("--eye_height", type=int, default=DEFAULT_INPUT_HEIGHT, help="Near-eye camera height; must match model input")
    parser.add_argument("--fpv_width", type=int, default=DEFAULT_FPV_WIDTH, help="FPV output width")
    parser.add_argument("--fpv_height", type=int, default=DEFAULT_FPV_HEIGHT, help="FPV output height")
    parser.add_argument("--eye_fps", type=float, default=DEFAULT_EYE_FPS, help="Near-eye camera FPS")
    parser.add_argument("--fpv_fps", type=float, default=DEFAULT_FPV_FPS, help="FPV camera FPS")
    parser.add_argument("--feature_mode", type=str, choices=["pupil_iris", "iris_only"], default="iris_only", help="Feature mode used for real-time calibration and tracking")
    parser.add_argument("--sync_tolerance_ms", type=float, default=DEFAULT_SYNC_TOLERANCE_MS, help="cam0/cam1 timestamp tolerance in ms; larger skew marks sync_stale")
    parser.add_argument("--rtsp_host", type=str, default=DEFAULT_RTSP_HOST, help="RTSP bind host")
    parser.add_argument("--rtsp_port", type=int, default=DEFAULT_RTSP_PORT, help="RTSP port")
    parser.add_argument("--rtsp_path", type=str, default=DEFAULT_RTSP_PATH, help="RTSP path")
    parser.add_argument("--rtsp_bitrate_kbps", type=int, default=DEFAULT_RTSP_BITRATE_KBPS, help="RTSP H.264 target bitrate")
    parser.add_argument("--metadata_stdout", dest="metadata_stdout", action="store_true", help="Output gaze/calibration metadata as JSON lines to stdout")
    parser.add_argument("--no-metadata_stdout", dest="metadata_stdout", action="store_false", help="Disable metadata output to stdout")
    parser.add_argument("--metadata_udp_host", type=str, default=None, help="Optional UDP host for metadata output")
    parser.add_argument("--metadata_udp_port", type=int, default=None, help="Optional UDP port for metadata output")
    parser.add_argument("--eye_hflip", action="store_true", help="Flip near-eye camera horizontally")
    parser.add_argument("--eye_vflip", action="store_true", help="Flip near-eye camera vertically")
    parser.add_argument("--fpv_hflip", action="store_true", help="Flip FPV camera horizontally")
    parser.add_argument("--fpv_vflip", action="store_true", help="Flip FPV camera vertically")
    parser.add_argument("--fpv_pixel_format", type=str, default=DEFAULT_FPV_PIXEL_FORMAT, help="FPV pixel format (default: BGR888)")
    parser.add_argument("--camera_stale_ms", type=float, default=DEFAULT_CAMERA_STALE_MS, help="Mark frame stale when camera has not updated for this duration")
    parser.add_argument("--reconnect_interval_ms", type=int, default=DEFAULT_RECONNECT_INTERVAL_MS, help="Auto-reconnect interval after camera errors")
    parser.add_argument("--calibration_settle_ms", type=int, default=DEFAULT_CALIBRATION_SETTLE_MS, help="Settle time after switching calibration points")
    parser.add_argument("--calibration_capture_ms", type=int, default=DEFAULT_CALIBRATION_CAPTURE_MS, help="Capture duration for each calibration point")
    parser.add_argument("--calibration_min_valid_frames", type=int, default=DEFAULT_CALIBRATION_MIN_VALID_FRAMES, help="Minimum valid frames required for each calibration point")
    parser.add_argument("--calibration_margin", type=float, default=DEFAULT_CALIBRATION_MARGIN, help="Normalized margin from edges for corner calibration points")
    parser.add_argument("--eye_preview", dest="eye_preview", action="store_true", help="Enable local eye-tracking preview window")
    parser.add_argument("--no-eye_preview", dest="eye_preview", action="store_false", help="Disable local eye-tracking preview window")
    parser.add_argument("--eye_preview_scale", type=float, default=DEFAULT_EYE_PREVIEW_SCALE, help="Scale factor for the eye preview window")
    parser.set_defaults(metadata_stdout=True, eye_preview=None)
    return parser.parse_args()


def resolve_display_host(rtsp_host: str) -> str:
    if rtsp_host not in {"0.0.0.0", "::"}:
        return rtsp_host

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def print_camera_inventory(eye_camera_id: int, fpv_camera_id: Optional[int]) -> None:
    cameras = discover_picamera_cameras()
    if len(cameras) == 0:
        raise RuntimeError("No Picamera2 cameras were detected.")

    print("detected cameras:")
    for camera in cameras:
        marker = []
        if camera.camera_id == eye_camera_id:
            marker.append("eye")
        if fpv_camera_id is not None and camera.camera_id == fpv_camera_id:
            marker.append("fpv")
        label = ",".join(marker) if len(marker) > 0 else "-"
        print(
            f"  id={camera.camera_id} [{label}] model={camera.model} "
            f"location={camera.location} rotation={camera.rotation} id={camera.identifier}"
        )

    camera_ids = {camera.camera_id for camera in cameras}
    if eye_camera_id not in camera_ids:
        raise RuntimeError(f"No camera found for eye_camera_id={eye_camera_id}.")
    if fpv_camera_id is not None and fpv_camera_id not in camera_ids:
        raise RuntimeError(f"No camera found for fpv_camera_id={fpv_camera_id}.")
    if fpv_camera_id is not None and eye_camera_id == fpv_camera_id:
        raise RuntimeError("eye_camera_id and fpv_camera_id must be different.")


def clear_tracking_state(
    quality_tracker: FeatureQualityTracker,
    kalman_filter: AdaptiveKalmanFilter2D,
) -> tuple[Optional[tuple[float, float]], Optional[float], bool]:
    quality_tracker.reset()
    kalman_filter.reset()
    displayed_screen_uv = None
    last_valid_screen_ts_ms = None
    tracking_valid = False
    return displayed_screen_uv, last_valid_screen_ts_ms, tracking_valid


def extract_geometry_result(pred_label_map):
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


def maybe_reconnect_camera(
    worker: PicameraStreamWorker,
    now_ms: float,
    last_attempt_ms: float,
    reconnect_interval_ms: int,
) -> tuple[float, Optional[str], bool]:
    health = worker.get_health()
    if health.running or now_ms - last_attempt_ms < reconnect_interval_ms:
        return last_attempt_ms, health.last_error, False

    try:
        worker.restart()
        return now_ms, None, True
    except Exception as exc:
        return now_ms, str(exc), False


def resolve_preview_enabled(eye_only_mode: bool, eye_preview: Optional[bool]) -> bool:
    if eye_preview is None:
        return bool(eye_only_mode)
    return bool(eye_preview)


def has_graphical_session() -> bool:
    display = os.environ.get("DISPLAY", "").strip()
    wayland_display = os.environ.get("WAYLAND_DISPLAY", "").strip()
    return len(display) > 0 or len(wayland_display) > 0


def initialize_eye_preview_window(enabled: bool, scale: float, eye_width: int, eye_height: int) -> bool:
    if not enabled:
        return False

    if not has_graphical_session():
        print("warning: no DISPLAY/WAYLAND_DISPLAY detected; disabling eye preview window.")
        return False

    if cv2 is None:
        print("warning: cv2 is unavailable; cannot enable eye preview window.")
        return False

    try:
        cv2.namedWindow(EYE_PREVIEW_WINDOW_NAME, cv2.WINDOW_NORMAL)
        width = max(320, int(round(float(eye_width) * float(scale))))
        height = max(240, int(round(float(eye_height) * float(scale))))
        cv2.resizeWindow(EYE_PREVIEW_WINDOW_NAME, width, height)

        # Probe GUI backend availability early to avoid repeated runtime warnings in the main loop.
        placeholder = np.full((max(eye_height, 120), max(eye_width, 160), 3), 24, dtype=np.uint8)
        cv2.putText(placeholder, "Eye preview is initializing...", (16, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)
        cv2.putText(placeholder, "Please wait for camera frames.", (16, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.imshow(EYE_PREVIEW_WINDOW_NAME, placeholder)
        cv2.waitKey(1)

        visible = cv2.getWindowProperty(EYE_PREVIEW_WINDOW_NAME, cv2.WND_PROP_VISIBLE)
        if visible < 1:
            print("warning: eye preview window is not visible after creation; disabling preview.")
            destroy_window(EYE_PREVIEW_WINDOW_NAME)
            return False
        return True
    except cv2.error as exc:
        print(f"warning: failed to create eye preview window: {exc}")
        destroy_window(EYE_PREVIEW_WINDOW_NAME)
        return False


def poll_preview_key(enabled: bool) -> Optional[str]:
    if not enabled or cv2 is None:
        return None

    key_code = cv2.waitKey(1)
    if key_code < 0:
        return None

    key = chr(key_code & 0xFF)
    return key.lower() if key else None


def format_screen_uv(screen_uv: Optional[tuple[float, float]]) -> str:
    if screen_uv is None:
        return "None"
    return f"({screen_uv[0]:.3f}, {screen_uv[1]:.3f})"


def draw_region_overlay(preview_bgr: np.ndarray, region, color: tuple[int, int, int]) -> None:
    if cv2 is None:
        return

    if region.ellipse is not None:
        center = (int(round(region.ellipse.center_x)), int(round(region.ellipse.center_y)))
        axes = (
            max(1, int(round(region.ellipse.major_axis / 2.0))),
            max(1, int(round(region.ellipse.minor_axis / 2.0))),
        )
        cv2.ellipse(
            preview_bgr,
            center=center,
            axes=axes,
            angle=float(region.ellipse.angle_deg),
            startAngle=0,
            endAngle=360,
            color=color,
            thickness=2,
        )

    if region.center_x is not None and region.center_y is not None:
        center = (int(round(region.center_x)), int(round(region.center_y)))
        cv2.circle(preview_bgr, center, 3, color, -1)


def draw_static_boundary_overlay(preview_bgr: np.ndarray, ellipse: Optional[EllipseResult], color: tuple[int, int, int]) -> None:
    if cv2 is None or ellipse is None:
        return

    center = (int(round(ellipse.center_x)), int(round(ellipse.center_y)))
    axes = (
        max(1, int(round(ellipse.major_axis / 2.0))),
        max(1, int(round(ellipse.minor_axis / 2.0))),
    )
    cv2.ellipse(
        preview_bgr,
        center=center,
        axes=axes,
        angle=float(ellipse.angle_deg),
        startAngle=0,
        endAngle=360,
        color=color,
        thickness=1,
    )
    cv2.circle(preview_bgr, center, 2, color, -1)


def build_eye_preview_frame(
    eye_preview_gray: np.ndarray,
    geometry_result,
    calibration_session: Optional[CalibrationSession],
    static_boundary_ellipse: Optional[EllipseResult],
    screen_uv: Optional[tuple[float, float]],
    tracking_valid: bool,
    feature_valid: bool,
    feature_confidence: float,
    feature_mode: str,
    fps: float,
    inference_ms: Optional[float],
    status_message: str,
    preview_scale: float,
) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("cv2 is unavailable; cannot build preview frame.")

    preview_bgr = cv2.cvtColor(eye_preview_gray, cv2.COLOR_GRAY2BGR)
    if geometry_result is not None:
        draw_region_overlay(preview_bgr, geometry_result.iris_geometry, color=(0, 255, 0))
        draw_region_overlay(preview_bgr, geometry_result.pupil_geometry, color=(0, 165, 255))
        draw_static_boundary_overlay(preview_bgr, static_boundary_ellipse, color=(255, 128, 0))

    if preview_scale != 1.0:
        width = max(1, int(round(preview_bgr.shape[1] * preview_scale)))
        height = max(1, int(round(preview_bgr.shape[0] * preview_scale)))
        interpolation = cv2.INTER_LINEAR if preview_scale > 1.0 else cv2.INTER_AREA
        preview_bgr = cv2.resize(preview_bgr, (width, height), interpolation=interpolation)

    active_point_name = None
    calibration_points = []
    calibrated = False
    if calibration_session is not None:
        calibration_points = calibration_session.points
        current_point = calibration_session.current_point
        active_point_name = None if current_point is None else current_point.name
        calibrated = calibration_session.is_calibrated

    preview_bgr = draw_screen_preview_panel(
        frame=preview_bgr,
        calibration_points=calibration_points,
        screen_uv=screen_uv,
        tracking_valid=tracking_valid,
        calibrated=calibrated,
        active_point_name=active_point_name,
    )

    infer_text = "N/A" if inference_ms is None else f"{inference_ms:.1f} ms"
    truncated_status = status_message if len(status_message) <= 56 else f"{status_message[:53]}..."
    text_lines = [
        f"feature_mode={feature_mode}",
        f"feature_valid={feature_valid} conf={feature_confidence:.3f} tracking_valid={tracking_valid}",
        f"fps={fps:.1f} inference={infer_text}",
        f"static_boundary={static_boundary_ellipse is not None}",
        f"screen_uv={format_screen_uv(screen_uv)}",
        f"status={truncated_status}",
        "keys: s=start x=cancel r=reset q=quit",
    ]

    y = 24
    for line in text_lines:
        cv2.putText(preview_bgr, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(preview_bgr, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
        y += 24

    return preview_bgr


def main() -> None:
    args = parse_args()
    if (args.metadata_udp_host is None) != (args.metadata_udp_port is None):
        raise RuntimeError("metadata_udp_host and metadata_udp_port must be provided together or omitted together.")

    eye_only_mode = bool(args.eye_only_mode)
    eye_preview_enabled = resolve_preview_enabled(eye_only_mode=eye_only_mode, eye_preview=args.eye_preview)
    eye_preview_scale = max(0.5, float(args.eye_preview_scale))

    runtime = PiOnnxSegmentationRuntime(model_path=args.model_path, backend="cpu")

    if runtime.input_width != args.eye_width or runtime.input_height != args.eye_height:
        raise RuntimeError(
            f"eye_width/eye_height must exactly match the model input. model input={runtime.input_width}x{runtime.input_height}, "
            f"current args={args.eye_width}x{args.eye_height}."
        )

    selected_fpv_camera_id = None if eye_only_mode else int(args.fpv_camera_id)
    print_camera_inventory(eye_camera_id=args.eye_camera_id, fpv_camera_id=selected_fpv_camera_id)

    # Initialize HighGUI before Picamera/RTSP threads to reduce GTK/Qt thread-context conflicts.
    eye_preview_enabled = initialize_eye_preview_window(
        enabled=eye_preview_enabled,
        scale=eye_preview_scale,
        eye_width=args.eye_width,
        eye_height=args.eye_height,
    )
    print("eye preview:", eye_preview_enabled)

    eye_worker = PicameraStreamWorker(
        PicameraStreamConfig(
            camera_id=args.eye_camera_id,
            width=args.eye_width,
            height=args.eye_height,
            pixel_format="YUV420",
            frame_rate=args.eye_fps,
            hflip=args.eye_hflip,
            vflip=args.eye_vflip,
        )
    )
    fpv_worker: Optional[PicameraStreamWorker] = None
    if not eye_only_mode:
        fpv_worker = PicameraStreamWorker(
            PicameraStreamConfig(
                camera_id=args.fpv_camera_id,
                width=args.fpv_width,
                height=args.fpv_height,
                pixel_format=args.fpv_pixel_format,
                frame_rate=args.fpv_fps,
                hflip=args.fpv_hflip,
                vflip=args.fpv_vflip,
            )
        )

    eye_worker.start()
    if fpv_worker is not None:
        fpv_worker.start()

    rtsp_server: Optional[RtspVideoServer] = None
    if fpv_worker is not None:
        rtsp_server = RtspVideoServer(
            width=args.fpv_width,
            height=args.fpv_height,
            fps=int(args.fpv_fps),
            host=args.rtsp_host,
            port=args.rtsp_port,
            path=args.rtsp_path,
            bitrate_kbps=args.rtsp_bitrate_kbps,
        )
        rtsp_server.start()

    metadata_publisher = GazeMetadataPublisher(
        stdout_enabled=args.metadata_stdout,
        udp_host=args.metadata_udp_host,
        udp_port=args.metadata_udp_port,
    )

    print("onnx providers:", runtime.providers)
    print("eye input:", f"{runtime.input_width}x{runtime.input_height}")
    if rtsp_server is not None:
        display_host = resolve_display_host(args.rtsp_host)
        display_rtsp_url = f"rtsp://{display_host}:{args.rtsp_port}/{args.rtsp_path.strip('/')}"
        print("rtsp encoder:", rtsp_server.encoder_name)
        print("rtsp url:", display_rtsp_url)
    else:
        print("rtsp:", "disabled (eye-only mode)")
    if args.metadata_udp_host is not None and args.metadata_udp_port is not None:
        print("metadata udp:", f"{args.metadata_udp_host}:{args.metadata_udp_port}")
    print("metadata stdout:", args.metadata_stdout)

    quality_tracker = FeatureQualityTracker(alpha=QUALITY_TRACKER_ALPHA)
    kalman_filter = AdaptiveKalmanFilter2D(
        process_noise=KALMAN_PROCESS_NOISE,
        base_measurement_noise=KALMAN_BASE_MEASUREMENT_NOISE,
        low_confidence_noise=KALMAN_LOW_CONFIDENCE_NOISE,
        jump_noise_scale=KALMAN_JUMP_NOISE_SCALE,
        max_covariance=KALMAN_MAX_COVARIANCE,
    )
    calibration_session: Optional[CalibrationSession] = None
    displayed_screen_uv: Optional[tuple[float, float]]
    last_valid_screen_ts_ms: Optional[float]
    tracking_valid: bool
    displayed_screen_uv, last_valid_screen_ts_ms, tracking_valid = clear_tracking_state(quality_tracker, kalman_filter)

    status_message = "Realtime pipeline started. Press s to begin nine-point calibration."
    last_eye_processed_ts: Optional[int] = None
    last_pushed_fpv_ts: Optional[int] = None
    last_eye_reconnect_ms = 0.0
    last_fpv_reconnect_ms = 0.0
    last_inference_ms: Optional[float] = None
    last_sync_skew_ms: Optional[float] = None
    last_sync_stale = False
    current_feature_valid = False
    current_feature_confidence = 0.0
    fps = 0.0
    prev_processed_time = time.monotonic()
    latest_eye_preview_gray: Optional[np.ndarray] = None
    latest_geometry_result = None
    latest_static_boundary_ellipse: Optional[EllipseResult] = None
    calibration_window_visible = False
    running = True

    try:
        with NonBlockingTerminalReader() as key_reader:
            while running:
                loop_now = time.monotonic()
                now_ms = loop_now * 1000.0
                now_ns = time.monotonic_ns()
                needs_push = False
                needs_metadata_publish = False

                terminal_key = key_reader.poll_key()
                preview_key = poll_preview_key(enabled=eye_preview_enabled)
                key = terminal_key if terminal_key is not None else preview_key
                if key == "q":
                    running = False
                    continue
                if key == "s":
                    calibration_session = begin_calibration_session(
                        now_ms=now_ms,
                        settle_ms=args.calibration_settle_ms,
                        capture_ms=args.calibration_capture_ms,
                        min_valid_frames=args.calibration_min_valid_frames,
                        margin=args.calibration_margin,
                        point_pattern="nine",
                    )
                    displayed_screen_uv, last_valid_screen_ts_ms, tracking_valid = clear_tracking_state(quality_tracker, kalman_filter)
                    current_feature_valid = False
                    current_feature_confidence = 0.0
                    latest_static_boundary_ellipse = None
                    status_message = f"Calibration started: {calibration_session.calibration_step}"
                    needs_push = True
                    needs_metadata_publish = True
                elif key == "x":
                    if calibration_session is not None:
                        calibration_session = cancel_calibration_session(calibration_session, reason="Calibration canceled by user.")
                    displayed_screen_uv, last_valid_screen_ts_ms, tracking_valid = clear_tracking_state(quality_tracker, kalman_filter)
                    current_feature_valid = False
                    current_feature_confidence = 0.0
                    latest_static_boundary_ellipse = None
                    status_message = "Current calibration was canceled."
                    needs_push = True
                    needs_metadata_publish = True
                elif key == "r":
                    calibration_session = None
                    displayed_screen_uv, last_valid_screen_ts_ms, tracking_valid = clear_tracking_state(quality_tracker, kalman_filter)
                    current_feature_valid = False
                    current_feature_confidence = 0.0
                    latest_static_boundary_ellipse = None
                    status_message = "Tracking and calibration have been reset."
                    needs_push = True
                    needs_metadata_publish = True

                last_eye_reconnect_ms, eye_reconnect_error, eye_reconnected = maybe_reconnect_camera(
                    worker=eye_worker,
                    now_ms=now_ms,
                    last_attempt_ms=last_eye_reconnect_ms,
                    reconnect_interval_ms=args.reconnect_interval_ms,
                )
                fpv_reconnect_error = None
                fpv_reconnected = False
                if fpv_worker is not None:
                    last_fpv_reconnect_ms, fpv_reconnect_error, fpv_reconnected = maybe_reconnect_camera(
                        worker=fpv_worker,
                        now_ms=now_ms,
                        last_attempt_ms=last_fpv_reconnect_ms,
                        reconnect_interval_ms=args.reconnect_interval_ms,
                    )

                if eye_reconnected or fpv_reconnected:
                    calibration_session = None
                    displayed_screen_uv, last_valid_screen_ts_ms, tracking_valid = clear_tracking_state(quality_tracker, kalman_filter)
                    current_feature_valid = False
                    current_feature_confidence = 0.0
                    latest_static_boundary_ellipse = None
                    status_message = "Camera reconnected; calibration has been cleared."
                    needs_push = True
                    needs_metadata_publish = True
                elif eye_reconnect_error is not None:
                    status_message = f"eye camera error; waiting to reconnect: {eye_reconnect_error}"
                elif fpv_reconnect_error is not None:
                    status_message = f"fpv camera error; waiting to reconnect: {fpv_reconnect_error}"

                eye_frame = eye_worker.get_latest_frame()
                fpv_frame = None if fpv_worker is None else fpv_worker.get_latest_frame()
                eye_health = eye_worker.get_health()
                eye_stale = is_frame_stale(eye_frame, stale_after_ms=args.camera_stale_ms, now_ns=now_ns)
                fpv_stale = False if fpv_worker is None else is_frame_stale(fpv_frame, stale_after_ms=args.camera_stale_ms, now_ns=now_ns)

                if eye_frame is not None and eye_frame.timestamp_ns != last_eye_processed_ts:
                    last_eye_processed_ts = eye_frame.timestamp_ns
                    if fpv_frame is not None:
                        pair = pair_eye_and_fpv_frames(
                            eye_frame=eye_frame,
                            fpv_frame=fpv_frame,
                            tolerance_ms=args.sync_tolerance_ms,
                        )
                        last_sync_skew_ms = pair.skew_ms
                        last_sync_stale = pair.sync_stale
                    else:
                        last_sync_skew_ms = None
                        last_sync_stale = False

                    eye_preview_gray, input_tensor = prepare_eye_tensor_from_yuv420(
                        eye_frame.frame,
                        input_width=runtime.input_width,
                        input_height=runtime.input_height,
                    )
                    latest_eye_preview_gray = eye_preview_gray
                    infer_start = time.perf_counter()
                    pred_label_map = runtime.predict_label_map(input_tensor)
                    last_inference_ms = (time.perf_counter() - infer_start) * 1000.0

                    geometry_result = extract_geometry_result(pred_label_map)
                    latest_geometry_result = geometry_result
                    calibration_model_for_feature = (
                        calibration_session.model
                        if calibration_session is not None and calibration_session.is_calibrated
                        else None
                    )
                    latest_static_boundary_ellipse = (
                        None if calibration_model_for_feature is None else calibration_model_for_feature.static_boundary_ellipse
                    )

                    feature_x, feature_y, current_feature_valid, feature_reasons = resolve_tracking_features(
                        geometry_result,
                        feature_mode=args.feature_mode,
                        reference_boundary_ellipse=latest_static_boundary_ellipse,
                    )
                    current_feature_confidence = quality_tracker.compute_confidence(
                        geometry_result=geometry_result,
                        feature_mode=args.feature_mode,
                        feature_x=feature_x,
                        feature_y=feature_y,
                        feature_valid=current_feature_valid,
                        feature_reasons=feature_reasons,
                    )
                    smoothed_x, smoothed_y = kalman_filter.update(
                        feature_x,
                        feature_y,
                        current_feature_valid,
                        current_feature_confidence,
                    )

                    delta_time = max(loop_now - prev_processed_time, 1e-6)
                    fps = 1.0 / delta_time
                    prev_processed_time = loop_now

                    if calibration_session is not None and calibration_session.is_active:
                        before_state = calibration_session.state
                        before_step = calibration_session.calibration_step
                        calibration_feature_valid = bool(
                            current_feature_valid
                            and current_feature_confidence >= MIN_CALIBRATION_CONFIDENCE
                            and smoothed_x is not None
                            and smoothed_y is not None
                        )
                        calibration_session = advance_calibration_session(
                            calibration_session,
                            now_ms=now_ms,
                            feature_dx=smoothed_x,
                            feature_dy=smoothed_y,
                            feature_valid=calibration_feature_valid,
                            feature_confidence=current_feature_confidence,
                            boundary_ellipse=geometry_result.outer_boundary_geometry.ellipse,
                        )
                        if calibration_session.state != before_state or calibration_session.calibration_step != before_step:
                            status_message = f"Calibration: {calibration_session.calibration_step} ({calibration_session.state})"
                        if calibration_session.state == "failed":
                            displayed_screen_uv, last_valid_screen_ts_ms, tracking_valid = clear_tracking_state(quality_tracker, kalman_filter)
                            current_feature_valid = False
                            current_feature_confidence = 0.0
                            latest_static_boundary_ellipse = None
                            status_message = calibration_session.failure_reason or "Calibration failed."
                        elif calibration_session.state == "completed":
                            status_message = "Nine-point calibration completed."

                    if (
                        calibration_model_for_feature is not None
                        and current_feature_valid
                        and current_feature_confidence >= MIN_TRACKING_CONFIDENCE
                        and smoothed_x is not None
                        and smoothed_y is not None
                    ):
                        predicted_screen_uv = predict_screen_point(
                            calibration_model_for_feature,
                            feature_dx=smoothed_x,
                            feature_dy=smoothed_y,
                            clamp=True,
                        )
                        if predicted_screen_uv is not None:
                            displayed_screen_uv = predicted_screen_uv
                            last_valid_screen_ts_ms = now_ms
                            tracking_valid = True
                        else:
                            tracking_valid = False
                    else:
                        tracking_valid = False

                    needs_push = True
                    needs_metadata_publish = True

                if not current_feature_valid or current_feature_confidence < MIN_TRACKING_CONFIDENCE:
                    tracking_valid = False

                if last_valid_screen_ts_ms is not None and now_ms - last_valid_screen_ts_ms > STALE_SCREEN_POINT_TIMEOUT_MS:
                    displayed_screen_uv = None

                if eye_stale:
                    tracking_valid = False
                    if last_valid_screen_ts_ms is None or now_ms - last_valid_screen_ts_ms > STALE_SCREEN_POINT_TIMEOUT_MS:
                        displayed_screen_uv = None
                    if eye_health.last_error is None:
                        status_message = "eye camera frame is stale."
                    needs_metadata_publish = True

                if needs_metadata_publish and eye_frame is not None:
                    metadata_packet = build_gaze_metadata_packet(
                        eye_timestamp_ns=eye_frame.timestamp_ns,
                        fpv_timestamp_ns=None if fpv_frame is None else fpv_frame.timestamp_ns,
                        screen_uv=displayed_screen_uv,
                        tracking_valid=tracking_valid,
                        feature_valid=current_feature_valid,
                        feature_mode=args.feature_mode,
                        calibration_session=calibration_session,
                        sync_skew_ms=last_sync_skew_ms,
                        sync_stale=last_sync_stale,
                        eye_stale=eye_stale,
                        fpv_stale=fpv_stale,
                        fps=fps,
                        inference_ms=last_inference_ms,
                        status_message=status_message,
                    )
                    metadata_publisher.publish(metadata_packet)

                if rtsp_server is not None and fpv_frame is not None and (fpv_frame.timestamp_ns != last_pushed_fpv_ts or needs_push):
                    base_frame = convert_fpv_frame_to_bgr(fpv_frame.frame, pixel_format=args.fpv_pixel_format)
                    rtsp_server.push_frame(base_frame)
                    last_pushed_fpv_ts = fpv_frame.timestamp_ns

                if eye_preview_enabled and cv2 is not None and latest_eye_preview_gray is not None:
                    try:
                        preview_frame = build_eye_preview_frame(
                            eye_preview_gray=latest_eye_preview_gray,
                            geometry_result=latest_geometry_result,
                            calibration_session=calibration_session,
                            static_boundary_ellipse=latest_static_boundary_ellipse,
                            screen_uv=displayed_screen_uv,
                            tracking_valid=tracking_valid,
                            feature_valid=current_feature_valid,
                            feature_confidence=current_feature_confidence,
                            feature_mode=args.feature_mode,
                            fps=fps,
                            inference_ms=last_inference_ms,
                            status_message=status_message,
                            preview_scale=eye_preview_scale,
                        )
                        cv2.imshow(EYE_PREVIEW_WINDOW_NAME, preview_frame)

                        if calibration_session is not None and calibration_session.is_active:
                            calibration_canvas = build_calibration_canvas(
                                size=(DEFAULT_CALIBRATION_WINDOW_WIDTH, DEFAULT_CALIBRATION_WINDOW_HEIGHT),
                                session=calibration_session,
                            )
                            cv2.imshow(CALIBRATION_WINDOW_NAME, calibration_canvas)
                            calibration_window_visible = True
                        elif calibration_window_visible:
                            destroy_window(CALIBRATION_WINDOW_NAME)
                            calibration_window_visible = False

                        if cv2.getWindowProperty(EYE_PREVIEW_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                            eye_preview_enabled = False
                            destroy_window(EYE_PREVIEW_WINDOW_NAME)
                            if calibration_window_visible:
                                destroy_window(CALIBRATION_WINDOW_NAME)
                                calibration_window_visible = False
                            status_message = "eye preview window closed."
                    except cv2.error as exc:
                        print(f"warning: eye preview window error; disabled: {exc}")
                        eye_preview_enabled = False
                        destroy_window(EYE_PREVIEW_WINDOW_NAME)
                        if calibration_window_visible:
                            destroy_window(CALIBRATION_WINDOW_NAME)
                            calibration_window_visible = False
                        status_message = "eye preview window unavailable; disabled automatically."

                time.sleep(0.005)
    finally:
        if fpv_worker is not None:
            fpv_worker.stop()
        eye_worker.stop()
        if rtsp_server is not None:
            rtsp_server.stop()
        if cv2 is not None:
            destroy_window(EYE_PREVIEW_WINDOW_NAME)
            destroy_window(CALIBRATION_WINDOW_NAME)
        metadata_publisher.close()


if __name__ == "__main__":
    main()
