from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from eyetrack.gaze import CalibrationSession


@dataclass(frozen=True)
class OverlayDebugState:
    fps: float
    inference_ms: Optional[float]
    feature_valid: bool
    calibration_state: str
    calibration_step: str
    tracking_valid: bool
    sync_skew_ms: Optional[float]
    sync_stale: bool
    eye_stale: bool
    fpv_stale: bool
    eye_camera_error: Optional[str]
    fpv_camera_error: Optional[str]
    feature_mode: str
    status_message: str
    rtsp_url: str


def _map_uv_to_frame(u: float, v: float, width: int, height: int) -> tuple[int, int]:
    px = int(round(np.clip(u, 0.0, 1.0) * max(width - 1, 0)))
    py = int(round(np.clip(v, 0.0, 1.0) * max(height - 1, 0)))
    return px, py


def draw_gaze_point(
    frame: np.ndarray,
    screen_uv: Optional[tuple[float, float]],
    tracking_valid: bool,
) -> np.ndarray:
    if screen_uv is None:
        return frame

    height, width = frame.shape[:2]
    px, py = _map_uv_to_frame(screen_uv[0], screen_uv[1], width=width, height=height)
    color = (0, 255, 0) if tracking_valid else (0, 165, 255)
    cv2.circle(frame, (px, py), 14, color, 2)
    cv2.circle(frame, (px, py), 3, color, -1)
    cv2.line(frame, (px - 20, py), (px + 20, py), color, 1)
    cv2.line(frame, (px, py - 20), (px, py + 20), color, 1)
    return frame


def draw_calibration_target(frame: np.ndarray, calibration_session: Optional[CalibrationSession]) -> np.ndarray:
    if calibration_session is None or not calibration_session.is_active:
        return frame

    point = calibration_session.current_point
    if point is None:
        return frame

    height, width = frame.shape[:2]
    px, py = _map_uv_to_frame(point.u, point.v, width=width, height=height)
    color = (0, 255, 255) if calibration_session.state == "settling" else (0, 255, 0)
    cv2.circle(frame, (px, py), 26, color, 3)
    cv2.circle(frame, (px, py), 4, color, -1)
    cv2.line(frame, (px - 40, py), (px + 40, py), color, 1)
    cv2.line(frame, (px, py - 40), (px, py + 40), color, 1)
    return frame


def draw_debug_panel(frame: np.ndarray, debug_state: OverlayDebugState, calibration_session: Optional[CalibrationSession]) -> np.ndarray:
    overlay = frame.copy()
    panel_width = min(430, max(300, frame.shape[1] // 3))
    panel_height = 240
    x0, y0 = 12, 12
    x1, y1 = x0 + panel_width, y0 + panel_height

    cv2.rectangle(overlay, (x0, y0), (x1, y1), (15, 15, 15), -1)
    frame[:] = cv2.addWeighted(overlay, 0.68, frame, 0.32, 0.0)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (220, 220, 220), 1)

    inference_text = "None" if debug_state.inference_ms is None else f"{debug_state.inference_ms:.1f}"
    skew_text = "None" if debug_state.sync_skew_ms is None else f"{debug_state.sync_skew_ms:.1f}"

    lines = [
        f"RTSP: {debug_state.rtsp_url}",
        f"FPS: {debug_state.fps:.1f}  inference_ms: {inference_text}",
        f"feature_mode: {debug_state.feature_mode}  feature_valid: {debug_state.feature_valid}",
        f"calibration_state: {debug_state.calibration_state}  step: {debug_state.calibration_step}",
        f"tracking_valid: {debug_state.tracking_valid}",
        f"sync_skew_ms: {skew_text}  sync_stale: {debug_state.sync_stale}",
        f"eye_stale: {debug_state.eye_stale}  fpv_stale: {debug_state.fpv_stale}",
        f"eye_error: {debug_state.eye_camera_error or 'none'}",
        f"fpv_error: {debug_state.fpv_camera_error or 'none'}",
        f"message: {debug_state.status_message}",
        "keys: s calibrate | x cancel | r reset | q quit",
    ]

    if calibration_session is not None and calibration_session.is_active:
        lines.append(
            f"capture_samples: {len(calibration_session.current_point_samples)}/{calibration_session.min_valid_frames}"
        )

    y = y0 + 24
    for line in lines:
        cv2.putText(frame, line, (x0 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        y += 20

    return frame


def compose_fpv_overlay(
    fpv_frame_bgr: np.ndarray,
    screen_uv: Optional[tuple[float, float]],
    tracking_valid: bool,
    calibration_session: Optional[CalibrationSession],
    debug_state: OverlayDebugState,
) -> np.ndarray:
    output = fpv_frame_bgr.copy()
    output = draw_gaze_point(output, screen_uv=screen_uv, tracking_valid=tracking_valid)
    output = draw_calibration_target(output, calibration_session=calibration_session)
    output = draw_debug_panel(output, debug_state=debug_state, calibration_session=calibration_session)
    return output
