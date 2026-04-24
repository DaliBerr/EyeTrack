from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from eyetrack.gaze import CalibrationPoint, CalibrationSession


def format_optional_pair(pair: Optional[Tuple[float, float]]) -> str:
    if pair is None:
        return "None"
    return f"({pair[0]:.3f}, {pair[1]:.3f})"


def ensure_fullscreen_window(window_name: str) -> None:
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def destroy_window(window_name: str) -> None:
    try:
        cv2.destroyWindow(window_name)
    except cv2.error:
        pass


def build_calibration_canvas(size: Tuple[int, int], session: CalibrationSession) -> np.ndarray:
    width, height = size
    width = max(width, 640)
    height = max(height, 480)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    point = session.current_point
    if point is None:
        return canvas

    px = int(round(point.u * (width - 1)))
    py = int(round(point.v * (height - 1)))
    color = (0, 255, 255) if session.state == "settling" else (0, 255, 0)

    cv2.circle(canvas, (px, py), 26, color, 3)
    cv2.circle(canvas, (px, py), 4, color, -1)
    cv2.line(canvas, (px - 40, py), (px + 40, py), color, 1)
    cv2.line(canvas, (px, py - 40), (px, py + 40), color, 1)

    lines = [
        f"{len(session.points)}-point calibration",
        f"point: {session.calibration_step}",
        f"phase: {session.state}",
        f"valid samples: {len(session.current_point_samples)}/{session.min_valid_frames}",
        "Keep your gaze on the target",
        "Press x to cancel",
    ]

    y = height - 150
    for line in lines:
        cv2.putText(canvas, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        y += 34

    return canvas


def draw_screen_preview_panel(
    frame: np.ndarray,
    calibration_points: list[CalibrationPoint],
    screen_uv: Optional[Tuple[float, float]],
    tracking_valid: bool,
    calibrated: bool,
    active_point_name: Optional[str],
    panel_size: int = 220,
) -> np.ndarray:
    if not calibrated and active_point_name is None:
        return frame

    output = frame.copy()
    frame_height, frame_width = output.shape[:2]
    x0 = frame_width - panel_size - 10
    y0 = 120
    x1 = x0 + panel_size
    y1 = y0 + panel_size

    overlay = output.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (20, 20, 20), -1)
    output = cv2.addWeighted(overlay, 0.7, output, 0.3, 0.0)
    cv2.rectangle(output, (x0, y0), (x1, y1), (255, 255, 255), 1)
    cv2.putText(output, "Screen Preview", (x0 + 12, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    pad = 24
    left = x0 + pad
    top = y0 + pad + 10
    right = x1 - pad
    bottom = y1 - pad
    cv2.rectangle(output, (left, top), (right, bottom), (140, 140, 140), 1)

    for point in calibration_points:
        px = int(round(left + point.u * (right - left)))
        py = int(round(top + point.v * (bottom - top)))
        color = (180, 180, 180)
        radius = 5
        if point.name == active_point_name:
            color = (0, 255, 255)
            radius = 8
        elif calibrated:
            color = (100, 200, 100)
        cv2.circle(output, (px, py), radius, color, -1)

    if screen_uv is not None:
        gaze_x = int(round(left + screen_uv[0] * (right - left)))
        gaze_y = int(round(top + screen_uv[1] * (bottom - top)))
        gaze_color = (0, 255, 0) if tracking_valid else (0, 165, 255)
        cv2.circle(output, (gaze_x, gaze_y), 9, gaze_color, 2)
        cv2.circle(output, (gaze_x, gaze_y), 2, gaze_color, -1)

    status_text = "tracking" if tracking_valid else ("calibrated" if calibrated else "calibrating")
    cv2.putText(output, status_text, (x0 + 12, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return output
