from __future__ import annotations

import time
from typing import Any, Optional

from eyetrack.raspi.types import CameraFrame, FramePair


def resolve_frame_timestamp_ns(metadata: Optional[dict[str, Any]], fallback_monotonic_ns: Optional[int] = None) -> int:
    if fallback_monotonic_ns is None:
        fallback_monotonic_ns = time.monotonic_ns()

    if metadata is None:
        return int(fallback_monotonic_ns)

    for key in ("SensorTimestamp", "FrameTimestamp", "Timestamp"):
        value = metadata.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return int(value)

    return int(fallback_monotonic_ns)


def pair_eye_and_fpv_frames(
    eye_frame: CameraFrame,
    fpv_frame: CameraFrame,
    tolerance_ms: float,
) -> FramePair:
    skew_ms = abs(fpv_frame.timestamp_ns - eye_frame.timestamp_ns) / 1_000_000.0
    return FramePair(
        eye_frame=eye_frame,
        fpv_frame=fpv_frame,
        skew_ms=float(skew_ms),
        sync_stale=bool(skew_ms > float(tolerance_ms)),
        tolerance_ms=float(tolerance_ms),
    )


def compute_frame_age_ms(frame: Optional[CameraFrame], now_ns: Optional[int] = None) -> Optional[float]:
    if frame is None:
        return None

    if now_ns is None:
        now_ns = time.monotonic_ns()

    return max(0.0, float(now_ns - frame.captured_monotonic_ns) / 1_000_000.0)


def is_frame_stale(frame: Optional[CameraFrame], stale_after_ms: float, now_ns: Optional[int] = None) -> bool:
    age_ms = compute_frame_age_ms(frame=frame, now_ns=now_ns)
    if age_ms is None:
        return True
    return bool(age_ms > float(stale_after_ms))
