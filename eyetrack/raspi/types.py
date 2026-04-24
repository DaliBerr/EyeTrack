from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class CameraDeviceInfo:
    camera_id: int
    model: str
    location: str
    rotation: str
    identifier: str


@dataclass(frozen=True)
class CameraFrame:
    camera_id: int
    frame: np.ndarray
    timestamp_ns: int
    captured_monotonic_ns: int
    frame_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FramePair:
    eye_frame: CameraFrame
    fpv_frame: CameraFrame
    skew_ms: float
    sync_stale: bool
    tolerance_ms: float


@dataclass(frozen=True)
class CameraHealth:
    running: bool
    last_error: Optional[str]
    latest_timestamp_ns: Optional[int]
