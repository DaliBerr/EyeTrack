"""Raspberry Pi realtime runtime helpers."""

from eyetrack.raspi.camera import (
    PicameraStreamConfig,
    PicameraStreamWorker,
    convert_fpv_frame_to_bgr,
    discover_picamera_cameras,
    prepare_eye_tensor_from_yuv420,
)
from eyetrack.raspi.metadata import (
    GazeMetadataPacket,
    GazeMetadataPublisher,
    build_gaze_metadata_packet,
)
from eyetrack.raspi.overlay import OverlayDebugState, compose_fpv_overlay
from eyetrack.raspi.rtsp import RtspVideoServer
from eyetrack.raspi.runtime import PiOnnxSegmentationRuntime
from eyetrack.raspi.sync import (
    compute_frame_age_ms,
    is_frame_stale,
    pair_eye_and_fpv_frames,
    resolve_frame_timestamp_ns,
)
from eyetrack.raspi.terminal import NonBlockingTerminalReader
from eyetrack.raspi.types import CameraDeviceInfo, CameraFrame, FramePair

__all__ = [
    "CameraDeviceInfo",
    "CameraFrame",
    "FramePair",
    "GazeMetadataPacket",
    "GazeMetadataPublisher",
    "NonBlockingTerminalReader",
    "OverlayDebugState",
    "PiOnnxSegmentationRuntime",
    "PicameraStreamConfig",
    "PicameraStreamWorker",
    "RtspVideoServer",
    "build_gaze_metadata_packet",
    "compose_fpv_overlay",
    "compute_frame_age_ms",
    "convert_fpv_frame_to_bgr",
    "discover_picamera_cameras",
    "is_frame_stale",
    "pair_eye_and_fpv_frames",
    "prepare_eye_tensor_from_yuv420",
    "resolve_frame_timestamp_ns",
]
