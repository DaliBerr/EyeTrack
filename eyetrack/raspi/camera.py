from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

from eyetrack.raspi.sync import resolve_frame_timestamp_ns
from eyetrack.raspi.types import CameraDeviceInfo, CameraFrame, CameraHealth


def require_picamera2():
    try:
        from picamera2 import Picamera2
    except ImportError as exc:
        raise RuntimeError(
            "缺少 picamera2。请在树莓派上安装 python3-picamera2 后再运行树莓派实时脚本。"
        ) from exc

    return Picamera2


def require_libcamera_transform():
    try:
        from libcamera import Transform
    except ImportError as exc:
        raise RuntimeError("缺少 libcamera Python 绑定，无法配置相机翻转。") from exc

    return Transform


@dataclass(frozen=True)
class PicameraStreamConfig:
    camera_id: int
    width: int
    height: int
    pixel_format: str
    frame_rate: float = 30.0
    hflip: bool = False
    vflip: bool = False
    buffer_count: int = 4
    stream_name: str = "main"


def discover_picamera_cameras() -> list[CameraDeviceInfo]:
    Picamera2 = require_picamera2()
    raw_infos = Picamera2.global_camera_info()
    devices: list[CameraDeviceInfo] = []
    for index, info in enumerate(raw_infos):
        devices.append(
            CameraDeviceInfo(
                camera_id=index,
                model=str(info.get("Model", "unknown")),
                location=str(info.get("Location", "unknown")),
                rotation=str(info.get("Rotation", "unknown")),
                identifier=str(info.get("Id", "unknown")),
            )
        )
    return devices


def extract_y_plane_from_yuv420(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    if frame.ndim != 2:
        raise ValueError(f"YUV420 输入应为二维数组，实际 shape={frame.shape}")
    if frame.shape[1] != width or frame.shape[0] < height:
        raise ValueError(f"YUV420 输入尺寸异常，期望至少 {(height, width)}，实际 {frame.shape}")
    return frame[:height, :width]


def prepare_eye_tensor_from_yuv420(
    frame: np.ndarray,
    input_width: int,
    input_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    y_plane = extract_y_plane_from_yuv420(frame=frame, height=input_height, width=input_width)
    if y_plane.shape != (input_height, input_width):
        raise ValueError(
            f"眼动输入尺寸与模型输入不匹配，当前 {y_plane.shape}，模型需要 {(input_height, input_width)}。"
        )

    preview = np.ascontiguousarray(y_plane)
    tensor = (preview.astype(np.float32) / 255.0)[None, None, :, :]
    return preview, tensor


def convert_fpv_frame_to_bgr(frame: np.ndarray, pixel_format: str) -> np.ndarray:
    normalized = pixel_format.strip().upper()

    if normalized == "BGR888":
        return np.ascontiguousarray(frame)

    if normalized == "RGB888":
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if normalized in {"XBGR8888", "ABGR8888", "RGBA8888"}:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    if normalized in {"XRGB8888", "ARGB8888", "BGRA8888"}:
        return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

    if normalized == "YUV420":
        return cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)

    raise ValueError(f"不支持的 FPV 像素格式: {pixel_format}")


class PicameraStreamWorker:
    def __init__(self, config: PicameraStreamConfig):
        self.config = config
        self._camera: Any = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest_frame: Optional[CameraFrame] = None
        self._last_error: Optional[str] = None
        self._frame_index = 0
        self._running = False

    def _open_camera(self) -> Any:
        Picamera2 = require_picamera2()
        Transform = require_libcamera_transform()

        camera = Picamera2(self.config.camera_id)
        video_config = camera.create_video_configuration(
            main={
                "size": (self.config.width, self.config.height),
                "format": self.config.pixel_format,
            },
            buffer_count=int(self.config.buffer_count),
            controls={"FrameRate": float(self.config.frame_rate)},
            transform=Transform(hflip=bool(self.config.hflip), vflip=bool(self.config.vflip)),
        )
        camera.configure(video_config)
        camera.start()
        return camera

    def _close_camera(self) -> None:
        camera = self._camera
        self._camera = None
        if camera is None:
            return

        try:
            camera.stop()
        except Exception:
            pass

        close_fn = getattr(camera, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._last_error = None
        self._latest_frame = None
        self._camera = self._open_camera()
        self._thread = threading.Thread(target=self._capture_loop, name=f"picam-{self.config.camera_id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        self._running = False
        self._close_camera()

    def restart(self) -> None:
        self.stop()
        self.start()

    def get_latest_frame(self) -> Optional[CameraFrame]:
        with self._lock:
            return self._latest_frame

    def get_health(self) -> CameraHealth:
        with self._lock:
            latest_timestamp_ns = None if self._latest_frame is None else self._latest_frame.timestamp_ns
            return CameraHealth(
                running=bool(self._running),
                last_error=self._last_error,
                latest_timestamp_ns=latest_timestamp_ns,
            )

    def _capture_loop(self) -> None:
        self._running = True
        try:
            while not self._stop_event.is_set():
                if self._camera is None:
                    break

                capture_context = getattr(self._camera, "captured_request", None)
                if not callable(capture_context):
                    raise RuntimeError("当前 picamera2 版本缺少 captured_request()，无法安全同步图像和 metadata。")

                with capture_context() as request:
                    array = request.make_array(self.config.stream_name)
                    metadata = dict(request.get_metadata())

                captured_monotonic_ns = time.monotonic_ns()
                timestamp_ns = resolve_frame_timestamp_ns(metadata=metadata, fallback_monotonic_ns=captured_monotonic_ns)
                frame = CameraFrame(
                    camera_id=self.config.camera_id,
                    frame=np.ascontiguousarray(array),
                    timestamp_ns=timestamp_ns,
                    captured_monotonic_ns=captured_monotonic_ns,
                    frame_index=self._frame_index,
                    metadata=metadata,
                )
                self._frame_index += 1

                with self._lock:
                    self._latest_frame = frame
                    self._last_error = None
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
        finally:
            self._running = False
            self._close_camera()
