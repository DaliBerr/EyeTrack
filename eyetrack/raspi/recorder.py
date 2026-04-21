from __future__ import annotations

import os
import threading
from datetime import datetime
from typing import Optional

import numpy as np

from eyetrack.raspi.rtsp import build_h264_encoder_fragment, require_gstreamer, resolve_h264_encoder_name


def _quote_gst_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def resolve_recording_h264_encoder_name(Gst) -> str:
    try:
        return resolve_h264_encoder_name(Gst, allow_software_fallback=False)
    except RuntimeError as exc:
        if Gst.ElementFactory.find("x264enc") is not None:
            raise RuntimeError(
                "record mode requires a Raspberry Pi hardware H.264 encoder; x264enc software fallback is disabled."
            ) from exc
        raise RuntimeError(
            "record mode requires a Raspberry Pi hardware H.264 encoder (expected one of v4l2h264enc, v4l2slh264enc, omxh264enc)."
        ) from exc


def build_recording_pipeline_launch(
    width: int,
    height: int,
    fps: int,
    encoder_fragment: str,
    output_path: str,
) -> str:
    caps = f"video/x-raw,format=BGR,width={int(width)},height={int(height)},framerate={max(int(fps), 1)}/1"
    quoted_output_path = _quote_gst_string(output_path)
    return (
        f"appsrc name=src is-live=true block=false format=time do-timestamp=true caps={caps} "
        f"! queue leaky=downstream max-size-buffers=2 "
        f"! videoconvert ! {encoder_fragment} ! matroskamux ! filesink location={quoted_output_path} sync=false async=false"
    )


def build_recording_output_path(record_dir: str, now: Optional[datetime] = None) -> str:
    timestamp = (datetime.now() if now is None else now).strftime("%Y%m%d_%H%M%S")
    candidate = os.path.abspath(os.path.join(record_dir, f"fpv_gaze_{timestamp}.mkv"))
    if not os.path.exists(candidate):
        return candidate

    suffix = 1
    while True:
        candidate = os.path.abspath(os.path.join(record_dir, f"fpv_gaze_{timestamp}_{suffix:02d}.mkv"))
        if not os.path.exists(candidate):
            return candidate
        suffix += 1


class FpvVideoRecorder:
    def __init__(
        self,
        width: int,
        height: int,
        fps: int,
        output_dir: str,
        bitrate_kbps: int = 4000,
    ):
        self.width = int(width)
        self.height = int(height)
        self.fps = max(int(fps), 1)
        self.output_dir = os.path.abspath(output_dir)
        self.bitrate_kbps = int(bitrate_kbps)

        self.Gst = None
        self.encoder_name: Optional[str] = None
        self.output_path: Optional[str] = None
        self.launch: Optional[str] = None
        self._pipeline = None
        self._appsrc = None
        self._bus = None
        self._push_lock = threading.Lock()
        self._pts_ns = 0
        self._frame_duration_ns = int(1_000_000_000 / self.fps)

    @property
    def is_recording(self) -> bool:
        return self._pipeline is not None and self._appsrc is not None and self.output_path is not None

    def start(self) -> str:
        if self.is_recording and self.output_path is not None:
            return self.output_path

        self.Gst = require_gstreamer()[1]
        self.encoder_name = resolve_recording_h264_encoder_name(self.Gst)
        encoder_fragment = build_h264_encoder_fragment(
            encoder_name=self.encoder_name,
            fps=self.fps,
            bitrate_kbps=self.bitrate_kbps,
        )

        os.makedirs(self.output_dir, exist_ok=True)
        self.output_path = build_recording_output_path(self.output_dir)
        self.launch = build_recording_pipeline_launch(
            width=self.width,
            height=self.height,
            fps=self.fps,
            encoder_fragment=encoder_fragment,
            output_path=self.output_path,
        )

        pipeline = self.Gst.parse_launch(self.launch)
        appsrc = pipeline.get_by_name("src")
        if appsrc is None:
            pipeline.set_state(self.Gst.State.NULL)
            self._clear_pipeline_state()
            raise RuntimeError("record pipeline is missing appsrc.")

        appsrc.set_property("format", self.Gst.Format.TIME)
        appsrc.set_property("is-live", True)
        appsrc.set_property("block", False)

        bus = pipeline.get_bus()
        state_result = pipeline.set_state(self.Gst.State.PLAYING)
        if state_result == self.Gst.StateChangeReturn.FAILURE:
            pipeline.set_state(self.Gst.State.NULL)
            self._clear_pipeline_state()
            raise RuntimeError(f"failed to start record pipeline with encoder {self.encoder_name}.")

        self._pipeline = pipeline
        self._appsrc = appsrc
        self._bus = bus
        self._pts_ns = 0
        return self.output_path

    def stop(self) -> Optional[str]:
        pipeline = self._pipeline
        appsrc = self._appsrc
        bus = self._bus
        output_path = self.output_path
        if pipeline is None or appsrc is None:
            self._clear_pipeline_state()
            return output_path

        error_message: Optional[str] = None
        try:
            with self._push_lock:
                appsrc.emit("end-of-stream")

            if bus is not None:
                message = bus.timed_pop_filtered(
                    5 * self.Gst.SECOND,
                    self.Gst.MessageType.EOS | self.Gst.MessageType.ERROR,
                )
                if message is None:
                    error_message = "timed out while finalizing MKV recording."
                elif message.type == self.Gst.MessageType.ERROR:
                    err, debug = message.parse_error()
                    error_message = str(err if debug is None else f"{err} ({debug})")
        finally:
            pipeline.set_state(self.Gst.State.NULL)
            self._clear_pipeline_state()

        if error_message is not None:
            raise RuntimeError(error_message)
        return output_path

    def push_frame(self, frame_bgr: np.ndarray) -> bool:
        if not self.is_recording or self._appsrc is None:
            return False
        if frame_bgr.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"record output frame size mismatch: expected {(self.height, self.width)}, got {frame_bgr.shape[:2]}"
            )

        with self._push_lock:
            if self._appsrc is None or not self.is_recording:
                return False

            frame = np.ascontiguousarray(frame_bgr)
            buffer = self.Gst.Buffer.new_allocate(None, frame.nbytes, None)
            buffer.fill(0, frame.tobytes())
            buffer.pts = self._pts_ns
            buffer.dts = self._pts_ns
            buffer.duration = self._frame_duration_ns
            self._pts_ns += self._frame_duration_ns
            result = self._appsrc.emit("push-buffer", buffer)
            return result == self.Gst.FlowReturn.OK

    def _clear_pipeline_state(self) -> None:
        self._pipeline = None
        self._appsrc = None
        self._bus = None
        self.output_path = None
        self._pts_ns = 0
