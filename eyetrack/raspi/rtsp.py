from __future__ import annotations

import threading
from typing import Optional

import numpy as np


def require_gstreamer():
    try:
        import gi

        gi.require_version("Gst", "1.0")
        gi.require_version("GstRtspServer", "1.0")
        from gi.repository import GLib, Gst, GstRtspServer
    except ImportError as exc:
        raise RuntimeError(
            "缺少 GStreamer Python 绑定。请在树莓派上安装 python3-gi、gir1.2-gst-rtsp-server-1.0 与相关 gstreamer 插件。"
        ) from exc

    Gst.init(None)
    return GLib, Gst, GstRtspServer


def resolve_h264_encoder_name(Gst, allow_software_fallback: bool = True) -> str:
    encoder_candidates = ["v4l2h264enc", "v4l2slh264enc", "omxh264enc"]
    if allow_software_fallback:
        encoder_candidates.append("x264enc")

    for encoder_name in encoder_candidates:
        if Gst.ElementFactory.find(encoder_name) is not None:
            return encoder_name
    raise RuntimeError("未找到可用的 H.264 编码器，请安装对应的 GStreamer encoder 插件。")


def build_h264_encoder_fragment(encoder_name: str, fps: int, bitrate_kbps: int) -> str:
    if encoder_name == "x264enc":
        return (
            f"x264enc tune=zerolatency speed-preset=ultrafast bitrate={int(bitrate_kbps)} "
            f"key-int-max={max(int(fps), 1)} bframes=0 cabac=false sliced-threads=true threads=2 byte-stream=true "
            f"! h264parse"
        )
    return f"{encoder_name} ! h264parse"


class RtspVideoServer:
    def __init__(
        self,
        width: int,
        height: int,
        fps: int,
        host: str,
        port: int,
        path: str,
        bitrate_kbps: int = 4000,
    ):
        self.width = int(width)
        self.height = int(height)
        self.fps = max(int(fps), 1)
        self.host = str(host)
        self.port = int(port)
        self.path = str(path).strip("/")
        self.bitrate_kbps = int(bitrate_kbps)

        self.GLib, self.Gst, self.GstRtspServer = require_gstreamer()
        encoder_name = resolve_h264_encoder_name(self.Gst)
        encoder_fragment = build_h264_encoder_fragment(
            encoder_name=encoder_name,
            fps=self.fps,
            bitrate_kbps=self.bitrate_kbps,
        )
        self.encoder_name = encoder_name
        self.url = f"rtsp://{self.host}:{self.port}/{self.path}"
        self._appsrc = None
        self._push_lock = threading.Lock()
        self._pts_ns = 0
        self._frame_duration_ns = int(1_000_000_000 / self.fps)

        caps = f"video/x-raw,format=BGR,width={self.width},height={self.height},framerate={self.fps}/1"
        launch = (
            f"( appsrc name=src is-live=true block=false format=time do-timestamp=true caps={caps} "
            f"! queue leaky=downstream max-size-buffers=2 "
            f"! videoconvert ! {encoder_fragment} ! rtph264pay name=pay0 pt=96 config-interval=1 )"
        )

        self.server = self.GstRtspServer.RTSPServer.new()
        if hasattr(self.server, "set_address"):
            self.server.set_address(self.host)
        self.server.set_service(str(self.port))

        mounts = self.server.get_mount_points()
        factory = self.GstRtspServer.RTSPMediaFactory.new()
        factory.set_launch(launch)
        factory.set_shared(True)
        factory.connect("media-configure", self._on_media_configure)
        mounts.add_factory(f"/{self.path}", factory)
        self.factory = factory

        self.loop = self.GLib.MainLoop()
        self._loop_thread: Optional[threading.Thread] = None

    def _on_media_configure(self, factory, media) -> None:
        del factory
        element = media.get_element()
        appsrc = element.get_child_by_name("src")
        if appsrc is None:
            raise RuntimeError("RTSP pipeline 中未找到 appsrc。")
        appsrc.set_property("format", self.Gst.Format.TIME)
        appsrc.set_property("is-live", True)
        appsrc.set_property("block", False)
        self._appsrc = appsrc

    def start(self) -> None:
        if self._loop_thread is not None and self._loop_thread.is_alive():
            return
        self.server.attach(None)
        self._loop_thread = threading.Thread(target=self.loop.run, name="rtsp-server", daemon=True)
        self._loop_thread.start()

    def stop(self) -> None:
        if self.loop.is_running():
            self.loop.quit()
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=2.0)
        self._loop_thread = None
        self._appsrc = None

    def push_frame(self, frame_bgr: np.ndarray) -> bool:
        if self._appsrc is None:
            return False
        if frame_bgr.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"RTSP 输出帧尺寸不匹配，期望 {(self.height, self.width)}，实际 {frame_bgr.shape[:2]}"
            )

        with self._push_lock:
            if self._appsrc is None:
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
