from __future__ import annotations

import json
import socket
import sys
from dataclasses import asdict, dataclass
from typing import Optional

from eyetrack.gaze import CalibrationSession


@dataclass(frozen=True)
class GazeMetadataPacket:
    eye_timestamp_ns: Optional[int]
    fpv_timestamp_ns: Optional[int]
    screen_uv: Optional[tuple[float, float]]
    tracking_valid: bool
    feature_valid: bool
    feature_mode: str
    calibration_state: str
    calibration_step: str
    calibrated: bool
    calibration_target_uv: Optional[tuple[float, float]]
    sync_skew_ms: Optional[float]
    sync_stale: bool
    eye_stale: bool
    fpv_stale: bool
    fps: float
    inference_ms: Optional[float]
    status_message: str

    def to_dict(self) -> dict:
        payload = asdict(self)
        if self.screen_uv is not None:
            payload["screen_uv"] = [float(self.screen_uv[0]), float(self.screen_uv[1])]
        if self.calibration_target_uv is not None:
            payload["calibration_target_uv"] = [
                float(self.calibration_target_uv[0]),
                float(self.calibration_target_uv[1]),
            ]
        return payload

    def to_json_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=True, separators=(",", ":"))


def build_gaze_metadata_packet(
    eye_timestamp_ns: Optional[int],
    fpv_timestamp_ns: Optional[int],
    screen_uv: Optional[tuple[float, float]],
    tracking_valid: bool,
    feature_valid: bool,
    feature_mode: str,
    calibration_session: Optional[CalibrationSession],
    sync_skew_ms: Optional[float],
    sync_stale: bool,
    eye_stale: bool,
    fpv_stale: bool,
    fps: float,
    inference_ms: Optional[float],
    status_message: str,
) -> GazeMetadataPacket:
    calibration_state = "idle" if calibration_session is None else calibration_session.state
    calibration_step = "none" if calibration_session is None else calibration_session.calibration_step
    calibrated = bool(calibration_session is not None and calibration_session.is_calibrated)
    active_point = None if calibration_session is None else calibration_session.current_point
    calibration_target_uv = None if active_point is None else (active_point.u, active_point.v)
    return GazeMetadataPacket(
        eye_timestamp_ns=eye_timestamp_ns,
        fpv_timestamp_ns=fpv_timestamp_ns,
        screen_uv=screen_uv,
        tracking_valid=bool(tracking_valid),
        feature_valid=bool(feature_valid),
        feature_mode=str(feature_mode),
        calibration_state=calibration_state,
        calibration_step=calibration_step,
        calibrated=calibrated,
        calibration_target_uv=calibration_target_uv,
        sync_skew_ms=sync_skew_ms,
        sync_stale=bool(sync_stale),
        eye_stale=bool(eye_stale),
        fpv_stale=bool(fpv_stale),
        fps=float(fps),
        inference_ms=inference_ms,
        status_message=str(status_message),
    )


class GazeMetadataPublisher:
    def __init__(
        self,
        stdout_enabled: bool = True,
        udp_host: Optional[str] = None,
        udp_port: Optional[int] = None,
    ) -> None:
        self.stdout_enabled = bool(stdout_enabled)
        self.udp_host = udp_host
        self.udp_port = udp_port
        self._sock: Optional[socket.socket] = None
        if udp_host is not None and udp_port is not None:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def publish(self, packet: GazeMetadataPacket) -> None:
        line = packet.to_json_line()
        if self.stdout_enabled:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

        if self._sock is not None and self.udp_host is not None and self.udp_port is not None:
            self._sock.sendto(line.encode("utf-8"), (self.udp_host, int(self.udp_port)))

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None
