from __future__ import annotations

import argparse
import socket
import time
from dataclasses import dataclass
from typing import Optional

from eyetrack.config import DEFAULT_INPUT_HEIGHT, DEFAULT_INPUT_WIDTH
from eyetrack.gaze import (
    CalibrationSession,
    advance_calibration_session,
    begin_calibration_session,
    cancel_calibration_session,
    extract_gaze_features_from_label_map,
    predict_screen_point,
    resolve_tracking_features,
)
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
KERNEL_SIZE = 3
IRIS_MIN_AREA = 100
PUPIL_MIN_AREA = 20
MAX_VALID_NORM_RADIUS = 0.85

EMA_ALPHA = 0.35
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


@dataclass
class OnlineEMAFilter:
    alpha: float
    x: Optional[float] = None
    y: Optional[float] = None

    def reset(self) -> None:
        self.x = None
        self.y = None

    def update(self, x: Optional[float], y: Optional[float], valid: bool) -> tuple[Optional[float], Optional[float]]:
        if not valid or x is None or y is None:
            return self.x, self.y

        if self.x is None or self.y is None:
            self.x = float(x)
            self.y = float(y)
        else:
            self.x = float(self.alpha * x + (1.0 - self.alpha) * self.x)
            self.y = float(self.alpha * y + (1.0 - self.alpha) * self.y)

        return self.x, self.y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="树莓派 5 双 CSI 实时视线元数据 + FPV 视频")
    parser.add_argument("--model_path", type=str, required=True, help="量化 ONNX 模型路径，仅支持 .onnx")
    parser.add_argument("--eye_camera_id", type=int, default=0, help="近眼红外相机 ID")
    parser.add_argument("--fpv_camera_id", type=int, default=1, help="FPV 相机 ID")
    parser.add_argument("--eye_width", type=int, default=DEFAULT_INPUT_WIDTH, help="近眼相机宽度，必须与模型输入一致")
    parser.add_argument("--eye_height", type=int, default=DEFAULT_INPUT_HEIGHT, help="近眼相机高度，必须与模型输入一致")
    parser.add_argument("--fpv_width", type=int, default=DEFAULT_FPV_WIDTH, help="FPV 输出宽度")
    parser.add_argument("--fpv_height", type=int, default=DEFAULT_FPV_HEIGHT, help="FPV 输出高度")
    parser.add_argument("--eye_fps", type=float, default=DEFAULT_EYE_FPS, help="近眼相机帧率")
    parser.add_argument("--fpv_fps", type=float, default=DEFAULT_FPV_FPS, help="FPV 相机帧率")
    parser.add_argument("--feature_mode", type=str, choices=["pupil_iris", "iris_only"], default="iris_only", help="实时校准与跟踪使用的特征模式")
    parser.add_argument("--sync_tolerance_ms", type=float, default=DEFAULT_SYNC_TOLERANCE_MS, help="cam0/cam1 时间戳容差，超过则标记 sync_stale")
    parser.add_argument("--rtsp_host", type=str, default=DEFAULT_RTSP_HOST, help="RTSP 绑定地址")
    parser.add_argument("--rtsp_port", type=int, default=DEFAULT_RTSP_PORT, help="RTSP 端口")
    parser.add_argument("--rtsp_path", type=str, default=DEFAULT_RTSP_PATH, help="RTSP 路径")
    parser.add_argument("--rtsp_bitrate_kbps", type=int, default=DEFAULT_RTSP_BITRATE_KBPS, help="RTSP H.264 目标码率")
    parser.add_argument("--metadata_stdout", dest="metadata_stdout", action="store_true", help="将 gaze/cali 元数据以 JSON 行格式输出到 stdout")
    parser.add_argument("--no-metadata_stdout", dest="metadata_stdout", action="store_false", help="禁用 stdout 元数据输出")
    parser.add_argument("--metadata_udp_host", type=str, default=None, help="可选 UDP 元数据目标地址")
    parser.add_argument("--metadata_udp_port", type=int, default=None, help="可选 UDP 元数据目标端口")
    parser.add_argument("--eye_hflip", action="store_true", help="近眼相机水平翻转")
    parser.add_argument("--eye_vflip", action="store_true", help="近眼相机垂直翻转")
    parser.add_argument("--fpv_hflip", action="store_true", help="FPV 相机水平翻转")
    parser.add_argument("--fpv_vflip", action="store_true", help="FPV 相机垂直翻转")
    parser.add_argument("--fpv_pixel_format", type=str, default=DEFAULT_FPV_PIXEL_FORMAT, help="FPV 像素格式，默认 BGR888")
    parser.add_argument("--camera_stale_ms", type=float, default=DEFAULT_CAMERA_STALE_MS, help="相机帧超过该时长未更新时标记 stale")
    parser.add_argument("--reconnect_interval_ms", type=int, default=DEFAULT_RECONNECT_INTERVAL_MS, help="相机异常后的自动重连间隔")
    parser.add_argument("--calibration_settle_ms", type=int, default=DEFAULT_CALIBRATION_SETTLE_MS, help="校准点切换后的稳定等待时长")
    parser.add_argument("--calibration_capture_ms", type=int, default=DEFAULT_CALIBRATION_CAPTURE_MS, help="每个校准点的采样时长")
    parser.add_argument("--calibration_min_valid_frames", type=int, default=DEFAULT_CALIBRATION_MIN_VALID_FRAMES, help="每个校准点要求的最少有效帧数")
    parser.add_argument("--calibration_margin", type=float, default=DEFAULT_CALIBRATION_MARGIN, help="四角校准点距边缘的归一化留白")
    parser.set_defaults(metadata_stdout=True)
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


def print_camera_inventory(eye_camera_id: int, fpv_camera_id: int) -> None:
    cameras = discover_picamera_cameras()
    if len(cameras) == 0:
        raise RuntimeError("未发现任何 Picamera2 相机。")

    print("detected cameras:")
    for camera in cameras:
        marker = []
        if camera.camera_id == eye_camera_id:
            marker.append("eye")
        if camera.camera_id == fpv_camera_id:
            marker.append("fpv")
        label = ",".join(marker) if len(marker) > 0 else "-"
        print(
            f"  id={camera.camera_id} [{label}] model={camera.model} "
            f"location={camera.location} rotation={camera.rotation} id={camera.identifier}"
        )

    camera_ids = {camera.camera_id for camera in cameras}
    if eye_camera_id not in camera_ids:
        raise RuntimeError(f"未找到 eye_camera_id={eye_camera_id} 对应的相机。")
    if fpv_camera_id not in camera_ids:
        raise RuntimeError(f"未找到 fpv_camera_id={fpv_camera_id} 对应的相机。")
    if eye_camera_id == fpv_camera_id:
        raise RuntimeError("eye_camera_id 与 fpv_camera_id 不能相同。")


def clear_tracking_state(
    ema_filter: OnlineEMAFilter,
) -> tuple[Optional[tuple[float, float]], Optional[float], bool]:
    ema_filter.reset()
    displayed_screen_uv = None
    last_valid_screen_ts_ms = None
    tracking_valid = False
    return displayed_screen_uv, last_valid_screen_ts_ms, tracking_valid


def extract_geometry_result(pred_label_map):
    return extract_gaze_features_from_label_map(
        pred_label_map=pred_label_map,
        valid_mask=None,
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


def main() -> None:
    args = parse_args()
    if (args.metadata_udp_host is None) != (args.metadata_udp_port is None):
        raise RuntimeError("metadata_udp_host 与 metadata_udp_port 必须同时提供，或同时省略。")

    runtime = PiOnnxSegmentationRuntime(model_path=args.model_path, backend="cpu")

    if runtime.input_width != args.eye_width or runtime.input_height != args.eye_height:
        raise RuntimeError(
            f"eye_width/eye_height 必须与模型输入完全一致。模型输入={runtime.input_width}x{runtime.input_height}，"
            f"当前参数={args.eye_width}x{args.eye_height}。"
        )

    print_camera_inventory(eye_camera_id=args.eye_camera_id, fpv_camera_id=args.fpv_camera_id)

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
    fpv_worker.start()

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

    display_host = resolve_display_host(args.rtsp_host)
    display_rtsp_url = f"rtsp://{display_host}:{args.rtsp_port}/{args.rtsp_path.strip('/')}"
    print("onnx providers:", runtime.providers)
    print("eye input:", f"{runtime.input_width}x{runtime.input_height}")
    print("rtsp encoder:", rtsp_server.encoder_name)
    print("rtsp url:", display_rtsp_url)
    if args.metadata_udp_host is not None and args.metadata_udp_port is not None:
        print("metadata udp:", f"{args.metadata_udp_host}:{args.metadata_udp_port}")
    print("metadata stdout:", args.metadata_stdout)

    ema_filter = OnlineEMAFilter(alpha=EMA_ALPHA)
    calibration_session: Optional[CalibrationSession] = None
    displayed_screen_uv: Optional[tuple[float, float]]
    last_valid_screen_ts_ms: Optional[float]
    tracking_valid: bool
    displayed_screen_uv, last_valid_screen_ts_ms, tracking_valid = clear_tracking_state(ema_filter)

    status_message = "树莓派实时已启动，按 s 开始五点校准。"
    last_eye_processed_ts: Optional[int] = None
    last_pushed_fpv_ts: Optional[int] = None
    last_eye_reconnect_ms = 0.0
    last_fpv_reconnect_ms = 0.0
    last_inference_ms: Optional[float] = None
    last_sync_skew_ms: Optional[float] = None
    last_sync_stale = False
    current_feature_valid = False
    fps = 0.0
    prev_processed_time = time.monotonic()
    running = True

    try:
        with NonBlockingTerminalReader() as key_reader:
            while running:
                loop_now = time.monotonic()
                now_ms = loop_now * 1000.0
                now_ns = time.monotonic_ns()
                needs_push = False
                needs_metadata_publish = False

                key = key_reader.poll_key()
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
                    )
                    displayed_screen_uv, last_valid_screen_ts_ms, tracking_valid = clear_tracking_state(ema_filter)
                    status_message = f"开始校准: {calibration_session.calibration_step}"
                    needs_push = True
                    needs_metadata_publish = True
                elif key == "x":
                    if calibration_session is not None:
                        calibration_session = cancel_calibration_session(calibration_session, reason="用户取消校准。")
                    displayed_screen_uv, last_valid_screen_ts_ms, tracking_valid = clear_tracking_state(ema_filter)
                    status_message = "已取消当前校准。"
                    needs_push = True
                    needs_metadata_publish = True
                elif key == "r":
                    calibration_session = None
                    displayed_screen_uv, last_valid_screen_ts_ms, tracking_valid = clear_tracking_state(ema_filter)
                    status_message = "已重置跟踪与校准。"
                    needs_push = True
                    needs_metadata_publish = True

                last_eye_reconnect_ms, eye_reconnect_error, eye_reconnected = maybe_reconnect_camera(
                    worker=eye_worker,
                    now_ms=now_ms,
                    last_attempt_ms=last_eye_reconnect_ms,
                    reconnect_interval_ms=args.reconnect_interval_ms,
                )
                last_fpv_reconnect_ms, fpv_reconnect_error, fpv_reconnected = maybe_reconnect_camera(
                    worker=fpv_worker,
                    now_ms=now_ms,
                    last_attempt_ms=last_fpv_reconnect_ms,
                    reconnect_interval_ms=args.reconnect_interval_ms,
                )

                if eye_reconnected or fpv_reconnected:
                    calibration_session = None
                    displayed_screen_uv, last_valid_screen_ts_ms, tracking_valid = clear_tracking_state(ema_filter)
                    status_message = "相机已重连，校准已清空。"
                    needs_push = True
                    needs_metadata_publish = True
                elif eye_reconnect_error is not None:
                    status_message = f"eye 相机异常，等待重连: {eye_reconnect_error}"
                elif fpv_reconnect_error is not None:
                    status_message = f"fpv 相机异常，等待重连: {fpv_reconnect_error}"

                eye_frame = eye_worker.get_latest_frame()
                fpv_frame = fpv_worker.get_latest_frame()
                eye_health = eye_worker.get_health()
                fpv_health = fpv_worker.get_health()
                eye_stale = is_frame_stale(eye_frame, stale_after_ms=args.camera_stale_ms, now_ns=now_ns)
                fpv_stale = is_frame_stale(fpv_frame, stale_after_ms=args.camera_stale_ms, now_ns=now_ns)

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

                    _, input_tensor = prepare_eye_tensor_from_yuv420(
                        eye_frame.frame,
                        input_width=runtime.input_width,
                        input_height=runtime.input_height,
                    )
                    infer_start = time.perf_counter()
                    pred_label_map = runtime.predict_label_map(input_tensor)
                    last_inference_ms = (time.perf_counter() - infer_start) * 1000.0

                    geometry_result = extract_geometry_result(pred_label_map)
                    feature_x, feature_y, current_feature_valid, _ = resolve_tracking_features(
                        geometry_result,
                        feature_mode=args.feature_mode,
                    )
                    smoothed_x, smoothed_y = ema_filter.update(feature_x, feature_y, current_feature_valid)

                    delta_time = max(loop_now - prev_processed_time, 1e-6)
                    fps = 1.0 / delta_time
                    prev_processed_time = loop_now

                    if calibration_session is not None and calibration_session.is_active:
                        before_state = calibration_session.state
                        before_step = calibration_session.calibration_step
                        calibration_session = advance_calibration_session(
                            calibration_session,
                            now_ms=now_ms,
                            feature_dx=smoothed_x,
                            feature_dy=smoothed_y,
                            feature_valid=bool(current_feature_valid and smoothed_x is not None and smoothed_y is not None),
                        )
                        if calibration_session.state != before_state or calibration_session.calibration_step != before_step:
                            status_message = f"校准: {calibration_session.calibration_step} ({calibration_session.state})"
                        if calibration_session.state == "failed":
                            displayed_screen_uv, last_valid_screen_ts_ms, tracking_valid = clear_tracking_state(ema_filter)
                            status_message = calibration_session.failure_reason or "校准失败。"
                        elif calibration_session.state == "completed":
                            status_message = "五点校准完成。"

                    if calibration_session is not None and calibration_session.is_calibrated and current_feature_valid:
                        predicted_screen_uv = predict_screen_point(
                            calibration_session.model,
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

                    needs_push = True
                    needs_metadata_publish = True

                if not current_feature_valid:
                    tracking_valid = False

                if last_valid_screen_ts_ms is not None and now_ms - last_valid_screen_ts_ms > STALE_SCREEN_POINT_TIMEOUT_MS:
                    displayed_screen_uv = None

                if eye_stale:
                    tracking_valid = False
                    if last_valid_screen_ts_ms is None or now_ms - last_valid_screen_ts_ms > STALE_SCREEN_POINT_TIMEOUT_MS:
                        displayed_screen_uv = None
                    if eye_health.last_error is None:
                        status_message = "eye 相机帧已过期。"
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

                if fpv_frame is not None and (fpv_frame.timestamp_ns != last_pushed_fpv_ts or needs_push):
                    base_frame = convert_fpv_frame_to_bgr(fpv_frame.frame, pixel_format=args.fpv_pixel_format)
                    rtsp_server.push_frame(base_frame)
                    last_pushed_fpv_ts = fpv_frame.timestamp_ns

                time.sleep(0.005)
    finally:
        fpv_worker.stop()
        eye_worker.stop()
        rtsp_server.stop()
        metadata_publisher.close()


if __name__ == "__main__":
    main()
