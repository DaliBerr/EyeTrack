import sys
import tempfile
import types
import unittest
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import numpy as np

from eyetrack.gaze import begin_calibration_session


def _install_desktop_realtime_import_stubs() -> None:
    if "torch" not in sys.modules:
        torch_stub = types.ModuleType("torch")
        nn_stub = types.ModuleType("torch.nn")

        class _DummyModule:
            pass

        def _no_grad(func=None):
            if func is None:
                def _decorator(inner):
                    return inner
                return _decorator
            return func

        torch_stub.no_grad = _no_grad
        torch_stub.device = object
        torch_stub.Tensor = object
        torch_stub.nn = nn_stub
        nn_stub.Module = _DummyModule
        sys.modules["torch"] = torch_stub
        sys.modules["torch.nn"] = nn_stub

    if "eyetrack.data.preprocessing" not in sys.modules:
        preprocessing_stub = types.ModuleType("eyetrack.data.preprocessing")

        @dataclass
        class ResizeMeta:
            source_width: int
            source_height: int
            input_width: int
            input_height: int
            scale_x: float
            scale_y: float

        def preprocess_bgr_frame(*args, **kwargs):
            raise RuntimeError("preprocess_bgr_frame stub should not be called in this test suite.")

        preprocessing_stub.ResizeMeta = ResizeMeta
        preprocessing_stub.preprocess_bgr_frame = preprocess_bgr_frame
        sys.modules["eyetrack.data.preprocessing"] = preprocessing_stub

    if "eyetrack.deployment.onnx_tools" not in sys.modules:
        onnx_stub = types.ModuleType("eyetrack.deployment.onnx_tools")
        onnx_stub.build_onnx_session = lambda *args, **kwargs: None
        onnx_stub.require_onnx = lambda: None
        onnx_stub.require_onnxruntime = lambda: None
        sys.modules["eyetrack.deployment.onnx_tools"] = onnx_stub

    if "eyetrack.models.unet" not in sys.modules:
        unet_stub = types.ModuleType("eyetrack.models.unet")

        class UNet:
            def __init__(self, *args, **kwargs):
                pass

        unet_stub.UNet = UNet
        sys.modules["eyetrack.models.unet"] = unet_stub

    if "eyetrack.runtime" not in sys.modules:
        runtime_stub = types.ModuleType("eyetrack.runtime")
        runtime_stub.autocast_context = lambda *args, **kwargs: nullcontext()
        runtime_stub.resolve_device = lambda device: types.SimpleNamespace(type=device)
        sys.modules["eyetrack.runtime"] = runtime_stub

    if "eyetrack.training.checkpoints" not in sys.modules:
        checkpoint_stub = types.ModuleType("eyetrack.training.checkpoints")
        checkpoint_stub.load_checkpoint_flexible = lambda *args, **kwargs: {}
        checkpoint_stub.resolve_model_metadata = lambda *args, **kwargs: {}
        sys.modules["eyetrack.training.checkpoints"] = checkpoint_stub


_install_desktop_realtime_import_stubs()

from realtime_eye_direction import (
    DEMO_PANEL_ANCHOR_CENTER_LEFT,
    DEMO_RENDER_STATE_FINISHED,
    DEMO_RENDER_STATE_IDLE,
    DEMO_RENDER_STATE_READY,
    DEMO_RENDER_STATE_RENDERING,
    DemoGazeSample,
    DemoVideoSourceInfo,
    PreviewPanelSpec,
    ROIBox,
    build_demo_gaze_sample,
    compose_calibration_output_frame,
    compose_demo_output_frame,
    compute_demo_panel_layout,
    create_demo_video_writer,
    draw_demo_gaze_overlay,
    interpolate_screen_uv,
    parse_args,
    resolve_demo_calibration_output_path,
    resolve_demo_output_path,
    resolve_demo_render_state,
    resolve_interpolated_demo_gaze,
    start_demo_rendering,
    validate_demo_render_start_request,
)


class RealtimeDesktopDemoModeTests(unittest.TestCase):
    @staticmethod
    def build_calibrated_session():
        session = begin_calibration_session(
            now_ms=0.0,
            settle_ms=500,
            capture_ms=1000,
            min_valid_frames=5,
            point_pattern="nine",
        )
        session.state = "completed"
        session.model = object()
        return session

    @staticmethod
    def build_demo_source_info() -> DemoVideoSourceInfo:
        return DemoVideoSourceInfo(
            path="D:/tmp/demo.mp4",
            width=1280,
            height=720,
            fps=30.0,
            frame_count=120,
        )

    def test_parse_args_accepts_demo_options(self) -> None:
        args = parse_args(["--demo_fpv_video", "demo.mp4", "--demo_output", "out.mp4"])

        self.assertEqual(args.demo_fpv_video, "demo.mp4")
        self.assertEqual(args.demo_output, "out.mp4")

    def test_resolve_demo_output_path_defaults_next_to_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "fpv_clip.mov"
            output_path = Path(resolve_demo_output_path(str(source_path), None))

            self.assertEqual(output_path.parent, source_path.parent.resolve())
            self.assertEqual(output_path.name, "fpv_clip_gaze_demo.mp4")

    def test_resolve_demo_calibration_output_path_appends_calibration_suffix(self) -> None:
        calibration_output_path = Path(resolve_demo_calibration_output_path("D:/tmp/fpv_clip_gaze_demo.mp4"))

        self.assertEqual(calibration_output_path.name, "fpv_clip_gaze_demo_calibration.mp4")

    def test_validate_demo_render_start_request_requires_valid_roi(self) -> None:
        reason = validate_demo_render_start_request(
            demo_source_info=self.build_demo_source_info(),
            selected_roi=None,
            calibration_session=self.build_calibrated_session(),
            rendering_active=False,
        )

        self.assertEqual(reason, "Please select a valid ROI before starting demo rendering.")

    def test_validate_demo_render_start_request_requires_completed_calibration(self) -> None:
        session = begin_calibration_session(
            now_ms=0.0,
            settle_ms=500,
            capture_ms=1000,
            min_valid_frames=5,
            point_pattern="nine",
        )
        reason = validate_demo_render_start_request(
            demo_source_info=self.build_demo_source_info(),
            selected_roi=ROIBox(0, 0, 120, 120),
            calibration_session=session,
            rendering_active=False,
        )

        self.assertEqual(reason, "Demo rendering requires a completed calibration.")

    def test_create_demo_video_writer_rejects_unopened_writer(self) -> None:
        fake_writer = mock.Mock()
        fake_writer.isOpened.return_value = False

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "demo.mp4"
            with mock.patch("realtime_eye_direction.cv2.VideoWriter", return_value=fake_writer):
                with self.assertRaises(RuntimeError):
                    create_demo_video_writer(self.build_demo_source_info(), str(output_path))

        fake_writer.release.assert_called_once_with()

    def test_start_demo_rendering_rewinds_capture_before_opening_writer(self) -> None:
        fake_capture = mock.Mock()
        fake_writer = mock.Mock()

        with mock.patch("realtime_eye_direction.create_demo_video_writer", return_value=fake_writer) as create_writer:
            writer = start_demo_rendering(
                source_capture=fake_capture,
                source_info=self.build_demo_source_info(),
                output_path="demo_output.mp4",
            )

        self.assertIs(writer, fake_writer)
        fake_capture.set.assert_called_once()
        create_writer.assert_called_once()

    def test_compute_demo_panel_layout_prefers_horizontal_on_large_frame(self) -> None:
        panels = [
            PreviewPanelSpec(image=np.zeros((10, 10), dtype=np.uint8), label="ROI", desired_size=(220, 140), use_gray=True),
            PreviewPanelSpec(image=np.zeros((10, 10), dtype=np.uint8), label="Seg", desired_size=(220, 140), use_gray=True),
        ]

        placements = compute_demo_panel_layout(frame_width=1280, frame_height=720, panels=panels)

        self.assertEqual(len(placements), 2)
        self.assertEqual(placements[0].y, placements[1].y)
        self.assertLess(placements[0].x, placements[1].x)

    def test_compute_demo_panel_layout_falls_back_to_vertical_when_horizontal_too_small(self) -> None:
        panels = [
            PreviewPanelSpec(image=np.zeros((10, 10), dtype=np.uint8), label="ROI", desired_size=(220, 140), use_gray=True),
            PreviewPanelSpec(image=np.zeros((10, 10), dtype=np.uint8), label="Seg", desired_size=(220, 140), use_gray=True),
        ]

        placements = compute_demo_panel_layout(frame_width=180, frame_height=240, panels=panels)

        self.assertEqual(len(placements), 2)
        self.assertEqual(placements[0].x, placements[1].x)
        self.assertGreater(placements[0].y, placements[1].y)

    def test_compute_demo_panel_layout_supports_center_left_vertical_anchor(self) -> None:
        panels = [
            PreviewPanelSpec(image=np.zeros((10, 10), dtype=np.uint8), label="ROI", desired_size=(220, 140), use_gray=True),
            PreviewPanelSpec(image=np.zeros((10, 10), dtype=np.uint8), label="Seg", desired_size=(220, 140), use_gray=True),
        ]

        placements = compute_demo_panel_layout(
            frame_width=640,
            frame_height=480,
            panels=panels,
            anchor=DEMO_PANEL_ANCHOR_CENTER_LEFT,
            prefer_horizontal=False,
        )

        self.assertEqual(len(placements), 2)
        self.assertGreater(placements[0].x, 10)
        self.assertEqual(placements[0].x, placements[1].x)
        self.assertGreater(placements[1].y, placements[0].y)

    def test_interpolate_screen_uv_linearly_blends_points(self) -> None:
        uv = interpolate_screen_uv((0.2, 0.4), (0.6, 0.8), 0.25)

        self.assertIsNotNone(uv)
        self.assertAlmostEqual(uv[0], 0.3)
        self.assertAlmostEqual(uv[1], 0.5)

    def test_resolve_interpolated_demo_gaze_uses_linear_interpolation_between_samples(self) -> None:
        previous_sample = DemoGazeSample(timestamp_s=0.0, screen_uv=(0.2, 0.2), tracking_valid=False)
        current_sample = DemoGazeSample(timestamp_s=1.0, screen_uv=(0.6, 0.4), tracking_valid=True)

        uv, tracking_valid = resolve_interpolated_demo_gaze(
            previous_sample=previous_sample,
            current_sample=current_sample,
            target_timestamp_s=0.75,
        )

        self.assertIsNotNone(uv)
        self.assertAlmostEqual(uv[0], 0.5)
        self.assertAlmostEqual(uv[1], 0.35)
        self.assertTrue(tracking_valid)

    def test_build_demo_gaze_sample_normalizes_values(self) -> None:
        sample = build_demo_gaze_sample(timestamp_s=-1.0, screen_uv=(1, 2), tracking_valid=1)

        self.assertEqual(sample.timestamp_s, 0.0)
        self.assertEqual(sample.screen_uv, (1.0, 2.0))
        self.assertTrue(sample.tracking_valid)

    def test_draw_demo_gaze_overlay_draws_emphasized_marker(self) -> None:
        frame = np.zeros((200, 400, 3), dtype=np.uint8)

        output = draw_demo_gaze_overlay(frame, screen_uv=(0.5, 0.5), tracking_valid=True)

        marker_region = output[70:130, 170:230]
        green_pixels = np.argwhere(np.all(marker_region == np.array([0, 255, 0], dtype=np.uint8), axis=2))
        shadow_pixels = np.argwhere(np.all(marker_region == np.array([0, 0, 0], dtype=np.uint8), axis=2))

        self.assertGreater(len(green_pixels), 0)
        self.assertGreater(len(shadow_pixels), 0)

    def test_compose_demo_output_frame_draws_marker_and_panels(self) -> None:
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        roi_gray_preview = np.full((140, 220), 128, dtype=np.uint8)
        pred_label_map = np.full((140, 220), 3, dtype=np.uint8)

        output = compose_demo_output_frame(
            fpv_frame_bgr=frame,
            screen_uv=(0.5, 0.5),
            tracking_valid=True,
            roi_gray_preview=roi_gray_preview,
            pred_label_map=pred_label_map,
        )

        self.assertEqual(output.shape, frame.shape)
        ring_slice = output[70:130, 170:230]
        self.assertTrue(np.any(np.all(ring_slice == np.array([0, 255, 0], dtype=np.uint8), axis=2)))
        self.assertGreater(int(output[180, 20].sum()), 0)
        self.assertGreater(int(output[180, 240].sum()), 0)

    def test_compose_calibration_output_frame_places_panels_away_from_bottom_left(self) -> None:
        calibration_canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        roi_gray_preview = np.full((140, 220), 128, dtype=np.uint8)
        pred_label_map = np.full((140, 220), 3, dtype=np.uint8)

        output = compose_calibration_output_frame(
            calibration_canvas_bgr=calibration_canvas,
            roi_gray_preview=roi_gray_preview,
            pred_label_map=pred_label_map,
        )

        self.assertGreater(int(output[170, 120].sum()), 0)
        self.assertEqual(int(output[460, 20].sum()), 0)

    def test_resolve_demo_render_state_transitions_between_ready_rendering_finished_and_idle(self) -> None:
        source_info = self.build_demo_source_info()
        selected_roi = ROIBox(0, 0, 120, 120)
        session = self.build_calibrated_session()

        ready_state = resolve_demo_render_state(
            current_state=DEMO_RENDER_STATE_IDLE,
            demo_source_info=source_info,
            selected_roi=selected_roi,
            calibration_session=session,
            rendering_active=False,
        )
        rendering_state = resolve_demo_render_state(
            current_state=ready_state,
            demo_source_info=source_info,
            selected_roi=selected_roi,
            calibration_session=session,
            rendering_active=True,
        )
        finished_state = resolve_demo_render_state(
            current_state=DEMO_RENDER_STATE_FINISHED,
            demo_source_info=source_info,
            selected_roi=selected_roi,
            calibration_session=session,
            rendering_active=False,
        )
        idle_state = resolve_demo_render_state(
            current_state=finished_state,
            demo_source_info=source_info,
            selected_roi=None,
            calibration_session=session,
            rendering_active=False,
        )

        self.assertEqual(ready_state, DEMO_RENDER_STATE_READY)
        self.assertEqual(rendering_state, DEMO_RENDER_STATE_RENDERING)
        self.assertEqual(finished_state, DEMO_RENDER_STATE_FINISHED)
        self.assertEqual(idle_state, DEMO_RENDER_STATE_IDLE)


if __name__ == "__main__":
    unittest.main()
