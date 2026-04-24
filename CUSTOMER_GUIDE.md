# EyeTrack Customer Guide

## 1. Purpose and Audience

This document explains how the EyeTrack repository works for customer delivery and technical handover.

Who this guide is for:
- Project managers who need a clear system-level understanding.
- Integration engineers who need to run, deploy, and troubleshoot.
- Technical reviewers who need to understand module boundaries and library choices.

Scope of this guide:
- Focus on inference and deployment workflows.
- Keep training and quantization as concise background.
- Separate Desktop and Raspberry Pi pipelines.
- Include tools, libraries, call chains, inputs, and outputs.

Out of scope:
- Internal experiment notes.
- Local machine absolute paths.
- Unstable export paths not recommended for production.

---

## 2. What the Repository Delivers

EyeTrack is an eye-segmentation and gaze-estimation system built around a lightweight U-Net segmentation model.

Main delivered capabilities:
- Segmentation model inference on eye images.
- Geometry extraction from segmentation maps (iris and pupil features).
- Realtime gaze tracking demo on desktop.
- Realtime dual-camera gaze runtime on Raspberry Pi 5.
- ONNX export, ONNX Runtime evaluation, and performance benchmark tooling.

Main artifacts produced:
- PyTorch checkpoints: `.pth`
- ONNX models: `.onnx`
- Segmentation outputs: `.npy`
- Geometry CSV reports: `.csv`
- Optional recorded demonstration videos: `.mp4` (desktop), `.mkv` (Pi record mode)

---

## 3. High-Level Architecture

### 3.1 Shared Core Pipeline (Desktop + Pi)

The same core logic is used across both runtime targets:

1. Frame acquisition
2. Preprocessing to model input tensor
3. Segmentation inference (PyTorch or ONNX Runtime)
4. Label map post-processing
5. Iris/pupil feature extraction
6. Calibration mapping to screen coordinates
7. Output rendering and metadata publication

### 3.2 Core Modules and Responsibilities

- `eyetrack/gaze.py`
  - Geometric feature extraction from segmentation masks.
  - Calibration model and calibration state machine.
  - Screen coordinate prediction from calibrated features.

- `eyetrack/realtime_gaze.py`
  - Calibration canvas rendering.
  - Screen preview panel rendering.
  - Realtime UI helper functions.

- `eyetrack/deployment/onnx_tools.py`
  - ONNX session creation.
  - ONNX model preprocessing/checking.
  - Calibration data reader for PTQ.
  - ONNX model evaluation utilities.

- `eyetrack/runtime.py`
  - Device selection (`auto/cpu/cuda`).
  - AMP context handling for desktop PyTorch inference/training.

---

## 4. Desktop Side (PC) - Detailed Workflow

### 4.1 Customer-Facing Desktop Use Cases

- Realtime demo with webcam: `realtime_eye_direction.py`
- Offline segmentation to NPY: `predict_segmentation_to_npy.py`
- Batch sequence processing: `batch_predict_and_extract_sequences.py`
- Geometry extraction report generation: `extract_geometry_from_segmentation.py`
- ONNX model export/evaluation/benchmark:
  - `export_unet_to_onnx.py`
  - `evaluate_onnx_model.py`
  - `benchmark_onnx_model.py`

### 4.2 Desktop Realtime Call Chain

Entry:
- `realtime_eye_direction.py`

Primary call chain:
- Camera frame -> `preprocess_bgr_frame(...)` in `eyetrack/data/preprocessing.py`
- Model inference:
  - PyTorch path: U-Net in `eyetrack/models/unet.py`
  - ONNX path: `build_onnx_session(...)` from `eyetrack/deployment/onnx_tools.py`
- Segmentation map -> `extract_gaze_features_from_label_map(...)` in `eyetrack/gaze.py`
- Tracking feature selection -> `resolve_tracking_features(...)` in `eyetrack/gaze.py`
- Calibration lifecycle:
  - Start: `begin_calibration_session(...)`
  - Advance: `advance_calibration_session(...)`
  - Predict: `predict_screen_point(...)`
- Optional visualization helpers from `eyetrack/realtime_gaze.py`

### 4.3 Desktop Batch and Analysis Chain

Batch processing entry:
- `batch_predict_and_extract_sequences.py`

Flow:
1. Enumerate sequence directories.
2. Run model inference to label maps via `run_prediction_to_npy(...)` in `eyetrack/workflows/predict.py`.
3. Extract per-frame geometry via `process_prediction_directory(...)` in `extract_geometry_from_segmentation.py`.
4. Produce CSV outputs for downstream analysis.

### 4.4 ONNX Lifecycle on Desktop

- Export checkpoint -> ONNX:
  - `eyetrack/workflows/export_onnx.py`
  - Function: `export_checkpoint_to_onnx(...)`
- Quantize ONNX (PTQ):
  - `eyetrack/workflows/quantize_onnx.py`
  - Function: `quantize_onnx_model(...)`
- Evaluate ONNX accuracy:
  - `eyetrack/workflows/evaluate_onnx.py`
- Benchmark ONNX latency:
  - `eyetrack/workflows/benchmark_onnx.py`

Important stability note:
- `qat_qdq_export` is intentionally blocked in current workflow.
- Recommended production path is:
  - Export stable float ONNX first.
  - Then run ONNX Runtime static quantization.

### 4.5 Desktop Commands (Customer-Safe Templates)

Realtime demo:
```bash
python realtime_eye_direction.py \
  --model_path ./checkpoints/unet_b16_384x240_fp32.onnx \
  --camera_index 0
```

Offline prediction:
```bash
python predict_segmentation_to_npy.py \
  --checkpoint_path ./checkpoints/best_unet_b8_384x240_amp.pth \
  --image_dir path/to/images \
  --output_dir ./pred_validation_npy
```

Geometry extraction:
```bash
python extract_geometry_from_segmentation.py \
  --pred_dir ./pred_validation_npy \
  --output_csv ./geometry_result.csv
```

ONNX evaluation:
```bash
python evaluate_onnx_model.py \
  --model_path ./checkpoints/unet_b8_384x240_fp32.onnx \
  --root_dir path/to/openeds \
  --split validation
```

---

## 5. Raspberry Pi Side - Detailed Workflow

### 5.1 Deployment Goal

The Raspberry Pi pipeline is built for realtime edge operation with dual CSI cameras:
- Eye camera for segmentation and feature extraction.
- FPV camera for streamed or recorded output.

Main entry:
- `realtime_eye_direction_pi.py`

### 5.2 Pi Runtime Module Map

- `eyetrack/raspi/camera.py`
  - Picamera2 stream worker threads.
  - Eye tensor preparation from YUV420.
  - FPV frame format conversion.

- `eyetrack/raspi/runtime.py`
  - `PiOnnxSegmentationRuntime`.
  - Enforces ONNX-only, CPU backend, quantized-model expectation.

- `eyetrack/raspi/sync.py`
  - Frame timestamp resolution.
  - Eye/FPV frame pairing and skew checks.
  - Staleness checks.

- `eyetrack/raspi/rtsp.py`
  - RTSP server setup.
  - GStreamer H.264 pipeline.
  - Live frame push to RTSP output.

- `eyetrack/raspi/recorder.py`
  - Local MKV recording pipeline.
  - GStreamer-based H.264 recording flow.

- `eyetrack/raspi/metadata.py`
  - Structured metadata packet.
  - JSON-line stdout and optional UDP publisher.

- `eyetrack/raspi/overlay.py`
  - FPV overlay composition for gaze and status visualization.

### 5.3 Pi Realtime Lifecycle

1. Parse runtime arguments and validate mode combinations.
2. Load quantized ONNX model through `PiOnnxSegmentationRuntime`.
3. Discover and initialize eye and FPV cameras.
4. Start output mode:
- RTSP mode, or
- Record mode.
5. Main loop per cycle:
- Read latest camera frames.
- Run frame synchronization checks.
- Convert eye frame to model tensor.
- Inference -> label map.
- Extract geometry features via shared `eyetrack/gaze.py`.
- Update confidence and Kalman smoothing.
- Run calibration/tracking state machine.
- Publish metadata packet.
- Push FPV frame to RTSP or recorder.
6. Handle user controls (`s`, `x`, `r`, `q`, record key).
7. Graceful shutdown of camera/output/publisher resources.

### 5.4 Pi Metadata Contract

The metadata packet includes:
- Timestamps: eye and FPV
- Current screen UV prediction
- Output mode and recording status
- Tracking and feature validity flags
- Calibration state and current calibration step
- Synchronization health (`sync_skew_ms`, stale flags)
- FPS and inference latency
- Human-readable status message

Published channels:
- JSON lines to stdout
- Optional UDP stream

### 5.5 Pi Runtime Commands (Customer-Safe Templates)

RTSP mode:
```bash
python realtime_eye_direction_pi.py \
  --model_path ./checkpoints/unet_b16_384x240_int8_qdq.onnx \
  --eye_camera_id 0 \
  --fpv_camera_id 1 \
  --feature_mode iris_only
```

Record mode:
```bash
python realtime_eye_direction_pi.py \
  --model_path ./checkpoints/unet_b16_384x240_int8_qdq.onnx \
  --eye_camera_id 0 \
  --fpv_camera_id 1 \
  --feature_mode iris_only \
  --fpv_output_mode record \
  --record_dir ./recordings
```

---

## 6. Tools and Libraries Inventory

This section lists customer-relevant tools and libraries currently used in the repository.

### 6.1 Internal Tools (Repository Scripts)

Model and data workflows:
- `Train.py`
- `Step_Train.py`
- `preprocess_openeds_dataset.py`
- `predict_segmentation_to_npy.py`
- `batch_predict_and_extract_sequences.py`
- `extract_geometry_from_segmentation.py`
- `analyze_sequence_geometry.py`

Deployment workflows:
- `export_unet_to_onnx.py`
- `quantize_onnx_model.py`
- `evaluate_onnx_model.py`
- `benchmark_onnx_model.py`

Realtime runtime:
- `realtime_eye_direction.py`
- `realtime_eye_direction_pi.py`

### 6.2 Python Runtime Libraries (Direct, Function-Oriented)

Core ML and numeric:
- `torch`: model definition, training, desktop inference path.
- `numpy`: arrays, preprocessing, geometry computations.

Vision and image IO:
- `opencv-python` (`cv2`): image transforms, contour/ellipse geometry, overlays.
- `Pillow` (`PIL`): image loading and format handling.

ONNX deployment:
- `onnx`: model load/check/infer-shape/metadata operations.
- `onnxruntime`: inference runtime and quantization toolkit integration.

Data processing and reporting:
- `pandas`: tabular sequence processing and CSV analytics scripts.

Visualization:
- `matplotlib`: plotting in sequence analysis workflows.

Progress and utility:
- `tqdm`: training/evaluation progress rendering.

Networking and dataset support:
- `requests`, `kagglehub`, `kagglesdk`: optional dataset retrieval and HTTP operations.

### 6.3 Python Environment Support Packages (Pinned in requirements)

The repository also pins ecosystem packages that support the above stack, including:
- `protobuf`, `typing_extensions`, `packaging`, `setuptools`
- `networkx`, `sympy`, `mpmath`, `filelock`, `fsspec`
- `python-dateutil`, `tzdata`, `six`
- `contourpy`, `cycler`, `fonttools`, `kiwisolver`, `pyparsing`
- `certifi`, `charset-normalizer`, `idna`, `urllib3`, `colorama`
- `Jinja2`, `MarkupSafe`, `PyYAML`

Notes:
- Some packages are direct imports in repository code.
- Others are transitive ecosystem dependencies kept pinned for stable environments.

### 6.4 Raspberry Pi System Packages (OS-Level, Not pip)

Required by Pi runtime stack:
- `python3-picamera2`
- `python3-opencv`
- `python3-gi`
- `gir1.2-gst-rtsp-server-1.0`
- `gstreamer1.0-tools`
- `gstreamer1.0-plugins-base`
- `gstreamer1.0-plugins-good`
- `gstreamer1.0-plugins-bad`
- `gstreamer1.0-plugins-ugly`

Why this split matters:
- Camera drivers, GStreamer and GI bindings are platform-level capabilities.
- They must come from Raspberry Pi OS packages for compatibility.

---

## 7. Data Contracts and Artifacts

### 7.1 Dataset Layout (OpenEDS Style)

Expected structure:
```text
dataset_root/
  train/
    images/
    labels/
    masks/
  validation/
    images/
    labels/
    masks/
```

### 7.2 Key Artifact Semantics

- Checkpoint `.pth`
  - Contains model state and metadata (input size, channels, etc.).
- Segmentation `.npy`
  - Per-pixel class label map.
- Geometry CSV
  - Frame-level iris/pupil geometry, offsets and normalized features.
- ONNX `.onnx`
  - Fixed-shape deployment model with embedded metadata.

---

## 8. Brief Training and Quantization Background

This section is intentionally short because production delivery focuses on inference.

Training summary:
- Main training flow is implemented in `eyetrack/workflows/train.py`.
- Data source is OpenEDS-style segmentation dataset.
- Output is a best-checkpoint artifact for downstream export/inference.

Quantization summary:
- PTQ uses ONNX Runtime static quantization in `eyetrack/workflows/quantize_onnx.py`.
- Calibration readers and fallback logic are included for memory-sensitive environments.
- Recommended production route remains float ONNX export plus PTQ quantization.

---

## 9. Operational Troubleshooting

Desktop common checks:
- Verify model input resolution matches preprocessing configuration.
- Confirm webcam index and permissions.
- If using CUDA, verify runtime device availability.

Pi common checks:
- Confirm both CSI cameras are detected and IDs are correct.
- Confirm eye frame shape matches model input dimensions.
- If RTSP is unavailable, verify GStreamer plugins and encoder availability.
- If sync is unstable, inspect timestamp skew and stale flags in metadata.

Model compatibility checks:
- Pi runtime requires ONNX model and expects quantized graph characteristics.
- Pi runtime accepts CPU backend only.

---

## 10. Known Limits and Customer Notes

- Raspberry Pi runtime is intentionally optimized for CPU ONNX Runtime path.
- Some advanced/export experimental options are present but not customer-recommended for production.
- For best supportability, use the documented commands and standard artifact names.

---

## 11. Suggested Handover Package

For customer delivery, include:
- This guide (`CUSTOMER_GUIDE.md`)
- A validated ONNX model set (float and quantized)
- One desktop startup command sheet
- One Raspberry Pi startup command sheet
- A sample metadata capture log and one geometry CSV example

This package format minimizes onboarding effort and shortens integration time.
