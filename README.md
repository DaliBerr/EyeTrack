# EyeTrack

A PyTorch-based eye semantic segmentation and geometric analysis project, including:

- Lightweight U-Net training and validation on an OpenEDS-style dataset
- Grayscale eye segmentation prediction, outputting label maps in `.npy` format
- Extraction of iris / pupil geometric parameters from segmentation results
- Sequence-level batch processing, abnormal-frame cleaning, smoothing, and visual analysis
- Real-time gaze-direction demo with a camera
- ONNX export, static PTQ quantization, and ONNX Runtime benchmarking

## Dependencies

Python 3.10+ is recommended.

```bash
pip install torch numpy opencv-python pillow matplotlib pandas onnx onnxruntime
```

## Data Organization

Training and validation data are organized by default in the OpenEDS style:

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

- `images`: grayscale eye images in `.png` format
- `labels`: segmentation labels in `.npy` format
- `masks`: valid-region masks in `.png` format

## Main Scripts

- `Train.py`: train the model and save checkpoints
- `predict_segmentation_to_npy.py`: run segmentation prediction on an image directory
- `preprocess_openeds_dataset.py`: offline resize the OpenEDS-style dataset to a fixed size
- `export_unet_to_onnx.py`: export a PyTorch checkpoint to fixed-input ONNX
- `quantize_onnx_model.py`: run ONNX Runtime static PTQ
- `evaluate_onnx_model.py`: evaluate the ONNX model and optionally compare it with the PyTorch checkpoint
- `benchmark_onnx_model.py`: run a simple latency benchmark for the ONNX Runtime backend
- `extract_geometry_from_segmentation.py`: extract geometric parameters from segmentation results to CSV
- `batch_predict_and_extract_sequences.py`: batch-process `S_*` sequence directories
- `baseline_filter_sequences.py` / `baseline_filter_sequences_v2.py` / `baseline_filter_sequences_v3.py`: temporal anomaly cleaning and smoothing
- `analyze_sequence_geometry.py`: generate sequence analysis CSV files and curve plots
- `realtime_eye_direction.py`: camera real-time demo
- `realtime_eye_direction_pi.py`: Raspberry Pi 5 + Picamera2 dual-CSI real-time gaze metadata entry point
- `main.py`: dataset sample inspection entry point

## Common Commands

### Mobile Standard Preset (Recommended, b8@384x240)

Train a b8 model:

```bash
python Train.py \
  --root_dir path/to/openeds \
  --save_checkpoint_path ./checkpoints/best_unet_b8_384x240_amp.pth \
  --base_channels 8 \
  --input_width 384 \
  --input_height 240 \
  --no-use_mask \
  --amp
```

Preprocess the dataset offline:

```bash
python preprocess_openeds_dataset.py \
  --input_root path/to/openeds_raw \
  --output_root path/to/openeds_384x240 \
  --splits train,validation \
  --input_width 384 \
  --input_height 240 \
  --process_mask
```

Run segmentation prediction on a single image directory:

```bash
python predict_segmentation_to_npy.py \
  --checkpoint_path ./checkpoints/best_unet_b8_384x240_amp.pth \
  --image_dir path/to/images \
  --output_dir ./pred_validation_npy
```

Export to ONNX:

```bash
python export_unet_to_onnx.py \
  --checkpoint_path ./checkpoints/best_unet_b8_384x240_amp.pth \
  --onnx_path ./checkpoints/unet_b8_384x240_fp32.onnx
```

Run PTQ:

```bash
python quantize_onnx_model.py \
  --model_path ./checkpoints/unet_b8_384x240_fp32.onnx \
  --calibration_root path/to/openeds \
  --output_path ./checkpoints/unet_b8_384x240_int8_qdq.onnx \
  --quant_format qdq \
  --activation_type qint8 \
  --weight_type qint8 \
  --calibration_method percentile \
  --auto_fallback_to_minmax_on_oom \
  --auto_fallback_to_u8u8 \
  --validation_root path/to/openeds
```

Evaluate ONNX and compare against PyTorch:

```bash
python evaluate_onnx_model.py \
  --model_path ./checkpoints/unet_b8_384x240_fp32.onnx \
  --root_dir path/to/openeds \
  --split validation \
  --checkpoint_path ./checkpoints/best_unet_b8_384x240_amp.pth
```

Batch-process temporal sequences:

```bash
python batch_predict_and_extract_sequences.py \
  --sequence_root path/to/sequences \
  --checkpoint_path ./checkpoints/best_unet_b8_384x240_amp.pth \
  --output_root ./sequence_outputs
```

Extract geometric parameters from prediction results:

```bash
python extract_geometry_from_segmentation.py \
  --pred_dir ./pred_validation_npy \
  --output_csv ./geometry_result.csv
```

Run smoothing and summary statistics on `sequence_outputs`:

```bash
python baseline_filter_sequences_v3.py \
  --input ./sequence_outputs \
  --output_dir ./baseline_v3_out
```

Generate sequence analysis plots:

```bash
python analyze_sequence_geometry.py \
  --input_root ./sequence_outputs \
  --output_root ./sequence_analysis
```

### Compatibility Baseline (b16@384x240)

The following commands are kept for the older baseline / compatibility workflow. The repository's default constants and default paths still point to b16:

```bash
python Train.py \
  --root_dir path/to/openeds \
  --save_checkpoint_path ./checkpoints/best_unet_b16_384x240_amp.pth \
  --base_channels 16 \
  --input_width 384 \
  --input_height 240 \
  --no-use_mask \
  --amp
```

```bash
python export_unet_to_onnx.py \
  --checkpoint_path ./checkpoints/best_unet_b16_384x240_amp.pth \
  --onnx_path ./checkpoints/unet_b16_384x240_fp32.onnx
```

```bash
python quantize_onnx_model.py \
  --model_path ./checkpoints/unet_b16_384x240_fp32.onnx \
  --calibration_root path/to/openeds \
  --output_path ./checkpoints/unet_b16_384x240_int8_qdq.onnx \
  --quant_format qdq \
  --activation_type qint8 \
  --weight_type qint8 \
  --calibration_method percentile \
  --auto_fallback_to_minmax_on_oom \
  --auto_fallback_to_u8u8 \
  --validation_root path/to/openeds
```

## Notes

- `Train.py` and some visualization scripts contain local default paths, so you need to adjust them for your environment before running.
- For newer full checkpoints, the PyTorch training, prediction, visualization, evaluation, export, and real-time scripts will preferentially read checkpoint metadata to automatically restore `base_channels`, input size, and AMP configuration.
- The current mobile recommended preset is `b8@384x240`, and `best_unet_b8_384x240_amp.pth`, `unet_b8_384x240_fp32.onnx`, and `unet_b8_384x240_int8_qdq.onnx` are the suggested artifact names.
- Quantization will, by default, fall back to `MinMax + a smaller calibration_limit` when histogram calibration runs out of memory. If you do not want this behavior, explicitly pass `--no-auto_fallback_to_minmax_on_oom`.
- The current lightweight model default configuration is `base_channels=16`, input size `384x240`, and preprocessing mode `raw grayscale + resize + normalize`.
- The model class count is 4, with `iris=2` and `pupil=3`.
- Output directories such as `checkpoints/`, `sequence_outputs/`, `sequence_analysis/`, and `vis_val/` are already excluded in `.gitignore`.

## Raspberry Pi 5 Dual-CSI Real-Time Run

The Raspberry Pi-specific real-time entry point is `realtime_eye_direction_pi.py`, which does not reuse the desktop version `realtime_eye_direction.py`.

### Extra Raspberry Pi Dependencies

On Raspberry Pi OS, it is recommended to install system packages first:

```bash
sudo apt install -y python3-picamera2 python3-opencv python3-gi \
  gir1.2-gst-rtsp-server-1.0 gstreamer1.0-tools \
  gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly
```

Then install the Python dependencies for the Pi scripts:

```bash
pip install -r requirements_pi.txt
```

### Launch Example

```bash
python realtime_eye_direction_pi.py \
  --model_path ./checkpoints/unet_b16_384x240_int8_qdq.onnx \
  --eye_camera_id 0 \
  --fpv_camera_id 1 \
  --feature_mode iris_only
```

Default behavior:

- `cam0` uses direct `YUV420@384x240` capture for the near-eye infrared image, without ROI or CPU-side resize.
- `cam1` is used as the FPV background output, and the default RTSP address is `rtsp://<pi-ip>:8554/fpv`.
- The Pi side does not draw gaze or calibration overlays directly into the FPV video; RTSP only transmits the raw FPV video.
- Gaze, calibration targets, and status are emitted as JSON lines to stdout, and can optionally also be sent via UDP to a receiver overlay using `--metadata_udp_host/--metadata_udp_port`.
- Keyboard controls: `s` start calibration, `x` cancel calibration, `r` reset tracking and calibration, `q` quit.
