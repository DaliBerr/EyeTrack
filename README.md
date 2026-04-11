# EyeTrack

基于 PyTorch 的眼部语义分割与几何分析项目，包含：

- OpenEDS 风格数据集上的轻量 U-Net 训练与验证
- 灰度眼图分割预测，输出标签图 `.npy`
- 从分割结果中提取 iris / pupil 几何参数
- 序列级批处理、异常帧清洗、平滑与可视化分析
- 摄像头实时视线方向演示
- ONNX 导出、静态 PTQ 量化与 ONNX Runtime 基准

## 依赖

建议使用 Python 3.10+。

```bash
pip install torch numpy opencv-python pillow matplotlib pandas onnx onnxruntime
```

## 数据组织

训练和验证默认按 OpenEDS 风格组织：

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

- `images`: 灰度眼图 `.png`
- `labels`: 分割标签 `.npy`
- `masks`: 有效区域掩码 `.png`

## 主要脚本

- `Train.py`: 训练模型并保存 checkpoint
- `predict_segmentation_to_npy.py`: 对图像目录做分割预测
- `preprocess_openeds_dataset.py`: 离线缩放 OpenEDS 风格数据集到固定尺寸
- `export_unet_to_onnx.py`: 将 PyTorch checkpoint 导出为固定输入 ONNX
- `quantize_onnx_model.py`: 执行 ONNX Runtime 静态 PTQ
- `evaluate_onnx_model.py`: 评估 ONNX 模型并可选对比 PyTorch checkpoint
- `benchmark_onnx_model.py`: 对 ONNX Runtime backend 做简单时延基准
- `extract_geometry_from_segmentation.py`: 从分割结果提取几何参数到 CSV
- `batch_predict_and_extract_sequences.py`: 批量处理 `S_*` 序列目录
- `baseline_filter_sequences.py` / `baseline_filter_sequences_v2.py` / `baseline_filter_sequences_v3.py`: 时序异常清洗与平滑
- `analyze_sequence_geometry.py`: 生成序列分析 CSV 和曲线图
- `realtime_eye_direction.py`: 摄像头实时演示
- `realtime_eye_direction_pi.py`: 树莓派 5 + Picamera2 双 CSI 实时视线元数据
- `main.py`: 数据样本检查入口

## 常用命令

### 移动端标准预设（推荐，b8@384x240）

训练 b8 模型：

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

离线缩放数据集：

```bash
python preprocess_openeds_dataset.py \
  --input_root path/to/openeds_raw \
  --output_root path/to/openeds_384x240 \
  --splits train,validation \
  --input_width 384 \
  --input_height 240 \
  --process_mask
```

对单个图像目录做分割预测：

```bash
python predict_segmentation_to_npy.py \
  --checkpoint_path ./checkpoints/best_unet_b8_384x240_amp.pth \
  --image_dir path/to/images \
  --output_dir ./pred_validation_npy
```

导出 ONNX：

```bash
python export_unet_to_onnx.py \
  --checkpoint_path ./checkpoints/best_unet_b8_384x240_amp.pth \
  --onnx_path ./checkpoints/unet_b8_384x240_fp32.onnx
```

执行 PTQ：

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

评估 ONNX 并对比 PyTorch：

```bash
python evaluate_onnx_model.py \
  --model_path ./checkpoints/unet_b8_384x240_fp32.onnx \
  --root_dir path/to/openeds \
  --split validation \
  --checkpoint_path ./checkpoints/best_unet_b8_384x240_amp.pth
```

批量处理时序序列：

```bash
python batch_predict_and_extract_sequences.py \
  --sequence_root path/to/sequences \
  --checkpoint_path ./checkpoints/best_unet_b8_384x240_amp.pth \
  --output_root ./sequence_outputs
```

从预测结果提取几何参数：

```bash
python extract_geometry_from_segmentation.py \
  --pred_dir ./pred_validation_npy \
  --output_csv ./geometry_result.csv
```

对 `sequence_outputs` 做平滑和摘要统计：

```bash
python baseline_filter_sequences_v3.py \
  --input ./sequence_outputs \
  --output_dir ./baseline_v3_out
```

生成序列分析图：

```bash
python analyze_sequence_geometry.py \
  --input_root ./sequence_outputs \
  --output_root ./sequence_analysis
```

### 兼容基线（b16@384x240）

以下命令保留给旧 baseline / 兼容流程，仓库默认常量和默认路径仍指向 b16：

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

## 说明

- `Train.py` 和部分可视化脚本里包含本地默认路径，运行前需要按你的环境修改。
- 对于新版完整 checkpoint，PyTorch 侧训练续训、预测、可视化、评估、导出和实时脚本会优先读取 checkpoint metadata 自动恢复 `base_channels`、输入尺寸和 AMP 配置。
- 移动端当前推荐预设为 `b8@384x240`，建议使用 `best_unet_b8_384x240_amp.pth`、`unet_b8_384x240_fp32.onnx`、`unet_b8_384x240_int8_qdq.onnx` 这组产物命名。
- 量化默认会在 histogram 校准 OOM 时自动回退到 `MinMax + 较小 calibration_limit`，若不需要该行为可显式传 `--no-auto_fallback_to_minmax_on_oom`。
- 当前轻量模型默认配置为 `base_channels=16`、输入尺寸 `384x240`、预处理模式 `raw grayscale + resize + normalize`。
- 模型默认类别数为 4，代码中约定 `iris=2`、`pupil=3`。
- 结果目录如 `checkpoints/`、`sequence_outputs/`、`sequence_analysis/`、`vis_val/` 已在 `.gitignore` 中排除。

## 树莓派 5 双 CSI 实时运行

树莓派专用实时入口是 `realtime_eye_direction_pi.py`，不会复用桌面版 `realtime_eye_direction.py`。

### 树莓派额外依赖

建议在 Raspberry Pi OS 上通过系统包安装：

```bash
sudo apt install -y python3-picamera2 python3-opencv python3-gi \
  gir1.2-gst-rtsp-server-1.0 gstreamer1.0-tools \
  gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly
```

再安装树莓派脚本的 Python 依赖：

```bash
pip install -r requirements_pi.txt
```

### 启动示例

```bash
python realtime_eye_direction_pi.py \
  --model_path ./checkpoints/unet_b16_384x240_int8_qdq.onnx \
  --eye_camera_id 0 \
  --fpv_camera_id 1 \
  --feature_mode iris_only
```

默认行为：

- `cam0` 使用 `YUV420@384x240` 直接采集近眼红外图，不走 ROI，不做 CPU resize。
- `cam1` 作为 FPV 输出底图，RTSP 默认地址为 `rtsp://<pi-ip>:8554/fpv`。
- 树莓派端不会把 gaze/calibration 图形直接画进 FPV 视频；RTSP 只发送原始 FPV 视频。
- gaze、校准目标和状态会以 JSON 行输出到 stdout，可选再用 `--metadata_udp_host/--metadata_udp_port` 通过 UDP 额外发送给接收端叠加。
- 终端按键：`s` 开始校准，`x` 取消校准，`r` 重置跟踪与校准，`q` 退出。
