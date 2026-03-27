# EyeTrack

基于 PyTorch 的眼部语义分割与几何分析项目，包含：

- OpenEDS 风格数据集上的 U-Net 训练与验证
- 灰度眼图分割预测，输出标签图 `.npy`
- 从分割结果中提取 iris / pupil 几何参数
- 序列级批处理、异常帧清洗、平滑与可视化分析
- 摄像头实时视线方向演示

## 依赖

建议使用 Python 3.10+。

```bash
pip install torch numpy opencv-python pillow matplotlib pandas
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
- `extract_geometry_from_segmentation.py`: 从分割结果提取几何参数到 CSV
- `batch_predict_and_extract_sequences.py`: 批量处理 `S_*` 序列目录
- `baseline_filter_sequences.py` / `baseline_filter_sequences_v2.py` / `baseline_filter_sequences_v3.py`: 时序异常清洗与平滑
- `analyze_sequence_geometry.py`: 生成序列分析 CSV 和曲线图
- `realtime_eye_direction.py`: 摄像头实时演示
- `main.py`: 数据样本检查入口

## 常用命令

对单个图像目录做分割预测：

```bash
python predict_segmentation_to_npy.py \
  --checkpoint_path ./checkpoints/best_unet_openeds.pth \
  --image_dir path/to/images \
  --output_dir ./pred_validation_npy
```

从预测结果提取几何参数：

```bash
python extract_geometry_from_segmentation.py \
  --pred_dir ./pred_validation_npy \
  --output_csv ./geometry_result.csv
```

批量处理时序序列：

```bash
python batch_predict_and_extract_sequences.py \
  --sequence_root path/to/sequences \
  --checkpoint_path ./checkpoints/best_unet_openeds.pth \
  --output_root ./sequence_outputs
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

## 说明

- `Train.py` 和部分可视化脚本里包含本地默认路径，运行前需要按你的环境修改。
- 模型默认类别数为 4，代码中约定 `iris=2`、`pupil=3`。
- 结果目录如 `checkpoints/`、`sequence_outputs/`、`sequence_analysis/`、`vis_val/` 已在 `.gitignore` 中排除。
