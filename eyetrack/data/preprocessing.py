from dataclasses import dataclass

import cv2
import numpy as np
import torch


@dataclass
class ResizeMeta:
    source_width: int
    source_height: int
    input_width: int
    input_height: int
    scale_x: float
    scale_y: float


def resize_gray_image(image: np.ndarray, input_width: int, input_height: int) -> np.ndarray:
    """
    summary: 将灰度图 resize 到模型输入尺寸
    param image: 原始灰度图，范围建议为 0~1
    param input_width: 输出宽度
    param input_height: 输出高度
    return: resize 后的 float32 灰度图
    """
    height, width = image.shape[:2]
    if width == input_width and height == input_height:
        return image.astype(np.float32, copy=False)

    resized = cv2.resize(image, (input_width, input_height), interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float32)


def resize_label_map(label: np.ndarray, input_width: int, input_height: int) -> np.ndarray:
    """
    summary: 将标签图按最近邻 resize 到模型输入尺寸
    param label: 原始标签图
    param input_width: 输出宽度
    param input_height: 输出高度
    return: resize 后的 int64 标签图
    """
    height, width = label.shape[:2]
    if width == input_width and height == input_height:
        return label.astype(np.int64, copy=False)

    resized = cv2.resize(label.astype(np.int32), (input_width, input_height), interpolation=cv2.INTER_NEAREST)
    return resized.astype(np.int64)


def resize_binary_mask(mask: np.ndarray, input_width: int, input_height: int) -> np.ndarray:
    """
    summary: 将二值 mask 按最近邻 resize 到模型输入尺寸
    param mask: 原始 mask
    param input_width: 输出宽度
    param input_height: 输出高度
    return: resize 后的 uint8 二值 mask
    """
    height, width = mask.shape[:2]
    if width == input_width and height == input_height:
        return (mask > 0).astype(np.uint8, copy=False)

    resized = cv2.resize(mask.astype(np.uint8), (input_width, input_height), interpolation=cv2.INTER_NEAREST)
    return (resized > 0).astype(np.uint8)


def normalize_gray_image(image: np.ndarray) -> np.ndarray:
    """
    summary: 将灰度图规范化到 0~1 float32
    param image: 输入图像
    return: 归一化结果
    """
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    return image


def preprocess_gray_image(image: np.ndarray, input_width: int, input_height: int) -> np.ndarray:
    """
    summary: 对灰度图执行 raw-resize 预处理
    param image: 输入灰度图
    param input_width: 输出宽度
    param input_height: 输出高度
    return: 预处理后的 float32 图像
    """
    normalized = normalize_gray_image(image)
    return resize_gray_image(normalized, input_width=input_width, input_height=input_height)


def preprocess_gray_image_to_tensor(image: np.ndarray, input_width: int, input_height: int) -> torch.Tensor:
    """
    summary: 将灰度图预处理为 BxCxHxW 中的单样本张量
    param image: 输入灰度图
    param input_width: 输出宽度
    param input_height: 输出高度
    return: 形状为 1xHxW 的 float32 张量
    """
    processed = preprocess_gray_image(image=image, input_width=input_width, input_height=input_height)
    return torch.from_numpy(processed).unsqueeze(0).float()


def preprocess_bgr_frame(
    frame_bgr: np.ndarray,
    input_width: int,
    input_height: int,
) -> tuple[np.ndarray, torch.Tensor, ResizeMeta]:
    """
    summary: 对 BGR 帧执行灰度化与 raw-resize 预处理
    param frame_bgr: 输入 BGR 图像
    param input_width: 模型输入宽度
    param input_height: 模型输入高度
    return: 预览灰度图、模型输入张量、映射元信息
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    processed = preprocess_gray_image(gray, input_width=input_width, input_height=input_height)

    source_height, source_width = gray.shape[:2]
    meta = ResizeMeta(
        source_width=source_width,
        source_height=source_height,
        input_width=input_width,
        input_height=input_height,
        scale_x=source_width / float(input_width),
        scale_y=source_height / float(input_height),
    )

    preview = (processed * 255.0).clip(0, 255).astype(np.uint8)
    tensor = torch.from_numpy(processed).unsqueeze(0).unsqueeze(0).float()
    return preview, tensor, meta
