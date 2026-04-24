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
    summary: resize modelinput
    param image:, 0~1
    param input_width: output
    param input_height: output
    return: resize float32
    """
    height, width = image.shape[:2]
    if width == input_width and height == input_height:
        return image.astype(np.float32, copy=False)

    resized = cv2.resize(image, (input_width, input_height), interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float32)


def resize_label_map(label: np.ndarray, input_width: int, input_height: int) -> np.ndarray:
    """
    summary: label resize modelinput
    param label: label
    param input_width: output
    param input_height: output
    return: resize int64 label
    """
    height, width = label.shape[:2]
    if width == input_width and height == input_height:
        return label.astype(np.int64, copy=False)

    resized = cv2.resize(label.astype(np.int32), (input_width, input_height), interpolation=cv2.INTER_NEAREST)
    return resized.astype(np.int64)


def resize_binary_mask(mask: np.ndarray, input_width: int, input_height: int) -> np.ndarray:
    """
    summary: mask resize modelinput
    param mask: mask
    param input_width: output
    param input_height: output
    return: resize uint8 mask
    """
    height, width = mask.shape[:2]
    if width == input_width and height == input_height:
        return (mask > 0).astype(np.uint8, copy=False)

    resized = cv2.resize(mask.astype(np.uint8), (input_width, input_height), interpolation=cv2.INTER_NEAREST)
    return (resized > 0).astype(np.uint8)


def normalize_gray_image(image: np.ndarray) -> np.ndarray:
    """
    summary: 0~1 float32
    param image: inputimage
    return:
    """
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    return image


def preprocess_gray_image(image: np.ndarray, input_width: int, input_height: int) -> np.ndarray:
    """
    summary: raw-resize process
    param image: input
    param input_width: output
    param input_height: output
    return: process float32 image
    """
    normalized = normalize_gray_image(image)
    return resize_gray_image(normalized, input_width=input_width, input_height=input_height)


def preprocess_gray_image_to_tensor(image: np.ndarray, input_width: int, input_height: int) -> torch.Tensor:
    """
    summary: process BxCxHxW sample
    param image: input
    param input_width: output
    param input_height: output
    return: 1xHxW float32
    """
    processed = preprocess_gray_image(image=image, input_width=input_width, input_height=input_height)
    return torch.from_numpy(processed).unsqueeze(0).float()


def preprocess_bgr_frame(
    frame_bgr: np.ndarray,
    input_width: int,
    input_height: int,
) -> tuple[np.ndarray, torch.Tensor, ResizeMeta]:
    """
    summary: BGR raw-resize process
    param frame_bgr: input BGR image
    param input_width: modelinput
    param input_height: modelinput
    return:, modelinput,
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
