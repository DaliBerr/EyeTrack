import argparse
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image

from eyetrack.config import DEFAULT_INPUT_HEIGHT, DEFAULT_INPUT_WIDTH, DEFAULT_ONNX_PATH
from eyetrack.data.preprocessing import preprocess_gray_image
from eyetrack.deployment.onnx_tools import build_onnx_session, load_manifest_paths, summarize_profile_providers


def benchmark_onnx_model(
    model_path: str = DEFAULT_ONNX_PATH,
    backend: str = "cpu",
    input_manifest: str | None = None,
    input_width: int = DEFAULT_INPUT_WIDTH,
    input_height: int = DEFAULT_INPUT_HEIGHT,
    warmup_runs: int = 20,
    benchmark_runs: int = 100,
    enable_profiling: bool = False,
) -> None:
    """
    summary: 对 ONNX 模型做简单时延基准
    param model_path: ONNX 模型路径
    param backend: cpu 或 nnapi
    param input_manifest: 文本清单，每行一个图像路径
    param input_width: 模型输入宽度
    param input_height: 模型输入高度
    param warmup_runs: 预热轮数
    param benchmark_runs: 正式计时轮数
    param enable_profiling: 是否开启 ORT profiling
    return: 无
    """
    if input_manifest is None:
        raise ValueError("必须提供 --input_manifest，且每行一个图像路径。")

    image_paths = load_manifest_paths(input_manifest)
    session = build_onnx_session(model_path=model_path, backend=backend, enable_profiling=enable_profiling)
    input_name = session.get_inputs()[0].name

    def load_input(index: int) -> np.ndarray:
        image_path = Path(image_paths[index % len(image_paths)])
        image = Image.open(image_path).convert("L")
        image = np.array(image, dtype=np.float32)
        image = preprocess_gray_image(image=image, input_width=input_width, input_height=input_height)
        return np.expand_dims(np.expand_dims(image, axis=0), axis=0).astype(np.float32)

    for run_idx in range(warmup_runs):
        session.run(None, {input_name: load_input(run_idx)})

    latencies_ms = []
    for run_idx in range(benchmark_runs):
        inputs = load_input(run_idx)
        start_time = time.perf_counter()
        session.run(None, {input_name: inputs})
        end_time = time.perf_counter()
        latencies_ms.append((end_time - start_time) * 1000.0)

    profile_path = session.end_profiling() if enable_profiling else None
    provider_summary = summarize_profile_providers(profile_path) if profile_path is not None else {}

    print("backend:", backend)
    print("available providers:", session.get_providers())
    print(f"runs: warmup={warmup_runs}, benchmark={benchmark_runs}")
    print(f"latency mean : {statistics.mean(latencies_ms):.3f} ms")
    print(f"latency median: {statistics.median(latencies_ms):.3f} ms")
    print(f"latency p90  : {np.percentile(latencies_ms, 90):.3f} ms")
    if profile_path is not None:
        print("profile path:", profile_path)
        print("provider summary:", provider_summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对 ONNX 分割模型执行简单时延基准")
    parser.add_argument("--model_path", type=str, default=DEFAULT_ONNX_PATH, help="ONNX 模型路径")
    parser.add_argument("--backend", type=str, default="cpu", choices=["cpu", "nnapi"], help="ONNX Runtime backend")
    parser.add_argument("--input_manifest", type=str, required=True, help="文本清单，每行一个图像路径")
    parser.add_argument("--input_width", type=int, default=DEFAULT_INPUT_WIDTH, help="模型输入宽度")
    parser.add_argument("--input_height", type=int, default=DEFAULT_INPUT_HEIGHT, help="模型输入高度")
    parser.add_argument("--warmup_runs", type=int, default=20, help="预热轮数")
    parser.add_argument("--benchmark_runs", type=int, default=100, help="正式计时轮数")
    parser.add_argument("--enable_profiling", action="store_true", help="启用 ORT profiling")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_onnx_model(
        model_path=args.model_path,
        backend=args.backend,
        input_manifest=args.input_manifest,
        input_width=args.input_width,
        input_height=args.input_height,
        warmup_runs=args.warmup_runs,
        benchmark_runs=args.benchmark_runs,
        enable_profiling=args.enable_profiling,
    )


if __name__ == "__main__":
    main()
