import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT_DIR / 'tools'
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import numpy as np
import torch

from iassd_postprocess import postprocess_outputs
from iassd_trt_runtime import IASSDTensorRTRunner
from validate_iassd_trt_engine import (
    TRT_DTYPE_TO_TORCH,
    OnnxValidationWrapper,
    ValidationDataset,
    build_example_points,
    cfg,
    cfg_from_list,
    cfg_from_yaml_file,
    common_utils,
    execute_context,
    get_tensor_dtype,
    get_tensor_name,
    get_tensor_shape,
    is_input_tensor,
    load_trt_engine,
    resolve_tools_path,
    set_input_shape_if_needed,
    tensor_to_numpy,
)

from pcdet.models import build_network


def parse_args():
    parser = argparse.ArgumentParser(description='IA-SSD PyTorch raw forward와 direct TensorRT 실행 시간 분리 측정')
    parser.add_argument('--cfg_file', type=str, default='tools/cfgs/kitti_models/IA-SSD.yaml')
    parser.add_argument('--ckpt', type=str, default='tools/IA-SSD.pth')
    parser.add_argument('--engine_file', type=str, required=True)
    parser.add_argument('--plugin_library', type=str, required=True)
    parser.add_argument('--sample_data_path', type=str, default=None)
    parser.add_argument('--sample_ext', type=str, default='.bin')
    parser.add_argument('--num_points', type=int, default=16384)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--score_thresh', type=float, default=None)
    parser.add_argument('--report_file', type=str, default=None)
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


def load_config(cfg_file, set_cfgs):
    old_cwd = Path.cwd()
    try:
        import os
        os.chdir(TOOLS_DIR)
        cfg_from_yaml_file(str(cfg_file), cfg)
    finally:
        os.chdir(old_cwd)
    if set_cfgs is not None:
        cfg_from_list(set_cfgs, cfg)


def percentile(values, ratio):
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int(round((len(sorted_values) - 1) * ratio))
    return float(sorted_values[index])


def summarize_ms(values):
    return {
        'iterations': len(values),
        'min_ms': float(min(values)) if values else 0.0,
        'max_ms': float(max(values)) if values else 0.0,
        'mean_ms': float(statistics.mean(values)) if values else 0.0,
        'median_ms': float(statistics.median(values)) if values else 0.0,
        'p90_ms': percentile(values, 0.90),
        'p95_ms': percentile(values, 0.95),
        'fps_from_mean': float(1000.0 / statistics.mean(values)) if values else 0.0,
    }


def time_call(fn, warmup, iterations, sync_cuda=True):
    for _ in range(warmup):
        fn()
    if sync_cuda:
        torch.cuda.synchronize()

    timings = []
    for _ in range(iterations):
        if sync_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if sync_cuda:
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000.0)
    return timings


def build_dataset(args):
    return ValidationDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        root_path=Path(args.sample_data_path).parent if args.sample_data_path is not None else Path.cwd(),
        sample_data_path=args.sample_data_path,
        ext=args.sample_ext,
        num_points=args.num_points,
    )


def build_torch_wrapper(args, dataset):
    logger = common_utils.create_logger()
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=str(resolve_tools_path(args.ckpt).resolve()), logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    wrapper = OnnxValidationWrapper(model).cuda()
    wrapper.eval()
    return wrapper


def make_trt_gpu_runner(engine, points):
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError('TensorRT execution context 생성 실패')

    tensor_count = engine.num_io_tensors if hasattr(engine, 'num_io_tensors') else engine.num_bindings
    input_names = []
    bindings = {}

    for index in range(tensor_count):
        name = get_tensor_name(engine, index)
        if is_input_tensor(engine, name, index):
            input_names.append(name)

    if len(input_names) != 1:
        raise RuntimeError(f'입력 1개 engine만 지원합니다: {input_names}')

    input_name = input_names[0]
    trt_points = points.contiguous()
    set_input_shape_if_needed(context, engine, input_name, trt_points)
    bindings[input_name] = int(trt_points.data_ptr())

    output_tensors = {}
    for index in range(tensor_count):
        name = get_tensor_name(engine, index)
        if name == input_name:
            continue
        trt_dtype = get_tensor_dtype(engine, name, index)
        torch_dtype = TRT_DTYPE_TO_TORCH.get(trt_dtype)
        if torch_dtype is None:
            raise RuntimeError(f'지원하지 않는 TensorRT dtype입니다: {name} {trt_dtype}')
        output_tensor = torch.empty(get_tensor_shape(engine, context, name, index), dtype=torch_dtype, device=trt_points.device)
        output_tensors[name] = output_tensor
        bindings[name] = int(output_tensor.data_ptr())

    stream = torch.cuda.current_stream().cuda_stream

    def run():
        ok = execute_context(context, engine, bindings, stream)
        if not ok:
            raise RuntimeError('TensorRT engine 실행 실패')
        return output_tensors

    return run


def time_postprocess(outputs, args):
    post_cfg = cfg.MODEL.POST_PROCESSING
    nms_cfg = post_cfg.NMS_CONFIG
    score_thresh = args.score_thresh if args.score_thresh is not None else post_cfg.SCORE_THRESH

    def run():
        return postprocess_outputs(
            outputs,
            score_thresh=score_thresh,
            nms_thresh=float(nms_cfg.NMS_THRESH),
            nms_pre_maxsize=int(nms_cfg.NMS_PRE_MAXSIZE),
            nms_post_maxsize=int(nms_cfg.NMS_POST_MAXSIZE),
        )

    return time_call(run, args.warmup, args.iterations, sync_cuda=False), run()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA를 사용할 수 없습니다.')

    cfg_file = resolve_tools_path(args.cfg_file).resolve()
    load_config(cfg_file, args.set_cfgs)
    dataset = build_dataset(args)
    points = build_example_points(dataset).cuda()

    wrapper = build_torch_wrapper(args, dataset)
    with torch.no_grad():
        torch_timings = time_call(lambda: wrapper(points), args.warmup, args.iterations)
        torch_outputs = wrapper(points)
        torch_outputs_np = {
            'batch_cls_preds': tensor_to_numpy(torch_outputs[0]),
            'batch_box_preds': tensor_to_numpy(torch_outputs[1]),
        }

    engine_path, plugin_path, engine = load_trt_engine(args.engine_file, args.plugin_library)
    trt_gpu_run = make_trt_gpu_runner(engine, points)
    trt_gpu_timings = time_call(trt_gpu_run, args.warmup, args.iterations)

    # ROS TensorRT 노드가 실제 사용하는 wrapper 경로다. 여기에는 output GPU->CPU 복사가 포함된다.
    trt_wrapper = IASSDTensorRTRunner(args.engine_file, args.plugin_library)
    trt_wrapper_timings = time_call(lambda: trt_wrapper.infer(points), args.warmup, args.iterations, sync_cuda=False)
    trt_outputs_np = trt_wrapper.infer(points)

    torch_post_timings, torch_detections = time_postprocess(torch_outputs_np, args)
    trt_post_timings, trt_detections = time_postprocess(trt_outputs_np, args)

    cuda_device = torch.cuda.current_device()
    report = {
        'hardware': {
            'platform': platform.platform(),
            'machine': platform.machine(),
            'cuda_device': torch.cuda.get_device_name(cuda_device),
            'cuda_device_index': cuda_device,
        },
        'software': {
            'python': platform.python_version(),
            'torch': torch.__version__,
            'torch_cuda': torch.version.cuda,
        },
        'config': {
            'cfg_file': str(cfg_file),
            'ckpt': str(resolve_tools_path(args.ckpt).resolve()),
            'engine_file': str(engine_path),
            'plugin_library': str(plugin_path),
            'num_points': args.num_points,
            'warmup': args.warmup,
            'iterations': args.iterations,
            'preprocessing_included': False,
            'post_processing_included_in_raw_forward': False,
            'trt_gpu_only_excludes_output_cpu_copy': True,
            'trt_wrapper_includes_output_cpu_copy': True,
        },
        'input_points': {
            'shape': list(points.shape),
            'dtype': str(points.dtype),
            'device': str(points.device),
        },
        'pytorch_raw_forward': summarize_ms(torch_timings),
        'direct_trt_gpu_only': summarize_ms(trt_gpu_timings),
        'direct_trt_wrapper_with_cpu_copy': summarize_ms(trt_wrapper_timings),
        'pytorch_numpy_postprocess': summarize_ms(torch_post_timings),
        'trt_numpy_postprocess': summarize_ms(trt_post_timings),
        'detections': {
            'pytorch_count': int(torch_detections['pred_scores'].shape[0]),
            'trt_count': int(trt_detections['pred_scores'].shape[0]),
        },
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.report_file is not None:
        report_path = Path(args.report_file)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
