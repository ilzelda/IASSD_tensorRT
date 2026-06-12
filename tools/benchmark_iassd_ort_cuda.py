import argparse
import ctypes
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from validate_iassd_ort_model import (
    OnnxValidationWrapper,
    ValidationDataset,
    build_example_points,
    cfg,
    cfg_from_list,
    cfg_from_yaml_file,
    common_utils,
    install_spconv_import_stub,
    ort,
    resolve_tools_path,
    tensor_to_numpy,
    TOOLS_DIR,
)
from validate_iassd_trt_engine import (
    TRT_DTYPE_TO_TORCH,
    execute_context,
    get_tensor_dtype,
    get_tensor_name,
    get_tensor_shape,
    is_input_tensor,
    load_trt_engine,
    set_input_shape_if_needed,
)


install_spconv_import_stub()

from pcdet.models import build_network


def parse_args():
    parser = argparse.ArgumentParser(description='IA-SSD PyTorch와 ORT CUDA EP raw forward benchmark')
    parser.add_argument('--cfg_file', type=str, default='tools/cfgs/kitti_models/IA-SSD.yaml')
    parser.add_argument('--ckpt', type=str, default='tools/IA-SSD.pth')
    parser.add_argument('--onnx_file', type=str, required=True, help='benchmark할 IA-SSD ONNX 파일')
    parser.add_argument('--ort_op_library', type=str, required=True, help='libiassd_ort_ops.so 경로')
    parser.add_argument('--trt_engine_file', type=str, default=None, help='선택 측정할 direct TensorRT engine 파일')
    parser.add_argument('--trt_plugin_library', type=str, default=None, help='TensorRT plugin library 경로')
    parser.add_argument('--sample_data_path', type=str, default=None, help='샘플 포인트클라우드 파일 또는 디렉터리 경로')
    parser.add_argument('--sample_ext', type=str, default='.bin', help='샘플 포인트클라우드 확장자')
    parser.add_argument('--num_points', type=int, default=16384, help='synthetic 입력 생성 시 포인트 수')
    parser.add_argument('--warmup', type=int, default=20, help='측정 전 warmup 반복 수')
    parser.add_argument('--iterations', type=int, default=100, help='측정 반복 수')
    parser.add_argument('--providers', type=str, default='CUDAExecutionProvider', help='쉼표로 구분한 ORT provider 목록')
    parser.add_argument('--report_file', type=str, default=None, help='benchmark report JSON 저장 경로')
    parser.add_argument('--skip_pytorch', action='store_true', help='PyTorch 기준 forward 측정을 건너뜀')
    parser.add_argument('--skip_iobinding', action='store_true', help='ORT CUDA IO binding 측정을 건너뜀')
    parser.add_argument('--measure_iobinding_copy', action='store_true', help='IO binding 출력의 GPU->CPU 복사 시간도 별도 측정')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER, help='추가 config override')
    return parser.parse_args()


def load_config(cfg_file, set_cfgs):
    old_cwd = Path.cwd()
    try:
        # OpenPCDet config의 _BASE_CONFIG_는 tools/ 기준 상대경로를 사용한다.
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


def time_pytorch_forward(wrapper, points, warmup, iterations):
    with torch.no_grad():
        for _ in range(warmup):
            wrapper(points)
        torch.cuda.synchronize()

        timings = []
        for _ in range(iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            wrapper(points)
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000.0)
    return timings


def make_ort_session(onnx_file, ort_op_library, providers):
    options = ort.SessionOptions()
    options.register_custom_ops_library(str(ort_op_library))
    session = ort.InferenceSession(str(onnx_file), sess_options=options, providers=providers)
    return session


def load_trt_plugin_library(trt_plugin_library):
    if trt_plugin_library is None:
        return
    plugin_path = Path(trt_plugin_library).resolve()
    if not plugin_path.exists():
        raise FileNotFoundError(f'TensorRT plugin library가 없습니다: {plugin_path}')
    ctypes.CDLL(str(plugin_path), mode=ctypes.RTLD_GLOBAL)


def time_ort_session_run(session, points_np, warmup, iterations):
    input_name = session.get_inputs()[0].name

    for _ in range(warmup):
        session.run(None, {input_name: points_np})
    torch.cuda.synchronize()

    timings = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        session.run(None, {input_name: points_np})
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000.0)
    return timings


def make_cuda_iobinding(session, points_np, device_id):
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    ort_input = ort.OrtValue.ortvalue_from_numpy(points_np, 'cuda', device_id)
    io_binding = session.io_binding()
    io_binding.bind_ortvalue_input(input_name, ort_input)
    for output_name in output_names:
        io_binding.bind_output(output_name, 'cuda', device_id)
    return io_binding


def time_ort_iobinding_run(session, points_np, warmup, iterations, device_id):
    io_binding = make_cuda_iobinding(session, points_np, device_id)

    for _ in range(warmup):
        session.run_with_iobinding(io_binding)
    torch.cuda.synchronize()

    timings = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        session.run_with_iobinding(io_binding)
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000.0)
    return timings


def time_ort_iobinding_copy_outputs(session, points_np, warmup, iterations, device_id):
    io_binding = make_cuda_iobinding(session, points_np, device_id)

    for _ in range(warmup):
        session.run_with_iobinding(io_binding)
        io_binding.copy_outputs_to_cpu()
    torch.cuda.synchronize()

    timings = []
    for _ in range(iterations):
        session.run_with_iobinding(io_binding)
        torch.cuda.synchronize()
        start = time.perf_counter()
        io_binding.copy_outputs_to_cpu()
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000.0)
    return timings


def make_trt_runner(engine, points):
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError('TensorRT execution context 생성 실패')

    tensor_count = engine.num_io_tensors if hasattr(engine, 'num_io_tensors') else engine.num_bindings
    input_names = []
    output_names = []
    bindings = {}
    output_tensors = {}

    for index in range(tensor_count):
        name = get_tensor_name(engine, index)
        if is_input_tensor(engine, name, index):
            input_names.append(name)
        else:
            output_names.append(name)

    if len(input_names) != 1:
        raise RuntimeError(f'현재 benchmark는 입력 1개 engine만 지원합니다: inputs={input_names}')

    input_name = input_names[0]
    trt_points = points.contiguous()
    set_input_shape_if_needed(context, engine, input_name, trt_points)
    bindings[input_name] = int(trt_points.data_ptr())

    for index in range(tensor_count):
        name = get_tensor_name(engine, index)
        if name == input_name:
            continue
        shape = get_tensor_shape(engine, context, name, index)
        trt_dtype = get_tensor_dtype(engine, name, index)
        torch_dtype = TRT_DTYPE_TO_TORCH.get(trt_dtype)
        if torch_dtype is None:
            raise RuntimeError(f'지원하지 않는 TensorRT dtype입니다: {name} {trt_dtype}')
        output_tensor = torch.empty(shape, dtype=torch_dtype, device=trt_points.device)
        output_tensors[name] = output_tensor
        bindings[name] = int(output_tensor.data_ptr())

    return context, bindings, output_names, output_tensors


def time_trt_engine(engine, points, warmup, iterations):
    context, bindings, _, _ = make_trt_runner(engine, points)
    stream = torch.cuda.current_stream().cuda_stream

    for _ in range(warmup):
        ok = execute_context(context, engine, bindings, stream)
        if not ok:
            raise RuntimeError('TensorRT engine 실행 실패')
    torch.cuda.synchronize()

    timings = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        ok = execute_context(context, engine, bindings, stream)
        if not ok:
            raise RuntimeError('TensorRT engine 실행 실패')
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000.0)
    return timings


def build_runtime_report(args, cfg_file, ckpt_file, onnx_file, ort_op_library, points, providers, session):
    cuda_device = torch.cuda.current_device()
    return {
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
            'onnxruntime': ort.__version__,
        },
        'config': {
            'cfg_file': str(cfg_file),
            'ckpt': str(ckpt_file),
            'onnx_file': str(onnx_file),
            'ort_op_library': str(ort_op_library),
            'trt_engine_file': str(Path(args.trt_engine_file).resolve()) if args.trt_engine_file else None,
            'trt_plugin_library': str(Path(args.trt_plugin_library).resolve()) if args.trt_plugin_library else None,
            'providers_requested': providers,
            'providers_active': session.get_providers(),
            'num_points': args.num_points,
            'warmup': args.warmup,
            'iterations': args.iterations,
            'post_processing_included': False,
            'preprocessing_included': False,
            'ort_session_run_includes_numpy_io': True,
            'ort_iobinding_reuses_cuda_input': True,
            'ort_iobinding_outputs_on_cuda': True,
        },
        'input_points': {
            'shape': list(points.shape),
            'dtype': str(points.dtype),
            'device': str(points.device),
        },
    }


def main():
    args = parse_args()
    logger = common_utils.create_logger()

    onnx_file = Path(args.onnx_file).resolve()
    ort_op_library = Path(args.ort_op_library).resolve()
    cfg_file = resolve_tools_path(args.cfg_file).resolve()
    ckpt_file = resolve_tools_path(args.ckpt).resolve()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA를 사용할 수 없습니다.')
    if not onnx_file.exists():
        raise FileNotFoundError(f'ONNX 파일이 없습니다: {onnx_file}')
    if not ort_op_library.exists():
        raise FileNotFoundError(f'custom op library가 없습니다: {ort_op_library}')
    if args.trt_engine_file is not None and args.trt_plugin_library is None:
        raise ValueError('--trt_engine_file을 쓰려면 --trt_plugin_library가 필요합니다.')
    if not cfg_file.exists():
        raise FileNotFoundError(f'config 파일이 없습니다: {cfg_file}')
    if not ckpt_file.exists():
        raise FileNotFoundError(f'checkpoint 파일이 없습니다: {ckpt_file}')
    load_trt_plugin_library(args.trt_plugin_library)

    load_config(cfg_file, args.set_cfgs)
    dataset = ValidationDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        root_path=Path(args.sample_data_path).parent if args.sample_data_path is not None else Path.cwd(),
        sample_data_path=args.sample_data_path,
        ext=args.sample_ext,
        num_points=args.num_points,
    )

    points = build_example_points(dataset).cuda()
    points_np = tensor_to_numpy(points)
    providers = [provider.strip() for provider in args.providers.split(',') if provider.strip()]
    session = make_ort_session(onnx_file, ort_op_library, providers)

    report = build_runtime_report(args, cfg_file, ckpt_file, onnx_file, ort_op_library, points, providers, session)

    if not args.skip_pytorch:
        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
        model.load_params_from_file(filename=str(ckpt_file), logger=logger, to_cpu=True)
        model.cuda()
        model.eval()
        wrapper = OnnxValidationWrapper(model).cuda()
        wrapper.eval()
        report['pytorch_raw_forward'] = summarize_ms(
            time_pytorch_forward(wrapper, points, args.warmup, args.iterations)
        )

    report['ort_cuda_session_run'] = summarize_ms(
        time_ort_session_run(session, points_np, args.warmup, args.iterations)
    )

    if not args.skip_iobinding:
        cuda_device = torch.cuda.current_device()
        report['ort_cuda_iobinding_run'] = summarize_ms(
            time_ort_iobinding_run(session, points_np, args.warmup, args.iterations, cuda_device)
        )
        if args.measure_iobinding_copy:
            report['ort_cuda_iobinding_copy_outputs_to_cpu'] = summarize_ms(
                time_ort_iobinding_copy_outputs(
                    session,
                    points_np,
                    args.warmup,
                    args.iterations,
                    cuda_device,
                )
            )

    if args.trt_engine_file is not None:
        engine_path, plugin_path, engine = load_trt_engine(args.trt_engine_file, args.trt_plugin_library)
        report['config']['trt_engine_file'] = str(engine_path)
        report['config']['trt_plugin_library'] = str(plugin_path)
        report['direct_trt_engine_run'] = summarize_ms(
            time_trt_engine(engine, points, args.warmup, args.iterations)
        )

    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.report_file is not None:
        report_path = Path(args.report_file)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
