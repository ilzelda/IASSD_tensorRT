import argparse
import ctypes
import glob
import json
import sys
import types
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT_DIR / 'tools'
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import numpy as np
import tensorrt as trt
import torch
import torch.nn as nn

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate


def install_spconv_import_stub():
    try:
        import spconv  # noqa: F401
        return
    except ImportError:
        pass

    class SparseConvolution(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            raise RuntimeError('이 검증 환경에는 spconv이 설치되어 있지 않아 sparse convolution 모델은 실행할 수 없습니다.')

    class SparseModule(nn.Module):
        pass

    class SparseConvTensor:
        def __init__(self, *args, **kwargs):
            raise RuntimeError('이 검증 환경에는 spconv이 설치되어 있지 않아 SparseConvTensor를 사용할 수 없습니다.')

    spconv_stub = types.ModuleType('spconv')
    spconv_pytorch_stub = types.ModuleType('spconv.pytorch')
    conv_stub = types.SimpleNamespace(SparseConvolution=SparseConvolution)

    for module in (spconv_stub, spconv_pytorch_stub):
        module.conv = conv_stub
        module.SparseModule = SparseModule
        module.SparseSequential = nn.Sequential
        module.SparseConvTensor = SparseConvTensor
        module.SubMConv3d = SparseConvolution
        module.SparseConv3d = SparseConvolution
        module.SparseInverseConv3d = SparseConvolution

    sys.modules['spconv'] = spconv_stub
    sys.modules['spconv.pytorch'] = spconv_pytorch_stub


install_spconv_import_stub()

from pcdet.models import build_network
from pcdet.utils import common_utils


class ValidationDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, root_path, sample_data_path=None, ext='.bin', num_points=16384):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=False, root_path=root_path)
        self.ext = ext
        self.num_points = num_points
        self.sample_data_path = Path(sample_data_path) if sample_data_path is not None else None
        self.sample_file_list = []

        if self.sample_data_path is not None:
            if self.sample_data_path.is_dir():
                self.sample_file_list = glob.glob(str(self.sample_data_path / f'*{self.ext}'))
                self.sample_file_list.sort()
            else:
                self.sample_file_list = [str(self.sample_data_path)]

    def __len__(self):
        return max(1, len(self.sample_file_list))

    def _generate_synthetic_points(self):
        rng = np.random.default_rng(1024)
        raw_feature_dim = len(self.point_feature_encoder.src_feature_list)
        points = np.zeros((self.num_points, raw_feature_dim), dtype=np.float32)

        point_cloud_range = self.point_cloud_range
        points[:, 0] = rng.uniform(point_cloud_range[0], point_cloud_range[3], size=self.num_points)
        points[:, 1] = rng.uniform(point_cloud_range[1], point_cloud_range[4], size=self.num_points)
        points[:, 2] = rng.uniform(point_cloud_range[2], point_cloud_range[5], size=self.num_points)

        if raw_feature_dim > 3:
            points[:, 3:] = rng.random((self.num_points, raw_feature_dim - 3), dtype=np.float32)

        return points

    def __getitem__(self, index):
        raw_feature_dim = len(self.point_feature_encoder.src_feature_list)

        if self.sample_file_list:
            sample_file = self.sample_file_list[index % len(self.sample_file_list)]
            if sample_file.endswith('.bin'):
                points = np.fromfile(sample_file, dtype=np.float32).reshape(-1, raw_feature_dim)
            elif sample_file.endswith('.npy'):
                points = np.load(sample_file)
            else:
                raise NotImplementedError(f'지원하지 않는 입력 형식입니다: {sample_file}')
        else:
            points = self._generate_synthetic_points()

        return self.prepare_data(
            data_dict={
                'points': points,
                'frame_id': index,
            }
        )


class OnnxValidationWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, points):
        batch_dict = {
            'points': points,
            'batch_size': 1,
        }
        for cur_module in self.model.module_list:
            batch_dict = cur_module(batch_dict)
        return batch_dict['batch_cls_preds'], batch_dict['batch_box_preds']


def build_example_points(dataset):
    sample_dict = dataset[0]
    batch_dict = dataset.collate_batch([sample_dict])
    return torch.from_numpy(batch_dict['points']).float()


def tensor_to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def describe_array(array):
    return {
        'shape': list(array.shape),
        'dtype': str(array.dtype),
    }


def compare_outputs(reference_outputs, target_outputs):
    report = {}
    for name, reference_output in reference_outputs.items():
        target_output = target_outputs[name]
        diff = target_output.astype(np.float64) - reference_output.astype(np.float64)
        report[name] = {
            'reference': describe_array(reference_output),
            'target': describe_array(target_output),
            'shape_match': list(reference_output.shape) == list(target_output.shape),
            'max_abs_error': float(np.max(np.abs(diff))) if diff.size else 0.0,
            'mean_abs_error': float(np.mean(np.abs(diff))) if diff.size else 0.0,
        }
    return report


TRT_DTYPE_TO_TORCH = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT32: torch.int32,
    trt.DataType.BOOL: torch.bool,
}


def parse_args():
    parser = argparse.ArgumentParser(description='IA-SSD direct TensorRT engine 실행 검증')
    parser.add_argument('--cfg_file', type=str, default='tools/cfgs/kitti_models/IA-SSD.yaml')
    parser.add_argument('--ckpt', type=str, default='tools/IA-SSD.pth')
    parser.add_argument('--engine_file', type=str, required=True, help='검증할 TensorRT engine 파일')
    parser.add_argument('--plugin_library', type=str, required=True, help='libiassd_trt_plugins.so 경로')
    parser.add_argument('--ort_onnx_file', type=str, default=None, help='선택 비교용 ONNX 파일')
    parser.add_argument('--ort_op_library', type=str, default=None, help='선택 비교용 libiassd_ort_ops.so 경로')
    parser.add_argument('--sample_data_path', type=str, default=None, help='샘플 포인트클라우드 파일 또는 디렉터리 경로')
    parser.add_argument('--sample_ext', type=str, default='.bin', help='샘플 포인트클라우드 확장자')
    parser.add_argument('--num_points', type=int, default=16384, help='synthetic 입력 생성 시 포인트 수')
    parser.add_argument('--skip_torch', action='store_true', help='PyTorch raw output 비교를 생략')
    parser.add_argument('--skip_ort', action='store_true', help='ORT CUDA raw output 비교를 생략')
    parser.add_argument('--report_file', type=str, default=None, help='shape/error report JSON 저장 경로')
    parser.add_argument('--dump_outputs_npz', type=str, default=None, help='PyTorch/ORT/TRT raw output tensor를 NPZ로 저장')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER, help='추가 config override')
    return parser.parse_args()


def resolve_tools_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    root_path = ROOT_DIR / path
    if root_path.exists():
        return root_path
    return TOOLS_DIR / path


def load_plugin_library(plugin_library):
    plugin_path = Path(plugin_library).resolve()
    if not plugin_path.exists():
        raise FileNotFoundError(f'TensorRT plugin library가 없습니다: {plugin_path}')
    ctypes.CDLL(str(plugin_path), mode=ctypes.RTLD_GLOBAL)
    return plugin_path


def load_trt_engine(engine_file, plugin_library):
    engine_path = Path(engine_file).resolve()
    if not engine_path.exists():
        raise FileNotFoundError(f'TensorRT engine 파일이 없습니다: {engine_path}')
    plugin_path = load_plugin_library(plugin_library)
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError('TensorRT engine deserialize 실패')
    return engine_path, plugin_path, engine


def get_tensor_name(engine, index):
    if hasattr(engine, 'get_tensor_name'):
        return engine.get_tensor_name(index)
    return engine.get_binding_name(index)


def get_tensor_shape(engine, context, name, index):
    if hasattr(engine, 'get_tensor_shape'):
        shape = tuple(int(dim) for dim in engine.get_tensor_shape(name))
    else:
        shape = tuple(int(dim) for dim in engine.get_binding_shape(index))
    if any(dim < 0 for dim in shape):
        if hasattr(context, 'get_tensor_shape'):
            shape = tuple(int(dim) for dim in context.get_tensor_shape(name))
        else:
            shape = tuple(int(dim) for dim in context.get_binding_shape(index))
    return shape


def get_tensor_dtype(engine, name, index):
    if hasattr(engine, 'get_tensor_dtype'):
        return engine.get_tensor_dtype(name)
    return engine.get_binding_dtype(index)


def is_input_tensor(engine, name, index):
    if hasattr(engine, 'get_tensor_mode'):
        return engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
    return engine.binding_is_input(index)


def set_input_shape_if_needed(context, engine, input_name, input_tensor):
    shape = tuple(int(dim) for dim in input_tensor.shape)
    if hasattr(context, 'set_input_shape'):
        engine_shape = tuple(int(dim) for dim in engine.get_tensor_shape(input_name))
        if any(dim < 0 for dim in engine_shape):
            context.set_input_shape(input_name, shape)
    else:
        binding_index = engine.get_binding_index(input_name)
        engine_shape = tuple(int(dim) for dim in engine.get_binding_shape(binding_index))
        if any(dim < 0 for dim in engine_shape):
            context.set_binding_shape(binding_index, shape)


def execute_context(context, engine, bindings, stream):
    if hasattr(context, 'execute_async_v3'):
        for index in range(engine.num_io_tensors):
            name = get_tensor_name(engine, index)
            context.set_tensor_address(name, bindings[name])
        return context.execute_async_v3(stream_handle=stream)
    binding_list = [0] * engine.num_bindings
    for index in range(engine.num_bindings):
        name = get_tensor_name(engine, index)
        binding_list[index] = bindings[name]
    return context.execute_async_v2(bindings=binding_list, stream_handle=stream)


def run_trt(engine, points):
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
        raise RuntimeError(f'현재 스크립트는 입력 1개 engine만 지원합니다: inputs={input_names}')

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

    stream = torch.cuda.current_stream().cuda_stream
    ok = execute_context(context, engine, bindings, stream)
    if not ok:
        raise RuntimeError('TensorRT engine 실행 실패')
    torch.cuda.synchronize()

    return {
        name: tensor_to_numpy(output_tensors[name])
        for name in output_names
    }


def build_dataset(args):
    cfg_file = resolve_tools_path(args.cfg_file).resolve()
    old_cwd = Path.cwd()
    try:
        import os
        os.chdir(TOOLS_DIR)
        cfg_from_yaml_file(str(cfg_file), cfg)
    finally:
        os.chdir(old_cwd)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    dataset = ValidationDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        root_path=Path(args.sample_data_path).parent if args.sample_data_path is not None else Path.cwd(),
        sample_data_path=args.sample_data_path,
        ext=args.sample_ext,
        num_points=args.num_points,
    )
    return cfg_file, dataset


def run_torch_model(args, dataset, points):
    cfg_file = resolve_tools_path(args.cfg_file).resolve()
    ckpt_file = resolve_tools_path(args.ckpt).resolve()
    if not ckpt_file.exists():
        raise FileNotFoundError(f'checkpoint 파일이 없습니다: {ckpt_file}')

    logger = common_utils.create_logger()
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=str(ckpt_file), logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    wrapper = OnnxValidationWrapper(model).cuda()
    wrapper.eval()

    with torch.no_grad():
        batch_cls_preds, batch_box_preds = wrapper(points)
    torch.cuda.synchronize()
    return cfg_file, ckpt_file, {
        'batch_cls_preds': tensor_to_numpy(batch_cls_preds),
        'batch_box_preds': tensor_to_numpy(batch_box_preds),
    }


def run_ort_cuda(onnx_file, ort_op_library, points_np):
    from validate_iassd_ort_model import run_ort

    return run_ort(
        onnx_file,
        ort_op_library,
        points_np,
        ['CUDAExecutionProvider'],
    )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA를 사용할 수 없습니다.')

    engine_path, plugin_path, engine = load_trt_engine(args.engine_file, args.plugin_library)
    cfg_file, dataset = build_dataset(args)
    points = build_example_points(dataset).cuda()
    trt_outputs = run_trt(engine, points)

    report = {
        'cfg_file': str(cfg_file),
        'engine_file': str(engine_path),
        'plugin_library': str(plugin_path),
        'num_points': args.num_points,
        'input_points': {
            'shape': list(points.shape),
            'dtype': str(points.dtype),
            'device': str(points.device),
        },
        'trt_outputs': {
            name: {
                'shape': list(value.shape),
                'dtype': str(value.dtype),
            }
            for name, value in trt_outputs.items()
        },
    }
    dump_arrays = {
        f'trt::{name}': value
        for name, value in trt_outputs.items()
    }

    if not args.skip_torch:
        _, ckpt_file, torch_outputs = run_torch_model(args, dataset, points)
        report['ckpt'] = str(ckpt_file)
        report['torch_vs_trt'] = compare_outputs(torch_outputs, trt_outputs)
        dump_arrays.update({
            f'torch::{name}': value
            for name, value in torch_outputs.items()
        })

    should_run_ort = not args.skip_ort and args.ort_onnx_file is not None and args.ort_op_library is not None
    if should_run_ort:
        ort_outputs = run_ort_cuda(
            Path(args.ort_onnx_file).resolve(),
            Path(args.ort_op_library).resolve(),
            tensor_to_numpy(points),
        )
        report['ort_onnx_file'] = str(Path(args.ort_onnx_file).resolve())
        report['ort_op_library'] = str(Path(args.ort_op_library).resolve())
        report['ort_vs_trt'] = compare_outputs(ort_outputs, trt_outputs)
        dump_arrays.update({
            f'ort::{name}': value
            for name, value in ort_outputs.items()
        })

    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.report_file is not None:
        report_path = Path(args.report_file)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')

    if args.dump_outputs_npz is not None:
        dump_path = Path(args.dump_outputs_npz)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dump_path, **dump_arrays)


if __name__ == '__main__':
    main()
