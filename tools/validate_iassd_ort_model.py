import argparse
import glob
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT_DIR / 'tools'
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import types

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.utils import common_utils


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


def parse_args():
    parser = argparse.ArgumentParser(description='IA-SSD ONNX Runtime 모델 단위 검증')
    parser.add_argument('--cfg_file', type=str, default='tools/cfgs/kitti_models/IA-SSD.yaml')
    parser.add_argument('--ckpt', type=str, default='tools/IA-SSD.pth')
    parser.add_argument('--onnx_file', type=str, required=True, help='검증할 IA-SSD ONNX 파일')
    parser.add_argument('--ort_op_library', type=str, required=True, help='libiassd_ort_ops.so 경로')
    parser.add_argument('--sample_data_path', type=str, default=None, help='샘플 포인트클라우드 파일 또는 디렉터리 경로')
    parser.add_argument('--sample_ext', type=str, default='.bin', help='샘플 포인트클라우드 확장자')
    parser.add_argument('--num_points', type=int, default=16384, help='synthetic 입력 생성 시 포인트 수')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda'], help='현재 검증은 CUDA 경로만 지원')
    parser.add_argument('--providers', type=str, default='CUDAExecutionProvider', help='쉼표로 구분한 ORT provider 목록')
    parser.add_argument('--report_file', type=str, default=None, help='shape/error report JSON 저장 경로')
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


def compare_outputs(torch_outputs, ort_outputs):
    report = {}
    for name, torch_output in torch_outputs.items():
        ort_output = ort_outputs[name]
        diff = ort_output.astype(np.float64) - torch_output.astype(np.float64)
        report[name] = {
            'torch': describe_array(torch_output),
            'ort': describe_array(ort_output),
            'shape_match': list(torch_output.shape) == list(ort_output.shape),
            'max_abs_error': float(np.max(np.abs(diff))) if diff.size else 0.0,
            'mean_abs_error': float(np.mean(np.abs(diff))) if diff.size else 0.0,
        }
    return report


def run_ort(onnx_file, ort_op_library, points_np, providers):
    options = ort.SessionOptions()
    options.register_custom_ops_library(str(ort_op_library))
    session = ort.InferenceSession(str(onnx_file), sess_options=options, providers=providers)

    input_name = session.get_inputs()[0].name
    ort_raw_outputs = session.run(None, {input_name: points_np})
    output_names = [output.name for output in session.get_outputs()]

    return {
        name: output
        for name, output in zip(output_names, ort_raw_outputs)
    }


def main():
    args = parse_args()
    logger = common_utils.create_logger()

    onnx_file = Path(args.onnx_file).resolve()
    ort_op_library = Path(args.ort_op_library).resolve()
    cfg_file = resolve_tools_path(args.cfg_file).resolve()
    ckpt_file = resolve_tools_path(args.ckpt).resolve()
    if not onnx_file.exists():
        raise FileNotFoundError(f'ONNX 파일이 없습니다: {onnx_file}')
    if not ort_op_library.exists():
        raise FileNotFoundError(f'custom op library가 없습니다: {ort_op_library}')
    if not cfg_file.exists():
        raise FileNotFoundError(f'config 파일이 없습니다: {cfg_file}')
    if not ckpt_file.exists():
        raise FileNotFoundError(f'checkpoint 파일이 없습니다: {ckpt_file}')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA를 사용할 수 없습니다.')

    old_cwd = Path.cwd()
    try:
        # OpenPCDet config의 _BASE_CONFIG_는 tools/ 기준 상대경로를 사용한다.
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

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=str(ckpt_file), logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    points = build_example_points(dataset).cuda()
    wrapper = OnnxValidationWrapper(model).cuda()
    wrapper.eval()

    with torch.no_grad():
        batch_cls_preds, batch_box_preds = wrapper(points)
    torch.cuda.synchronize()

    torch_outputs = {
        'batch_cls_preds': tensor_to_numpy(batch_cls_preds),
        'batch_box_preds': tensor_to_numpy(batch_box_preds),
    }
    providers = [provider.strip() for provider in args.providers.split(',') if provider.strip()]
    ort_outputs = run_ort(onnx_file, ort_op_library, tensor_to_numpy(points), providers)

    report = {
        'cfg_file': str(cfg_file),
        'ckpt': str(ckpt_file),
        'onnx_file': str(onnx_file),
        'ort_op_library': str(ort_op_library),
        'providers_requested': providers,
        'num_points': args.num_points,
        'input_points': {
            'shape': list(points.shape),
            'dtype': str(points.dtype),
            'device': str(points.device),
        },
        'outputs': compare_outputs(torch_outputs, ort_outputs),
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.report_file is not None:
        report_path = Path(args.report_file)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
