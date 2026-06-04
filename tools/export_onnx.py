import _init_path

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network
from pcdet.utils import common_utils


class ExportDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, root_path, sample_data_path=None, ext='.bin', num_points=65536):
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


class OnnxExportWrapper(nn.Module):
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


def parse_config():
    parser = argparse.ArgumentParser(description='IA-SSD ONNX export')
    parser.add_argument('--cfg_file', type=str, default='tools/cfgs/waymo_models/IA-SSD.yaml')
    parser.add_argument('--ckpt', type=str, default='tools/IA-SSD.pth')
    parser.add_argument('--output_file', type=str, default='ia_ssd.onnx')
    parser.add_argument('--sample_data_path', type=str, default=None,
                        help='샘플 포인트클라우드 파일 또는 디렉터리 경로')
    parser.add_argument('--sample_ext', type=str, default='.bin', help='샘플 포인트클라우드 확장자')
    parser.add_argument('--num_points', type=int, default=65536, help='synthetic 입력 생성 시 포인트 수')
    parser.add_argument('--device', type=str, default='cuda', help='export에 사용할 장치')
    parser.add_argument('--opset_version', type=int, default=17, help='ONNX opset 버전')
    parser.add_argument('--shape_report_file', type=str, default=None,
                        help='입력과 NMS 전 raw 출력 shape를 저장할 JSON 경로')
    parser.add_argument('--dump_raw_output_file', type=str, default=None,
                        help='NMS 전 raw 출력 tensor를 저장할 NPZ 경로')
    parser.add_argument('--skip_export', action='store_true',
                        help='raw forward 검증만 수행하고 ONNX export를 건너뜀')
    parser.add_argument('--use_iassd_custom_ops', action='store_true',
                        help='IA-SSD custom ONNX op placeholder 경로로 export')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='추가 config override')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def build_example_batch(export_dataset):
    sample_dict = export_dataset[0]
    batch_dict = export_dataset.collate_batch([sample_dict])
    points = torch.from_numpy(batch_dict['points']).float()
    return points


def describe_tensor(tensor):
    return {
        'shape': list(tensor.shape),
        'dtype': str(tensor.dtype),
        'device': str(tensor.device),
    }


def write_shape_report(path, args, example_points, raw_outputs):
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        'cfg_file': args.cfg_file,
        'ckpt': args.ckpt,
        'sample_data_path': args.sample_data_path,
        'num_points': args.num_points,
        'device': args.device,
        'input_points': describe_tensor(example_points),
        'outputs': {
            name: describe_tensor(tensor)
            for name, tensor in raw_outputs.items()
        },
    }

    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')


def dump_raw_outputs(path, raw_outputs):
    dump_path = Path(path)
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dump_path,
        **{
            name: tensor.detach().cpu().numpy()
            for name, tensor in raw_outputs.items()
        },
    )


def sanitize_onnx_ir_bool_int_attrs(onnx_program):
    model = getattr(onnx_program, 'model', None)
    if model is None:
        return 0

    graphs = [model.graph]
    graphs.extend(getattr(model, 'functions', {}).values())

    fixed_count = 0
    for graph in graphs:
        for node in graph:
            for attr in getattr(node, 'attributes', {}).values():
                attr_type = getattr(attr, 'type', None)
                if getattr(attr_type, 'name', None) == 'INT' and isinstance(getattr(attr, 'value', None), bool):
                    attr.value = int(attr.value)
                    fixed_count += 1

    return fixed_count


def ensure_iassd_opset_import(onnx_program):
    model = getattr(onnx_program, 'model', None)
    if model is None:
        return False

    opset_imports = getattr(model, 'opset_imports', None)
    if opset_imports is None:
        return False

    if opset_imports.get('IASSD') == 1:
        return False

    opset_imports['IASSD'] = 1
    return True


def export_with_iassd_custom_ops(wrapper, example_points, output_path, logger):
    from torch.onnx._internal.exporter import _core

    from pcdet.ops.onnx_custom_ops import build_iassd_onnx_registry, export_placeholders_enabled

    logger.info('IA-SSD custom ONNX op placeholder 경로를 사용합니다.')
    logger.info('PyTorch 2.5 내부 ONNX exporter API를 사용하므로 opset_version 인자는 이 경로에서 적용되지 않을 수 있습니다.')

    registry = build_iassd_onnx_registry()
    with torch.no_grad(), export_placeholders_enabled():
        onnx_program = _core.export(
            wrapper,
            (example_points,),
            registry=registry,
            input_names=['points'],
            output_names=['batch_cls_preds', 'batch_box_preds'],
        )
    fixed_count = sanitize_onnx_ir_bool_int_attrs(onnx_program)
    if fixed_count:
        logger.info('ONNX IR bool-valued INT attribute 보정 완료: %d개', fixed_count)
    if ensure_iassd_opset_import(onnx_program):
        logger.info('IASSD custom op domain opset import 추가 완료: IASSD=1')
    onnx_program.save(output_path)


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA를 사용할 수 없습니다. --device 값을 cpu로 바꾸거나 CUDA 환경을 확인하세요.')
    device = torch.device(args.device)

    export_dataset = ExportDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        root_path=Path(args.sample_data_path).parent if args.sample_data_path is not None else Path.cwd(),
        sample_data_path=args.sample_data_path,
        ext=args.sample_ext,
        num_points=args.num_points,
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=export_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.to(device)
    model.eval()

    example_points = build_example_batch(export_dataset).to(device)
    wrapper = OnnxExportWrapper(model).to(device)
    wrapper.eval()

    logger.info('예제 입력 points shape: %s', tuple(example_points.shape))

    with torch.no_grad():
        batch_cls_preds, batch_box_preds = wrapper(example_points)

    raw_outputs = {
        'batch_cls_preds': batch_cls_preds,
        'batch_box_preds': batch_box_preds,
    }
    logger.info('raw batch_cls_preds shape: %s', tuple(batch_cls_preds.shape))
    logger.info('raw batch_box_preds shape: %s', tuple(batch_box_preds.shape))

    if args.shape_report_file is not None:
        write_shape_report(args.shape_report_file, args, example_points, raw_outputs)
        logger.info('shape report 저장 완료: %s', args.shape_report_file)

    if args.dump_raw_output_file is not None:
        dump_raw_outputs(args.dump_raw_output_file, raw_outputs)
        logger.info('raw output 저장 완료: %s', args.dump_raw_output_file)

    if args.skip_export:
        logger.info('--skip_export가 지정되어 ONNX export를 건너뜁니다.')
        return

    logger.info('ONNX export 시작: %s', output_path)

    with torch.no_grad():
        if args.use_iassd_custom_ops:
            export_with_iassd_custom_ops(wrapper, example_points, output_path, logger)
        else:
            torch.onnx.export(
                wrapper,
                (example_points,),
                str(output_path),
                opset_version=args.opset_version,
                input_names=['points'],
                output_names=['batch_cls_preds', 'batch_box_preds'],
                dynamo=True,
            )

    logger.info('ONNX export 완료: %s', output_path)


if __name__ == '__main__':
    main()
