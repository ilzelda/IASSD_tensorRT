import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import helper

from validate_iassd_ort_model import (
    OnnxValidationWrapper,
    ValidationDataset,
    build_example_points,
    cfg,
    cfg_from_list,
    cfg_from_yaml_file,
    common_utils,
    compare_outputs,
    install_spconv_import_stub,
    ort,
    resolve_tools_path,
    tensor_to_numpy,
    TOOLS_DIR,
)


install_spconv_import_stub()

from pcdet.models import build_network


DEFAULT_ONNX_DEBUG_OUTPUTS = [
    'gather_points',
    'gather_points_1',
    'gather_points_2',
    'gather_points_3',
    'gather_points_4',
    '_to_copy_1',
    '_to_copy_2',
    'squeeze_1',
    'relu_6',
    'squeeze_2',
    'squeeze_3',
    'cat_5',
    'relu_13',
    'relu_14',
    'convolution_15',
    'sigmoid',
    'relu_21',
    'convolution_24',
    'sigmoid_1',
    'convolution_26',
    'add',
    'batch_cls_preds',
    'batch_box_preds',
]


def parse_args():
    parser = argparse.ArgumentParser(description='IA-SSD PyTorch/ORT 중간 tensor parity 추적')
    parser.add_argument('--cfg_file', type=str, default='tools/cfgs/kitti_models/IA-SSD.yaml')
    parser.add_argument('--ckpt', type=str, default='tools/IA-SSD.pth')
    parser.add_argument('--onnx_file', type=str, required=True, help='검증할 IA-SSD ONNX 파일')
    parser.add_argument('--ort_op_library', type=str, required=True, help='libiassd_ort_ops.so 경로')
    parser.add_argument('--sample_data_path', type=str, default=None, help='샘플 포인트클라우드 파일 또는 디렉터리 경로')
    parser.add_argument('--sample_ext', type=str, default='.bin', help='샘플 포인트클라우드 확장자')
    parser.add_argument('--num_points', type=int, default=16384, help='synthetic 입력 생성 시 포인트 수')
    parser.add_argument('--providers', type=str, default='CUDAExecutionProvider', help='쉼표로 구분한 ORT provider 목록')
    parser.add_argument('--onnx_debug_outputs', type=str, default=','.join(DEFAULT_ONNX_DEBUG_OUTPUTS), help='추가로 뽑을 ONNX tensor 이름 목록')
    parser.add_argument('--report_file', type=str, default=None, help='trace report JSON 저장 경로')
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


def find_value_info(model, name):
    for value_info in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        if value_info.name == name:
            return value_info
    return None


def make_debug_onnx(onnx_file, output_names, output_file):
    model = onnx.load(str(onnx_file))
    existing_outputs = {output.name for output in model.graph.output}
    missing_outputs = []

    for output_name in output_names:
        if output_name in existing_outputs:
            continue
        value_info = find_value_info(model, output_name)
        if value_info is None:
            missing_outputs.append(output_name)
            continue
        model.graph.output.append(helper.make_tensor_value_info(
            value_info.name,
            value_info.type.tensor_type.elem_type,
            [dim.dim_value if dim.dim_value else dim.dim_param for dim in value_info.type.tensor_type.shape.dim],
        ))

    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as err:
        print(f'ONNX checker 경고: debug output 추가 모델 검사를 건너뜁니다: {err}')
    onnx.save(model, str(output_file))
    return missing_outputs


def collect_torch_trace(model, points):
    batch_dict = {
        'points': points,
        'batch_size': 1,
    }
    original_topk = torch.topk
    topk_records = []

    def traced_topk(*args, **kwargs):
        values, indices = original_topk(*args, **kwargs)
        topk_records.append((values.detach(), indices.detach()))
        return values, indices

    try:
        torch.topk = traced_topk
        for cur_module in model.module_list:
            batch_dict = cur_module(batch_dict)
    finally:
        torch.topk = original_topk

    trace = {
        'batch_cls_preds': tensor_to_numpy(batch_dict['batch_cls_preds']),
        'batch_box_preds': tensor_to_numpy(batch_dict['batch_box_preds']),
    }

    for idx, (values, indices) in enumerate(topk_records):
        trace[f'topk_values_{idx}'] = tensor_to_numpy(values)
        trace[f'topk_indices_{idx}'] = tensor_to_numpy(indices.int())

    for idx, xyz in enumerate(batch_dict.get('encoder_xyz', [])):
        trace[f'encoder_xyz_{idx}'] = tensor_to_numpy(xyz)
        trace[f'encoder_xyz_{idx}_bcn'] = tensor_to_numpy(xyz.transpose(1, 2).contiguous())

    for idx, features in enumerate(batch_dict.get('encoder_features', [])):
        if features is None:
            continue
        trace[f'encoder_features_{idx}'] = tensor_to_numpy(features)
        trace[f'encoder_features_{idx}_bnc'] = tensor_to_numpy(features.transpose(1, 2).contiguous())

    for idx, sa_ins_pred in enumerate(batch_dict.get('sa_ins_preds', [])):
        if len(sa_ins_pred) == 0:
            continue
        logits = sa_ins_pred[..., 1:]
        trace[f'sa_ins_logits_{idx}'] = tensor_to_numpy(logits)
        trace[f'sa_ins_logits_{idx}_bcn'] = tensor_to_numpy(logits.transpose(1, 2).contiguous())
        trace[f'sa_ins_score_{idx}'] = tensor_to_numpy(torch.sigmoid(logits.max(dim=-1).values))

    if 'centers' in batch_dict:
        centers = batch_dict['centers'][:, 1:4].view(1, -1, 3)
        trace['centers_xyz'] = tensor_to_numpy(centers)
        trace['centers_xyz_bcn'] = tensor_to_numpy(centers.transpose(1, 2).contiguous())
    if 'centers_origin' in batch_dict:
        centers_origin = batch_dict['centers_origin'][:, 1:4].view(1, -1, 3)
        trace['centers_origin_xyz'] = tensor_to_numpy(centers_origin)
        trace['centers_origin_xyz_bcn'] = tensor_to_numpy(centers_origin.transpose(1, 2).contiguous())
    if 'ctr_offsets' in batch_dict:
        ctr_offsets = batch_dict['ctr_offsets'][:, 1:4].view(1, -1, 3)
        trace['ctr_offsets_xyz'] = tensor_to_numpy(ctr_offsets)
        trace['ctr_offsets_xyz_bcn'] = tensor_to_numpy(ctr_offsets.transpose(1, 2).contiguous())

    return trace


def run_ort_debug(onnx_file, ort_op_library, points_np, providers):
    options = ort.SessionOptions()
    options.register_custom_ops_library(str(ort_op_library))
    session = ort.InferenceSession(str(onnx_file), sess_options=options, providers=providers)

    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    outputs = session.run(output_names, {input_name: points_np})
    return {name: output for name, output in zip(output_names, outputs)}


def array_summary(array):
    return {
        'shape': list(array.shape),
        'dtype': str(array.dtype),
    }


def compare_pair(torch_name, torch_array, ort_name, ort_array):
    result = {
        'torch_name': torch_name,
        'ort_name': ort_name,
        'torch': array_summary(torch_array),
        'ort': array_summary(ort_array),
        'shape_match': list(torch_array.shape) == list(ort_array.shape),
    }
    if not result['shape_match']:
        return result

    if not np.issubdtype(torch_array.dtype, np.number) or not np.issubdtype(ort_array.dtype, np.number):
        return result

    diff = ort_array.astype(np.float64) - torch_array.astype(np.float64)
    result.update({
        'max_abs_error': float(np.max(np.abs(diff))) if diff.size else 0.0,
        'mean_abs_error': float(np.mean(np.abs(diff))) if diff.size else 0.0,
    })
    return result


def compare_named_pairs(torch_trace, ort_trace):
    pairs = [
        ('encoder_xyz_1_bcn', 'gather_points'),
        ('encoder_features_1', 'relu_6'),
        ('encoder_xyz_2_bcn', 'gather_points_1'),
        ('encoder_features_2', 'cat_5'),
        ('encoder_features_2', 'relu_13'),
        ('encoder_features_2', 'relu_14'),
        ('sa_ins_logits_1_bcn', 'convolution_15'),
        ('sa_ins_score_1', 'sigmoid'),
        ('topk_indices_0', '_to_copy_1'),
        ('encoder_xyz_3_bcn', 'gather_points_2'),
        ('sa_ins_logits_2_bcn', 'convolution_24'),
        ('sa_ins_score_2', 'sigmoid_1'),
        ('topk_indices_1', '_to_copy_2'),
        ('encoder_xyz_4_bcn', 'gather_points_3'),
        ('encoder_features_3', 'relu_14'),
        ('encoder_features_4', 'relu_21'),
        ('encoder_features_4', 'gather_points_4'),
        ('ctr_offsets_xyz_bcn', 'convolution_26'),
        ('centers_xyz', 'add'),
        ('batch_cls_preds', 'batch_cls_preds'),
        ('batch_box_preds', 'batch_box_preds'),
    ]

    report = []
    for torch_name, ort_name in pairs:
        if torch_name not in torch_trace or ort_name not in ort_trace:
            report.append({
                'torch_name': torch_name,
                'ort_name': ort_name,
                'status': 'missing',
            })
            continue
        report.append(compare_pair(torch_name, torch_trace[torch_name], ort_name, ort_trace[ort_name]))
    return report


def find_shape_matches(torch_trace, ort_trace):
    matches = []
    for ort_name, ort_array in ort_trace.items():
        for torch_name, torch_array in torch_trace.items():
            if list(ort_array.shape) != list(torch_array.shape):
                continue
            if not np.issubdtype(ort_array.dtype, np.number) or not np.issubdtype(torch_array.dtype, np.number):
                continue
            result = compare_pair(torch_name, torch_array, ort_name, ort_array)
            matches.append(result)

    matches.sort(key=lambda item: item.get('mean_abs_error', float('inf')))
    return matches[:30]


def main():
    args = parse_args()
    logger = common_utils.create_logger()

    onnx_file = Path(args.onnx_file).resolve()
    ort_op_library = Path(args.ort_op_library).resolve()
    cfg_file = resolve_tools_path(args.cfg_file).resolve()
    ckpt_file = resolve_tools_path(args.ckpt).resolve()
    output_names = [name.strip() for name in args.onnx_debug_outputs.split(',') if name.strip()]
    providers = [provider.strip() for provider in args.providers.split(',') if provider.strip()]

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA를 사용할 수 없습니다.')
    if not onnx_file.exists():
        raise FileNotFoundError(f'ONNX 파일이 없습니다: {onnx_file}')
    if not ort_op_library.exists():
        raise FileNotFoundError(f'custom op library가 없습니다: {ort_op_library}')

    load_config(cfg_file, args.set_cfgs)
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
        torch_trace = collect_torch_trace(model, points)
        batch_cls_preds, batch_box_preds = wrapper(points)
    torch.cuda.synchronize()
    torch_trace['wrapper_batch_cls_preds'] = tensor_to_numpy(batch_cls_preds)
    torch_trace['wrapper_batch_box_preds'] = tensor_to_numpy(batch_box_preds)

    with tempfile.TemporaryDirectory() as temp_dir:
        debug_onnx_file = Path(temp_dir) / 'iassd_debug_outputs.onnx'
        missing_outputs = make_debug_onnx(onnx_file, output_names, debug_onnx_file)
        ort_trace = run_ort_debug(debug_onnx_file, ort_op_library, tensor_to_numpy(points), providers)

    report = {
        'cfg_file': str(cfg_file),
        'ckpt': str(ckpt_file),
        'onnx_file': str(onnx_file),
        'ort_op_library': str(ort_op_library),
        'providers_requested': providers,
        'num_points': args.num_points,
        'missing_onnx_debug_outputs': missing_outputs,
        'named_pairs': compare_named_pairs(torch_trace, ort_trace),
        'best_shape_matches': find_shape_matches(torch_trace, ort_trace),
        'final_outputs': compare_outputs(
            {
                'batch_cls_preds': torch_trace['batch_cls_preds'],
                'batch_box_preds': torch_trace['batch_box_preds'],
            },
            {
                'batch_cls_preds': ort_trace['batch_cls_preds'],
                'batch_box_preds': ort_trace['batch_box_preds'],
            },
        ),
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.report_file is not None:
        report_path = Path(args.report_file)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
