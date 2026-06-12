import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT_DIR / 'tools'
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file
from pcdet.utils import common_utils
from validate_iassd_ort_model import run_ort
from validate_iassd_trt_engine import (
    ValidationDataset,
    build_example_points,
    load_trt_engine,
    run_trt,
    tensor_to_numpy,
)
from pcdet.models import build_network


def parse_args():
    parser = argparse.ArgumentParser(description='IA-SSD raw output의 post-processing 결과 비교')
    parser.add_argument('--cfg_file', type=str, default='tools/cfgs/kitti_models/IA-SSD.yaml')
    parser.add_argument('--ckpt', type=str, default='tools/IA-SSD.pth')
    parser.add_argument('--base_onnx_file', type=str, required=True, help='기준 ORT ONNX 파일')
    parser.add_argument('--candidate_onnx_file', type=str, default=None, help='비교할 ORT ONNX 파일')
    parser.add_argument('--ort_op_library', type=str, required=True, help='libiassd_ort_ops.so 경로')
    parser.add_argument('--engine_file', type=str, default=None, help='비교할 TensorRT engine 파일')
    parser.add_argument('--plugin_library', type=str, default=None, help='TensorRT plugin library 경로')
    parser.add_argument('--sample_data_path', type=str, default=None, help='샘플 포인트클라우드 파일 또는 디렉터리 경로')
    parser.add_argument('--sample_ext', type=str, default='.bin', help='샘플 포인트클라우드 확장자')
    parser.add_argument('--num_points', type=int, default=16384, help='synthetic 입력 생성 시 포인트 수')
    parser.add_argument('--seed', type=int, default=1024, help='입력 point sampling과 PyTorch 실행 재현용 seed')
    parser.add_argument('--include_pytorch', action='store_true', help='PyTorch raw inference 결과도 함께 비교')
    parser.add_argument('--skip_postprocess', action='store_true', help='NMS post-processing을 생략하고 raw 후보만 비교')
    parser.add_argument(
        '--postprocess_mode',
        type=str,
        default='model',
        choices=['model', 'numpy_nms'],
        help='model은 기존 CUDA NMS를 사용하고, numpy_nms는 비교용 CPU NMS를 사용',
    )
    parser.add_argument('--report_file', type=str, default=None, help='비교 report JSON 저장 경로')
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


def load_config(cfg_file, set_cfgs):
    cfg_path = resolve_tools_path(cfg_file).resolve()
    old_cwd = Path.cwd()
    try:
        import os
        os.chdir(TOOLS_DIR)
        cfg_from_yaml_file(str(cfg_path), cfg)
    finally:
        os.chdir(old_cwd)
    if set_cfgs is not None:
        cfg_from_list(set_cfgs, cfg)
    return cfg_path


def build_dataset(args):
    return ValidationDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        root_path=Path(args.sample_data_path).parent if args.sample_data_path is not None else Path.cwd(),
        sample_data_path=args.sample_data_path,
        ext=args.sample_ext,
        num_points=args.num_points,
    )


def build_postprocess_model(args, dataset):
    ckpt_file = resolve_tools_path(args.ckpt).resolve()
    if not ckpt_file.exists():
        raise FileNotFoundError(f'checkpoint 파일이 없습니다: {ckpt_file}')
    logger = common_utils.create_logger()
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=str(ckpt_file), logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    return ckpt_file, model


def ensure_raw_tensor(array, device):
    return torch.from_numpy(array).to(device=device, dtype=torch.float32)


def postprocess_outputs(model, outputs, device):
    cls_preds = ensure_raw_tensor(outputs['batch_cls_preds'], device)
    box_preds = ensure_raw_tensor(outputs['batch_box_preds'], device)
    batch_dict = {
        'batch_size': 1,
        'batch_cls_preds': cls_preds,
        'batch_box_preds': box_preds,
        'cls_preds_normalized': False,
    }
    if box_preds.dim() == 2:
        batch_dict['batch_index'] = torch.zeros(box_preds.shape[0], dtype=torch.long, device=device)
    with torch.no_grad():
        pred_dicts, _ = model.post_processing(batch_dict)
    pred_dict = pred_dicts[0]
    return {
        'pred_boxes': tensor_to_numpy(pred_dict['pred_boxes']),
        'pred_scores': tensor_to_numpy(pred_dict['pred_scores']),
        'pred_labels': tensor_to_numpy(pred_dict['pred_labels']),
    }


def run_pytorch_outputs(model, points):
    batch_dict = {
        'points': points,
        'batch_size': 1,
    }
    with torch.no_grad():
        for cur_module in model.module_list:
            batch_dict = cur_module(batch_dict)
    return {
        'batch_cls_preds': tensor_to_numpy(batch_dict['batch_cls_preds']),
        'batch_box_preds': tensor_to_numpy(batch_dict['batch_box_preds']),
    }


def boxes_to_aligned_bev(boxes):
    if boxes.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)
    centers = boxes[:, 0:2].astype(np.float32)
    sizes = boxes[:, 3:5].astype(np.float32)
    headings = boxes[:, 6].astype(np.float32)
    use_dx_dy = np.abs(np.cos(headings)) >= np.abs(np.sin(headings))
    aligned_sizes = sizes.copy()
    aligned_sizes[~use_dx_dy] = aligned_sizes[~use_dx_dy][:, ::-1]
    half_sizes = aligned_sizes * 0.5
    return np.concatenate([centers - half_sizes, centers + half_sizes], axis=1)


def aligned_bev_iou(reference_box, target_boxes):
    if target_boxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    lt = np.maximum(reference_box[:2], target_boxes[:, :2])
    rb = np.minimum(reference_box[2:], target_boxes[:, 2:])
    wh = np.maximum(rb - lt, 0.0)
    inter = wh[:, 0] * wh[:, 1]
    reference_area = max((reference_box[2] - reference_box[0]) * (reference_box[3] - reference_box[1]), 0.0)
    target_area = np.maximum(target_boxes[:, 2] - target_boxes[:, 0], 0.0) * np.maximum(
        target_boxes[:, 3] - target_boxes[:, 1],
        0.0,
    )
    union = reference_area + target_area - inter
    return inter / np.maximum(union, 1e-8)


def numpy_aligned_bev_nms(boxes, scores, nms_thresh, pre_maxsize, post_maxsize):
    if scores.shape[0] == 0:
        return np.asarray([], dtype=np.int64)
    order = np.argsort(-scores, kind='mergesort')
    if pre_maxsize is not None:
        order = order[:pre_maxsize]
    bev_boxes = boxes_to_aligned_bev(boxes[order, :7])
    keep = []
    candidate_order = np.arange(order.shape[0], dtype=np.int64)
    while candidate_order.size > 0 and len(keep) < post_maxsize:
        current = candidate_order[0]
        keep.append(current)
        if candidate_order.size == 1:
            break
        ious = aligned_bev_iou(bev_boxes[current], bev_boxes[candidate_order[1:]])
        candidate_order = candidate_order[1:][ious <= nms_thresh]
    return order[np.asarray(keep, dtype=np.int64)]


def postprocess_outputs_numpy(outputs):
    post_process_cfg = cfg.MODEL.POST_PROCESSING
    nms_config = post_process_cfg.NMS_CONFIG
    candidates = raw_outputs_to_candidates(outputs)
    scores = candidates['pred_scores']
    boxes = candidates['pred_boxes']
    labels = candidates['pred_labels']

    # 비교 안정성을 위한 CPU fallback이다. 실제 CUDA rotated NMS와 완전 동일하지 않고,
    # heading에 맞춘 axis-aligned BEV 근사로 suppress 여부만 재현한다.
    score_thresh = post_process_cfg.SCORE_THRESH
    if score_thresh is not None:
        score_mask = scores >= float(score_thresh)
        original_indices = np.nonzero(score_mask)[0]
        filtered_scores = scores[score_mask]
        filtered_boxes = boxes[score_mask]
    else:
        original_indices = np.arange(scores.shape[0], dtype=np.int64)
        filtered_scores = scores
        filtered_boxes = boxes

    selected = numpy_aligned_bev_nms(
        filtered_boxes,
        filtered_scores,
        float(nms_config.NMS_THRESH),
        int(nms_config.NMS_PRE_MAXSIZE),
        int(nms_config.NMS_POST_MAXSIZE),
    )
    selected = original_indices[selected]
    return {
        'pred_boxes': boxes[selected].astype(np.float32),
        'pred_scores': scores[selected].astype(np.float32),
        'pred_labels': labels[selected].astype(np.int64),
    }


def canonical_order(detections):
    scores = detections['pred_scores']
    labels = detections['pred_labels']
    boxes = detections['pred_boxes']
    if scores.size == 0:
        return np.asarray([], dtype=np.int64)
    rounded_boxes = np.round(boxes.astype(np.float64), decimals=5)
    keys = [rounded_boxes[:, index] for index in range(rounded_boxes.shape[1] - 1, -1, -1)]
    keys.append(labels.astype(np.int64))
    keys.append(-scores.astype(np.float64))
    return np.lexsort(tuple(keys))


def reorder_detections(detections, order):
    return {
        name: value[order]
        for name, value in detections.items()
    }


def summarize_detections(detections):
    scores = detections['pred_scores']
    labels = detections['pred_labels']
    return {
        'count': int(scores.shape[0]),
        'score_min': float(np.min(scores)) if scores.size else None,
        'score_max': float(np.max(scores)) if scores.size else None,
        'labels': labels.astype(np.int64).tolist(),
    }


def sigmoid(array):
    return 1.0 / (1.0 + np.exp(-array.astype(np.float64)))


def raw_outputs_to_candidates(outputs):
    cls_preds = outputs['batch_cls_preds']
    box_preds = outputs['batch_box_preds']
    if cls_preds.ndim == 3:
        cls_preds = cls_preds[0]
    if box_preds.ndim == 3:
        box_preds = box_preds[0]
    cls_scores = sigmoid(cls_preds)
    labels = np.argmax(cls_scores, axis=-1).astype(np.int64) + 1
    scores = np.max(cls_scores, axis=-1).astype(np.float32)
    return {
        'pred_boxes': box_preds.astype(np.float32),
        'pred_scores': scores,
        'pred_labels': labels,
    }


def compare_arrays(reference, target):
    common_count = min(reference.shape[0], target.shape[0])
    if common_count == 0:
        return {
            'common_count': int(common_count),
            'max_abs_error': None,
            'mean_abs_error': None,
        }
    diff = target[:common_count].astype(np.float64) - reference[:common_count].astype(np.float64)
    return {
        'common_count': int(common_count),
        'max_abs_error': float(np.max(np.abs(diff))),
        'mean_abs_error': float(np.mean(np.abs(diff))),
    }


def compare_detections(reference, target):
    reference_order = canonical_order(reference)
    target_order = canonical_order(target)
    reference_sorted = reorder_detections(reference, reference_order)
    target_sorted = reorder_detections(target, target_order)
    common_count = min(reference['pred_scores'].shape[0], target['pred_scores'].shape[0])
    labels_match = bool(
        reference_sorted['pred_labels'].shape == target_sorted['pred_labels'].shape
        and np.array_equal(reference_sorted['pred_labels'], target_sorted['pred_labels'])
    )
    return {
        'reference': summarize_detections(reference),
        'target': summarize_detections(target),
        'count_match': int(reference['pred_scores'].shape[0]) == int(target['pred_scores'].shape[0]),
        'labels_match_after_canonical_sort': labels_match,
        'common_count': int(common_count),
        'scores': compare_arrays(reference_sorted['pred_scores'], target_sorted['pred_scores']),
        'boxes': compare_arrays(reference_sorted['pred_boxes'], target_sorted['pred_boxes']),
    }


def compare_all_to_base_and_candidate(detections):
    comparisons = {}
    names = list(detections.keys())
    for name in names:
        if name != 'base_ort':
            comparisons[f'base_ort_vs_{name}'] = compare_detections(detections['base_ort'], detections[name])
    if 'candidate_ort' in detections and 'candidate_trt' in detections:
        comparisons['candidate_ort_vs_candidate_trt'] = compare_detections(
            detections['candidate_ort'],
            detections['candidate_trt'],
        )
    return comparisons


def compare_against_reference(detections, reference_name):
    if reference_name not in detections:
        return {}
    comparisons = {}
    for name in detections:
        if name == reference_name:
            continue
        comparisons[f'{reference_name}_vs_{name}'] = compare_detections(detections[reference_name], detections[name])
    return comparisons


def run_ort_outputs(onnx_file, ort_op_library, points):
    return run_ort(
        Path(onnx_file).resolve(),
        Path(ort_op_library).resolve(),
        tensor_to_numpy(points),
        ['CUDAExecutionProvider'],
    )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA를 사용할 수 없습니다.')
    if args.engine_file is not None and args.plugin_library is None:
        raise ValueError('--engine_file을 쓰려면 --plugin_library가 필요합니다.')

    cfg_file = load_config(args.cfg_file, args.set_cfgs)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dataset = build_dataset(args)
    ckpt_file = resolve_tools_path(args.ckpt).resolve()
    model = None
    if args.include_pytorch or (not args.skip_postprocess and args.postprocess_mode == 'model'):
        ckpt_file, model = build_postprocess_model(args, dataset)
    points = build_example_points(dataset).cuda()

    raw_outputs = {}
    if args.include_pytorch:
        raw_outputs['pytorch'] = run_pytorch_outputs(model, points)
    raw_outputs['base_ort'] = run_ort_outputs(args.base_onnx_file, args.ort_op_library, points)
    if args.candidate_onnx_file is not None:
        raw_outputs['candidate_ort'] = run_ort_outputs(args.candidate_onnx_file, args.ort_op_library, points)
    engine_path = None
    plugin_path = None
    if args.engine_file is not None:
        engine_path, plugin_path, engine = load_trt_engine(args.engine_file, args.plugin_library)
        raw_outputs['candidate_trt'] = run_trt(engine, points)

    raw_candidates = {
        name: raw_outputs_to_candidates(outputs)
        for name, outputs in raw_outputs.items()
    }
    raw_comparisons = compare_all_to_base_and_candidate(raw_candidates)
    raw_pytorch_comparisons = compare_against_reference(raw_candidates, 'pytorch')

    detections = {}
    comparisons = {}
    if not args.skip_postprocess:
        if args.postprocess_mode == 'model':
            detections = {
                name: postprocess_outputs(model, outputs, points.device)
                for name, outputs in raw_outputs.items()
            }
        else:
            detections = {
                name: postprocess_outputs_numpy(outputs)
                for name, outputs in raw_outputs.items()
            }
        comparisons = compare_all_to_base_and_candidate(detections)
    pytorch_comparisons = compare_against_reference(detections, 'pytorch')

    report = {
        'cfg_file': str(cfg_file),
        'ckpt': str(ckpt_file),
        'base_onnx_file': str(Path(args.base_onnx_file).resolve()),
        'candidate_onnx_file': str(Path(args.candidate_onnx_file).resolve()) if args.candidate_onnx_file else None,
        'ort_op_library': str(Path(args.ort_op_library).resolve()),
        'engine_file': str(engine_path) if engine_path is not None else None,
        'plugin_library': str(plugin_path) if plugin_path is not None else None,
        'num_points': args.num_points,
        'seed': args.seed,
        'include_pytorch': args.include_pytorch,
        'skip_postprocess': args.skip_postprocess,
        'postprocess_mode': args.postprocess_mode,
        'raw_candidates': {
            name: summarize_detections(candidates)
            for name, candidates in raw_candidates.items()
        },
        'raw_candidate_comparisons': raw_comparisons,
        'raw_pytorch_comparisons': raw_pytorch_comparisons,
        'detections': {
            name: summarize_detections(detection)
            for name, detection in detections.items()
        },
        'comparisons': comparisons,
        'pytorch_comparisons': pytorch_comparisons,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.report_file is not None:
        report_path = Path(args.report_file)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
