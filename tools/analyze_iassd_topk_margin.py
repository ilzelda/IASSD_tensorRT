import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='IA-SSD TopK score margin 분석')
    parser.add_argument('--npz_file', type=str, required=True, help='validate_iassd_trt_engine.py가 저장한 NPZ 파일')
    parser.add_argument('--score_name', type=str, default='sub_4', help='분석할 TopK 입력 score tensor 이름')
    parser.add_argument('--index_name', type=str, default='getitem_56', help='분석할 TopK index tensor 이름')
    parser.add_argument('--topk', type=int, default=None, help='TopK k 값. 생략하면 index tensor shape에서 추론')
    parser.add_argument('--report_file', type=str, default=None, help='분석 결과 JSON 저장 경로')
    return parser.parse_args()


def as_2d(array):
    array = np.asarray(array)
    if array.ndim == 1:
        return array.reshape(1, -1)
    if array.ndim == 2:
        return array
    return array.reshape(array.shape[0], -1)


def summarize_array(array):
    array = np.asarray(array)
    return {
        'shape': list(array.shape),
        'dtype': str(array.dtype),
    }


def summarize_score_diff(reference_scores, target_scores):
    diff = target_scores.astype(np.float64) - reference_scores.astype(np.float64)
    abs_diff = np.abs(diff)
    return {
        'max_abs': float(np.max(abs_diff)) if abs_diff.size else 0.0,
        'mean_abs': float(np.mean(abs_diff)) if abs_diff.size else 0.0,
        'p50_abs': float(np.percentile(abs_diff, 50)) if abs_diff.size else 0.0,
        'p90_abs': float(np.percentile(abs_diff, 90)) if abs_diff.size else 0.0,
        'p99_abs': float(np.percentile(abs_diff, 99)) if abs_diff.size else 0.0,
    }


def sorted_desc(scores):
    return np.sort(scores.astype(np.float64))[::-1]


def summarize_reference_margin(scores, k):
    ordered = sorted_desc(scores)
    if k <= 0 or k >= ordered.size:
        return {}
    top_values = ordered[:k]
    tail_values = ordered[k:]
    boundary_gap = float(top_values[-1] - tail_values[0])
    adjacent_gaps = top_values[:-1] - top_values[1:]
    thresholds = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3]
    return {
        'kth_score': float(top_values[-1]),
        'next_score': float(tail_values[0]),
        'boundary_gap': boundary_gap,
        'min_adjacent_gap_in_topk': float(np.min(adjacent_gaps)) if adjacent_gaps.size else 0.0,
        'mean_adjacent_gap_in_topk': float(np.mean(adjacent_gaps)) if adjacent_gaps.size else 0.0,
        'small_gap_counts_in_topk': {
            str(threshold): int(np.count_nonzero(adjacent_gaps <= threshold))
            for threshold in thresholds
        },
    }


def summarize_index_overlap(reference_indices, target_indices, reference_scores):
    reference_indices = reference_indices.astype(np.int64)
    target_indices = target_indices.astype(np.int64)
    batch_reports = []
    total_ref = 0
    total_intersection = 0

    for batch_index, (ref_row, target_row, score_row) in enumerate(zip(reference_indices, target_indices, reference_scores)):
        ref_set = set(int(value) for value in ref_row)
        target_set = set(int(value) for value in target_row)
        intersection = ref_set & target_set
        missing = sorted(ref_set - target_set)
        extra = sorted(target_set - ref_set)
        total_ref += len(ref_set)
        total_intersection += len(intersection)

        missing_scores = score_row[missing] if missing else np.asarray([], dtype=np.float64)
        extra_scores = score_row[extra] if extra else np.asarray([], dtype=np.float64)
        batch_reports.append({
            'batch_index': batch_index,
            'reference_count': len(ref_set),
            'target_count': len(target_set),
            'intersection_count': len(intersection),
            'intersection_ratio': float(len(intersection) / len(ref_set)) if ref_set else 1.0,
            'missing_count': len(missing),
            'extra_count': len(extra),
            'missing_reference_score_min': float(np.min(missing_scores)) if missing_scores.size else None,
            'missing_reference_score_max': float(np.max(missing_scores)) if missing_scores.size else None,
            'extra_reference_score_min': float(np.min(extra_scores)) if extra_scores.size else None,
            'extra_reference_score_max': float(np.max(extra_scores)) if extra_scores.size else None,
        })

    return {
        'total_reference_count': total_ref,
        'total_intersection_count': total_intersection,
        'total_intersection_ratio': float(total_intersection / total_ref) if total_ref else 1.0,
        'batches': batch_reports,
    }


def main():
    args = parse_args()
    npz_path = Path(args.npz_file)
    arrays = np.load(npz_path)

    reference_score_key = f'ort::{args.score_name}'
    target_score_key = f'trt::{args.score_name}'
    reference_index_key = f'ort::{args.index_name}'
    target_index_key = f'trt::{args.index_name}'
    required_keys = [reference_score_key, target_score_key, reference_index_key, target_index_key]
    missing_keys = [key for key in required_keys if key not in arrays]
    if missing_keys:
        raise KeyError(f'NPZ에 필요한 tensor가 없습니다: {missing_keys}')

    reference_scores = as_2d(arrays[reference_score_key])
    target_scores = as_2d(arrays[target_score_key])
    reference_indices = as_2d(arrays[reference_index_key])
    target_indices = as_2d(arrays[target_index_key])
    k = args.topk if args.topk is not None else reference_indices.shape[1]

    report = {
        'npz_file': str(npz_path.resolve()),
        'score_name': args.score_name,
        'index_name': args.index_name,
        'topk': int(k),
        'reference_score': summarize_array(reference_scores),
        'target_score': summarize_array(target_scores),
        'reference_index': summarize_array(reference_indices),
        'target_index': summarize_array(target_indices),
        'score_diff': summarize_score_diff(reference_scores, target_scores),
        'index_overlap': summarize_index_overlap(reference_indices, target_indices, reference_scores),
        'reference_margin_by_batch': [
            summarize_reference_margin(score_row, k)
            for score_row in reference_scores
        ],
    }

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.report_file is not None:
        report_path = Path(args.report_file)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(text + '\n')


if __name__ == '__main__':
    main()
