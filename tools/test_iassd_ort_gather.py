import argparse
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, helper

from pcdet.ops.pointnet2.pointnet2_batch import pointnet2_utils


def parse_args():
    parser = argparse.ArgumentParser(description='IASSD::GatherPoints ORT custom op 단위 테스트')
    parser.add_argument('--ort_op_library', type=str, required=True, help='libiassd_ort_ops.so 경로')
    parser.add_argument('--batch_size', type=int, default=1, help='테스트 batch 크기')
    parser.add_argument('--channels', type=int, default=4, help='feature channel 수')
    parser.add_argument('--num_points', type=int, default=512, help='입력 feature 포인트 수')
    parser.add_argument('--npoint', type=int, default=64, help='gather할 포인트 수')
    parser.add_argument('--seed', type=int, default=2048, help='입력 생성 seed')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda'], help='현재 custom op는 CUDA EP만 지원')
    return parser.parse_args()


def make_gather_model(path, batch_size, channels, num_points, npoint):
    graph = helper.make_graph(
        nodes=[
            helper.make_node(
                'GatherPoints',
                inputs=['features', 'idx'],
                outputs=['output'],
                domain='IASSD',
            )
        ],
        name='iassd_gather_test',
        inputs=[
            helper.make_tensor_value_info('features', TensorProto.FLOAT, [batch_size, channels, num_points]),
            helper.make_tensor_value_info('idx', TensorProto.INT32, [batch_size, npoint]),
        ],
        outputs=[
            helper.make_tensor_value_info('output', TensorProto.FLOAT, [batch_size, channels, npoint])
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_operatorsetid('', 17),
            helper.make_operatorsetid('IASSD', 1),
        ],
        producer_name='iassd_ort_gather_test',
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)


def make_inputs(args):
    rng = np.random.default_rng(args.seed)
    features_np = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(args.batch_size, args.channels, args.num_points),
    ).astype(np.float32)
    idx_np = rng.integers(
        low=0,
        high=args.num_points,
        size=(args.batch_size, args.npoint),
        dtype=np.int32,
    )
    return features_np, idx_np


def run_torch_gather(features_np, idx_np):
    features = torch.from_numpy(features_np).cuda().contiguous()
    idx = torch.from_numpy(idx_np).cuda().contiguous()
    with torch.no_grad():
        output = pointnet2_utils.gather_operation(features, idx)
    torch.cuda.synchronize()
    return output.cpu().numpy()


def run_ort_gather(model_path, library_path, features_np, idx_np):
    options = ort.SessionOptions()
    options.register_custom_ops_library(str(library_path))
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=['CUDAExecutionProvider'])
    return session.run(['output'], {'features': features_np, 'idx': idx_np})[0]


def main():
    args = parse_args()
    library_path = Path(args.ort_op_library).resolve()
    if not library_path.exists():
        raise FileNotFoundError(f'custom op library가 없습니다: {library_path}')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA를 사용할 수 없습니다.')

    features_np, idx_np = make_inputs(args)

    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / 'iassd_gather_test.onnx'
        make_gather_model(model_path, args.batch_size, args.channels, args.num_points, args.npoint)

        torch_output = run_torch_gather(features_np, idx_np)
        ort_output = run_ort_gather(model_path, library_path, features_np, idx_np)

    if not np.array_equal(torch_output, ort_output):
        max_abs_diff = float(np.max(np.abs(torch_output - ort_output)))
        diff_count = int(np.count_nonzero(torch_output != ort_output))
        raise AssertionError(f'Gather output mismatch: diff_count={diff_count}, max_abs_diff={max_abs_diff}')

    print('IASSD::GatherPoints ORT custom op 테스트 통과')
    print(
        f'features_shape={tuple(features_np.shape)} '
        f'idx_shape={tuple(idx_np.shape)} '
        f'output_shape={tuple(ort_output.shape)}'
    )


if __name__ == '__main__':
    main()
