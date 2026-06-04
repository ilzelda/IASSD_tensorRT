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
    parser = argparse.ArgumentParser(description='IASSD::BallQuery ORT custom op 단위 테스트')
    parser.add_argument('--ort_op_library', type=str, required=True, help='libiassd_ort_ops.so 경로')
    parser.add_argument('--batch_size', type=int, default=1, help='테스트 batch 크기')
    parser.add_argument('--num_points', type=int, default=512, help='입력 xyz 포인트 수')
    parser.add_argument('--npoint', type=int, default=64, help='query center 포인트 수')
    parser.add_argument('--radius', type=float, default=0.2, help='BallQuery radius')
    parser.add_argument('--nsample', type=int, default=16, help='각 query center의 최대 neighbor 수')
    parser.add_argument('--seed', type=int, default=4096, help='입력 생성 seed')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda'], help='현재 custom op는 CUDA EP만 지원')
    return parser.parse_args()


def make_ball_query_model(path, batch_size, num_points, npoint, radius, nsample):
    graph = helper.make_graph(
        nodes=[
            helper.make_node(
                'BallQuery',
                inputs=['xyz', 'new_xyz'],
                outputs=['idx'],
                domain='IASSD',
                radius_f=radius,
                nsample_i=nsample,
            )
        ],
        name='iassd_ball_query_test',
        inputs=[
            helper.make_tensor_value_info('xyz', TensorProto.FLOAT, [batch_size, num_points, 3]),
            helper.make_tensor_value_info('new_xyz', TensorProto.FLOAT, [batch_size, npoint, 3]),
        ],
        outputs=[
            helper.make_tensor_value_info('idx', TensorProto.INT32, [batch_size, npoint, nsample])
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_operatorsetid('', 17),
            helper.make_operatorsetid('IASSD', 1),
        ],
        producer_name='iassd_ort_ball_query_test',
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)


def make_inputs(args):
    rng = np.random.default_rng(args.seed)
    xyz_np = rng.random((args.batch_size, args.num_points, 3), dtype=np.float32)
    center_idx = rng.integers(
        low=0,
        high=args.num_points,
        size=(args.batch_size, args.npoint),
        dtype=np.int64,
    )
    new_xyz_np = np.empty((args.batch_size, args.npoint, 3), dtype=np.float32)
    for batch_idx in range(args.batch_size):
        new_xyz_np[batch_idx] = xyz_np[batch_idx, center_idx[batch_idx]]
    return xyz_np, new_xyz_np


def run_torch_ball_query(xyz_np, new_xyz_np, radius, nsample):
    xyz = torch.from_numpy(xyz_np).cuda().contiguous()
    new_xyz = torch.from_numpy(new_xyz_np).cuda().contiguous()
    with torch.no_grad():
        idx = pointnet2_utils.ball_query(radius, nsample, xyz, new_xyz)
    torch.cuda.synchronize()
    return idx.cpu().numpy()


def run_ort_ball_query(model_path, library_path, xyz_np, new_xyz_np):
    options = ort.SessionOptions()
    options.register_custom_ops_library(str(library_path))
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=['CUDAExecutionProvider'])
    return session.run(['idx'], {'xyz': xyz_np, 'new_xyz': new_xyz_np})[0]


def main():
    args = parse_args()
    library_path = Path(args.ort_op_library).resolve()
    if not library_path.exists():
        raise FileNotFoundError(f'custom op library가 없습니다: {library_path}')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA를 사용할 수 없습니다.')

    xyz_np, new_xyz_np = make_inputs(args)

    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / 'iassd_ball_query_test.onnx'
        make_ball_query_model(
            model_path,
            args.batch_size,
            args.num_points,
            args.npoint,
            args.radius,
            args.nsample,
        )

        torch_idx = run_torch_ball_query(xyz_np, new_xyz_np, args.radius, args.nsample)
        ort_idx = run_ort_ball_query(model_path, library_path, xyz_np, new_xyz_np)

    if not np.array_equal(torch_idx, ort_idx):
        diff_count = int(np.count_nonzero(torch_idx != ort_idx))
        max_diff = int(np.max(np.abs(torch_idx.astype(np.int64) - ort_idx.astype(np.int64))))
        raise AssertionError(f'BallQuery index mismatch: diff_count={diff_count}, max_diff={max_diff}')

    print('IASSD::BallQuery ORT custom op 테스트 통과')
    print(
        f'xyz_shape={tuple(xyz_np.shape)} '
        f'new_xyz_shape={tuple(new_xyz_np.shape)} '
        f'idx_shape={tuple(ort_idx.shape)} '
        f'radius={args.radius} nsample={args.nsample}'
    )


if __name__ == '__main__':
    main()
