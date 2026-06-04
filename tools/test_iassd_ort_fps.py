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
    parser = argparse.ArgumentParser(description='IASSD::FarthestPointSampling ORT custom op 단위 테스트')
    parser.add_argument('--ort_op_library', type=str, required=True, help='libiassd_ort_ops.so 경로')
    parser.add_argument('--batch_size', type=int, default=1, help='테스트 batch 크기')
    parser.add_argument('--num_points', type=int, default=512, help='입력 포인트 수')
    parser.add_argument('--npoint', type=int, default=64, help='FPS로 샘플링할 포인트 수')
    parser.add_argument('--seed', type=int, default=1024, help='입력 생성 seed')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda'], help='현재 custom op는 CUDA EP만 지원')
    return parser.parse_args()


def make_fps_model(path, batch_size, num_points, npoint):
    graph = helper.make_graph(
        nodes=[
            helper.make_node(
                'FarthestPointSampling',
                inputs=['xyz'],
                outputs=['idx'],
                domain='IASSD',
                npoint_i=npoint,
            )
        ],
        name='iassd_fps_test',
        inputs=[
            helper.make_tensor_value_info('xyz', TensorProto.FLOAT, [batch_size, num_points, 3])
        ],
        outputs=[
            helper.make_tensor_value_info('idx', TensorProto.INT32, [batch_size, npoint])
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_operatorsetid('', 17),
            helper.make_operatorsetid('IASSD', 1),
        ],
        producer_name='iassd_ort_fps_test',
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)


def run_torch_fps(xyz_np, npoint):
    xyz = torch.from_numpy(xyz_np).cuda().contiguous()
    with torch.no_grad():
        idx = pointnet2_utils.farthest_point_sample(xyz, npoint)
    torch.cuda.synchronize()
    return idx.cpu().numpy()


def run_ort_fps(model_path, library_path, xyz_np):
    options = ort.SessionOptions()
    options.register_custom_ops_library(str(library_path))
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=['CUDAExecutionProvider'])
    return session.run(['idx'], {'xyz': xyz_np})[0]


def main():
    args = parse_args()
    library_path = Path(args.ort_op_library).resolve()
    if not library_path.exists():
        raise FileNotFoundError(f'custom op library가 없습니다: {library_path}')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA를 사용할 수 없습니다.')

    rng = np.random.default_rng(args.seed)
    xyz_np = rng.random((args.batch_size, args.num_points, 3), dtype=np.float32)

    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / 'iassd_fps_test.onnx'
        make_fps_model(model_path, args.batch_size, args.num_points, args.npoint)

        torch_idx = run_torch_fps(xyz_np, args.npoint)
        ort_idx = run_ort_fps(model_path, library_path, xyz_np)

    if not np.array_equal(torch_idx, ort_idx):
        diff_count = int(np.count_nonzero(torch_idx != ort_idx))
        max_diff = int(np.max(np.abs(torch_idx.astype(np.int64) - ort_idx.astype(np.int64))))
        raise AssertionError(f'FPS index mismatch: diff_count={diff_count}, max_diff={max_diff}')

    print('IASSD::FarthestPointSampling ORT custom op 테스트 통과')
    print(f'input_shape={tuple(xyz_np.shape)} output_shape={tuple(ort_idx.shape)} npoint={args.npoint}')


if __name__ == '__main__':
    main()
