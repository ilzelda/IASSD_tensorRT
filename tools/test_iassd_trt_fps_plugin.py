import argparse
import ctypes
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import onnx
import tensorrt as trt
import torch
from onnx import TensorProto, helper

from pcdet.ops.pointnet2.pointnet2_batch import pointnet2_utils


def parse_args():
    parser = argparse.ArgumentParser(description='IASSD::FarthestPointSampling TensorRT plugin 빌드 테스트')
    parser.add_argument('--plugin_library', type=str, required=True, help='libiassd_trt_plugins.so 경로')
    parser.add_argument('--batch_size', type=int, default=1, help='테스트 batch 크기')
    parser.add_argument('--num_points', type=int, default=512, help='입력 포인트 수')
    parser.add_argument('--npoint', type=int, default=64, help='FPS로 샘플링할 포인트 수')
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
        name='iassd_trt_fps_test',
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
        producer_name='iassd_trt_fps_test',
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)


def build_engine(model_path):
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, '')
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    model_bytes = Path(model_path).read_bytes()
    if not parser.parse(model_bytes):
        messages = []
        for index in range(parser.num_errors):
            messages.append(str(parser.get_error(index)))
        raise RuntimeError('TensorRT ONNX parse 실패:\n' + '\n'.join(messages))

    config = builder.create_builder_config()
    config.max_workspace_size = 256 << 20
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError('TensorRT engine build 실패')
    return serialized


def run_torch_fps(xyz_np, npoint):
    xyz = torch.from_numpy(xyz_np).cuda().contiguous()
    with torch.no_grad():
        idx = pointnet2_utils.farthest_point_sample(xyz, npoint)
    torch.cuda.synchronize()
    return idx.cpu().numpy()


def run_trt_fps(serialized_engine, xyz_np, npoint):
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    if engine is None:
        raise RuntimeError('TensorRT engine deserialize 실패')
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError('TensorRT execution context 생성 실패')

    xyz = torch.from_numpy(xyz_np).cuda().contiguous()
    idx = torch.empty((xyz_np.shape[0], npoint), dtype=torch.int32, device='cuda')
    bindings = [0] * engine.num_bindings
    bindings[engine.get_binding_index('xyz')] = int(xyz.data_ptr())
    bindings[engine.get_binding_index('idx')] = int(idx.data_ptr())

    stream = torch.cuda.current_stream().cuda_stream
    ok = context.execute_async_v2(bindings=bindings, stream_handle=stream)
    if not ok:
        raise RuntimeError('TensorRT execute_async_v2 실패')
    torch.cuda.synchronize()
    return idx.cpu().numpy()


def main():
    args = parse_args()
    plugin_library = Path(args.plugin_library).resolve()
    if not plugin_library.exists():
        raise FileNotFoundError(f'TensorRT plugin library가 없습니다: {plugin_library}')

    # plugin creator 등록을 위해 shared library를 먼저 로드한다.
    ctypes.CDLL(str(plugin_library), mode=ctypes.RTLD_GLOBAL)

    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / 'iassd_trt_fps_test.onnx'
        make_fps_model(model_path, args.batch_size, args.num_points, args.npoint)
        serialized = build_engine(model_path)

    rng = np.random.default_rng(1024)
    xyz_np = rng.random((args.batch_size, args.num_points, 3), dtype=np.float32)
    torch_idx = run_torch_fps(xyz_np, args.npoint)
    trt_idx = run_trt_fps(serialized, xyz_np, args.npoint)
    if not np.array_equal(torch_idx, trt_idx):
        diff_count = int(np.count_nonzero(torch_idx != trt_idx))
        max_diff = int(np.max(np.abs(torch_idx.astype(np.int64) - trt_idx.astype(np.int64))))
        raise AssertionError(f'TensorRT FPS index mismatch: diff_count={diff_count}, max_diff={max_diff}')

    print('IASSD::FarthestPointSampling TensorRT plugin engine build/execute 테스트 통과')
    print(
        f'input_shape=({args.batch_size}, {args.num_points}, 3) '
        f'output_shape=({args.batch_size}, {args.npoint}) '
        f'engine_bytes={len(bytes(serialized))}'
    )


if __name__ == '__main__':
    main()
