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
    parser = argparse.ArgumentParser(description='IA-SSD TensorRT PointNet2 plugin 단위 테스트')
    parser.add_argument('--plugin_library', type=str, required=True, help='libiassd_trt_plugins.so 경로')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_points', type=int, default=512)
    parser.add_argument('--channels', type=int, default=4)
    parser.add_argument('--npoint', type=int, default=64)
    parser.add_argument('--nsample', type=int, default=16)
    parser.add_argument('--radius', type=float, default=0.2)
    return parser.parse_args()


def save_model(path, nodes, inputs, outputs):
    graph = helper.make_graph(nodes=nodes, name='iassd_trt_plugin_test', inputs=inputs, outputs=outputs)
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_operatorsetid('', 17),
            helper.make_operatorsetid('IASSD', 1),
        ],
        producer_name='iassd_trt_plugin_test',
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)


def build_engine(model_path):
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(Path(model_path).read_bytes()):
        errors = [parser.get_error(index).desc() for index in range(parser.num_errors)]
        raise RuntimeError('TensorRT ONNX parse 실패:\n' + '\n'.join(errors))
    config = builder.create_builder_config()
    if hasattr(config, 'set_memory_pool_limit') and hasattr(trt, 'MemoryPoolType'):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 << 20)
    else:
        config.max_workspace_size = 256 << 20
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError('TensorRT engine build 실패')
    return serialized


def run_engine(serialized_engine, input_tensors, output_specs):
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    if engine is None:
        raise RuntimeError('TensorRT engine deserialize 실패')
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError('TensorRT execution context 생성 실패')

    outputs = {}
    bindings = [0] * engine.num_bindings
    for name, tensor in input_tensors.items():
        bindings[engine.get_binding_index(name)] = int(tensor.data_ptr())
    for name, shape, dtype in output_specs:
        output = torch.empty(shape, dtype=dtype, device='cuda')
        outputs[name] = output
        bindings[engine.get_binding_index(name)] = int(output.data_ptr())

    stream = torch.cuda.current_stream().cuda_stream
    ok = context.execute_async_v2(bindings=bindings, stream_handle=stream)
    if not ok:
        raise RuntimeError('TensorRT engine 실행 실패')
    torch.cuda.synchronize()
    return {name: output.cpu().numpy() for name, output in outputs.items()}


def assert_equal(name, expected, actual):
    if not np.array_equal(expected, actual):
        diff = actual.astype(np.float64) - expected.astype(np.float64)
        raise AssertionError(
            f'{name} mismatch: max_abs={float(np.max(np.abs(diff)))}, '
            f'mean_abs={float(np.mean(np.abs(diff)))}, '
            f'diff_count={int(np.count_nonzero(diff))}'
        )


def test_farthest_point_sampling(temp_dir, args, rng):
    model_path = Path(temp_dir) / 'farthest_point_sampling.onnx'
    save_model(
        model_path,
        nodes=[
            helper.make_node(
                'FarthestPointSampling',
                inputs=['xyz'],
                outputs=['idx'],
                domain='IASSD',
                npoint_i=args.npoint,
            )
        ],
        inputs=[
            helper.make_tensor_value_info('xyz', TensorProto.FLOAT, [args.batch_size, args.num_points, 3]),
        ],
        outputs=[
            helper.make_tensor_value_info('idx', TensorProto.INT32, [args.batch_size, args.npoint])
        ],
    )
    xyz = torch.from_numpy(rng.random((args.batch_size, args.num_points, 3), dtype=np.float32)).cuda()
    with torch.no_grad():
        expected = pointnet2_utils.furthest_point_sample(xyz, args.npoint).cpu().numpy()
    actual = run_engine(
        build_engine(model_path),
        {'xyz': xyz.contiguous()},
        [('idx', expected.shape, torch.int32)],
    )['idx']
    assert_equal('FarthestPointSampling', expected, actual)


def test_gather(temp_dir, args, rng):
    model_path = Path(temp_dir) / 'gather.onnx'
    save_model(
        model_path,
        nodes=[
            helper.make_node('GatherPoints', inputs=['features', 'idx'], outputs=['output'], domain='IASSD')
        ],
        inputs=[
            helper.make_tensor_value_info('features', TensorProto.FLOAT, [args.batch_size, args.channels, args.num_points]),
            helper.make_tensor_value_info('idx', TensorProto.INT32, [args.batch_size, args.npoint]),
        ],
        outputs=[
            helper.make_tensor_value_info('output', TensorProto.FLOAT, [args.batch_size, args.channels, args.npoint])
        ],
    )
    features = torch.from_numpy(rng.random((args.batch_size, args.channels, args.num_points), dtype=np.float32)).cuda()
    idx_np = rng.integers(0, args.num_points, size=(args.batch_size, args.npoint), dtype=np.int32)
    idx = torch.from_numpy(idx_np).cuda()
    with torch.no_grad():
        expected = pointnet2_utils.gather_operation(features, idx).cpu().numpy()
    actual = run_engine(
        build_engine(model_path),
        {'features': features.contiguous(), 'idx': idx.contiguous()},
        [('output', expected.shape, torch.float32)],
    )['output']
    assert_equal('GatherPoints', expected, actual)


def test_ball_query(temp_dir, args, rng):
    model_path = Path(temp_dir) / 'ball_query.onnx'
    save_model(
        model_path,
        nodes=[
            helper.make_node(
                'BallQuery',
                inputs=['xyz', 'new_xyz'],
                outputs=['idx'],
                domain='IASSD',
                radius_f=args.radius,
                nsample_i=args.nsample,
            )
        ],
        inputs=[
            helper.make_tensor_value_info('xyz', TensorProto.FLOAT, [args.batch_size, args.num_points, 3]),
            helper.make_tensor_value_info('new_xyz', TensorProto.FLOAT, [args.batch_size, args.npoint, 3]),
        ],
        outputs=[
            helper.make_tensor_value_info('idx', TensorProto.INT32, [args.batch_size, args.npoint, args.nsample])
        ],
    )
    xyz = torch.from_numpy(rng.random((args.batch_size, args.num_points, 3), dtype=np.float32)).cuda()
    new_xyz = xyz[:, :args.npoint, :].contiguous()
    with torch.no_grad():
        expected = pointnet2_utils.ball_query(args.radius, args.nsample, xyz, new_xyz).cpu().numpy()
    actual = run_engine(
        build_engine(model_path),
        {'xyz': xyz.contiguous(), 'new_xyz': new_xyz.contiguous()},
        [('idx', expected.shape, torch.int32)],
    )['idx']
    assert_equal('BallQuery', expected, actual)


def test_group(temp_dir, args, rng):
    model_path = Path(temp_dir) / 'group.onnx'
    save_model(
        model_path,
        nodes=[
            helper.make_node('GroupPoints', inputs=['features', 'idx'], outputs=['output'], domain='IASSD')
        ],
        inputs=[
            helper.make_tensor_value_info('features', TensorProto.FLOAT, [args.batch_size, args.channels, args.num_points]),
            helper.make_tensor_value_info('idx', TensorProto.INT32, [args.batch_size, args.npoint, args.nsample]),
        ],
        outputs=[
            helper.make_tensor_value_info('output', TensorProto.FLOAT, [args.batch_size, args.channels, args.npoint, args.nsample])
        ],
    )
    features = torch.from_numpy(rng.random((args.batch_size, args.channels, args.num_points), dtype=np.float32)).cuda()
    idx_np = rng.integers(0, args.num_points, size=(args.batch_size, args.npoint, args.nsample), dtype=np.int32)
    idx = torch.from_numpy(idx_np).cuda()
    with torch.no_grad():
        expected = pointnet2_utils.grouping_operation(features, idx).cpu().numpy()
    actual = run_engine(
        build_engine(model_path),
        {'features': features.contiguous(), 'idx': idx.contiguous()},
        [('output', expected.shape, torch.float32)],
    )['output']
    assert_equal('GroupPoints', expected, actual)


def main():
    args = parse_args()
    plugin_library = Path(args.plugin_library).resolve()
    if not plugin_library.exists():
        raise FileNotFoundError(f'TensorRT plugin library가 없습니다: {plugin_library}')
    ctypes.CDLL(str(plugin_library), mode=ctypes.RTLD_GLOBAL)

    rng = np.random.default_rng(1024)
    with tempfile.TemporaryDirectory() as temp_dir:
        test_farthest_point_sampling(temp_dir, args, rng)
        test_gather(temp_dir, args, rng)
        test_ball_query(temp_dir, args, rng)
        test_group(temp_dir, args, rng)

    print('IA-SSD TensorRT PointNet2 plugin 단위 테스트 통과')
    print(
        f'batch_size={args.batch_size} num_points={args.num_points} '
        f'channels={args.channels} npoint={args.npoint} nsample={args.nsample} radius={args.radius}'
    )


if __name__ == '__main__':
    main()
