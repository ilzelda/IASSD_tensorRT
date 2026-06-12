import argparse
import ctypes
import json
from pathlib import Path

import tensorrt as trt


def parse_args():
    parser = argparse.ArgumentParser(description='IA-SSD ONNX를 TensorRT direct engine으로 빌드')
    parser.add_argument('--onnx_file', type=str, required=True, help='TensorRT parser에 넣을 ONNX 파일')
    parser.add_argument('--engine_file', type=str, default=None, help='저장할 TensorRT engine 파일')
    parser.add_argument(
        '--plugin_library',
        type=str,
        action='append',
        default=[],
        help='dlopen할 TensorRT plugin shared library. 여러 번 지정 가능',
    )
    parser.add_argument('--fp16', action='store_true', help='FP16 builder flag 활성화')
    parser.add_argument('--workspace_mb', type=int, default=2048, help='TensorRT workspace 크기(MB)')
    parser.add_argument('--batch_size', type=int, default=1, help='동적 batch dim에 사용할 batch size')
    parser.add_argument('--num_points', type=int, default=16384, help='동적 point dim에 사용할 point 수')
    parser.add_argument('--feature_dim', type=int, default=5, help='동적 feature dim에 사용할 feature 수')
    parser.add_argument('--strict_parse', action='store_true', help='parse 실패 시 즉시 종료')
    parser.add_argument('--verbose', action='store_true', help='TensorRT verbose 로그 활성화')
    parser.add_argument('--disable_tf32', action='store_true', help='FP32 parity 확인을 위해 TensorRT TF32 tactic 사용을 비활성화')
    parser.add_argument(
        '--tactic_sources',
        type=str,
        default=None,
        help='쉼표로 구분한 TensorRT tactic source 제한값. 예: CUBLAS,CUDNN',
    )
    parser.add_argument(
        '--log_level',
        type=str,
        choices=['verbose', 'info', 'warning', 'error'],
        default='warning',
        help='TensorRT logger 출력 수준',
    )
    return parser.parse_args()


def load_plugin_libraries(plugin_libraries):
    loaded = []
    for library in plugin_libraries:
        library_path = Path(library).resolve()
        if not library_path.exists():
            raise FileNotFoundError(f'TensorRT plugin library가 없습니다: {library_path}')
        ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)
        loaded.append(str(library_path))
    return loaded


def set_workspace(config, workspace_bytes):
    if hasattr(config, 'set_memory_pool_limit') and hasattr(trt, 'MemoryPoolType'):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:
        # 구형 TensorRT Python binding 호환 경로다.
        config.max_workspace_size = workspace_bytes


def resolve_dynamic_dim(dim, index, args):
    if dim >= 0:
        return dim
    if index == 0:
        return args.batch_size
    if index == 1:
        return args.num_points
    if index == 2:
        return args.feature_dim
    raise ValueError(f'자동 profile 값을 정할 수 없는 dynamic dim index입니다: {index}')


def make_profile_shape(dims, args):
    return tuple(resolve_dynamic_dim(int(dim), index, args) for index, dim in enumerate(dims))


def configure_profiles(builder, network, config, args):
    profile = builder.create_optimization_profile()
    has_dynamic_input = False

    for input_index in range(network.num_inputs):
        tensor = network.get_input(input_index)
        dims = tuple(int(dim) for dim in tensor.shape)
        if any(dim < 0 for dim in dims):
            has_dynamic_input = True
            shape = make_profile_shape(dims, args)
            profile.set_shape(tensor.name, shape, shape, shape)

    if has_dynamic_input:
        config.add_optimization_profile(profile)


def configure_tactic_sources(config, tactic_sources):
    if tactic_sources is None:
        return None
    if not hasattr(config, 'set_tactic_sources') or not hasattr(trt, 'TacticSource'):
        raise RuntimeError('현재 TensorRT Python binding은 tactic source 제한을 지원하지 않습니다.')

    source_names = [name.strip() for name in tactic_sources.split(',') if name.strip()]
    source_mask = 0
    for source_name in source_names:
        if not hasattr(trt.TacticSource, source_name):
            raise ValueError(f'지원하지 않는 TensorRT tactic source입니다: {source_name}')
        source_mask |= 1 << int(getattr(trt.TacticSource, source_name))
    config.set_tactic_sources(source_mask)
    return source_names


def describe_network(network):
    inputs = []
    outputs = []
    layers = []

    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        inputs.append({
            'name': tensor.name,
            'shape': tuple(int(dim) for dim in tensor.shape),
            'dtype': str(tensor.dtype),
        })

    for index in range(network.num_outputs):
        tensor = network.get_output(index)
        outputs.append({
            'name': tensor.name,
            'shape': tuple(int(dim) for dim in tensor.shape),
            'dtype': str(tensor.dtype),
        })

    for index in range(network.num_layers):
        layer = network.get_layer(index)
        layers.append({
            'index': index,
            'name': layer.name,
            'type': str(layer.type),
        })

    return {
        'inputs': inputs,
        'outputs': outputs,
        'num_layers': network.num_layers,
        'layers_head': layers[:30],
    }


def collect_parser_errors(parser):
    return [
        {
            'index': index,
            'code': int(parser.get_error(index).code()),
            'desc': parser.get_error(index).desc(),
        }
        for index in range(parser.num_errors)
    ]


def build_engine(args):
    onnx_file = Path(args.onnx_file).resolve()
    if not onnx_file.exists():
        raise FileNotFoundError(f'ONNX 파일이 없습니다: {onnx_file}')

    logger_levels = {
        'verbose': trt.Logger.VERBOSE,
        'info': trt.Logger.INFO,
        'warning': trt.Logger.WARNING,
        'error': trt.Logger.ERROR,
    }
    logger_level = trt.Logger.VERBOSE if args.verbose else logger_levels[args.log_level]
    logger = trt.Logger(logger_level)
    loaded_plugins = load_plugin_libraries(args.plugin_library)
    trt.init_libnvinfer_plugins(logger, '')

    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    parsed = parser.parse_from_file(str(onnx_file))
    parser_errors = collect_parser_errors(parser)
    report = {
        'onnx_file': str(onnx_file),
        'engine_file': str(Path(args.engine_file).resolve()) if args.engine_file else None,
        'plugin_libraries': loaded_plugins,
        'fp16': args.fp16,
        'disable_tf32': args.disable_tf32,
        'tactic_sources': args.tactic_sources,
        'workspace_mb': args.workspace_mb,
        'parse_ok': bool(parsed),
        'parser_errors': parser_errors,
    }

    if not parsed:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if args.strict_parse:
            raise RuntimeError('TensorRT ONNX parse 실패')
        return report

    report['network'] = describe_network(network)

    config = builder.create_builder_config()
    set_workspace(config, args.workspace_mb * 1024 * 1024)
    if args.fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if args.disable_tf32 and hasattr(trt.BuilderFlag, 'TF32'):
        config.clear_flag(trt.BuilderFlag.TF32)
    report['tactic_sources_active'] = configure_tactic_sources(config, args.tactic_sources)
    configure_profiles(builder, network, config, args)

    serialized_engine = builder.build_serialized_network(network, config)
    report['build_ok'] = serialized_engine is not None
    if serialized_engine is None:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        raise RuntimeError('TensorRT engine build 실패')

    if args.engine_file is not None:
        engine_file = Path(args.engine_file).resolve()
        engine_file.parent.mkdir(parents=True, exist_ok=True)
        engine_file.write_bytes(bytes(serialized_engine))
        report['engine_size_bytes'] = engine_file.stat().st_size
    else:
        report['engine_size_bytes'] = len(bytes(serialized_engine))

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main():
    args = parse_args()
    build_engine(args)


if __name__ == '__main__':
    main()
