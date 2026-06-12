import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


TENSOR_DTYPE_TO_NUMPY = {
    TensorProto.FLOAT: np.float32,
    TensorProto.FLOAT16: np.float16,
    TensorProto.DOUBLE: np.float64,
    TensorProto.INT8: np.int8,
    TensorProto.INT16: np.int16,
    TensorProto.INT32: np.int32,
    TensorProto.INT64: np.int64,
    TensorProto.UINT8: np.uint8,
    TensorProto.UINT16: np.uint16,
    TensorProto.UINT32: np.uint32,
    TensorProto.UINT64: np.uint64,
    TensorProto.BOOL: np.bool_,
}


def parse_args():
    parser = argparse.ArgumentParser(description='TensorRT parser용 ONNX graph 정리')
    parser.add_argument('--input', type=str, required=True, help='입력 ONNX 파일')
    parser.add_argument('--output', type=str, required=True, help='저장할 ONNX 파일')
    parser.add_argument('--skip_check', action='store_true', help='ONNX checker 검사를 건너뜀')
    parser.add_argument(
        '--disabled_lowerings',
        type=str,
        default='',
        help='쉼표로 구분한 비활성 lowering 이름. 예: batch_norm,aten_split',
    )
    parser.add_argument(
        '--enable_conv_batch_norm_fold',
        action='store_true',
        help='실험용 Conv+BatchNorm fold를 활성화',
    )
    parser.add_argument(
        '--sort_topk_indices',
        action='store_true',
        help='실험용 TopK index 오름차순 정렬을 활성화',
    )
    return parser.parse_args()


def parse_disabled_lowerings(value):
    return {name.strip() for name in value.split(',') if name.strip()}


def run_lowering(name, disabled_lowerings, lowering_fn, graph):
    if name in disabled_lowerings:
        return 0
    return lowering_fn(graph)


def get_attribute(node, name, default=None):
    for attribute in node.attribute:
        if attribute.name == name:
            return helper.get_attribute_value(attribute)
    return default


def get_constant_value(node):
    value = get_attribute(node, 'value')
    if value is not None:
        return numpy_helper.to_array(value)

    value_int = get_attribute(node, 'value_int')
    if value_int is not None:
        return np.asarray(value_int, dtype=np.int64)

    value_ints = get_attribute(node, 'value_ints')
    if value_ints is not None:
        return np.asarray(value_ints, dtype=np.int64)

    value_float = get_attribute(node, 'value_float')
    if value_float is not None:
        return np.asarray(value_float, dtype=np.float32)

    value_floats = get_attribute(node, 'value_floats')
    if value_floats is not None:
        return np.asarray(value_floats, dtype=np.float32)

    return None


def evaluate_constant_graph(graph, constants):
    local_constants = dict(constants)

    for node in graph.node:
        folded_values = fold_node(node, local_constants)
        if folded_values is None:
            return None
        for output_name, value in zip(node.output, folded_values):
            if output_name:
                local_constants[output_name] = value

    outputs = []
    for output in graph.output:
        if output.name not in local_constants:
            return None
        outputs.append(local_constants[output.name])
    return outputs


def evaluate_repeat_branch(graph, constants):
    for node in graph.node:
        if node.op_type != 'Tile' or len(node.input) != 2:
            continue
        repeat_name = node.input[1]
        source_name = None
        for candidate in graph.node:
            if candidate.op_type == 'Cast' and candidate.output and candidate.output[0] == repeat_name and candidate.input:
                repeat_name = candidate.input[0]
            if candidate.op_type == 'Expand' and len(candidate.input) >= 1 and candidate.output and candidate.output[0] == node.input[0]:
                source_name = candidate.input[0]
        if source_name is None:
            continue
        if source_name not in constants or repeat_name not in constants:
            continue
        source = np.asarray(constants[source_name])
        repeats = np.asarray(constants[repeat_name], dtype=np.int64)
        if repeats.ndim != 1 or repeats.size < source.ndim:
            continue
        padded_shape = (1,) * (repeats.size - source.ndim) + source.shape
        return [np.tile(source.reshape(padded_shape), repeats)]

    return None


def fold_node(node, constants):
    if node.op_type == 'Constant' and node.domain == '':
        value = get_constant_value(node)
        if value is None:
            return None
        return [value]

    inputs = []
    for name in node.input:
        if name not in constants:
            return None
        inputs.append(constants[name])

    if node.domain != '':
        return None

    if node.op_type == 'Cast':
        target_type = get_attribute(node, 'to')
        numpy_dtype = TENSOR_DTYPE_TO_NUMPY.get(target_type)
        if numpy_dtype is None:
            return None
        return [inputs[0].astype(numpy_dtype)]

    if node.op_type == 'Identity':
        return [inputs[0]]

    if node.op_type == 'Size':
        return [np.asarray(inputs[0].size, dtype=np.int64)]

    if node.op_type == 'Shape':
        return [np.asarray(inputs[0].shape, dtype=np.int64)]

    if node.op_type == 'Equal':
        return [np.equal(inputs[0], inputs[1])]

    if node.op_type == 'Reshape':
        allowzero = int(get_attribute(node, 'allowzero', 0))
        shape = inputs[1].astype(np.int64).tolist()
        if allowzero == 0:
            source_shape = list(inputs[0].shape)
            shape = [
                source_shape[index] if dim == 0 and index < len(source_shape) else dim
                for index, dim in enumerate(shape)
            ]
        try:
            return [np.reshape(inputs[0], shape)]
        except ValueError:
            return None

    if node.op_type == 'Concat':
        axis = int(get_attribute(node, 'axis', 0))
        return [np.concatenate(inputs, axis=axis)]

    if node.op_type == 'Expand':
        try:
            return [np.broadcast_to(inputs[0], tuple(inputs[1].astype(np.int64).tolist()))]
        except ValueError:
            return None

    if node.op_type == 'Tile':
        try:
            return [np.tile(inputs[0], inputs[1].astype(np.int64))]
        except ValueError:
            return None

    if node.op_type == 'Unsqueeze':
        axes = inputs[1].astype(np.int64).tolist() if len(inputs) > 1 else get_attribute(node, 'axes')
        value = inputs[0]
        for axis in sorted(int(axis) for axis in axes):
            value = np.expand_dims(value, axis)
        return [value]

    if node.op_type == 'Squeeze':
        axes = inputs[1].astype(np.int64).tolist() if len(inputs) > 1 else get_attribute(node, 'axes')
        if axes is None:
            try:
                return [np.squeeze(inputs[0])]
            except ValueError:
                return None
        try:
            return [np.squeeze(inputs[0], axis=tuple(int(axis) for axis in axes))]
        except ValueError:
            return None

    if node.op_type == 'If':
        condition = bool(np.asarray(inputs[0]).item())
        branch = get_attribute(node, 'then_branch' if condition else 'else_branch')
        if branch is None:
            return None
        folded_values = evaluate_constant_graph(branch, constants)
        if folded_values is not None:
            return folded_values
        return evaluate_repeat_branch(branch, constants)

    return None


def make_initializer(name, value):
    value = np.asarray(value)
    return numpy_helper.from_array(value, name=name)


def collect_value_types(graph):
    value_types = {}

    for value_info in list(graph.input) + list(graph.output) + list(graph.value_info):
        tensor_type = value_info.type.tensor_type
        if tensor_type.HasField('elem_type'):
            value_types[value_info.name] = tensor_type.elem_type

    for initializer in graph.initializer:
        value_types[initializer.name] = initializer.data_type

    return value_types


def infer_value_types(graph):
    value_types = collect_value_types(graph)
    unary_same_type_ops = {
        'Abs',
        'Ceil',
        'Clip',
        'Exp',
        'Floor',
        'Log',
        'Neg',
        'Reciprocal',
        'Relu',
        'Sigmoid',
        'Sqrt',
        'Tanh',
    }
    variadic_same_type_ops = {
        'Add',
        'Concat',
        'Div',
        'Max',
        'Min',
        'Mul',
        'Sub',
    }

    changed = True
    while changed:
        changed = False
        for node in graph.node:
            inferred_type = None
            if node.op_type == 'Cast' and len(node.output) == 1:
                inferred_type = get_attribute(node, 'to')
            elif node.op_type in {'Shape', 'Size'}:
                inferred_type = TensorProto.INT64
            elif node.op_type in unary_same_type_ops and len(node.input) >= 1:
                inferred_type = value_types.get(node.input[0])
            elif node.op_type in variadic_same_type_ops:
                for input_name in node.input:
                    inferred_type = value_types.get(input_name)
                    if inferred_type is not None:
                        break

            if inferred_type is None:
                continue

            for output_name in node.output:
                if output_name and value_types.get(output_name) != inferred_type:
                    value_types[output_name] = inferred_type
                    changed = True

    return value_types


def collect_value_ranks(graph):
    value_ranks = {}

    for value_info in list(graph.input) + list(graph.output) + list(graph.value_info):
        tensor_type = value_info.type.tensor_type
        if tensor_type.HasField('shape'):
            value_ranks[value_info.name] = len(tensor_type.shape.dim)

    for initializer in graph.initializer:
        value_ranks[initializer.name] = len(initializer.dims)

    return value_ranks


def collect_value_shapes(graph):
    value_shapes = {}

    for value_info in list(graph.input) + list(graph.output) + list(graph.value_info):
        tensor_type = value_info.type.tensor_type
        if not tensor_type.HasField('shape'):
            continue
        shape = []
        known = True
        for dim in tensor_type.shape.dim:
            if dim.HasField('dim_value'):
                shape.append(int(dim.dim_value))
            else:
                known = False
                break
        if known:
            value_shapes[value_info.name] = tuple(shape)

    for initializer in graph.initializer:
        value_shapes[initializer.name] = tuple(int(dim) for dim in initializer.dims)

    return value_shapes


def lower_castlike_nodes(graph):
    value_types = infer_value_types(graph)
    lowered_count = 0

    for node in graph.node:
        if node.domain != '' or node.op_type != 'CastLike':
            continue
        if len(node.input) != 2:
            continue
        target_type = value_types.get(node.input[1])
        if target_type is None:
            continue
        node.op_type = 'Cast'
        del node.input[1:]
        del node.attribute[:]
        node.attribute.extend([helper.make_attribute('to', int(target_type))])
        lowered_count += 1

    return lowered_count


def lower_isscalar_nodes(graph):
    value_ranks = collect_value_ranks(graph)
    lowered_initializers = {}
    kept_nodes = []
    lowered_count = 0

    for node in graph.node:
        if node.op_type != 'IsScalar' or len(node.input) != 1 or len(node.output) != 1:
            kept_nodes.append(node)
            continue
        input_rank = value_ranks.get(node.input[0])
        if input_rank is None:
            kept_nodes.append(node)
            continue
        lowered_initializers[node.output[0]] = make_initializer(node.output[0], np.asarray(input_rank == 0, dtype=np.bool_))
        lowered_count += 1

    if lowered_count == 0:
        return 0

    del graph.node[:]
    graph.node.extend(kept_nodes)
    graph.initializer.extend(lowered_initializers.values())
    return lowered_count


def lower_aten_squeeze_dim_nodes(graph):
    lowered_initializers = []
    lowered_count = 0

    for node in graph.node:
        if node.domain != 'pkg.onnxscript.torch_lib' or node.op_type != 'aten_squeeze_dim':
            continue
        dim = get_attribute(node, 'dim')
        if dim is None or len(node.input) != 1 or len(node.output) != 1:
            continue
        input_name = node.input[0]
        axes_name = f'{node.output[0]}__trt_squeeze_axes'
        lowered_initializers.append(make_initializer(axes_name, np.asarray([int(dim)], dtype=np.int64)))
        node.domain = ''
        node.op_type = 'Squeeze'
        del node.input[:]
        node.input.extend([input_name, axes_name])
        del node.attribute[:]
        lowered_count += 1

    if lowered_count == 0:
        return 0

    graph.initializer.extend(lowered_initializers)
    return lowered_count


def lower_aten_unsqueeze_nodes(graph):
    lowered_initializers = []
    lowered_count = 0

    for node in graph.node:
        if node.domain != 'pkg.onnxscript.torch_lib' or node.op_type != 'aten_unsqueeze':
            continue
        dim = get_attribute(node, 'dim')
        if dim is None or len(node.input) != 1 or len(node.output) != 1:
            continue
        input_name = node.input[0]
        axes_name = f'{node.output[0]}__trt_unsqueeze_axes'
        lowered_initializers.append(make_initializer(axes_name, np.asarray([int(dim)], dtype=np.int64)))
        node.domain = ''
        node.op_type = 'Unsqueeze'
        del node.input[:]
        node.input.extend([input_name, axes_name])
        del node.attribute[:]
        lowered_count += 1

    if lowered_count == 0:
        return 0

    graph.initializer.extend(lowered_initializers)
    return lowered_count


def collect_input_uses(graph):
    input_uses = {}
    for node in graph.node:
        for input_name in node.input:
            if not input_name:
                continue
            input_uses[input_name] = input_uses.get(input_name, 0) + 1
    return input_uses


def fold_conv_native_batch_norm_inference_nodes(graph):
    original_nodes = list(graph.node)
    input_uses = collect_input_uses(graph)
    constants = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in graph.initializer
    }
    consumers = {}
    for node in graph.node:
        for input_name in node.input:
            if not input_name:
                continue
            consumers.setdefault(input_name, []).append(node)

    skipped_nodes = set()
    folded_initializers = []
    folded_count = 0

    for conv_node in graph.node:
        if conv_node.op_type != 'Conv' or len(conv_node.input) < 2 or len(conv_node.output) != 1:
            continue
        conv_output = conv_node.output[0]
        output_consumers = consumers.get(conv_output, [])
        if len(output_consumers) != 1:
            continue
        batch_norm_node = output_consumers[0]
        if (
            batch_norm_node.domain != 'pkg.onnxscript.torch_lib'
            or batch_norm_node.op_type != '_aten_native_batch_norm_inference_onnx'
            or len(batch_norm_node.input) != 5
            or len(batch_norm_node.output) < 1
        ):
            continue
        if any(input_uses.get(output_name, 0) != 0 for output_name in batch_norm_node.output[1:]):
            continue

        conv_input_name = conv_node.input[0]
        weight_name = conv_node.input[1]
        bias_name = conv_node.input[2] if len(conv_node.input) >= 3 and conv_node.input[2] else None
        needed_names = [weight_name, batch_norm_node.input[1], batch_norm_node.input[2], batch_norm_node.input[3], batch_norm_node.input[4]]
        if bias_name is not None:
            needed_names.append(bias_name)
        if any(name not in constants for name in needed_names):
            continue

        conv_weight = np.asarray(constants[weight_name])
        if conv_weight.ndim < 1:
            continue
        out_channels = conv_weight.shape[0]
        bn_weight = np.asarray(constants[batch_norm_node.input[1]], dtype=np.float32).reshape(-1)
        bn_bias = np.asarray(constants[batch_norm_node.input[2]], dtype=np.float32).reshape(-1)
        bn_mean = np.asarray(constants[batch_norm_node.input[3]], dtype=np.float32).reshape(-1)
        bn_var = np.asarray(constants[batch_norm_node.input[4]], dtype=np.float32).reshape(-1)
        if any(value.shape[0] != out_channels for value in [bn_weight, bn_bias, bn_mean, bn_var]):
            continue

        epsilon = float(get_attribute(batch_norm_node, 'eps', 1e-5))
        conv_bias = (
            np.asarray(constants[bias_name], dtype=np.float32).reshape(-1)
            if bias_name is not None
            else np.zeros(out_channels, dtype=np.float32)
        )
        if conv_bias.shape[0] != out_channels:
            continue

        scale = bn_weight / np.sqrt(bn_var + epsilon)
        scale_shape = (out_channels,) + (1,) * (conv_weight.ndim - 1)
        folded_weight = (conv_weight.astype(np.float32) * scale.reshape(scale_shape)).astype(conv_weight.dtype)
        folded_bias = ((conv_bias - bn_mean) * scale + bn_bias).astype(np.float32)
        folded_weight_name = f'{weight_name}__trt_folded_batch_norm'
        folded_bias_name = f'{batch_norm_node.output[0]}__trt_folded_batch_norm_bias'
        folded_initializers.extend([
            make_initializer(folded_weight_name, folded_weight),
            make_initializer(folded_bias_name, folded_bias),
        ])

        del conv_node.input[:]
        conv_node.input.extend([conv_input_name, folded_weight_name, folded_bias_name])
        del conv_node.output[:]
        conv_node.output.extend([batch_norm_node.output[0]])
        skipped_nodes.add(id(batch_norm_node))
        folded_count += 1

    if folded_count == 0:
        return 0

    del graph.node[:]
    graph.node.extend(node for node in original_nodes if id(node) not in skipped_nodes)
    graph.initializer.extend(folded_initializers)
    return folded_count


def sort_topk_index_cast_nodes(graph):
    topk_index_outputs = {}
    for node in graph.node:
        if node.op_type != 'TopK' or len(node.output) < 2 or len(node.input) < 2:
            continue
        topk_index_outputs[node.output[1]] = node.input[1]

    if not topk_index_outputs:
        return 0

    new_nodes = []
    lowered_count = 0
    for node in graph.node:
        if node.op_type != 'Cast' or len(node.input) != 1 or node.input[0] not in topk_index_outputs:
            new_nodes.append(node)
            continue

        source_name = node.input[0]
        k_name = topk_index_outputs[source_name]
        prefix = f'{node.output[0]}__trt_sort_topk_indices'
        index_float = f'{prefix}_float'
        sorted_float = f'{prefix}_sorted_float'
        sorted_order = f'{prefix}_order'
        sorted_int64 = f'{prefix}_int64'

        new_nodes.extend([
            helper.make_node('Cast', [source_name], [index_float], name=f'{prefix}_cast_to_float', to=TensorProto.FLOAT),
            helper.make_node('TopK', [index_float, k_name], [sorted_float, sorted_order], name=f'{prefix}_topk', axis=-1, largest=0, sorted=1),
            helper.make_node('Cast', [sorted_float], [sorted_int64], name=f'{prefix}_cast_to_int64', to=TensorProto.INT64),
        ])
        del node.input[:]
        node.input.extend([sorted_int64])
        new_nodes.append(node)
        lowered_count += 1

    if lowered_count == 0:
        return 0

    del graph.node[:]
    graph.node.extend(new_nodes)
    return lowered_count


def lower_native_batch_norm_inference_nodes(graph):
    input_uses = collect_input_uses(graph)
    value_ranks = collect_value_ranks(graph)
    lowered_initializers = []
    new_nodes = []
    lowered_count = 0

    for node in graph.node:
        if node.domain != 'pkg.onnxscript.torch_lib' or node.op_type != '_aten_native_batch_norm_inference_onnx':
            new_nodes.append(node)
            continue
        if len(node.input) != 5 or len(node.output) < 1:
            new_nodes.append(node)
            continue
        if any(input_uses.get(output_name, 0) != 0 for output_name in node.output[1:]):
            new_nodes.append(node)
            continue
        input_rank = value_ranks.get(node.input[0])
        if input_rank is None or input_rank < 2:
            new_nodes.append(node)
            continue
        epsilon = get_attribute(node, 'eps', 1e-5)

        prefix = f'{node.output[0]}__trt_batch_norm'
        broadcast_shape_name = f'{prefix}_broadcast_shape'
        eps_name = f'{prefix}_eps'
        reshape_shape = np.asarray([1, -1] + [1] * (input_rank - 2), dtype=np.int64)
        lowered_initializers.extend([
            make_initializer(broadcast_shape_name, reshape_shape),
            make_initializer(eps_name, np.asarray([float(epsilon)], dtype=np.float32)),
        ])

        weight_reshape = f'{prefix}_weight'
        bias_reshape = f'{prefix}_bias'
        mean_reshape = f'{prefix}_mean'
        var_reshape = f'{prefix}_var'
        centered = f'{prefix}_centered'
        var_eps = f'{prefix}_var_eps'
        std = f'{prefix}_std'
        normalized = f'{prefix}_normalized'
        scaled = f'{prefix}_scaled'

        new_nodes.extend([
            helper.make_node('Reshape', [node.input[1], broadcast_shape_name], [weight_reshape], name=f'{prefix}_reshape_weight'),
            helper.make_node('Reshape', [node.input[2], broadcast_shape_name], [bias_reshape], name=f'{prefix}_reshape_bias'),
            helper.make_node('Reshape', [node.input[3], broadcast_shape_name], [mean_reshape], name=f'{prefix}_reshape_mean'),
            helper.make_node('Reshape', [node.input[4], broadcast_shape_name], [var_reshape], name=f'{prefix}_reshape_var'),
            helper.make_node('Sub', [node.input[0], mean_reshape], [centered], name=f'{prefix}_sub_mean'),
            helper.make_node('Add', [var_reshape, eps_name], [var_eps], name=f'{prefix}_add_eps'),
            helper.make_node('Sqrt', [var_eps], [std], name=f'{prefix}_sqrt'),
            helper.make_node('Div', [centered, std], [normalized], name=f'{prefix}_div_std'),
            helper.make_node('Mul', [normalized, weight_reshape], [scaled], name=f'{prefix}_mul_weight'),
            helper.make_node('Add', [scaled, bias_reshape], [node.output[0]], name=f'{prefix}_add_bias'),
        ])
        lowered_count += 1

    if lowered_count == 0:
        return 0

    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend(lowered_initializers)
    return lowered_count


def lower_reducemax_with_initializer_axes_nodes(graph):
    constants = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in graph.initializer
    }
    for node in graph.node:
        if node.domain == '' and node.op_type == 'Constant' and len(node.output) == 1:
            value = get_constant_value(node)
            if value is not None:
                constants[node.output[0]] = value
    value_shapes = collect_value_shapes(graph)
    lowered_initializers = []
    new_nodes = []
    lowered_count = 0

    for node in graph.node:
        if node.domain != '' or node.op_type != 'ReduceMax':
            new_nodes.append(node)
            continue
        if len(node.input) != 2 or len(node.output) != 1:
            new_nodes.append(node)
            continue
        axes_value = constants.get(node.input[1])
        if axes_value is None:
            new_nodes.append(node)
            continue
        axes = np.asarray(axes_value, dtype=np.int64).reshape(-1)
        if axes.size != 1:
            new_nodes.append(node)
            continue
        keepdims = int(get_attribute(node, 'keepdims', 1))
        if keepdims != 0:
            new_nodes.append(node)
            continue
        input_shape = value_shapes.get(node.input[0])
        if input_shape is None:
            new_nodes.append(node)
            continue
        rank = len(input_shape)
        axis = int(axes[0])
        if axis < 0:
            axis += rank
        if axis < 0 or axis >= rank:
            new_nodes.append(node)
            continue
        axis_size = input_shape[axis]
        if axis_size <= 0 or axis_size > 16:
            new_nodes.append(node)
            continue

        prefix = f'{node.output[0]}__trt_reducemax'
        axes_name = f'{prefix}_axes'
        steps_name = f'{prefix}_steps'
        squeeze_axes_name = f'{prefix}_squeeze_axes'
        lowered_initializers.extend([
            make_initializer(axes_name, np.asarray([axis], dtype=np.int64)),
            make_initializer(steps_name, np.asarray([1], dtype=np.int64)),
            make_initializer(squeeze_axes_name, np.asarray([axis], dtype=np.int64)),
        ])

        squeezed_outputs = []
        for index in range(axis_size):
            start_name = f'{prefix}_{index}_starts'
            end_name = f'{prefix}_{index}_ends'
            slice_output = f'{prefix}_{index}_slice'
            squeeze_output = f'{prefix}_{index}_squeeze'
            lowered_initializers.extend([
                make_initializer(start_name, np.asarray([index], dtype=np.int64)),
                make_initializer(end_name, np.asarray([index + 1], dtype=np.int64)),
            ])
            new_nodes.extend([
                helper.make_node(
                    'Slice',
                    [node.input[0], start_name, end_name, axes_name, steps_name],
                    [slice_output],
                    name=f'{prefix}_{index}_slice_node',
                ),
                helper.make_node(
                    'Squeeze',
                    [slice_output, squeeze_axes_name],
                    [squeeze_output],
                    name=f'{prefix}_{index}_squeeze_node',
                ),
            ])
            squeezed_outputs.append(squeeze_output)

        if len(squeezed_outputs) == 1:
            new_nodes.append(
                helper.make_node(
                    'Identity',
                    [squeezed_outputs[0]],
                    [node.output[0]],
                    name=f'{prefix}_identity_node',
                )
            )
        else:
            new_nodes.append(
                helper.make_node(
                    'Max',
                    squeezed_outputs,
                    [node.output[0]],
                    name=f'{prefix}_max_node',
                )
            )
        lowered_count += 1

    if lowered_count == 0:
        return 0

    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend(lowered_initializers)
    return lowered_count


def lower_aten_repeat_nodes(graph):
    lowered_count = 0

    for node in graph.node:
        if node.domain != 'pkg.onnxscript.torch_lib' or node.op_type != 'aten_repeat':
            continue
        if len(node.input) != 2 or len(node.output) != 1:
            continue
        node.domain = ''
        node.op_type = 'Tile'
        del node.attribute[:]
        lowered_count += 1

    return lowered_count


def lower_aten_where_nodes(graph):
    lowered_count = 0

    for node in graph.node:
        if node.domain != 'pkg.onnxscript.torch_lib' or node.op_type != 'aten_where':
            continue
        if len(node.input) != 3 or len(node.output) != 1:
            continue
        node.domain = ''
        node.op_type = 'Where'
        del node.attribute[:]
        lowered_count += 1

    return lowered_count


def lower_aten_exp_nodes(graph):
    lowered_count = 0

    for node in graph.node:
        if node.domain != 'pkg.onnxscript.torch_lib' or node.op_type != 'aten_exp':
            continue
        if len(node.input) != 1 or len(node.output) != 1:
            continue
        node.domain = ''
        node.op_type = 'Exp'
        del node.attribute[:]
        lowered_count += 1

    return lowered_count


def lower_split_to_sequence_nodes(graph):
    constants = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in graph.initializer
    }
    sequence_sources = {}
    lowered_initializers = []
    lowered_count = 0

    for node in graph.node:
        if node.op_type != 'SplitToSequence' or len(node.input) < 2 or len(node.output) != 1:
            continue
        split_value = constants.get(node.input[1])
        if split_value is None or np.asarray(split_value).size != 1:
            continue
        keepdims = int(get_attribute(node, 'keepdims', 1))
        if keepdims != 1:
            continue
        sequence_sources[node.output[0]] = (
            node.input[0],
            int(np.asarray(split_value).item()),
            int(get_attribute(node, 'axis', 0)),
        )

    if not sequence_sources:
        return 0

    for node in graph.node:
        if node.op_type != 'SequenceAt' or len(node.input) != 2 or len(node.output) != 1:
            continue
        if node.input[0] not in sequence_sources or node.input[1] not in constants:
            continue
        index_value = np.asarray(constants[node.input[1]])
        if index_value.size != 1:
            continue
        data_name, split_size, axis = sequence_sources[node.input[0]]
        index = int(index_value.item())
        prefix = f'{node.output[0]}__trt_sequence_slice'
        start_name = f'{prefix}_starts'
        end_name = f'{prefix}_ends'
        axes_name = f'{prefix}_axes'
        steps_name = f'{prefix}_steps'
        lowered_initializers.extend([
            make_initializer(start_name, np.asarray([index * split_size], dtype=np.int64)),
            make_initializer(end_name, np.asarray([(index + 1) * split_size], dtype=np.int64)),
            make_initializer(axes_name, np.asarray([axis], dtype=np.int64)),
            make_initializer(steps_name, np.asarray([1], dtype=np.int64)),
        ])
        node.domain = ''
        node.op_type = 'Slice'
        del node.input[:]
        node.input.extend([data_name, start_name, end_name, axes_name, steps_name])
        del node.attribute[:]
        lowered_count += 1

    graph.initializer.extend(lowered_initializers)
    return lowered_count


def lower_aten_split_getitem_nodes(graph):
    constants = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in graph.initializer
    }
    value_ranks = collect_value_ranks(graph)
    split_sources = {}
    lowered_initializers = []
    lowered_count = 0

    for node in graph.node:
        if node.domain != 'pkg.onnxscript.torch_lib' or node.op_type != 'aten_split':
            continue
        if len(node.input) < 2 or len(node.output) != 1:
            continue
        split_value = constants.get(node.input[1])
        if split_value is None or np.asarray(split_value).size != 1:
            continue
        axis = int(get_attribute(node, 'dim', 0))
        input_rank = value_ranks.get(node.input[0])
        if input_rank is not None and axis < 0:
            axis += input_rank
        split_sources[node.output[0]] = (
            node.input[0],
            int(np.asarray(split_value).item()),
            axis,
        )

    if not split_sources:
        return 0

    for node in graph.node:
        if node.domain != 'pkg.onnxscript.torch_lib' or node.op_type != 'aten_getitem':
            continue
        if len(node.input) != 2 or len(node.output) != 1:
            continue
        if node.input[0] not in split_sources or node.input[1] not in constants:
            continue
        index_value = np.asarray(constants[node.input[1]])
        if index_value.size != 1:
            continue
        data_name, split_size, axis = split_sources[node.input[0]]
        index = int(index_value.item())
        prefix = f'{node.output[0]}__trt_aten_split_slice'
        start_name = f'{prefix}_starts'
        end_name = f'{prefix}_ends'
        axes_name = f'{prefix}_axes'
        steps_name = f'{prefix}_steps'
        lowered_initializers.extend([
            make_initializer(start_name, np.asarray([index * split_size], dtype=np.int64)),
            make_initializer(end_name, np.asarray([(index + 1) * split_size], dtype=np.int64)),
            make_initializer(axes_name, np.asarray([axis], dtype=np.int64)),
            make_initializer(steps_name, np.asarray([1], dtype=np.int64)),
        ])
        node.domain = ''
        node.op_type = 'Slice'
        del node.input[:]
        node.input.extend([data_name, start_name, end_name, axes_name, steps_name])
        del node.attribute[:]
        lowered_count += 1

    graph.initializer.extend(lowered_initializers)
    return lowered_count


def remove_dead_nodes(graph):
    needed_values = {output.name for output in graph.output}
    kept_nodes_reversed = []
    removed_count = 0

    for node in reversed(graph.node):
        node_outputs = {output_name for output_name in node.output if output_name}
        if node_outputs & needed_values:
            kept_nodes_reversed.append(node)
            needed_values.update(input_name for input_name in node.input if input_name)
        else:
            removed_count += 1

    if removed_count == 0:
        return 0

    del graph.node[:]
    graph.node.extend(reversed(kept_nodes_reversed))
    return removed_count


def remove_unused_maxpool_indices(graph):
    used_values = {input_name for node in graph.node for input_name in node.input if input_name}
    graph_outputs = {output.name for output in graph.output}
    removed_count = 0

    for node in graph.node:
        if node.op_type != 'MaxPool' or len(node.output) <= 1:
            continue
        removable_outputs = [
            output_name
            for output_name in node.output[1:]
            if output_name not in used_values and output_name not in graph_outputs
        ]
        if len(removable_outputs) != len(node.output) - 1:
            continue
        del node.output[1:]
        removed_count += len(removable_outputs)

    return removed_count


def sanitize_graph(graph, disabled_lowerings=None, enable_conv_batch_norm_fold=False, sort_topk_indices=False):
    disabled_lowerings = disabled_lowerings or set()
    lowered_castlike_count = run_lowering('castlike', disabled_lowerings, lower_castlike_nodes, graph)
    lowered_isscalar_count = run_lowering('isscalar', disabled_lowerings, lower_isscalar_nodes, graph)
    lowered_aten_squeeze_count = run_lowering('aten_squeeze', disabled_lowerings, lower_aten_squeeze_dim_nodes, graph)
    lowered_aten_unsqueeze_count = run_lowering('aten_unsqueeze', disabled_lowerings, lower_aten_unsqueeze_nodes, graph)
    folded_conv_batch_norm_count = 0
    if enable_conv_batch_norm_fold:
        folded_conv_batch_norm_count = run_lowering(
            'conv_batch_norm',
            disabled_lowerings,
            fold_conv_native_batch_norm_inference_nodes,
            graph,
        )
    lowered_batch_norm_count = run_lowering('batch_norm', disabled_lowerings, lower_native_batch_norm_inference_nodes, graph)
    lowered_aten_repeat_count = run_lowering('aten_repeat', disabled_lowerings, lower_aten_repeat_nodes, graph)
    lowered_aten_where_count = run_lowering('aten_where', disabled_lowerings, lower_aten_where_nodes, graph)
    lowered_aten_exp_count = run_lowering('aten_exp', disabled_lowerings, lower_aten_exp_nodes, graph)
    constants = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in graph.initializer
    }
    folded_initializers = {}
    kept_nodes = []
    folded_count = 0

    for node in graph.node:
        folded_values = fold_node(node, constants)
        if folded_values is None:
            kept_nodes.append(node)
            continue

        folded_count += 1
        for output_name, value in zip(node.output, folded_values):
            if not output_name:
                continue
            constants[output_name] = value
            folded_initializers[output_name] = make_initializer(output_name, value)

    old_initializer_names = {initializer.name for initializer in graph.initializer}
    new_initializers = [
        initializer
        for initializer in graph.initializer
        if initializer.name not in folded_initializers
    ]
    new_initializers.extend(
        initializer
        for name, initializer in folded_initializers.items()
        if name not in old_initializer_names
    )

    del graph.node[:]
    graph.node.extend(kept_nodes)
    del graph.initializer[:]
    graph.initializer.extend(new_initializers)

    lowered_reducemax_count = run_lowering('reducemax', disabled_lowerings, lower_reducemax_with_initializer_axes_nodes, graph)
    sorted_topk_index_count = 0
    if sort_topk_indices:
        sorted_topk_index_count = run_lowering(
            'sort_topk_indices',
            disabled_lowerings,
            sort_topk_index_cast_nodes,
            graph,
        )
    lowered_sequence_count = run_lowering('sequence', disabled_lowerings, lower_split_to_sequence_nodes, graph)
    lowered_aten_split_count = run_lowering('aten_split', disabled_lowerings, lower_aten_split_getitem_nodes, graph)
    removed_dead_node_count = run_lowering('dead_nodes', disabled_lowerings, remove_dead_nodes, graph)
    removed_maxpool_index_count = run_lowering('maxpool_indices', disabled_lowerings, remove_unused_maxpool_indices, graph)

    return (
        folded_count,
        lowered_castlike_count,
        lowered_isscalar_count,
        lowered_aten_squeeze_count,
        lowered_aten_unsqueeze_count,
        folded_conv_batch_norm_count,
        lowered_batch_norm_count,
        lowered_reducemax_count,
        sorted_topk_index_count,
        lowered_aten_repeat_count,
        lowered_aten_where_count,
        lowered_aten_exp_count,
        lowered_sequence_count,
        lowered_aten_split_count,
        removed_dead_node_count,
        removed_maxpool_index_count,
    )


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    disabled_lowerings = parse_disabled_lowerings(args.disabled_lowerings)

    model = onnx.load(str(input_path))
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        pass
    (
        folded_count,
        lowered_castlike_count,
        lowered_isscalar_count,
        lowered_aten_squeeze_count,
        lowered_aten_unsqueeze_count,
        folded_conv_batch_norm_count,
        lowered_batch_norm_count,
        lowered_reducemax_count,
        sorted_topk_index_count,
        lowered_aten_repeat_count,
        lowered_aten_where_count,
        lowered_aten_exp_count,
        lowered_sequence_count,
        lowered_aten_split_count,
        removed_dead_node_count,
        removed_maxpool_index_count,
    ) = sanitize_graph(
        model.graph,
        disabled_lowerings=disabled_lowerings,
        enable_conv_batch_norm_fold=args.enable_conv_batch_norm_fold,
        sort_topk_indices=args.sort_topk_indices,
    )

    if not args.skip_check:
        onnx.checker.check_model(model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    print(
        f'TensorRT용 ONNX 정리 완료: {output_path} '
        f'folded_nodes={folded_count} '
        f'lowered_castlike_nodes={lowered_castlike_count} '
        f'lowered_isscalar_nodes={lowered_isscalar_count} '
        f'lowered_aten_squeeze_nodes={lowered_aten_squeeze_count} '
        f'lowered_aten_unsqueeze_nodes={lowered_aten_unsqueeze_count} '
        f'folded_conv_batch_norm_nodes={folded_conv_batch_norm_count} '
        f'lowered_batch_norm_nodes={lowered_batch_norm_count} '
        f'lowered_reducemax_nodes={lowered_reducemax_count} '
        f'sorted_topk_index_nodes={sorted_topk_index_count} '
        f'lowered_aten_repeat_nodes={lowered_aten_repeat_count} '
        f'lowered_aten_where_nodes={lowered_aten_where_count} '
        f'lowered_aten_exp_nodes={lowered_aten_exp_count} '
        f'lowered_sequence_nodes={lowered_sequence_count} '
        f'lowered_aten_split_nodes={lowered_aten_split_count} '
        f'removed_dead_nodes={removed_dead_node_count} '
        f'removed_maxpool_indices={removed_maxpool_index_count} '
        f'enable_conv_batch_norm_fold={args.enable_conv_batch_norm_fold} '
        f'sort_topk_indices={args.sort_topk_indices} '
        f'disabled_lowerings={",".join(sorted(disabled_lowerings)) or "none"}'
    )


if __name__ == '__main__':
    main()
