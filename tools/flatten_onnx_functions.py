import argparse
import copy
from pathlib import Path

import onnx


def parse_args():
    parser = argparse.ArgumentParser(description='ONNX local function 호출을 graph node로 평탄화')
    parser.add_argument('--input', type=str, required=True, help='입력 ONNX 파일')
    parser.add_argument('--output', type=str, required=True, help='평탄화된 ONNX 파일')
    parser.add_argument('--domains', type=str, default=None, help='쉼표로 구분한 평탄화 대상 function domain 목록')
    parser.add_argument('--exclude_functions', type=str, default=None, help='쉼표로 구분한 평탄화 제외 function 이름 목록')
    parser.add_argument('--remove_identity', action='store_true', help='평탄화 후 Identity node를 제거')
    parser.add_argument('--remove_identity_outputs', action='store_true', help='graph output에 연결된 Identity까지 제거')
    parser.add_argument('--skip_check', action='store_true', help='ONNX checker 검사를 건너뜀')
    return parser.parse_args()


def make_function_map(model):
    return {
        (function.domain, function.name): function
        for function in model.functions
    }


def filter_function_map(function_map, domains, exclude_functions):
    if domains is None:
        domain_filtered = function_map
    else:
        domain_set = {
            domain.strip()
            for domain in domains.split(',')
            if domain.strip()
        }
        domain_filtered = {
            key: function
            for key, function in function_map.items()
            if key[0] in domain_set
        }
    if exclude_functions is None:
        return domain_filtered
    excluded_names = {
        name.strip()
        for name in exclude_functions.split(',')
        if name.strip()
    }
    return {
        key: function
        for key, function in domain_filtered.items()
        if key[1] not in excluded_names
    }


def make_attribute_map(node):
    return {
        attribute.name: attribute
        for attribute in node.attribute
    }


def clone_attribute_with_name(attribute, name):
    cloned = copy.deepcopy(attribute)
    cloned.name = name
    cloned.ref_attr_name = ''
    return cloned


def substitute_attributes(function_node, call_attributes):
    attributes = []
    for attribute in function_node.attribute:
        if attribute.ref_attr_name:
            if attribute.ref_attr_name not in call_attributes:
                raise KeyError(f'호출 node에 function attribute가 없습니다: {attribute.ref_attr_name}')
            attributes.append(clone_attribute_with_name(call_attributes[attribute.ref_attr_name], attribute.name))
        else:
            attributes.append(copy.deepcopy(attribute))
    return attributes


def make_value_mapper(function, call_node, scope):
    value_map = {}
    for function_input, call_input in zip(function.input, call_node.input):
        value_map[function_input] = call_input
    for function_output, call_output in zip(function.output, call_node.output):
        value_map[function_output] = call_output

    def map_value(name):
        if not name:
            return name
        if name in value_map:
            return value_map[name]
        mapped_name = f'{scope}__{name}'
        value_map[name] = mapped_name
        return mapped_name

    return map_value


def rewrite_graph_names(graph, map_value):
    for value_info in list(graph.input) + list(graph.output) + list(graph.value_info):
        value_info.name = map_value(value_info.name)
    for initializer in graph.initializer:
        initializer.name = map_value(initializer.name)
    for sparse_initializer in graph.sparse_initializer:
        sparse_initializer.values.name = map_value(sparse_initializer.values.name)
        sparse_initializer.indices.name = map_value(sparse_initializer.indices.name)
    for node in graph.node:
        node.input[:] = [map_value(name) for name in node.input]
        node.output[:] = [map_value(name) for name in node.output]
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                rewrite_graph_names(attribute.g, map_value)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for subgraph in attribute.graphs:
                    rewrite_graph_names(subgraph, map_value)


def clone_function_node(function_node, call_node, function, function_map, map_value, scope_base, function_node_index):
    scope = f'{scope_base}__{function.name}_{function_node_index}'
    call_attributes = make_attribute_map(call_node)

    cloned = copy.deepcopy(function_node)
    cloned.name = f'{scope}__{function_node.op_type}'
    cloned.input[:] = [map_value(name) for name in function_node.input]
    cloned.output[:] = [map_value(name) for name in function_node.output]
    del cloned.attribute[:]
    cloned.attribute.extend(substitute_attributes(function_node, call_attributes))

    for attribute in cloned.attribute:
        if attribute.type == onnx.AttributeProto.GRAPH:
            rewrite_graph_names(attribute.g, map_value)
            inline_graph(attribute.g, function_map)
        elif attribute.type == onnx.AttributeProto.GRAPHS:
            for subgraph in attribute.graphs:
                rewrite_graph_names(subgraph, map_value)
                inline_graph(subgraph, function_map)

    return cloned


def inline_call_node(call_node, function, function_map, call_index):
    inlined_nodes = []
    scope_base = call_node.name or f'node_{call_index}'
    map_value = make_value_mapper(function, call_node, scope_base)
    for function_node_index, function_node in enumerate(function.node):
        cloned = clone_function_node(
            function_node,
            call_node,
            function,
            function_map,
            map_value,
            scope_base,
            function_node_index,
        )
        nested_function = function_map.get((cloned.domain, cloned.op_type))
        if nested_function is None:
            inlined_nodes.append(cloned)
        else:
            inlined_nodes.extend(inline_call_node(cloned, nested_function, function_map, call_index))
    return inlined_nodes


def inline_graph(graph, function_map):
    new_nodes = []
    for call_index, node in enumerate(graph.node):
        function = function_map.get((node.domain, node.op_type))
        if function is None:
            cloned = copy.deepcopy(node)
            for attribute in cloned.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    inline_graph(attribute.g, function_map)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for subgraph in attribute.graphs:
                        inline_graph(subgraph, function_map)
            new_nodes.append(cloned)
        else:
            new_nodes.extend(inline_call_node(node, function, function_map, call_index))

    del graph.node[:]
    graph.node.extend(new_nodes)


def replace_node_inputs(graph, replacements):
    for node in graph.node:
        node.input[:] = [replacements.get(name, name) for name in node.input]
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                replace_node_inputs(attribute.g, replacements)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for subgraph in attribute.graphs:
                    replace_node_inputs(subgraph, replacements)


def replace_graph_outputs(graph, replacements):
    for output in graph.output:
        output.name = replacements.get(output.name, output.name)


def remove_identity_nodes(graph, remove_identity_outputs=False):
    replacements = {}
    graph_outputs = {output.name for output in graph.output}
    kept_nodes = []

    for node in graph.node:
        if node.domain == '' and node.op_type == 'Identity' and len(node.input) == 1 and len(node.output) == 1:
            input_name = replacements.get(node.input[0], node.input[0])
            output_name = node.output[0]
            if output_name in graph_outputs:
                if remove_identity_outputs:
                    replacements[output_name] = input_name
                else:
                    kept_nodes.append(node)
            else:
                replacements[output_name] = input_name
            continue
        kept_nodes.append(node)

    del graph.node[:]
    graph.node.extend(kept_nodes)
    replace_node_inputs(graph, replacements)
    if remove_identity_outputs:
        replace_graph_outputs(graph, replacements)

    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                remove_identity_nodes(attribute.g, remove_identity_outputs=remove_identity_outputs)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for subgraph in attribute.graphs:
                    remove_identity_nodes(subgraph, remove_identity_outputs=remove_identity_outputs)


def prune_opset_imports(model):
    used_domains = {node.domain for node in model.graph.node}
    used_domains.update(function.domain for function in model.functions)
    used_domains.add('')
    kept_imports = [
        opset_import
        for opset_import in model.opset_import
        if opset_import.domain in used_domains
    ]
    del model.opset_import[:]
    model.opset_import.extend(kept_imports)


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    model = onnx.load(str(input_path))
    all_function_map = make_function_map(model)
    function_map = filter_function_map(all_function_map, args.domains, args.exclude_functions)
    inline_graph(model.graph, function_map)
    if args.remove_identity:
        remove_identity_nodes(model.graph, remove_identity_outputs=args.remove_identity_outputs)
    remaining_functions = [
        function
        for key, function in all_function_map.items()
        if key not in function_map
    ]
    del model.functions[:]
    model.functions.extend(remaining_functions)
    prune_opset_imports(model)

    if not args.skip_check:
        onnx.checker.check_model(model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    print(f'ONNX local function 평탄화 완료: {output_path}')


if __name__ == '__main__':
    main()
