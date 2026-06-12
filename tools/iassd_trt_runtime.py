import ctypes
from pathlib import Path

import tensorrt as trt
import torch


TRT_DTYPE_TO_TORCH = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT32: torch.int32,
    trt.DataType.BOOL: torch.bool,
}


class IASSDTensorRTRunner:
    def __init__(self, engine_file, plugin_library):
        self.engine_file = Path(engine_file).resolve()
        self.plugin_library = Path(plugin_library).resolve()
        if not self.engine_file.exists():
            raise FileNotFoundError(f'TensorRT engine 파일이 없습니다: {self.engine_file}')
        if not self.plugin_library.exists():
            raise FileNotFoundError(f'TensorRT plugin library가 없습니다: {self.plugin_library}')

        ctypes.CDLL(str(self.plugin_library), mode=ctypes.RTLD_GLOBAL)
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, '')
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(self.engine_file.read_bytes())
        if self.engine is None:
            raise RuntimeError('TensorRT engine deserialize 실패')
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError('TensorRT execution context 생성 실패')

        self.input_names, self.output_names = self._collect_io_names()
        if len(self.input_names) != 1:
            raise RuntimeError(f'현재 IA-SSD TensorRT runner는 입력 1개만 지원합니다: {self.input_names}')

    def _tensor_count(self):
        return self.engine.num_io_tensors if hasattr(self.engine, 'num_io_tensors') else self.engine.num_bindings

    def _tensor_name(self, index):
        if hasattr(self.engine, 'get_tensor_name'):
            return self.engine.get_tensor_name(index)
        return self.engine.get_binding_name(index)

    def _is_input(self, name, index):
        if hasattr(self.engine, 'get_tensor_mode'):
            return self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        return self.engine.binding_is_input(index)

    def _tensor_dtype(self, name, index):
        if hasattr(self.engine, 'get_tensor_dtype'):
            return self.engine.get_tensor_dtype(name)
        return self.engine.get_binding_dtype(index)

    def _engine_shape(self, name, index):
        if hasattr(self.engine, 'get_tensor_shape'):
            return tuple(int(dim) for dim in self.engine.get_tensor_shape(name))
        return tuple(int(dim) for dim in self.engine.get_binding_shape(index))

    def _context_shape(self, name, index):
        if hasattr(self.context, 'get_tensor_shape'):
            return tuple(int(dim) for dim in self.context.get_tensor_shape(name))
        return tuple(int(dim) for dim in self.context.get_binding_shape(index))

    def _collect_io_names(self):
        input_names = []
        output_names = []
        for index in range(self._tensor_count()):
            name = self._tensor_name(index)
            if self._is_input(name, index):
                input_names.append(name)
            else:
                output_names.append(name)
        return input_names, output_names

    def _set_input_shape_if_needed(self, input_name, input_tensor):
        shape = tuple(int(dim) for dim in input_tensor.shape)
        if hasattr(self.context, 'set_input_shape'):
            engine_shape = tuple(int(dim) for dim in self.engine.get_tensor_shape(input_name))
            if any(dim < 0 for dim in engine_shape):
                self.context.set_input_shape(input_name, shape)
            return

        binding_index = self.engine.get_binding_index(input_name)
        engine_shape = tuple(int(dim) for dim in self.engine.get_binding_shape(binding_index))
        if any(dim < 0 for dim in engine_shape):
            self.context.set_binding_shape(binding_index, shape)

    def _output_shape(self, name, index):
        shape = self._engine_shape(name, index)
        if any(dim < 0 for dim in shape):
            shape = self._context_shape(name, index)
        return shape

    def _execute(self, bindings):
        stream = torch.cuda.current_stream().cuda_stream
        if hasattr(self.context, 'execute_async_v3'):
            for index in range(self._tensor_count()):
                name = self._tensor_name(index)
                self.context.set_tensor_address(name, bindings[name])
            return self.context.execute_async_v3(stream_handle=stream)

        binding_list = [0] * self.engine.num_bindings
        for index in range(self.engine.num_bindings):
            name = self._tensor_name(index)
            binding_list[index] = bindings[name]
        return self.context.execute_async_v2(bindings=binding_list, stream_handle=stream)

    def infer(self, points):
        input_name = self.input_names[0]
        input_tensor = points.contiguous()
        if not input_tensor.is_cuda:
            input_tensor = input_tensor.cuda()
        if input_tensor.dtype != torch.float32:
            input_tensor = input_tensor.float()

        self._set_input_shape_if_needed(input_name, input_tensor)
        bindings = {input_name: int(input_tensor.data_ptr())}
        output_tensors = {}

        for index in range(self._tensor_count()):
            name = self._tensor_name(index)
            if name == input_name:
                continue
            trt_dtype = self._tensor_dtype(name, index)
            torch_dtype = TRT_DTYPE_TO_TORCH.get(trt_dtype)
            if torch_dtype is None:
                raise RuntimeError(f'지원하지 않는 TensorRT dtype입니다: {name} {trt_dtype}')
            output_tensor = torch.empty(self._output_shape(name, index), dtype=torch_dtype, device=input_tensor.device)
            output_tensors[name] = output_tensor
            bindings[name] = int(output_tensor.data_ptr())

        ok = self._execute(bindings)
        if not ok:
            raise RuntimeError('TensorRT engine 실행 실패')
        torch.cuda.synchronize()

        return {
            name: output_tensors[name].detach().cpu().numpy()
            for name in self.output_names
        }
