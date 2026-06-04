from contextlib import contextmanager

import torch

try:
    from torch.library import custom_op
except ImportError:
    custom_op = None

try:
    import onnxscript
    from onnxscript import FLOAT, INT32
    from onnxscript import values
except ImportError:
    onnxscript = None
    FLOAT = None
    INT32 = None
    values = None


_EXPORT_PLACEHOLDERS_ENABLED = False
_IASSD_ONNX_OPSET = values.Opset('IASSD', 1) if values is not None else None


@contextmanager
def export_placeholders_enabled():
    global _EXPORT_PLACEHOLDERS_ENABLED
    old_value = _EXPORT_PLACEHOLDERS_ENABLED
    _EXPORT_PLACEHOLDERS_ENABLED = True
    try:
        yield
    finally:
        _EXPORT_PLACEHOLDERS_ENABLED = old_value


def is_export_placeholders_enabled():
    return _EXPORT_PLACEHOLDERS_ENABLED


def _missing_custom_op(*args, **kwargs):
    raise RuntimeError('IA-SSD ONNX export placeholder를 사용하려면 torch.library.custom_op가 필요합니다.')


if custom_op is not None:
    @custom_op('iassd::farthest_point_sampling', mutates_args=())
    def farthest_point_sampling(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        raise RuntimeError('farthest_point_sampling은 ONNX export placeholder 전용입니다.')

    @farthest_point_sampling.register_fake
    def _(xyz, npoint: int):
        return torch.empty((xyz.shape[0], npoint), dtype=torch.int32, device=xyz.device)
else:
    farthest_point_sampling = _missing_custom_op


if onnxscript is not None:
    @onnxscript.script()
    def onnx_farthest_point_sampling(xyz: FLOAT[...], npoint: int) -> INT32[...]:
        return _IASSD_ONNX_OPSET.FarthestPointSampling(xyz, npoint_i=npoint)
else:
    onnx_farthest_point_sampling = None


if custom_op is not None:
    @custom_op('iassd::gather_points', mutates_args=())
    def gather_points(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        raise RuntimeError('gather_points는 ONNX export placeholder 전용입니다.')

    @gather_points.register_fake
    def _(features, idx):
        return torch.empty(
            (features.shape[0], features.shape[1], idx.shape[1]),
            dtype=features.dtype,
            device=features.device,
        )
else:
    gather_points = _missing_custom_op


if onnxscript is not None:
    @onnxscript.script()
    def onnx_gather_points(features: FLOAT[...], idx: INT32[...]) -> FLOAT[...]:
        return _IASSD_ONNX_OPSET.GatherPoints(features, idx)
else:
    onnx_gather_points = None


if custom_op is not None:
    @custom_op('iassd::ball_query', mutates_args=())
    def ball_query(xyz: torch.Tensor, new_xyz: torch.Tensor, radius: float, nsample: int) -> torch.Tensor:
        raise RuntimeError('ball_query는 ONNX export placeholder 전용입니다.')

    @ball_query.register_fake
    def _(xyz, new_xyz, radius: float, nsample: int):
        return torch.empty(
            (xyz.shape[0], new_xyz.shape[1], nsample),
            dtype=torch.int32,
            device=xyz.device,
        )
else:
    ball_query = _missing_custom_op


if onnxscript is not None:
    @onnxscript.script()
    def onnx_ball_query(xyz: FLOAT[...], new_xyz: FLOAT[...], radius: float, nsample: int) -> INT32[...]:
        return _IASSD_ONNX_OPSET.BallQuery(xyz, new_xyz, radius_f=radius, nsample_i=nsample)
else:
    onnx_ball_query = None


if custom_op is not None:
    @custom_op('iassd::group_points', mutates_args=())
    def group_points(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        raise RuntimeError('group_points는 ONNX export placeholder 전용입니다.')

    @group_points.register_fake
    def _(features, idx):
        return torch.empty(
            (features.shape[0], features.shape[1], idx.shape[1], idx.shape[2]),
            dtype=features.dtype,
            device=features.device,
        )
else:
    group_points = _missing_custom_op


if onnxscript is not None:
    @onnxscript.script()
    def onnx_group_points(features: FLOAT[...], idx: INT32[...]) -> FLOAT[...]:
        return _IASSD_ONNX_OPSET.GroupPoints(features, idx)
else:
    onnx_group_points = None


def build_iassd_onnx_registry():
    if custom_op is None:
        raise ImportError('IA-SSD ONNX export custom op registry를 만들려면 torch.library.custom_op가 필요합니다.')
    if onnxscript is None:
        raise ImportError('IA-SSD ONNX export custom op registry를 만들려면 onnxscript가 필요합니다.')

    from torch.onnx._internal.exporter import _registration

    registry = _registration.ONNXRegistry.from_torchlib()
    registry.register_op(torch.ops.iassd.farthest_point_sampling.default, onnx_farthest_point_sampling)
    registry.register_op(torch.ops.iassd.gather_points.default, onnx_gather_points)
    registry.register_op(torch.ops.iassd.ball_query.default, onnx_ball_query)
    registry.register_op(torch.ops.iassd.group_points.default, onnx_group_points)
    return registry
